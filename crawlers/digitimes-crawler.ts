/**
 * DIGITIMES Calendar Crawler
 *
 * Flow: calendar page → extract links + dates → crawl each article with Readability.
 * Same n8n payload shape as EE Times and SemiAnalysis so workflows can reuse mapping.
 *
 * Usage:
 *   pnpm digitimes
 *   MAX_ARTICLES=10 HEADLESS=false pnpm digitimes
 *   DATE_FROM=2026-02-10 DATE_TO=2026-02-12 pnpm digitimes
 *
 * Env:
 *   MAX_ARTICLES     Max articles to crawl (default: all). Use 0 to crawl all collected.
 *   HEADLESS        Set to "false" to show browser (recommended; headless may be blocked).
 *   DATE_FROM       Start of date range (YYYY-MM-DD). Only crawl articles on or after this date.
 *   DATE_TO         End of date range (YYYY-MM-DD). Only crawl articles on or before this date.
 *   OUTPUT_PATH     If set, also write n8n JSON to this path (e.g. /configs/digitimes-latest.json).
 *   OUTPUT_DIR      Directory for timestamped snapshots (default: ./output).
 *   CRAWLER_WEBHOOK_URL If set, POST the results here (N8N_WEBHOOK_URL: deprecated alias). No POST by default.
 */

import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { mkdir, writeFile } from 'fs/promises';

import { Readability } from '@mozilla/readability';
import * as cheerio from 'cheerio';
import { JSDOM } from 'jsdom';
import puppeteer, { type Browser, type Page, type BrowserContext } from 'puppeteer-core';

import {
  getChromePath,
  sleep,
  parseDate,
  isInDateRange,
  isEarlierThanDateFrom,
  parseCrawlerConfig,
  installEsbuildNameShim,
  resolveOutputDir,
  DEFAULT_DELAYS,
  DEFAULT_TIMEOUTS,
  CONTENT_THRESHOLDS,
  type CrawlerConfig,
} from './shared/util-functions.js';

// ─── Constants ─────────────────────────────────────────────────────────────

const __dirname = dirname(fileURLToPath(import.meta.url));
const CALENDAR_URL = process.env.LIST_URL || 'https://www.digitimes.com/calendar.php?d=14d';
const BASE_URL = 'https://www.digitimes.com';
const OUTPUT_DIR = resolveOutputDir(join(__dirname, 'output'));
const SOURCE_LABEL = 'digitimes';

const DIGITIMES_DELAYS = {
  PAGE_LOAD_MS: 5000,
  ARTICLE_LOAD_MS: 2000,
  BETWEEN_ARTICLES_MS: 3500, // Longer delay when using fresh context
  MODAL_CLOSE_MS: 500,
  CALENDAR_WAIT_MS: 2000,
} as const;

const DIGITIMES_SELECTORS = {
  calendar: '#calendar',
  calendarContent: '#calendar > div:nth-child(2)',
  timeTitle: '.time-title',
  mediaContainer: '.media-container',
  articleContent: [
    '.article-content',
    '.article-body',
    '.story-content',
    '.post-content',
    '#article-content',
    '[class*="article"]:not([class*="login"]):not([class*="subscribe"])',
    'article',
  ],
  modalClose: [
    '.modal-close',
    '.paywall-close',
    '[class*="close"][class*="modal"]',
    'button[aria-label*="close" i]',
    '.overlay-close',
  ],
} as const;

const BROWSER_ARGS = [
  '--no-sandbox',
  '--disable-setuid-sandbox',
  '--disable-blink-features=AutomationControlled',
  '--disable-dev-shm-usage',
] as const;

const USER_AGENT =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

const PAYWALL_INDICATORS = [
  'save my user id and password',
  'members only',
  'subscribers only',
  'please login',
  'please log in',
  'subscribe to continue',
  'sign in to continue',
  'this content is for subscribers',
  'premium content',
  'paid subscribers',
] as const;

// ─── Types ────────────────────────────────────────────────────────────────

/** One item per article, same shape as EE Times / SemiAnalysis so n8n pipelines can reuse mapping. */
export interface N8nCrawlItem {
  url: string;
  listTitle: string;
  listAuthor: string;
  listDate: string;
  title: string;
  siteName: string;
  content: string;
  crawledAt: string;
  source: string;
}

interface ArticleLink {
  date: string;
  title: string;
  url: string;
}

interface ExtractedContent {
  articleContent: string;
  articleTitle: string;
  siteName: string;
}

// ─── Paywall Detection ─────────────────────────────────────────────────────

/**
 * Check if content appears to be paywall/login page content.
 *
 * @param content - Content text to check
 * @returns True if paywall indicators are found
 */
function isPaywallContent(content: string): boolean {
  if (!content || content.length < 50) return false;
  const lower = content.toLowerCase();
  return PAYWALL_INDICATORS.some((indicator) => lower.includes(indicator));
}

// ─── Content Extraction ────────────────────────────────────────────────────

/**
 * Extract article content from HTML using Readability with fallbacks.
 *
 * Detects and handles paywall/login pages.
 *
 * @param html - Full HTML content of article page
 * @param url - Article URL (for Readability context)
 * @returns Object with articleContent, articleTitle, siteName
 */
function extractArticleContent(html: string, url: string): ExtractedContent {
  let articleContent = '';
  let articleTitle = '';
  let siteName = '';

  const $ = cheerio.load(html);

  // Check for paywall indicators in the page
  const pageText = $('body').text().toLowerCase();
  const hasPaywallIndicators = isPaywallContent(pageText);

  // Try Readability first
  try {
    const dom = new JSDOM(html, { url });
    const reader = new Readability(dom.window.document);
    const article = reader.parse();

    if (article) {
      articleTitle = (article.title || '').trim();
      if (article.textContent?.trim()) {
        articleContent = article.textContent.trim();
      } else if (article.content) {
        const $content = cheerio.load(article.content);
        articleContent = $content.root().text().trim();
      }
      siteName = (article.siteName || '').trim();

      // Check if Readability extracted paywall content
      if (isPaywallContent(articleContent)) {
        articleContent = '';
      } else if (articleContent.length > CONTENT_THRESHOLDS.MIN_CONTENT_LENGTH) {
        return { articleContent, articleTitle, siteName };
      }
    }
  } catch (error) {
    console.warn(`Readability extraction failed for ${url}:`, error instanceof Error ? error.message : String(error));
  }

  // Fallback: Try DIGITIMES-specific selectors (avoid paywall areas)
  for (const selector of DIGITIMES_SELECTORS.articleContent) {
    const element = $(selector).first();
    if (element.length) {
      // Remove common paywall/login elements
      element.find('.login, .subscribe, .paywall, [class*="member"], [id*="login"]').remove();
      const text = element.text().trim();
      if (text.length > CONTENT_THRESHOLDS.MIN_CONTENT_LENGTH && !isPaywallContent(text)) {
        articleContent = text;
        break;
      }
    }
  }

  // If still no good content, try main content area but exclude login/paywall sections
  if (!articleContent || articleContent.length < CONTENT_THRESHOLDS.MIN_CONTENT_LENGTH) {
    const main = $('main, .main-content, #main-content');
    if (main.length) {
      main.find('.login, .subscribe, .paywall, [class*="member"], [id*="login"], form').remove();
      const text = main.text().trim();
      if (text.length > CONTENT_THRESHOLDS.MIN_CONTENT_LENGTH && !isPaywallContent(text)) {
        articleContent = text;
      }
    }
  }

  // Last resort: extract paragraphs but filter out paywall text
  if (!articleContent || articleContent.length < CONTENT_THRESHOLDS.MIN_CONTENT_LENGTH || isPaywallContent(articleContent)) {
    const paras = $('p')
      .not('.login, .subscribe, .paywall, [class*="member"]')
      .slice(0, 30)
      .map((_, el) => $(el).text().trim())
      .get()
      .filter((t) => t.length > CONTENT_THRESHOLDS.MIN_PARAGRAPH_LENGTH && !isPaywallContent(t))
      .join(' ');
    if (paras.length > CONTENT_THRESHOLDS.MIN_CONTENT_LENGTH) {
      articleContent = paras;
    }
  }

  // Final check: if content is still paywall, mark as empty
  if (isPaywallContent(articleContent)) {
    articleContent = '';
  }

  if (!articleTitle) {
    const titleText = $('title').text().trim();
    if (titleText && !isPaywallContent(titleText)) articleTitle = titleText;
  }

  if (!siteName) {
    siteName =
      $('meta[property="og:site_name"]').attr('content') ||
      $('meta[name="application-name"]').attr('content') ||
      '';
  }

  return { articleContent, articleTitle, siteName };
}

// ─── Browser Setup ────────────────────────────────────────────────────────

/**
 * Configure browser page with headers and viewport.
 *
 * @param page - Puppeteer page instance
 * @param referer - Optional referer header
 * @returns Promise that resolves when configuration is complete
 */
async function configurePage(page: Page, referer?: string): Promise<void> {
  // Must run before the first navigation: see installEsbuildNameShim.
  await installEsbuildNameShim(page);
  await page.setViewport({ width: 1920, height: 1080 });
  const headers: Record<string, string> = {
    'Accept-Language': 'en-US,en;q=0.9',
    Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
  };
  if (referer) {
    headers.Referer = referer;
  }
  await page.setExtraHTTPHeaders(headers);
  await page.setUserAgent(USER_AGENT);
}

/**
 * Launch and configure browser instance.
 *
 * @param config - Crawler configuration
 * @returns Configured browser instance
 * @throws Error if browser launch fails
 */
async function setupBrowser(config: CrawlerConfig): Promise<Browser> {
  try {
    const chromePath = getChromePath();
    console.log('Using Chrome:', chromePath);

    const browser = await puppeteer.launch({
      executablePath: chromePath,
      headless: config.headless,
      args: [...BROWSER_ARGS],
    });

    return browser;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to launch browser: ${errorMessage}`);
  }
}

/**
 * Close any modals/popups on the page.
 *
 * @param page - Puppeteer page instance
 * @returns Promise that resolves when modal handling is complete
 */
async function closeModals(page: Page): Promise<void> {
  try {
    for (const selector of DIGITIMES_SELECTORS.modalClose) {
      const closeBtn = await page.$(selector);
      if (closeBtn) {
        await closeBtn.click();
        await sleep(DIGITIMES_DELAYS.MODAL_CLOSE_MS);
      }
    }
  } catch {
    // Ignore modal handling errors
  }
}

// ─── Article Crawling ───────────────────────────────────────────────────────

/**
 * Crawl a single article using a fresh browser context.
 *
 * Uses fresh context per article to avoid "first article free, then paywall" behavior.
 *
 * @param browser - Puppeteer browser instance
 * @param articleLink - Article link to crawl
 * @param calendarUrl - Calendar URL for referer header
 * @returns Extracted content or null if extraction fails
 */
async function crawlArticleWithFreshContext(
  browser: Browser,
  articleLink: ArticleLink,
  calendarUrl: string
): Promise<ExtractedContent | null> {
  const articleContext = await browser.createBrowserContext();
  const articlePage = await articleContext.newPage();

  try {
    await configurePage(articlePage, calendarUrl);
    await articlePage.goto(articleLink.url, {
      waitUntil: 'load',
      timeout: DEFAULT_TIMEOUTS.LONG_NAVIGATION_MS,
    });
    await sleep(DIGITIMES_DELAYS.ARTICLE_LOAD_MS);
    await sleep(1000);
    await closeModals(articlePage);

    const html = await articlePage.content();
    return extractArticleContent(html, articleLink.url);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`Failed to crawl article ${articleLink.url}:`, errorMessage);
    return null;
  } finally {
    await articleContext.close();
  }
}

/**
 * Process article links: filter by date range and crawl each article.
 *
 * @param browser - Puppeteer browser instance
 * @param articleLinks - Article links extracted from calendar
 * @param config - Crawler configuration
 * @param calendarUrl - Calendar URL for referer header
 * @returns Array of crawled articles
 */
async function processArticleLinks(
  browser: Browser,
  articleLinks: ArticleLink[],
  config: CrawlerConfig,
  calendarUrl: string
): Promise<N8nCrawlItem[]> {
  const results: N8nCrawlItem[] = [];
  const seenUrls = new Set<string>();
  let shouldStopScraping = false;

  for (const link of articleLinks) {
    if (config.maxArticles > 0 && results.length >= config.maxArticles) break;
    if (shouldStopScraping) break;
    if (seenUrls.has(link.url)) continue;
    seenUrls.add(link.url);

    const articleDate = parseDate(link.date);

    if (isEarlierThanDateFrom(articleDate, config.dateFrom)) {
      console.log(
        `   ⏹️ Stopping: article date ${link.date || '(no date)'} is earlier than DATE_FROM (${config.dateFrom}).`
      );
      shouldStopScraping = true;
      break;
    }

    if (!isInDateRange(articleDate, config.dateFrom, config.dateTo)) {
      console.log(`   ⏭️ skipped (outside date range) ${link.date || '(no date)'}`);
      await sleep(200);
      continue;
    }

    console.log(`[${results.length + 1}${config.maxArticles > 0 ? `/${config.maxArticles}` : ''}] ${link.title}`);

    const extractedContent = await crawlArticleWithFreshContext(browser, link, calendarUrl);

    const crawledAt = new Date().toISOString();
    const isPaywalled =
      !extractedContent ||
      !extractedContent.articleContent ||
      extractedContent.articleContent.length < CONTENT_THRESHOLDS.MIN_CONTENT_LENGTH ||
      isPaywallContent(extractedContent.articleContent);

    results.push({
      url: link.url,
      listTitle: link.title,
      listAuthor: '',
      listDate: link.date,
      title: extractedContent?.articleTitle || link.title,
      siteName: extractedContent?.siteName || '',
      content: isPaywalled ? '' : extractedContent?.articleContent || '',
      crawledAt,
      source: SOURCE_LABEL,
    });

    if (isPaywalled) {
      console.log(`   ⚠️  Paywall detected - content not available`);
    } else {
      console.log(`   ✅ ${extractedContent?.articleContent.length || 0} chars`);
    }
    console.log('---------------------------------------------------');

    await sleep(DIGITIMES_DELAYS.BETWEEN_ARTICLES_MS);
  }

  return results;
}

/**
 * Crawl calendar page and extract article links.
 *
 * @param page - Puppeteer page instance
 * @param calendarUrl - Calendar page URL
 * @returns Array of article links
 */
async function crawlCalendarPage(page: Page, calendarUrl: string): Promise<ArticleLink[]> {
  console.log(`[Calendar] Fetching: ${calendarUrl}`);

  try {
    await page.goto(calendarUrl, {
      waitUntil: 'load',
      timeout: DEFAULT_TIMEOUTS.LONG_NAVIGATION_MS,
    });
  } catch (error) {
    console.warn(`Navigation timeout, but page may have loaded. Error: ${error}`);
    // Try to continue anyway - page might be partially loaded
  }

  await sleep(DIGITIMES_DELAYS.PAGE_LOAD_MS);

  // Wait for calendar container with a reasonable timeout
  try {
    await page.waitForSelector(DIGITIMES_SELECTORS.calendar, { timeout: 15000 });
    console.log('Calendar container found');
  } catch (error) {
    console.warn('Calendar container not found after wait, proceeding anyway...');
  }

  // Additional wait for dynamic content
  await sleep(DIGITIMES_DELAYS.CALENDAR_WAIT_MS);

  // Handle dialogs/modals
  page.on('dialog', async (dialog) => {
    console.log(`Dialog detected: ${dialog.message()}`);
    await dialog.dismiss();
  });

  // Try to close any modals/popups
  await closeModals(page);

  // page.evaluate serializes only the passed function — helpers and constants from Node.js
  // scope are not available in the browser. Inject them first via addScriptTag.
  await page.addScriptTag({
    content: `
      const _DT_SEL = ${JSON.stringify({
        calendar: DIGITIMES_SELECTORS.calendar,
        calendarContent: DIGITIMES_SELECTORS.calendarContent,
        timeTitle: DIGITIMES_SELECTORS.timeTitle,
        mediaContainer: DIGITIMES_SELECTORS.mediaContainer,
      })};
      const _DT_MIN_TITLE = ${CONTENT_THRESHOLDS.MIN_TITLE_LENGTH};
      function extractDateFromDayBlock(b) {
        const e = b.querySelector(_DT_SEL.timeTitle);
        return e ? (e.textContent || '').replace(/\\s+/g, ' ').trim() : '';
      }
      function extractLinksFromMediaContainer(mc, baseUrl, date) {
        const result = [];
        for (const link of Array.from(mc.querySelectorAll('a[href]'))) {
          const href = link.getAttribute('href');
          const title = (link.textContent || '').trim();
          if (!href || !title || title.length < _DT_MIN_TITLE) continue;
          let url = href.startsWith('/') ? baseUrl + href : href.startsWith('http') ? href : baseUrl + '/' + href;
          if (!url.includes('/news/') && !url.includes('/calendar')) continue;
          result.push({ date: date.replace(/\\s+/g, ' ').trim(), title: title.replace(/\\s+/g, ' ').trim(), url });
        }
        return result;
      }
      function extractCalendarArticlesPrimary(baseUrl) {
        const articles = [];
        let currentDate = '';
        const calendar = document.querySelector(_DT_SEL.calendar);
        if (!calendar) { console.warn('Could not find #calendar container'); return []; }
        const calendarContent = calendar.querySelector(_DT_SEL.calendarContent) || calendar;
        for (const dayBlock of Array.from(calendarContent.children)) {
          const dateText = extractDateFromDayBlock(dayBlock);
          if (dateText) currentDate = dateText;
          const mc = dayBlock.querySelector(_DT_SEL.mediaContainer);
          if (!mc) continue;
          articles.push(...extractLinksFromMediaContainer(mc, baseUrl, currentDate));
        }
        return articles;
      }
      function extractCalendarArticlesAlternative(baseUrl) {
        const articles = [];
        let currentDate = '';
        const calendar = document.querySelector(_DT_SEL.calendar);
        if (!calendar) return [];
        const calendarContent = calendar.querySelector(_DT_SEL.calendarContent) || calendar;
        const walker = document.createTreeWalker(calendarContent, NodeFilter.SHOW_ELEMENT, {
          acceptNode: function(node) {
            if (node.tagName === 'DIV' && node.className && String(node.className).toLowerCase().includes('time')) return NodeFilter.FILTER_ACCEPT;
            if (node.tagName === 'A' && node.hasAttribute('href')) return NodeFilter.FILTER_ACCEPT;
            return NodeFilter.FILTER_SKIP;
          }
        });
        let node;
        while ((node = walker.nextNode())) {
          if (node.tagName === 'DIV') {
            const t = (node.textContent || '').replace(/\\s+/g, ' ').trim();
            if (t && t.length > 5) currentDate = t;
          }
          if (node.tagName === 'A') {
            const href = node.getAttribute('href');
            const title = (node.textContent || '').trim();
            if (!href || !title || title.length < _DT_MIN_TITLE) continue;
            if (!href.includes('/news/') && !href.includes('/calendar')) continue;
            const url = href.startsWith('/') ? baseUrl + href : href.startsWith('http') ? href : null;
            if (!url) continue;
            articles.push({ date: (currentDate || 'Unknown').replace(/\\s+/g, ' ').trim(), title: title.replace(/\\s+/g, ' ').trim(), url });
          }
        }
        return articles;
      }
    `,
  });

  // Extract article links from the calendar page
  const articleLinks = (await page.evaluate((baseUrl: string) => {
    const g = globalThis as any;
    const articles = g.extractCalendarArticlesPrimary(baseUrl) as ArticleLink[];
    if (articles.length === 0) {
      console.log('Trying alternative extraction method...');
      return g.extractCalendarArticlesAlternative(baseUrl) as ArticleLink[];
    }
    return articles;
  }, BASE_URL)) as ArticleLink[];

  console.log(`Found ${articleLinks.length} articles on calendar page`);
  return articleLinks;
}

// ─── Output & Webhook ────────────────────────────────────────────────────────

/**
 * Save results to files.
 *
 * @param results - Array of crawled articles
 * @param config - Crawler configuration
 * @returns Paths to saved JSON files
 * @throws Error if file write fails
 */
async function saveResults(results: N8nCrawlItem[], config: CrawlerConfig): Promise<{ jsonPath: string; n8nPath: string }> {
  try {
    await mkdir(OUTPUT_DIR, { recursive: true });
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const jsonPath = join(OUTPUT_DIR, `digitimes-calendar_${timestamp}.json`);
    const n8nPath = join(OUTPUT_DIR, `digitimes-n8n_${timestamp}.json`);
    const jsonContent = JSON.stringify(results, null, 2);

    await writeFile(jsonPath, jsonContent, 'utf-8');
    console.log(`\n✅ Done. Results saved to ${jsonPath}`);

    await writeFile(n8nPath, jsonContent, 'utf-8');
    console.log(`📁 n8n JSON: ${n8nPath}`);

    // Write to OUTPUT_PATH if specified
    if (config.outputPath) {
      await mkdir(dirname(config.outputPath), { recursive: true });
      await writeFile(config.outputPath, jsonContent, 'utf-8');
      console.log(`   Also written to OUTPUT_PATH: ${config.outputPath}`);
    }

    return { jsonPath, n8nPath };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to save results: ${errorMessage}`);
  }
}

/**
 * Send results to n8n webhook.
 *
 * @param results - Array of crawled articles
 * @param webhookUrl - Webhook URL
 * @throws Error if webhook request fails
 */
async function sendToWebhook(results: N8nCrawlItem[], webhookUrl: string): Promise<void> {
  try {
    const response = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(results),
    });

    console.log(`   Sent to n8n webhook: ${response.status}`);

    if (!response.ok) {
      const responseText = await response.text();
      console.error('   n8n response:', responseText.slice(0, 300));
      throw new Error(`Webhook returned status ${response.status}: ${responseText.slice(0, 200)}`);
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`   Failed to send to n8n webhook ${webhookUrl}:`, errorMessage);
    throw error;
  }
}

// ─── Main Entry Point ────────────────────────────────────────────────────────

/**
 * Main crawler function.
 *
 * Orchestrates browser setup, calendar page crawling, article extraction, and result saving.
 *
 * @throws Error if critical operations fail
 */
async function main(): Promise<void> {
  const config = parseCrawlerConfig();

  console.log('Calendar URL:', CALENDAR_URL);
  console.log('Max articles:', config.maxArticles === 0 ? 'all' : config.maxArticles);
  if (config.dateFrom || config.dateTo) {
    console.log('Date range:', [config.dateFrom || '(none)', config.dateTo || '(none)'].join(' ~ '));
  }
  console.log('Headless:', config.headless);
  console.log('');

  const browser = await setupBrowser(config);

  try {
    const page = await browser.newPage();
    await configurePage(page);

    const articleLinks = await crawlCalendarPage(page, CALENDAR_URL);
    const results = await processArticleLinks(browser, articleLinks, config, CALENDAR_URL);

    await saveResults(results, config);

    if (config.webhookUrl) {
      await sendToWebhook(results, config.webhookUrl);
    }

    console.log('🎉 Crawl finished.');
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error('Crawler error:', errorMessage);
    throw error;
  } finally {
    await browser.close();
  }
}

// ─── Execution ────────────────────────────────────────────────────────────

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
