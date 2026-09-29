/**
 * DIGITIMES Calendar crawler
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
 *   N8N_WEBHOOK_URL n8n webhook URL (default: http://localhost:5678/webhook/digitimes-crawl).
 */

import { existsSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { mkdir, writeFile } from 'fs/promises';

import { Readability } from '@mozilla/readability';
import * as cheerio from 'cheerio';
import dayjs from 'dayjs';
import { JSDOM } from 'jsdom';
import puppeteer from 'puppeteer-core';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CALENDAR_URL = 'https://www.digitimes.com/calendar.php?d=7d&dt_ref=tabs';
const BASE_URL = 'https://www.digitimes.com';
const OUTPUT_DIR = join(__dirname, 'output');
const DELAY_PAGE_MS = 5000;
const DELAY_ARTICLE_MS = 2000;
const DELAY_BETWEEN_ARTICLES_MS = 3500; // Longer delay when using fresh context
const GOTO_TIMEOUT_MS = 90_000;
const SOURCE_LABEL = 'digitimes';

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

interface Article extends N8nCrawlItem {}

function getChromePath(): string {
  if (process.env.PUPPETEER_EXECUTABLE_PATH && existsSync(process.env.PUPPETEER_EXECUTABLE_PATH)) {
    return process.env.PUPPETEER_EXECUTABLE_PATH;
  }
  const linux = ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome'];
  for (const p of linux) {
    if (existsSync(p)) return p;
  }
  const mac = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  if (existsSync(mac)) return mac;
  const chromium = '/Applications/Chromium.app/Contents/MacOS/Chromium';
  if (existsSync(chromium)) return chromium;
  throw new Error('Chrome not found. Set PUPPETEER_EXECUTABLE_PATH or install Chrome/Chromium.');
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Parse date string from DIGITIMES format (e.g., "Feb 12, 2026").
 * Returns Date object or null if parsing fails.
 */
function parseDate(dateStr: string | null | undefined): Date | null {
  if (!dateStr || !String(dateStr).trim()) return null;
  const d = new Date(String(dateStr).trim());
  return Number.isNaN(d.getTime()) ? null : d;
}

/**
 * True if date is within [from, to] (inclusive, start/end of day).
 * If both from and to are null/undefined, returns true (no filter).
 */
function isInDateRange(date: Date | null, from: string | null | undefined, to: string | null | undefined): boolean {
  if (!from && !to) return true;
  if (date == null) return true;
  const t = date.getTime();
  if (Number.isNaN(t)) return true;
  if (from) {
    const fromStart = dayjs(from).startOf('day').valueOf();
    if (t < fromStart) return false;
  }
  if (to) {
    const toEnd = dayjs(to).endOf('day').valueOf();
    if (t > toEnd) return false;
  }
  return true;
}

/**
 * True if date is earlier than DATE_FROM (calendar is newest-first; we can stop).
 */
function isEarlierThanDateFrom(date: Date | null, dateFrom: string | null | undefined): boolean {
  if (!dateFrom || date == null) return false;
  const t = date.getTime();
  if (Number.isNaN(t)) return false;
  const fromStart = dayjs(dateFrom).startOf('day').valueOf();
  return t < fromStart;
}

/**
 * Check if content appears to be paywall/login page content.
 */
function isPaywallContent(content: string): boolean {
  if (!content || content.length < 50) return false;
  const lower = content.toLowerCase();
  const paywallIndicators = [
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
  ];
  return paywallIndicators.some((indicator) => lower.includes(indicator));
}

/**
 * Extract article content from HTML using Readability, with fallbacks.
 * Detects and handles paywall/login pages.
 */
function extractArticleContent(html: string, url: string): { articleContent: string; articleTitle: string; siteName: string } {
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
      } else if (articleContent.length > 200) {
        // Good content found
        return { articleContent, articleTitle, siteName };
      }
    }
  } catch (_) {
    // Fall through to fallback extraction
  }

  // Fallback: Try DIGITIMES-specific selectors (avoid paywall areas)
  const digitimesSelectors = [
    '.article-content',
    '.article-body',
    '.story-content',
    '.post-content',
    '#article-content',
    '[class*="article"]:not([class*="login"]):not([class*="subscribe"])',
    'article',
  ];

  for (const sel of digitimesSelectors) {
    const element = $(sel).first();
    if (element.length) {
      // Remove common paywall/login elements
      element.find('.login, .subscribe, .paywall, [class*="member"], [id*="login"]').remove();
      const text = element.text().trim();
      if (text.length > 200 && !isPaywallContent(text)) {
        articleContent = text;
        break;
      }
    }
  }

  // If still no good content, try main content area but exclude login/paywall sections
  if (!articleContent || articleContent.length < 200) {
    const main = $('main, .main-content, #main-content');
    if (main.length) {
      main.find('.login, .subscribe, .paywall, [class*="member"], [id*="login"], form').remove();
      const text = main.text().trim();
      if (text.length > 200 && !isPaywallContent(text)) {
        articleContent = text;
      }
    }
  }

  // Last resort: extract paragraphs but filter out paywall text
  if (!articleContent || articleContent.length < 200 || isPaywallContent(articleContent)) {
    const paras = $('p')
      .not('.login, .subscribe, .paywall, [class*="member"]')
      .slice(0, 30)
      .map((i, el) => $(el).text().trim())
      .get()
      .filter((t) => t.length > 30 && !isPaywallContent(t))
      .join(' ');
    if (paras.length > 200) articleContent = paras;
  }

  // Final check: if content is still paywall, mark as empty
  if (isPaywallContent(articleContent)) {
    articleContent = '';
  }

  if (!articleTitle) {
    const t = $('title').text().trim();
    if (t && !isPaywallContent(t)) articleTitle = t;
  }
  if (!siteName) {
    siteName =
      $('meta[property="og:site_name"]').attr('content') ||
      $('meta[name="application-name"]').attr('content') ||
      '';
  }

  return { articleContent, articleTitle, siteName };
}

/**
 * Extract article links from the calendar page HTML.
 * Used inside page.evaluate — must be self-contained (no outer refs).
 */
function extractCalendarArticles(baseUrl: string): Array<{ date: string; title: string; url: string }> {
  const articles: Article[] = [];
  let currentDate = '';

  // Find the main calendar container
  const calendar = document.querySelector('#calendar');
  if (!calendar) {
    console.warn('Could not find #calendar container');
    return [];
  }

  // Find the second child div which contains the day blocks
  const calendarContent = calendar.querySelector('#calendar > div:nth-child(2)') || calendar;

  // Find all day blocks (direct children divs)
  const dayBlocks = Array.from(calendarContent.children) as HTMLElement[];

  for (const dayBlock of dayBlocks) {
    // Extract date from .time-title
    const dateElement = dayBlock.querySelector('.time-title');
    if (dateElement) {
      // Normalize whitespace - replace all whitespace sequences with single space
      const dateText = (dateElement.textContent || '').replace(/\s+/g, ' ').trim();
      if (dateText) {
        currentDate = dateText;
      }
    }

    // Extract articles from .media-container
    const mediaContainer = dayBlock.querySelector('.media-container');
    if (!mediaContainer) continue;

    // Find all article links within the media container
    const articleLinks = mediaContainer.querySelectorAll('a[href]');

    for (const link of Array.from(articleLinks)) {
      const href = link.getAttribute('href');
      const title = link.textContent?.trim();

      if (!href || !title || title.length < 10) continue;

      // Make URL absolute if relative
      let fullUrl: string;
      if (href.startsWith('/')) {
        fullUrl = `${baseUrl}${href}`;
      } else if (href.startsWith('http')) {
        fullUrl = href;
      } else {
        fullUrl = `${baseUrl}/${href}`;
      }

      // Filter out non-article URLs
      if (!fullUrl.includes('/news/') && !fullUrl.includes('/calendar')) {
        continue;
      }

      articles.push({
        date: currentDate.replace(/\s+/g, ' ').trim(),
        title: title.replace(/\s+/g, ' ').trim(),
        url: fullUrl,
      });
    }
  }

  // If we didn't find articles with the expected structure, try alternative approach
  if (articles.length === 0) {
    console.log('Trying alternative extraction method...');

    // Look for all links in the calendar area
    const allLinks = calendar.querySelectorAll('a[href]');
    currentDate = '';

    // Walk through elements to find date headers and article links
    const walker = document.createTreeWalker(
      calendarContent,
      NodeFilter.SHOW_ELEMENT,
      {
        acceptNode: (node) => {
          const el = node as HTMLElement;
          if (el.tagName === 'DIV' && el.className && String(el.className).toLowerCase().includes('time')) {
            return NodeFilter.FILTER_ACCEPT;
          }
          if (el.tagName === 'A' && el.hasAttribute('href')) {
            return NodeFilter.FILTER_ACCEPT;
          }
          return NodeFilter.FILTER_SKIP;
        },
      }
    );

    let node: Node | null;
    while ((node = walker.nextNode())) {
      const el = node as HTMLElement;

      // Check if this is a date header
      if (el.tagName === 'DIV') {
        // Normalize whitespace - replace all whitespace sequences with single space
        const dateText = (el.textContent || '').replace(/\s+/g, ' ').trim();
        if (dateText && dateText.length > 5) {
          currentDate = dateText;
        }
      }

      // Check if this is an article link
      if (el.tagName === 'A') {
        const href = el.getAttribute('href');
        const title = el.textContent?.trim();

        if (!href || !title || title.length < 10) continue;

        // Filter out non-article links
        if (!href.includes('/news/') && !href.includes('/calendar')) continue;

        let fullUrl: string;
        if (href.startsWith('/')) {
          fullUrl = `${baseUrl}${href}`;
        } else if (href.startsWith('http')) {
          fullUrl = href;
        } else {
          continue;
        }

        articles.push({
          date: (currentDate || 'Unknown').replace(/\s+/g, ' ').trim(),
          title: title.replace(/\s+/g, ' ').trim(),
          url: fullUrl,
        });
      }
    }
  }

  return articles;
}

async function main(): Promise<void> {
  const maxArticles = process.env.MAX_ARTICLES !== undefined ? Number(process.env.MAX_ARTICLES) : 0;
  const headless = process.env.HEADLESS !== 'false';
  const dateFrom = process.env.DATE_FROM || null;
  const dateTo = process.env.DATE_TO || null;

  console.log('Using Chrome:', getChromePath());
  console.log('Calendar URL:', CALENDAR_URL);
  console.log('Max articles:', maxArticles === 0 ? 'all' : maxArticles);
  if (dateFrom || dateTo) {
    console.log('Date range:', [dateFrom || '(none)', dateTo || '(none)'].join(' ~ '));
  }
  console.log('Headless:', headless);
  console.log('');

  const browser = await puppeteer.launch({
    executablePath: getChromePath(),
    headless,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-blink-features=AutomationControlled',
      '--disable-dev-shm-usage',
    ],
  });

  const results: Article[] = [];
  const seenUrls = new Set<string>();

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });
    await page.setExtraHTTPHeaders({
      'Accept-Language': 'en-US,en;q=0.9',
      Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    });
    await page.setUserAgent(
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    );

    // Handle potential dialogs/modals
    page.on('dialog', async (dialog) => {
      console.log(`Dialog detected: ${dialog.message()}`);
      await dialog.dismiss();
    });

    console.log(`[Calendar] Fetching: ${CALENDAR_URL}`);

    try {
      await page.goto(CALENDAR_URL, { waitUntil: 'load', timeout: GOTO_TIMEOUT_MS });
    } catch (error) {
      console.warn(`Navigation timeout, but page may have loaded. Error: ${error}`);
      // Try to continue anyway - page might be partially loaded
    }
    
    await sleep(DELAY_PAGE_MS);
    
    // Wait for calendar container with a reasonable timeout
    try {
      await page.waitForSelector('#calendar', { timeout: 15000 });
      console.log('Calendar container found');
    } catch (error) {
      console.warn('Calendar container not found after wait, proceeding anyway...');
    }
    
    // Additional wait for dynamic content
    await sleep(2000);
    
    // Try to close any modals/popups that might be blocking
    try {
      const closeButtons = await page.$$('button[aria-label*="close" i], .modal-close, .popup-close, [class*="close"][class*="button"]');
      for (const btn of closeButtons) {
        try {
          await btn.click();
          await sleep(500);
        } catch (e) {
          // Ignore click errors
        }
      }
    } catch (e) {
      // Ignore modal handling errors
    }

    // Extract article links from the calendar page
    interface ArticleLink {
      date: string;
      title: string;
      url: string;
    }
    const articleLinks = await page.evaluate(extractCalendarArticles, BASE_URL) as ArticleLink[];

    console.log(`Found ${articleLinks.length} articles on calendar page`);

    // Filter by date range, deduplicate, then crawl each article (same flow as EE Times)
    let shouldStopScraping = false;
    for (const link of articleLinks) {
      if (maxArticles > 0 && results.length >= maxArticles) break;
      if (shouldStopScraping) break;
      if (seenUrls.has(link.url)) continue;
      seenUrls.add(link.url);

      const articleDate = parseDate(link.date);

      if (isEarlierThanDateFrom(articleDate, dateFrom)) {
        console.log(
          `   ⏹️ Stopping: article date ${link.date || '(no date)'} is earlier than DATE_FROM (${dateFrom}).`
        );
        shouldStopScraping = true;
        break;
      }
      if (!isInDateRange(articleDate, dateFrom, dateTo)) {
        console.log(`   ⏭️ skipped (outside date range) ${link.date || '(no date)'}`);
        await sleep(200);
        continue;
      }

      console.log(`[${results.length + 1}${maxArticles > 0 ? `/${maxArticles}` : ''}] ${link.title}`);

      // Use a fresh browser context per article so each request has no cookies/session.
      // This avoids "first article free, then paywall" behavior.
      const articleContext = await browser.createBrowserContext();
      const articlePage = await articleContext.newPage();

      try {
        await articlePage.setViewport({ width: 1920, height: 1080 });
        await articlePage.setUserAgent(
          'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        );
        // Referer from calendar so the site thinks we came from their list page
        await articlePage.setExtraHTTPHeaders({
          Referer: CALENDAR_URL,
          'Accept-Language': 'en-US,en;q=0.9',
          Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        });

        await articlePage.goto(link.url, { waitUntil: 'load', timeout: GOTO_TIMEOUT_MS });
        await sleep(DELAY_ARTICLE_MS);

        await sleep(1000);
        try {
          const modalSelectors = [
            '.modal-close',
            '.paywall-close',
            '[class*="close"][class*="modal"]',
            'button[aria-label*="close" i]',
            '.overlay-close',
          ];
          for (const selector of modalSelectors) {
            const closeBtn = await articlePage.$(selector);
            if (closeBtn) {
              await closeBtn.click();
              await sleep(500);
            }
          }
        } catch (_) {
          /* ignore */
        }

        const html = await articlePage.content();
        const { articleContent, articleTitle, siteName } = extractArticleContent(html, link.url);

        const crawledAt = new Date().toISOString();
        const isPaywalled = !articleContent || articleContent.length < 200 || isPaywallContent(articleContent);

        results.push({
          url: link.url,
          listTitle: link.title,
          listAuthor: '',
          listDate: link.date,
          title: articleTitle || link.title,
          siteName,
          content: isPaywalled ? '' : articleContent,
          crawledAt,
          source: SOURCE_LABEL,
        });

        if (isPaywalled) {
          console.log(`   ⚠️  Paywall detected - content not available`);
        } else {
          console.log(`   ✅ ${articleContent.length} chars`);
        }
        console.log('---------------------------------------------------');
      } catch (err) {
        console.log(`   ❌ ${err instanceof Error ? err.message : String(err)}`);
        results.push({
          url: link.url,
          listTitle: link.title,
          listAuthor: '',
          listDate: link.date,
          title: link.title,
          siteName: '',
          content: '',
          crawledAt: new Date().toISOString(),
          source: SOURCE_LABEL,
        });
      } finally {
        await articleContext.close();
      }

      await sleep(DELAY_BETWEEN_ARTICLES_MS);
    }

    // ——— Save results (same pattern as EE Times + SemiAnalysis) ———
    const n8nJson = JSON.stringify(results, null, 2);

    await mkdir(OUTPUT_DIR, { recursive: true });
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const jsonPath = join(OUTPUT_DIR, `digitimes-calendar_${timestamp}.json`);
    await writeFile(jsonPath, n8nJson, 'utf-8');
    console.log(`\n✅ Done. Results saved to ${jsonPath}`);

    const n8nPath = join(OUTPUT_DIR, `digitimes-n8n_${timestamp}.json`);
    await writeFile(n8nPath, n8nJson, 'utf-8');
    console.log(`📁 n8n JSON: ${n8nPath}`);

    const outputPath = process.env.OUTPUT_PATH || null;
    if (outputPath) {
      await mkdir(dirname(outputPath), { recursive: true });
      await writeFile(outputPath, n8nJson, 'utf-8');
      console.log(`   Also written to OUTPUT_PATH: ${outputPath}`);
    }

    const webhookUrl = process.env.N8N_WEBHOOK_URL || 'http://localhost:5678/webhook/digitimes-crawl';
    if (webhookUrl) {
      try {
        const response = await fetch(webhookUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: n8nJson,
        });
        console.log(`   Sent to n8n webhook: ${response.status}`);
        if (!response.ok) {
          const text = await response.text();
          console.error('   n8n response:', text.slice(0, 300));
        }
      } catch (e) {
        console.error('   Failed to send to n8n', e);
      }
    }

    console.log('🎉 Crawl finished.');
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
