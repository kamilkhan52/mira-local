/**
 * EE Times Designline crawler
 *
 * Flow: list page → extract links + dates → pagination → crawl each article with Readability.
 * Default list page: https://www.eetimes.com/category/news-analysis/designline/memory-designline/
 *
 * Usage:
 *   pnpm micron:eetimes
 *   MAX_ARTICLES=5 HEADLESS=false pnpm micron:eetimes
 *   DATE_FROM=2026-01-01 DATE_TO=2026-01-31 pnpm micron:eetimes
 *   LIST_URL=https://www.eetimes.com/category/news-analysis/designline/storage-designline/ pnpm micron:eetimes
 *
 * Env:
 *   LIST_URL        EE Times designline URL to crawl (default: memory-designline).
 *   MAX_ARTICLES     Max articles to crawl (default 3). Use 0 to crawl all collected.
 *   HEADLESS        Set to "false" to show browser (recommended; headless may be blocked).
 *   DATE_FROM       Start of date range (YYYY-MM-DD). Only scrape articles on or after this date.
 *   DATE_TO         End of date range (YYYY-MM-DD). Only scrape articles on or before this date.
 *   OUTPUT_PATH     If set, also write JSON to this path (e.g. /configs/eetimes-latest.json for n8n).
 *   N8N_WEBHOOK_URL n8n webhook URL (default: http://localhost:5678/webhook/eetimes-crawl).
 */

import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { mkdir, writeFile } from 'fs/promises';

import { Readability } from '@mozilla/readability';
import * as cheerio from 'cheerio';
import { JSDOM } from 'jsdom';
import puppeteer, { type Browser, type Page } from 'puppeteer-core';

import {
  getChromePath,
  sleep,
  parseDate,
  isInDateRange,
  isEarlierThanDateFrom,
  parseCrawlerConfig,
  DEFAULT_DELAYS,
  DEFAULT_TIMEOUTS,
  CONTENT_THRESHOLDS,
  type CrawlerConfig,
  type ParsedAuthorDate,
} from './shared/util-functions.js';

// ─── Constants ─────────────────────────────────────────────────────────────

const __dirname = dirname(fileURLToPath(import.meta.url));
const LIST_URL = process.env.LIST_URL || 'https://www.eetimes.com/category/news-analysis/designline/memory-designline/';
const OUTPUT_DIR = join(__dirname, 'output');
const SOURCE_LABEL = 'eetimes';

const EETIMES_SELECTORS = {
  featured: '.categoryFeatured-title',
  headline: '.headline-title',
  card: '.card-title',
  dateInfo: '.categoryFeatured-info, .headline-info, .card-info',
  pagination: 'nav',
  content: [
    'article',
    '.article',
    '.content',
    '.post-content',
    '.entry-content',
    '.story-content',
    '.article-content',
    '.post-body',
    '.entry-body',
    'main',
  ],
} as const;

const BROWSER_ARGS = [
  '--no-sandbox',
  '--disable-setuid-sandbox',
  '--single-process', // Avoids WS endpoint timeout in Docker (fewer processes, no zygote wait).
  '--disable-blink-features=AutomationControlled',
  '--disable-dev-shm-usage',
  '--disable-gpu',
  '--no-first-run',
  '--disable-software-rasterizer',
  '--disable-extensions',
  '--disable-background-networking',
  '--disable-default-apps',
  '--disable-sync',
  '--metrics-recording-only',
  '--mute-audio',
  '--no-zygote',
  '--disable-logging',
  '--log-level=3', // Fatal only; reduces DBus/Chrome stderr noise in Docker.
] as const;

const USER_AGENT =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

// ─── Types ────────────────────────────────────────────────────────────────

export interface N8nCrawlItem {
  url: string;
  listTitle: string;
  listAuthor: string;
  listDate: string;
  title: string;
  siteName: string;
  content: string;
  crawledAt: string;
}

interface ListArticle {
  url: string;
  title: string;
  date: string;
}

interface ExtractedContent {
  articleContent: string;
  articleTitle: string;
  siteName: string;
}

// ─── Date Parsing ─────────────────────────────────────────────────────────

/**
 * Split raw "By Author Date" string into { author, date }.
 *
 * Handles formats like "By\t\tMajeed Ahmad \t\tJanuary 26, 2026"
 *
 * @param raw - Raw date string from page
 * @returns Object with author and date strings
 */
function parseAuthorAndDate(raw: string | null | undefined): ParsedAuthorDate {
  if (!raw || !String(raw).trim()) return { author: '', date: '' };

  let normalized = String(raw).trim().replace(/\s+/g, ' ');
  const byMatch = normalized.match(/^By\s+/i);
  if (byMatch) {
    normalized = normalized.slice(byMatch[0].length).trim();
  }

  const dateRegex =
    /(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}[./]\d{4}/;

  const dateMatch = normalized.match(dateRegex);
  if (dateMatch) {
    const date = dateMatch[0].trim();
    const author = normalized.slice(0, dateMatch.index).replace(/\s+/g, ' ').trim();
    return { author, date };
  }

  return { author: normalized, date: '' };
}

/**
 * Parse listDate string (e.g. "January 26, 2026") to Date.
 *
 * @param listDateStr - Date string from list page
 * @returns Date object or null if parsing fails
 */
function parseListDateToDate(listDateStr: string | null | undefined): Date | null {
  return parseDate(listDateStr);
}

// ─── List Page Extraction ────────────────────────────────────────────────

/**
 * Extract all article links + dates from list page.
 *
 * Selectors (from page structure):
 *   - .categoryFeatured-title a (segment-main featured) + .categoryFeatured-info
 *   - .headline-title a (segment-one) + .headline-info
 *   - .card-title a (segment-main cards) + .card-info
 *
 * Used inside page.evaluate — must be self-contained (no outer refs).
 *
 * @param listPageUrl - Current list page URL
 * @returns Array of article objects with url, title, date
 */
function extractListArticles(listPageUrl: string): ListArticle[] {
  const listPath = new URL(listPageUrl).pathname.replace(/\/$/, '') || '/';
  const seenUrls = new Set<string>();
  const articles: ListArticle[] = [];

  /**
   * Check if URL is a valid article URL.
   */
  function isArticleUrl(href: string): boolean {
    try {
      const url = new URL(href);
      if (url.origin !== 'https://www.eetimes.com') return false;

      const path = url.pathname.replace(/\/$/, '') || '/';
      if (path.includes('/category/') || path.includes('/tag/') || path.includes('/author/')) {
        return false;
      }
      if (url.search || url.hash) return false;
      if (path === listPath) return false;

      const segments = path.split('/').filter(Boolean);
      return segments.length >= 1 && segments.length <= 3;
    } catch {
      return false;
    }
  }

  /**
   * Extract date string from row element.
   */
  function dateFromRow(row: Element | null): string {
    if (!row) return '';
    const dateSelector = '.categoryFeatured-info, .headline-info, .card-info';
    const dateElement = row.querySelector(dateSelector);
    if (!dateElement && row.parentElement) {
      return row.parentElement.querySelector(dateSelector)?.textContent?.trim() || '';
    }
    return dateElement?.textContent?.trim() || '';
  }

  /**
   * Add article to results if valid and not seen.
   */
  function addArticle(url: string, title: string, date: string): void {
    if (!url || seenUrls.has(url)) return;
    if (!isArticleUrl(url)) return;

    seenUrls.add(url);
    articles.push({
      url,
      title: (title || '').trim(),
      date: (date || '').trim(),
    });
  }

  // Extract featured articles
  document.querySelectorAll('.categoryFeatured-title').forEach((titleElement: Element) => {
    const link = titleElement.querySelector('a') as HTMLAnchorElement | null;
    if (!link || !link.href) return;
    const row = titleElement.parentElement;
    addArticle(link.href, link.textContent || '', dateFromRow(row || null));
  });

  // Extract headline articles
  document.querySelectorAll('.headline-title a').forEach((link: Element) => {
    const anchor = link as HTMLAnchorElement;
    if (!anchor.href) return;
    const titleDiv = link.closest('.headline-title');
    const row = titleDiv ? titleDiv.parentElement : link.closest('div');
    addArticle(anchor.href, anchor.textContent || '', dateFromRow(row || null));
  });

  // Extract card articles
  document.querySelectorAll('.card-title a').forEach((link: Element) => {
    const anchor = link as HTMLAnchorElement;
    if (!anchor.href) return;
    const titleDiv = link.closest('.card-title');
    const row = titleDiv ? titleDiv.parentElement : link.closest('div');
    addArticle(anchor.href, anchor.textContent || '', dateFromRow(row || null));
  });

  return articles;
}

/**
 * Find "Next" pagination link.
 *
 * Tries: next|›|» in nav, then nav div a:nth-child(2), then rel="next".
 * Used inside page.evaluate — must be self-contained.
 *
 * @param currentListUrl - Current list page URL
 * @returns Next page URL or null if not found
 */
function getNextPageLink(currentListUrl: string): string | null {
  const nav = document.querySelector('nav');
  if (!nav) return null;

  const links = Array.from(nav.querySelectorAll('div a'));
  const nextLink = links.find((link) => /next|›|»/i.test((link.textContent || '').trim()));
  if (nextLink?.href && nextLink.href !== currentListUrl) {
    return nextLink.href;
  }

  const secondLink = nav.querySelector('div a:nth-child(2)');
  if (secondLink?.href && secondLink.href !== currentListUrl) {
    return secondLink.href;
  }

  const relNext =
    document.querySelector('a[rel="next"]') || document.querySelector('link[rel="next"]');
  if (relNext) {
    const href = (relNext as HTMLAnchorElement).href || relNext.getAttribute('href');
    if (href) {
      try {
        const absoluteUrl = new URL(href, document.baseURI).href;
        if (absoluteUrl !== currentListUrl) return absoluteUrl;
      } catch {
        // Invalid URL, ignore
      }
    }
  }

  return null;
}

// ─── Content Extraction ───────────────────────────────────────────────────

/**
 * Extract article content using Readability with fallback selectors.
 *
 * @param html - Full HTML content of article page
 * @param url - Article URL (for Readability context)
 * @returns Object with articleContent, articleTitle, siteName
 */
function extractArticleContent(html: string, url: string): ExtractedContent {
  let articleContent = '';
  let articleTitle = '';
  let siteName = '';

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

      if (articleContent.length > CONTENT_THRESHOLDS.MIN_CONTENT_LENGTH) {
        return { articleContent, articleTitle, siteName };
      }
    }
  } catch (error) {
    console.warn(`Readability extraction failed for ${url}:`, error instanceof Error ? error.message : String(error));
  }

  // Fallback: Try common content selectors
  const $ = cheerio.load(html);
  for (const selector of EETIMES_SELECTORS.content) {
    const text = $(selector).text().trim();
    if (text.length > CONTENT_THRESHOLDS.MIN_CONTENT_LENGTH) {
      articleContent = text;
      break;
    }
  }

  // Last resort: Extract paragraphs
  if (!articleContent || articleContent.length < CONTENT_THRESHOLDS.MIN_CONTENT_LENGTH) {
    const paragraphs = $('p')
      .slice(0, 20)
      .map((_, element) => $(element).text().trim())
      .get()
      .filter((text) => text.length > CONTENT_THRESHOLDS.MIN_PARAGRAPH_LENGTH)
      .join(' ');

    if (paragraphs.length > CONTENT_THRESHOLDS.MIN_SNIPPET_LENGTH) {
      articleContent = paragraphs;
    }
  }

  // Extract title if not found
  if (!articleTitle) {
    const titleText = $('title').text().trim();
    if (titleText) articleTitle = titleText;
  }

  // Extract site name if not found
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
 * @returns Promise that resolves when configuration is complete
 */
async function configurePage(page: Page): Promise<void> {
  await page.setViewport({ width: 1920, height: 1080 });
  await page.setExtraHTTPHeaders({
    'Accept-Language': 'en-US,en;q=0.9',
    Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
  });
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
      timeout: 120_000,
      protocolTimeout: 120_000,
      dumpio: false, // Don't pipe Chrome stderr (avoids DBus/Chrome noise in n8n execution output).
    });

    return browser;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to launch browser: ${errorMessage}`);
  }
}

// ─── Article Crawling ────────────────────────────────────────────────────

/**
 * Crawl a single article and extract content.
 *
 * @param page - Puppeteer page instance
 * @param articleUrl - URL of article to crawl
 * @returns Extracted content or null if extraction fails
 */
async function crawlArticle(page: Page, articleUrl: string): Promise<ExtractedContent | null> {
  try {
    await page.goto(articleUrl, {
      waitUntil: 'load',
      timeout: DEFAULT_TIMEOUTS.NAVIGATION_MS,
    });
    await sleep(DEFAULT_DELAYS.ARTICLE_LOAD_MS);

    const html = await page.content();
    return extractArticleContent(html, articleUrl);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`Failed to crawl article ${articleUrl}:`, errorMessage);
    return null;
  }
}

/**
 * Process articles from a list page.
 *
 * Filters by date range, crawls each article, and collects results.
 *
 * @param page - Puppeteer page instance
 * @param listArticles - Articles extracted from list page
 * @param config - Crawler configuration
 * @param results - Array to append results to
 * @param seenUrls - Set of already processed URLs
 * @returns Object indicating if scraping should stop
 */
async function processListArticles(
  page: Page,
  listArticles: ListArticle[],
  config: CrawlerConfig,
  results: N8nCrawlItem[],
  seenUrls: Set<string>
): Promise<{ shouldStop: boolean }> {
  for (const article of listArticles) {
    if (config.maxArticles > 0 && results.length >= config.maxArticles) {
      return { shouldStop: true };
    }

    if (seenUrls.has(article.url)) continue;
    seenUrls.add(article.url);

    const { author: listAuthor, date: listDate } = parseAuthorAndDate(article.date);
    const articleDate = parseListDateToDate(listDate);

    // Early stop if date is before DATE_FROM
    if (isEarlierThanDateFrom(articleDate, config.dateFrom)) {
      console.log(
        `   ⏹️ Stopping: article date ${listDate || '(no date)'} is earlier than DATE_FROM (${config.dateFrom}).`
      );
      return { shouldStop: true };
    }

    // Skip if outside date range
    if (!isInDateRange(articleDate, config.dateFrom, config.dateTo)) {
      console.log(`   ⏭️ skipped (outside date range) ${listDate || '(no date)'}`);
      await sleep(200);
      continue;
    }

    console.log(
      `[${results.length + 1}${config.maxArticles > 0 ? `/${config.maxArticles}` : ''}] ${article.title || article.url}`
    );

    const extractedContent = await crawlArticle(page, article.url);

    if (extractedContent) {
      results.push({
        url: article.url,
        listTitle: article.title,
        listAuthor,
        listDate,
        title: extractedContent.articleTitle || article.title,
        siteName: extractedContent.siteName,
        content: extractedContent.articleContent,
        crawledAt: new Date().toISOString(),
      });
      console.log(`   ✅ ${extractedContent.articleContent.length} chars`);
    } else {
      console.log(`   ❌ Failed to extract content`);
    }

    await sleep(DEFAULT_DELAYS.ARTICLE_LOAD_MS);
  }

  return { shouldStop: false };
}

/**
 * Crawl list pages with pagination.
 *
 * @param page - Puppeteer page instance
 * @param config - Crawler configuration
 * @returns Array of crawled articles
 */
async function crawlListPages(page: Page, config: CrawlerConfig): Promise<N8nCrawlItem[]> {
  const results: N8nCrawlItem[] = [];
  const seenUrls = new Set<string>();
  let currentListUrl = LIST_URL;
  let pageNumber = 1;
  let shouldStopScraping = false;

  while (!shouldStopScraping) {
    if (config.maxArticles > 0 && results.length >= config.maxArticles) break;

    console.log(`[List] Fetching page ${pageNumber}: ${currentListUrl}`);

    try {
      await page.goto(currentListUrl, {
        waitUntil: 'load',
        timeout: DEFAULT_TIMEOUTS.NAVIGATION_MS,
      });
      await sleep(DEFAULT_DELAYS.PAGE_LOAD_MS);

      // Use arrow function without TypeScript types to avoid serialization issues
      // Puppeteer serializes this function to send to browser context
      const listArticlesRaw = await page.evaluate(
        (listPageUrl) => {
          const listPath = new URL(listPageUrl).pathname.replace(/\/$/, '') || '/';
          const seenUrls = new Set();
          const articles = [];

          function isArticleUrl(href) {
            try {
              const url = new URL(href);
              if (url.origin !== 'https://www.eetimes.com') return false;
              const path = url.pathname.replace(/\/$/, '') || '/';
              if (path.includes('/category/') || path.includes('/tag/') || path.includes('/author/')) {
                return false;
              }
              if (url.search || url.hash) return false;
              if (path === listPath) return false;
              const segments = path.split('/').filter(Boolean);
              return segments.length >= 1 && segments.length <= 3;
            } catch {
              return false;
            }
          }

          function dateFromRow(row) {
            if (!row) return '';
            const dateSelector = '.categoryFeatured-info, .headline-info, .card-info';
            let dateElement = row.querySelector(dateSelector);
            if (!dateElement && row.parentElement) {
              dateElement = row.parentElement.querySelector(dateSelector);
            }
            return dateElement && dateElement.textContent ? dateElement.textContent.trim() : '';
          }

          function addArticle(url, title, date) {
            if (!url || seenUrls.has(url)) return;
            if (!isArticleUrl(url)) return;
            seenUrls.add(url);
            articles.push({
              url: url,
              title: (title || '').trim(),
              date: (date || '').trim()
            });
          }

          document.querySelectorAll('.categoryFeatured-title').forEach((titleElement) => {
            const link = titleElement.querySelector('a');
            if (!link || !link.href) return;
            const row = titleElement.parentElement;
            addArticle(link.href, link.textContent, dateFromRow(row));
          });

          document.querySelectorAll('.headline-title a').forEach((link) => {
            if (!link.href) return;
            const titleDiv = link.closest('.headline-title');
            const row = titleDiv ? titleDiv.parentElement : link.closest('div');
            addArticle(link.href, link.textContent, dateFromRow(row));
          });

          document.querySelectorAll('.card-title a').forEach((link) => {
            if (!link.href) return;
            const titleDiv = link.closest('.card-title');
            const row = titleDiv ? titleDiv.parentElement : link.closest('div');
            addArticle(link.href, link.textContent, dateFromRow(row));
          });

          return articles;
        },
        currentListUrl
      );
      const listArticles = Array.isArray(listArticlesRaw) ? (listArticlesRaw as ListArticle[]) : [];

      const { shouldStop } = await processListArticles(page, listArticles, config, results, seenUrls);
      shouldStopScraping = shouldStop;

      if (config.maxArticles > 0 && results.length >= config.maxArticles) break;
      if (shouldStopScraping) break;

      // Check for next page
      await page.goto(currentListUrl, {
        waitUntil: 'load',
        timeout: DEFAULT_TIMEOUTS.NAVIGATION_MS,
      });
      await sleep(1000);

      const nextHref = (await page.evaluate(
        (currentListUrl) => {
          const nav = document.querySelector('nav');
          if (!nav) return null;

          const links = Array.from(nav.querySelectorAll('div a'));
          const nextLink = links.find((link) => /next|›|»/i.test((link.textContent || '').trim()));
          if (nextLink && nextLink.href && nextLink.href !== currentListUrl) {
            return nextLink.href;
          }

          const secondLink = nav.querySelector('div a:nth-child(2)');
          if (secondLink && secondLink.href && secondLink.href !== currentListUrl) {
            return secondLink.href;
          }

          const relNext = document.querySelector('a[rel="next"]') || document.querySelector('link[rel="next"]');
          if (relNext) {
            const href = relNext.href || relNext.getAttribute('href');
            if (href) {
              try {
                const absoluteUrl = new URL(href, document.baseURI).href;
                if (absoluteUrl !== currentListUrl) return absoluteUrl;
              } catch {
                // Invalid URL, ignore
              }
            }
          }

          return null;
        },
        currentListUrl
      )) as string | null;
      if (!nextHref || nextHref === currentListUrl) {
        console.log('[List] No more pagination.');
        break;
      }

      currentListUrl = nextHref;
      pageNumber++;
      await sleep(DEFAULT_DELAYS.PAGINATION_MS);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error(`Error crawling list page ${currentListUrl}:`, errorMessage);
      throw error;
    }
  }

  return results;
}

// ─── Output & Webhook ────────────────────────────────────────────────────

/**
 * Save results to files.
 *
 * @param results - Array of crawled articles
 * @param config - Crawler configuration
 * @returns Path to saved JSON file
 * @throws Error if file write fails
 */
async function saveResults(results: N8nCrawlItem[], config: CrawlerConfig): Promise<string> {
  try {
    await mkdir(OUTPUT_DIR, { recursive: true });
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const jsonPath = join(OUTPUT_DIR, `eetimes-memory-designline_${timestamp}.json`);
    const jsonContent = JSON.stringify(results, null, 2);

    await writeFile(jsonPath, jsonContent, 'utf-8');
    console.log(`\n✅ Done. Results saved to ${jsonPath}`);

    // Write to OUTPUT_PATH if specified
    if (config.outputPath) {
      await mkdir(dirname(config.outputPath), { recursive: true });
      await writeFile(config.outputPath, jsonContent, 'utf-8');
      console.log(`   Also written to OUTPUT_PATH: ${config.outputPath}`);
    }

    return jsonPath;
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

    console.log(`Sent to n8n: ${response.status}`);

    if (!response.ok) {
      const responseText = await response.text();
      console.error('n8n response:', responseText.slice(0, 500));
      throw new Error(`Webhook returned status ${response.status}: ${responseText.slice(0, 200)}`);
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error(`Failed to send to n8n webhook ${webhookUrl}:`, errorMessage);
    throw error;
  }
}

// ─── Main Entry Point ────────────────────────────────────────────────────

/**
 * Main crawler function.
 *
 * Orchestrates browser setup, list page crawling, article extraction, and result saving.
 *
 * @throws Error if critical operations fail
 */
async function main(): Promise<void> {
  const config = parseCrawlerConfig();

  console.log('List URL:', LIST_URL);
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

    const results = await crawlListPages(page, config);
    await saveResults(results, config);

    if (config.webhookUrl) {
      await sendToWebhook(results, config.webhookUrl);
    }
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
