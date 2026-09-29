/**
 * SemiAnalysis Newsletter Crawler (Full Archive Edition)
 *
 * Strategy: "Infinite History"
 * 1. Discovery: Loops through the hidden Archive API to get article slugs.
 * 2. Detail: Fetches full content for each slug via the Post API.
 *
 * Usage:
 *   pnpm semianalysis
 *   DATE_FROM=2026-01-01 DATE_TO=2026-01-31 pnpm semianalysis
 *
 * Env:
 *   DATE_FROM      Start of date range (YYYY-MM-DD). Only include articles on or after this date.
 *   DATE_TO        End of date range (YYYY-MM-DD). Only include articles on or before this date.
 *   MAX_ARTICLES   Max articles to crawl (default 5). Use 0 to crawl all in range.
 *   OUTPUT_PATH    If set, also write n8n JSON array to this path (e.g. /configs/semianalysis-latest.json).
 *   OUTPUT_DIR      Directory for timestamped snapshots (default: ./output).
 *   CRAWLER_WEBHOOK_URL If set, POST the JSON array here (N8N_WEBHOOK_URL: deprecated alias). No POST by default.
 */

import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { mkdir, writeFile } from 'fs/promises';

import axios, { type AxiosInstance } from 'axios';
import Bottleneck from 'bottleneck';
import * as cheerio from 'cheerio';

import {
  parseDate,
  isInDateRange,
  isEarlierThanDateFrom,
  resolveOutputDir,
  resolveWebhookUrl,
  type CrawlerConfig,
} from './shared/util-functions.js';

// ─── Constants ─────────────────────────────────────────────────────────────

const __dirname = dirname(fileURLToPath(import.meta.url));
const BASE_URL = 'https://newsletter.semianalysis.com';
const OUTPUT_DIR = resolveOutputDir(join(__dirname, 'output'));
const API_ARCHIVE = `${BASE_URL}/api/v1/archive`;
const API_POSTS_BASE = `${BASE_URL}/api/v1/posts`;

const USER_AGENT =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';

const SOURCE_LABEL = 'semianalysis';
const DEFAULT_AUTHOR = 'SemiAnalysis';

// Rate limiting: 1 request every 1.5 seconds
const RATE_LIMIT_MIN_TIME_MS = 1500;
const RATE_LIMIT_MAX_CONCURRENT = 1;
const API_ARCHIVE_LIMIT = 50; // Max allowed by Substack
const API_TIMEOUT_MS = 30_000; // 30 seconds timeout for API requests
const SNIPPET_MAX_LENGTH = 200;

// ─── Types ─────────────────────────────────────────────────────────────────

/** Response from the Archive API (list of articles) */
interface ArchiveEntry {
  id: number;
  title: string;
  slug: string;
  post_date: string;
  type: string; // 'newsletter', 'podcast', etc.
}

/** Response from the Detail API (single article) */
interface SubstackApiResponse {
  id?: number;
  title?: string;
  body_html?: string;
  post_date?: string;
  reaction_count?: number;
  [key: string]: unknown;
}

interface Article {
  slug: string;
  title: string;
  publishedAt: string;
  likes: number;
  cleanSnippet: string;
  bodyText: string;
}

/** One item per article, same shape as EE Times crawler so n8n pipelines can reuse mapping. */
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
  likes?: number;
}

interface SemiAnalysisConfig {
  dateFrom: string | null;
  dateTo: string | null;
  maxArticles: number;
  outputPath: string | null;
  webhookUrl: string | null;
}

// ─── Helpers ───────────────────────────────────────────────────────────────

/**
 * Strip HTML tags and normalize whitespace.
 *
 * @param html - HTML string to process
 * @returns Plain text with normalized whitespace
 */
function stripHtml(html: string): string {
  const $ = cheerio.load(html);
  return $('body').text().replace(/\s+/g, ' ').trim();
}

/**
 * Truncate text to maximum length with ellipsis.
 *
 * @param text - Text to truncate
 * @param maxLength - Maximum length
 * @returns Truncated text with ellipsis if needed
 */
function truncateSnippet(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + '…';
}

/**
 * Build canonical post URL (Substack uses /p/<slug>).
 *
 * @param slug - Article slug
 * @returns Full URL to article
 */
function postUrl(slug: string): string {
  return `${BASE_URL}/p/${slug}`;
}

/**
 * Convert crawled articles to n8n-ready array (same shape as EE Times webhook payload).
 *
 * @param articles - Array of crawled articles
 * @param crawledAt - ISO timestamp of crawl
 * @returns Array of n8n-compatible items
 */
function articlesToN8nItems(articles: Article[], crawledAt: string): N8nCrawlItem[] {
  return articles.map((article) => ({
    url: postUrl(article.slug),
    listTitle: article.title,
    listAuthor: DEFAULT_AUTHOR,
    listDate: article.publishedAt,
    title: article.title,
    siteName: SOURCE_LABEL,
    content: article.bodyText,
    crawledAt,
    source: SOURCE_LABEL,
    likes: article.likes,
  }));
}

/**
 * Parse crawler configuration from environment variables.
 *
 * @returns SemiAnalysisConfig object
 */
function parseCrawlerConfig(): SemiAnalysisConfig {
  return {
    dateFrom: process.env.DATE_FROM || null,
    dateTo: process.env.DATE_TO || null,
    maxArticles:
      process.env.MAX_ARTICLES !== undefined ? Number(process.env.MAX_ARTICLES) : 5,
    outputPath: process.env.OUTPUT_PATH || null,
    webhookUrl: resolveWebhookUrl(),
  };
}

// ─── API Client Setup ──────────────────────────────────────────────────────

/**
 * Create rate-limited Axios client.
 *
 * @returns Rate-limited fetch function
 */
function createRateLimitedClient(): (url: string) => Promise<unknown> {
  const limiter = new Bottleneck({
    minTime: RATE_LIMIT_MIN_TIME_MS,
    maxConcurrent: RATE_LIMIT_MAX_CONCURRENT,
  });

  const client: AxiosInstance = axios.create({
    headers: {
      'User-Agent': USER_AGENT,
      Accept: 'application/json',
    },
    timeout: API_TIMEOUT_MS,
  });

  // Wrapper to rate-limit any axios call
  return limiter.wrap(async (url: string) => {
    try {
      const response = await client.get(url);
      return response.data;
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error(`❌ Request failed: ${url} - ${errorMessage}`);
      return null;
    }
  });
}

// ─── Discovery Phase ───────────────────────────────────────────────────────

/**
 * Fetch archive page and extract slugs in date range.
 *
 * @param fetch - Rate-limited fetch function
 * @param offset - Archive offset
 * @param config - Crawler configuration
 * @returns Object with slugs array and shouldStop flag
 */
async function fetchArchivePage(
  fetch: (url: string) => Promise<unknown>,
  offset: number,
  config: SemiAnalysisConfig
): Promise<{ slugs: string[]; shouldStop: boolean }> {
  const archiveUrl = `${API_ARCHIVE}?sort=new&search=&offset=${offset}&limit=${API_ARCHIVE_LIMIT}`;
  process.stdout.write(`   Fetching offset ${offset}... `);

  const data = (await fetch(archiveUrl)) as ArchiveEntry[] | null;

  if (!data || data.length === 0) {
    console.log('\n🏁 End of archive reached.');
    return { slugs: [], shouldStop: true };
  }

  const slugs: string[] = [];
  let shouldStop = false;

  for (const item of data) {
    const postDate = item.post_date ? parseDate(item.post_date) : null;

    if (isEarlierThanDateFrom(postDate, config.dateFrom)) {
      console.log(
        `\n   ⏹️ Stopping: article date ${item.post_date || '(no date)'} is earlier than DATE_FROM (${config.dateFrom}).`
      );
      shouldStop = true;
      break;
    }

    if (!isInDateRange(postDate, config.dateFrom, config.dateTo)) {
      continue;
    }

    slugs.push(item.slug);
  }

  console.log(`Found ${slugs.length} in range.`);
  return { slugs, shouldStop };
}

/**
 * Discover all article slugs in date range.
 *
 * @param fetch - Rate-limited fetch function
 * @param config - Crawler configuration
 * @returns Array of article slugs
 */
async function discoverArticleSlugs(
  fetch: (url: string) => Promise<unknown>,
  config: SemiAnalysisConfig
): Promise<string[]> {
  console.log('🚀 Starting Full Archive Discovery...');

  const allSlugs: string[] = [];
  let offset = 0;
  let keepGoing = true;

  while (keepGoing) {
    const { slugs, shouldStop } = await fetchArchivePage(fetch, offset, config);

    // Add slugs from this batch before checking if we should stop
    allSlugs.push(...slugs);

    if (shouldStop) {
      keepGoing = false;
      break;
    }

    offset += API_ARCHIVE_LIMIT;
  }

  console.log(`\n✅ Discovery Complete. Found ${allSlugs.length} total articles.`);
  console.log(`   Starting Phase 2: Detail Extraction...`);

  return allSlugs;
}

// ─── Detail Extraction Phase ───────────────────────────────────────────────

/**
 * Fetch and parse article detail from API.
 *
 * @param fetch - Rate-limited fetch function
 * @param slug - Article slug
 * @returns Article object or null if fetch fails
 */
async function fetchArticleDetail(
  fetch: (url: string) => Promise<unknown>,
  slug: string
): Promise<Article | null> {
  const detailUrl = `${API_POSTS_BASE}/${slug}`;
  const postData = (await fetch(detailUrl)) as SubstackApiResponse | null;

  if (!postData) {
    return null;
  }

  const bodyHtml = postData.body_html || '';
  const cleanText = stripHtml(bodyHtml);
  const snippet = truncateSnippet(cleanText, SNIPPET_MAX_LENGTH);

  return {
    slug,
    title: postData.title ?? '(no title)',
    publishedAt: postData.post_date ?? '',
    likes: postData.reaction_count ?? 0,
    cleanSnippet: snippet,
    bodyText: cleanText,
  };
}

/**
 * Extract details for multiple articles.
 *
 * @param fetch - Rate-limited fetch function
 * @param slugs - Array of article slugs
 * @param maxArticles - Maximum articles to fetch (0 = all)
 * @returns Array of article objects
 */
async function extractArticleDetails(
  fetch: (url: string) => Promise<unknown>,
  slugs: string[],
  maxArticles: number
): Promise<Article[]> {
  const slugsToFetch = maxArticles > 0 ? slugs.slice(0, maxArticles) : slugs;
  const articles: Article[] = [];

  for (const slug of slugsToFetch) {
    const article = await fetchArticleDetail(fetch, slug);

    if (article) {
      articles.push(article);

      console.log('---------------------------------------------------');
      console.log(`📄 Title:   ${article.title}`);
      console.log(`🔗 Slug:    ${slug}`);
      console.log(`📅 Date:    ${article.publishedAt}`);
      console.log(`👍 Likes:   ${article.likes}`);
      console.log(`📝 Content: ${article.cleanSnippet}`);
    }
  }

  return articles;
}

// ─── Output & Webhook ──────────────────────────────────────────────────────

/**
 * Save results to files.
 *
 * @param articles - Array of crawled articles
 * @param n8nItems - Array of n8n-compatible items
 * @param config - Crawler configuration
 * @returns Paths to saved files
 * @throws Error if file write fails
 */
async function saveResults(
  articles: Article[],
  n8nItems: N8nCrawlItem[],
  config: SemiAnalysisConfig
): Promise<{ fullPath: string; n8nPath: string }> {
  try {
    await mkdir(OUTPUT_DIR, { recursive: true });
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

    // Full data (articles + metadata) for debugging
    const fullPath = join(OUTPUT_DIR, `semianalysis-crawl-${timestamp}.json`);
    const crawledAt = new Date().toISOString();
    await writeFile(fullPath, JSON.stringify({ crawledAt, articles }, null, 2), 'utf-8');
    console.log(`\n📁 Full output: ${fullPath}`);

    // n8n-ready JSON (array of items, same shape as EE Times webhook)
    const n8nPath = join(OUTPUT_DIR, `semianalysis-n8n-${timestamp}.json`);
    await writeFile(n8nPath, JSON.stringify(n8nItems, null, 2), 'utf-8');
    console.log(`📁 n8n JSON: ${n8nPath}`);

    // Write to OUTPUT_PATH if specified
    if (config.outputPath) {
      await mkdir(dirname(config.outputPath), { recursive: true });
      await writeFile(config.outputPath, JSON.stringify(n8nItems, null, 2), 'utf-8');
      console.log(`   Also written to OUTPUT_PATH: ${config.outputPath}`);
    }

    return { fullPath, n8nPath };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to save results: ${errorMessage}`);
  }
}

/**
 * Send results to n8n webhook.
 *
 * @param n8nItems - Array of n8n-compatible items
 * @param webhookUrl - Webhook URL
 * @throws Error if webhook request fails
 */
async function sendToWebhook(n8nItems: N8nCrawlItem[], webhookUrl: string): Promise<void> {
  try {
    const response = await axios.post(webhookUrl, n8nItems, {
      headers: { 'Content-Type': 'application/json' },
      timeout: API_TIMEOUT_MS,
    });

    console.log(`   Sent to n8n webhook: ${response.status}`);

    if (!(response.status >= 200 && response.status < 300)) {
      console.error('   n8n response:', String(response.data).slice(0, 300));
      throw new Error(`Webhook returned status ${response.status}`);
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error('   Failed to send to n8n:', errorMessage);
    throw error;
  }
}

// ─── Main Entry Point ────────────────────────────────────────────────────────

/**
 * Main crawler function.
 *
 * Orchestrates discovery, detail extraction, and result saving.
 *
 * @throws Error if critical operations fail
 */
async function main(): Promise<void> {
  const config = parseCrawlerConfig();

  console.log('Max articles:', config.maxArticles === 0 ? 'all' : config.maxArticles);
  if (config.dateFrom || config.dateTo) {
    console.log('Date range:', [config.dateFrom || '(none)', config.dateTo || '(none)'].join(' ~ '));
  }
  console.log('');

  // Setup rate-limited API client
  const fetch = createRateLimitedClient();

  // Phase 1: Discovery
  const slugs = await discoverArticleSlugs(fetch, config);

  // Phase 2: Detail Extraction
  const articles = await extractArticleDetails(fetch, slugs, config.maxArticles);

  // Save results
  const crawledAt = new Date().toISOString();
  const n8nItems = articlesToN8nItems(articles, crawledAt);
  await saveResults(articles, n8nItems, config);

  // Send to webhook if configured
  if (config.webhookUrl) {
    await sendToWebhook(n8nItems, config.webhookUrl);
  }

  console.log('🎉 Crawl finished.');
}

// ─── Execution ────────────────────────────────────────────────────────────

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
