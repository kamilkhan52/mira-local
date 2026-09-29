/**
 * TrendForce News Crawler (HTTP-only)
 *
 * Flow: list pages (SSR) → extract links + dates → optionally fetch each article and extract content.
 * Same N8nCrawlItem shape as EE Times / SemiAnalysis / Digitimes so n8n pipelines can reuse mapping.
 *
 * Usage:
 *   pnpm trendforce
 *   DATE_FROM=2026-01-01 DATE_TO=2026-03-11 MAX_ARTICLES=20 pnpm trendforce
 *   FETCH_ARTICLE_BODY=false pnpm trendforce  (list-only, content empty)
 *
 * Env:
 *   LIST_URL_TEMPLATE   URL template with {} as page placeholder (default: https://www.trendforce.com/news/page/{}/). Use to target a specific category, e.g. https://www.trendforce.com/news/category/semiconductors/page/{}/
 *   DATE_FROM           Start of date range (YYYY-MM-DD). Only include articles on or after this date.
 *   DATE_TO             End of date range (YYYY-MM-DD). Only include articles on or before this date.
 *   MAX_ARTICLES        Max articles to crawl (default 5). Use 0 to crawl all in range.
 *   OUTPUT_PATH         If set, also write n8n JSON to this path (e.g. /configs/trendforce-latest.json).
 *   OUTPUT_DIR          Directory for timestamped snapshots (default: ./output).
 *   CRAWLER_WEBHOOK_URL If set, POST the JSON array here (N8N_WEBHOOK_URL: deprecated alias). No POST by default.
 *   PAGE_START          First list page number (default 1).
 *   PAGE_END            Last list page number (default: crawl until no articles or non-200).
 *   FETCH_ARTICLE_BODY  Set to "false" to skip fetching article body (list-only; content will be empty).
 */

import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { mkdir, writeFile } from 'fs/promises';

import axios, { type AxiosInstance } from 'axios';
import Bottleneck from 'bottleneck';
import * as cheerio from 'cheerio';
import { JSDOM } from 'jsdom';
import { Readability } from '@mozilla/readability';

import {
  parseDate,
  isInDateRange,
  isEarlierThanDateFrom,
  CONTENT_THRESHOLDS,
  resolveOutputDir,
  resolveWebhookUrl,
} from './shared/util-functions.js';

// ─── Constants ─────────────────────────────────────────────────────────────

const __dirname = dirname(fileURLToPath(import.meta.url));
const BASE_URL = 'https://www.trendforce.com';
const LIST_URL_TEMPLATE = process.env.LIST_URL_TEMPLATE || `${BASE_URL}/news/page/{}/`;
const OUTPUT_DIR = resolveOutputDir(join(__dirname, 'output'));
const SOURCE_LABEL = 'trendforce';

const USER_AGENT =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36';

const RATE_LIMIT_MIN_TIME_MS = 1800;
const RATE_LIMIT_MAX_CONCURRENT = 1;
const REQUEST_TIMEOUT_MS = 15_000;

const LIST_SELECTORS = {
  item: '.insight-list-item',
  titleLink: 'h2.text-ellipsis-2 a',
  dateIcon: 'i.fa-calendar',
} as const;

/** Try common date formats from TrendForce list pages; fallback to parseDate. */
function parseListDate(dateStr: string | null | undefined): Date | null {
  const parsed = parseDate(dateStr);
  if (parsed) return parsed;
  if (!dateStr || !String(dateStr).trim()) return null;
  const s = String(dateStr).trim();
  // YYYY/MM/DD (common in Asia)
  const slash = /^(\d{4})\/(\d{1,2})\/(\d{1,2})/.exec(s);
  if (slash) {
    const d = new Date(Number(slash[1]), Number(slash[2]) - 1, Number(slash[3]));
    if (!Number.isNaN(d.getTime())) return d;
  }
  // DD-MM-YYYY or DD/MM/YYYY
  const dmy = /^(\d{1,2})[-\/](\d{1,2})[-\/](\d{4})/.exec(s);
  if (dmy) {
    const d = new Date(Number(dmy[3]), Number(dmy[2]) - 1, Number(dmy[1]));
    if (!Number.isNaN(d.getTime())) return d;
  }
  return null;
}

const ARTICLE_CONTENT_SELECTORS = [
  'article',
  '.article-content',
  '.article-body',
  '.content',
  '.post-content',
  '.entry-content',
  'main',
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
  source?: string;
}

interface ListArticle {
  url: string;
  title: string;
  date: string;
}

interface TrendForceConfig {
  dateFrom: string | null;
  dateTo: string | null;
  maxArticles: number;
  outputPath: string | null;
  webhookUrl: string | null;
  pageStart: number;
  pageEnd: number | null;
  fetchArticleBody: boolean;
}

interface ExtractedContent {
  articleContent: string;
  articleTitle: string;
  siteName: string;
}

// ─── Config ────────────────────────────────────────────────────────────────

function parseCrawlerConfig(): TrendForceConfig {
  const pageEndEnv = process.env.PAGE_END;
  return {
    dateFrom: process.env.DATE_FROM || null,
    dateTo: process.env.DATE_TO || null,
    maxArticles:
      process.env.MAX_ARTICLES !== undefined ? Number(process.env.MAX_ARTICLES) : 5,
    outputPath: process.env.OUTPUT_PATH || null,
    webhookUrl: resolveWebhookUrl(),
    pageStart:
      process.env.PAGE_START !== undefined ? Number(process.env.PAGE_START) : 1,
    pageEnd: pageEndEnv !== undefined && pageEndEnv !== '' ? Number(pageEndEnv) : null,
    fetchArticleBody: process.env.FETCH_ARTICLE_BODY !== 'false',
  };
}

// ─── HTTP Client ────────────────────────────────────────────────────────────

interface FetchResult {
  data: string;
  status: number;
}

function createRateLimitedClient(): (url: string) => Promise<FetchResult | null> {
  const limiter = new Bottleneck({
    minTime: RATE_LIMIT_MIN_TIME_MS,
    maxConcurrent: RATE_LIMIT_MAX_CONCURRENT,
  });

  const client: AxiosInstance = axios.create({
    headers: {
      'User-Agent': USER_AGENT,
      'Accept-Language': 'en-US,en;q=0.9',
      Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      Referer: 'https://www.google.com/',
    },
    timeout: REQUEST_TIMEOUT_MS,
    responseType: 'text',
  });

  return limiter.wrap(async (url: string): Promise<FetchResult | null> => {
    try {
      const response = await client.get<string>(url);
      return { data: response.data, status: response.status };
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`❌ Request failed: ${url} - ${message}`);
      return null;
    }
  });
}

// ─── URL helper ────────────────────────────────────────────────────────────

function resolveUrl(href: string): string {
  if (!href || !href.trim()) return '';
  const trimmed = href.trim();
  if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) return trimmed;
  const base = BASE_URL.endsWith('/') ? BASE_URL : BASE_URL + '/';
  const path = trimmed.startsWith('/') ? trimmed : '/' + trimmed;
  return new URL(path, base).href;
}

// ─── Phase 1: List crawl ────────────────────────────────────────────────────

function parseListPage(html: string, listPageUrl: string): ListArticle[] {
  const $ = cheerio.load(html);
  const articles: ListArticle[] = [];

  $(LIST_SELECTORS.item).each((_, el) => {
    const $el = $(el);
    const titleTag = $el.find(LIST_SELECTORS.titleLink).first();
    if (!titleTag.length) return;

    const href = titleTag.attr('href');
    const title = titleTag.text().trim();
    if (!href || !title) return;

    const link = resolveUrl(href);

    const dateIcon = $el.find(LIST_SELECTORS.dateIcon).first();
    let date = 'No Date';
    if (dateIcon.length) {
      const parentText = dateIcon.parent().text().trim();
      if (parentText) {
        // Parent often contains date + title + snippet + "View More"; keep only the date.
        const isoMatch = parentText.match(/\d{4}-\d{2}-\d{2}/);
        if (isoMatch) {
          date = isoMatch[0];
        } else {
          const firstLine = parentText.split(/\r?\n/)[0]?.trim() || parentText;
          date = firstLine.slice(0, 50); // avoid dragging in long text
        }
      }
    }

    articles.push({ url: link, title, date });
  });

  return articles;
}

async function discoverListArticles(
  fetch: (url: string) => Promise<FetchResult | null>,
  config: TrendForceConfig
): Promise<ListArticle[]> {
  console.log('🚀 Phase 1: List crawl...');
  const all: ListArticle[] = [];
  const seenUrls = new Set<string>();
  let page = config.pageStart;
  const endPage = config.pageEnd ?? 1e6;

  while (page <= endPage) {
    const url = LIST_URL_TEMPLATE.replace('{}', String(page));
    console.log(`   Fetching page ${page}: ${url}`);

    const result = await fetch(url);
    if (!result) break;
    if (result.status !== 200) {
      console.log(`   Non-200 status ${result.status}, stopping.`);
      break;
    }

    const items = parseListPage(result.data, url);
    if (items.length === 0) {
      console.log('   No articles on this page, stopping.');
      break;
    }

    let added = 0;
    for (const item of items) {
      if (seenUrls.has(item.url)) continue;
      const articleDate = parseListDate(item.date);
      // When date is unparseable and we have a range, stop to avoid crawling forever
      if (articleDate == null && (config.dateFrom || config.dateTo)) {
        console.log(
          `   ⏹️ Stopping: unparseable date "${item.date}" (cannot compare to range).`
        );
        page = endPage + 1;
        break;
      }
      if (isEarlierThanDateFrom(articleDate, config.dateFrom)) {
        console.log(
          `   ⏹️ Stopping: article date ${item.date} is earlier than DATE_FROM (${config.dateFrom}).`
        );
        page = endPage + 1;
        break;
      }
      if (!isInDateRange(articleDate, config.dateFrom, config.dateTo)) continue;
      seenUrls.add(item.url);
      all.push(item);
      added++;
      if (config.maxArticles > 0 && all.length >= config.maxArticles) break;
    }
    console.log(`   Found ${added} in range (total ${all.length}).`);
    if (config.maxArticles > 0 && all.length >= config.maxArticles) break;
    if (page > endPage) break; // stopped early due to date range
    page++;
  }

  const toUse = config.maxArticles > 0 ? all.slice(0, config.maxArticles) : all;
  console.log(`\n✅ Phase 1 done. ${toUse.length} articles to process.`);
  return toUse;
}

// ─── Phase 2: Article content extraction ────────────────────────────────────

function extractArticleContent(html: string, url: string): ExtractedContent {
  let articleContent = '';
  let articleTitle = '';
  let siteName = '';

  try {
    const dom = new JSDOM(html, { url });
    const reader = new Readability(dom.window.document);
    const article = reader.parse();

    if (article) {
      articleTitle = (article.title || '').trim();
      if (article.textContent?.trim()) {
        articleContent = article.textContent.trim();
      } else if (article.content) {
        const $c = cheerio.load(article.content);
        articleContent = $c.root().text().trim();
      }
      siteName = (article.siteName || '').trim();
      if (articleContent.length > CONTENT_THRESHOLDS.MIN_CONTENT_LENGTH) {
        return { articleContent, articleTitle, siteName };
      }
    }
  } catch (err) {
    console.warn(
      `Readability failed for ${url}:`,
      err instanceof Error ? err.message : String(err)
    );
  }

  const $ = cheerio.load(html);
  for (const selector of ARTICLE_CONTENT_SELECTORS) {
    const text = $(selector).first().text().trim();
    if (text.length > CONTENT_THRESHOLDS.MIN_CONTENT_LENGTH) {
      articleContent = text;
      break;
    }
  }
  if (!articleContent || articleContent.length < CONTENT_THRESHOLDS.MIN_CONTENT_LENGTH) {
    const paras = $('p')
      .slice(0, 20)
      .map((_, el) => $(el).text().trim())
      .get()
      .filter((t) => t.length >= CONTENT_THRESHOLDS.MIN_PARAGRAPH_LENGTH)
      .join(' ');
    if (paras.length > CONTENT_THRESHOLDS.MIN_SNIPPET_LENGTH) articleContent = paras;
  }
  if (!articleTitle) articleTitle = $('title').text().trim() || '';
  if (!siteName) {
    siteName =
      $('meta[property="og:site_name"]').attr('content') ||
      $('meta[name="application-name"]').attr('content') ||
      '';
  }
  return { articleContent, articleTitle, siteName };
}

async function fetchArticleBodies(
  fetch: (url: string) => Promise<FetchResult | null>,
  listArticles: ListArticle[],
  config: TrendForceConfig
): Promise<N8nCrawlItem[]> {
  if (!config.fetchArticleBody) {
    const crawledAt = new Date().toISOString();
    return listArticles.map((a) => ({
      url: a.url,
      listTitle: a.title,
      listAuthor: '',
      listDate: a.date,
      title: a.title,
      siteName: SOURCE_LABEL,
      content: '',
      crawledAt,
      source: SOURCE_LABEL,
    }));
  }

  console.log('\n🚀 Phase 2: Fetching article bodies...');
  const crawledAt = new Date().toISOString();
  const results: N8nCrawlItem[] = [];

  for (let i = 0; i < listArticles.length; i++) {
    const a = listArticles[i];
    console.log(`   [${i + 1}/${listArticles.length}] ${a.title.slice(0, 50)}…`);
    const result = await fetch(a.url);
    if (!result || result.status !== 200) {
      results.push({
        url: a.url,
        listTitle: a.title,
        listAuthor: '',
        listDate: a.date,
        title: a.title,
        siteName: SOURCE_LABEL,
        content: '',
        crawledAt,
        source: SOURCE_LABEL,
      });
      continue;
    }
    const extracted = extractArticleContent(result.data, a.url);
    results.push({
      url: a.url,
      listTitle: a.title,
      listAuthor: '',
      listDate: a.date,
      title: extracted.articleTitle || a.title,
      siteName: extracted.siteName || SOURCE_LABEL,
      content: extracted.articleContent,
      crawledAt,
      source: SOURCE_LABEL,
    });
    console.log(`   ✅ ${extracted.articleContent.length} chars`);
  }

  return results;
}

// ─── Output & Webhook ────────────────────────────────────────────────────────

async function saveResults(
  n8nItems: N8nCrawlItem[],
  config: TrendForceConfig
): Promise<{ fullPath: string; n8nPath: string }> {
  await mkdir(OUTPUT_DIR, { recursive: true });
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const fullPath = join(OUTPUT_DIR, `trendforce-crawl-${timestamp}.json`);
  const n8nPath = join(OUTPUT_DIR, `trendforce-n8n-${timestamp}.json`);
  const jsonContent = JSON.stringify(n8nItems, null, 2);

  await writeFile(fullPath, jsonContent, 'utf-8');
  console.log(`\n📁 Full output: ${fullPath}`);
  await writeFile(n8nPath, jsonContent, 'utf-8');
  console.log(`📁 n8n JSON: ${n8nPath}`);

  if (config.outputPath) {
    await mkdir(dirname(config.outputPath), { recursive: true });
    await writeFile(config.outputPath, jsonContent, 'utf-8');
    console.log(`   Also written to OUTPUT_PATH: ${config.outputPath}`);
  }

  return { fullPath, n8nPath };
}

async function sendToWebhook(n8nItems: N8nCrawlItem[], webhookUrl: string): Promise<void> {
  try {
    const response = await axios.post(webhookUrl, n8nItems, {
      headers: { 'Content-Type': 'application/json' },
      timeout: REQUEST_TIMEOUT_MS,
    });
    console.log(`   Sent to n8n webhook: ${response.status}`);
    if (!(response.status >= 200 && response.status < 300)) {
      console.error('   n8n response:', String(response.data).slice(0, 300));
      throw new Error(`Webhook returned status ${response.status}`);
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    console.error('   Failed to send to n8n:', message);
    throw error;
  }
}

// ─── Main ────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const config = parseCrawlerConfig();

  console.log('List URL template:', LIST_URL_TEMPLATE.replace('{}', 'N'));
  console.log('Max articles:', config.maxArticles === 0 ? 'all' : config.maxArticles);
  console.log('Fetch article body:', config.fetchArticleBody);
  if (config.dateFrom || config.dateTo) {
    console.log(
      'Date range:',
      [config.dateFrom || '(none)', config.dateTo || '(none)'].join(' ~ ')
    );
  }
  if (config.pageEnd !== null) {
    console.log('Page range:', config.pageStart, '..', config.pageEnd);
  }
  console.log('');

  const fetch = createRateLimitedClient();
  const listArticles = await discoverListArticles(fetch, config);
  const n8nItems = await fetchArticleBodies(fetch, listArticles, config);
  await saveResults(n8nItems, config);

  if (config.webhookUrl) {
    await sendToWebhook(n8nItems, config.webhookUrl);
  }

  console.log('🎉 Crawl finished.');
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
