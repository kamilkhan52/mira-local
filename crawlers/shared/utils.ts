/**
 * Shared utilities for crawler scripts
 *
 * Provides common functions for Chrome path detection, date parsing, and error handling.
 */

import { existsSync } from 'fs';

// ─── Constants ─────────────────────────────────────────────────────────────

export const CHROME_PATHS = {
  linux: ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome'],
  mac: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  chromium: '/Applications/Chromium.app/Contents/MacOS/Chromium',
} as const;

export const DEFAULT_DELAYS = {
  PAGE_LOAD_MS: 3000,
  ARTICLE_LOAD_MS: 2000,
  BETWEEN_ARTICLES_MS: 2000,
  PAGINATION_MS: 2000,
} as const;

export const DEFAULT_TIMEOUTS = {
  NAVIGATION_MS: 60_000,
  LONG_NAVIGATION_MS: 90_000,
} as const;

export const CONTENT_THRESHOLDS = {
  MIN_CONTENT_LENGTH: 200,
  MIN_TITLE_LENGTH: 10,
  MIN_PARAGRAPH_LENGTH: 20,
  MIN_SNIPPET_LENGTH: 50,
} as const;

// ─── Types ────────────────────────────────────────────────────────────────

export interface CrawlerConfig {
  maxArticles: number;
  headless: boolean;
  dateFrom: string | null;
  dateTo: string | null;
  outputPath: string | null;
  webhookUrl: string | null;
}

export interface ParsedAuthorDate {
  author: string;
  date: string;
}

// ─── Chrome Path Detection ────────────────────────────────────────────────

/**
 * Finds Chrome/Chromium executable path.
 *
 * Checks PUPPETEER_EXECUTABLE_PATH env var first, then common system paths.
 *
 * @returns Chrome executable path
 * @throws Error if Chrome not found
 */
export function getChromePath(): string {
  const envPath = process.env.PUPPETEER_EXECUTABLE_PATH;
  if (envPath && existsSync(envPath)) {
    return envPath;
  }

  for (const path of CHROME_PATHS.linux) {
    if (existsSync(path)) return path;
  }

  if (existsSync(CHROME_PATHS.mac)) return CHROME_PATHS.mac;
  if (existsSync(CHROME_PATHS.chromium)) return CHROME_PATHS.chromium;

  throw new Error(
    'Chrome not found. Set PUPPETEER_EXECUTABLE_PATH or install Chrome/Chromium.'
  );
}

// ─── Date Utilities ────────────────────────────────────────────────────────

/**
 * Parses date string to Date object.
 *
 * @param dateStr - Date string in various formats
 * @returns Date object or null if parsing fails
 */
export function parseDate(dateStr: string | null | undefined): Date | null {
  if (!dateStr || !String(dateStr).trim()) return null;
  const date = new Date(String(dateStr).trim());
  return Number.isNaN(date.getTime()) ? null : date;
}

/**
 * Checks if date is within specified range (inclusive, start/end of day).
 *
 * @param date - Date to check (Date object or null)
 * @param from - Start date string (YYYY-MM-DD) or null
 * @param to - End date string (YYYY-MM-DD) or null
 * @returns True if date is in range or no filter specified
 */
export function isInDateRange(
  date: Date | null,
  from: string | null | undefined,
  to: string | null | undefined
): boolean {
  if (!from && !to) return true;
  if (date == null) return true;

  const timestamp = date.getTime();
  if (Number.isNaN(timestamp)) return true;

  if (from) {
    const fromStart = new Date(from).setHours(0, 0, 0, 0);
    if (timestamp < fromStart) return false;
  }

  if (to) {
    const toEnd = new Date(to).setHours(23, 59, 59, 999);
    if (timestamp > toEnd) return false;
  }

  return true;
}

/**
 * Checks if date is earlier than DATE_FROM (for early stopping in newest-first lists).
 *
 * @param date - Date to check (Date object or null)
 * @param dateFrom - Start date string (YYYY-MM-DD) or null
 * @returns True if date is earlier than dateFrom
 */
export function isEarlierThanDateFrom(
  date: Date | null,
  dateFrom: string | null | undefined
): boolean {
  if (!dateFrom || date == null) return false;

  const timestamp = date.getTime();
  if (Number.isNaN(timestamp)) return false;

  const fromStart = new Date(dateFrom).setHours(0, 0, 0, 0);
  return timestamp < fromStart;
}

// ─── Async Utilities ──────────────────────────────────────────────────────

/**
 * Sleep for specified milliseconds.
 *
 * @param ms - Milliseconds to sleep
 * @returns Promise that resolves after delay
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ─── Output / Webhook / Browser Shims ─────────────────────────────────────

/**
 * Resolves the optional webhook the crawler POSTs its results to.
 *
 * Crawlers no longer POST anywhere by default (they used to target a local
 * n8n webhook, which failed the whole crawl with exit 1 when n8n was not
 * running). Set CRAWLER_WEBHOOK_URL to opt in. N8N_WEBHOOK_URL is accepted as
 * a deprecated alias; an empty value disables the POST either way.
 *
 * @returns Webhook URL or null when no POST should happen
 */
export function resolveWebhookUrl(): string | null {
  const explicit = (process.env.CRAWLER_WEBHOOK_URL || '').trim();
  if (explicit) return explicit;
  const legacy = (process.env.N8N_WEBHOOK_URL || '').trim();
  if (legacy) {
    console.warn('N8N_WEBHOOK_URL is deprecated; set CRAWLER_WEBHOOK_URL instead.');
    return legacy;
  }
  return null;
}

/**
 * Directory for timestamped crawl snapshots. OUTPUT_DIR (set by the Python
 * pipeline to its data temp dir) wins; otherwise the crawler-local fallback.
 *
 * @param fallback - Default directory (usually <crawlers>/output)
 * @returns Directory path
 */
export function resolveOutputDir(fallback: string): string {
  const dir = (process.env.OUTPUT_DIR || '').trim();
  return dir || fallback;
}

/**
 * tsx/esbuild compiles with keepNames, which wraps named inner functions in
 * `__name(fn, "name")`. Functions passed to page.evaluate are serialized and
 * run in the browser, where that helper does not exist, so evaluate throws
 * `ReferenceError: __name is not defined`. Defining a no-op `__name` in every
 * document before any page script runs makes the serialized code valid.
 * Must be called before the first navigation.
 */
export const ESBUILD_NAME_SHIM =
  'globalThis.__name = globalThis.__name || ((target) => target);';

export async function installEsbuildNameShim(page: {
  evaluateOnNewDocument: (source: string) => Promise<unknown>;
}): Promise<void> {
  await page.evaluateOnNewDocument(ESBUILD_NAME_SHIM);
}

// ─── Configuration Parsing ────────────────────────────────────────────────

/**
 * Parses environment variables into crawler configuration.
 *
 * @returns CrawlerConfig object
 */
export function parseCrawlerConfig(): CrawlerConfig {
  return {
    maxArticles:
      process.env.MAX_ARTICLES !== undefined
        ? Number(process.env.MAX_ARTICLES)
        : 3,
    headless: process.env.HEADLESS !== 'false',
    dateFrom: process.env.DATE_FROM || null,
    dateTo: process.env.DATE_TO || null,
    outputPath: process.env.OUTPUT_PATH || null,
    webhookUrl: resolveWebhookUrl(),
  };
}
