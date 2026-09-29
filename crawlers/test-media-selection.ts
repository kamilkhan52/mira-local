/**
 * Media Intelligence Selection Logic Test
 *
 * Runs the article selection prompt for every topic profile over a 7-day window,
 * using the cached source files in /configs/. Prints which articles each topic
 * selects so you can verify the topic-specific guidance is working correctly.
 *
 * Usage:
 *   ANTHROPIC_API_KEY=<key> npx tsx test-media-selection.ts
 *   ANCHOR_DATE=2026-04-21 npx tsx test-media-selection.ts   (override end date)
 *   LOOKBACK_DAYS=14 npx tsx test-media-selection.ts          (override window)
 *   PROFILE=storage-innovation npx tsx test-media-selection.ts (single profile)
 */

import { readFileSync } from 'fs';
import { resolve, join, dirname } from 'path';
import { fileURLToPath } from 'url';
import axios from 'axios';

// ─── Config ────────────────────────────────────────────────────────────────

const __dirname = dirname(fileURLToPath(import.meta.url));
const CONFIGS_DIR = resolve(__dirname, '../configs');
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY || '';
const ANCHOR_DATE_STR = (process.env.ANCHOR_DATE || new Date().toISOString()).slice(0, 10);
const LOOKBACK_DAYS = Number(process.env.LOOKBACK_DAYS ?? 7);
const PROFILE_FILTER = process.env.PROFILE || '';
const MAX_SELECT = 6;
const MODEL = 'claude-haiku-4-5-20251001';

if (!ANTHROPIC_API_KEY) {
  console.error('ERROR: ANTHROPIC_API_KEY not set');
  process.exit(1);
}

const anchorDate = new Date(`${ANCHOR_DATE_STR}T23:59:59Z`);
const fromDate = new Date(anchorDate);
fromDate.setDate(fromDate.getDate() - LOOKBACK_DAYS);

// ─── Types ──────────────────────────────────────────────────────────────────

interface RawArticle {
  url?: string;
  listTitle?: string;
  title?: string;
  content?: string;
  summary?: string;
  listDate?: string;
  crawledAt?: string;
  source?: string;
  [key: string]: unknown;
}

interface NormalizedArticle {
  index: number;
  source: string;
  title: string;
  snippet: string;
  url: string;
  listDate: string;
}

interface Profile {
  profile_id: string;
  topic: { name: string; focus: string };
  media?: {
    selection_guidance?: string;
    eetimes?: { list_url?: string };
    trendforce?: { list_url_template?: string };
  };
}

interface SelectionResult {
  indices: number[];
  reasoning: string;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function parseDate(raw: string): Date | null {
  if (!raw) return null;
  // Handle ISO strings, "Apr 21 2026", "April 20, 2026", "04.15.2026", etc.
  const cleaned = raw.trim().replace(/^(\d{2})\.(\d{2})\.(\d{4})$/, '$3-$2-$1');
  const d = new Date(cleaned);
  return isNaN(d.getTime()) ? null : d;
}

function isInRange(raw: string): boolean {
  const d = parseDate(raw);
  if (!d) return true; // include unknown dates rather than silently dropping
  return d >= fromDate && d <= anchorDate;
}

function loadSource(filename: string, sourceName: string): NormalizedArticle[] {
  const path = join(CONFIGS_DIR, filename);
  let raw: RawArticle[] = [];
  try {
    raw = JSON.parse(readFileSync(path, 'utf-8'));
  } catch {
    console.warn(`  [warn] Could not read ${filename}`);
    return [];
  }
  if (!Array.isArray(raw)) return [];

  return raw
    .map((a) => ({
      index: 0, // set after merge
      source: sourceName,
      title: (a.listTitle || a.title || '').trim(),
      snippet: ((a.content || a.summary || '') as string).slice(0, 400).replace(/\s+/g, ' ').trim(),
      url: (a.url || '').trim(),
      listDate: (a.listDate || a.crawledAt || '').slice(0, 24),
    }))
    .filter((a) => a.title && isInRange(a.listDate));
}

function mergeAndIndex(articleSets: NormalizedArticle[][]): NormalizedArticle[] {
  return articleSets.flat().map((a, i) => ({ ...a, index: i }));
}

// ─── LLM Call ───────────────────────────────────────────────────────────────

async function selectArticles(
  articles: NormalizedArticle[],
  topicFocus: string,
  selectionGuidance: string
): Promise<SelectionResult> {
  const articlesJson = JSON.stringify(
    articles.map((a) => ({
      index: a.index,
      source: a.source,
      title: a.title,
      snippet: a.snippet.slice(0, 300),
      listDate: a.listDate,
    })),
    null,
    2
  );

  const guidanceLine = selectionGuidance
    ? `\n\nTopic-specific guidance:\n${selectionGuidance}`
    : '';

  const system =
    `You are selecting the most relevant media articles about ${topicFocus}.` +
    `${guidanceLine}\n\n` +
    `Output ONLY valid JSON with no markdown fences.`;

  const userPrompt =
    `Select up to ${MAX_SELECT} articles most relevant to: ${topicFocus}\n` +
    `Date window: ${fromDate.toISOString().slice(0, 10)} → ${anchorDate.toISOString().slice(0, 10)}` +
    `${guidanceLine}\n\n` +
    `Articles (${articles.length} total):\n${articlesJson}\n\n` +
    `Output ONLY valid JSON:\n` +
    `{"selected_indices": [<integers matching index field>], "reasoning": "<one sentence>"}`;

  const response = await axios.post(
    'https://api.anthropic.com/v1/messages',
    {
      model: MODEL,
      max_tokens: 1024,
      system,
      messages: [{ role: 'user', content: userPrompt }],
    },
    {
      headers: {
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
      },
      timeout: 30_000,
    }
  );

  const raw = (response.data.content[0].text as string)
    .replace(/```json\n?/g, '')
    .replace(/```\n?/g, '')
    .trim();

  const parsed = JSON.parse(raw);
  return {
    indices: Array.isArray(parsed.selected_indices) ? parsed.selected_indices : [],
    reasoning: parsed.reasoning || '',
  };
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  // Load profiles
  const profilesFile = JSON.parse(
    readFileSync(join(CONFIGS_DIR, 'memory-innovation-profile.json'), 'utf-8')
  );
  const profiles: Profile[] = profilesFile.profiles;

  // Load all articles once
  const allArticles = mergeAndIndex([
    loadSource('digitimes-latest.json', 'digitimes'),
    loadSource('trendforce-latest.json', 'trendforce'),
    loadSource('eetimes-latest.json', 'eetimes'),
    loadSource('semianalysis-latest.json', 'semianalysis'),
  ]);

  const line = '─'.repeat(72);
  const dblLine = '═'.repeat(72);

  console.log(`\n${dblLine}`);
  console.log(`  MEDIA INTELLIGENCE SELECTION TEST`);
  console.log(`  Window : ${fromDate.toISOString().slice(0, 10)} → ${anchorDate.toISOString().slice(0, 10)} (${LOOKBACK_DAYS} days)`);
  console.log(`  Sources: digitimes(${loadSource('digitimes-latest.json', 'd').length}) trendforce(${loadSource('trendforce-latest.json', 't').length}) eetimes(${loadSource('eetimes-latest.json', 'e').length}) semianalysis(${loadSource('semianalysis-latest.json', 's').length})`);
  console.log(`  Total articles in window: ${allArticles.length}`);
  console.log(`  Model  : ${MODEL}`);
  console.log(`${dblLine}\n`);

  const profilesToTest = profiles.filter(
    (p) => !PROFILE_FILTER || p.profile_id === PROFILE_FILTER
  );

  for (const profile of profilesToTest) {
    const { profile_id, topic, media } = profile;
    const topicFocus = topic.focus;
    const selectionGuidance = media?.selection_guidance || '';

    console.log(`\n${dblLine}`);
    console.log(`  PROFILE  : ${profile_id}`);
    console.log(`  Focus    : ${topicFocus}`);
    if (selectionGuidance) {
      console.log(`  Guidance : ${selectionGuidance}`);
    }
    console.log(dblLine);

    // Show per-source count
    const bySource = allArticles.reduce<Record<string, number>>((acc, a) => {
      acc[a.source] = (acc[a.source] || 0) + 1;
      return acc;
    }, {});
    console.log(
      `  Articles : ${allArticles.length} total — ` +
        Object.entries(bySource)
          .map(([s, c]) => `${s}(${c})`)
          .join('  ')
    );

    if (allArticles.length === 0) {
      console.log('\n  No articles in date range — skipping.\n');
      continue;
    }

    console.log(`\n  Calling ${MODEL}...`);
    let result: SelectionResult;
    try {
      result = await selectArticles(allArticles, topicFocus, selectionGuidance);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`\n  ERROR: ${msg}\n`);
      continue;
    }

    console.log(`\n  SELECTED (${result.indices.length}/${MAX_SELECT}):`);
    if (result.reasoning) {
      console.log(`  Reasoning: ${result.reasoning}`);
    }
    console.log(line);

    for (const idx of result.indices) {
      const a = allArticles.find((x) => x.index === idx);
      if (!a) {
        console.log(`  [!] index ${idx} not found`);
        continue;
      }
      const dateStr = a.listDate.slice(0, 10);
      const src = a.source.padEnd(12);
      console.log(`  [${src}] (${dateStr})  ${a.title}`);
    }

    console.log(line);

    // Show what was NOT selected (to verify exclusions are working)
    const selectedSet = new Set(result.indices);
    const rejected = allArticles.filter((a) => !selectedSet.has(a.index));
    const rejectedBySource = rejected.reduce<Record<string, number>>((acc, a) => {
      acc[a.source] = (acc[a.source] || 0) + 1;
      return acc;
    }, {});
    console.log(
      `  Excluded : ${rejected.length} articles — ` +
        Object.entries(rejectedBySource)
          .map(([s, c]) => `${s}(${c})`)
          .join('  ')
    );

    // Brief pause between profiles to avoid rate limits
    await new Promise((r) => setTimeout(r, 800));
  }

  console.log(`\n${dblLine}`);
  console.log('  Test complete.');
  console.log(`${dblLine}\n`);
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
