"""Verbatim n8n Code-node JavaScript (live workflow "Memory Innovation
Research Assistant", iz3yMcSlkWIQhRmn) plus a tiny Node.js harness that runs
a node body against fixture items, so the Python port can be compared with
the real n8n logic. Tests that need it skip when `node` is not installed.

Only what these nodes touch is emulated: $json, $input, items, $('Node'),
$prevNode, and a minimal Luxon DateTime (fromISO / minus / toFormat / isValid).
"""
import json
import shutil
import subprocess

import pytest

NODE_BIN = shutil.which("node")
requires_node = pytest.mark.skipif(NODE_BIN is None, reason="node is not installed")

_HARNESS = r'''
const fs = require('fs');
const spec = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
console.log = (...a) => process.stderr.write(a.join(' ') + '\n');  // keep stdout for the result
class DT {
  constructor(d) { this.d = d; this.isValid = !!d; }
  static fromISO(s) {
    const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(s));
    if (!m) return new DT(null);
    const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
    if (d.getUTCMonth() !== +m[2] - 1 || d.getUTCDate() !== +m[3]) return new DT(null);
    return new DT(d);
  }
  minus({ days }) { return new DT(new Date(this.d.getTime() - days * 86400000)); }
  toFormat(f) {
    const y = this.d.getUTCFullYear();
    const m = String(this.d.getUTCMonth() + 1).padStart(2, '0');
    const dd = String(this.d.getUTCDate()).padStart(2, '0');
    return f === 'yyyy-MM-dd' ? `${y}-${m}-${dd}` : `${y}${m}${dd}`;
  }
}
const wrap = (arr) => ({ first: () => ({ json: arr[0] }), all: () => arr.map(j => ({ json: j })) });
const $ = (name) => {
  if (!(name in spec.nodes)) throw new Error(`node ${name} not in fixture`);
  return wrap(spec.nodes[name]);
};
const input = spec.input || [];
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
// `items` is a sandbox global in n8n (node bodies may shadow it with const).
globalThis.items = input.map(j => ({ json: j }));
const fn = new AsyncFunction('$json', '$input', '$', '$prevNode', 'DateTime', '$now', spec.code);
fn(input[0] || {}, wrap(input), $, { name: spec.prevNode || '' }, DT, DT.fromISO(spec.now))
  .then((out) => { process.stdout.write(JSON.stringify({ ok: out })); })
  .catch((e) => { process.stdout.write(JSON.stringify({ error: String(e && e.message || e) })); });
'''


def run_node(tmp_path, name: str, input_items=None, nodes=None, now: str = "2026-09-28",
             prev_node: str = ""):
    """Run NODE_JS[name]; returns the node's return value with each n8n item
    unwrapped to its json, or raises RuntimeError with the node's error."""
    harness = tmp_path / "n8n_harness.js"
    harness.write_text(_HARNESS)
    spec = tmp_path / "n8n_spec.json"
    spec.write_text(json.dumps({"code": NODE_JS[name], "input": input_items or [],
                                "nodes": nodes or {}, "now": now, "prevNode": prev_node}))
    proc = subprocess.run([NODE_BIN, str(harness), str(spec)], capture_output=True, text=True,
                          timeout=60)
    if proc.returncode != 0:
        raise AssertionError(f"node harness crashed: {proc.stderr}")
    result = json.loads(proc.stdout)
    if "error" in result:
        raise RuntimeError(result["error"])
    out = result["ok"]
    if isinstance(out, list):
        return [o.get("json", o) if isinstance(o, dict) else o for o in out]
    if isinstance(out, dict) and "json" in out:
        return out["json"]
    return out


NODE_JS = {
    'Set Run Mode': r'''const config = $json.config || {};
const triggerNode = ($json.triggerNode || $prevNode.name || '').toLowerCase();

let mode;
if (triggerNode.includes('daily')) {
  mode = 'daily';
} else if (triggerNode.includes('weekly')) {
  mode = 'weekly';
} else if (triggerNode.includes('monthly')) {
  mode = 'monthly';
} else {
  mode = config.default_mode || 'weekly';
}

const parseBool = (value, fallback = false) => {
  if (value === undefined || value === null || value === '') return fallback;
  return [true, 'true', 1, '1', 'yes', 'on'].includes(value);
};

const rawTestOverride = $json.test_mode_override;
const isTestMode = parseBool(rawTestOverride, false);
const rawLlmCacheBypass = $json.llm_cache_bypass;
const llmCacheBypass = parseBool(rawLlmCacheBypass, false);
const rawTrendEnabledOverride = $json.trend_enabled_override;
const trendEnabled = rawTrendEnabledOverride !== undefined && rawTrendEnabledOverride !== null && rawTrendEnabledOverride !== ''
  ? parseBool(rawTrendEnabledOverride, true)
  : (((config.modes || {})[mode] || {}).trend_enabled !== undefined ? ((config.modes || {})[mode] || {}).trend_enabled : true);

const profileId = config.profile_id || config.topic?.name || 'default';
const profileSlug = String(profileId).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'default';

const modeConfig = (config.modes && config.modes[mode]) || {};
const overrideDays = Number($json.lookback_days_override);
const hasOverride = Number.isFinite(overrideDays) && overrideDays > 0;
const lookbackDays = hasOverride
  ? overrideDays
  : (modeConfig.lookback_days ?? (mode === 'daily' ? 1 : mode === 'weekly' ? 7 : 30));
const rangeDays = Math.max(0, lookbackDays - 1);
const selectionRangeLabel = modeConfig.selection_range_label || `${modeConfig.selection_min || 0}-${modeConfig.selection_max || 0}`.replace(/^0-0$/, mode === 'daily' ? '2-5' : '5-10');
const maxSelection = modeConfig.max_selection || (mode === 'daily' ? 10 : mode === 'weekly' ? 20 : 20);
const reportSelectionRangeLabel = modeConfig.report_selection_range_label || `${modeConfig.report_selection_min || 0}-${modeConfig.report_selection_max || 0}`.replace(/^0-0$/, mode === 'daily' ? '3-5' : '7-10');
const reportMaxSelection = Number(modeConfig.report_max_selection || (mode === 'daily' ? 5 : 10));
const maxLimitOverride = $json.max_limit ?? $json.maxLimit ?? null;

const rawCurrentDateOverride = ($json.current_date_override || '').toString().slice(0, 10);
const validCurrentDateOverride = /^\d{4}-\d{2}-\d{2}$/.test(rawCurrentDateOverride) ? rawCurrentDateOverride : '';
const anchorDate = validCurrentDateOverride || $now.toFormat('yyyy-MM-dd');

const buildDurationLabel = (days) => {
  const d = Number(days) || 0;
  if (d <= 0) return 'recent period';
  if (d === 1) return 'last day';
  if (d === 7) return 'last week';
  if (d % 30 === 0) {
    const months = Math.round(d / 30);
    return months === 1 ? 'last month' : `last ${months} months`;
  }
  if (d % 7 === 0 && d < 60) {
    const weeks = Math.round(d / 7);
    return weeks === 1 ? 'last week' : `last ${weeks} weeks`;
  }
  return `last ${d} days`;
};

const toTitleCase = (text) => (text || '')
  .split(' ')
  .map(word => word ? word[0].toUpperCase() + word.slice(1) : '')
  .join(' ')
  .trim();

const periodLabel = buildDurationLabel(lookbackDays);
const periodTitle = toTitleCase(periodLabel);
const digestLabel = `${config.topic?.name || 'Research'} Digest (${periodTitle})`;
const anchorDt = DateTime.fromISO(anchorDate);
const endDate = anchorDt.isValid ? anchorDt.toFormat('yyyy-MM-dd') : $now.toFormat('yyyy-MM-dd');
const startDate = anchorDt.isValid
  ? anchorDt.minus({ days: rangeDays }).toFormat('yyyy-MM-dd')
  : $now.minus({ days: rangeDays }).toFormat('yyyy-MM-dd');
const periodRange = `${startDate} to ${endDate}`;

return [{
  json: {
    ...$json,
    mode,
    isTestMode,
    llm_cache_bypass: llmCacheBypass,
    trend_enabled: trendEnabled,
    profileId,
    profileSlug,
    lookbackDays,
    periodLabel,
    periodTitle,
    periodRange,
    digestLabel,
    selectionRangeLabel,
    maxSelection,
    reportSelectionRangeLabel,
    reportMaxSelection,
    maxLimit: maxLimitOverride,
    topicName: config.topic?.name || 'Research',
    topicFocus: config.topic?.focus || 'research',
    assistantSignature: config.topic?.assistant_signature || 'Research Assistant',
    currentDate: endDate,
    config
  }
}];
''',
    'Score Completeness Gate': r'''const thresholdRelevance = Number($('Set Run Mode').first().json.config?.thresholds?.relevance_score_min ?? 1);
const thresholdCredibility = Number($('Set Run Mode').first().json.config?.thresholds?.credibility_tier_min ?? 1);

const toNumber = (v) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

const normalizeId = (value) => String(value || '')
  .replace(/^https?:\/\/arxiv\.org\/abs\//, '')
  .replace(/^https?:\/\/arxiv\.org\/pdf\//, '')
  .replace(/\.pdf$/i, '');

const parseOutput = (data) => {
  const out = data?.output;
  if (out && typeof out === 'object') return out;
  if (typeof out === 'string') {
    try {
      const cleaned = out.replace(/```json\s*\n?/gi, '').replace(/```\s*$/i, '').trim();
      const parsed = JSON.parse(cleaned);
      return parsed && typeof parsed === 'object' ? parsed : null;
    } catch (_) {
      return null;
    }
  }
  return null;
};

const originalItems = $('Combine Original Data with PDF Extraction').all().map(i => i.json || {});
const classifiedItems = items.map(i => i.json || {});

const classifiedById = new Map();
for (const data of classifiedItems) {
  const parsed = parseOutput(data);
  const key = normalizeId(data.id || data.arxiv_id || parsed?.arxiv_id || '');
  if (!key || !parsed) continue;

  classifiedById.set(key, {
    source: data,
    output: {
      arxiv_id: parsed.arxiv_id || data.id || data.arxiv_id || '',
      primary_topic: parsed.primary_topic || '',
      secondary_topics: Array.isArray(parsed.secondary_topics) ? parsed.secondary_topics : [],
      potential_impact: parsed.potential_impact || '',
      relevance_score: toNumber(parsed.relevance_score),
      key_findings: parsed.key_findings || '',
      actionable: parsed.actionable || ''
    }
  });
}

const valid = [];
const dropped = [];

for (const base of originalItems) {
  const key = normalizeId(base.id || base.arxiv_id || base.output?.arxiv_id || '');
  const cls = classifiedById.get(key);

  const merged = cls
    ? {
        ...base,
        ...cls.source,
        output: { ...(base.output || {}), ...(cls.output || {}) },
      }
    : { ...base };

  const rel = toNumber(merged.output?.relevance_score ?? merged.relevance_score);
  const cred = toNumber(merged.credibility_tier);

  if (rel === null || cred === null) {
    dropped.push({
      arxiv_id: key,
      missing_relevance_score: rel === null,
      missing_credibility_tier: cred === null
    });
    continue;
  }

  valid.push({
    json: {
      ...merged,
      output: {
        ...(merged.output || {}),
        relevance_score: rel
      },
      credibility_tier: cred,
      score_validation: {
        has_relevance_score: true,
        has_credibility_tier: true,
        dropped_count: 0,
        threshold_relevance: thresholdRelevance,
        threshold_credibility: thresholdCredibility,
        originals_count: originalItems.length,
        classified_count: classifiedById.size
      }
    }
  });
}

const droppedCount = dropped.length;
for (const item of valid) {
  item.json.score_validation.dropped_count = droppedCount;
}

if (droppedCount > 0) {
  console.log(`[Score Completeness Gate] Dropped ${droppedCount} papers missing required scores before threshold filtering.`);
}

return valid;
''',
    'Rank and Cap Filtered Papers': r'''const runMode = $('Set Run Mode').first().json || {};
const cfg = runMode.config || {};

const maxPapers = Number(cfg.thresholds?.max_filtered_papers ?? 100);
const relevanceWeightRaw = Number(cfg.thresholds?.combined_relevance_weight ?? 0.6);
const credibilityWeightRaw = Number(cfg.thresholds?.combined_credibility_weight ?? 0.4);
const weightSum = relevanceWeightRaw + credibilityWeightRaw;
const relevanceWeight = weightSum > 0 ? relevanceWeightRaw / weightSum : 0.6;
const credibilityWeight = weightSum > 0 ? credibilityWeightRaw / weightSum : 0.4;

const toNum = (v, d = 0) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
};

const ranked = items
  .map((item) => {
    const j = item.json || {};
    const relevance = toNum(j.output?.relevance_score ?? j.relevance_score, 0);
    const credibility = toNum(j.credibility_tier, 0);
    const combinedScore = (relevance * relevanceWeight) + (credibility * credibilityWeight);

    return {
      json: {
        ...j,
        combined_score: Number(combinedScore.toFixed(4)),
        score_weights: {
          relevance: Number(relevanceWeight.toFixed(4)),
          credibility: Number(credibilityWeight.toFixed(4))
        }
      }
    };
  })
  .sort((a, b) => {
    const aScore = toNum(a.json.combined_score, 0);
    const bScore = toNum(b.json.combined_score, 0);
    if (bScore !== aScore) return bScore - aScore;

    const aRel = toNum(a.json.output?.relevance_score ?? a.json.relevance_score, 0);
    const bRel = toNum(b.json.output?.relevance_score ?? b.json.relevance_score, 0);
    if (bRel !== aRel) return bRel - aRel;

    const aCred = toNum(a.json.credibility_tier, 0);
    const bCred = toNum(b.json.credibility_tier, 0);
    return bCred - aCred;
  });

const limit = Math.max(1, maxPapers);
const capped = ranked.slice(0, limit);

console.log(`[Rank and Cap Filtered Papers] Input=${items.length}, Output=${capped.length}, max=${limit}`);

return capped;
''',
    'Format for Selection Agent': r'''const papers = $input.all().map(item => ({
  arxiv_id: item.json.id,
  title: item.json.title,
  authors: item.json.author,
  affiliations: item.json.affiliations,
  credibility_tier: item.json.credibility_tier,
  credibility_reasoning: item.json.credibility_reasoning,
  primary_topic: item.json.output.primary_topic,
  secondary_topics: item.json.output.secondary_topics,
  potential_impact: item.json.output.potential_impact,
  relevance_score: item.json.output.relevance_score,
  key_findings: item.json.output.key_findings,
  abstract: item.json.summary || item.json.output.summary
}));

return {
  json: {
    all_papers: papers,
    paper_count: papers.length,
    run_date: papers[0]?.run_date || new Date().toISOString().split('T')[0]
  }
};
''',
    'Build Selection Prompt': r'''const runMode = $('Set Run Mode').first().json || {};
const config = runMode.config || {};
const mode = runMode.mode || 'weekly';
const modeConfig = config.modes?.[mode] || {};
const periodLabel = runMode.periodLabel || 'recent period';
const selectionRangeLabel = modeConfig.selection_range_label || `${modeConfig.selection_min || 0}-${modeConfig.selection_max || 0}`;

const applyTemplate = (value, vars) => (value || '').replace(/{{\s*([^}]+)\s*}}/g, (_, key) => {
  const v = vars[key.trim()];
  return v === undefined || v === null ? '' : v;
});

const template = config.prompts?.selection?.user || '';
const systemTemplate = config.prompts?.selection?.system || '';

const prompt = applyTemplate(template, {
  period_label: periodLabel,
  topic_focus: config.topic?.focus || 'research',
  paper_count: $json.paper_count,
  all_papers_json: JSON.stringify($json.all_papers, null, 2),
  selection_range_label: selectionRangeLabel,
  max_selection: modeConfig.max_selection || 0
});

const system = applyTemplate(systemTemplate, {
  max_selection: modeConfig.max_selection || 0
});

return [{
  json: {
    ...$json,
    selection_prompt: prompt,
    selection_system: system
  }
}];
''',
    'Validate Selection Output': r'''const item = $input.first().json || {};
const output = item.output || {};

const asArray = (v) => Array.isArray(v) ? v : [];
const asText = (v) => typeof v === 'string' ? v.trim() : '';

const selected = asArray(output.selected_papers)
  .filter(p => p && typeof p === 'object' && asText(p.arxiv_id));
const remaining = asArray(output.remaining_papers)
  .filter(p => p && typeof p === 'object' && asText(p.arxiv_id));

if (selected.length === 0) {
  const raw = JSON.stringify(output).slice(0, 800);
  throw new Error(`Paper Selection Agent produced no valid selected_papers JSON. output preview: ${raw}`);
}

return [{
  json: {
    output: {
      reasoning: asText(output.reasoning),
      selected_papers: selected.map((p, i) => ({
        arxiv_id: asText(p.arxiv_id),
        selection_reasoning: asText(p.selection_reasoning),
        priority_rank: Number.isFinite(Number(p.priority_rank)) ? Number(p.priority_rank) : (i + 1)
      })),
      remaining_papers: remaining.map((p) => ({
        arxiv_id: asText(p.arxiv_id),
        exclusion_reasoning: asText(p.exclusion_reasoning)
      }))
    }
  }
}];
''',
    'Build Deep Analysis Prompt': r'''const config = $('Set Run Mode').first().json.config || {};
const template = config.prompts?.analysis?.user || '';
const system = config.prompts?.analysis?.system || '';

const applyTemplate = (value, vars) => (value || '').replace(/{{\s*([^}]+)\s*}}/g, (_, key) => {
  const v = vars[key.trim()];
  return v === undefined || v === null ? '' : v;
});

return items.map(item => {
  const data = item.json;
  const authors = Array.isArray(data.author) ? data.author.join(', ') : (data.author ?? '');

  const prompt = applyTemplate(template, {
    arxiv_id: data.selected_papers?.arxiv_id || data.id,
    title: data.title,
    authors,
    primary_topic: data.output?.primary_topic || '',
    selection_reasoning: data.selected_papers?.selection_reasoning || '',
    full_text: data.full_text || '',
    pdf_analysis_performed: data.pdf_download_success ? 'true' : 'false'
  });

  return {
    json: {
      ...data,
      analysis_prompt: prompt,
      analysis_system: system
    }
  };
});
''',
    'Compute Paper Statistics': r'''const toNum = (v) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

const round1 = (v) => Math.round((v || 0) * 10) / 10;

const allPapers = $('Merge Classification Results1').all().map(item => item.json || {});
const abstractsAnalyzed = $('Remove Duplicates').all().length;

const relevanceScores = allPapers
  .map(p => toNum(p.output?.relevance_score))
  .filter(v => v !== null);
const credibilityScores = allPapers
  .map(p => toNum(p.credibility_tier))
  .filter(v => v !== null);

const avgRelevance = relevanceScores.length
  ? relevanceScores.reduce((a, b) => a + b, 0) / relevanceScores.length
  : 0;
const avgCredibility = credibilityScores.length
  ? credibilityScores.reduce((a, b) => a + b, 0) / credibilityScores.length
  : 0;

const highCredibilityCount = allPapers.filter(p => toNum(p.credibility_tier) >= 8).length;

const topicCounts = {};
allPapers.forEach(p => {
  const topic = p.output?.primary_topic || 'Unknown';
  topicCounts[topic] = (topicCounts[topic] || 0) + 1;
});

const topicDistribution = Object.entries(topicCounts)
  .sort((a, b) => b[1] - a[1])
  .slice(0, 10)
  .map(([topic, count]) => ({ topic, count }));

const impactCounts = { Breakthrough: 0, High: 0, Medium: 0, Low: 0 };
allPapers.forEach(p => {
  const impact = p.output?.potential_impact || 'Unknown';
  if (impactCounts[impact] !== undefined) {
    impactCounts[impact] += 1;
  }
});

return [{
  json: {
    stats: {
      abstracts_analyzed: abstractsAnalyzed,
      total_relevant_papers: allPapers.length,
      avg_relevance_score: round1(avgRelevance),
      avg_credibility_tier: round1(avgCredibility),
      high_credibility_count: highCredibilityCount,
      topic_distribution: topicDistribution,
      impact_distribution: impactCounts
    }
  }
}];
''',
    'Media - Map EE Times to Pipeline': r'''const raw = ($input.first().json.stdout || '').trim();
const runMode = $('Set Run Mode').first().json || {};
const runDate = (runMode.currentDate || '').toString().slice(0, 10) || new Date().toISOString().slice(0, 10);
let arr = [];
if (raw) { try { const data = JSON.parse(raw); arr = Array.isArray(data) ? data : []; } catch (_) {} }
if (arr.length === 0) return [{ json: { type: 'eetimes_empty', _noResults: true, source: 'eetimes' } }];
const maxSummary = 6000;
function toPublished(s) { if (!s || !String(s).trim()) return ''; const d = new Date(String(s).trim()); return Number.isNaN(d.getTime()) ? '' : d.toISOString(); }
return arr.map(r => ({ json: { id: r.url||'', title: (r.title||r.listTitle||'').trim(), summary: (r.content||'').trim().slice(0, maxSummary), author: r.listAuthor ? [r.listAuthor] : [], published: toPublished(r.listDate), category: ['eetimes'], run_date: runDate, source: 'eetimes', url: r.url||'', listAuthor: r.listAuthor||'', listDate: r.listDate||'' } }));
''',
    'Media - Split Summary and Articles': r'''const items = $input.all();
const runMode = $('Set Run Mode').first().json || {};
const summaryTypes = ['eetimes_summary', 'semianalysis_summary', 'trendforce_summary'];
const rawArticles = items.filter(i => !i.json._noResults && !summaryTypes.includes(i.json.type));
const summaries = items.filter(i => summaryTypes.includes(i.json.type));

let dateFrom = summaries.map(i => i.json.date_from).filter(Boolean).sort()[0] || '';
let dateTo = summaries.map(i => i.json.date_to).filter(Boolean).sort().reverse()[0] || '';

if (runMode.periodRange && runMode.periodRange.includes(' to ')) {
  const [from, to] = String(runMode.periodRange).split(' to ');
  if (/^\d{4}-\d{2}-\d{2}$/.test(from || '')) dateFrom = from;
  if (/^\d{4}-\d{2}-\d{2}$/.test(to || '')) dateTo = to;
}

const parseDateOnly = (value) => {
  if (!value) return '';
  const s = String(value).trim();
  if (!s) return '';
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;
  const dt = new Date(s);
  if (Number.isNaN(dt.getTime())) return '';
  return dt.toISOString().slice(0, 10);
};

const inWindow = (article) => {
  if (!dateFrom || !dateTo) return true;
  const articleDate = parseDateOnly(article.date || article.listDate || article.published || '');
  if (!articleDate) return false;
  return articleDate >= dateFrom && articleDate <= dateTo;
};

const articles = rawArticles
  .map(i => i.json)
  .filter(inWindow);

const period_range = (dateFrom && dateTo) ? `${dateFrom} to ${dateTo}` : (runMode.periodRange || '');
const summary = { period_range, date_from: dateFrom, date_to: dateTo, total_articles: articles.length };
return [{ json: { summary, articles, article_count: articles.length } }];
''',
    'Media - Build Selection Prompt': r'''const item = $input.first().json;
const mediaCfg = $('Media - Load Config').first().json || {};
const runMode = $('Set Run Mode').first().json || {};
const config = mediaCfg.config || runMode.config || {};
const topicFocus = mediaCfg.topicFocus || runMode.topicFocus || 'memory technology and semiconductors';
const periodLabel = runMode.periodLabel || mediaCfg.period_label || 'this period';
const periodRange = item.summary?.period_range || runMode.periodRange || mediaCfg.period_range || 'the period';
const maxSelect = 5;
const articles = item.articles || [];
const articlesJson = JSON.stringify(articles.map((a, i) => ({
  index: i, source: a.source, title: a.title,
  summary: (a.summary || '').slice(0, 500), url: a.url, listDate: a.listDate
})), null, 2);

const applyTemplate = (value, vars) => (value || '').replace(/{{\s*([^}]+)\s*}}/g, (_, key) => {
  const v = vars[key.trim()];
  return v === undefined || v === null ? '' : v;
});

const mediaPrompts = config.prompts?.media || {};
const defaultSystem = `Select the top ${maxSelect} most newsworthy articles about ${topicFocus}. Try to include articles from multiple sources (EE Times, SemiAnalysis, TrendForce). Output ONLY valid JSON.`;
const defaultPrompt = `Select up to ${maxSelect} articles from the following ${articles.length} articles about ${topicFocus} for ${periodLabel} (${periodRange}).\n\nArticles:\n${articlesJson}\n\nOutput ONLY valid JSON: {"selected_indices": [0, 1, 2, ...]}`;

const vars = {
  topic_focus: topicFocus,
  period_label: periodLabel,
  period_range: periodRange,
  article_count: articles.length,
  max_select: maxSelect,
  articles_json: articlesJson
};

const system = applyTemplate(mediaPrompts.selection_system || defaultSystem, vars);
const prompt = applyTemplate(mediaPrompts.selection_user || defaultPrompt, vars);

return [{ json: { ...item, selection_prompt: prompt, selection_system: system, max_select: maxSelect } }];
''',
    'Media - Apply Selection': r'''const item = $('Media - Build Selection Prompt').first().json;
let indices = [];
const inputData = $input.first().json;
const out = inputData?.output || inputData;
if (out && typeof out === 'object') {
  if (Array.isArray(out.selected_indices)) indices = out.selected_indices;
  else if (out.selected_papers && Array.isArray(out.selected_papers))
    indices = out.selected_papers.map(p => p.index || 0).filter(i => typeof i === 'number');
}
if (indices.length === 0) {
  const rawText = inputData?.text || inputData?.output || JSON.stringify(inputData || {});
  const m = String(rawText).match(/"selected_indices"\s*:\s*\[([^\]]+)\]/);
  if (m) try { indices = JSON.parse('[' + m[1] + ']'); } catch (_) {}
}
const articles = (item.articles || []).filter((_, i) => indices.includes(i));
const selected = articles.length ? articles : (item.articles || []).slice(0, 5);
return [{ json: { summary: item.summary, articles: selected, article_count: selected.length } }];
''',
    'Media - Build Summarize Prompt': r'''const item = $input.first().json;
const runMode = $('Set Run Mode').first().json || {};
const mediaCfg = $('Media - Load Config').first().json || {};
const config = mediaCfg.config || runMode.config || {};
const topicFocus = mediaCfg.topicFocus || runMode.topicFocus || 'memory technology and semiconductors';
const periodLabel = runMode.periodLabel || mediaCfg.period_label || 'this period';
const periodRange = item.summary?.period_range || runMode.periodRange || mediaCfg.period_range || 'the period';
const articles = item.articles || [];
if (articles.length === 0) return [{ json: { ...item, articles: [] } }];

const maxContent = 3000;
const list = articles.map((a, i) =>
  `--- Article ${i + 1} [${(a.source || '').toUpperCase()}] ---\nTitle: ${a.title || 'Untitled'}\nDate: ${a.listDate || ''}\nURL: ${a.url || ''}\nContent:\n${(a.summary || '').slice(0, maxContent)}`
).join('\n\n');

const applyTemplate = (value, vars) => (value || '').replace(/{{\s*([^}]+)\s*}}/g, (_, key) => {
  const v = vars[key.trim()];
  return v === undefined || v === null ? '' : v;
});

const mediaPrompts = config.prompts?.media || {};
const defaultPrompt = `Normalize each news article for ${topicFocus} in ${periodLabel} (${periodRange}). For each, output one object in the SAME ORDER with:
- title: article title (unchanged)
- short_summary: 2-3 clear sentences about the main point (no author/date lines, no tables)
- date: YYYY-MM-DD or empty string
- source: copy the source tag from the article header (eetimes / semianalysis / trendforce)

Return ONLY a JSON array. No code fences.

${list}`;

const vars = {
  topic_focus: topicFocus,
  period_label: periodLabel,
  period_range: periodRange,
  article_count: articles.length,
  articles_text: list
};

const prompt = applyTemplate(mediaPrompts.summarize_user || defaultPrompt, vars);
const system = applyTemplate(mediaPrompts.summarize_system || 'Output only a JSON array of objects with keys: title, short_summary, date, source. Same number and order as articles. No markdown.', vars);

return [{ json: { ...item, summarize_prompt: prompt, summarize_system: system } }];
''',
    'Media - Parse Summarized Articles': r'''const prev = $('Media - Build Summarize Prompt').first().json;
const item = $input.first().json;
const articles = prev.articles || [];
if (articles.length === 0) return [{ json: { summary: prev.summary, articles: [], article_count: 0 } }];
let parsed = [];
let raw = '';
if (item.output && typeof item.output === 'object') {
  const o = item.output;
  if (Array.isArray(o)) parsed = o;
  else if (Array.isArray(o.articles)) parsed = o.articles;
  else raw = JSON.stringify(o);
}
if (parsed.length === 0 && !raw)
  raw = (item.text || (item.output && typeof item.output === 'string' ? item.output : '') || '').toString();
if (raw) {
  const cleaned = raw.replace(/^\s*```(?:json)?\s*\n?/i, '').replace(/\n?```\s*$/i, '').trim();
  try { const p = JSON.parse(cleaned); parsed = Array.isArray(p) ? p : (p.articles ? p.articles : [p]); } catch (_) {}
}
function normDate(s) {
  if (!s || typeof s !== 'string') return '';
  const t = s.trim().toLowerCase();
  if (!t || t.includes('not') || t === 'n/a' || t === 'null') return '';
  if (/^\d{4}-\d{2}-\d{2}$/.test(t)) return t;
  try { const d = new Date(t); if (!Number.isNaN(d.getTime())) return d.toISOString().slice(0, 10); } catch (_) {}
  return '';
}
const merged = articles.map((orig, i) => {
  const s = parsed[i] || {};
  return {
    title: s.title ?? orig.title ?? '',
    short_summary: s.short_summary ?? (orig.summary || '').slice(0, 400),
    date: normDate(s.date) || normDate(orig.listDate || orig.date || ''),
    url: orig.url ?? '',
    source: s.source || orig.source || 'unknown'
  };
});
return [{ json: { summary: prev.summary, articles: merged, article_count: merged.length } }];
''',
    'Media - Prepare Media Output For Parent': r'''const item = $input.first().json || {};
const summary = item.summary || {};
const articles = Array.isArray(item.articles) ? item.articles : [];
const periodRange = summary.period_range || [summary.date_from, summary.date_to].filter(Boolean).join(' to ') || '';

const mediaMarkdown = articles.length
  ? articles.map((a, idx) => {
      const title = a.title || `Untitled ${idx + 1}`;
      const link = a.url ? `[${title}](${a.url})` : title;
      const source = (a.source || 'unknown').toString();
      const date = a.date ? ` (${a.date})` : '';
      const text = (a.short_summary || '').toString().trim();
      return `- ${link} — ${source}${date}\n  ${text}`;
    }).join('\n')
  : '- No media articles available for this period.';

return [{
  json: {
    media_intelligence: {
      period_range: periodRange,
      article_count: articles.length,
      articles
    },
    media_period_range: periodRange,
    media_article_count: articles.length,
    media_articles: articles,
    media_markdown: mediaMarkdown
  }
}];
''',
}


@requires_node
def test_harness_runs_a_node(tmp_path):
    out = run_node(tmp_path, "Validate Selection Output",
                   input_items=[{"output": {"selected_papers": [{"arxiv_id": " x "}]}}])
    assert out[0]["output"]["selected_papers"] == [
        {"arxiv_id": "x", "selection_reasoning": "", "priority_rank": 1}]
