"""Email HTML rendering — port of the n8n nodes "Convert Markdown to HTML2" and
"Apply Email Styling" (Memory Innovation Research Assistant workflow).

Also hosts the small JavaScript-semantics helpers (Number(), Math.round,
toFixed, JSON.stringify, String()) that mira.report needs to reproduce the
n8n Code nodes byte-for-byte where it matters (prompt text, record JSON,
styled HTML).

Parity notes (see tests/test_port_back_render.py):
  * Markdown: n8n uses showdown with simplifiedAutoLink. We use Python-Markdown
    (core + fenced code, NO nl2br — showdown's simpleLineBreaks is off) and add
    showdown's header ids (``id="deepdivetopresearch"``) and bare-URL
    autolinking. Tables are not enabled in showdown's defaults, so not here.
  * Palette: n8n reads ``config.email.palette`` / ``config.email.theme``, but a
    profile's ``email`` block is keyed by mode, so those are always undefined
    and every live email uses the TEAL palette — the per-mode ``colors`` in the
    profile are never read. ``resolve_palette`` reproduces that by default and
    only uses the profile colours when ``email_use_profile_colors`` is set.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timezone
from decimal import ROUND_HALF_UP, Decimal

import markdown as _md


# --------------------------------------------------------------------------
# JavaScript semantics helpers
# --------------------------------------------------------------------------

class _Undefined:
    """Sentinel for JS ``undefined``: dropped from objects by js_json, like
    JSON.stringify does, and distinct from ``None`` (JS ``null``)."""

    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "UNDEF"


UNDEF = _Undefined()

_JS_NUM_RE = re.compile(r"[+-]?(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)")


def js_number(v) -> float:
    """JS ``Number(v)``. ``None`` is treated as ``undefined`` (NaN) unless the
    caller maps it; ``UNDEF`` is NaN too."""
    if v is None or v is UNDEF:
        return math.nan
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return 0.0
        if _JS_NUM_RE.fullmatch(s):
            return float(s)
        if s in ("Infinity", "+Infinity"):
            return math.inf
        if s == "-Infinity":
            return -math.inf
        if re.fullmatch(r"0[xX][0-9a-fA-F]+", s):
            return float(int(s, 16))
        return math.nan
    if isinstance(v, list):
        if not v:
            return 0.0
        if len(v) == 1:
            return js_number(js_str(v[0]))
        return math.nan
    return math.nan


def js_null_number(v) -> float:
    """JS ``Number(v)`` where a Python ``None`` stands for JS ``null`` (-> 0)."""
    return 0.0 if v is None else js_number(v)


def is_finite(n: float) -> bool:
    return isinstance(n, (int, float)) and math.isfinite(n)


def js_round(x: float) -> float:
    """``Math.round``: half rounds toward +Infinity."""
    if not math.isfinite(x):
        return x
    return float(math.floor(x + 0.5))


def js_to_fixed(x: float, digits: int) -> str:
    """``Number.prototype.toFixed`` (round half up on the exact binary value)."""
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(x).quantize(q, rounding=ROUND_HALF_UP))


def js_num_str(x) -> str:
    """How JS prints a number (``String(n)`` / template literal)."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == int(x) and abs(x) < 1e21:
        return str(int(x))
    return repr(x)


def js_str(v) -> str:
    """JS ``String(v)`` for the scalar values the n8n templates interpolate."""
    if v is None:
        return "null"
    if v is UNDEF:
        return "undefined"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return js_num_str(v)
    if isinstance(v, list):
        return ",".join("" if (x is None or x is UNDEF) else js_str(x) for x in v)
    if isinstance(v, dict):
        return "[object Object]"
    return str(v)


def _jsonable(obj, in_array: bool = False):
    if obj is UNDEF:
        return None
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        if obj == int(obj) and abs(obj) < 1e21:
            return int(obj)
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if v is not UNDEF}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, True) for v in obj]
    return str(obj)


def js_json(obj, indent: int | None = 2) -> str:
    """``JSON.stringify(obj, null, indent)``: integral floats print without
    ``.0``, NaN/Infinity become null, UNDEF keys are dropped, non-ASCII kept."""
    if indent is None:
        return json.dumps(_jsonable(obj), ensure_ascii=False, separators=(",", ":"))
    return json.dumps(_jsonable(obj), ensure_ascii=False, indent=indent)


def js_iso_now(now: datetime | None = None) -> str:
    """``new Date().toISOString()`` → ``2026-09-03T05:33:04.961Z``."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


# --------------------------------------------------------------------------
# Convert Markdown to HTML2 (showdown, simplifiedAutoLink: true)
# --------------------------------------------------------------------------

_ATX_RE = re.compile(r"^(#{1,6})[ \t]*(.+?)[ \t]*#*[ \t]*$")
_SETEXT_RE = re.compile(r"^(=+|-+)[ \t]*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_HEADING_TAG_RE = re.compile(r"<h([1-6])>(.*?)</h\1>", re.S)


def _raw_heading_texts(md_text: str) -> list[str]:
    """Raw (pre-span-gamut) heading texts in document order, like showdown's
    headers sub-parser sees them — showdown derives the id from this text."""
    out: list[str] = []
    in_fence = False
    lines = md_text.split("\n")
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _ATX_RE.match(line)
        if m:
            out.append(m.group(2))
            continue
        if (i + 1 < len(lines) and line.strip() and not line.startswith(("    ", "\t"))
                and _SETEXT_RE.match(lines[i + 1])):
            out.append(line.strip())
    return out


def _showdown_header_id(raw: str, counts: dict[str, int]) -> str:
    title = re.sub(r"[^A-Za-z0-9_]", "", raw).lower()
    if counts.get(title):
        n = counts[title]
        counts[title] = n + 1
        return f"{title}-{n}"
    counts[title] = 1
    return title


_AUTOLINK_RE = re.compile(r"((?:https?|ftp)://[^'\">\s]+?\.[^'\">\s]+|www\.[^'\">\s]+?\.[^'\">\s]+)")


def _autolink_text(text: str) -> str:
    def repl(m: re.Match) -> str:
        url = m.group(1)
        href = url if not url.lower().startswith("www.") else f"http://{url}"
        return f'<a href="{href}">{url}</a>'
    return _AUTOLINK_RE.sub(repl, text)


def _simplified_autolink(html: str) -> str:
    """Link bare URLs in text nodes (not inside tags, <a>, <code> or <pre>)."""
    parts = re.split(r"(<[^>]+>)", html)
    depth = {"a": 0, "code": 0, "pre": 0}
    out = []
    for part in parts:
        if part.startswith("<") and part.endswith(">"):
            m = re.match(r"<(/?)(a|code|pre)\b", part, re.I)
            if m:
                tag = m.group(2).lower()
                depth[tag] = max(0, depth[tag] + (-1 if m.group(1) else 1))
            out.append(part)
        elif part and not any(depth.values()):
            out.append(_autolink_text(part))
        else:
            out.append(part)
    return "".join(out)


def markdown_to_html(md_text: str) -> str:
    """n8n "Convert Markdown to HTML2": markdown → HTML fragment."""
    md_text = md_text or ""
    html = _md.markdown(md_text, extensions=["fenced_code"])
    raw_titles = _raw_heading_texts(md_text)
    rendered = list(_HEADING_TAG_RE.finditer(html))
    counts: dict[str, int] = {}
    use_raw = len(raw_titles) == len(rendered)
    pieces, last = [], 0
    for idx, m in enumerate(rendered):
        raw = raw_titles[idx] if use_raw else re.sub(r"<[^>]+>", "", m.group(2))
        hid = _showdown_header_id(raw, counts)
        pieces.append(html[last:m.start()])
        pieces.append(f'<h{m.group(1)} id="{hid}">{m.group(2)}</h{m.group(1)}>')
        last = m.end()
    pieces.append(html[last:])
    return _simplified_autolink("".join(pieces))


# --------------------------------------------------------------------------
# Apply Email Styling
# --------------------------------------------------------------------------

PALETTES: dict[str, dict[str, str]] = {
    "teal": {
        "primary": "#0891b2",
        "primaryDark": "#0e7490",
        "primaryLight": "#06b6d4",
        "secondary": "#06b6d4",
        "text": "#2d3748",
        "textLight": "#4a5568",
        "textMuted": "#718096",
        "border": "#e2e8f0",
        "borderLight": "#edf2f7",
        "background": "#f7fafc",
        "backgroundAlt": "#ecfeff",
        "white": "#ffffff",
        "accent": "#14b8a6",
        "warning": "#f59e0b",
        "pageBackground": "#e0f2fe",
    },
    "violet": {
        "primary": "#667eea",
        "primaryDark": "#5a67d8",
        "secondary": "#764ba2",
        "text": "#2d3748",
        "textLight": "#4a5568",
        "textMuted": "#718096",
        "border": "#e2e8f0",
        "borderLight": "#edf2f7",
        "background": "#f7fafc",
        "backgroundAlt": "#edf2f7",
        "white": "#ffffff",
        "accent": "#48bb78",
        "warning": "#ed8936",
        "pageBackground": "#ede9fe",
    },
}

FONT_SCALE = 1.5
BASE_FONT_SIZE = 16
H1_SIZE = BASE_FONT_SIZE * 1.75
H2_SIZE = BASE_FONT_SIZE * 1.5
H3_SIZE = BASE_FONT_SIZE * 1.25
H4_SIZE = BASE_FONT_SIZE * 1.1
DATE_FONT_SIZE = BASE_FONT_SIZE * 0.9375
FOOTER_FONT_SIZE = BASE_FONT_SIZE * 0.8125
FOOTER_SMALL_FONT_SIZE = BASE_FONT_SIZE * 0.75

_n = js_num_str  # shorthand for number interpolation in templates
# Whitespace-only template lines in the n8n node (kept explicit so editors
# that strip trailing whitespace cannot change the output).
_SP8 = " " * 8
_SP10 = " " * 10


def resolve_palette(config: dict, use_profile_colors: bool | None = None) -> dict:
    """Colour palette, resolved the way n8n's Apply Email Styling does.

    n8n: ``paletteOverride = config.email?.palette || config.email_palette``;
    ``theme = config.email?.theme || config.email_theme``; violet when the theme
    mentions violet/purple, else teal. Our config carries the per-mode block as
    ``email_cfg`` — ``email_cfg.palette`` / ``email_cfg.theme`` are honoured as
    the equivalent override hooks (no profile sets them today).

    The profile's per-mode ``email_cfg.colors`` are NOT used by n8n (always
    teal in production). Pass ``use_profile_colors=True`` or set config key
    ``email_use_profile_colors: true`` to opt into them.
    """
    email_cfg = config.get("email_cfg") or {}
    override = email_cfg.get("palette") or config.get("email_palette")
    theme = str(email_cfg.get("theme") or config.get("email_theme") or "").lower()
    if use_profile_colors is None:
        use_profile_colors = bool(config.get("email_use_profile_colors"))
    if override:
        colors = dict(override)
    elif use_profile_colors and email_cfg.get("colors"):
        colors = dict(email_cfg["colors"])
    elif "violet" in theme or "purple" in theme:
        colors = dict(PALETTES["violet"])
    else:
        colors = dict(PALETTES["teal"])
    if not colors.get("pageBackground"):
        colors["pageBackground"] = PALETTES["teal"]["pageBackground"]
    return colors


def format_header_date(iso_date: str | None, today: date | None = None) -> str:
    """``toLocaleDateString('en-US', {weekday:'long', year, month:'long', day})``
    → ``Monday, August 31, 2026``."""
    s = str(iso_date or "")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            d = date.fromisoformat(s)
        except ValueError:
            d = today or date.today()
    else:
        d = today or date.today()
    return f"{d.strftime('%A')}, {d.strftime('%B')} {d.day}, {d.year}"


def scale_font_sizes(html: str, factor: float = FONT_SCALE) -> str:
    if not html or factor == 1:
        return html

    def repl(m: re.Match) -> str:
        scaled = js_round(float(m.group(1)) * factor * 100) / 100
        return f"font-size: {_n(scaled)}px"
    return re.sub(r"font-size:\s*([0-9]*\.?[0-9]+)px", repl, html, flags=re.I)


def _esc(s) -> str:
    """JS ``String(s || '')`` then escape & < >."""
    text = "" if (not s and s is not True) else js_str(s)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_stats_dashboard(html: str, colors: dict, dashboard_data: dict | None,
                          base_font_size: float = BASE_FONT_SIZE) -> str:
    """Replace the "<h2>… in Numbers …</h2>" section with the stats dashboard."""
    heading = re.search(r"<h2[^>]*>([^<]*in Numbers[^<]*)</h2>", html, re.I)
    if not heading:
        return html
    heading_text = (heading.group(1) or "In Numbers").strip()
    heading_start = heading.start()
    heading_end = heading.end()
    nxt = re.compile(r"<h2[^>]*>.*?</h2>", re.I).search(html, heading_end)
    scope_end = nxt.start() if nxt else len(html)

    if not dashboard_data or not dashboard_data.get("counts"):
        return html

    counts = dashboard_data.get("counts") or {}
    score_cards = dashboard_data.get("score_cards") or {}
    all_scores = score_cards.get("all_papers") or {}
    selected_scores = score_cards.get("selected_papers") or {}
    themes = dashboard_data.get("themes") if isinstance(dashboard_data.get("themes"), list) else []

    def as_num(v):
        n = js_null_number(v)
        return n if math.isfinite(n) else None

    def fmt_count(v) -> str:
        n = as_num(v)
        return "-" if n is None else js_num_str(js_round(n))

    def score_value(v) -> str:
        n = as_num(v)
        return "-" if n is None else js_to_fixed(n, 1)

    all_theme_total = sum((as_num(t.get("all_count")) or 0) for t in themes) or 1
    seg_colors = [
        colors.get("primary"),
        colors.get("accent"),
        colors.get("primaryLight") or colors.get("secondary"),
        colors.get("warning"),
        "#667eea",
        "#ed8936",
        "#e53e3e",
    ]

    rows = []
    for i, t in enumerate(themes[:8]):
        all_count = as_num(t.get("all_count")) or 0
        color = seg_colors[i % len(seg_colors)]
        pct = max(1, js_round((all_count / all_theme_total) * 100))
        bar_width = max(12, js_round((all_count / all_theme_total) * 180))
        rem_width = max(0, 180 - bar_width)
        rows.append(f"""<tr>
      <td width="16" style="padding: 8px 0;">
        <div style="width: 10px; height: 10px; background-color: {color}; border-radius: 2px;">&nbsp;</div>
      </td>
      <td style="padding: 8px 8px 8px 0; font-size: 13px; color: {colors.get('text')}; font-weight: 600;">{_esc(t.get('topic'))}</td>
      <td width="42" align="right" style="padding: 8px 10px 8px 0; font-size: 12px; color: {colors.get('textMuted')};">{_n(pct)}%</td>
      <td width="190" style="padding: 8px 0;">
        <table role="presentation" width="180" cellspacing="0" cellpadding="0" border="0" style="border-collapse: collapse;">
          <tr>
            <td width="{_n(bar_width)}" bgcolor="{color}" style="background-color: {color}; height: 10px; line-height: 10px; font-size: 1px;">&nbsp;</td>
            <td width="{_n(rem_width)}" bgcolor="{colors.get('border')}" style="background-color: {colors.get('border')}; height: 10px; line-height: 10px; font-size: 1px;">&nbsp;</td>
          </tr>
        </table>
      </td>
    </tr>""")
    theme_rows = "".join(rows)

    def score_panel(title, rel, cred) -> str:
        return f"""
    <td width="50%" style="padding: 0 6px;" valign="top">
      <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="border: 1px solid {colors.get('border')}; border-radius: 8px;">
        <tr><td style="padding: 14px 14px 12px 14px;">
          <p style="margin: 0 0 8px 0; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: {colors.get('textMuted')};">{title}</p>
          <p style="margin: 0; font-size: 14px; color: {colors.get('text')};">Avg relevance: <strong>{score_value(rel)}/10</strong></p>
          <p style="margin: 6px 0 0 0; font-size: 14px; color: {colors.get('text')};">Avg credibility: <strong>{score_value(cred)}/10</strong></p>
        </td></tr>
      </table>
    </td>"""

    def card(label, value, dark=False) -> str:
        bg = colors.get("primary") if dark else colors.get("backgroundAlt")
        border = "none" if dark else f"1px solid {colors.get('border')}"
        label_color = "rgba(255,255,255,0.82)" if dark else (colors.get("primaryDark") or colors.get("primary"))
        value_color = "#ffffff" if dark else colors.get("primary")
        return f"""
    <td width="33%" style="padding: 0 {'2' if dark else '4'}px;" valign="top">
      <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background-color: {bg}; border-radius: 8px; border: {border};">
        <tr><td style="padding: 14px 10px; text-align: center;">
          <p style="margin: 0 0 1px 0; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: {label_color};">{label}</p>
          <p style="margin: 0; font-size: 30px; font-weight: 800; color: {value_color}; line-height: 1.15; letter-spacing: -1px;">{value}</p>
        </td></tr>
      </table>
    </td>"""

    dashboard = f"""
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin: 40px 0 20px 0;">
      <tr><td style="padding-bottom: 10px; border-bottom: 3px solid {colors.get('primary')};">
        <p style="margin: 0; color: {colors.get('text')}; font-size: {_n(base_font_size * 1.5)}px; font-weight: 700; letter-spacing: -0.3px;">{heading_text}</p>
      </td></tr>
    </table>

    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin: 0 0 20px 0;">
      <tr>
        {card('Abstracts Analyzed', fmt_count(counts.get('abstracts_analyzed', UNDEF)), False)}
        {card('Full-text Papers Analyzed', fmt_count(counts.get('all_papers_considered', UNDEF)), True)}
        {card('Selected Papers', fmt_count(counts.get('selected_papers', UNDEF)), False)}
      </tr>
    </table>

    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin: 0 0 20px 0; background-color: {colors.get('background')}; border-radius: 8px; border: 1px solid {colors.get('border')};">
      <tr><td style="padding: 18px 20px;">
        <p style="margin: 0 0 10px 0; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: {colors.get('textLight')};">Research Themes (All Papers)</p>
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">{theme_rows}</table>
      </td></tr>
    </table>

    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin: 0 0 8px 0;">
      <tr>
        {score_panel('All Papers', all_scores.get('avg_relevance_score', UNDEF), all_scores.get('avg_credibility_tier', UNDEF))}
        {score_panel('Selected Papers', selected_scores.get('avg_relevance_score', UNDEF), selected_scores.get('avg_credibility_tier', UNDEF))}
      </tr>
    </table>
  """
    return html[:heading_start] + dashboard + html[scope_end:]


_CARD_SPLIT_RE = re.compile(
    r"(?=<(?:p |p>|div |div>|table |table>|h[1-6][ >]|ul[ >]|ul>|ol[ >]|ol>|hr|blockquote))", re.I)
_ARXIV_CARD_RE = re.compile(r'(<p style="[^"]*">)\s*(<a href="http[^"]*arxiv[^"]*")', re.I)
_P_OPEN_RE = re.compile(r"^<p[\s>]", re.I)


def process_content(html: str, colors: dict, dashboard_data: dict | None) -> str:
    if not html:
        return "<p>No content available</p>"

    html = build_stats_dashboard(html, colors, dashboard_data)
    c = colors
    p = html

    p = re.sub(r"<h2([^>]*)>(.*?)</h2>", lambda m: f"""<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin: 40px 0 24px 0;">
      <tr>
        <td style="padding-bottom: 12px; border-bottom: 3px solid {c.get('primary')};">
          <h2{m.group(1)} style="margin: 0; color: {c.get('text')}; font-size: {_n(H2_SIZE)}px; font-weight: 700; letter-spacing: -0.3px;">{m.group(2)}</h2>
        </td>
      </tr>
    </table>""", p, flags=re.I)

    p = re.sub(r"<h3([^>]*)>(.*?)</h3>", lambda m: f"""<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin: 28px 0 16px 0;">
      <tr>
        <td style="background-color: {c.get('backgroundAlt')}; padding: 14px 18px; border-left: 4px solid {c.get('primary')}; border-radius: 0 6px 6px 0;">
          <h3{m.group(1)} style="margin: 0; color: {c.get('primaryDark') or c.get('primary')}; font-size: {_n(H3_SIZE)}px; font-weight: 600;">{m.group(2)}</h3>
        </td>
      </tr>
    </table>""", p, flags=re.I)

    p = re.sub(r"<h4([^>]*)>(.*?)</h4>", lambda m: (
        f'<h4{m.group(1)} style="margin: 20px 0 8px 0; color: {c.get("text")}; '
        f'font-size: {_n(H4_SIZE)}px; font-weight: 600;">{m.group(2)}</h4>'), p, flags=re.I)

    p = re.sub(r"<p>", f'<p style="margin: 0 0 16px 0; color: {c.get("text")}; '
               f'font-size: {_n(BASE_FONT_SIZE)}px; line-height: 1.7;">', p, flags=re.I)

    p = re.sub(r'<a\s+href="([^"]+)"([^>]*)>', lambda m: (
        f'<a href="{m.group(1)}" style="color: {c.get("primary")}; text-decoration: none; font-weight: 600;">'),
        p, flags=re.I)

    p = re.sub(r"<strong>(.*?)</strong>", lambda m: (
        f'<strong style="color: {c.get("text")}; font-weight: 700;">{m.group(1)}</strong>'), p, flags=re.I)

    p = re.sub(r"<em>", f'<em style="color: {c.get("textLight")}; font-style: italic;">', p, flags=re.I)
    p = re.sub(r"<ul>", f'<ul style="margin: 16px 0; padding-left: 24px; color: {c.get("text")};">', p, flags=re.I)
    p = re.sub(r"<li>", f'<li style="margin-bottom: 12px; line-height: 1.6; font-size: {_n(BASE_FONT_SIZE)}px; '
               f'padding-left: 8px;">', p, flags=re.I)
    p = re.sub(r"<ol>", f'<ol style="margin: 16px 0; padding-left: 24px; color: {c.get("text")};">', p, flags=re.I)
    p = re.sub(r"<blockquote>", (
        f'<blockquote style="margin: 20px 0; background-color: {c.get("backgroundAlt")}; '
        f'border-left: 4px solid {c.get("accent")}; padding: 16px 20px; border-radius: 0 8px 8px 0; '
        f'color: {c.get("textLight")}; font-size: {_n(BASE_FONT_SIZE)}px; line-height: 1.6; font-style: italic;">'),
        p, flags=re.I)
    p = re.sub(r"<hr\s*/?>", f'<hr style="border: none; border-top: 1px solid {c.get("border")}; margin: 32px 0;">',
               p, flags=re.I)
    p = re.sub(r"\(Note: Full-text analysis unavailable\)", (
        f'<span style="display: inline-block; margin-top: 8px; padding: 4px 10px; '
        f'background-color: {c.get("backgroundAlt")}; border-radius: 4px; '
        f'font-size: {_n(BASE_FONT_SIZE * 0.85)}px; color: {c.get("textMuted")}; font-style: italic;">'
        f'\U0001F4C4 Note: Full-text analysis unavailable</span>'), p, flags=re.I)

    # Wrap each arXiv-link paragraph plus its following <p> siblings in a card.
    card_style = (f"background-color: {c.get('white')}; border: 1px solid {c.get('border')}; "
                  f"border-radius: 8px; padding: 16px; margin: 12px 0;")
    chunks = _CARD_SPLIT_RE.split(p)
    if chunks and chunks[0] == "" and len(chunks) > 1:
        chunks = chunks[1:]  # JS split() does not emit the leading empty chunk
    result = []
    in_card = False
    for chunk in chunks:
        is_arxiv = bool(_ARXIV_CARD_RE.search(chunk))
        is_para = bool(_P_OPEN_RE.match(chunk.strip()))
        if is_arxiv:
            if in_card:
                result.append("</div>")
            result.append(f'<div style="{card_style}">')
            result.append(chunk)
            in_card = True
        elif in_card and is_para:
            result.append(chunk)
        else:
            if in_card:
                result.append("</div>")
                in_card = False
            result.append(chunk)
    if in_card:
        result.append("</div>")
    return "".join(result)


def style_email(html_content: str, subject: str | None, *, header_title: str,
                digest_label: str, current_date: str | None, colors: dict,
                footer_blurb: str | None = None, stats_dashboard: dict | None = None,
                today: date | None = None) -> str:
    """n8n "Apply Email Styling": wrap the markdown HTML in the email template.

    ``header_title`` = runMode.topicName, ``digest_label`` = runMode.digestLabel
    (``"<topic> Digest (<Period Title>)"``), ``footer_blurb`` defaults to
    ``"Digest of <header_title> from arXiv"``. Returns the font-scaled HTML
    (``html_body`` in n8n)."""
    subject = subject or "Research Digest"
    footer_blurb = footer_blurb or f"Digest of {header_title} from arXiv"
    date_str = format_header_date(current_date, today)
    c = colors
    body = process_content(html_content or "", c, stats_dashboard)
    styled = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{subject}</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #ffffff; color: #1a202c; -webkit-font-smoothing: antialiased;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background-color: #ffffff;">
    <tr>
      <td align="center" style="padding: 12px 8px;">
{_SP8}
        <!-- Main Container -->
        <table role="presentation" width="920" cellspacing="0" cellpadding="0" border="0" style="background-color: #ffffff; border-radius: 0; box-shadow: none; overflow: hidden; max-width: 100%;">
{_SP10}
          <!-- Header -->
          <tr>
            <td style="background-color: {c.get('primary')}; padding: 32px 32px; text-align: center;">
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                <tr>
                  <td align="center">
                    <p style="margin: 0 0 8px 0; color: rgba(255,255,255,0.9); font-size: 14px; font-weight: 500; text-transform: uppercase; letter-spacing: 2px;">{digest_label}</p>
                    <h1 style="margin: 0; color: #ffffff; font-size: {_n(H1_SIZE)}px; font-weight: 700; letter-spacing: -0.5px; line-height: 1.2;">{header_title}</h1>
                    <p style="margin: 16px 0 0 0; color: rgba(255,255,255,0.95); font-size: {_n(DATE_FONT_SIZE)}px; font-weight: 500;">{date_str}</p>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
{_SP10}
          <!-- Decorative Accent Bar -->
          <tr>
            <td style="background-color: {c.get('accent')}; height: 6px;"></td>
          </tr>
{_SP10}
          <!-- Content -->
          <tr>
            <td style="padding: 28px 32px;">
              <div style="color: {c.get('text')}; font-size: {_n(BASE_FONT_SIZE)}px; line-height: 1.7;">
                {body}
              </div>
            </td>
          </tr>
{_SP10}
          <!-- Footer Divider -->
          <tr>
            <td style="padding: 0 32px;">
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                <tr>
                  <td style="border-top: 1px solid {c.get('border')};"></td>
                </tr>
              </table>
            </td>
          </tr>
{_SP10}
          <!-- Footer -->
          <tr>
            <td style="background-color: {c.get('background')}; padding: 24px 32px;">
              <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                <tr>
                  <td align="center">
                    <p style="margin: 0 0 8px 0; color: {c.get('textLight')}; font-size: {_n(FOOTER_FONT_SIZE)}px; font-weight: 600;">
                      Memory Innovation Research Assistant
                    </p>
                    <p style="margin: 0; color: {c.get('textMuted')}; font-size: {_n(FOOTER_SMALL_FONT_SIZE)}px; line-height: 1.5;">
                      {footer_blurb}
                    </p>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
{_SP10}
        </table>
{_SP8}
      </td>
    </tr>
  </table>
</body>
</html>"""
    return scale_font_sizes(styled, FONT_SCALE)
