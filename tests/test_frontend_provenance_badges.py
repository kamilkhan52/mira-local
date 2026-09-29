from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frontend_renders_storage_and_joined_provenance_as_domain_chips():
    """Entities and relations expose every provenance domain, including Storage."""
    app_js = (ROOT / "chatbot/static/app.js").read_text()
    style_css = (ROOT / "chatbot/static/style.css").read_text()

    assert "function createProvenanceBadges(domain)" in app_js
    assert ".split('+')" in app_js
    assert "createBadge(domain, domain)" in app_js
    assert "createProvenanceBadges(ent.domain)" in app_js
    assert "createProvenanceBadges(rel.domain)" in app_js
    assert "domain === 'storage'" in app_js
    assert "domain === 'both' || String(domain || '').includes('+')" in app_js
    assert "--badge-storage-bg" in style_css
    assert ".badge-storage" in style_css
