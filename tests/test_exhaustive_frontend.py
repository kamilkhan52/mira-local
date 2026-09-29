from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "chatbot/static/app.js").read_text()
INDEX = (ROOT / "chatbot/static/index.html").read_text()


def test_research_job_is_persisted_and_reattached_on_load():
    assert "active_research_job_id" in APP_JS
    assert "localStorage.setItem(ACTIVE_RESEARCH_JOB_KEY" in APP_JS
    assert "localStorage.removeItem(ACTIVE_RESEARCH_JOB_KEY" in APP_JS
    assert "/api/research/active" in APP_JS
    assert "reattachActiveResearch" in APP_JS
    assert "response.status === 404" in APP_JS
    assert "clearActiveResearch()" in APP_JS


def test_chat_uses_exhaustive_job_and_handles_active_conflict():
    assert "fetch('/api/research/chat'" in APP_JS or (
        "researchFetch('/api/research/chat'" in APP_JS
    )
    assert "active_job_id" in APP_JS
    assert "/events" in APP_JS
    assert "data.detail === 'exhaustive research is disabled'" in APP_JS
    assert "research_job_id" in APP_JS
    assert "preliminary" not in APP_JS.casefold()


def test_hypothesis_ui_uses_shared_research_queue():
    assert "researchFetch('/api/research/hypotheses'" in APP_JS
    hypothesis_submit = APP_JS[APP_JS.index("async function hypSubmit"):]
    assert "localStorage.setItem(ACTIVE_RESEARCH_JOB_KEY, jobId)" in (
        hypothesis_submit
    )
    assert "showResearchProgress()" in hypothesis_submit
    assert "attachResearchJob(jobId)" in hypothesis_submit
    assert "if (!response.ok && !jobId)" in hypothesis_submit


def test_shared_job_completion_renders_nested_hypothesis_result():
    finish = APP_JS[
        APP_JS.index("function finishResearch"):
        APP_JS.index("async function attachResearchJob")
    ]
    assert "record?.kind === 'hypotheses'" in finish
    assert "record?.result?.result?.markdown" in finish
    assert "hypRenderDossier(markdown)" in finish


def test_progress_surface_has_all_domains_costs_and_cancel():
    for marker in (
        "research-memory-coverage",
        "research-optical-coverage",
        "research-storage-coverage",
        "research-estimated-cost",
        "research-actual-cost",
        "research-cancel",
    ):
        assert marker in INDEX


def test_final_markdown_is_sanitized_and_real_provenance_is_rendered():
    assert "DOMPurify.sanitize(marked.parse(markdown))" in APP_JS
    assert "citation.domains.join('+')" in APP_JS
    assert "createProvenanceBadges(ref.domain)" in APP_JS
