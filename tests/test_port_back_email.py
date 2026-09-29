"""SMTP delivery, recipient resolution, palette resolution and markdown
conversion for the n8n back-half port. No network: smtplib is faked."""
import email
import json
from pathlib import Path

import pytest

from mira import render
from mira import report as R

FX = Path(__file__).parent / "fixtures" / "port_back"
CONFIG = json.loads((FX / "pipeline_fixture.json").read_text())["config"]
SMTP_VARS = ("MIRA_SMTP_HOST", "MIRA_SMTP_PORT", "MIRA_SMTP_USER", "MIRA_SMTP_PASSWORD", "MIRA_SMTP_FROM",
             "MIRA_SMTP_SECURITY", "MIRA_RECIPIENTS", "RECIPIENT_EMAIL", "GMAIL_USER", "GMAIL_APP_PASSWORD")


class FakeSMTP:
    instances: list = []

    def __init__(self, host, port, **kw):
        self.host, self.port, self.kw, self.log = host, port, kw, []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.log.append("quit")

    def ehlo(self):
        self.log.append("ehlo")

    def starttls(self, context=None):
        self.log.append("starttls")

    def login(self, user, password):
        self.log.append(("login", user, password))

    def sendmail(self, sender, to, msg):
        self.log.append(("sendmail", sender, list(to)))
        self.message = email.message_from_string(msg)


class FakeSSL(FakeSMTP):
    pass


@pytest.fixture
def smtp(monkeypatch):
    for v in SMTP_VARS:
        monkeypatch.delenv(v, raising=False)
    FakeSMTP.instances = []
    monkeypatch.setattr(R.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(R.smtplib, "SMTP_SSL", FakeSSL)
    return FakeSMTP


# ------------------------------------------------------------ settings --

def test_smtp_settings_generic_starttls_default():
    s = R.smtp_settings({"MIRA_SMTP_HOST": "smtp.office365.com", "MIRA_SMTP_USER": "bot@corp.com",
                         "MIRA_SMTP_PASSWORD": "pw"})
    assert (s.host, s.port, s.security, s.user, s.sender) == ("smtp.office365.com", 587, "starttls",
                                                              "bot@corp.com", "bot@corp.com")


def test_smtp_settings_465_is_ssl_and_from_override():
    s = R.smtp_settings({"MIRA_SMTP_HOST": "mail.corp", "MIRA_SMTP_PORT": "465", "MIRA_SMTP_USER": "u",
                         "MIRA_SMTP_PASSWORD": "p", "MIRA_SMTP_FROM": "MIRA <mira@corp.com>"})
    assert (s.port, s.security, s.sender) == (465, "ssl", "MIRA <mira@corp.com>")


def test_smtp_settings_unauthenticated_relay():
    s = R.smtp_settings({"MIRA_SMTP_HOST": "relay.corp", "MIRA_SMTP_PORT": "25", "MIRA_SMTP_SECURITY": "none",
                         "MIRA_SMTP_FROM": "mira@corp.com"})
    assert (s.security, s.user, s.password) == ("none", None, None)


def test_smtp_settings_gmail_fallback_only_when_generic_unset():
    s = R.smtp_settings({"GMAIL_USER": "me@gmail.com", "GMAIL_APP_PASSWORD": "app"})
    assert (s.host, s.port, s.security, s.sender) == ("smtp.gmail.com", 587, "starttls", "me@gmail.com")
    s2 = R.smtp_settings({"GMAIL_USER": "me@gmail.com", "GMAIL_APP_PASSWORD": "app",
                          "MIRA_SMTP_HOST": "mail.corp", "MIRA_SMTP_FROM": "x@corp"})
    assert s2.host == "mail.corp"


@pytest.mark.parametrize("env,msg", [
    ({}, "No SMTP configuration"),
    ({"GMAIL_USER": "me@gmail.com"}, "No SMTP configuration"),
    ({"MIRA_SMTP_HOST": "h"}, "MIRA_SMTP_FROM"),
    ({"MIRA_SMTP_HOST": "h", "MIRA_SMTP_FROM": "a@b", "MIRA_SMTP_PORT": "abc"}, "integer"),
    ({"MIRA_SMTP_HOST": "h", "MIRA_SMTP_FROM": "a@b", "MIRA_SMTP_SECURITY": "tls13"}, "MIRA_SMTP_SECURITY"),
])
def test_smtp_settings_errors(env, msg):
    with pytest.raises(EnvironmentError, match=msg):
        R.smtp_settings(env)


def test_recipient_precedence():
    env = {"MIRA_RECIPIENTS": "a@x.com, b@x.com ,", "RECIPIENT_EMAIL": "legacy@x.com"}
    assert R.resolve_recipients(["arg@x.com"], {}, env) == ["arg@x.com"]
    assert R.resolve_recipients("c@x.com,d@x.com", {}, env) == ["c@x.com", "d@x.com"]
    assert R.resolve_recipients(None, {"recipient_email": "cfg@x.com"}, env) == ["a@x.com", "b@x.com"]
    assert R.resolve_recipients(None, {"recipient_email": "cfg@x.com"}, {"RECIPIENT_EMAIL": "legacy@x.com"}) == \
        ["legacy@x.com"]
    assert R.resolve_recipients(None, {"recipient_email": "cfg@x.com"}, {}) == ["cfg@x.com"]
    assert R.resolve_recipients(None, {}, {}) == []


# -------------------------------------------------------------- sending --

def _result(tmp_path, pdf=True):
    pdf_path = None
    if pdf:
        pdf_path = tmp_path / "report-memory-x.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
    return {"html": "<html><body><p>Hi</p></body></html>", "subject": "MIRA  Digest - Test subject",
            "pdf_path": pdf_path}


def test_deliver_report_starttls_with_pdf_attachment(tmp_path, smtp, monkeypatch):
    monkeypatch.setenv("MIRA_SMTP_HOST", "smtp.corp.com")
    monkeypatch.setenv("MIRA_SMTP_USER", "bot@corp.com")
    monkeypatch.setenv("MIRA_SMTP_PASSWORD", "pw")
    monkeypatch.setenv("MIRA_RECIPIENTS", "a@corp.com,b@corp.com")
    info = R.deliver_report(_result(tmp_path), CONFIG)
    (conn,) = smtp.instances
    assert type(conn) is FakeSMTP and (conn.host, conn.port) == ("smtp.corp.com", 587)
    assert conn.log[:3] == ["ehlo", "starttls", "ehlo"]
    assert ("login", "bot@corp.com", "pw") in conn.log
    assert ("sendmail", "bot@corp.com", ["a@corp.com", "b@corp.com"]) in conn.log
    msg = conn.message
    assert msg.get_content_type() == "multipart/mixed" and msg["To"] == "a@corp.com, b@corp.com"
    parts = [p for p in msg.walk() if p.get_filename()]
    assert [p.get_filename() for p in parts] == ["research-report.pdf"]
    assert parts[0].get_content_type() == "application/pdf"
    html_parts = [p for p in msg.walk() if p.get_content_type() == "text/html"]
    assert html_parts and "<p>Hi</p>" in html_parts[0].get_payload(decode=True).decode()
    assert info == {"sent": True, "dry_run": False, "recipients": ["a@corp.com", "b@corp.com"],
                    "attachments": ["research-report.pdf"], "host": "smtp.corp.com"}


def test_deliver_report_ssl_without_pdf(tmp_path, smtp, monkeypatch):
    monkeypatch.setenv("MIRA_SMTP_HOST", "smtp.corp.com")
    monkeypatch.setenv("MIRA_SMTP_PORT", "465")
    monkeypatch.setenv("MIRA_SMTP_FROM", "mira@corp.com")
    R.deliver_report(_result(tmp_path, pdf=False), CONFIG, recipients=["only@corp.com"])
    (conn,) = smtp.instances
    assert type(conn) is FakeSSL and conn.port == 465
    assert "starttls" not in conn.log and not any(isinstance(x, tuple) and x[0] == "login" for x in conn.log)
    assert conn.message.get_content_type() == "multipart/alternative"
    assert ("sendmail", "mira@corp.com", ["only@corp.com"]) in conn.log


def test_deliver_report_dry_run_never_connects(tmp_path, smtp):
    info = R.deliver_report(_result(tmp_path), CONFIG, recipients="x@corp.com", dry_run=True)
    assert smtp.instances == []
    assert info["dry_run"] is True and info["sent"] is False
    assert info["recipients"] == ["x@corp.com"] and info["attachments"] == ["research-report.pdf"]


def test_deliver_report_requires_recipients(tmp_path, smtp):
    with pytest.raises(ValueError, match="recipients"):
        R.deliver_report(_result(tmp_path), {}, dry_run=True)


def test_realtime_style_send_email_uses_config_recipient(smtp, monkeypatch):
    """mira.realtime calls send_email(html, subject, {**config, "recipient_email": sub})."""
    monkeypatch.setenv("GMAIL_USER", "me@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "app")
    monkeypatch.setenv("MIRA_RECIPIENTS", "everyone@corp.com")
    html = R.to_html("## Alert\n\nBody", "Alert subject", CONFIG)
    R.send_email(html, "Alert subject", {**CONFIG, "recipient_email": "sub@corp.com"})
    (conn,) = smtp.instances
    assert (conn.host, conn.port) == ("smtp.gmail.com", 587)
    assert ("sendmail", "me@gmail.com", ["sub@corp.com"]) in conn.log
    assert conn.message["Subject"] == "Alert subject"


# -------------------------------------------------------------- palette --

def test_default_palette_is_teal_like_n8n_even_with_profile_colors():
    assert CONFIG["email_cfg"]["colors"]["primary"] == "#667eea"  # weekly profile colours...
    assert render.resolve_palette(CONFIG)["primary"] == "#0891b2"  # ...are not what n8n renders


def test_profile_colors_opt_in_and_overrides():
    assert render.resolve_palette({**CONFIG, "email_use_profile_colors": True})["primary"] == "#667eea"
    assert render.resolve_palette(CONFIG, use_profile_colors=True)["accent"] == "#48bb78"
    cfg = {**CONFIG, "email_cfg": {**CONFIG["email_cfg"], "palette": {"primary": "#123456"}}}
    pal = render.resolve_palette(cfg)
    assert pal["primary"] == "#123456" and pal["pageBackground"] == "#e0f2fe"
    assert render.resolve_palette({"email_theme": "Violet"})["primary"] == "#667eea"


def test_styled_email_structure_uses_resolved_palette():
    rm = R.resolve_run_mode(CONFIG)
    dash = {"counts": {"abstracts_analyzed": 57, "all_papers_considered": 6, "selected_papers": 4},
            "score_cards": {"all_papers": {"avg_relevance_score": 5.9, "avg_credibility_tier": 6.8},
                            "selected_papers": {"avg_relevance_score": 8.5, "avg_credibility_tier": 8.3}},
            "themes": [{"topic": "HBM & Stacked DRAM", "all_count": 2, "selected_count": 1}]}
    md_text = ("## Executive Summary\n\n[Paper](http://arxiv.org/abs/2609.00001v1)\n\nSummary.\n\n"
               "## This Last Week in Numbers\n\n- stats line\n\n## Closing\n\nBye")
    for cfg, primary in ((CONFIG, "#0891b2"), ({**CONFIG, "email_use_profile_colors": True}, "#667eea")):
        html = R.render_email_html(md_text, "Subject here", cfg, rm, dash)
        assert html.startswith("<!DOCTYPE html>") and "<title>Subject here</title>" in html
        assert f"background-color: {primary}; padding: 32px 32px" in html
        assert "Memory Technology Research Digest (Last 8 Days)" in html
        assert "Monday, September 28, 2026" in html
        assert "Digest of Memory Technology Research from arXiv" in html
        assert "stats line" not in html and "HBM &amp; Stacked DRAM" in html  # dashboard replaced the list
        assert '<div style="background-color: #ffffff; border: 1px solid' in html  # arXiv paper card
        assert "font-size: 24px" in html and "font-size: 16px; line-height: 1.7" not in html  # 1.5x scaling


# ------------------------------------------------------------- markdown --

def test_markdown_conversion_matches_showdown_conventions():
    html = render.markdown_to_html(
        "## Deep Dive: Top Research\n\nline one\nline two https://example.com/a.b\n\n"
        "## Deep Dive: Top Research\n\n- [x](http://arxiv.org/abs/1v1)\n")
    assert '<h2 id="deepdivetopresearch">Deep Dive: Top Research</h2>' in html
    assert '<h2 id="deepdivetopresearch-1">' in html  # showdown de-duplicates ids
    assert "<br" not in html  # simpleLineBreaks is off in n8n
    assert '<a href="https://example.com/a.b">https://example.com/a.b</a>' in html  # simplifiedAutoLink
    assert html.count("<a ") == 2
