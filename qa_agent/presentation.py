"""Shared, local-only presentation helpers for the Web UI and HTML reports."""

from datetime import datetime, timezone
from html import escape


UI_CSS = """\
:root{color-scheme:light;--ink:#17212f;--muted:#5c6878;--line:#d9e0e8;--surface:#fff;--canvas:#f5f7fa;--blue:#174ea6}
*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--blue);text-decoration-thickness:1px;text-underline-offset:2px}a:hover{text-decoration-thickness:2px}
.shell{max-width:1180px;margin:0 auto;padding:0 1.25rem}.topbar{background:#fff;border-bottom:1px solid var(--line)}
.topbar-inner{min-height:64px;display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap}
.brand{font-size:1.08rem;font-weight:750;text-decoration:none;color:var(--ink)}.nav-links{display:flex;gap:.35rem;flex-wrap:wrap}
.nav-links a,.button{display:inline-block;padding:.48rem .72rem;border-radius:7px;text-decoration:none;font-weight:620}
.nav-links a{color:#344256}.nav-links a[aria-current=page]{background:#eaf1fb;color:#103f85}.button{border:1px solid #b9c7d8;background:#fff;color:#174ea6}
.button.primary{border-color:#174ea6;background:#174ea6;color:white}.main{padding:1.6rem 0 3rem}.breadcrumbs{color:var(--muted);font-size:.92rem;margin:0 0 .8rem}
.breadcrumbs span{padding:0 .35rem;color:#8792a0}.page-heading{margin:0 0 1.15rem}.page-heading h1{font-size:clamp(1.65rem,3vw,2.2rem);line-height:1.2;margin:.15rem 0 .35rem;letter-spacing:-.025em;overflow-wrap:anywhere}
.lead,.muted{color:var(--muted)}.lead{margin:.2rem 0}.panel,.card{background:var(--surface);border:1px solid var(--line);border-radius:11px}
.panel{padding:1.15rem;margin:1rem 0}.panel h2{font-size:1.18rem;margin:0 0 .85rem}.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.8rem;margin:1rem 0}
.card{padding:.9rem 1rem}.card-label{color:var(--muted);font-size:.88rem}.card-value{font-size:1.45rem;font-weight:730;margin-top:.2rem}.card-sub{font-size:.86rem;color:var(--muted)}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;min-width:680px}th,td{text-align:left;padding:.68rem .72rem;border-bottom:1px solid #e7ebf0;vertical-align:top}
th{font-size:.84rem;color:#4e5b6c;background:#f8fafc;font-weight:700}tr:last-child td{border-bottom:0}.badge{display:inline-flex;align-items:center;gap:.25rem;border:1px solid transparent;border-radius:999px;padding:.19rem .58rem;font-size:.8rem;font-weight:730;line-height:1.35;white-space:nowrap}
.badge.success{background:#e5f5eb;color:#155c31;border-color:#b8e2c5}.badge.danger,.badge.product-failure{background:#fff0ed;color:#8d2519;border-color:#f1c4bc}
.badge.warning,.badge.setup-failure,.badge.cleanup-failure{background:#fff6df;color:#704b00;border-color:#ead69b}.badge.drift{background:#f2edff;color:#523391;border-color:#d5c7f3}
.badge.infrastructure{background:#fff0dc;color:#754100;border-color:#edd0a2}.badge.neutral{background:#eef1f5;color:#435064;border-color:#d7dee7}.badge.workflow{background:#eaf1fb;color:#174784;border-color:#cadaf2}
.status-line{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}.meta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.75rem}
.section-heading{display:flex;align-items:center;justify-content:space-between;gap:.8rem;flex-wrap:wrap}.row-sub-badge{margin-top:.35rem}
.meta-item{padding:.65rem .75rem;background:#f8fafc;border-radius:8px}.meta-label{display:block;color:var(--muted);font-size:.82rem}.meta-value{font-weight:620;overflow-wrap:anywhere}
.step-card{border:1px solid var(--line);border-left:4px solid #97a5b6;border-radius:9px;background:#fff;padding:1rem;margin:.8rem 0}
.step-card.passed{border-left-color:#298450}.step-card.failed{border-left-color:#ba3425}.step-card.blocked{border-left-color:#a87918;background:#fffdf7}
.step-heading{display:flex;gap:.6rem;align-items:flex-start;flex-wrap:wrap}.step-heading h3{font-size:1.05rem;margin:0;flex:1 1 250px}.detail-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:.7rem;margin:.65rem 0}
.detail-box{background:#f8fafc;border-radius:7px;padding:.72rem}.detail-box h4{margin:0 0 .3rem;font-size:.85rem;color:#4e5b6c}.detail-box p{margin:.15rem 0;overflow-wrap:anywhere}
.notice{border-radius:8px;padding:.75rem .9rem;margin:.75rem 0;background:#eef4fc;border:1px solid #c9d9ee}.notice.danger{background:#fff0ed;border-color:#f1c4bc}.notice.warning{background:#fff6df;border-color:#ead69b}
.empty-state,.error-state{padding:1.2rem;text-align:center;border:1px dashed #bac5d2;border-radius:9px;background:#fafbfd;color:#536174}.error-state{border-style:solid;background:#fff5f3}
.filters{display:flex;align-items:end;gap:.75rem;flex-wrap:wrap}.field{display:grid;gap:.25rem}.field label{font-size:.85rem;font-weight:650}.field select{min-width:155px;padding:.48rem;border:1px solid #aebaca;border-radius:6px;background:#fff;color:var(--ink);font:inherit}
.field input,.field textarea{width:min(100%,720px);padding:.55rem .62rem;border:1px solid #aebaca;border-radius:6px;background:#fff;color:var(--ink);font:inherit}.field textarea{resize:vertical}
:focus-visible{outline:3px solid #3478c2;outline-offset:2px}
.authoring-entry .field{margin:.8rem 0}.authoring-entry .field input,.authoring-entry .field textarea{width:100%;max-width:720px}
.authoring-entry textarea,.testcase-editor textarea{min-height:12rem}.authoring-submit{margin-top:.2rem}
.authoring-error{padding:.65rem .75rem;border:1px solid #f1c4bc;border-radius:7px;background:#fff0ed;color:#8d2519}
.voice-controls{display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;margin:.25rem 0}
.voice-button:disabled{opacity:.68;cursor:wait}.voice-status{margin:0;min-height:1.5em}.voice-privacy{font-size:.86rem;margin:.15rem 0 .8rem}
.product-flow{display:flex;flex-wrap:wrap;gap:.45rem;list-style:none;margin:1.15rem 0 0;padding:0}
.product-flow li{padding:.25rem .6rem;border-radius:6px;background:#f1f5fa;font-weight:650}
.product-flow li:not(:last-child)::after{content:" →";padding-left:.45rem;color:#5c6878}
.testcase-editor .field{margin:.65rem 0}.field-error{color:#8d2519;font-size:.88rem}.edited-indicator{margin-left:.45rem}
.provider-card{padding:1.2rem}.provider-card h2{margin:.15rem 0}.provider-meta{display:flex;gap:.5rem 1.4rem;flex-wrap:wrap}.provider-meta p{margin:.6rem 0}.provider-actions form{margin:0}.provider-model-form,.provider-key-form{display:flex;align-items:end;gap:.65rem;flex-wrap:wrap;margin:.8rem 0}.provider-model-form .field,.provider-key-form .field{margin:0;flex:1 1 260px}.provider-model-form input,.provider-key-form input{max-width:540px}.provider-health{margin:.5rem 0;font-weight:650}
.actions{display:flex;gap:.55rem;flex-wrap:wrap;margin:.8rem 0}.id-code,code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}.id-code{font-size:.9rem}
.eyebrow{color:var(--muted);font-size:.85rem;font-weight:700;letter-spacing:.04em;margin:0}.technical-details{margin:.7rem 0;color:var(--muted)}.technical-details summary,.plan-version-details summary{cursor:pointer}.plan-version-details{display:inline-block;margin-left:.45rem}.plan-version-details>p,.technical-details>p{margin:.35rem 0}
.evidence-preview{display:block;max-width:min(100%,540px);max-height:380px;object-fit:contain;border:1px solid var(--line);border-radius:7px;margin:.45rem 0}
footer{color:var(--muted);font-size:.85rem;padding:1rem 0;border-top:1px solid var(--line)}
@media(max-width:640px){.shell{padding:0 .8rem}.topbar-inner{padding:.55rem 0}.main{padding-top:1rem}.panel{padding:.85rem}.card-value{font-size:1.25rem}}
@media(max-width:640px){.voice-controls{align-items:flex-start}.product-flow{gap:.3rem}.product-flow li{padding:.2rem .45rem}}
.progress-event{display:flex;gap:.65rem;align-items:baseline;padding:.42rem 0;border-bottom:1px solid #eef1f5}
.is-disabled{opacity:.65;cursor:wait}
.progress-steps{list-style:none;padding:0;margin:0}
.progress-step{display:grid;grid-template-columns:1.5rem 1fr;gap:.65rem;padding:.8rem 0;border-bottom:1px solid #eef1f5}
.progress-step>span:last-child{min-width:0;overflow-wrap:anywhere}.progress-step strong{display:block;overflow-wrap:anywhere}.progress-failure{display:block;color:#8d2519;margin-top:.2rem;overflow-wrap:anywhere}
.progress-symbol{font-size:1.1rem;font-weight:700}
.progress-step-state,.progress-step .muted{display:block;margin-top:.2rem;font-size:.9rem}
.progress-result[hidden]{display:none}
[data-progress-result-content]{display:flex;align-items:center;gap:1rem;flex-wrap:wrap}
[data-progress-result-content] p{margin:0}
.progress-evidence{display:block;margin-top:.2rem;font-size:.9rem;color:var(--muted)}
"""


def escape_html(value: object) -> str:
    return escape(str(value), quote=True)


def format_timestamp(value: datetime | None) -> str:
    if value is None:
        return "Unavailable"
    normalized = (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )
    return f"{normalized.day} {normalized:%b %Y, %H:%M} UTC"


def format_duration(milliseconds: int | None) -> str:
    if milliseconds is None:
        return "Unavailable"
    seconds = max(0, milliseconds) / 1000
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remaining = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remaining}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def short_id(value: object, length: int = 8) -> str:
    full = str(value)
    return f"{full[:length]}…" if len(full) > length else full


def outcome_label(value: str) -> str:
    return {
        "PRODUCT_FAILURE": "PRODUCT FAILURE",
        "AUTOMATION_DRIFT": "AUTOMATION DRIFT",
        "AUTOMATION_EXECUTION_ERROR": "AUTOMATION EXECUTION ERROR",
        "INFRASTRUCTURE_ERROR": "INFRA ERROR",
        "SETUP_FAILURE": "SETUP FAILURE",
        "CLEANUP_FAILURE": "CLEANUP FAILURE",
    }.get(value, value.replace("_", " "))


def outcome_tone(value: str) -> str:
    return {
        "PASSED": "success",
        "PRODUCT_FAILURE": "product-failure",
        "AUTOMATION_DRIFT": "drift",
        "AUTOMATION_EXECUTION_ERROR": "infrastructure",
        "INFRASTRUCTURE_ERROR": "infrastructure",
        "SETUP_FAILURE": "setup-failure",
        "CLEANUP_FAILURE": "cleanup-failure",
        "FAILED": "danger",
    }.get(value, "neutral")


def failure_message(value: str) -> str:
    """Short, stable explanations safe for progress pages and JSON."""
    return {
        "PASSED": "The run completed successfully.",
        "PRODUCT_FAILURE": "The automation completed the check and detected unexpected product behavior.",
        "AUTOMATION_DRIFT": "The saved automation no longer matches the current UI.",
        "AUTOMATION_EXECUTION_ERROR": "Automation was generated but could not be executed reliably.",
        "INFRASTRUCTURE_ERROR": "The test could not be reliably executed because of a browser, network, or runtime problem.",
        "SETUP_FAILURE": "Required test conditions could not be established.",
        "AI_GENERATION_ERROR": "The AI could not generate a valid test definition.",
        "AUTOMATION_GENERATION_ERROR": "Executable automation could not be generated for this test.",
        "INVALID_TESTCASE": "The selected TestCase is invalid or no longer available.",
        "MISSING_AUTOMATION": "Validation or Regression requires saved automation for every step.",
        "EXECUTION_ERROR": "An unexpected execution error occurred.",
        "CLEANUP_FAILURE": "The run completed, but cleanup did not succeed.",
        "FAILED": "The run did not pass.",
    }.get(value, "The run could not be completed.")


def status_tone(value: str) -> str:
    return {
        "PASSED": "success",
        "FAILED": "danger",
        "BLOCKED": "warning",
        "PENDING": "neutral",
        "RUNNING": "neutral",
        "SUCCEEDED": "success",
        "PRECONDITION_NOT_ESTABLISHED": "setup-failure",
        "SETUP_INFRASTRUCTURE_ERROR": "infrastructure",
    }.get(value, "neutral")


def badge(label: str, tone: str | None = None, *, title: str | None = None) -> str:
    display = outcome_label(label)
    color = tone or status_tone(label)
    title_attribute = f' title="{escape_html(title)}"' if title else ""
    return (
        f'<span class="badge {escape_html(color)}"{title_attribute}>'
        f"{escape_html(display)}</span>"
    )
