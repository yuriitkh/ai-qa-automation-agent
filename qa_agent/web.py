"""Read-only localhost UI for persisted TestRun history."""

import argparse
import html
import mimetypes
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

from qa_agent.reporting import RunReportGenerator, RunStepReport, RunAttemptReport, RunEvidenceReport
from qa_agent.run_history import RunHistoryService
from qa_agent.storage import create_sqlite_storage


_CSS = """body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1f2937}
h1,h2{color:#111827}a{color:#1d4ed8}.summary,.panel{border:1px solid #d1d5db;border-radius:8px;padding:1rem;margin:1rem 0}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:.65rem;border-bottom:1px solid #e5e7eb}
.badge{display:inline-block;border-radius:999px;padding:.2rem .65rem;background:#e5e7eb}
.passed{background:#dcfce7;color:#166534}.failed{background:#fee2e2;color:#991b1b}
.blocked{background:#fef3c7;color:#92400e}.muted{color:#6b7280}.evidence{max-width:700px;max-height:500px}
"""


@dataclass(frozen=True)
class WebResponse:
    status: int
    content_type: str
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def html(cls, status: int, body: str) -> "WebResponse":
        return cls(status, "text/html; charset=utf-8", body.encode("utf-8"))

    @classmethod
    def json(cls, status: int, body: str) -> "WebResponse":
        return cls(status, "application/json; charset=utf-8", body.encode("utf-8"))


class LocalWebApplication:
    """Route requests through history/report services, never raw SQL."""

    def __init__(
        self,
        run_history: RunHistoryService,
        reports: RunReportGenerator | None = None,
        evidence_root: str | Path | None = None,
    ) -> None:
        self._run_history = run_history
        self._reports = reports or RunReportGenerator()
        self._evidence_root = Path(evidence_root).expanduser() if evidence_root else None

    def handle(self, method: str, target: str) -> WebResponse:
        if method.upper() != "GET":
            return WebResponse.html(405, self._page("Method not allowed", "<p>Read-only UI.</p>"))
        path = urlsplit(target).path
        if path == "/":
            return WebResponse.html(200, self._dashboard())

        evidence_parts = path.strip("/").split("/")
        if len(evidence_parts) == 5 and evidence_parts[0] == "runs" and evidence_parts[2] == "evidence":
            return self._serve_evidence(
                evidence_parts[1], evidence_parts[3], evidence_parts[4]
            )

        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "runs" and parts[2] in {"report.json", "report.html"}:
            run_id = _parse_uuid(parts[1])
            if run_id is None:
                return self._not_found()
            detail = self._run_history.get_detail(run_id)
            if detail is None:
                return self._not_found()
            report = self._reports.generate_history(detail)
            if parts[2] == "report.json":
                return WebResponse.json(200, report.to_json())
            return WebResponse.html(200, self._reports.to_html(
                report,
                evidence_url=lambda _step, attempt, evidence, index: self._evidence_url(
                    run_id, attempt, index
                ),
            ))
        if len(parts) == 2 and parts[0] == "runs":
            run_id = _parse_uuid(parts[1])
            if run_id is None:
                return self._not_found()
            detail = self._run_history.get_detail(run_id)
            if detail is None:
                return self._not_found()
            report = self._reports.generate_history(detail)
            return WebResponse.html(200, self._reports.to_html(
                report,
                evidence_url=lambda _step, attempt, evidence, index: self._evidence_url(
                    run_id, attempt, index
                ),
            ))
        if len(parts) == 2 and parts[0] == "test-cases":
            test_case_id = _parse_uuid(parts[1])
            if test_case_id is None:
                return self._not_found()
            return self._test_case_page(test_case_id)
        return self._not_found()

    def _dashboard(self) -> str:
        runs = self._run_history.list_recent(50)
        rows = []
        for record in runs:
            rows.append(
                "<tr>"
                f'<td><a href="/runs/{record.run_id}">{_e(record.run_id)}</a></td>'
                f'<td><a href="/test-cases/{record.test_case_id}">{_e(record.test_case_name)}</a></td>'
                f"<td>{_e(record.workflow_type.value)}</td>"
                f'<td><span class="badge {_e(record.status.value.lower())}">{_e(record.status.value)}</span></td>'
                f"<td>{_e(record.started_at.isoformat())}</td>"
                f"<td>{_duration(record.duration_ms)}</td></tr>"
            )
        table = (
            "<table><thead><tr><th>Run ID</th><th>TestCase</th><th>Workflow</th>"
            "<th>Status</th><th>Started</th><th>Duration</th></tr></thead><tbody>"
            + ("".join(rows) if rows else '<tr><td colspan="6">No runs yet.</td></tr>')
            + "</tbody></table>"
        )
        return self._page("AI QA Agent", "<h1>AI QA Agent</h1><section class='panel'>"
                          f"<h2>Recent Runs</h2>{table}</section>")

    def _test_case_page(self, test_case_id: UUID) -> WebResponse:
        records = self._run_history.list_for_test_case(test_case_id, 50)
        if not records:
            return self._not_found()
        latest = records[0]
        steps = "".join(
            f"<li><strong>{step.order}. {_e(step.name)}</strong> — {_e(step.expected)}</li>"
            for step in sorted(latest.steps, key=lambda item: item.order)
        )
        conditions = "".join(
            f"<li>{_e(condition.description)}"
            + (f" <span class='muted'>({_e(condition.status)})</span>" if condition.status else "")
            + "</li>"
            for condition in latest.preconditions
        )
        history_rows = "".join(
            f'<tr><td><a href="/runs/{record.run_id}">{_e(record.run_id)}</a></td>'
            f"<td>{_e(record.workflow_type.value)}</td><td>{_e(record.status.value)}</td>"
            f"<td>{_e(record.started_at.isoformat())}</td><td>{_duration(record.duration_ms)}</td></tr>"
            for record in records
        )
        body = (
            f"<h1>{_e(latest.test_case_name)}</h1><p>{_e(latest.test_case_description)}</p>"
            f"<p class='muted'>Definition shown from the most recent recorded run.</p>"
            f"<p>TestCase ID: <code>{_e(latest.test_case_id)}</code></p>"
            f"<section class='panel'><h2>Steps</h2><ol>{steps}</ol></section>"
        )
        if latest.preconditions:
            body += f"<section class='panel'><h2>Preconditions</h2><ul>{conditions}</ul></section>"
        if latest.segments:
            body += "<section class='panel'><h2>Execution Segments</h2><ul>" + "".join(
                f"<li>Segment {segment.order}: {_e(segment.base_url or 'No URL')} "
                f"({len(segment.test_step_ids)} steps)</li>"
                for segment in latest.segments
            ) + "</ul></section>"
        body += (
            "<section class='panel'><h2>Run History</h2><table><thead><tr>"
            "<th>Run ID</th><th>Workflow</th><th>Status</th><th>Started</th><th>Duration</th>"
            "</tr></thead><tbody>" + history_rows + "</tbody></table></section>"
        )
        return WebResponse.html(200, self._page("TestCase history", body))

    def _serve_evidence(self, run_id_text: str, execution_id_text: str, index_text: str) -> WebResponse:
        run_id = _parse_uuid(run_id_text)
        execution_id = _parse_uuid(execution_id_text)
        if run_id is None or execution_id is None or not index_text.isdecimal():
            return self._not_found()
        detail = self._run_history.get_detail(run_id)
        if detail is None or not any(
            ref.execution_id == execution_id for ref in detail.record.executions
        ):
            return self._not_found()
        reference = next(
            ref for ref in detail.record.executions
            if ref.execution_id == execution_id
        )
        execution = detail.executions.get(execution_id)
        index = int(index_text)
        if (
            execution is None
            or index >= len(execution.evidence)
            or index >= len(reference.evidence)
            or execution.evidence[index].id != reference.evidence[index].id
            or execution.evidence[index].type.value != "SCREENSHOT"
            or self._evidence_root is None
        ):
            return self._not_found()
        try:
            root = self._evidence_root.resolve(strict=True)
            evidence_path = Path(execution.evidence[index].path)
            if not evidence_path.is_absolute():
                evidence_path = root / evidence_path
            resolved = evidence_path.resolve(strict=True)
            if not resolved.is_relative_to(root) or not resolved.is_file():
                return self._not_found()
            if resolved.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                return self._not_found()
            content_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
            safe_name = resolved.name.replace("\r", "").replace("\n", "").replace('"', "")
            return WebResponse(
                200,
                content_type,
                resolved.read_bytes(),
                {"Content-Disposition": f'inline; filename="{safe_name}"'},
            )
        except (OSError, RuntimeError, ValueError):
            return self._not_found()

    @staticmethod
    def _evidence_url(
        run_id: UUID,
        attempt: RunAttemptReport,
        evidence_index: int,
    ) -> str:
        return f"/runs/{run_id}/evidence/{attempt.execution_id}/{evidence_index}"

    def _page(self, title: str, body: str) -> str:
        return (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>"
            f"<nav><a href='/'>AI QA Agent</a></nav>{body}</body></html>"
        )

    def _not_found(self) -> WebResponse:
        return WebResponse.html(404, self._page("Not found", "<h1>Not found</h1>"))


def create_application(
    database_path: str | Path | None = None,
    evidence_directory: str | Path | None = None,
) -> LocalWebApplication:
    storage = create_sqlite_storage(database_path)
    return LocalWebApplication(
        storage.run_history,
        evidence_root=evidence_directory,
    )


def serve(
    application: LocalWebApplication,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            response = application.handle("GET", self.path)
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'unsafe-inline'; img-src 'self'; object-src 'none'",
            )
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(response.body)

        def do_POST(self) -> None:
            response = application.handle("POST", self.path)
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.end_headers()
            self.wfile.write(response.body)

    server = ThreadingHTTPServer((host, port), Handler)
    try:
        print(f"AI QA Agent UI listening at http://{host}:{port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Browse persisted AI QA Agent runs.")
    parser.add_argument("--database", default=None, help="SQLite database path")
    parser.add_argument("--evidence-directory", default=None, help="allowed screenshot directory")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (defaults to localhost)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    serve(create_application(args.database, args.evidence_directory), args.host, args.port)
    return 0


def _parse_uuid(value: str) -> UUID | None:
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        return None


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _duration(milliseconds: int | None) -> str:
    return f"{milliseconds} ms" if milliseconds is not None else "—"


if __name__ == "__main__":
    raise SystemExit(main())
