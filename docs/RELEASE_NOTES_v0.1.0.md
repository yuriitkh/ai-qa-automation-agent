# Demo v0.1.0 release preparation

Prepared locally from main baseline `1e2b0a6`. No commit, push, tag, GitHub workflow
execution or published release is performed by M5.

## Changes

* All HTTP mutations require a per-server CSRF token. Server-rendered and dynamic
  POST forms carry it; same-origin and cross-site checks apply to execution,
  authoring, approvals, provider settings, suites and export submissions.
* All HTTP reads and downloads validate a single loopback Host and the listening
  port, closing DNS-rebinding access to data, exports and diagnostics.
* Ambiguous request framing/security headers and oversized bodies are rejected;
  socket reads have a bounded timeout. Unexpected errors return generic responses.
  Request logs omit user-controlled targets, query values and exception contents.
* Responses use no-store, same-origin referrers, framing/base/form restrictions and the
  existing MIME/script policy. Evidence containment, escaping, export secret
  guards, review fingerprints and Quality Gates are preserved.
* Offline GitHub Actions configuration uses Python 3.13, pinned dependencies,
  Chromium, full pytest coverage with exactly the two established external-test
  exclusions, and whitespace validation. No API keys or uploaded artifacts.
* Installation/quick-start, architecture, privacy notes, troubleshooting and a
  reproducible demo checklist are documented.
* Automated release regressions exercise fake LLM generation through real gates,
  approval, actual Chromium Validation, Automation Ready and independent Python
  export execution, with temporary SQLite and evidence.

## Compatibility

The default server remains loopback-bound. Old persisted TestCases retain their
existing review behavior; new unapproved cases cannot execute. Trusted in-process
application calls remain supported. HTTP clients must obtain the current token
from a rendered page and submit `_csrf` or `X-QA-CSRF` on POST, use a loopback Host
with the correct port, and send unambiguous Content-Length framing. Page refresh
is required after server restart because tokens rotate. Remote/proxy hosting is
outside this local demo's supported contract.

## Verification and remaining limits

See [M5 verification record](M5_SECURITY_CI_DEMO.md) for measured results and
[demo guide](DEMO_V0_1_0.md) for installation commands and limitations.
Live LLM reliability, TypeScript runtime, C# runtime, fresh-machine installation
and actual remote GitHub execution remain **NOT TESTED**. The browser-only demo
does not implement account creation or persistent registration. Synthetic seeded
history is not a PASS claim. This release preparation does not establish
commercial production readiness.
