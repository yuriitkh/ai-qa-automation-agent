"""Standalone runtime templates used by the existing TestPlan exporter."""

PYTHON_RUNTIME = '''import os
from urllib.parse import urlsplit, urlunsplit


def export_url(value: str, source_base: str) -> str:
    configured = os.environ.get("BASE_URL")
    if not configured:
        return value
    target, source, override = urlsplit(value), urlsplit(source_base), urlsplit(configured)
    if override.scheme not in {"http", "https"} or not override.netloc or override.username or override.password or override.path not in {"", "/"} or override.query or override.fragment:
        raise ValueError("BASE_URL must be an http(s) origin without credentials, path, query or fragment")
    if (target.scheme, target.netloc) != (source.scheme, source.netloc):
        return value
    return urlunsplit((override.scheme, override.netloc, target.path, target.query, target.fragment))


def export_timeout(name: str, default: int) -> int:
    value = int(os.environ.get(name, os.environ.get("TIMEOUT_MS", str(default))))
    if value <= 0:
        raise ValueError("Timeouts must be positive milliseconds")
    return value
'''

PYTHON_CONFTEST = '''import os
import pytest
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="session")
def browser():
    name = os.environ.get("BROWSER", "chromium")
    if name not in {"chromium", "firefox", "webkit"}:
        raise ValueError("BROWSER must be chromium, firefox or webkit")
    headless = os.environ.get("HEADLESS", "true").lower()
    if headless not in {"true", "false", "1", "0"}:
        raise ValueError("HEADLESS must be true, false, 1 or 0")
    with sync_playwright() as playwright:
        browser = getattr(playwright, name).launch(headless=headless in {"true", "1"})
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def page(browser):
    context = browser.new_context()
    try:
        yield context.new_page()
    finally:
        context.close()
'''

TYPESCRIPT_RUNTIME = '''function exportUrl(value: string, sourceBase: string): string {
  const configured = process.env.BASE_URL;
  if (!configured) return value;
  const target = new URL(value), source = new URL(sourceBase), override = new URL(configured);
  if (!['http:', 'https:'].includes(override.protocol) || override.username || override.password || override.pathname !== '/' || override.search || override.hash)
    throw new Error('BASE_URL must be an http(s) origin without credentials, path, query or fragment');
  return target.origin === source.origin ? override.origin + target.pathname + target.search + target.hash : value;
}

function exportTimeout(name: string, fallback: number): number {
  const value = Number(process.env[name] ?? process.env.TIMEOUT_MS ?? fallback);
  if (!Number.isInteger(value) || value <= 0) throw new Error('Timeouts must be positive milliseconds');
  return value;
}
'''

CSHARP_RUNTIME = '''    private static string ExportUrl(string value, string sourceBase)
    {
        var configured = System.Environment.GetEnvironmentVariable("BASE_URL");
        if (string.IsNullOrEmpty(configured)) return value;
        var target = new System.Uri(value);
        var source = new System.Uri(sourceBase);
        var replacement = new System.Uri(configured);
        if ((replacement.Scheme != "http" && replacement.Scheme != "https") || replacement.UserInfo != "" || replacement.AbsolutePath != "/" || replacement.Query != "" || replacement.Fragment != "")
            throw new System.ArgumentException("BASE_URL must be an http(s) origin without credentials, path, query or fragment");
        return target.GetLeftPart(System.UriPartial.Authority) == source.GetLeftPart(System.UriPartial.Authority)
            ? replacement.GetLeftPart(System.UriPartial.Authority) + target.PathAndQuery + target.Fragment : value;
    }

    private static int ExportTimeout(string name, int fallback)
    {
        var raw = System.Environment.GetEnvironmentVariable(name) ?? System.Environment.GetEnvironmentVariable("TIMEOUT_MS");
        var value = raw == null ? fallback : int.Parse(raw, System.Globalization.CultureInfo.InvariantCulture);
        if (value <= 0) throw new System.ArgumentException("Timeouts must be positive milliseconds");
        return value;
    }
'''

PYTHON_CI = '''name: Standalone tests
on: [workflow_dispatch, push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      BASE_URL: ${{ vars.BASE_URL }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'
      - run: python -m pip install -r requirements.txt
      - run: python -m playwright install --with-deps chromium
      - run: python -m pytest -q
'''
