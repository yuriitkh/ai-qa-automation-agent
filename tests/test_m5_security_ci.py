"""HTTP boundary and offline workflow regressions; no providers or real storage."""

from contextlib import contextmanager
from html.parser import HTMLParser
from http.client import HTTPConnection
import json
from pathlib import Path
import socket
import subprocess
from threading import Thread
from unittest.mock import Mock

import playwright
import pytest

from qa_agent.web import WebResponse, _parse_form_body, create_http_server
from tests.test_test_case_quality_editing import application


@contextmanager
def server_for(app):
    server = create_http_server(app, port=0)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)


def request(port, method, target, body=None, headers=None):
    connection = HTTPConnection('127.0.0.1', port, timeout=5)
    try:
        connection.request(method, target, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


@pytest.mark.parametrize('target', [
    '/drafts', '/drafts/not-a-uuid/save', '/test-cases/manual',
    '/test-cases/manual/prepare', '/test-cases/generate',
    '/test-cases/not-a-uuid/approve', '/test-cases/not-a-uuid/run',
    '/test-cases/not-a-uuid/approve-validation', '/test-cases/export', '/export',
    '/test-suites', '/test-suites/not-a-uuid/run', '/settings/providers',
    '/settings/diagnostics', '/test-cases/not-a-uuid/edit', '/system/health/refresh',
])
def test_every_network_mutation_requires_csrf_before_dispatch(target):
    app, _, _ = application()
    dispatch = app._handle_post = Mock(return_value=WebResponse.json(200, '{}'))
    with server_for(app) as port:
        assert request(port, 'POST', target, 'name=example')[0] == 403
        dispatch.assert_not_called()
        assert request(port, 'POST', target, 'name=example', {'X-QA-CSRF': app._csrf_token})[0] == 200
        dispatch.assert_called_once()


@pytest.mark.parametrize('headers,body', [
    ({'Origin': 'https://attacker.invalid'}, ''),
    ({'Origin': 'null'}, ''),
    ({'Sec-Fetch-Site': 'cross-site'}, ''),
    ({'X-QA-CSRF': 'wrong'}, ''),
    ({}, '_csrf=wrong&_csrf=wrong'),
    ({}, b'name=\xff'),
])
def test_invalid_mutation_origin_token_or_encoding_never_dispatches(headers, body):
    app, _, _ = application()
    dispatch = app._handle_post = Mock(return_value=WebResponse.json(200, '{}'))
    headers = {'X-QA-CSRF': app._csrf_token, **headers}
    with server_for(app) as port:
        assert request(port, 'POST', '/drafts', body, headers)[0] == 403
        dispatch.assert_not_called()


@pytest.mark.parametrize('host', [
    'attacker.invalid', '127.0.0.1.attacker.invalid', 'attacker@127.0.0.1',
    'localhost/path', 'localhost?query', 'localhost#fragment',
    'localhost:bad', 'localhost:1', 'localhost ', '',
])
def test_read_and_export_endpoints_reject_invalid_hosts(host):
    app, _, _ = application()
    dispatch = app.handle = Mock(return_value=WebResponse.json(200, '{}'))
    with server_for(app) as port:
        assert request(port, 'GET', '/export', headers={'Host': host})[0] == 403
        dispatch.assert_not_called()


@pytest.mark.parametrize('extra_headers,expected', [
    ([('Host', 'localhost')], 403),
    ([('Content-Length', '0'), ('Content-Length', '0')], 400),
    ([('Transfer-Encoding', 'chunked'), ('Content-Length', '0')], 400),
    ([('Content-Length', '-1')], 400),
    ([('Content-Length', '900000000')], 413),
    ([('Content-Length', '0'), ('Origin', 'null'), ('Origin', 'null')], 400),
    ([], 400),
])
def test_ambiguous_headers_and_request_framing_are_rejected(extra_headers, expected):
    app, _, _ = application()
    dispatch = app.handle = Mock(return_value=WebResponse.json(200, '{}'))
    with server_for(app) as port:
        connection = HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            connection.putrequest('POST', '/drafts', skip_host=True)
            connection.putheader('Host', f'127.0.0.1:{port}')
            for key, value in extra_headers:
                connection.putheader(key, value)
            connection.endheaders()
            response = connection.getresponse()
            assert response.status == expected
            response.read()
            dispatch.assert_not_called()
        finally:
            connection.close()


def test_safe_error_headers_and_logs_do_not_expose_request_or_exception(capsys):
    app, _, _ = application()
    app.handle = Mock(side_effect=RuntimeError('secret-key-SYNTHETIC-fixture'))
    with server_for(app) as port:
        status, headers, body = request(port, 'GET', '/?token=SYNTHETIC-query-secret')
    assert status == 500
    assert headers['Cache-Control'] == 'no-store'
    assert headers['Referrer-Policy'] == 'same-origin'
    assert headers['X-Frame-Options'] == 'DENY'
    assert "frame-ancestors 'none'" in headers['Content-Security-Policy']
    output = body.decode() + capsys.readouterr().err
    assert 'SYNTHETIC' not in output and 'Traceback' not in output


class Forms(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form':
            self.current = {'method': attrs.get('method', 'get'), 'tokens': []}
            self.forms.append(self.current)
        if tag == 'input' and attrs.get('name') == '_csrf' and self.current is not None:
            self.current['tokens'].append(attrs['value'])

    def handle_endtag(self, tag):
        if tag == 'form':
            self.current = None


def test_rendered_post_forms_have_one_token_and_support_repeated_bulk_fields():
    app, case, _ = application()
    for path in ['/', '/drafts/new', '/test-cases/manual', f'/test-cases/{case.id}', f'/test-cases/{case.id}/edit', '/export']:
        parser = Forms()
        parser.feed(app.handle('GET', path).body.decode())
        assert parser.forms, path
        for form in parser.forms:
            if form['method'].lower() == 'post':
                assert form['tokens'] == [app._csrf_token], path
    app._handle_post = Mock(return_value=WebResponse.json(200, '{}'))
    with server_for(app) as port:
        status, _, _ = request(port, 'POST', '/test-cases/export',
            f'_csrf={app._csrf_token}&test_case_id=one&test_case_id=two',
            {'Origin': f'http://127.0.0.1:{port}'})
        assert status == 200


def test_form_byte_limits_and_csrf_transport_field_do_not_change_business_fields():
    assert _parse_form_body('name=é', max_bytes=6)[1] == 'The request was too large.'
    assert _parse_form_body('_csrf=token&name=case') == ({'name': ['case']}, None)
    assert _parse_form_body('_csrf=one&_csrf=two')[1] is not None


def test_default_server_binding_and_socket_timeout_are_local_and_bounded():
    app, _, _ = application()
    server = create_http_server(app, port=0)
    client = socket.create_connection(server.server_address, timeout=5)
    try:
        connection, _ = server.get_request()
        try:
            assert server.server_address[0] == '127.0.0.1'
            assert connection.gettimeout() == 10
        finally:
            connection.close()
    finally:
        client.close()
        server.server_close()


@pytest.mark.parametrize('target', ['http://attacker.invalid/', '//attacker.invalid/', '/#fragment', 'http://[invalid'])
def test_invalid_targets_are_safe_client_errors(target):
    app, _, _ = application()
    assert app.handle('GET', target, headers={}).status == 400


def test_ci_yaml_runs_the_complete_offline_suite_without_credentials_or_artifacts():
    # Playwright already bundles a YAML parser and Node; no added dependency.
    driver = Path(playwright.__file__).parent / 'driver'
    node = driver / ('node.exe' if (driver / 'node.exe').exists() else 'node')
    bundle = driver / 'package/lib/utilsBundle.js'
    workflow = Path(__file__).resolve().parents[1] / '.github/workflows/offline-tests.yml'
    script = "const fs=require('fs'); const {yaml}=require(process.argv[1]); process.stdout.write(JSON.stringify(yaml.parse(fs.readFileSync(process.argv[2], 'utf8'))));"
    result = subprocess.run([str(node), '-e', script, str(bundle), str(workflow)], capture_output=True, text=True, check=True, timeout=10)
    parsed = json.loads(result.stdout)
    assert set(parsed['on']) == {'push', 'pull_request', 'workflow_dispatch'}
    assert parsed['permissions'] == {'contents': 'read'}
    job = parsed['jobs']['offline-tests']
    assert job['runs-on'] == 'ubuntu-24.04'
    assert job['env']['LLM_PROVIDER_ORDER'] == ''
    steps = job['steps']
    assert next(s for s in steps if s.get('uses', '').startswith('actions/setup-python@'))['with']['python-version'] == '3.13'
    commands = [s['run'] for s in steps if 'run' in s]
    assert 'python -m pip install -r requirements.txt' in commands
    assert 'python -m playwright install --with-deps chromium' in commands
    assert 'git diff --check' in commands
    assert 'git show --format= --check HEAD' in commands
    assert commands[-1].split() == ['python', '-m', 'pytest', '-q',
        '--ignore=tests/integration/test_openrouter.py',
        '--deselect=tests/test_browser_discovery.py::BrowserDiscoveryTests::test_selenium_disabled_input_visible_text_selector_matches_its_element']
    text = workflow.read_text()
    assert 'secrets.' not in text and 'upload-artifact' not in text and 'continue-on-error' not in text
