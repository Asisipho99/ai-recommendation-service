"""The CORS allowlist must never resolve to something that blocks everything.

A variable that is set but empty is the case that actually broke production:
os.getenv returns '' (not None), so a getenv default is skipped, ''.split(',')
yields [''], and flask-cors then matches no origin at all - including
localhost. The service still answers preflights 200, just without an
Access-Control-Allow-Origin header, so the browser blocks every real request
and the only symptom is OPTIONS traffic with no matching GET.

Run in subprocesses because the allowlist is resolved once at import time.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]

SNIPPET = "import app, json; print(json.dumps(app.allowed_origins))"

EXPECTED_DEFAULT = {
    "http://localhost:4200",
    "http://localhost:4201",
    "https://shopping-admin-production.up.railway.app",
    "https://shopping-customer-production.up.railway.app",
}


def resolve(value):
    """Import the app with CORS_ALLOWED_ORIGINS set to `value`."""
    env = dict(os.environ)
    if value is None:
        env.pop("CORS_ALLOWED_ORIGINS", None)
    else:
        env["CORS_ALLOWED_ORIGINS"] = value
    # app.py calls load_dotenv(), which defaults to override=False, so an
    # explicitly set variable here wins over .env. That is what lets these
    # cases be asserted on the variable alone.
    proc = subprocess.run([sys.executable, "-c", SNIPPET], env=env,
                          cwd=SERVICE_ROOT, capture_output=True, text=True,
                          timeout=180)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_empty_string_falls_back_to_defaults():
    # The production bug: set but empty.
    assert set(resolve("")) == EXPECTED_DEFAULT


def test_only_separators_falls_back_to_defaults():
    assert set(resolve(" , , ")) == EXPECTED_DEFAULT


def test_whitespace_only_falls_back_to_defaults():
    assert set(resolve("   ")) == EXPECTED_DEFAULT


def test_real_values_are_trimmed():
    got = resolve("  https://a.example.com , https://b.example.com  ")
    assert got == ["https://a.example.com", "https://b.example.com"]


def test_trailing_comma_does_not_create_an_empty_origin():
    got = resolve("https://a.example.com,")
    assert got == ["https://a.example.com"]
    assert "" not in got


def test_allowlist_is_never_empty():
    for value in (None, "", "  ", ",", ",,,"):
        assert resolve(value), f"empty allowlist for {value!r}"
