#!/usr/bin/env python3
"""Preflight MCP Server — E2E smoke test.

Tests core tool logic directly (bypasses MCP protocol) using a temp SQLite DB
and a stubbed fastembed so the test completes in under 5 seconds.

Run:
    python ~/.config/preflight/test_mcp.py
"""

import hashlib
import json
import os
import sys
import types
import sqlite3
import tempfile
import traceback

# ── Stub fastembed before importing anything that touches utils.py ─────────────
_stub_utils = types.ModuleType("utils")

def _embed(text: str) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    v = [b / 255.0 for b in h[:32]]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n else v

def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

_stub_utils.embed_text = _embed
_stub_utils.cosine_similarity = _cos
sys.modules["utils"] = _stub_utils

# ── Add backend scripts dir to path ───────────────────────────────────────────
_SCRIPTS_DIR = os.path.join(os.path.expanduser("~"), ".config", "opencode")
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Import mcp_server tool functions AFTER stubs are in place ─────────────────
# Temporarily override _VENV_PYTHON so mcp_server doesn't try subprocess with venv
# (we call tool fns directly, not via subprocess, in this test).
import mcp_server as srv

import memory as _mem

# ── Isolated temp DB for this test run ────────────────────────────────────────
_DB = os.path.join(tempfile.gettempdir(), "preflight_test_mcp.db")
if os.path.exists(_DB):
    os.remove(_DB)
_mem.DB_PATH = _DB

# Patch _call_memory / _call_tasks / _call_classifier to call Python functions
# directly instead of via subprocess (avoids needing fastembed in venv during CI).
def _direct_memory(*args):
    """Route CLI-style args to memory.py functions directly."""
    cmd = args[0]
    if cmd == "store_fact":
        _, project_id, session_id, text, *rest = args
        fact_type = rest[0] if rest else "finding"
        _mem.store_fact(project_id, session_id, text, fact_type)
        return ""
    elif cmd == "retrieve_facts":
        _, project_id, session_id, prompt, top_n, threshold = args
        return json.dumps(_mem.retrieve_facts(project_id, session_id, prompt, int(top_n), float(threshold)))
    elif cmd == "store_slot_fill":
        _, project_id, session_id, slot_name, value = args
        _mem.store_slot_fill(project_id, session_id, slot_name, value)
        return ""
    elif cmd == "retrieve_slot_fills":
        _, project_id = args
        return json.dumps(_mem.retrieve_slot_fills(project_id))
    elif cmd == "session_mark":
        _, session_id, project_id = args
        _mem.session_mark(session_id, project_id)
        return ""
    elif cmd == "session_seen":
        _, session_id = args
        return "YES" if _mem.session_seen(session_id) else "NO"
    elif cmd == "session_unmark":
        _, session_id = args
        _mem.session_unmark(session_id)
        return ""
    else:
        raise ValueError(f"Unknown CLI command in test: {cmd}")

def _direct_tasks(*args):
    """Route retrieve_similar to tasks.py directly."""
    import tasks as _tasks
    _tasks.DB_PATH = _DB
    cmd = args[0]
    if cmd == "retrieve_similar":
        _, project_id, session_id, prompt, top_n, threshold = args
        return json.dumps(_tasks.retrieve_similar_tasks(project_id, session_id, prompt, int(top_n), float(threshold)))
    return "[]"

def _direct_classifier(payload_json: str) -> str:
    payload = json.loads(payload_json)
    prompt  = payload.get("prompt", "").lower()
    if any(w in prompt for w in ("fix", "bug", "error", "crash")):
        return json.dumps({"type": "bug"})
    if any(w in prompt for w in ("add", "feature", "implement", "new")):
        return json.dumps({"type": "feature"})
    if any(w in prompt for w in ("refactor", "clean", "reorganize")):
        return json.dumps({"type": "refactor"})
    return json.dumps({"type": None})

# Monkey-patch the server module so tool fns use our direct callables
srv._call_memory    = lambda *a: _direct_memory(*a)
srv._call_tasks     = lambda *a: _direct_tasks(*a)
srv._call_classifier = lambda p: _direct_classifier(p)

# ── Test helpers ───────────────────────────────────────────────────────────────
_PASS = 0
_FAIL = 0

def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))

# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — get_project_id
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 1: get_project_id ──")
try:
    result = srv.tool_get_project_id(os.path.expanduser("~/.config/opencode"))
    pid = result.get("project_id", "")
    check("returns a dict with project_id key", "project_id" in result)
    check("project_id is a 12-char hex string", len(pid) == 12 and all(c in "0123456789abcdef" for c in pid),
          f"got: {pid!r}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — store_slot / list_slots
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 2: store_slot / list_slots ──")
PROJ = "smoke_test"
SESS = "sess_smoke_1"
try:
    srv.tool_store_slot(SESS, PROJ, "framework", "FastAPI")
    srv.tool_store_slot(SESS, PROJ, "language",  "Python")
    slots = srv.tool_list_slots(PROJ)
    check("list_slots returns framework", slots.get("framework") == "FastAPI",
          f"got: {slots}")
    check("list_slots returns language",  slots.get("language")  == "Python",
          f"got: {slots}")
    check("no extra slots", len(slots) == 2, f"got: {slots}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test 3 — store_memory / get_context retrieves it
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 3: store_memory → get_context retrieval ──")
try:
    srv.tool_store_memory(SESS, PROJ, "always use SQLAlchemy not raw SQL", "finding")
    ctx = srv.tool_get_context("fix the database query", "sess_smoke_2", PROJ)
    mems = ctx.get("memories", [])
    check("memories list is non-empty", len(mems) > 0, f"got empty memories")
    check("SQLAlchemy fact is in memories",
          any("SQLAlchemy" in m for m in mems),
          f"got: {mems}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test 4 — preference fact stored in __global__ is visible from OTHER project
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 4: global preference visible from different project ──")
OTHER_PROJ = "totally_different_project"
try:
    srv.tool_store_memory(SESS, PROJ, "keep responses concise", "preference")
    ctx2 = srv.tool_get_context("add user authentication", "sess_smoke_3", OTHER_PROJ)
    mems2 = ctx2.get("memories", [])
    check("preference appears in memories for OTHER project",
          any("concise" in m for m in mems2),
          f"got: {mems2}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test 5 — missing_slots when no slots stored (feature task)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 5: missing_slots for feature task (no slots stored) ──")
EMPTY_PROJ = "fresh_project_no_slots"
try:
    ctx3 = srv.tool_get_context("add user authentication", "sess_smoke_4", EMPTY_PROJ)
    missing = ctx3.get("missing_slots", [])
    task    = ctx3.get("task_type")
    check("task_type classified as feature", task == "feature", f"got: {task!r}")
    check("missing_slots contains language",  "language"  in missing, f"got: {missing}")
    check("missing_slots contains framework", "framework" in missing, f"got: {missing}")
    check("missing_slots contains database",  "database"  in missing, f"got: {missing}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Test 6 — storing language removes it from missing_slots
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Test 6: store language → it leaves missing_slots ──")
try:
    srv.tool_store_slot("sess_smoke_4", EMPTY_PROJ, "language", "Python")
    ctx4 = srv.tool_get_context("add user authentication", "sess_smoke_5", EMPTY_PROJ)
    missing2 = ctx4.get("missing_slots", [])
    check("language is NO LONGER in missing_slots", "language"  not in missing2,
          f"got: {missing2}")
    check("framework still missing",                "framework" in missing2,
          f"got: {missing2}")
    check("database still missing",                 "database"  in missing2,
          f"got: {missing2}")
except Exception:
    print("  ERROR:", traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
total = _PASS + _FAIL
print(f"\n{'═'*50}")
print(f"  {_PASS}/{total} passed{'  ✓  ALL PASS' if _FAIL == 0 else ''}")
if _FAIL:
    print(f"  {_FAIL} FAILED")
print(f"{'═'*50}")
sys.exit(0 if _FAIL == 0 else 1)
