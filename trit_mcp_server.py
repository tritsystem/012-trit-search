"""
012 OBSERVE MCP Server

Exposes OBSERVE's compressed semantic code search as an MCP (Model
Context Protocol) server, so AI coding assistants (Claude Code, Claude
Desktop, or any MCP-compatible client) can call it as a tool — search
your local, fine-tuned, ternary-compressed code index directly from a
conversation, without leaving your editor or copy-pasting results.

This wraps the exact same SearchEngine used by trit_app.py (OBSERVE's
GUI) — same model, same compressed index, same search logic. No GUI is
launched; this runs headless over stdio, the standard local-MCP transport.

Setup:
  pip install mcp
  Build an index first with trit_app.py (click INDEX CODEBASE) or
  trit_search.py --index — this server reads the existing index, it
  does not build one itself.

Usage (standalone test):
  python trit_mcp_server.py

Usage (as an MCP server, e.g. in Claude Code's mcp config):
  {
    "mcpServers": {
      "observe": {
        "command": "python",
        "args": ["C:/path/to/012-ternary/trit_mcp_server.py"]
      }
    }
  }
"""
import sys
import json
import subprocess
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trit_app import SearchEngine
from observe_pipeline import group_chunks_by_project, NON_PROJECT_HINTS

try:
    # mcp-gateway is a separate, shared local package (also used by Spikeling's MCP
    # server) -- not published anywhere, so it is genuinely optional: importing this
    # module (and running the server) must work without it, for CI, for a fresh
    # `pip install`, and for anyone whose machine doesn't have a sibling mcp-gateway
    # checkout. When it IS present, real audit logging + rate limiting apply.
    sys.path.insert(0, str(Path.home() / "OneDrive" / "Documents" / "mcp-gateway"))
    from wrap import make_gateway
except ImportError:
    def make_gateway(server_name, audit_path, default_rate_per_min=60):
        class _NoopLimiter:
            def set_limit(self, *a, **kw):
                pass

        def gated(tool_name, rate_per_min=None):
            def deco(fn):
                return fn
            return deco

        return gated, None, _NoopLimiter()

from mcp.server.fastmcp import FastMCP

INDEX_DIR  = str(Path.home() / ".trit-search" / "index")
MODEL_PATH = str(Path(__file__).resolve().parent / "models" / "code-minilm")
if not Path(MODEL_PATH).exists():
    MODEL_PATH = "all-MiniLM-L6-v2"  # fall back to baseline if not fine-tuned locally

mcp = FastMCP("observe")
engine = SearchEngine()

# Local-only audit log + rate limiting (see mcp-gateway/README-less package --
# self-contained, single-operator scope; not wired into observe-api's separate
# hosted commerce endpoints, which already have their own rate limiting).
gated, _audit, _limiter = make_gateway(
    "observe", str(Path.home() / ".trit-search" / "audit_log.jsonl"))
_limiter.set_limit("apply_and_verify", 20)  # the one tool that writes real files -- tightest limit

# ── Stable vs experimental tool surface ───────────────────────────────────────
# The stable, benchmarked surface is search_code / query_codebase /
# index_status (see paper/token_reduction_findings.md and
# paper/quality_benchmark_findings.md for the measurements behind them).
# Everything else (propose_change, apply_and_verify, and the entanglement
# family) is experimental and known-unstable; it only registers when the
# environment sets OBSERVE_EXPERIMENTAL=1.
import os
EXPERIMENTAL = os.environ.get("OBSERVE_EXPERIMENTAL", "") == "1"

def experimental_tool(fn):
    return mcp.tool()(fn) if EXPERIMENTAL else fn

import time

_loaded = {"done": False, "error": None, "loading_started": False}
_status = {"msg": "Not started", "ts": 0.0}
_load_lock = threading.Lock()
_pending = {"attempt": None}   # holds the one AttemptState currently "live"

# Real measured root cause (2026-08-12, see paper/mcp_stall_findings.md):
# a stuck load is NOT the ~55s sentence_transformers import chain running
# slow -- py-spy attached to a genuinely-hung server process showed the
# background thread parked at zero CPU inside importlib's create_module()
# for faiss's native _swigfaiss.pyd, i.e. blocked *inside* the OS-level
# LoadLibrary call itself, with amsi.dll/MpOav.dll (Windows Defender/AMSI)
# loaded into the same process. That's a real native DLL-load stall (almost
# certainly AV real-time-scan + Windows loader-lock interaction), not
# anything this Python code is doing -- outside this codebase's control,
# and NOT reproducible on demand (a warm AV scan cache makes a fresh
# in-isolation re-import fast, which is why isolated repro attempts didn't
# hang). What IS a real, fixable, in-codebase bug: once that native call
# wedges, a Python thread blocked inside a C extension can never be killed
# or interrupted from pure Python -- so the *previous* version of this
# function derived "time since last status" from the module-level
# _status["ts"], read fresh at the top of every call and compared to
# whatever time had ALREADY elapsed. Once a real stall pushed that gap past
# STALL_LIMIT once, every later call recomputed the same ever-growing gap,
# hit the stall branch on its very first 0.5s poll, and returned WITHOUT
# ever resetting loading_started -- so the server reported the exact same
# "not ready" message forever, with no way to recover short of killing the
# process (this is what actually produced the 2-day-stuck PID tonight).
#
# Fix, two parts:
#  1) Track "time since last real progress" PER ATTEMPT, in an isolated
#     AttemptState object captured by that attempt's own on_status closure
#     -- not a shared module-level timestamp. This preserves the original
#     (correct) intent of tolerating a load that's merely slow but still
#     genuinely progressing (important: this machine's real processes can
#     contend with each other, per tonight's multi-spawn observations, so
#     "elapsed since attempt START" alone would falsely abandon a healthy
#     but contended load -- measured and confirmed by testing with
#     shrunk-down limits; verified this per-progress design tolerates it).
#     Critically, this isolation also means that IF a leaked, abandoned,
#     truly-wedged thread ever does unblock later and calls on_status, it
#     only ever updates its OWN AttemptState -- it can never bleed into or
#     reset a newer attempt's stall clock.
#  2) On genuine abandonment, reset loading_started so the NEXT call starts
#     a real fresh attempt on a brand-new SearchEngine() instance (never
#     reusing the singleton) -- the old wedged thread is leaked (Python
#     genuinely cannot kill a thread blocked in a C extension), but it only
#     ever touches its own now-unreferenced engine object. The module-level
#     `engine` is swapped in only once a load actually completes. _load_lock
#     guards the check-and-start of a new attempt, also closing a real (if
#     narrow) pre-existing race where two concurrent tool calls could both
#     see loading_started=False and both start a load.
#
# Honest limit on (2), found via LIVE evidence, not theory: after deploying
# this fix, py-spy on the actual live server showed a real, deeper wall --
# the retried attempt's thread wasn't stuck in create_module like the first
# one; it was blocked in importlib's _lock_unlock_module, waiting to acquire
# CPython's own per-module-name import lock for `faiss`, which the FIRST,
# still-wedged thread was still holding (it never got far enough to release
# it). This is a real, unavoidable CPython constraint: import locks are
# keyed by module name and shared process-wide, so ANY new thread that
# tries to `import faiss` again -- no matter how many fresh SearchEngine
# attempts we spawn -- will always queue behind the same lock a wedged
# thread holds forever. Retrying within the same process literally cannot
# out-run this specific failure mode; only killing the whole process (which
# releases all its locks with it) can. So: MAX_SAME_PROCESS_RETRIES caps
# how many times this process will spawn a fresh in-process attempt before
# giving up on retrying at all and saying so plainly -- better than an
# infinite chain of equally-doomed retries that only leak more zombie
# threads while implying progress that can't actually happen.
STALL_ABANDON_LIMIT = 150   # real seconds since this attempt's last progress
                             # signal (not since it started) before treating
                             # it as genuinely wedged rather than slow
PER_CALL_WAIT_CAP = 180     # a single tool call blocks polling at most this
                             # long before returning control to the caller
MAX_SAME_PROCESS_RETRIES = 2   # after this many abandoned attempts, stop
                                 # retrying in-process and say so honestly

class _AttemptState:
    __slots__ = ("engine", "last_progress_ts")
    def __init__(self, engine):
        self.engine = engine
        self.last_progress_ts = time.monotonic()

_retry_count = {"n": 0}

def _ensure_loaded():
    global engine
    if _loaded["done"]:
        return

    if _retry_count["n"] > MAX_SAME_PROCESS_RETRIES:
        _loaded["error"] = (
            f"Gave up after {_retry_count['n']} abandoned load attempts in "
            f"this process -- real evidence (py-spy) shows retries queue "
            f"behind CPython's own import lock for a module a still-wedged "
            f"earlier thread holds forever, so more in-process retries "
            f"cannot succeed. This server process needs to be restarted "
            f"(kill it; your MCP client should respawn a clean one) -- "
            f"this is not something a code-level retry can fix."
        )
        return

    with _load_lock:
        if not _loaded["loading_started"]:
            new_engine = SearchEngine()
            attempt = _AttemptState(new_engine)

            def on_status(msg, _attempt=attempt):
                _status["msg"] = msg
                _status["ts"] = time.monotonic()
                _attempt.last_progress_ts = time.monotonic()
                print(f"[OBSERVE] {msg}", file=sys.stderr)

            _status["msg"] = "Starting..."
            _status["ts"] = time.monotonic()
            _pending["attempt"] = attempt
            new_engine.load(INDEX_DIR, MODEL_PATH, on_status)
            _loaded["loading_started"] = True
            _loaded["error"] = None

    attempt = _pending["attempt"]
    candidate = attempt.engine
    waited_this_call = 0.0
    while not candidate.ready:
        stalled_for = time.monotonic() - attempt.last_progress_ts
        if stalled_for > STALL_ABANDON_LIMIT or waited_this_call >= PER_CALL_WAIT_CAP:
            break
        time.sleep(0.5)
        waited_this_call += 0.5

    if candidate.ready:
        with _load_lock:
            engine = candidate   # atomic swap -- other tool calls read this global
            _loaded["done"] = True
            _loaded["error"] = None
        return

    stalled_for = time.monotonic() - attempt.last_progress_ts
    if stalled_for > STALL_ABANDON_LIMIT:
        # No progress signal for real, per-attempt-tracked seconds -- treat
        # as genuinely wedged, not just slow. Give up on THIS attempt for
        # good so the *next* call gets a real fresh try instead of
        # repeating this same check forever (up to MAX_SAME_PROCESS_RETRIES).
        with _load_lock:
            _loaded["loading_started"] = False
        _retry_count["n"] += 1
        remaining = MAX_SAME_PROCESS_RETRIES - _retry_count["n"]
        retry_note = (
            f"the next call will start a fresh one ({remaining} more "
            f"in-process retr{'y' if remaining == 1 else 'ies'} left before "
            f"this server gives up and asks for a restart instead)."
            if remaining > 0 else
            "no in-process retries are left -- the NEXT call will report "
            "that this server needs to be restarted."
        )
        _loaded["error"] = (
            f"Load attempt has had no progress for {stalled_for:.0f}s (last "
            f"status: \"{_status['msg']}\") -- treating it as genuinely "
            f"wedged (likely a native DLL load blocked outside Python's "
            f"control, e.g. AV real-time scanning interacting with the "
            f"Windows loader lock -- see comment above), not just slow. "
            f"Abandoning this attempt; {retry_note}"
        )
    else:
        _loaded["error"] = (
            f"Still loading (no progress signal for {stalled_for:.0f}s, "
            f"last status: \"{_status['msg']}\") -- try again shortly."
        )

@mcp.tool()
@gated("search_code")
def search_code(query: str, k: int = 10, project_dir: str = "") -> str:
    """
    Search the local OBSERVE code index by meaning, not exact keywords.

    Use this to find relevant code in the user's indexed codebase(s) when
    they ask about functionality, e.g. "where is health damage handled"
    or "find the function that retries network requests" — it searches
    by semantic similarity using a fine-tuned embedding model, so it
    finds conceptually relevant code even if it doesn't share words with
    the query.

    IMPORTANT — prefer Grep when you already know the exact identifier
    (function/variable/class name, error string, etc). Measured directly
    (see paper/grep_vs_semantic_findings.md): on 5 exact-keyword queries,
    Grep found the correct file 5/5 times; this tool's dedup+relevance-cutoff
    missed 2/5 even when the query WAS the literal function name. This tool's
    real advantage is on natural-language/conceptual queries where the exact
    identifier is unknown — there it matched Grep's recall using a single
    call instead of several sequential exploratory Grep guesses, at ~90%
    fewer tokens. Use Grep first if you can name the thing you're looking
    for; use semantic search when you can only describe it.

    Args:
        query: A natural-language description of what to find.
        k: Number of results to return (default 10).
        project_dir: Optional absolute path to scope results to a single
            indexed project's base directory (e.g. the user's current
            working directory). When set, results from other indexed
            codebases are excluded instead of crowding out the relevant
            project. Leave empty to search across all indexed codebases.

    Returns:
        A formatted list of matching files with relevance scores and
        code previews, ranked by relevance (highest first).
    """
    _ensure_loaded()
    if _loaded["error"]:
        return f"Error: {_loaded['error']}. Build an index first with trit_app.py or trit_search.py --index."

    results = engine.search(query, k=k, base_dir_filter=project_dir or None)
    if not results:
        if project_dir:
            return f"No results under {project_dir} — it may not be indexed, or try without project_dir to search all indexed codebases."
        return "No results — index may be empty. Build one with trit_app.py (INDEX CODEBASE) or trit_search.py --index."

    lines = [f"Found {len(results)} results for: \"{query}\"\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r['score']:.3f}] {r['path']}")
        lines.append(f"   {r['preview'][:150]}")
    return "\n".join(lines)

@experimental_tool
@gated("hybrid_search_code")
def hybrid_search_code(query: str, k: int = 10, project: str = "", chunk_type: str = "",
                        path_glob: str = "", expand_graph: bool = False) -> str:
    """
    search_code, extended with real structured filtering and optional
    graph expansion over a provenance/lineage layer built alongside the
    embedding index (see chunk_provenance.py). Results are still ranked
    by the same semantic similarity search_code uses -- filtering narrows
    the candidate set, it never re-ranks -- but each result now carries
    real metadata (exact project, line range, chunk type) instead of a
    bare path+preview, and can optionally show real call/import
    relationships to OTHER chunks, not just text matches to the query.

    Use this over search_code when you need to scope a search precisely
    (e.g. "only in project X", "only function definitions, not stray
    text/config") or when you want to see what a matched function calls
    or is called by, which pure semantic similarity cannot surface (a
    caller and callee can have zero textual/semantic overlap).

    Args:
        query: Natural-language description of what to find.
        k: Number of results to return (default 10).
        project: Optional exact project name to restrict to (see
            list_indexed_projects for real names in this index).
        chunk_type: Optional filter -- "function", "class", or "text"
            (unclassified; currently only Python chunks get real
            function/class tags from AST parsing).
        path_glob: Optional glob (e.g. "*.py", "*test*") to restrict by
            relative file path.
        expand_graph: If true, attach each result's real 1-hop call/import
            neighbors (Python-only; cross-project matches are tagged by
            how confidently they were resolved -- see chunk_provenance.py).

    Returns:
        Formatted results with project/line-range/chunk-type per hit, and
        real graph neighbors if expand_graph=True. Reports itself
        unavailable (not an error) if `python chunk_provenance.py --build`
        hasn't been run yet.
    """
    _ensure_loaded()
    if _loaded["error"]:
        return f"Error: {_loaded['error']}. Build an index first with trit_app.py or trit_search.py --index."
    from hybrid_search import hybrid_search, format_results
    result = hybrid_search(engine, query, k=k, project=project or None, chunk_type=chunk_type or None,
                            path_glob=path_glob or None, expand_graph=expand_graph)
    return format_results(result)

@mcp.tool()
@gated("query_codebase")
def query_codebase(query: str, k: int = 8, project_dir: str = "") -> str:
    """
    Token-tight variant of search_code: same semantic search, but
    deduplicated (one result per file, best-scoring chunk only),
    relevance-filtered (drops low-scoring results below a fraction of the
    top score instead of always returning a fixed k), and formatted as
    one compact line per result instead of search_code's two-line,
    blank-line-separated format.

    Use this instead of search_code when you want the smallest reasonable
    token footprint and don't need every low-relevance result — e.g. when
    scanning many queries in one session, or working in a tight context
    budget. Use search_code instead when you want the full, uncollapsed
    result set (e.g. multiple chunks from the same file matter, or you
    want to see low-scoring results too).

    IMPORTANT — prefer Grep when you already know the exact identifier.
    This tool's dedup+relevance-cutoff is more aggressive than search_code's
    and measured worse on exact-keyword queries as a result (3/5 recall vs
    Grep's 5/5, see paper/grep_vs_semantic_findings.md) — it can drop a file
    even when the query is a dead-exact keyword match. Its real advantage is
    natural-language/conceptual queries where no exact identifier is known.

    Args:
        query: A natural-language description of what to find.
        k: Maximum number of results to return after dedup/filtering (default 8).
        project_dir: Optional absolute path to scope results to one indexed project.

    Returns:
        One compact line per result: "path  score  preview".
    """
    _ensure_loaded()
    if _loaded["error"]:
        return f"Error: {_loaded['error']}. Build an index first with trit_app.py or trit_search.py --index."

    # over-fetch so dedup/filtering still has enough of a pool to choose from
    raw = engine.search(query, k=max(k * 4, 20), base_dir_filter=project_dir or None)
    if not raw:
        if project_dir:
            return f"No results under {project_dir}."
        return "No results."

    # dedup: keep only the best-scoring chunk per file
    best_per_path = {}
    for r in raw:
        if r["path"] not in best_per_path or r["score"] > best_per_path[r["path"]]["score"]:
            best_per_path[r["path"]] = r
    deduped = sorted(best_per_path.values(), key=lambda r: -r["score"])

    # relevance cutoff: drop results far below the top score instead of
    # padding out to a fixed k with noise (always keep at least 1)
    top_score = deduped[0]["score"]
    threshold = top_score * 0.7
    filtered = [r for r in deduped if r["score"] >= threshold] or deduped[:1]
    final = filtered[:k]

    lines = []
    for r in final:
        preview = " ".join(r["preview"].split())[:90]   # collapse whitespace, tight preview
        lines.append(f"{r['path']}  {r['score']:.2f}  {preview}")
    return "\n".join(lines)

OLLAMA_MODEL = "qwen2.5-coder:7b"
# Separate model for the semantic-role verification call — measured
# directly: qwen2.5-coder:7b (code-focused) missed a real semantic
# mismatch that deepseek-r1:7b (reasoning-focused) caught correctly, on
# the identical prompt (see paper/observe_ollama_findings.md). Generation
# and verification are different tasks; the best model for one is not
# necessarily the best model for the other, even both being 7B-scale.
OLLAMA_VERIFY_MODEL = "deepseek-r1:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    import urllib.request, json as _json
    body = _json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return _json.loads(resp.read())["response"].strip()

LOW_CONFIDENCE_THRESHOLD = 4.0   # below this top-score, treat search context as weak (see trit_cutoff_sweep_test.py for real score ranges)

@experimental_tool
@gated("propose_change")
def propose_change(request: str, project_dir: str = "", model: str = OLLAMA_MODEL) -> str:
    """
    For a vague, ambiguous code-change request (e.g. "make turrets shoot
    faster", "make the shop cheaper") with no specific number or exact
    behavior stated, proposes ONE concrete, grounded default and phrases
    a short confirmation question — instead of silently guessing or
    blocking on the ambiguity.

    Grounds the proposal in REAL existing code: runs a semantic search
    over the indexed project first, and includes the actual matched
    code (with any real existing numeric values) in the prompt. The
    model is REQUIRED to quote the literal source line it grounds a
    proposal in, or explicitly say no relevant existing value was found
    — measured testing (see paper/) found it will otherwise sometimes
    fabricate a plausible-sounding but fake specific detail when the
    search context is weak or truncated, which is worse than an honest
    "no grounding available."

    Uses a local model via Ollama (standard GGUF quantization) running
    entirely on this machine — no cloud call. NOTE: this project's own
    experimental ternary weight quantization was tested on a 7B model
    and produced incoherent garbage (see weight_quantization_findings.md)
    — this tool deliberately uses Ollama's proven quantization instead.

    Use this BEFORE implementing an ambiguous request, so the specific
    number/behavior chosen is explicit, grounded, and confirmable, not
    a silent guess buried inside a code change.

    Args:
        request: The user's plain-language request, as given.
        project_dir: Absolute path to scope the grounding search to one
            indexed project. Strongly recommended — without it, the
            proposal has no real code to anchor to.
        model: Which Ollama model to use (default qwen2.5-coder:7b).
            Pass a different installed model name (e.g. "deepseek-r1:7b")
            to compare hallucination/grounding behavior across models.

    Returns:
        A short proposal: what's ambiguous, a specific suggested value
        (with a literal quote of its real source, or an explicit
        "no grounding found" statement), and a confirmation question.
    """
    _ensure_loaded()
    context = ""
    confidence_note = ""
    if not _loaded["error"]:
        results = engine.search(request, k=5, base_dir_filter=project_dir or None)
        if results:
            top_score = results[0]["score"]
            lines = []
            for r in results:
                preview = " ".join(r["preview"].split())[:350]   # widened from 150 — reduces truncation cutting off the real value
                lines.append(f"{r['path']} (score={r['score']:.2f}): {preview}")
            context = "\n\nReal existing code found in this project, relevant to the request:\n" + "\n".join(lines)
            if top_score < LOW_CONFIDENCE_THRESHOLD:
                confidence_note = (
                    f"\n\nNOTE: the best match above scored only {top_score:.2f}, a WEAK match "
                    "(low confidence). Do not treat this as reliable grounding — if nothing above "
                    "is clearly relevant, say so explicitly rather than using it."
                )
        else:
            confidence_note = "\n\nNOTE: no search results were found at all. There is no existing code to ground a proposal in."

    prompt = (
        "A user gave this code-change request: \"" + request + "\"\n"
        + context + confidence_note + "\n\n"
        "If a specific real value relevant to this request appears in the code above, "
        "quote it directly and propose a moderate, easily-reversible adjustment to it. "
        "If nothing above is clearly relevant, propose a reasonable generic default "
        "instead — do not invent a specific-sounding detail (a number, mechanism, or "
        "field name) that does not literally appear in the search results above. "
        "If the request is already fully specific, just say so in one sentence. "
        "Respond in 2-4 sentences maximum, no preamble."
    )
    try:
        raw = _call_ollama(prompt, model=model)
        return _tag_with_verified_grounding(raw, context, request, model=model)
    except Exception as e:
        return f"Error calling local Ollama model ({model}): {e}. Is Ollama running?"

import re

_COMMON_WORDS = {
    "the", "a", "an", "is", "are", "to", "of", "in", "for", "and", "or",
    "this", "that", "with", "value", "code", "make", "adjust", "adjustment",
    "increase", "decrease", "consider", "could", "would", "should", "may",
    "specific", "real", "default", "generic", "moderate", "current", "change",
    "existing", "relevant", "request", "project", "provided", "snippet",
    "snippets", "shown", "above", "however", "reasonable", "without",
    "being", "than", "based", "using", "used", "already", "somewhat",
}
# Programming keywords across the languages this project indexes — these
# trivially co-occur in almost any two code samples and are not evidence
# of grounding in any SPECIFIC piece of code.
_KEYWORDS = {
    "if", "else", "elif", "for", "while", "def", "func", "var", "return",
    "continue", "break", "class", "import", "from", "true", "false", "none",
    "null", "and", "or", "not", "pass", "self", "print", "int", "float",
    "str", "bool", "void", "extends", "export", "const", "let", "static",
    "public", "private", "protected", "try", "except", "finally", "with",
    "as", "lambda", "yield", "async", "await",
}

def _distinctive_tokens(text: str, exclude: set = frozenset()) -> set:
    """
    Tokens worth treating as evidence of real grounding: numeric literals
    (2+ chars, so bare "1"/"0" don't count), and identifier-shaped words
    (snake_case/camelCase/dotted, length > 5) — filtering out common
    English filler, programming keywords, and (via `exclude`) words
    already present in the user's own request, since a query word
    trivially "matching" the context it was used to search for is
    circular, not evidence of real grounding.
    """
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*|\d+\.?\d*", text)
    out = set()
    for t in tokens:
        low = t.lower()
        if low in _COMMON_WORDS or low in _KEYWORDS or low in exclude:
            continue
        if t.replace(".", "").isdigit():
            cleaned = _clean_number(t)   # strip trailing sentence-period, e.g. "110." -> "110"
            if len(cleaned.replace(".", "")) >= 2:
                out.add(cleaned)   # numeric literal — e.g. "110", "0.92" — excludes bare "1", "0"
        elif "_" in t or any(c.isupper() for c in t[1:]) or len(t) > 5:
            out.add(t)   # identifier-shaped — e.g. "fire_rate", "N_DEFECTS", "threshold"
    return out

def _tag_with_verified_grounding(response: str, context: str, request: str = "", model: str = OLLAMA_MODEL) -> str:
    """
    Objectively checks whether the model's response shares distinctive
    tokens (numbers, identifiers) with the real search context, instead of
    trusting the model's own self-report (measured unreliable — see
    paper/observe_ollama_findings.md). Excludes the user's own query words
    (circular matching) and programming keywords (trivial co-occurrence).

    Additionally cross-checks numeric literals specifically: if the
    response states a number as if it were the CURRENT/existing value,
    but that exact number never appears anywhere in the real context,
    it's flagged separately — this catches the harder "right variable,
    fabricated number" case (e.g. correctly citing `threshold` but
    inventing `100` when the real value is `110`), which plain token
    overlap alone does not catch, since `threshold` alone would still
    match even with a wrong number attached to it.
    """
    if not context:
        return "[no search context available — unverified default] " + response

    query_words = {w.lower() for w in re.findall(r"[A-Za-z_]+", request)}
    response_tokens = _distinctive_tokens(response, exclude=query_words)
    context_tokens = _distinctive_tokens(context, exclude=query_words)
    shared = response_tokens & context_tokens

    # Numeric cross-check: numbers claimed in the response that never
    # appear anywhere in the real context at all (possible fabrication)
    response_numbers = {t for t in response_tokens if t.replace(".", "").isdigit()}
    context_numbers = {t for t in context_tokens if t.replace(".", "").isdigit()}
    unverified_numbers = response_numbers - context_numbers

    # Deterministic variable=value pair cross-check — catches the subtler
    # case token-overlap alone misses: a real variable name correctly
    # referenced, but paired with a number that never actually belonged
    # to it (a coincidental match elsewhere in context created false
    # confidence — e.g. citing spawn_timer.wait_time=1.0 when the real
    # pair is spawn_timer.wait_time=1.5, where "1.0" happens to appear
    # elsewhere in context attached to something unrelated).
    #
    # Only checked against pairs the response presents as CURRENT/EXISTING
    # state — a proposed NEW value is SUPPOSED to differ from context by
    # design (that's the point of a change), so proposed values are
    # excluded via _extract_current_state_pairs' proximity heuristic
    # (measured false-positive case: "changing k=10 to k=5" was originally
    # flagged as a "mismatch" against the real k=10, when it was actually
    # a correct, intentional proposal — see paper/observe_ollama_findings.md).
    context_pairs = _extract_assignment_pairs(context)
    response_pairs = _extract_current_state_pairs(response)
    mismatches = []
    for var, claimed_vals in response_pairs.items():
        for ctx_var, real_val in context_pairs.items():
            if var == ctx_var or var in ctx_var or ctx_var in var:
                # Only flag if NONE of the surviving claimed values for this
                # variable match reality — a single dict overwrite silently
                # dropping the correct claim (measured real bug: "threshold=100.
                # changing threshold=80 instead" kept only 80, discarding the
                # correct 100, because "instead" alone wasn't a recognized
                # proposal-phrase trigger) previously caused false mismatches.
                # Checking against ALL surviving values, not just the last
                # one seen, is robust to that class of bug even when the
                # phrase-trigger list is incomplete.
                if real_val not in claimed_vals:
                    shown = "/".join(claimed_vals)
                    mismatches.append(f'"{var}" claimed CURRENT={shown} but real value is {real_val}')
                break

    parts = []
    if shared:
        shown = ", ".join(sorted(shared)[:5])
        parts.append(f"shares real tokens: {shown}")
    else:
        parts.append("no shared distinctive tokens with search context")
    if unverified_numbers:
        shown_n = ", ".join(sorted(unverified_numbers)[:5])
        parts.append(f"numbers not found anywhere in context: {shown_n}")
    if mismatches:
        parts.append("VARIABLE=VALUE MISMATCH (verified wrong): " + "; ".join(mismatches))

    if mismatches:
        status = "FABRICATION DETECTED"
    elif shared and not unverified_numbers:
        status = "verified grounded"
    elif shared:
        status = "PARTIAL — grounded reference but unverified number(s)"
    else:
        status = "unverified"

    # Semantic-role check — a NARROWER, separate class of error that token
    # matching alone cannot catch: a number genuinely present in context,
    # correctly identified as "shared," but attached to a completely wrong
    # MEANING in the response (e.g. citing a real "1958" from an unrelated
    # historical reference elsewhere in the codebase as if it were a
    # tunable "benchmark year" parameter).
    #
    # Runs on any SHARED numeric token, regardless of overall status — NOT
    # gated on status == "verified grounded". Measured bug: that gate
    # required zero unverified numbers, but a proposed NEW value (e.g. the
    # response's own suggested replacement) is *always* unverified by
    # design, so nearly every real response landed on "PARTIAL" and the
    # check never ran in practice. Shared tokens can still have a wrong
    # semantic role even when other, unrelated numbers in the same
    # response are legitimately-unverified proposals — these are
    # independent concerns and must be checked independently. Skipped
    # entirely if a MISMATCH was already found (that's the stronger,
    # already-confirmed signal; no need for a second, costlier call).
    if not mismatches:
        shared_numbers = [t for t in shared if t.replace(".", "").isdigit()]
        for token in shared_numbers[:2]:   # cap at 2 checks — cost/latency limit
            verdict = _check_semantic_role(token, response, context)
            if verdict is False:
                status = "SEMANTIC MISMATCH DETECTED"
                parts.append(f'token "{token}" is real but its claimed meaning does not match its actual role in the source')
                break

    return f"[{status} — {'; '.join(parts)}] {response}"

def _extract_window(text: str, token: str, chars: int = 100) -> str:
    """Return the text surrounding one occurrence of `token`, as a stand-in
    for 'the sentence this token appears in' without needing real sentence
    parsing — good enough for a narrow yes/no role-matching prompt."""
    idx = text.find(token)
    if idx == -1:
        return ""
    return text[max(0, idx - chars): idx + len(token) + chars].strip()

# Consensus verification pool — deliberately diverse model families
# (code-specialized, reasoning-specialized, general-purpose), not just
# multiple sizes of the same model, since correlated failure modes would
# defeat the point of voting. Applies this project's own scoping rule
# from paper/npc_consensus_findings.md, confirmed in 5 independent domains
# tonight: voting beats trusting a single source specifically when
# individual signals are uncalibrated and fail unpredictably — which is
# exactly what was measured here (qwen2.5-coder:7b wrong, deepseek-r1:7b
# right, on the identical verification prompt).
CONSENSUS_VERIFY_MODELS = ("qwen2.5-coder:7b", "deepseek-r1:7b", "llama3.2:latest")

def _check_semantic_role(token: str, response: str, context: str, models=CONSENSUS_VERIFY_MODELS) -> bool:
    """
    Consensus verification: does this token's claimed role/meaning in the
    response actually match its real role in the source text? Polls
    multiple independent models and requires a MAJORITY to agree on
    "DIFFERENT" before flagging a mismatch — not a single model's opinion.

    Returns True (majority says matches, or no clear majority — err
    toward not flagging), False (majority says mismatch — real bug
    caught, cross-confirmed by multiple independent models), or None
    (no model could be reached at all).
    """
    response_snippet = _extract_window(response, token)
    context_snippet = _extract_window(context, token)
    if not response_snippet or not context_snippet:
        return None

    prompt = (
        f'Source text (real, from the codebase): "{context_snippet}"\n'
        f'Claim (from a proposed code change): "{response_snippet}"\n\n'
        f'Both mention the number {token}. Does the CLAIM use {token} with the '
        f'SAME meaning/role it actually has in the SOURCE text (e.g. the same '
        f'variable, setting, or concept) — or does the claim attach {token} to '
        f'a different, unrelated meaning than what it represents in the source? '
        f'Answer with exactly one word: SAME or DIFFERENT.'
    )

    votes = []
    for m in models:
        try:
            raw = _call_ollama(prompt, model=m).strip().upper()
            if "DIFFERENT" in raw:
                votes.append(False)
            elif "SAME" in raw:
                votes.append(True)
            # unparseable response from this model — no vote cast, not an error
        except Exception:
            pass   # this model unreachable/errored — skip it, don't block on one failure

    if not votes:
        return None   # no model could be reached at all
    different_votes = votes.count(False)
    same_votes = votes.count(True)
    # Majority rule — ties (e.g. 1-1 with one model unreachable) default
    # to NOT flagging, since this is a supplementary check and a false
    # "mismatch" alarm has its own cost (see paper/observe_ollama_findings.md
    # false-positive history throughout this whole build).
    return different_votes <= same_votes

_ASSIGNMENT_PATTERN = re.compile(
    r"([A-Za-z_][A-Za-z0-9_.]*)\s*(?::\s*\w+\s*)?=\s*([-+]?\d+\.?\d*)"
)
# Prose equivalents — real responses describe a value in many different
# ways ("threshold is set to 100", "fire_rate is currently 0.92", "might
# be around 100", "has a delay of 1 second"), not just code syntax or one
# fixed phrasing. This is an open-ended natural-language coverage problem
# (infinite ways to phrase "the current value is X") — these patterns
# cover the phrasings measured in real responses so far, not an exhaustive
# set. See paper/observe_ollama_findings.md for the specific cases each
# pattern was added to catch.
_PROSE_VALUE_PATTERNS = [
    re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s+is\s+(?:currently\s+|set\s+to\s+|)([-+]?\d+\.?\d*)"),
    re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s+(?:might\s+be|appears?\s+to\s+be|seems?\s+to\s+be)\s+(?:around\s+|about\s+|)([-+]?\d+\.?\d*)"),
    re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s+(?:currently\s+)?has\s+a\s+\w+\s+of\s+([-+]?\d+\.?\d*)"),
]

def _clean_number(val: str) -> str:
    """Strip a trailing bare '.' left over from sentence-ending punctuation
    the regex incidentally captured (e.g. "100." at the end of a sentence)."""
    return val[:-1] if val.endswith(".") and val.count(".") == 1 else val

def _extract_assignment_pairs(text: str) -> dict:
    """
    Deterministic extraction of (identifier, value) pairs from an
    `identifier = number` or `identifier: type = number` shape — matches
    the real patterns seen across this project's languages: GDScript
    (`@export var fire_rate: float = 0.25`, `spawn_timer.wait_time = 1.5`)
    and Python (`threshold=110`, `N_DEFECTS = 40`). No model involved —
    pure regex/structural matching, so it either finds a real match or it
    doesn't, with no hallucination risk in the check itself. Last match
    per identifier wins (in case of duplicates in one text blob).
    """
    pairs = {}
    for var, val in _ASSIGNMENT_PATTERN.findall(text):
        pairs[var] = _clean_number(val)
    for prose_pattern in _PROSE_VALUE_PATTERNS:
        for var, val in prose_pattern.findall(text):
            pairs.setdefault(var, _clean_number(val))
    return pairs

# Phrases that signal the number right after them is a PROPOSED new value,
# not a claim about the current/existing state — measured from real
# response phrasing patterns ("changing X to Y", "consider Y", "such as Y").
_PROPOSAL_LEAD_PHRASES = (
    " to ", " to`", "to →", "->", "consider ", "propose ", "suggest ",
    "such as ", "instead of ", "could be ", "moderate adjustment",
    "default of ", "a default ", "you might change it to",
)

def _extract_current_state_pairs(text: str, window: int = 30) -> dict:
    """
    Same extraction as _extract_assignment_pairs, but excludes any
    identifier=number match whose number is immediately preceded (within
    `window` characters) by a proposal-indicating phrase — those are the
    model's PROPOSED new value, which is supposed to differ from the real
    context by design, not a false claim about current state.

    Returns {var: [values]} — a LIST per variable, not a single value.
    Measured real bug with a single-value dict: "threshold=100. changing
    threshold=80 instead" kept only 80 (last-seen wins on dict overwrite),
    silently discarding the correct current-state claim of 100, because
    "instead" alone wasn't a recognized proposal-phrase trigger. Returning
    all surviving values lets the caller check "does ANY surviving claim
    match reality" instead of trusting whichever value happened to be
    extracted last — robust to gaps in the phrase-trigger list.
    """
    pairs = {}
    for pattern in (_ASSIGNMENT_PATTERN, *_PROSE_VALUE_PATTERNS):
        for m in pattern.finditer(text):
            var, val = m.group(1), _clean_number(m.group(2))
            # Also check inside the match itself for a leading "= " immediately
            # after words like "to"/"->" (covers "changing X to Y" where Y
            # isn't its own identifier=value pair but follows "to").
            lookback = text[max(0, m.start() - window): m.start()].lower()
            if any(p in lookback for p in _PROPOSAL_LEAD_PHRASES):
                continue   # this is a proposed value, not a current-state claim — skip
            pairs.setdefault(var, []).append(val)
    return pairs

@mcp.tool()
@gated("index_status")
def index_status() -> str:
    """
    Report the current OBSERVE index status — how many chunks/files are
    indexed, and which model is being used. Use this to check whether a
    search is likely to find anything before running search_code.
    """
    _ensure_loaded()
    if _loaded["error"]:
        return f"Not ready: {_loaded['error']}"
    n_chunks = len(engine.metadata)
    return f"OBSERVE index ready — {n_chunks:,} chunks indexed, model: {MODEL_PATH}"

GODOT_EXE = str(Path.home() / "Downloads" / "Godot_v4.6.2-stable_win64.exe" / "Godot_v4.6.2-stable_win64_console.exe")

def _find_git_root(path: Path):
    """Walk upward from `path` looking for a .git directory. Returns the
    repo root Path, or None if the file isn't inside any git repo."""
    for parent in [path.parent, *path.parents]:
        if (parent / ".git").exists():
            return parent
    return None

def _git_file_is_dirty(repo_root: Path, path: Path) -> bool:
    """True if this specific file has uncommitted changes RIGHT NOW,
    before this tool touches it. A dirty file means we cannot cleanly
    tell 'the user's own unsaved work' apart from 'this tool's edit' —
    reverting on failure would silently discard whatever the user hadn't
    saved yet, which is worse than doing nothing."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", str(path)],
            cwd=str(repo_root), capture_output=True, text=True, timeout=15
        )
        return bool(result.stdout.strip())
    except Exception:
        return True   # can't determine git state — treat as dirty, refuse by default

def _git_revert_file(repo_root: Path, path: Path) -> bool:
    """Durable revert via git — works even if this tool's own process
    already exited and restarted, unlike an in-memory Python variable."""
    try:
        result = subprocess.run(
            ["git", "checkout", "--", str(path)],
            cwd=str(repo_root), capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0
    except Exception:
        return False

@experimental_tool
@gated("apply_and_verify")
def apply_and_verify(file_path: str, old_text: str, new_text: str) -> str:
    """
    Closes the loop that search_code/query_codebase/propose_change stop
    short of: applies a real edit to a real file, then verifies it by
    ACTUALLY RUNNING the code — not just checking the text changed.

    SAFETY, layered, real fallbacks — no bypass exists for any of them:

    1. Git-dirty refusal, ALWAYS ENFORCED, no override parameter exists:
       if the target file already has uncommitted changes before this
       tool touches it, the edit is unconditionally refused. An in-memory
       revert on failure would go back to that pre-existing dirty state,
       silently mixing "the user's own unsaved work" with "this tool's
       edit" — there's no way to tell them apart afterward, and no
       parameter to disable this check. If you need to build on dirty
       state, commit or stash it yourself first with real git commands —
       that gives you a real, inspectable checkpoint to diff against,
       which a bypass flag would not.
    2. Durable git-based revert: if the file is clean and the edit breaks
       the syntax check, the file is restored via `git checkout --`, not
       an in-memory Python variable — this works even if the process
       crashes or is killed mid-operation, unlike a purely in-memory
       backup.
    3. Non-git files: if the file isn't inside a git repository at all,
       a literal `.bak` backup file is written alongside it before
       editing, so a durable recovery copy exists on disk regardless of
       whether this process survives to finish the operation.
    4. Ambiguous-match refusal: old_text must appear exactly once, or
       the edit is refused rather than guessing which occurrence to hit.
    5. Syntax verification: the edited file must still parse/load
       (py_compile for .py, Godot --check-only for .gd) or it's reverted.

    Honest scope, still true after all of the above: this verifies the
    file still parses/loads and (for git-tracked files) that a real,
    durable undo path exists — it does NOT confirm the new behavior is
    what you intended. That still needs a real test, same as the turret
    fire_rate / bulk-discount TDD examples built by hand earlier. This
    tool prevents "silently broken or lost work," not "wrong but valid
    code" — those need a genuine behavioral test, not a safety net.

    Args:
        file_path: Absolute path to the real file to edit.
        old_text: The exact existing text to replace — must appear
            exactly once in the file, or the edit is rejected.
        new_text: The replacement text.

    Returns:
        A report of exactly what safety checks ran, what happened, and
        (on failure) exactly how the file was restored to safety.
    """
    path = Path(file_path)
    if not path.exists():
        return f"FAILED — file does not exist: {file_path}"

    repo_root = _find_git_root(path)
    backup_path = None

    if repo_root is not None:
        if _git_file_is_dirty(repo_root, path):
            return (
                f"REFUSED — {file_path} already has uncommitted changes before this edit. "
                f"Reverting on failure would silently mix your existing unsaved work with "
                f"this tool's edit, with no way to separate them afterward. There is no "
                f"override for this check. Commit or stash your current changes first — "
                f"that gives you a real, inspectable checkpoint — then retry. "
                f"No edit was applied."
            )
    else:
        # Not a git repo at all — durable on-disk backup as the fallback,
        # since there's no `git checkout` safety net available here.
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    original = path.read_text(encoding="utf-8")
    count = original.count(old_text)
    if count == 0:
        if backup_path:
            backup_path.unlink(missing_ok=True)
        return f"FAILED — old_text not found in {file_path}. No edit applied."
    if count > 1:
        if backup_path:
            backup_path.unlink(missing_ok=True)
        return (f"FAILED — old_text appears {count} times in {file_path}, ambiguous which "
                f"one to replace. No edit applied. Include more surrounding context in old_text "
                f"to make it unique.")

    edited = original.replace(old_text, new_text)
    path.write_text(edited, encoding="utf-8")

    # Verify the file still parses/loads — catches a syntax-breaking edit
    ext = path.suffix.lower()
    if ext == ".py":
        result = subprocess.run([sys.executable, "-m", "py_compile", str(path)],
                                capture_output=True, text=True, timeout=30)
        ok = result.returncode == 0
        detail = result.stderr.strip() if not ok else "compiles cleanly"
    elif ext == ".gd":
        if not Path(GODOT_EXE).exists():
            ok = False
            detail = f"could not verify: Godot not found at {GODOT_EXE}"
        else:
            result = subprocess.run(
                [GODOT_EXE, "--headless", "--check-only", "--script", str(path)],
                capture_output=True, text=True, timeout=60, cwd=str(path.parent)
            )
            ok = result.returncode == 0 and "error" not in result.stderr.lower()
            detail = "loads cleanly in Godot" if ok else (result.stderr.strip() or result.stdout.strip())
    else:
        ok = True
        detail = f"no syntax checker for {ext} files — edit applied, unverified"

    if not ok:
        # Durable revert: git checkout for tracked files (works even across
        # process restarts), .bak restore for untracked files.
        if repo_root is not None:
            reverted = _git_revert_file(repo_root, path)
            method = "git checkout --" if reverted else "git revert FAILED — manual recovery needed"
        elif backup_path is not None:
            path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
            backup_path.unlink(missing_ok=True)
            method = ".bak file restore"
        else:
            path.write_text(original, encoding="utf-8")
            method = "in-memory restore (no durable backup existed)"
        return (f"REVERTED — edit broke the file. {detail}\n"
                f"Restored via: {method}\n"
                f"The file has been restored to its original state. No changes were kept.")

    if backup_path:
        backup_path.unlink(missing_ok=True)   # success — clean up the temp backup

    revert_hint = f"`git checkout -- {file_path}`" if repo_root else "no durable undo available (not a git repo)"
    return (f"APPLIED AND VERIFIED — {file_path} edited successfully, replacing:\n"
            f'  "{old_text}"\nwith:\n  "{new_text}"\n'
            f"Syntax check: {detail}\n"
            f"To undo this specific edit later: {revert_hint}\n"
            f"NOTE: this confirms the file still parses/loads — it does NOT confirm the new "
            f"behavior is what you intended. Write a real test for that if this change matters.")

ENTANGLEMENT_DB_PATH = str(Path(__file__).resolve().parent / "code_entanglement_db.json")

def _load_entanglement_db():
    p = Path(ENTANGLEMENT_DB_PATH)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

@experimental_tool
@gated("list_indexed_projects")
def list_indexed_projects() -> str:
    """
    List every distinct project OBSERVE's index has been mapped to.

    REAL BUG THIS FIXES (2026-09-14): this tool used to read ONLY the
    precomputed entanglement database (code_entanglement_db.json, written
    by trit_entanglement.py) and report just that. That database is built
    by many local-model (Ollama) summarization calls and takes tens of
    minutes, so it is NOT rebuilt automatically after
    observe_incremental_index.py + chunk_provenance.py --build add new
    projects to the live index -- it silently goes stale. Concretely: a
    real matomo-security-audit sweep found this tool reporting that
    NEITHER matomo nor zabbix were indexed, when both genuinely were
    (verified directly against chunk_provenance.db) -- a confident wrong
    answer with no warning that anything was stale.

    Fix: project names + chunk counts now always come from the LIVE,
    currently-loaded embedding index via group_chunks_by_project() (the
    same cheap, in-memory grouping chunk_provenance.py itself uses to
    assign each chunk's project) -- this can never be stale relative to
    what OBSERVE can actually search right now, since it IS what OBSERVE
    searches. The entanglement database, if present, is merged in
    read-only for its one piece of value the live index can't provide (an
    LLM-generated one-line summary per project) -- any project the live
    index has that the entanglement DB doesn't know about yet is now
    listed with an explicit "[not summarized -- entanglement DB is stale
    or was never built for this project]" flag instead of being silently
    omitted or misreported. There is currently no automatic trigger to
    rebuild the entanglement DB after indexing (its cost, tens of minutes
    of LLM calls, makes that unsafe to run implicitly inside an MCP tool
    call) -- run `python trit_entanglement.py` manually after adding new
    projects if you want fresh summaries; this tool's core answer (what's
    indexed, how much) no longer depends on that step being done.

    Returns:
        One line per project: name, LIVE chunk count, and an entanglement
        summary if one exists and is current -- otherwise a visible
        "[not summarized]" flag rather than a wrong or missing answer.
    """
    _ensure_loaded()
    if _loaded["error"]:
        return f"Error: {_loaded['error']}. Build an index first with trit_app.py or trit_search.py --index."

    live_groups = group_chunks_by_project(engine)
    live_counts = {name: len(idxs) for name, idxs in live_groups.items()}

    db = _load_entanglement_db()
    db_projects = db["projects"] if db else {}

    lines = []
    for name, count in sorted(live_counts.items(), key=lambda kv: -kv[1]):
        noise_tag = "  [non-project config/software]" if name.lower() in NON_PROJECT_HINTS else ""
        info = db_projects.get(name)
        if info is None:
            lines.append(f"{name} ({count} chunks, live){noise_tag}: [not summarized -- entanglement DB is "
                         f"stale or was never built for this project; run `python trit_entanglement.py`]")
            continue
        stale_tag = ""
        if info.get("chunk_count") != count:
            stale_tag = (f"  [STALE: entanglement DB has {info['chunk_count']} chunks for this project, "
                         f"live index has {count} -- summary below may not reflect current content]")
        claim_flag = "  [has unverified claims flagged]" if info.get("unsupported_claims") else ""
        one_line = info["summary"].split(".")[0][:150] if info.get("summary") else "(no summary)"
        lines.append(f"{name} ({count} chunks, live): {one_line}{claim_flag}{stale_tag}{noise_tag}")

    if db is None:
        lines.append("\n(No entanglement database found at all -- every project above is unsummarized. "
                     f"Run `python trit_entanglement.py` to build one at {ENTANGLEMENT_DB_PATH}.)")

    return "\n".join(lines) if lines else "No projects found in the live index."

@experimental_tool
@gated("get_project_summary")
def get_project_summary(project_name: str) -> str:
    """
    Get the full summary and any flagged unsupported claims for one
    project from the precomputed entanglement database. Use
    list_indexed_projects first to see exact project names.

    NOTE on reliability: summaries are generated by a local model sampling
    a handful of chunks, then cross-checked by a 3-model consensus vote
    for claims not actually supported by the sampled content (see
    paper/observe_ollama_findings.md — this exact verification step was
    built after a real hallucination: a summary confidently claimed an
    NPC trust-simulation project was "likely a first-person shooter,"
    with zero support in the actual sampled code). Any claims flagged
    below survived that check and should still be treated as informed
    guesses from a small sample, not verified fact.

    Args:
        project_name: Exact project name as shown by list_indexed_projects.

    Returns:
        The project's summary, chunk count, and any flagged claims.
    """
    db = _load_entanglement_db()
    if db is None:
        return "No entanglement database found. Run `python trit_entanglement.py` first."
    info = db["projects"].get(project_name)
    if info is None:
        available = ", ".join(sorted(db["projects"].keys()))
        return f"Project '{project_name}' not found. Available: {available}"

    out = [f"{project_name} — {info['chunk_count']} chunks", "", info["summary"]]
    if info.get("unsupported_claims"):
        out.append("\nFLAGGED — claims below were checked and found NOT clearly supported by the sampled content:")
        for c in info["unsupported_claims"]:
            out.append(f"  - {c[:300]}")
    return "\n".join(out)

@experimental_tool
@gated("get_entanglement")
def get_entanglement(project_a: str, project_b: str) -> str:
    """
    Get the measured cross-project relationship between two projects from
    the precomputed entanglement database — an entanglement score plus
    the actual evidence (specific matching file/chunk pairs), not just a
    bare number.

    Each piece of evidence includes BOTH a content-similarity score and
    an independent filename-similarity score, with a likely_genuine flag —
    content similarity alone cannot reliably separate genuine relationships
    (e.g. the same file duplicated in two locations) from coincidental
    matches (e.g. a C header and minified JavaScript scoring nearly as
    high purely because both are dense, low-natural-language text). Don't
    trust a high score alone — check whether likely_genuine is true and
    read the actual file paths shown.

    Args:
        project_a: First project name (see list_indexed_projects).
        project_b: Second project name — order doesn't matter, both
            orderings are checked.

    Returns:
        The entanglement score and up to 3 pieces of evidence, each
        labeled with content score, filename similarity, and whether
        it's likely a genuine relationship.
    """
    db = _load_entanglement_db()
    if db is None:
        return "No entanglement database found. Run `python trit_entanglement.py` first."

    match = None
    for pair in db["entanglement"]:
        if {pair["a"], pair["b"]} == {project_a, project_b}:
            match = pair
            break
    if match is None:
        return (f"No precomputed entanglement found between '{project_a}' and '{project_b}'. "
                f"Check exact names with list_indexed_projects — both projects must exist "
                f"and have been included in the same trit_entanglement.py run.")

    lines = [f"{match['a']} <-> {match['b']}: entanglement score = {match['score']:.3f}\n"]
    for e in match["evidence"]:
        tag = "LIKELY GENUINE" if e.get("likely_genuine") else "possibly coincidental"
        lines.append(f"[{tag}] content={e.get('content_score', e.get('score', 0)):.3f} "
                     f"filename_sim={e.get('filename_similarity', 0):.3f}")
        lines.append(f"  {e['a_path']} <-> {e['b_path']}")
        lines.append(f"  A: {e['a_preview'][:120]}")
        lines.append(f"  B: {e['b_preview'][:120]}\n")
    return "\n".join(lines)

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
