"""
Free (Ollama, $0 Claude API tokens) agentic coding loop -- extends
local_dispatch.py's read-only search+explain with a real "propose an edit,
apply it safely, verify it compiles, revert on failure" action.

Reuses trit_mcp_server.py's apply_and_verify() DIRECTLY, not a
reimplementation -- the exact same safety-reviewed logic OBSERVE's own MCP
tool already uses: git-dirty refusal (no override), ambiguous-match
refusal (old_text must appear exactly once), syntax verification
(py_compile for .py), durable git-based revert on failure. Edits made
through this script are ALSO covered by the same mcp-gateway hash-chained
audit log apply_and_verify already writes to -- one audit trail regardless
of whether Claude or a local model made the edit.

Same ReAct "one action per turn" pattern as methodlm.py's investigate()
loop -- proven, real, already working in this codebase -- pointed at a
local model and code tools instead of Claude and causal-analysis tools.
Ollama's native tool-calling response format was tested directly and found
unreliable for qwen2.5-coder:7b (it echoes a JSON tool-call as plain text
in `message.content` instead of populating the structured `tool_calls`
field), so this drives the loop via prompted JSON-in-text + manual
parsing over /api/generate -- the same approach already proven in
local_dispatch.py, not Ollama's native tools param.

NEVER commits or pushes. Edits land on disk, real and git-reversible, for
a human (or a separate explicit step) to review and commit.

Usage:
    python local_coding_agent.py "task description" --project methodlm [--max-turns 8]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from observe_pipeline import load_engine
from hybrid_search import hybrid_search, format_results
from local_dispatch import call_ollama, grep_project, PROJECT_ROOTS, OLLAMA_MODEL
from trit_mcp_server import apply_and_verify

MAX_TURNS_DEFAULT = 8
SYSTEM_TEMPLATE = """You are a coding agent working in the real "{project}" repository at {root}.
Respond with EXACTLY one JSON object per turn and NOTHING else -- no markdown fences, no explanation
outside the JSON. Available actions:

  {{"action": "search", "args": {{"query": "..."}}}}
      Semantic search over the indexed codebase. Good for "where is X handled".

  {{"action": "grep", "args": {{"terms": ["ExactName1", "ExactName2"]}}}}
      Exact-identifier search (case-sensitive substring). Use this when you know a real
      class/function name -- it is MORE reliable than semantic search for finding an exact definition.

  {{"action": "read_file", "args": {{"path": "relative/path.py"}}}}
      Read a real file's full text (relative to the repo root above).

  {{"action": "propose_edit", "args": {{"path": "relative/path.py", "old_text": "...", "new_text": "..."}}}}
      Apply a real, safety-checked edit. old_text must appear EXACTLY ONCE in the file or it is
      refused. The file is verified to still compile after the edit; if it doesn't, the edit is
      automatically reverted via git. You will be told exactly what happened.

  {{"action": "done", "args": {{"summary": "what you did and why"}}}}
      Finish. Only call this after you have actually verified your edit(s) succeeded.

Task: {task}

Rules: search or read BEFORE proposing an edit -- never propose an edit to a file you have not
actually read in this conversation. If propose_edit reports REFUSED or REVERTED, do not just retry
blindly -- read the reason and adjust (e.g. read more context so old_text is unique, or fix the
syntax issue that caused the revert). Keep edits small and targeted."""


def _extract_json(text):
    """The model was told to respond with ONLY a JSON object, but a small
    local model doesn't always comply -- find the first balanced {...}
    block rather than assuming text.strip() is valid JSON on its own."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    return text  # let json.loads raise its own real error if nothing balanced was found


def run_agent(task, project, max_turns=MAX_TURNS_DEFAULT, verbose=True):
    root = PROJECT_ROOTS.get(project)
    if not root or not root.exists():
        return {"task": task, "project": project, "status": "error",
                 "error": f"unknown or missing project root for '{project}'"}

    t0 = time.time()
    engine = load_engine()
    if verbose:
        print(f"[engine] loaded ({time.time()-t0:.1f}s)")

    system = SYSTEM_TEMPLATE.format(project=project, root=root, task=task)
    history = [system]
    log = []

    for turn in range(max_turns):
        raw = call_ollama("\n\n".join(history))
        try:
            action = json.loads(_extract_json(raw))
        except Exception as e:
            history.append(f"[assistant said]: {raw}")
            history.append(f"[system]: That was not valid JSON ({e}). Respond with EXACTLY one JSON object.")
            if verbose:
                print(f"[turn {turn}] unparseable response, nudging")
            continue

        kind = action.get("action")
        args = action.get("args", {}) or {}
        history.append(f"[assistant]: {json.dumps(action)}")

        if kind == "done":
            elapsed = time.time() - t0
            log.append({"turn": turn, "action": "done", "args": args})
            if verbose:
                print(f"[turn {turn}] done: {args.get('summary')}")
            return {"task": task, "project": project, "status": "done",
                     "summary": args.get("summary"), "log": log, "elapsed_s": round(elapsed, 1)}

        elif kind == "search":
            res = hybrid_search(engine, args.get("query", ""), k=5, project=project)
            result_text = format_results(res) if res.get("available") else str(res.get("reason"))

        elif kind == "grep":
            hits = grep_project(project, args.get("terms", []) or [])
            result_text = "\n".join(hits) if hits else "no matches"

        elif kind == "read_file":
            full = root / args.get("path", "")
            try:
                result_text = full.read_text(encoding="utf-8", errors="ignore")[:4000]
            except Exception as e:
                result_text = f"error reading file: {e}"

        elif kind == "propose_edit":
            full = str(root / args.get("path", ""))
            result_text = apply_and_verify(full, args.get("old_text", ""), args.get("new_text", ""))

        else:
            result_text = f"unknown action '{kind}' -- use search, grep, read_file, propose_edit, or done"

        log.append({"turn": turn, "action": kind, "args": args, "result_preview": str(result_text)[:300]})
        if verbose:
            print(f"[turn {turn}] {kind}({json.dumps(args)[:100]}) -> {str(result_text)[:200]}")
        history.append(f"[result]: {str(result_text)[:2500]}")

    elapsed = time.time() - t0
    return {"task": task, "project": project, "status": "max_turns_reached", "log": log,
             "elapsed_s": round(elapsed, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--project", required=True, choices=list(PROJECT_ROOTS.keys()))
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS_DEFAULT)
    args = ap.parse_args()
    result = run_agent(args.task, args.project, max_turns=args.max_turns)
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2, default=str))
