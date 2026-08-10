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
import subprocess
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

  {{"action": "run_python", "args": {{"code": "..."}}}}
      Execute a real, short Python snippet (via `python -c`, cwd = the repo root above, 15s
      timeout) and see its REAL stdout/stderr/exit code. Use this to actually TEST your edit
      behaves correctly -- propose_edit only checks that the file still COMPILES, it does NOT
      confirm your fix does what you think it does. A syntax-valid edit can still be functionally
      wrong (e.g. moving buggy code into a function that gets called unconditionally at import
      time is NOT a fix -- verify by actually importing/running it, don't just assume wrapping
      code in a function changes when it executes).

  {{"action": "done", "args": {{"summary": "what you did and why"}}}}
      Finish. Only call this AFTER you have used run_python to actually verify your edit produces
      the correct real behavior, not just that it compiles. A summary claiming something works
      that you have not actually tested is worse than admitting you couldn't fully verify it.

Task: {task}

Rules: search or read BEFORE proposing an edit -- never propose an edit to a file you have not
actually read in this conversation. If propose_edit reports REFUSED or REVERTED, do not just retry
blindly -- read the reason and adjust (e.g. read more context so old_text is unique, or fix the
syntax issue that caused the revert). Keep edits small and targeted. ALWAYS run_python to verify
behavior before calling done -- compiling is not the same as working."""


def run_python(code, cwd, timeout=15):
    """Real execution, not a syntax check -- this is what propose_edit's
    py_compile step cannot give the agent: actual behavior. Scoped
    deliberately narrower than a full shell (single `python -c` snippet,
    cwd pinned to the repo root, hard timeout) rather than an open shell
    action, since this is specifically for "does my edit actually work",
    not general command execution."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=str(cwd),
            capture_output=True, text=True, timeout=timeout
        )
        out = f"exit_code={result.returncode}\nstdout:\n{result.stdout[-1500:]}\nstderr:\n{result.stderr[-1500:]}"
        return out
    except subprocess.TimeoutExpired:
        return f"TIMED OUT after {timeout}s -- the snippet is likely hanging (e.g. blocking on real I/O)."
    except Exception as e:
        return f"error running snippet: {e}"


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


def load_relevant_lessons(engine, task, k=3):
    """Real "read past lessons before working" grounding, matching the
    Spikeling vault's own stated design ("Everything the agents do is
    logged here, and they read it back before working -- so past work and
    past mistakes shape new work") -- this agent didn't actually do that
    until now. Searched with project="Spikeling" + path_glob="Lessons/*"
    (NOT "*Lessons*" -- Path.match() only checks the filename against a
    single-segment pattern, confirmed directly; see hybrid_search.py's
    docstring for the same gotcha noted where it was found).

    Needs its OWN query refinement, not local_dispatch.py's refine_query()
    -- that one is tuned for CODE search ("what file/function"), and
    measured directly to produce the wrong kind of query here: for this
    exact task it returned "event_scanner.py argv error handling" (still
    surface-level, filename-anchored) which found nothing, while a
    genuinely conceptual query ("wrapping buggy code in a function does
    not defer execution timing") found this file's own new lesson at
    score 7.16. Lessons are general PRINCIPLES, not code -- the query
    needs to ask for the underlying pattern/mistake-class, not a
    code-search-shaped phrase."""
    concept_prompt = (
        f"Task: \"{task}\"\n\n"
        f"Forget code search for a moment. What GENERAL software-engineering concept, pattern, or "
        f"class of mistake might a past project lesson about this task be titled around? Write ONE "
        f"short phrase (5-10 words) describing the underlying PRINCIPLE, not the specific file/variable "
        f"names involved. Reply with ONLY the phrase."
    )
    concept_query = call_ollama(concept_prompt).strip().strip('"')
    res = hybrid_search(engine, concept_query, k=k, project="Spikeling", path_glob="Lessons/*")
    if not res.get("available") or not res.get("results"):
        return []
    seen_paths, lessons = set(), []
    for r in res["results"]:
        if r["rel_path"] in seen_paths:
            continue
        seen_paths.add(r["rel_path"])
        lessons.append(f"[{r['rel_path']}] {r['preview']}")
    return lessons


def run_agent(task, project, max_turns=MAX_TURNS_DEFAULT, verbose=True):
    root = PROJECT_ROOTS.get(project)
    if not root or not root.exists():
        return {"task": task, "project": project, "status": "error",
                 "error": f"unknown or missing project root for '{project}'"}

    t0 = time.time()
    engine = load_engine()
    if verbose:
        print(f"[engine] loaded ({time.time()-t0:.1f}s)")

    lessons = load_relevant_lessons(engine, task)
    if verbose:
        print(f"[lessons] {len(lessons)} relevant vault lesson(s) found")
    lessons_block = ""
    if lessons:
        lessons_block = ("\n\nRELEVANT PAST LESSONS from this project's vault (real prior mistakes -- "
                          "read these before acting, they may directly apply to this task):\n" +
                          "\n".join(f"- {l}" for l in lessons))

    system = SYSTEM_TEMPLATE.format(project=project, root=root, task=task) + lessons_block
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

        elif kind == "run_python":
            result_text = run_python(args.get("code", ""), root)

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
