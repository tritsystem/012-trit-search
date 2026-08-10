"""
Free local dispatch: keyword classify -> real hybrid_search() lookup -> local
Ollama explanation. Zero Claude API tokens -- the whole pipeline runs on this
machine's own hardware (Ollama), the same real pattern already used by
trit_entanglement.py's _call_ollama() and trit_mcp_server.py's
propose_change.

NOT a replacement for the Claude Code specialist-router workflow -- this has
no real agentic tool-use loop (no file edits, no multi-step exploration,
no ability to decide "I need one more lookup" mid-answer). It's specifically
for "explain these already-fetched real search results in plain English"
questions -- one deterministic round: classify, search, explain. For work
needing actual code changes or open-ended multi-step reasoning across a
codebase, use the Claude Code specialists (mini or full) instead.

Usage:
    python local_dispatch.py "which repo has the STDP learner and how does it relate to methodlm's REFUTE tool"
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from observe_pipeline import load_engine
from hybrid_search import hybrid_search, format_results

OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Real known project roots -- NOT derived from chunk_provenance.db's
# base_dir column, which for repos scanned as part of a broad root (whole
# Documents folder, or C:/ itself) stores that broad root, not the repo's
# own directory. Hand-kept, matching the same real paths used throughout
# this whole session.
PROJECT_ROOTS = {
    "Spikeling": Path.home() / "OneDrive" / "Documents" / "Spikeling",
    "012-ternary": Path.home() / "OneDrive" / "Documents" / "012-ternary",
    "methodlm": Path.home() / "OneDrive" / "Documents" / "methodlm",
    "server-guard": Path.home() / "OneDrive" / "Documents" / "server-guard",
    "tribe": Path.home() / "OneDrive" / "Documents" / "tribe",
    "mcp-gateway": Path.home() / "OneDrive" / "Documents" / "mcp-gateway",
    "topological-phononics": Path.home() / "OneDrive" / "Documents" / "topological-phononics",
}
GREP_EXTS = {".py", ".gd", ".js", ".ts", ".c", ".h", ".md"}
GREP_SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv", "search_index", "models"}


def extract_identifiers(question):
    """Real-identifier heuristic: ALL-CAPS acronyms (STDP, REFUTE) and
    CamelCase tokens (STDPLearner) are usually exact code names worth
    grepping verbatim -- semantic search is measurably weak at surfacing
    the exact class/function that DEFINES a term even when it's strong at
    finding prose that DISCUSSES it (see the vault Lesson this script's
    docstring references: OBSERVE's own README already says "prefer Grep
    when you already know the exact identifier")."""
    caps = re.findall(r"\b[A-Z]{3,}\b", question)
    camel = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]*)+\b", question)
    return sorted(set(caps) | set(camel))


def grep_project(project, terms, max_hits=8):
    """Plain local filesystem grep, zero cost, no Ollama needed for this
    part either -- exact substring match, case-sensitive (identifiers are
    usually case-meaningful).

    Real bug found and fixed while testing: an early cap applied during the
    directory walk in file-visit order meant a repo with many files that
    merely MENTION a term (Spikeling has dozens of `pyspike_*.py` files
    referencing "STDP" in comments/filenames) filled the cap before the walk
    ever reached the file that actually DEFINES it (`core/runtime/runtime.py`,
    several directories deep). Fixed by scanning the WHOLE tree first, then
    ranking definition sites (`class TERM` / `def TERM`) ahead of plain
    mentions -- a mention-heavy repo no longer buries the actual definition."""
    root = PROJECT_ROOTS.get(project)
    if not root or not root.exists() or not terms:
        return []
    mention_pattern = re.compile("|".join(re.escape(t) for t in terms))
    def_pattern = re.compile(r"\b(?:class|def)\s+\w*(?:" + "|".join(re.escape(t) for t in terms) + r")\w*\b")

    definitions, mentions = [], []
    for path in root.rglob("*"):
        if path.is_dir() or path.suffix.lower() not in GREP_EXTS:
            continue
        if any(part in GREP_SKIP for part in path.parts):
            continue
        try:
            for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if mention_pattern.search(line):
                    entry = f"{path.relative_to(root)}:{lineno}: {line.strip()[:160]}"
                    (definitions if def_pattern.search(line) else mentions).append(entry)
        except Exception:
            continue
    return (definitions + mentions)[:max_hits]

# Same domain/keyword mapping as .claude/workflows/specialist-router.js --
# kept in sync BY HAND (different runtime -- JS Workflow script vs. this
# plain Python script -- can't literally share code between them). If you
# edit the specialist list in one, edit the other.
SPECIALISTS = [
    {"project": "Spikeling", "keywords": ["spikeling", "stdp", ".spk", "neuromorphic", "lif neuron",
                                           "spiking network", "jet-engine", "jet engine", "sensor adapter",
                                           "synaptic", "spike_pipeline"]},
    {"project": "012-ternary", "keywords": ["observe", "012-ternary", "ternary compress", "chunk provenance",
                                             "hybrid_search", "hybrid search", "semantic search", "trit_app",
                                             "trit_mcp", "embedding index", "incremental index"]},
    {"project": "methodlm", "keywords": ["methodlm", "refute", "adjust", "preregister", "cinelli-hazlett",
                                          "dowhy", "robustness value", "causal reasoning", "interact tool",
                                          "strat tool"]},
    {"project": "server-guard", "keywords": ["server-guard", "packet capture", "osint", "blocklist", "firehol",
                                              "telemetry collector", "guard supervisor"]},
    {"project": "tribe", "keywords": ["tribe", "godot", "npc", "gdscript", "terrain_gen", "kuramoto",
                                       "trade_envoy", "tribemanager"]},
    {"project": "mcp-gateway", "keywords": ["mcp-gateway", "audit_log", "audit log", "rate limit",
                                             "hash-chained", "tamper-evident", "gated(", "rate_limiter"]},
    {"project": "topological-phononics", "keywords": ["topological-phononics", "ssh reservoir", "meta_ledger",
                                                        "meta-ledger", "quasicrystal", "defect-tolerance",
                                                        "chiral symmetry", "fibonacci"]},
]


def classify(question):
    """Same free substring-match approach as the JS router -- zero cost,
    deterministic. Returns real project names (not agent names -- there's
    no subagent here)."""
    q = question.lower()
    return [s["project"] for s in SPECIALISTS if any(kw in q for kw in s["keywords"])]


def call_ollama(prompt, model=OLLAMA_MODEL, timeout=120):
    """Same real HTTP call shape as trit_entanglement.py's _call_ollama()."""
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"].strip()


def refine_query(question, project):
    """A raw conversational question is a genuinely bad semantic-search
    query -- measured directly: passing the full question text for "which
    repo has the STDP learner... how does it relate to REFUTE" matched a
    DIFFERENT Spikeling research doc (about forward-forward learning, not
    STDP) and found ZERO results in methodlm, even though both are real,
    findable content -- confirmed separately via short, focused queries.
    This local call rewrites the question into a short, code-search-shaped
    query per project before searching. Still $0 API cost -- one more local
    Ollama round-trip, not a Claude call."""
    prompt = (
        f"Question: \"{question}\"\n\n"
        f"Write ONE short, focused search query (3-8 words, like you'd type into a code search engine, "
        f"not a full sentence) that would find the most relevant code in the \"{project}\" codebase for "
        f"answering this question. Reply with ONLY the query, nothing else."
    )
    return call_ollama(prompt).strip().strip('"')


def dispatch(question, k=5, verbose=True):
    t0 = time.time()
    projects = classify(question)
    if verbose:
        print(f"[classify] matched: {projects or '(none)'}")
    if not projects:
        return {"question": question, "projects": [], "answer":
                 "No project matched by keyword -- try naming the repo/topic more explicitly, "
                 "or fall back to the Claude Code specialist router for ambiguous phrasing."}

    engine = load_engine()
    if verbose:
        print(f"[engine] loaded ({time.time()-t0:.1f}s)")

    identifiers = extract_identifiers(question)
    if verbose:
        print(f"[identifiers] extracted for grep: {identifiers or '(none)'}")

    # Grep/definition sections go FIRST -- measured directly: even with the
    # correct grep hit (core/runtime/runtime.py:201, the real STDPLearner
    # class) present, qwen2.5-coder:7b kept citing the semantic-search
    # result instead when semantic search was listed first. Reordering so
    # the more-authoritative exact-identifier data has primacy, plus an
    # explicit priority instruction below, rather than trusting a small
    # local model to weigh two sections correctly on its own.
    grep_sections, semantic_sections = [], []
    for project in projects:
        grep_hits = grep_project(project, identifiers)
        if verbose:
            print(f"[grep] {project}: {len(grep_hits)} exact-identifier hit(s)")
        if grep_hits:
            grep_sections.append(f"=== Real exact-identifier grep hits ({', '.join(identifiers)}) in "
                                  f"project={project} -- AUTHORITATIVE for where something is actually "
                                  f"defined ===\n" + "\n".join(grep_hits))

        refined = refine_query(question, project)
        if verbose:
            print(f"[refine] {project}: \"{refined}\"")
        res = hybrid_search(engine, refined, k=k, project=project)
        if res.get("available"):
            semantic_sections.append(f"=== Real semantic search results for query \"{refined}\" scoped to "
                                      f"project={project} -- background/context only, LOWER priority than "
                                      f"the grep hits above if they conflict ===\n{format_results(res)}")
        else:
            semantic_sections.append(f"=== project={project}: {res.get('reason')} ===")
    combined = "\n\n".join(grep_sections + semantic_sections)

    prompt = (
        f"A user asked: \"{question}\"\n\n"
        f"Here are REAL results from the actual codebase (not invented -- ground your answer in these, "
        f"do not add facts they don't support). If the grep section and the semantic-search section name "
        f"DIFFERENT files for the same thing, TRUST THE GREP SECTION -- it found the actual definition; "
        f"the semantic-search section can surface related-but-wrong files.\n\n{combined}\n\n"
        f"Answer the question directly and honestly using ONLY what's shown above. If the results don't "
        f"show a connection or a clear answer, say so plainly rather than guessing or padding. Be concise."
    )
    answer = call_ollama(prompt)
    elapsed = time.time() - t0
    if verbose:
        print(f"[ollama] answered ({elapsed:.1f}s total, model={OLLAMA_MODEL}, $0 API cost)")
    return {"question": question, "projects": projects, "answer": answer, "elapsed_s": round(elapsed, 1)}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or (
        "which repo has the STDP learner implementation, and how does it relate to methodlm's REFUTE tool?"
    )
    result = dispatch(q)
    print("\n=== ANSWER ===")
    print(result["answer"])
    print(f"\n[{result.get('elapsed_s', '?')}s total, projects={result.get('projects')}]")
