"""
Hybrid search: OBSERVE's existing semantic ranking + real SQL structured
filtering + optional 1-hop graph expansion, over the provenance/lineage
layer built by chunk_provenance.py.

Pure semantic search (trit_app.py's SearchEngine.search) ranks by meaning
alone -- it cannot answer "find code about X, but only in project Y" or
"only function-level chunks, not stray text" without the caller manually
filtering a much larger result set by eye. This module does that filtering
in real SQL against chunk_provenance.db, using the SAME chunk_id the
embedding index uses, so results stay ranked by real cosine similarity and
gain real structured metadata (project, exact line range, chunk type,
source hash) instead of a bare path+preview string.

expand_graph=True adds one real capability semantic search structurally
cannot: for each hit, "what does this call / what calls this" from the
real AST-derived edges -- e.g. searching for a bug symptom finds the
function that logs it, then graph expansion surfaces the function that
actually computes the bad value, even if that function's own text has
no semantic overlap with the query at all.

Requires chunk_provenance.py --build to have been run at least once
(reports itself unavailable, not an error, if the DB is missing).
"""
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk_provenance import DB_PATH


def _db_available(db_path=DB_PATH):
    return os.path.exists(db_path)


def hybrid_search(engine, query, k=10, project=None, chunk_type=None, path_glob=None,
                   expand_graph=False, db_path=DB_PATH):
    """Real semantic candidates (over-fetched k*8, capped at 200) filtered
    in SQL by project/chunk_type/path_glob, re-truncated to k in the
    original semantic rank order (filtering never re-ranks). Each result
    carries real provenance; expand_graph attaches each hit's real 1-hop
    call/import neighbors.

    Returns {"available": False, "reason": ...} if the provenance DB
    hasn't been built yet -- caller should fall back to engine.search()
    directly rather than erroring."""
    if not _db_available(db_path):
        return {"available": False,
                "reason": "chunk_provenance.db not found -- run `python chunk_provenance.py --build` first."}

    over_fetch = min(max(k * 8, 40), 200)
    raw = engine.search(query, k=over_fetch)
    if not raw:
        return {"available": True, "results": []}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    where = ["chunk_id = ?"]
    filtered = []
    for r in raw:
        if "idx" not in r:
            continue  # engine.search() predates the idx field -- skip rather than crash
        row = conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (r["idx"],)).fetchone()
        if row is None:
            continue  # chunk exists in the embedding index but wasn't in the provenance build
            # (e.g. source file unreadable at build time -- see chunk_provenance.py's n_missing)
        if project and row["project"] != project:
            continue
        if chunk_type and row["chunk_type"] != chunk_type:
            continue
        if path_glob and not Path(row["rel_path"]).match(path_glob):
            continue
        entry = dict(r)
        entry.update({
            "project": row["project"], "rel_path": row["rel_path"],
            "line_start": row["line_start"], "line_end": row["line_end"],
            "chunk_type": row["chunk_type"], "source_hash": row["source_hash"],
        })
        filtered.append(entry)
        if len(filtered) >= k:
            break

    if expand_graph:
        for entry in filtered:
            cid = entry["idx"]
            # One JOINed, LIMITed query instead of an edges query + up to 10
            # separate per-target lookups (a real N+1 pattern found while
            # checking this for scalability) -- the LIMIT is in SQL, not a
            # Python-side slice after fetching everything, so a real hub
            # chunk with hundreds of edges doesn't pull them all into memory
            # just to discard most of them.
            rows = conn.execute(
                "SELECT e.edge_type, e.evidence, e.resolution, "
                "c.rel_path, c.line_start, c.project FROM edges e "
                "JOIN chunks c ON c.chunk_id = e.to_chunk_id "
                "WHERE e.from_chunk_id = ? AND e.to_chunk_id != e.from_chunk_id "
                "LIMIT 10", (cid,)).fetchall()
            related = []
            for r in rows:
                related.append({
                    "edge_type": r["edge_type"], "evidence": r["evidence"], "resolution": r["resolution"],
                    "target_path": r["rel_path"], "target_line": r["line_start"],
                    "target_project": r["project"],
                })
            entry["related"] = related

    conn.close()
    return {"available": True, "results": filtered}


def format_results(result):
    """Plain-text rendering matching search_code's existing MCP tool
    output style, extended with real provenance + (if requested) graph
    neighbors."""
    if not result.get("available"):
        return f"HYBRID SEARCH unavailable: {result.get('reason')}"
    results = result["results"]
    if not results:
        return "No results after filtering. Try relaxing project/chunk_type/path_glob, or check they match real values."
    lines = [f"Found {len(results)} results (hybrid: semantic rank + SQL filter)\n"]
    for i, r in enumerate(results, 1):
        loc = f"{r['rel_path']}:{r['line_start']}-{r['line_end']}"
        lines.append(f"{i}. [{r['score']:.3f}] {loc}  ({r['project']}, {r['chunk_type']})")
        lines.append(f"   {r['preview'][:150]}")
        for rel in r.get("related", []):
            lines.append(f"     -> {rel['edge_type']} {rel['evidence']} @ {rel['target_path']}:{rel['target_line']} "
                          f"[{rel['resolution']}]")
    return "\n".join(lines)
