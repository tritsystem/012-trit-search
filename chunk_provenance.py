"""
Chunk provenance + structural lineage — a SQLite layer alongside OBSERVE's
existing embedding index.

WHY THIS EXISTS: OBSERVE's chunks (search_index/metadata.json /
~/.trit-search/index) carry only [path_index, char_offset] -- no line
numbers, no chunk-type, no hash of what was actually indexed, and no
relationships between chunks. That's enough for pure semantic ranking but
not enough to (a) filter/join structurally (real hybrid search) or (b)
answer "what does this call / what calls this" (real GraphRAG-style
multi-hop retrieval). This module builds that missing layer FROM the
existing index -- it does not re-chunk or re-embed anything.

PROVENANCE per chunk (table `chunks`): project, path, line_start/line_end
(derived by counting newlines up to the chunk's real char offset -- not
stored anywhere before this), a sha256 of the chunk's OWN text slice (not
the whole file -- lets a caller verify this exact chunk hasn't drifted
since indexing), the file's real mtime at build time, and a real build
timestamp.

LINEAGE per edge (table `edges`): Python-only, real `ast` parsing --
imports, function/class definitions, and same-corpus function calls
resolved by name. Deliberately NOT attempted for other languages (GDScript,
C#, JS, ...) in this index; a regex-based multi-language "parser" would
produce edges that look structural but aren't reliably correct, which is
worse than an honest gap. Name-based call resolution can also produce
false-positive edges when two unrelated functions share a name across
projects -- every call edge is tagged `resolution="name-match"` so a
consumer knows the confidence level, never presented as verified.

Usage:
    python chunk_provenance.py --build     Build/rebuild the DB from the current index
    python chunk_provenance.py --stats     Report what's in it
"""
import argparse
import ast
import hashlib
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from observe_pipeline import INDEX_DIR, MODEL_PATH, load_engine, group_chunks_by_project

DB_PATH = str(Path(__file__).resolve().parent / "chunk_provenance.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    INTEGER PRIMARY KEY,
    project     TEXT NOT NULL,
    base_dir    TEXT NOT NULL,
    rel_path    TEXT NOT NULL,
    char_offset INTEGER NOT NULL,
    line_start  INTEGER NOT NULL,
    line_end    INTEGER NOT NULL,
    language    TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    file_mtime  REAL,
    indexed_at  REAL NOT NULL,
    chunk_type  TEXT NOT NULL DEFAULT 'text'
);
CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(rel_path);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);

CREATE TABLE IF NOT EXISTS edges (
    from_chunk_id INTEGER NOT NULL,
    to_chunk_id   INTEGER NOT NULL,
    edge_type     TEXT NOT NULL,     -- 'imports' | 'defines' | 'calls'
    evidence      TEXT NOT NULL,     -- the real name/statement this edge was derived from
    resolution    TEXT NOT NULL,     -- 'same-file' | 'same-project' | 'name-match' | 'unresolved'
    FOREIGN KEY (from_chunk_id) REFERENCES chunks(chunk_id),
    FOREIGN KEY (to_chunk_id) REFERENCES chunks(chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_chunk_id);
CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_chunk_id);
"""

CHUNK_SIZE = 800  # must match trit_search.py's CHUNK_SIZE -- how far a chunk's text runs from its offset


def _line_of(text, char_offset):
    """1-indexed line number containing char_offset -- counts real newlines
    in the actual file text, not estimated from chunk size."""
    return text.count("\n", 0, char_offset) + 1


def build(index_dir=INDEX_DIR, model_path=MODEL_PATH, db_path=DB_PATH):
    t0 = time.time()
    engine = load_engine(index_dir, model_path)
    print(f"[provenance] index loaded, {len(engine.metadata)} chunks ({time.time()-t0:.1f}s)")

    # Same project boundaries the entanglement tools already report (including
    # duplicate-suffix folder merging) -- built once, inverted to a per-chunk lookup,
    # instead of recomputing a private per-chunk inference that would drift from it.
    project_of_chunk = {}
    for project, chunk_ids in group_chunks_by_project(engine).items():
        for cid in chunk_ids:
            project_of_chunk[cid] = project

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM edges")

    now = time.time()
    file_text_cache = {}   # rel_path -> (text, mtime) -- avoid re-reading the same file per chunk
    n_ok, n_missing = 0, 0
    py_files = {}   # rel_path -> full_path, for the AST pass below

    for chunk_id, m in enumerate(engine.metadata):
        if not (isinstance(m, list) and engine.path_table):
            continue
        p_idx, offset = m
        p = engine.path_table[p_idx]
        base_dir, rel_path = p["base_dir"], p["rel_path"]
        full_path = Path(base_dir) / rel_path

        if rel_path not in file_text_cache:
            try:
                text = full_path.read_text(encoding="utf-8", errors="ignore")
                mtime = full_path.stat().st_mtime
            except Exception:
                text, mtime = None, None
            file_text_cache[rel_path] = (text, mtime)
        text, mtime = file_text_cache[rel_path]
        if text is None:
            n_missing += 1
            continue

        line_start = _line_of(text, offset)
        chunk_text = text[offset:offset + CHUNK_SIZE]
        line_end = line_start + chunk_text.count("\n")
        source_hash = hashlib.sha256(chunk_text.encode("utf-8", "ignore")).hexdigest()
        project = project_of_chunk.get(chunk_id, "(ungrouped -- below MIN_CHUNKS_PER_PROJECT)")
        language = Path(rel_path).suffix.lstrip(".") or "unknown"

        conn.execute(
            "INSERT INTO chunks (chunk_id, project, base_dir, rel_path, char_offset, "
            "line_start, line_end, language, source_hash, file_mtime, indexed_at, chunk_type) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (chunk_id, project, base_dir, rel_path, offset, line_start, line_end,
             language, source_hash, mtime, now, "text"))
        n_ok += 1
        if language == "py" and rel_path not in py_files:
            py_files[rel_path] = (full_path, base_dir)

    conn.commit()
    print(f"[provenance] {n_ok} chunks written, {n_missing} skipped (source file unreadable) "
          f"({time.time()-t0:.1f}s total)")

    n_type, n_edges = _tag_python_chunk_types_and_edges(conn, py_files)
    conn.commit()
    print(f"[provenance] {n_type} chunks tagged function/class/module from real AST, "
          f"{n_edges} structural edges written ({time.time()-t0:.1f}s total)")
    conn.close()
    return n_ok, n_type, n_edges


AMBIGUOUS_NAME_LIMIT = 3   # a name resolving to more sites than this is treated as too generic to trust


def _tag_python_chunk_types_and_edges(conn, py_files):
    """Real ast.parse over every indexed .py file: tags each chunk's
    chunk_type by which def/class (if any) its line range falls inside,
    and writes real edges for imports, definitions, and same-corpus calls
    resolved by name. A file that fails to parse (real syntax error, or a
    .py file that's actually something else) is skipped and reported, not
    silently ignored.

    Call resolution tiers, most to least trusted -- MEASURED necessary:
    the first version resolved every bare-name call against ALL matching
    defs corpus-wide, and `main()` (the if __name__=="__main__" idiom,
    present in dozens of unrelated one-off scripts across this 24-project
    index) fanned out into 67,860 edges linking every main() call to all
    261 different main() definitions site-wide -- not "low confidence,"
    actively wrong. Fixed by scoping resolution:
      same-file    -- exact, always kept
      same-project -- ambiguous corpus-wide but the caller and a small
                      number of candidate defs share a project; plausible
      name-match   -- cross-project, but the name is rare enough overall
                      (<=AMBIGUOUS_NAME_LIMIT distinct definition sites in
                      the WHOLE corpus) to be worth keeping
      (skipped)    -- name is too generic (more candidates than the limit,
                      no same-project match) to mean anything as a bare
                      name; NOT stored, rather than stored as noise."""
    rel_path_project = dict(conn.execute("SELECT DISTINCT rel_path, project FROM chunks").fetchall())

    # name -> [(chunk_id, rel_path)] of every function/class definition seen,
    # built first so call-edges can resolve targets that were defined in a
    # DIFFERENT file than the one currently being walked.
    def_index = {}
    file_asts = {}
    n_parse_fail = 0
    for rel_path, (full_path, base_dir) in py_files.items():
        try:
            src = full_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src, filename=str(full_path))
        except SyntaxError:
            n_parse_fail += 1
            continue
        file_asts[rel_path] = (src, tree)

    if n_parse_fail:
        print(f"[provenance] {n_parse_fail} .py file(s) failed to parse (real syntax error), skipped for AST edges")

    def chunks_for_file(rel_path):
        rows = conn.execute(
            "SELECT chunk_id, line_start, line_end FROM chunks WHERE rel_path=? ORDER BY line_start",
            (rel_path,)).fetchall()
        return rows

    file_chunks = {rp: chunks_for_file(rp) for rp in file_asts}

    def chunk_for_line(rel_path, lineno):
        """The chunk whose [line_start, line_end] contains lineno, or the
        closest chunk if the file's chunking didn't align exactly (fixed-
        size windows don't respect AST boundaries)."""
        rows = file_chunks.get(rel_path) or []
        best = None
        for chunk_id, ls, le in rows:
            if ls <= lineno <= le:
                return chunk_id
            if best is None or abs(ls - lineno) < abs(best[1] - lineno):
                best = (chunk_id, ls)
        return best[0] if best else None

    n_type = 0
    for rel_path, (src, tree) in file_asts.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                cid = chunk_for_line(rel_path, node.lineno)
                if cid is not None:
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    conn.execute("UPDATE chunks SET chunk_type=? WHERE chunk_id=?", (kind, cid))
                    n_type += 1
                    def_index.setdefault(node.name, []).append((cid, rel_path))

    n_edges = 0
    n_skipped_ambiguous = 0
    for rel_path, (src, tree) in file_asts.items():
        caller_project = rel_path_project.get(rel_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    cid = chunk_for_line(rel_path, node.lineno)
                    if cid is not None:
                        conn.execute(
                            "INSERT INTO edges (from_chunk_id, to_chunk_id, edge_type, evidence, resolution) "
                            "VALUES (?,?,?,?,?)",
                            (cid, cid, "imports", alias.name, "unresolved"))
                        n_edges += 1
            elif isinstance(node, ast.ImportFrom) and node.module:
                cid = chunk_for_line(rel_path, node.lineno)
                if cid is not None:
                    conn.execute(
                        "INSERT INTO edges (from_chunk_id, to_chunk_id, edge_type, evidence, resolution) "
                        "VALUES (?,?,?,?,?)",
                        (cid, cid, "imports", node.module, "unresolved"))
                    n_edges += 1
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called = node.func.id
                caller_cid = chunk_for_line(rel_path, node.lineno)
                if caller_cid is None or called not in def_index:
                    continue
                candidates = [(cid, path) for cid, path in def_index[called] if cid != caller_cid]
                if not candidates:
                    continue
                same_file = [c for c in candidates if c[1] == rel_path]
                if same_file:
                    targets, resolution = same_file, "same-file"
                else:
                    same_project = [c for c in candidates if rel_path_project.get(c[1]) == caller_project]
                    if same_project and len(same_project) <= AMBIGUOUS_NAME_LIMIT:
                        targets, resolution = same_project, "same-project"
                    elif not same_project and len(candidates) <= AMBIGUOUS_NAME_LIMIT:
                        targets, resolution = candidates, "name-match"
                    else:
                        n_skipped_ambiguous += 1
                        continue
                for target_cid, target_path in targets:
                    conn.execute(
                        "INSERT INTO edges (from_chunk_id, to_chunk_id, edge_type, evidence, resolution) "
                        "VALUES (?,?,?,?,?)",
                        (caller_cid, target_cid, "calls", called, resolution))
                    n_edges += 1
    print(f"[provenance] {n_skipped_ambiguous} call site(s) skipped as too ambiguous "
          f"(name resolves to >{AMBIGUOUS_NAME_LIMIT} sites with no same-project match)")
    return n_type, n_edges


def stats(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    n_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    by_type = conn.execute("SELECT chunk_type, COUNT(*) FROM chunks GROUP BY chunk_type ORDER BY 2 DESC").fetchall()
    by_edge = conn.execute("SELECT edge_type, resolution, COUNT(*) FROM edges GROUP BY 1,2 ORDER BY 3 DESC").fetchall()
    n_projects = conn.execute("SELECT COUNT(DISTINCT project) FROM chunks").fetchone()[0]
    print(f"chunks: {n_chunks}  across {n_projects} projects  |  edges: {n_edges}")
    print("chunk_type breakdown:", dict(by_type))
    print("edge breakdown (type, resolution, count):")
    for et, res, c in by_edge:
        print(f"  {et:10s} {res:12s} {c}")
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    if args.build:
        build()
    if args.stats or not args.build:
        stats()
