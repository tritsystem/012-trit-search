"""
Incremental append to OBSERVE's existing embedding index.

trit_app.py's build_index() (and trit_search.py --index) only know how to
do a FULL rebuild: scan everything under scan_dirs, encode, overwrite the
index. That's correct for "reindex everything" but wrong for "add these
specific new repos" -- the currently configured scan_dirs
(~/.trit-search/config.json) include the whole Documents folder and C:/
itself, so a full rebuild at that scope is a many-hour, near-whole-drive
scan. This script instead does exactly one thing: scan a SMALL, explicit
list of new directories, encode ONLY their chunks, and APPEND to the
existing index -- every existing chunk keeps its exact chunk_id (position
in the embeddings array), so chunk_provenance.py's already-built rows for
the old 90,099 chunks stay valid; only a rebuild of the provenance DB
(cheap -- it just re-derives from the index, doesn't re-embed) is needed
afterward to add rows for the new chunks.

Correctness of the append, by construction:
  - Old packed-ternary rows are concatenated with new ones (axis=0) --
    NEVER unpacked/repacked/modified, so nothing about the old 90,099
    vectors can be perturbed by this script.
  - Old path_table entries keep their exact indices; new paths get NEW
    indices appended after them.
  - Any (base_dir, rel_path) already in the existing path_table is
    skipped during the new scan, so re-running this against a directory
    that was already (partially) indexed does not create duplicate
    chunks for the same file.

Usage:
    python observe_incremental_index.py DIR [DIR ...]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trit_app import pack_ternary
from observe_pipeline import INDEX_DIR, MODEL_PATH

EXTS = {".py", ".gd", ".js", ".ts", ".cs", ".rs", ".go",
        ".c", ".cpp", ".h", ".java", ".lua", ".rb", ".php",
        ".swift", ".kt", ".dart", ".zig", ".md", ".sh", ".ps1"}
SKIP = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", "target", "models", "search_index",
    "ai_files", "addons",
    "AppData", "Temp", "Windows", "Program Files",
    "Program Files (x86)", "ProgramData",
    "$Recycle.Bin", "System Volume Information",
    "msys64", "mingw64", "mingw32", "Anaconda3", "miniconda3",
    "site-packages", ".cargo", ".rustup", ".nuget", ".gradle", ".m2",
    "Ableton", "Steam", "steamapps", "Epic Games", "Adobe", "Spotify",
}
SKIP_PATTERNS = ("_files", "_assets")
CHUNK_SIZE, CHUNK_STEP = 800, 700


def scan_new_chunks(dirs, already_indexed):
    """Same chunking rules as trit_app.py's build_index(), restricted to
    `dirs`, skipping any (base_dir, rel_path) already in the existing
    index."""
    chunks, files = [], 0
    for base in dirs:
        # Forward-slash form, matching the existing index's convention (its
        # base_dir strings come from scan_dirs in ~/.trit-search/config.json,
        # already forward-slashed) -- str(Path(...).resolve()) on Windows
        # gives native backslashes, which _infer_project_name's
        # CONTAINER_PREFIXES matching (forward-slash prefixes) silently fails
        # to strip, so new chunks got grouped under a garbled full-path
        # "project" instead of the real repo name. Caught via a real stats
        # check after the first live run, fixed via a one-time path_table
        # patch (_fix_basedir_slashes.py) plus this fix so it can't recur.
        base = str(Path(base).resolve()).replace("\\", "/")
        for root, subdirs, fnames in os.walk(base):
            subdirs[:] = [d for d in subdirs if d not in SKIP and not d.startswith(".")
                          and not any(p in d for p in SKIP_PATTERNS)]
            for fname in fnames:
                if Path(fname).suffix.lower() not in EXTS:
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, base)
                if (base, rel) in already_indexed:
                    continue
                try:
                    text = open(fpath, encoding="utf-8", errors="ignore").read()
                    if len(text.strip()) < 100:
                        continue
                    for i in range(0, len(text), CHUNK_STEP):
                        chunk = text[i:i + CHUNK_SIZE]
                        if len(chunk.strip()) > 50:
                            chunks.append({
                                "text": f"file:{rel}\n{chunk}",
                                "rel_path": rel, "base_dir": base, "offset": i,
                            })
                    files += 1
                except Exception:
                    pass
    return chunks, files


def append(dirs, index_dir=INDEX_DIR, model_path=MODEL_PATH):
    t0 = time.time()
    meta_path = os.path.join(index_dir, "metadata.json")
    trit_path = os.path.join(index_dir, "vectors_ternary.npy")
    vmeta_path = os.path.join(index_dir, "vectors_meta.json")

    raw = json.load(open(meta_path, encoding="utf-8"))
    old_path_table, old_chunks = raw["paths"], raw["chunks"]
    old_packed = np.load(trit_path)
    dim = json.load(open(vmeta_path)).get("dim", 384)
    n_old = len(old_chunks)
    print(f"[incremental] existing index: {n_old} chunks, {len(old_path_table)} files, "
          f"packed shape {old_packed.shape}")

    already_indexed = {(p["base_dir"], p["rel_path"]) for p in old_path_table}
    chunks, n_files = scan_new_chunks(dirs, already_indexed)
    print(f"[incremental] scanned {dirs}: {len(chunks)} new chunks from {n_files} new files")
    if not chunks:
        print("[incremental] nothing new to add (already indexed, or no matching files found)")
        return 0

    print("[incremental] loading model + encoding...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path)
    texts = [c["text"] for c in chunks]
    bs = 128
    vecs = []
    for i in range(0, len(texts), bs):
        v = model.encode(texts[i:i + bs], normalize_embeddings=True, show_progress_bar=False)
        vecs.append(v)
    vecs = np.vstack(vecs).astype("float32")
    assert vecs.shape[1] == dim, f"embedding dim {vecs.shape[1]} != existing index dim {dim}"

    # SAME quantization approach as build_index() -- 0.7x mean absolute
    # value as the ternary threshold, derived from this batch's own
    # embeddings (a stable property of the embedding model's output
    # distribution, not the specific text, so consistent with the
    # existing index's own per-build threshold without needing to
    # re-derive it jointly).
    t = 0.7 * np.abs(vecs).mean()
    trit_vecs = np.where(vecs > t, 1, np.where(vecs < -t, -1, 0)).astype("int8")
    new_packed = pack_ternary(trit_vecs)
    assert new_packed.shape[1] == old_packed.shape[1], "packed width mismatch -- dim changed?"

    # Append-only merge: old path indices/rows untouched, new ones start
    # right after the old table so every existing chunk_id is unchanged.
    path_idx = {}
    new_path_table = []
    for p in old_path_table:
        path_idx[(p["base_dir"], p["rel_path"])] = len(new_path_table)
        new_path_table.append(p)
    new_rows = []
    for c in chunks:
        key = (c["base_dir"], c["rel_path"])
        if key not in path_idx:
            path_idx[key] = len(new_path_table)
            new_path_table.append({"base_dir": c["base_dir"], "rel_path": c["rel_path"]})
        new_rows.append([path_idx[key], c["offset"]])

    merged_chunks = old_chunks + new_rows
    merged_packed = np.concatenate([old_packed, new_packed], axis=0)
    assert len(merged_chunks) == merged_packed.shape[0]
    assert merged_chunks[:n_old] == old_chunks, "old chunk rows were altered -- append is not safe, aborting write"
    assert np.array_equal(merged_packed[:n_old], old_packed), "old packed vectors were altered -- aborting write"

    json.dump({"paths": new_path_table, "chunks": merged_chunks},
               open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    np.save(trit_path, merged_packed)
    print(f"[incremental] wrote {len(merged_chunks)} total chunks "
          f"({len(merged_chunks) - n_old} new, {n_old} unchanged), "
          f"{len(new_path_table)} total files ({time.time()-t0:.1f}s)")
    return len(merged_chunks) - n_old


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    append(args.dirs)
