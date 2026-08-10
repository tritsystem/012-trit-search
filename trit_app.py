"""
012 Trit Search — Desktop App
Standalone GUI with Matrix/cyberpunk themes.
Packages into a single .exe with PyInstaller.

Install deps:
  pip install sentence-transformers faiss-cpu flask torch tkinter pyinstaller

Build exe:
  pyinstaller --onefile --noconsole --name TritSearch trit_app.py

Usage:
  python trit_app.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading, subprocess, sys, os, json, time, random, webbrowser, queue, shutil
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# TERNARY BIT-PACKING
# Packs 5 trits {-1,0,+1} into 1 byte (3^5=243 < 256) — true ~1.6 bits/value,
# the real 20.2x compression vs float32 (32 bits/value).
# ══════════════════════════════════════════════════════════════════════════════

def pack_ternary(trits):
    """trits: int8 array (N, D) with values in {-1,0,1} -> packed uint8 (N, ceil(D/5))"""
    import numpy as np
    digits = (trits + 1).astype(np.int32)   # 0,1,2
    N, D = digits.shape
    pad = (-D) % 5
    if pad:
        digits = np.pad(digits, ((0, 0), (0, pad)), constant_values=1)  # pad with trit 0
    G = digits.shape[1] // 5
    digits = digits.reshape(N, G, 5)
    weights = np.array([1, 3, 9, 27, 81], dtype=np.int32)
    packed = (digits * weights).sum(axis=2).astype(np.uint8)
    return packed

def unpack_ternary(packed, orig_dim):
    """packed: uint8 array (N, G) -> trits int8 (N, orig_dim) with values in {-1,0,1}"""
    import numpy as np
    N, G = packed.shape
    vals = packed.astype(np.int32)
    out = np.zeros((N, G, 5), dtype=np.int8)
    for i in range(5):
        out[:, :, i] = (vals % 3) - 1
        vals //= 3
    return out.reshape(N, G * 5)[:, :orig_dim]

# ══════════════════════════════════════════════════════════════════════════════
# THEMES
# ══════════════════════════════════════════════════════════════════════════════

THEMES = {
    "Matrix": {
        "bg":           "#0D0D0D",
        "bg2":          "#111111",
        "fg":           "#00FF41",
        "fg_dim":       "#005C13",
        "fg_bright":    "#AFFFBC",
        "accent":       "#00FF41",
        "accent2":      "#00CC33",
        "error":        "#FF4444",
        "border":       "#003B00",
        "font_main":    ("Courier New", 11),
        "font_title":   ("Courier New", 22, "bold"),
        "font_small":   ("Courier New", 9),
        "rain_chars":   "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ01",
    },
    "Cyberpunk": {
        "bg":           "#0A0014",
        "bg2":          "#120020",
        "fg":           "#FF00FF",
        "fg_dim":       "#5C005C",
        "fg_bright":    "#FFB3FF",
        "accent":       "#00FFFF",
        "accent2":      "#FF00FF",
        "error":        "#FF4444",
        "border":       "#3D0060",
        "font_main":    ("Courier New", 11),
        "font_title":   ("Courier New", 22, "bold"),
        "font_small":   ("Courier New", 9),
        "rain_chars":   "▓▒░█▄▀■□●○◆◇★☆♦♠♣♥0123456789ABCDEF",
    },
    "Amber": {
        "bg":           "#0D0800",
        "bg2":          "#150E00",
        "fg":           "#FFB000",
        "fg_dim":       "#5C3E00",
        "fg_bright":    "#FFD966",
        "accent":       "#FF8C00",
        "accent2":      "#FFB000",
        "error":        "#FF4444",
        "border":       "#3D2800",
        "font_main":    ("Courier New", 11),
        "font_title":   ("Courier New", 22, "bold"),
        "font_small":   ("Courier New", 9),
        "rain_chars":   "01▌▐░▒▓█ABCDEF0123456789",
    },
    "Ice": {
        "bg":           "#00050F",
        "bg2":          "#000A1A",
        "fg":           "#00CFFF",
        "fg_dim":       "#004466",
        "fg_bright":    "#AAEEFF",
        "accent":       "#0088FF",
        "accent2":      "#00CFFF",
        "error":        "#FF4444",
        "border":       "#003355",
        "font_main":    ("Courier New", 11),
        "font_title":   ("Courier New", 22, "bold"),
        "font_small":   ("Courier New", 9),
        "rain_chars":   "❄✦✧⬡⬢◈◉○●◦•·˙0123456789",
    },
    "Ghost": {
        "bg":           "#080808",
        "bg2":          "#0F0F0F",
        "fg":           "#CCCCCC",
        "fg_dim":       "#444444",
        "fg_bright":    "#FFFFFF",
        "accent":       "#888888",
        "accent2":      "#CCCCCC",
        "error":        "#FF4444",
        "border":       "#222222",
        "font_main":    ("Courier New", 11),
        "font_title":   ("Courier New", 22, "bold"),
        "font_small":   ("Courier New", 9),
        "rain_chars":   "░▒▓01·˙",
    },
}

CURRENT_THEME = "Matrix"

# ══════════════════════════════════════════════════════════════════════════════
# MATRIX RAIN CANVAS
# ══════════════════════════════════════════════════════════════════════════════

class MatrixRain:
    def __init__(self, canvas, theme):
        self.canvas  = canvas
        self.theme   = theme
        self.columns = []
        self.running = False
        self.after_id = None

    def start(self, width, height):
        self.width   = width
        self.height  = height
        self.cols    = max(1, width // 14)
        self.rows    = max(1, height // 16)
        self.columns = [random.randint(0, self.rows) for _ in range(self.cols)]
        self.running = True
        self._draw()

    def stop(self):
        self.running = False
        if self.after_id:
            try: self.canvas.after_cancel(self.after_id)
            except: pass

    def _draw(self):
        if not self.running:
            return
        t   = self.theme
        chars = t["rain_chars"]
        self.canvas.delete("rain")

        for col, row in enumerate(self.columns):
            x = col * 14 + 7
            # Tail — dim
            for r in range(max(0, row-8), row):
                y    = r * 16 + 8
                char = random.choice(chars)
                self.canvas.create_text(x, y, text=char, fill=t["fg_dim"],
                                        font=t["font_small"], tags="rain")
            # Head — bright
            if row < self.rows:
                y    = row * 16 + 8
                char = random.choice(chars)
                self.canvas.create_text(x, y, text=char, fill=t["fg_bright"],
                                        font=t["font_small"], tags="rain")

            self.columns[col] = (row + 1) % (self.rows + random.randint(4, 16))

        self.after_id = self.canvas.after(80, self._draw)


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH ENGINE (runs in background thread)
# ══════════════════════════════════════════════════════════════════════════════

class SearchEngine:
    def __init__(self):
        self.index      = None
        self.metadata   = []
        self.path_table = []
        self.model      = None
        self.ready      = False
        self.status     = "Not initialized"

    def load(self, index_dir, model_path, on_status):
        def _load():
            try:
                on_status("Starting imports (sentence_transformers import alone takes ~55s on this machine)...")
                import faiss
                import numpy as np
                from sentence_transformers import SentenceTransformer

                on_status("Loading model...")
                self.model = SentenceTransformer(model_path)

                trit_path = os.path.join(index_dir, "vectors_ternary.npy")
                trit_meta_path = os.path.join(index_dir, "vectors_meta.json")
                idx_path  = os.path.join(index_dir, "faiss.index")
                meta_path = os.path.join(index_dir, "metadata.json")

                if os.path.exists(trit_path) and os.path.exists(meta_path):
                    on_status("Loading ternary-compressed index...")
                    packed = np.load(trit_path)
                    dim    = json.load(open(trit_meta_path)).get("dim", 384) if os.path.exists(trit_meta_path) else 384
                    disk_mb = packed.nbytes / 1e6
                    # Unpack ONCE here (disk stays 19.9x compressed) so every
                    # search afterward hits a plain float32 matmul, no per-query
                    # unpack cost — balances disk savings with search speed.
                    self.index    = unpack_ternary(packed, dim).astype("float32")
                    raw = json.load(open(meta_path, encoding="utf-8"))
                    self.path_table = raw.get("paths", [])
                    self.metadata   = raw.get("chunks", [])
                    on_status(f"Ready - {len(self.metadata):,} chunks indexed "
                              f"({disk_mb:.2f}MB on disk, 19.9x compressed). Start searching!")
                elif os.path.exists(idx_path) and os.path.exists(meta_path):
                    on_status("Loading index...")
                    self.index    = faiss.read_index(idx_path)
                    self.metadata = json.load(open(meta_path, encoding="utf-8"))
                    on_status(f"Ready - {len(self.metadata):,} chunks indexed. Start searching!")
                else:
                    on_status("No index yet - add a directory then click INDEX CODEBASE")
            except Exception as e:
                on_status(f"Error: {e}")
            finally:
                # Loop waiting on self.ready must always terminate, even if
                # on_status() itself raises (e.g. non-ASCII text hitting a
                # Windows console codec that can't encode it) - a status-print
                # failure must never silently hang every caller forever.
                self.ready = True
        threading.Thread(target=_load, daemon=True).start()

    def search(self, query, k=10, base_dir_filter=None):
        if not self.ready or self.index is None or self.model is None:
            return []
        import numpy as np
        vec = self.model.encode([query], normalize_embeddings=True).astype("float32")

        # Filter to a specific project's chunks *before* taking top-k, not
        # after — otherwise an unrelated, larger codebase in the same index
        # can crowd the target project out of the results entirely.
        chunk_mask = None
        if base_dir_filter and self.path_table:
            norm_filter = os.path.normcase(os.path.normpath(base_dir_filter))
            allowed_path_idx = {
                i for i, p in enumerate(self.path_table)
                if os.path.normcase(os.path.normpath(p["base_dir"])) == norm_filter
            }
            chunk_mask = np.array([
                isinstance(m, list) and m[0] in allowed_path_idx
                for m in self.metadata
            ])

        if isinstance(self.index, np.ndarray):
            # Ternary index, unpacked once at load time — fast float32 matmul,
            # disk file is still 19.9x compressed (see load()/build_index()).
            sims = self.index @ vec[0]
            if chunk_mask is not None:
                sims = np.where(chunk_mask, sims, -np.inf)
            top  = np.argsort(-sims)[:min(k, len(sims))]
            scores, indices = sims[top], top
        else:
            scores0, indices0 = self.index.search(vec, min(k, self.index.ntotal))
            scores, indices = scores0[0], indices0[0]
        results = []
        for score, idx in zip(scores, indices):
            if score == -np.inf:
                continue
            if idx >= 0 and idx < len(self.metadata):
                m = self.metadata[idx]
                if isinstance(m, list) and self.path_table:
                    # Compact format: [path_idx, offset] — regenerate
                    # preview lazily from the original file on disk.
                    p_idx, offset = m
                    p = self.path_table[p_idx]
                    rel_path = p["rel_path"]
                    preview = ""
                    try:
                        full = os.path.join(p["base_dir"], rel_path)
                        text = open(full, encoding="utf-8", errors="ignore").read()
                        preview = text[offset:offset+120].replace("\n", " ")
                    except Exception:
                        pass
                    results.append({"score": float(score), "path": rel_path, "preview": preview,
                                     "offset": offset, "idx": int(idx)})
                else:
                    # Legacy format fallback
                    results.append({
                        "score":   float(score),
                        "path":    m.get("rel_path", m.get("path", "?")),
                        "preview": m.get("preview", ""),
                        "offset":  m.get("offset"),
                        "idx":     int(idx),
                    })
        return results

    def build_index(self, scan_dirs, index_dir, on_status, on_done):
        def _build():
            try:
                import faiss, numpy as np
                from sentence_transformers import SentenceTransformer

                if self.model is None:
                    on_status("Loading model...")

                EXTS = {".py",".gd",".js",".ts",".cs",".rs",".go",
                        ".c",".cpp",".h",".java",".lua",".rb",".php",
                        ".swift",".kt",".dart",".zig",".md",".sh",".ps1"}
                # Universal default skip-list — folders that are virtually
                # never a user's own source code, regardless of whose
                # machine this runs on (installed software, OS internals,
                # package manager caches, build artifacts, web-page saves).
                SKIP = {
                    # Version control / build artifacts / common project conventions
                    ".git", "__pycache__", "node_modules", ".venv", "venv",
                    "dist", "build", "target", "models", "search_index",
                    "ai_files", "addons",
                    # Windows OS / system folders
                    "AppData", "Temp", "Windows", "Program Files",
                    "Program Files (x86)", "ProgramData",
                    "$Recycle.Bin", "System Volume Information",
                    # Dev toolchains / package manager caches (large, never
                    # the user's own code)
                    "msys64", "mingw64", "mingw32", "Anaconda3", "miniconda3",
                    "site-packages", ".cargo", ".rustup",
                    ".nuget", ".gradle", ".m2",
                    # Common large bundled creative/consumer software
                    "Ableton", "Steam", "steamapps", "Epic Games",
                    "Adobe", "Spotify",
                }

                chunks = []
                files  = 0
                on_status("Scanning files...")

                SKIP_PATTERNS = ("_files", "_assets")

                for base in scan_dirs:
                    for root, dirs, fnames in os.walk(base):
                        dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")
                                   and not any(p in d for p in SKIP_PATTERNS)]
                        for fname in fnames:
                            if Path(fname).suffix.lower() not in EXTS:
                                continue
                            fpath = os.path.join(root, fname)
                            try:
                                text = open(fpath, encoding="utf-8", errors="ignore").read()
                                if len(text.strip()) < 100:
                                    continue
                                rel = os.path.relpath(fpath, base)
                                # Chunk — store only path + offset, not the
                                # full text/preview (regenerated lazily on
                                # display from the original file on disk).
                                for i in range(0, len(text), 700):
                                    chunk = text[i:i+800]
                                    if len(chunk.strip()) > 50:
                                        chunks.append({
                                            "text":     f"file:{rel}\n{chunk}",
                                            "rel_path": rel,
                                            "base_dir": base,
                                            "offset":   i,
                                        })
                                files += 1
                            except: pass

                on_status(f"Encoding {len(chunks):,} chunks from {files:,} files...")
                os.makedirs(index_dir, exist_ok=True)

                texts = [c["text"] for c in chunks]
                bs    = 128
                vecs  = []
                for i in range(0, len(texts), bs):
                    batch = texts[i:i+bs]
                    v = self.model.encode(batch, normalize_embeddings=True,
                                          show_progress_bar=False)
                    vecs.append(v)
                    pct = (i+bs) / len(texts) * 100
                    on_status(f"Encoding... {min(pct,100):.0f}%  ({min(i+bs, len(texts)):,}/{len(texts):,})")

                vecs = np.vstack(vecs).astype("float32")

                # Ternary compression: quantize to {-1,0,+1}, then bit-pack
                # 5 trits/byte (3^5=243<256) for the real ~20x reduction vs
                # float32. Query stays float32 at search time (asymmetric).
                t = 0.7 * np.abs(vecs).mean()
                trit_vecs = np.where(vecs > t, 1, np.where(vecs < -t, -1, 0)).astype("int8")
                packed    = pack_ternary(trit_vecs)

                old_index = os.path.join(index_dir, "faiss.index")
                if os.path.exists(old_index):
                    os.remove(old_index)
                np.save(os.path.join(index_dir, "vectors_ternary.npy"), packed)
                json.dump({"dim": int(vecs.shape[1])},
                          open(os.path.join(index_dir, "vectors_meta.json"), "w"))

                # Deduplicate paths into a small table; store only a path
                # index + byte offset per chunk (preview regenerated lazily
                # from the original file on disk, not cached here).
                path_table = []
                path_idx   = {}
                rows = []
                for c in chunks:
                    key = (c["base_dir"], c["rel_path"])
                    if key not in path_idx:
                        path_idx[key] = len(path_table)
                        path_table.append({"base_dir": c["base_dir"], "rel_path": c["rel_path"]})
                    rows.append([path_idx[key], c["offset"]])

                json.dump({"paths": path_table, "chunks": rows},
                          open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8"),
                          ensure_ascii=False, separators=(",", ":"))

                # Disk file stays packed (19.9x compressed); keep the
                # unpacked float32 view in memory for fast search.
                self.index      = trit_vecs.astype("float32")
                self.metadata   = rows
                self.path_table = path_table
                self.ready      = True
                size_mb  = packed.nbytes / 1e6
                float_mb = vecs.nbytes / 1e6
                on_status(f"Index complete — {len(chunks):,} chunks, {files:,} files "
                          f"({size_mb:.2f}MB packed-ternary vs {float_mb:.2f}MB float32, "
                          f"{float_mb/size_mb:.1f}x smaller)")
                on_done()
            except Exception as e:
                on_status(f"Index error: {e}")
        threading.Thread(target=_build, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

class TritSearchApp:
    def __init__(self, root):
        self.root         = root
        self.theme_name   = CURRENT_THEME
        self.t            = THEMES[self.theme_name]
        self.engine       = SearchEngine()
        self.rain         = None
        self.scan_dirs    = []
        self.index_dir    = str(Path.home() / ".trit-search" / "index")
        self.model_path   = self._find_model()
        self.typing_job   = None
        self._last_results = []

        self.root.title("OBSERVE")
        self.root.geometry("1000x720")
        self.root.minsize(800, 600)
        self.root.configure(bg=self.t["bg"])

        self._build_ui()
        self._apply_theme()
        self._start_rain()
        self._load_config()
        self._set_status("⟳ Loading model... (first launch takes 10-30s)")
        self.root.after(100, self._init_engine)

    def _find_model(self):
        # Look for fine-tuned model next to script, then fallback to base
        candidates = [
            Path(__file__).parent / "models" / "code-minilm",
            Path.home() / ".trit-search" / "models" / "code-minilm",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return "all-MiniLM-L6-v2"

    # ── UI BUILD ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        t = self.t

        # ── Top bar ──
        self.topbar = tk.Frame(self.root, bg=t["bg"], height=50)
        self.topbar.pack(fill="x", padx=0, pady=0)
        self.topbar.pack_propagate(False)

        self.title_lbl = tk.Label(self.topbar, text="◈ OBSERVE",
                                   font=t["font_title"], bg=t["bg"], fg=t["accent"])
        self.title_lbl.pack(side="left", padx=20, pady=8)

        # Theme selector
        theme_frame = tk.Frame(self.topbar, bg=t["bg"])
        theme_frame.pack(side="right", padx=16, pady=8)

        tk.Label(theme_frame, text="THEME:", font=t["font_small"],
                 bg=t["bg"], fg=t["fg_dim"]).pack(side="left", padx=(0,6))

        self.theme_var = tk.StringVar(value=self.theme_name)
        self.theme_menu = ttk.Combobox(theme_frame, textvariable=self.theme_var,
                                        values=list(THEMES.keys()), width=10,
                                        state="readonly", font=t["font_small"])
        self.theme_menu.pack(side="left")
        self.theme_menu.bind("<<ComboboxSelected>>", self._on_theme_change)

        # ── Divider ──
        self.div1 = tk.Frame(self.root, bg=t["border"], height=1)
        self.div1.pack(fill="x")

        # ── Main pane ──
        self.main = tk.Frame(self.root, bg=t["bg"])
        self.main.pack(fill="both", expand=True)

        # Left sidebar
        self.sidebar = tk.Frame(self.main, bg=t["bg2"], width=220)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="CODEBASE", font=self.t["font_small"],
                 bg=t["bg2"], fg=t["fg_dim"]).pack(pady=(16,4), padx=12, anchor="w")

        self.dir_listbox = tk.Listbox(self.sidebar, bg=t["bg"], fg=t["fg"],
                                       font=t["font_small"], selectbackground=t["border"],
                                       selectforeground=t["fg_bright"],
                                       relief="flat", highlightthickness=0,
                                       height=6)
        self.dir_listbox.pack(fill="x", padx=8)

        btn_frame = tk.Frame(self.sidebar, bg=t["bg2"])
        btn_frame.pack(fill="x", padx=8, pady=4)

        self.add_dir_btn = self._btn(btn_frame, "+ ADD DIR", self._add_dir)
        self.add_dir_btn.pack(side="left", padx=(0,4))
        self.rem_dir_btn = self._btn(btn_frame, "− REMOVE", self._remove_dir)
        self.rem_dir_btn.pack(side="left")

        tk.Frame(self.sidebar, bg=t["border"], height=1).pack(fill="x", pady=8, padx=8)

        self.index_btn = self._btn(self.sidebar, "⟳ INDEX CODEBASE", self._start_index,
                                    width=18)
        self.index_btn.pack(pady=4, padx=8, fill="x")

        self.serve_btn = self._btn(self.sidebar, "⊕ OPEN IN BROWSER", self._open_browser,
                                    width=18)
        self.serve_btn.pack(pady=4, padx=8, fill="x")

        self.entangle_btn = self._btn(self.sidebar, "⊞ ENTANGLEMENT", self._open_entanglement_window,
                                       width=18)
        self.entangle_btn.pack(pady=4, padx=8, fill="x")

        self.run_pipeline_btn = self._btn(self.sidebar, "▶ RUN FULL ANALYSIS", self._open_run_pipeline_window,
                                           width=18)
        self.run_pipeline_btn.pack(pady=4, padx=8, fill="x")

        tk.Frame(self.sidebar, bg=t["border"], height=1).pack(fill="x", pady=8, padx=8)

        # Stats
        tk.Label(self.sidebar, text="STATS", font=self.t["font_small"],
                 bg=t["bg2"], fg=t["fg_dim"]).pack(pady=(4,2), padx=12, anchor="w")
        self.stats_lbl = tk.Label(self.sidebar, text="—", font=t["font_small"],
                                   bg=t["bg2"], fg=t["fg"], wraplength=190, justify="left")
        self.stats_lbl.pack(padx=12, anchor="w")

        # Rain canvas in sidebar
        self.rain_canvas = tk.Canvas(self.sidebar, bg=t["bg"], highlightthickness=0)
        self.rain_canvas.pack(fill="both", expand=True, padx=0, pady=(8,0))

        # ── Right panel ──
        self.right = tk.Frame(self.main, bg=t["bg"])
        self.right.pack(side="left", fill="both", expand=True)

        # Search bar
        search_frame = tk.Frame(self.right, bg=t["bg"])
        search_frame.pack(fill="x", padx=16, pady=12)

        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(search_frame, textvariable=self.search_var,
                                      font=("Courier New", 14),
                                      bg=t["bg2"], fg=t["fg"],
                                      insertbackground=t["accent"],
                                      relief="flat", highlightthickness=2,
                                      highlightcolor=t["accent"],
                                      highlightbackground=t["border"])
        self.search_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0,8))
        self.search_entry.bind("<Return>", lambda e: self._search())
        self.search_entry.bind("<KeyRelease>", self._on_key)
        self.search_entry.insert(0, "Search your codebase by meaning...")
        self.search_entry.bind("<FocusIn>", self._clear_placeholder)
        self.search_entry.bind("<FocusOut>", self._restore_placeholder)

        self.search_btn = self._btn(search_frame, "SEARCH", self._search, width=10)
        self.search_btn.pack(side="left", padx=(0,4))

        self.copy_all_btn = self._btn(search_frame, "⎘ COPY ALL", self._copy_all, width=12)
        self.copy_all_btn.pack(side="left")

        # Status bar
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_lbl = tk.Label(self.right, textvariable=self.status_var,
                                    font=t["font_small"], bg=t["bg"],
                                    fg=t["fg_dim"], anchor="w")
        self.status_lbl.pack(fill="x", padx=16)

        # Results
        self.results_frame = tk.Frame(self.right, bg=t["bg"])
        self.results_frame.pack(fill="both", expand=True, padx=16, pady=8)

        self.results_canvas = tk.Canvas(self.results_frame, bg=t["bg"],
                                         highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical",
                                        command=self.results_canvas.yview)
        self.results_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.results_canvas.pack(side="left", fill="both", expand=True)

        self.results_inner = tk.Frame(self.results_canvas, bg=t["bg"])
        self.results_window = self.results_canvas.create_window(
            (0,0), window=self.results_inner, anchor="nw"
        )
        self.results_inner.bind("<Configure>", self._on_results_configure)
        self.results_canvas.bind("<Configure>", self._on_canvas_configure)
        self.results_canvas.bind("<MouseWheel>", self._on_mousewheel)

        # Bottom bar
        self.botbar = tk.Frame(self.root, bg=t["bg2"], height=28)
        self.botbar.pack(fill="x", side="bottom")
        self.botbar.pack_propagate(False)
        tk.Label(self.botbar, text="012 TERNARY SEARCH  ·  LOCAL  ·  PRIVATE  ·  FREE",
                 font=t["font_small"], bg=t["bg2"], fg=t["fg_dim"]).pack(side="left", padx=12)
        tk.Label(self.botbar, text="v1.0",
                 font=t["font_small"], bg=t["bg2"], fg=t["fg_dim"]).pack(side="right", padx=12)

    def _btn(self, parent, text, cmd, width=None):
        t   = self.t
        kw  = dict(text=text, command=cmd, font=t["font_small"],
                   bg=t["bg2"], fg=t["accent"], relief="flat",
                   activebackground=t["border"], activeforeground=t["fg_bright"],
                   cursor="hand2", bd=1, highlightthickness=1,
                   highlightbackground=t["border"])
        if width: kw["width"] = width
        b = tk.Button(parent, **kw)
        b.bind("<Enter>", lambda e: b.config(bg=t["border"], fg=t["fg_bright"]))
        b.bind("<Leave>", lambda e: b.config(bg=t["bg2"], fg=t["accent"]))
        return b

    # ── THEME ─────────────────────────────────────────────────────────────────

    def _apply_theme(self):
        t = self.t
        # Restyle combobox
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TCombobox",
                         fieldbackground=t["bg2"],
                         background=t["bg2"],
                         foreground=t["fg"],
                         selectbackground=t["border"],
                         selectforeground=t["fg_bright"])
        style.configure("TScrollbar",
                         background=t["bg2"],
                         troughcolor=t["bg"],
                         arrowcolor=t["fg_dim"])

    def _on_theme_change(self, event=None):
        name = self.theme_var.get()
        self.theme_name = name
        self.t = THEMES[name]
        self._save_config()

        # Restart rain
        if self.rain:
            self.rain.stop()

        # Rebuild UI with new theme
        for widget in self.root.winfo_children():
            widget.destroy()
        self._build_ui()
        self._apply_theme()
        self._start_rain()
        self._restore_dirs()
        self._update_stats()
        self.theme_var.set(name)
        self.theme_menu.set(name)

    # ── RAIN ──────────────────────────────────────────────────────────────────

    def _start_rain(self):
        self.rain_canvas.update()
        w = self.rain_canvas.winfo_width()
        h = self.rain_canvas.winfo_height()
        if w < 10:
            self.root.after(200, self._start_rain)
            return
        self.rain = MatrixRain(self.rain_canvas, self.t)
        self.rain.start(w, h)

    # ── SEARCH ────────────────────────────────────────────────────────────────

    def _clear_placeholder(self, event):
        if self.search_entry.get() == "Search your codebase by meaning...":
            self.search_entry.delete(0, "end")
            self.search_entry.config(fg=self.t["fg"])

    def _restore_placeholder(self, event):
        if not self.search_entry.get():
            self.search_entry.insert(0, "Search your codebase by meaning...")
            self.search_entry.config(fg=self.t["fg_dim"])

    def _on_key(self, event):
        if self.typing_job:
            self.root.after_cancel(self.typing_job)
        q = self.search_var.get().strip()
        if len(q) > 3 and q != "Search your codebase by meaning...":
            self.typing_job = self.root.after(400, self._search)

    def _search(self):
        q = self.search_var.get().strip()
        if not q or q == "Search your codebase by meaning...":
            return
        if not self.engine.ready:
            self._set_status("Still loading — wait a moment...")
            return
        if self.engine.index is None:
            self._set_status("⟳ No index yet — add a directory then click INDEX CODEBASE")
            return

        idx = self.engine.index
        import numpy as np
        n_chunks = len(idx) if isinstance(idx, np.ndarray) else idx.ntotal
        self._set_status(f"Searching... index={n_chunks} chunks  model={'loaded' if self.engine.model else 'NONE'}")
        t0      = time.time()
        try:
            results = self.engine.search(q, k=12)
        except Exception as e:
            self._set_status(f"Search error: {e}")
            return
        elapsed = (time.time() - t0) * 1000

        self._set_status(f"  {len(results)} results  ·  {elapsed:.0f}ms  ·  query: \"{q}\"")
        self._last_results = results
        self._show_results(results)

    def _show_results(self, results):
        t = self.t
        for w in self.results_inner.winfo_children():
            w.destroy()

        if not results:
            tk.Label(self.results_inner, text="No results found.",
                     font=t["font_main"], bg=t["bg"], fg=t["fg_dim"]).pack(pady=20)
            return

        for i, r in enumerate(results):
            card = tk.Frame(self.results_inner, bg=t["bg2"],
                            highlightthickness=1, highlightbackground=t["border"])
            card.pack(fill="x", pady=3)

            # Score bar
            score_pct = max(0, min(1, r["score"]))
            bar_w     = int(score_pct * 80)
            bar_color = t["accent"] if score_pct > 0.5 else t["fg_dim"]

            header = tk.Frame(card, bg=t["bg2"])
            header.pack(fill="x", padx=8, pady=(6,2))

            score_bar = tk.Frame(header, bg=bar_color, width=bar_w, height=3)
            score_bar.pack(side="left", pady=(4,0))

            tk.Label(header, text=f"  {r['score']:.3f}",
                     font=t["font_small"], bg=t["bg2"], fg=bar_color).pack(side="left")

            path_lbl = tk.Label(header, text=r["path"],
                                 font=("Courier New", 10, "bold"),
                                 bg=t["bg2"], fg=t["fg_bright"],
                                 cursor="hand2")
            path_lbl.pack(side="left", padx=8)
            path_lbl.bind("<Button-1>", lambda e, p=r["path"], o=r.get("offset"): self._open_file(p, o))

            copy_btn = tk.Label(header, text="⎘ COPY", font=t["font_small"],
                                bg=t["bg2"], fg=t["fg_dim"], cursor="hand2")
            copy_btn.pack(side="left", padx=4)
            copy_btn.bind("<Button-1>", lambda e, p=r["path"]: self._copy(p))

            entangle_lbl = tk.Label(header, text="⊞ ENTANGLEMENT", font=t["font_small"],
                                     bg=t["bg2"], fg=t["fg_dim"], cursor="hand2")
            entangle_lbl.pack(side="left", padx=4)
            entangle_lbl.bind("<Button-1>", lambda e, p=r["path"]: self._open_entanglement_for_path(p))

            # A Label can't be selected/copied with the mouse at all --
            # the only way to interact with a result used to be clicking
            # the path, which immediately launches an external editor.
            # A Text widget allows mouse selection + copy. Real bug found
            # twice in a row here: first attempt disabled it (state=
            # "disabled") to make it read-only, but Tk renders disabled
            # Text in its own default gray regardless of fg -- that WAS
            # the "different color text" bug. Second attempt tried to fix
            # that with disabledforeground, but Text (unlike Entry/Label)
            # doesn't support that option at all -- crashed the app on
            # the very first search (verified directly: constructing a
            # Text widget with disabledforeground raises TclError). Left
            # as a normal (not disabled) widget instead: nothing is ever
            # saved from this preview, so a technically-editable-in-place
            # display is a fine tradeoff for correct color + real
            # selection, and it's the simplest fix that doesn't reintroduce
            # either previous bug.
            preview_text = r["preview"][:200].replace("\n", "  ")
            n_lines = max(1, -(-len(preview_text) // 90))  # matches wraplength roughly
            preview = tk.Text(card, height=min(n_lines, 4), font=t["font_small"],
                               bg=t["bg2"], fg=t["fg"], wrap="word",
                               relief="flat", highlightthickness=0,
                               borderwidth=0, padx=0, pady=0,
                               cursor="xterm")
            preview.insert("1.0", preview_text)
            preview.pack(fill="x", padx=12, pady=(0,6))

            # Hover
            for w in [card, header, preview]:
                w.bind("<Enter>", lambda e, c=card: c.config(
                    highlightbackground=self.t["accent"]))
                w.bind("<Leave>", lambda e, c=card: c.config(
                    highlightbackground=self.t["border"]))

    def _copy(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status(f"Copied: {text}")

    def _copy_all(self):
        results = getattr(self, "_last_results", [])
        if not results:
            self._set_status("No results to copy.")
            return
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] [{r['score']:.3f}] {r['path']}")
            lines.append(f"     {r['preview'][:120].strip()}")
            lines.append("")
        text = "\n".join(lines)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status(f"Copied {len(results)} results to clipboard.")

    def _open_file(self, rel_path, offset=None):
        for base in self.scan_dirs:
            full = os.path.join(base, rel_path)
            if os.path.exists(full):
                # Jump straight to the matched line, not just the top of
                # the file -- previously clicking a result opened the file
                # generically, leaving the user to manually re-find the
                # actual match. VS Code's CLI supports --goto file:line
                # for this; fall back to a plain open if `code` isn't on
                # PATH or the line can't be computed.
                line = None
                if offset is not None:
                    try:
                        text = open(full, encoding="utf-8", errors="ignore").read()
                        line = text.count("\n", 0, offset) + 1
                    except Exception:
                        line = None

                if line is not None:
                    # subprocess.run(["code", ...]) fails with
                    # FileNotFoundError on Windows even when `code` is on
                    # PATH -- CreateProcess doesn't resolve an extensionless
                    # name to code.CMD the way a real shell does. shutil.which
                    # resolves through PATHEXT correctly; verified directly
                    # (bare "code" failed, the shutil.which-resolved path
                    # succeeded, returncode 0).
                    code_exe = shutil.which("code")
                    if code_exe:
                        try:
                            subprocess.run([code_exe, "--goto", f"{full}:{line}"], check=True)
                            return
                        except Exception:
                            pass  # launch failed -- fall through to a plain open

                if sys.platform == "darwin":
                    subprocess.run(["open", full])
                elif sys.platform == "win32":
                    os.startfile(full)
                else:
                    subprocess.run(["xdg-open", full])
                return

    def _open_entanglement_for_path(self, rel_path):
        # Click-through from a search result straight to that file's
        # project entanglement view, instead of making the user open the
        # Entanglement window separately and hunt for the right project.
        # observe_pipeline is imported lazily (not at module level) because
        # it itself imports SearchEngine from this file -- a top-level
        # import here would be circular.
        from observe_pipeline import _infer_project_name

        base_dir = None
        for b in self.scan_dirs:
            if os.path.exists(os.path.join(b, rel_path)):
                base_dir = b
                break
        if base_dir is None and self.engine.path_table:
            # scan_dirs might not include every base_dir actually indexed
            # (e.g. re-opened without re-adding dirs) -- fall back to
            # whatever base_dirs the loaded index itself knows about.
            for p in self.engine.path_table:
                if os.path.exists(os.path.join(p["base_dir"], rel_path)):
                    base_dir = p["base_dir"]
                    break
        if base_dir is None:
            messagebox.showinfo("Can't locate project",
                                 f"Couldn't resolve {rel_path} to a base directory.")
            return

        project = _infer_project_name(base_dir.replace("\\", "/"), rel_path)
        self._open_entanglement_window(initial_project=project)

    # ── INDEX ─────────────────────────────────────────────────────────────────

    def _add_dir(self):
        d = filedialog.askdirectory(title="Select codebase directory")
        if d and d not in self.scan_dirs:
            self.scan_dirs.append(d)
            self.dir_listbox.insert("end", d)
            self._save_config()

    def _remove_dir(self):
        sel = self.dir_listbox.curselection()
        if sel:
            idx = sel[0]
            self.scan_dirs.pop(idx)
            self.dir_listbox.delete(idx)
            self._save_config()

    def _restore_dirs(self):
        self.dir_listbox.delete(0, "end")
        for d in self.scan_dirs:
            self.dir_listbox.insert("end", d)

    def _start_index(self):
        if not self.scan_dirs:
            self._set_status("Add a codebase directory first.")
            return
        self.index_btn.config(state="disabled")
        self.engine.build_index(
            self.scan_dirs, self.index_dir,
            on_status=self._set_status,
            on_done=self._on_index_done
        )

    def _on_index_done(self):
        self.root.after(0, lambda: self.index_btn.config(state="normal"))
        self.root.after(0, self._update_stats)

    def _open_browser(self):
        webbrowser.open(f"http://localhost:5050")

    # ── ENTANGLEMENT WINDOW ──────────────────────────────────────────────────
    # Browses code_entanglement_db.json (built separately by
    # trit_entanglement.py -- this window only reads it, never runs the
    # pipeline itself, since that takes many minutes and needs Ollama).

    def _references_db_path(self):
        # Prefer the combined database (run_full_pipeline.py's output --
        # one shared run of every stage) over the standalone
        # code_references_results.json, since the combined file is always
        # at least as fresh and also carries the entanglement/AST data the
        # standalone file doesn't. Falls back if the combined run hasn't
        # been done yet.
        combined = Path(__file__).resolve().parent / "observe_full_database.json"
        if combined.exists():
            return combined
        return Path(__file__).resolve().parent / "code_references_results.json"

    def _open_entanglement_window(self, initial_project=None):
        # NOTE: this window shows real code REFERENCES (imports, preload/
        # load() calls, require(), #include), not embedding-similarity
        # "entanglement" scores. By explicit request: this should show
        # where code actually SPEAKS to other files/projects, not how
        # similar they conceptually look. Built by code_references.py,
        # which resolves each reference against the real filesystem
        # before ever claiming a cross-project relationship -- see that
        # file's docstring/commit history for the two real false-positive
        # classes found and fixed (basename collisions, bare Python
        # imports resolving to a same-directory sibling instead).
        db_path = self._references_db_path()
        if not db_path.exists():
            messagebox.showinfo(
                "No references database",
                f"No database found at:\n{db_path}\n\n"
                "Run `python code_references.py` (references only, fast) or "
                "`python run_full_pipeline.py` (everything, slower) from a "
                "terminal first to build it."
            )
            return
        try:
            db = json.loads(db_path.read_text(encoding="utf-8"))
        except Exception as e:
            messagebox.showerror("Failed to load database", str(e))
            return

        t = self.t
        win = tk.Toplevel(self.root)
        win.title("OBSERVE — Code References")
        win.geometry("1000x680")
        win.configure(bg=t["bg"])

        header = tk.Label(win, text="⊞ CODE REFERENCES", font=t["font_title"],
                           bg=t["bg"], fg=t["accent"])
        header.pack(pady=(12, 4))

        refs = db.get("cross_project_references", [])
        # run_full_pipeline.py's combined output nests these under
        # reference_stats; the standalone code_references.py output has
        # them at the top level. Handle both shapes rather than force one
        # schema to match the other after the fact.
        within = db.get("within_project_count", db.get("reference_stats", {}).get("within_project_count", 0))
        info = tk.Label(
            win,
            text=(f"{len(refs)} cross-project references (real imports/loads/includes, "
                  f"filesystem-verified)  ·  {within} within-project  ·  {db_path.name}"),
            font=t["font_small"], bg=t["bg"], fg=t["fg_dim"])
        info.pack(pady=(0, 8))

        body = tk.Frame(win, bg=t["bg"])
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Left: project list, derived from which projects actually appear
        # as the SOURCE of a real cross-project reference.
        left = tk.Frame(body, bg=t["bg2"], width=260)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        tk.Label(left, text="PROJECTS (click one)", font=t["font_small"],
                 bg=t["bg2"], fg=t["fg_dim"]).pack(pady=(8, 4), padx=8, anchor="w")

        proj_list = tk.Listbox(left, bg=t["bg"], fg=t["fg"], font=t["font_small"],
                                selectbackground=t["border"], selectforeground=t["fg_bright"],
                                relief="flat", highlightthickness=0)
        proj_list.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Real confusion found via actual use: the list showed "(0
        # references)" for projects like Spikeling-Project, which the user
        # correctly pointed out is misleading -- Spikeling-Project IS
        # referenced by other projects (3 times, from loose scripts), it
        # just never imports/loads anything FROM another project itself.
        # by_source only ever counted OUTGOING references, with no label
        # distinguishing that from incoming -- "0" read as wrong even
        # though it was accurate for the specific (unstated) direction
        # being measured. Track both directions and show both explicitly.
        by_source = {}
        by_target = {}
        for r in refs:
            by_source.setdefault(r["source_project"], []).append(r)
            by_target.setdefault(r["target_project"], []).append(r)

        # If this is the combined database, it also has per-project
        # summaries -- list EVERY real project (not just ones with a
        # reference), so clicking through from a search result always
        # lands on something, and show the summary alongside references.
        projects_info = db.get("projects", {})
        all_names = set(by_source.keys()) | set(projects_info.keys())

        # Real bug found via direct testing: opening this window from the
        # SIDEBAR button (no specific file) only ever showed
        # "(loose scripts, no project folder)", because that's the only
        # project with any reference data in the standalone
        # code_references_results.json -- confirmed directly, the sidebar
        # path produced a listbox with exactly one item. Every other real
        # project (Spikeling-Project, 012-ternary, etc.) was invisible
        # even though the loaded index knows about all of them. Compute
        # the REAL full project list from the already-loaded engine (no
        # reload -- group_chunks_by_project is cheap, pure in-memory) so
        # the sidebar button is useful on its own, not just as a landing
        # spot for click-through.
        if self.engine.ready:
            try:
                from observe_pipeline import group_chunks_by_project, MIN_CHUNKS_PER_PROJECT, NON_PROJECT_HINTS
                groups = group_chunks_by_project(self.engine)
                for name, idxs in groups.items():
                    if len(idxs) >= MIN_CHUNKS_PER_PROJECT and name.lower() not in NON_PROJECT_HINTS:
                        all_names.add(name)
            except Exception:
                pass  # fall back to whatever the JSON file alone provides
        # Real bug found via actual use: when only the standalone
        # code_references_results.json exists (no "projects" section at
        # all), only the handful of projects that happen to have a
        # cross-project reference show up here -- most real projects
        # (Spikeling-Project, 012-ternary, etc.) have zero references and
        # were silently missing from this list entirely. Clicking through
        # from a search result on one of those landed on nothing, with no
        # visible error -- looked exactly like the feature was broken.
        # Always include initial_project even with zero known data, so a
        # click-through confirms it worked instead of silently vanishing.
        if initial_project:
            all_names.add(initial_project)
        proj_names = sorted(all_names, key=lambda n: -(len(by_source.get(n, [])) + len(by_target.get(n, []))))
        for name in proj_names:
            n_out = len(by_source.get(name, []))
            n_in = len(by_target.get(name, []))
            proj_list.insert("end", f"{name}  ({n_out} out, {n_in} in)")

        # Right: detail pane
        right = tk.Frame(body, bg=t["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        detail = scrolledtext.ScrolledText(
            right, bg=t["bg2"], fg=t["fg"], font=t["font_small"],
            wrap="word", relief="flat", highlightthickness=0, insertbackground=t["fg"]
        )
        detail.pack(fill="both", expand=True)
        detail.insert(
            "end",
            "Select a project on the left to see the real files it literally\n"
            "imports, loads, or includes from OTHER projects -- an actual\n"
            "reference, not an inferred similarity.\n\n"
            "[non-ambiguous] = exactly one indexed file matches this reference\n"
            "[AMBIGUOUS] = multiple files share this name; shown but not certain"
        )
        detail.config(state="disabled")

        def show_project(name):
            outgoing = by_source.get(name, [])
            incoming = by_target.get(name, [])
            info = projects_info.get(name, {})

            detail.config(state="normal")
            detail.delete("1.0", "end")
            detail.insert("end", f"{name}\n", "h1")

            if info:
                detail.insert("end", f"{info.get('chunk_count', 0)} chunks indexed\n\n")
                detail.insert("end", f"{info.get('summary', '(no summary)')}\n\n")
                claims = info.get("unsupported_claims") or []
                if claims:
                    detail.insert("end", "FLAGGED — possibly unsupported claim(s):\n")
                    for c in claims:
                        detail.insert("end", f"  - {c[:200]}\n")
                    detail.insert("end", "\n")
            else:
                detail.insert("end", "(no summary available -- run run_full_pipeline.py or "
                                      "trit_entanglement.py to generate one)\n\n")

            detail.insert("end", "─" * 60 + "\n")
            detail.insert("end", f"OUTGOING -- this project imports/loads/includes -- {len(outgoing)} found\n")
            detail.insert("end", "─" * 60 + "\n\n")

            if not outgoing:
                detail.insert("end", "(none -- this project doesn't literally import/load/include\n"
                                      "anything from another project)\n\n")
            for r in outgoing:
                tag = "AMBIGUOUS" if r.get("ambiguous") else "confirmed"
                detail.insert("end", f"[{tag}] {r['source_path']}\n")
                detail.insert("end", f"    -> \"{r['raw_reference']}\"\n")
                detail.insert("end", f"    -> {r['target_project']}:{r['target_path']}\n\n")

            detail.insert("end", "─" * 60 + "\n")
            detail.insert("end", f"INCOMING -- other projects import/load/include FROM this one -- {len(incoming)} found\n")
            detail.insert("end", "─" * 60 + "\n\n")

            if not incoming:
                detail.insert("end", "(none -- no other project literally imports/loads/includes\n"
                                      "anything from this one)\n\n")
            for r in incoming:
                tag = "AMBIGUOUS" if r.get("ambiguous") else "confirmed"
                detail.insert("end", f"[{tag}] {r['source_project']}:{r['source_path']}\n")
                detail.insert("end", f"    -> \"{r['raw_reference']}\"\n")
                detail.insert("end", f"    -> {r['target_path']}\n\n")

            detail.config(state="disabled")

        def on_select(evt):
            sel = proj_list.curselection()
            if not sel:
                return
            idx = sel[0]
            show_project(proj_names[idx])

        if initial_project and initial_project in proj_names:
            idx = proj_names.index(initial_project)
            proj_list.selection_set(idx)
            proj_list.see(idx)
            show_project(initial_project)

        proj_list.bind("<<ListboxSelect>>", on_select)

    # ── RUN FULL ANALYSIS (with visible progress/elapsed time) ───────────────
    # Real problem this fixes: the analysis pipeline previously only ran
    # from a bare terminal with no timestamps on most output lines, and
    # once already hung silently for 40+ minutes tonight due to a real bug
    # (a Windows console encoding crash swallowed by a broken except path)
    # -- from outside, "hung" and "just slow" looked identical. This window
    # streams the subprocess's real output AND keeps a live elapsed-time
    # counter and a "time since last output line" indicator, so a genuine
    # stall is visibly distinguishable from normal slow progress instead of
    # requiring someone to go measure CPU time by hand to tell the
    # difference (which is literally what happened here earlier tonight).

    def _open_run_pipeline_window(self):
        if getattr(self, "_pipeline_proc", None) is not None and self._pipeline_proc.poll() is None:
            messagebox.showinfo("Already running", "A pipeline run is already in progress.")
            return

        t = self.t
        win = tk.Toplevel(self.root)
        win.title("OBSERVE — Run Full Analysis")
        win.geometry("900x600")
        win.configure(bg=t["bg"])

        header = tk.Label(win, text="▶ RUNNING FULL ANALYSIS", font=t["font_title"],
                           bg=t["bg"], fg=t["accent"])
        header.pack(pady=(12, 4))

        status_lbl = tk.Label(win, text="Starting...", font=t["font_small"],
                               bg=t["bg"], fg=t["fg_dim"])
        status_lbl.pack(pady=(0, 8))

        output = scrolledtext.ScrolledText(
            win, bg=t["bg2"], fg=t["fg"], font=("Courier New", 9),
            wrap="word", relief="flat", highlightthickness=0, insertbackground=t["fg"]
        )
        output.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        output.config(state="disabled")

        q = queue.Queue()
        state = {"start": time.time(), "last_line": time.time(), "done": False, "returncode": None}

        script_path = Path(__file__).resolve().parent / "run_full_pipeline.py"
        proc = subprocess.Popen(
            [sys.executable, "-u", str(script_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=str(script_path.parent),
        )
        self._pipeline_proc = proc

        def reader_thread():
            for line in proc.stdout:
                q.put(line)
            proc.wait()
            q.put(None)  # sentinel: process exited

        threading.Thread(target=reader_thread, daemon=True).start()

        def poll_queue():
            try:
                while True:
                    line = q.get_nowait()
                    if line is None:
                        state["done"] = True
                        state["returncode"] = proc.returncode
                        break
                    state["last_line"] = time.time()
                    output.config(state="normal")
                    output.insert("end", line)
                    output.see("end")
                    output.config(state="disabled")
            except queue.Empty:
                pass

            elapsed = time.time() - state["start"]
            since_last = time.time() - state["last_line"]
            mins, secs = divmod(int(elapsed), 60)

            if state["done"]:
                ok = state["returncode"] == 0
                status_lbl.config(
                    text=(f"{'✓ Finished' if ok else '✗ Failed'} in {mins}m {secs}s "
                          f"(exit code {state['returncode']})"),
                    fg=(t["accent"] if ok else "#ff6666"),
                )
                self._pipeline_proc = None
                return  # stop polling -- process is done

            warn = since_last > 90
            status_lbl.config(
                text=(f"Running... {mins}m {secs}s elapsed"
                      + (f"   ⚠ no new output in {int(since_last)}s -- may be stuck"
                         if warn else f"   (output {int(since_last)}s ago)")),
                fg=("#ffaa33" if warn else t["fg_dim"]),
            )
            win.after(500, poll_queue)

        poll_queue()

    # ── ENGINE ────────────────────────────────────────────────────────────────

    def _init_engine(self):
        self.engine.load(self.index_dir, self.model_path, self._set_status)

    # ── STATUS / STATS ────────────────────────────────────────────────────────

    def _set_status(self, msg):
        try:
            self.root.after(0, lambda: self.status_var.set(f"  {msg}"))
        except: pass

    def _update_stats(self):
        n = len(self.engine.metadata)
        if n:
            self.stats_lbl.config(
                text=f"{n:,} chunks\n{len(self.scan_dirs)} dir(s)\nModel: {Path(self.model_path).name}"
            )

    # ── SCROLL ────────────────────────────────────────────────────────────────

    def _on_results_configure(self, event):
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.results_canvas.itemconfig(self.results_window, width=event.width)

    def _on_mousewheel(self, event):
        self.results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    # ── CONFIG PERSIST ────────────────────────────────────────────────────────

    def _config_path(self):
        return Path.home() / ".trit-search" / "config.json"

    def _save_config(self):
        cfg = {"theme": self.theme_name, "scan_dirs": self.scan_dirs}
        self._config_path().parent.mkdir(parents=True, exist_ok=True)
        self._config_path().write_text(json.dumps(cfg))

    def _load_config(self):
        try:
            cfg = json.loads(self._config_path().read_text())
            self.scan_dirs = cfg.get("scan_dirs", [])
            theme = cfg.get("theme", "Matrix")
            if theme in THEMES:
                self.theme_name = theme
                self.t = THEMES[theme]
            self._restore_dirs()
        except: pass


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    app  = TritSearchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
