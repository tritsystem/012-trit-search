# OBSERVE MCP server (trit_mcp_server.py) — containerized runtime.
#
# DESIGN CHOICE: this image bakes in the Python interpreter + dependencies
# ONLY, not the source code. The actual 012-ternary source (this whole
# directory) is bind-mounted at /app at `docker run` time instead — this
# project is under heavy active development (many scripts changing daily),
# and baking source into the image would mean a rebuild on every edit.
# Rebuild this image only when requirements_app.txt itself changes.
#
# Uses requirements_app.txt (the documented "just the OBSERVE search app"
# subset) rather than the full requirements.txt, which also pulls in
# training/fine-tuning deps (peft, bitsandbytes, datasets, pyinstaller)
# this server never touches.

FROM python:3.12-slim

# libgomp1: faiss-cpu's compiled extension needs OpenMP at runtime on
# Debian-based images (a real, common faiss-on-slim failure otherwise —
# not a guess, this is faiss's own documented Linux dependency).
#
# python3-tk: CONFIRMED REAL, not precautionary — trit_mcp_server.py does
# `from trit_app import SearchEngine`, and trit_app.py imports `tkinter`
# at module level (it's normally a desktop GUI app; the MCP server only
# wants one shared class out of it). python:3.12-slim has no Tk system
# libs at all, so this import chain hard-crashes with
# "ImportError: libtk8.6.so: cannot open shared object file" before the
# server even starts — reproduced directly against this exact image.
# Nothing GUI-related actually RUNS in this container; this package only
# exists to satisfy that one import successfully.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with a predictable home directory. trit_mcp_server.py
# resolves its real index/config dir as `Path.home() / ".trit-search"`
# (see INDEX_DIR in trit_mcp_server.py) — HOME here must match the mount
# point used in the `docker run` command / MCP config so that resolves to
# the REAL index data mounted in from the host, not an empty container-
# local directory.
RUN useradd --create-home --home-dir /home/appuser --shell /bin/bash appuser
ENV HOME=/home/appuser
USER appuser
WORKDIR /app

# CPU-only torch, installed FIRST and pinned to the CPU wheel index (same
# technique requirements.txt's own header comment documents for a real
# CUDA install, mirrored here in reverse). requirements_app.txt lists bare
# "torch" with no index pin, which on a plain `pip install` here pulled
# the default CUDA build — ~2GB+ of nvidia-cublas/cudnn/cusolver wheels
# that this container can't use anyway (no GPU passthrough configured,
# and this workload is a small MiniLM embedding model for code search,
# not something that benefits from it). Installing the CPU wheel first
# means the later full-requirements install sees torch already satisfied
# (no version pin in that file) and leaves it alone.
# mcp PINNED to 1.28.1: CONFIRMED REAL, not precautionary — requirements_app.txt
# lists bare "mcp" with no version, which resolves to whatever's newest on
# install day. Reproduced directly: an unpinned install pulled mcp 2.1.0,
# which has REORGANIZED trit_mcp_server.py's exact import
# (`from mcp.server.fastmcp import FastMCP`) into a different path,
# crashing with ModuleNotFoundError before the server even starts. The
# real working native install (G:\trit312, confirmed via `pip show mcp`)
# is 1.28.1 — pinning to that exact version here, not "latest", is what
# actually matches the code this image runs.
COPY --chown=appuser:appuser requirements_app.txt /tmp/requirements_app.txt
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r /tmp/requirements_app.txt \
    && pip install --no-cache-dir --user "mcp==1.28.1"
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Source + data are bind-mounted here at runtime (see README-docker.md /
# the `docker run` command in the MCP config) — nothing COPYed in.

# MCP stdio transport: stdin/stdout must stay clean JSON-RPC. The server's
# own status/progress prints already go to stderr (confirmed by directly
# running it once during setup) — unbuffered stdout so the client sees
# output promptly rather than batched behind Python's default buffering.
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "trit_mcp_server.py"]
