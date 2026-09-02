#!/usr/bin/env bash
# OBSERVE - one-line installer (macOS / Linux)
#
#   curl -fsSL https://raw.githubusercontent.com/tritsystem/012-trit-search/main/install.sh | bash
#
# Installs into an isolated venv (~/.observe) so it never touches your system
# Python, then puts `observe`, `observe-search`, and `observe-mcp` on your PATH.
# Nothing leaves your machine; the only downloads are PyPI packages + the model.
set -euo pipefail

REPO="https://github.com/tritsystem/012-trit-search.git"
INSTALL_DIR="${OBSERVE_HOME:-$HOME/.observe}"
VENV="$INSTALL_DIR/venv"
BIN_DIR="$HOME/.local/bin"

say() { printf '  %s\n' "$*"; }
echo ""
echo "  OBSERVE - local semantic code search"
echo "  ===================================="

# 1. Find a Python >= 3.10 ----------------------------------------------------
PY=""
for c in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)' 2>/dev/null; then
        PY="$c"; break
    fi
done
if [ -z "$PY" ]; then
    echo "  ERROR: Python 3.10+ is required but was not found."
    echo "         Install it from https://python.org and re-run."
    exit 1
fi
say "Python : $($PY --version 2>&1)"
say "Install: $INSTALL_DIR"

# 2. Isolated virtual environment --------------------------------------------
mkdir -p "$INSTALL_DIR"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
PIP="$VENV/bin/pip"

# 3. Dependencies: CPU-only torch FIRST (keeps the install ~180MB not ~2GB) ---
say "Upgrading pip..."
"$PIP" install --upgrade --quiet pip
say "Installing CPU-only PyTorch (~180MB, one time)..."
"$PIP" install --quiet torch --index-url https://download.pytorch.org/whl/cpu
say "Installing OBSERVE + dependencies..."
"$PIP" install --quiet "git+$REPO"

# 4. Expose the commands on PATH ---------------------------------------------
mkdir -p "$BIN_DIR"
for cmd in observe observe-search observe-mcp observe-demo; do
    [ -f "$VENV/bin/$cmd" ] && ln -sf "$VENV/bin/$cmd" "$BIN_DIR/$cmd"
done

echo ""
echo "  Done. Commands installed:"
echo "     observe          desktop GUI  (needs Tk: 'apt install python3-tk' on bare Linux)"
echo "     observe-search   command-line search"
echo "     observe-mcp      MCP server for editors/agents"
echo ""
case ":$PATH:" in
    *":$BIN_DIR:"*) echo "  Try:  observe-search --help" ;;
    *) echo "  Add this to your shell profile, then restart your terminal:"
       echo "      export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
echo ""
