# OBSERVE — local semantic code search, with receipts

OBSERVE is a desktop semantic code search tool that runs entirely on your
machine: a fine-tuned embedding model, a ternary-compressed index, a GUI,
a CLI, and an MCP server that plugs the search into Claude Code, Claude
Desktop, Cursor, or any MCP-compatible AI assistant.

Nothing leaves your computer. No API keys, no cloud indexing, no
telemetry. Your code is embedded locally, stored locally, searched
locally.

## Why it exists (the measured version)

Every claim below is from a written-up, reproducible benchmark in this
repo — including the ones that are unflattering:

| Claim | Number | Evidence |
|---|---|---|
| Fine-tuned model beats the stock baseline on code search | 96% vs 92% (hard benchmark), 92% vs 76.7% (real OSS code) | [`trit_benchmark.py`](trit_benchmark.py), [`trit_oss_test.py`](trit_oss_test.py) |
| Index is small | ~20× compressed on disk (58k chunks ≈ 4.5 MB) | ternary quantization, [`paper/012_paper.md`](paper/012_paper.md) |
| `query_codebase` saves tokens for AI assistants | **66.3% fewer tokens** than the plain search tool on the same queries | [`paper/token_reduction_findings.md`](paper/token_reduction_findings.md) |
| …and the savings aren't free | dedup/cutoff can drop correct answers; root-caused and fixed with function-boundary chunking (chunk recall 29% → 86%) | [`paper/quality_benchmark_findings.md`](paper/quality_benchmark_findings.md) |
| **Grep beats it when you know the identifier** | Grep 5/5 vs 3/5 on exact-name queries | [`paper/grep_vs_semantic_findings.md`](paper/grep_vs_semantic_findings.md) |
| Where it actually wins | vocabulary mismatch: legacy renames, domain jargon, concept-only queries — one call instead of several exploratory greps, at ~90% fewer tokens | same doc |

**The honest scope in one sentence: if you can *name* the thing, grep for
it; if you can only *describe* it, OBSERVE finds it.** The MCP tool
descriptions say exactly this to the AI assistant, so it routes queries
correctly on its own.

## Install + demo, one line

```bash
pip install "git+https://github.com/tritsystem/012-trit-search.git@project" && observe-demo
```

That installs OBSERVE, indexes the directory you ran it from into a
throwaway demo index (your real index is never touched), runs three
natural-language searches against it, and prints the setup commands for
the GUI and the Claude Code MCP integration. Point it somewhere specific
with `observe-demo /path/to/repo`.

(Most of the install time is `torch` + `sentence-transformers`; the first
demo run also downloads the embedding model if the fine-tuned one isn't
present. GPU is optional — if your torch build sees CUDA, embedding and
search use it automatically.)

Installed commands: `observe` (GUI) · `observe-search` (CLI) ·
`observe-mcp` (MCP server) · `observe-demo`.

Working from a clone instead:

```bash
pip install -r requirements_app.txt     # sentence-transformers, faiss-cpu, flask, torch, mcp, tiktoken
```

## Use

**Desktop app (index + search):**
```bash
python trit_app.py        # click INDEX CODEBASE, point it at your repo(s), then search
```

**CLI / HTTP API:**
```bash
python trit_search.py --index /path/to/repo    # build/update an index
python trit_search.py "where do we retry failed uploads"
```

**MCP server (Claude Code, Claude Desktop, Cursor, …):** build an index
first (the server reads it, it doesn't build), then register:

```bash
claude mcp add observe -- python C:/path/to/012-ternary/trit_mcp_server.py
```

or in a `.mcp.json` / client config:

```json
{
  "mcpServers": {
    "observe": {
      "command": "python",
      "args": ["C:/path/to/012-ternary/trit_mcp_server.py"]
    }
  }
}
```

The server exposes **three stable tools**:

- `search_code(query, k, project_dir)` — full semantic results with scores
  and previews;
- `query_codebase(query, k, project_dir)` — the token-tight variant
  (deduplicated, relevance-cutoff, compact formatting — the measured
  66.3% saving);
- `index_status()` — chunk count, model, index health.

## Stable vs experimental

The three tools above are the product; they're benchmarked, and their
failure modes are documented. The repo also contains experimental tools —
`propose_change`, `apply_and_verify`, the cross-project entanglement
family (`list_indexed_projects`, `get_project_summary`,
`get_entanglement`), and `hybrid_search_code` (real SQL structured
filtering + optional 1-hop call/import graph expansion over a
provenance/lineage layer -- see `chunk_provenance.py`; spot-checked
against known real relationships, not yet benchmarked the way
`query_codebase` was, so it stays experimental until it is).
**These are unstable and not part of the supported surface.** They stay unregistered unless you explicitly opt in:

```bash
OBSERVE_EXPERIMENTAL=1 python trit_mcp_server.py
```

If you don't set that variable, they don't exist as far as your MCP
client is concerned.

## How it works, briefly

Code is split at function boundaries (not blind character windows — that
was a measured recall bug, see the quality benchmark), embedded with a
MiniLM fine-tuned on code (`models/code-minilm`, with automatic fallback
to stock `all-MiniLM-L6-v2` if the fine-tuned weights aren't present),
ternary-quantized for a ~20× smaller index, and searched with FAISS. The
GUI, CLI, and MCP server share one `SearchEngine`, one model, one index
(`~/.trit-search/index`).

## License

MIT, same as the repo — see [LICENSE](LICENSE).
