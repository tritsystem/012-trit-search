# Contributing

Thanks for looking. This is a research monorepo with one shipped product
(**OBSERVE**) inside it, maintained by one person. Contributions are welcome
within that reality.

## Ground rules

1. **Claims are backed by a script that runs.** If you add or change a
   measured result, include the code that produces it and say what hardware /
   dataset you ran it on. No estimated numbers.
2. **Negative results are results.** A change that *doesn't* help is worth a PR
   if it's measured and written up honestly.
3. **Keep the product modules clean.** The files that ship as OBSERVE are listed
   in `pyproject.toml` under `py-modules`. Changes there get more scrutiny than
   changes to the research scripts.
4. **Disclose AI assistance** if you used it — a line in the PR description is
   enough. See [AI_DISCLOSURE.md](AI_DISCLOSURE.md) for how the maintainer does it.

## Setup

```bash
git clone https://github.com/tritsystem/012-trit-search
cd 012-trit-search
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before opening a PR

```bash
python -m pytest -q                # tests that exist run green
python -m ruff check .             # lint (or: pip install ruff first)
```

- One logical change per PR.
- Explain what you measured, not just what you changed.
- If it touches OBSERVE's search behaviour, run `observe-search` against a real
  repo and paste the before/after.

## Reporting bugs

Open an issue with: OS, Python version, what you ran, what happened, what you
expected. For OBSERVE search-quality issues, include the query and the repo (or a
minimal repo that reproduces it).

## Security

Don't open a public issue — see [SECURITY.md](SECURITY.md).
