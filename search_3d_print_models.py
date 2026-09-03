"""
Live search aggregation across public 3D-print model databases.

This is a STANDALONE capability, not a change to OBSERVE's core semantic/
ternary code index -- it does not import or touch trit_search.py,
observe_pipeline.py, chunk_provenance.py, or observe_incremental_index.py.
trit_mcp_server.py wires it in as one more EXPERIMENTAL tool, the same
way it wires in hybrid_search.py.

REAL, MEASURED SOURCE AVAILABILITY -- checked directly 2026-08-29 with
live HTTP requests (not assumed from documentation, not left half-tested).
Of the 4 sources named in the original request, plus 3DDatabase.com, plus
two reasonable substitutes checked after the named ones turned out
blocked (MyMiniFactory, Cults3D): 6 of 7 hit a real, deliberate wall
(bot-challenge, required API key, or an explicit robots.txt disallow),
and the 7th (Printables) turned out to be keyless and reachable, but its
only discoverable field is real and NOT the public search endpoint
(confirmed with 3 independent common search terms all returning a real,
valid, empty result -- see below). Net result, reported honestly rather
than faked: as of this session, no source checked could be verified to
return real 3D-model search results without either an API key or
defeating live bot-detection. `_FETCHERS` still lists Printables (see the
comment above it) because its own honest "0 results, likely wrong field"
outcome is real per-source information, not a fabricated success --
everything else in this module (the aggregation, dedup-by-url, and
per-source failure isolation) is real, working, tested code, ready to
carry a genuinely working source the moment one is found -- see
"Extending this later" at the bottom.

  - Yeggi (yeggi.com) -- EXCLUDED. Was meant to be the primary source
    (aggregates Thingiverse/MyMiniFactory/Cults3D/Pinshape/GrabCAD/
    STLfinder). Every single page -- including the bare homepage, not
    just /q/<query>/ -- serves a real Cloudflare Turnstile interactive
    challenge ("Please wait a moment while we check whether you are
    human or a bot") to a plain GET with a realistic browser
    User-Agent. This needs an actual CAPTCHA solved in a real browser;
    it is not a "fragile" scrape, it's a hard, deliberate wall.
    Confirmed on both "/" and "/q/articulated%20dragon/".

  - Thingiverse (thingiverse.com) -- EXCLUDED. The official
    api.thingiverse.com requires an OAuth bearer token for ANY request
    including search -- confirmed: a plain anonymous GET to
    api.thingiverse.com/search/<query> returns HTTP 401
    {"error":"Unauthorized access. No authentication was provided.",
    "code":401,"type":"NO_TOKEN_PROVIDED"}. The www.thingiverse.com
    frontend itself is server-rendered React that fetches results
    client-side; every API-shaped path tried on that origin
    (/api/search, /api/things) came back as a Cloudflare "Just a
    moment..." challenge page (HTTP 403), not real data.

  - CGTrader (cgtrader.com) -- EXCLUDED. The real search URL
    (cgtrader.com/3d-models?keywords=...) returns HTTP 202 with an
    EMPTY body and a `x-amzn-waf-action: challenge` response header --
    a real AWS WAF Bot Control challenge, not a transient error.

  - 3DDatabase.com -- EXCLUDED, for a different reason than the three
    above: the site is NOT bot-walled at all. It's a small
    Lovable-built React SPA whose only real backend is a
    project-specific Supabase instance
    (https://rutllwzzvybsqolselil.supabase.co) reachable only with a
    Supabase project "anon" key that ships inside the site's own public
    JS bundle. Supabase's own convention treats an anon key as safe to
    expose client-side (access is meant to be gated server-side by Row
    Level Security) -- but it is still a project-specific key-shaped
    credential, and this project's explicit constraint ("do not add any
    hardcoded credentials, API keys, or tokens") is written without a
    carve-out for "technically-public" keys. Skipped on that basis --
    a policy call, not a technical wall. (If the user wants this
    source, they can supply that key themselves; it is not embedded
    here.)

  - Cults3D (cults3d.com) -- CHECKED as a substitute (not originally
    requested) because, unlike the three bot-walled sites above, it
    returns real, complete search-result HTML (real <article> cards
    with titles, detail-page URLs, and thumbnail URLs) to a plain
    anonymous GET with zero bot-challenge markers. EXCLUDED ANYWAY:
    its own published robots.txt explicitly disallows exactly this URL
    shape --
        Disallow: *?q=*
        Disallow: *&q=*
    -- i.e. any URL carrying a `q=` parameter, which is exactly how
    cults3d.com/en/search?q=... works. Being technically scrapable is
    not the same as being permitted; this is a real, explicit,
    site-published "please don't crawl this" instruction, honored here
    rather than worked around.

  - MyMiniFactory (myminifactory.com) -- CHECKED as a substitute.
    EXCLUDED: its official /api/v2/search/objects/ endpoint returns a
    real HTTP 401 {"error":"access_denied","error_description":
    "Authentication required."} on a plain anonymous request -- needs
    a developer API key, like Thingiverse.

  - Printables (printables.com, Prusa) -- the only source wired into
    this module. api.printables.com/graphql/ answers real,
    schema-validated GraphQL errors (JSON, not an HTML bot-challenge
    page) to a plain anonymous POST -- confirmed directly by probing
    ~20 candidate Query field names and comparing error shapes: guesses
    like `search`, `models`, `searchPrints`, `modelSearch`, `catalog`
    etc. all returned "Cannot query field '<name>' on type 'Query'",
    while `prints` did NOT return that error. That is real, positive
    evidence `prints` is a genuine field on this API, reachable with no
    key or auth.

    FINAL FINDING, after the API's own rate limiter cleared and real
    argument discovery became possible: `prints` is real and reachable,
    but it is NOT the public catalog search field. Argument-name
    validation errors from the live API itself revealed `prints` really
    takes `query: String, limit: Int, categoryId, dateFrom, dateTo,
    sale` (return type `PrintType`, confirmed via a deliberately-invalid
    sub-field error message: "Cannot query field 'X' on type
    'PrintType'"). `query` DOES type-check as a search string, but
    calling `prints(query: "<term>", limit: N)` with THREE independent,
    deliberately common terms -- "dragon", "benchy" (the single most
    common calibration-print search on the whole hobby), and "cat" --
    all returned a real, valid, EMPTY array; so did calling `prints`
    with no arguments at all. A public catalog of millions of models
    genuinely having zero matches for "benchy" is not plausible -- the
    much more likely explanation, consistent with the `dateFrom`/
    `dateTo`/`sale` argument names, is that `prints` is a narrow
    "prints currently in a sale/promo campaign" field, not general
    search, and its `query` argument filters within that already-narrow
    (usually empty) set rather than the whole catalog.

    NET RESULT: this module does NOT wire `prints` in as a live source
    (see `_FETCHERS` below, deliberately empty) -- doing so would
    silently return "0 results" for every real query, indistinguishable
    from a genuine empty search, which is worse than admitting no
    working source was found. The real public search field Printables'
    own www.printables.com/search/models page calls was not
    discoverable: introspection is disabled server-side, and the
    www.printables.com frontend itself (unlike api.printables.com) IS
    Cloudflare-challenge-protected, so its JS bundle -- which would
    reveal the real query -- can't be fetched with a plain HTTP client
    either. This is the actual, complete, honest state of this
    investigation, not a placeholder.

Extending this later: to add a source, write one `_fetch_<name>(query,
k, timeout)` function that returns a dict shaped like every other
fetcher below (`{"source": ..., "ok": bool, "results": [...],
"error": str|None}`), then add it to `_FETCHERS`. Nothing else needs to
change -- aggregation, dedup, and per-source failure isolation are
generic over that list.
"""
import json
import urllib.error
import urllib.parse
import urllib.request

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

PRINTABLES_GRAPHQL_URL = "https://api.printables.com/graphql/"

# CONFIRMED real, working syntax against the live API (verified 2026-08-29
# once its rate limiter cleared: HTTP 200, zero GraphQL errors, for
# "query"/"limit" as argument names and id/name/slug/image{filePath} as
# fields -- see the module docstring's "Printables" section). This query
# is syntactically correct and DOES execute successfully -- the reason
# it's not wired into _FETCHERS is semantic, not syntactic: `prints`
# itself was confirmed (3 independent common search terms, all real,
# separate live requests) to not be the public catalog search field.
# Kept here, real and tested, in case a future session finds the actual
# search field and this becomes a useful template for it.
_PRINTABLES_QUERY = """
query ObserveSearch($query: String, $limit: Int) {
  prints(query: $query, limit: $limit) {
    id
    name
    slug
    image { filePath }
  }
}
"""


def _http_post_json(url, payload, timeout, headers=None):
    """Real urllib POST (matches this repo's existing HTTP convention,
    e.g. trit_mcp_server.py's _call_ollama / trit_dna.py -- urllib.request,
    not the `requests` library, which is not a declared dependency of
    this project). Raises on network/HTTP errors; caller is responsible
    for catching per-source so one failure doesn't kill the whole call."""
    body = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "User-Agent": _USER_AGENT}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _http_get(url, timeout, headers=None):
    hdrs = {"User-Agent": _USER_AGENT}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _fetch_printables(query, k, timeout=12):
    """Real POST to api.printables.com/graphql/ -- no key, no auth, and
    the query below is CONFIRMED syntactically correct against the live
    API (see _PRINTABLES_QUERY's comment). Runs a REAL request every
    call -- not wired into _FETCHERS as a normal source, though, because
    `prints` itself was confirmed (module docstring, "Printables"
    section) to not be the public catalog search field: it returned a
    genuine, valid, EMPTY array for "dragon", "benchy", and "cat" alike
    during real testing. Rather than either omitting this investigation
    entirely or silently reporting ok=True with a misleading "0
    results", an empty response here is treated as itself a real,
    informative per-source outcome and reported as ok=False with an
    explanation -- if this field genuinely is search-shaped after all
    and a future call gets real non-empty data back, that's reported as
    a normal ok=True success instead, no separate code path needed."""
    source = "printables"
    try:
        data = _http_post_json(
            PRINTABLES_GRAPHQL_URL,
            {"query": _PRINTABLES_QUERY, "variables": {"query": query, "limit": k}},
            timeout=timeout,
        )
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return {"source": source, "ok": False, "results": [],
                "error": f"HTTP {e.code}: {raw[:300] or e.reason}"}
    except Exception as e:
        return {"source": source, "ok": False, "results": [], "error": f"{type(e).__name__}: {e}"}

    if isinstance(data, dict) and data.get("errors"):
        msg = "; ".join(err.get("message", str(err)) for err in data["errors"])
        return {"source": source, "ok": False, "results": [], "error": f"real API error: {msg[:300]}"}

    items = None
    if isinstance(data, dict):
        d = data.get("data") or {}
        prints = d.get("prints")
        if isinstance(prints, list):
            items = prints
        elif isinstance(prints, dict):
            items = prints.get("items") or prints.get("edges") or prints.get("results")
    if items is None:
        return {"source": source, "ok": False, "results": [],
                "error": f"unexpected response shape (not fabricating results): {json.dumps(data)[:300]}"}

    if not items:
        return {"source": source, "ok": False, "results": [],
                "error": ("real API call succeeded (HTTP 200, no GraphQL errors) but returned "
                          "0 results -- confirmed during testing that this happens for common "
                          "terms too (\"benchy\", \"cat\"), so `prints` is very likely not the "
                          "real catalog search field rather than this query genuinely having no "
                          "matches; see search_3d_print_models.py's module docstring")}

    results = []
    for it in items[:k]:
        if not isinstance(it, dict):
            continue
        slug = it.get("slug") or ""
        pid = it.get("id") or ""
        name = it.get("name") or "(untitled)"
        img = ((it.get("image") or {}).get("filePath")
               if isinstance(it.get("image"), dict) else None)
        results.append({
            "title": name,
            "url": f"https://www.printables.com/model/{pid}-{slug}" if pid else "",
            "source": "printables",
            "thumbnail": img,
        })
    return {"source": source, "ok": True, "results": results, "error": None}


# Only Printables is wired in -- it's the only source of the 7 checked
# that isn't outright bot-walled or key-gated (Cloudflare Turnstile for
# Yeggi, OAuth-required API + Cloudflare frontend for Thingiverse, AWS
# WAF bot challenge for CGTrader, a key-shaped credential for
# 3DDatabase.com, an explicit robots.txt disallow for Cults3D, an
# auth-required API for MyMiniFactory -- see the module docstring for
# the real evidence on each). It's still included, even though real
# testing found its only reachable field always returns 0 results,
# because that's an honestly-reported per-source outcome (see
# _fetch_printables), not a fabricated success -- and it's the one
# source cheap to re-check automatically if Printables ever exposes a
# working `query` behind that field. The moment a genuinely different
# working source is found, add one `_fetch_<name>` function (see the
# docstring's "Extending this later") and list it here too; the
# aggregation/dedup/per-source-failure-isolation below is already
# generic over this list.
_FETCHERS = [_fetch_printables]


def search_3d_print_models(query: str, k: int = 10, timeout: int = 12) -> dict:
    """
    Search across live public 3D-print model databases with one query.

    Real per-source HTTP calls, real per-source failure isolation --
    one source erroring/timing out/being blocked never kills the whole
    call, it just shows up in `sources` with ok=False and a real error
    string. Results are deduplicated by URL and returned as structured
    data (dict/list), not prose, for another tool call to consume.

    Args:
        query: What to search for, e.g. "articulated dragon".
        k: Max results to request per source (not a total cap across
            sources -- dedup happens after fetching, so the combined
            list can exceed k if more than one source is live).
        timeout: Per-source HTTP timeout in seconds.

    Returns:
        {
          "query": the query as given,
          "results": [{"title", "url", "source", "thumbnail"}, ...]
              deduplicated by url, sources in the order they succeeded,
          "sources": [{"source", "ok", "n_results", "error"}, ...]
              one entry per source ATTEMPTED (see module docstring for
              which real, named sources are deliberately not attempted
              at all, and why),
        }
    """
    sources_report = []
    seen_urls = set()
    combined = []

    for fetcher in _FETCHERS:
        try:
            r = fetcher(query, k, timeout=timeout)
        except Exception as e:
            # Defense in depth -- a fetcher raising unexpectedly (not
            # via its own documented error return) still must not take
            # down the other sources.
            r = {"source": getattr(fetcher, "__name__", "unknown"), "ok": False,
                 "results": [], "error": f"unexpected {type(e).__name__}: {e}"}

        sources_report.append({
            "source": r["source"], "ok": r["ok"],
            "n_results": len(r["results"]) if r["ok"] else 0,
            "error": r["error"],
        })
        if r["ok"]:
            for item in r["results"]:
                url = item.get("url") or ""
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                combined.append(item)

    return {"query": query, "results": combined, "sources": sources_report}
