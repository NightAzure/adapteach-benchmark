"""
Download CS1QA, MBPP, StaQC/SO, and CoNaLa datasets and convert to the AdapTeach
query format: {"id": "...", "category": "...", "topic": "...", "query": "..."}

Outputs:
  bench/queries_cs1qa.jsonl     -- from CS1QA (NAACL 2022, real student questions)
  bench/queries_mbpp.jsonl      -- from MBPP (beginner Python problems)
  bench/queries_staqc.jsonl     -- from Stack Overflow Python questions (conceptual)
  bench/queries_conala.jsonl    -- from CoNaLa (StackOverflow NL→code intents)
  bench/queries_external.jsonl  -- all combined and deduplicated

Usage:
  py bench/fetch_external_queries.py
  py bench/fetch_external_queries.py --sources cs1qa mbpp staqc
  py bench/fetch_external_queries.py --max-per-source 200 --topics loops conditionals
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ── Quality filters ───────────────────────────────────────────────────────────
# CS1QA is from a specific Python CS1 course using Hubo robots.
# Many questions are course/assignment-specific, not general Python concepts.
# These patterns indicate low-value queries we should skip.
_NOISE_PATTERNS = [
    r"\bhubo\b", r"\btask\s*\d", r"\blab\s*\d", r"\bhw\s*\d", r"\bassignment\b",
    r"\bgrader\b", r"\bsubmit\b", r"\bsubmission\b", r"\bpiazza\b", r"\bcanvas\b",
    r"\bprofessor\b", r"\bta\b", r"\bcourse\b", r"\bgrade\b", r"\bdue\b",
    r"thank(s| you)", r"^(yes|no|ok|okay|sure|got it|i see|i understand)",
    r"^\w{1,15}$",  # single word / very short
]
import re as _re
_NOISE_RE = _re.compile("|".join(_NOISE_PATTERNS), _re.IGNORECASE)

# CS1QA: only keep questions that start with a recognized question opener —
# these are genuine conceptual questions, not context-dependent chat replies
_QUESTION_OPENERS = _re.compile(
    r"^(how|what|why|when|where|which|can|could|should|do|does|did|is|are|was|were|will|would)\b",
    _re.IGNORECASE,
)

# Must contain at least one Python programming term to pass quality gate
_PYTHON_TERMS = [
    "variable", "assign", "loop", "for ", "while", "if ", "elif", "else",
    "function", "def ", "return", "print", "import", "class", "list", "dict",
    "string", "int", "bool", "float", "type", "error", "exception", "index",
    "parameter", "argument", "scope", "range", "iterate", "condition",
    "recursion", "lambda", "python", "code", "syntax", "indent", "bracket",
]


def _is_quality_query(text: str, require_question_opener: bool = False) -> bool:
    """Return True if text looks like a genuine Python programming question."""
    if len(text) < 25:
        return False
    if _NOISE_RE.search(text):
        return False
    lower = text.lower()
    if not any(term in lower for term in _PYTHON_TERMS):
        return False
    if require_question_opener and not _QUESTION_OPENERS.match(text.strip()):
        return False
    return True


# ── Topic classification ──────────────────────────────────────────────────────
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "variables": [
        "variable", "assign", "assignment", "scope", "name", "binding",
        "global", "local", "nonlocal", "unbound", "namespace", "type",
        "integer", "string", "float", "boolean", "mutable", "immutable",
        "aliasing", "reference", "object", "value", "rebind",
    ],
    "loops": [
        "loop", "for loop", "while loop", "while", "iterate", "iteration",
        "range", "break", "continue", "enumerate", "repeat", "infinite",
        "nested loop", "for each", "foreach", "traverse", "counter",
    ],
    "conditionals": [
        "if", "elif", "else", "condition", "conditional", "boolean",
        "true", "false", "compare", "comparison", "branch", "ternary",
        "match", "switch", "guard", "truthy", "falsy", "is none",
    ],
    "functions": [
        "function", "def ", "return", "parameter", "argument", "call",
        "lambda", "recursive", "recursion", "closure", "decorator",
        "generator", "yield", "default argument", "keyword argument",
        "positional", "built-in", "method", "signature",
    ],
}

CORE_TOPICS = list(TOPIC_KEYWORDS.keys())


def _classify_topic(text: str) -> str | None:
    """Return the most likely topic label, or None if no clear match."""
    lower = text.lower()
    scores: dict[str, int] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score:
            scores[topic] = score
    if not scores:
        return None
    return max(scores, key=lambda t: scores[t])


def _classify_category(text: str, source_category: str | None = None) -> str:
    """Map raw category labels to one of: concept_explanation, debugging, misconception."""
    if source_category:
        cat = source_category.lower()
        if any(k in cat for k in ("debug", "error", "fix", "bug", "wrong")):
            return "debugging"
        if any(k in cat for k in ("misconception", "misunderstanding", "correct", "myth")):
            return "misconception"
        if any(k in cat for k in ("concept", "explain", "what", "how", "describe")):
            return "concept_explanation"
    lower = text.lower()
    if any(k in lower for k in ("bug", "error", "fix", "wrong", "broken", "problem", "traceback", "exception")):
        return "debugging"
    if any(k in lower for k in ("misconception", "incorrect", "wrong belief", "correct this", "actually")):
        return "misconception"
    return "concept_explanation"


def _stable_id(prefix: str, text: str) -> str:
    return f"{prefix}-{hashlib.md5(text.encode()).hexdigest()[:10]}"


def _fetch_json(url: str) -> list | dict:
    req = urllib.request.Request(url, headers={"User-Agent": "AdapTeach-Fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "AdapTeach-Fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


# ── CS1QA ─────────────────────────────────────────────────────────────────────
CS1QA_SPLITS = [
    "train_cleaned",
    "dev_cleaned",
    "test_cleaned",
]
CS1QA_BASE = "https://raw.githubusercontent.com/cyoon47/CS1QA/main/data/final/cleaned/equal"

# CS1QA questionType values → our category
CS1QA_CATEGORY_MAP = {
    "error":             "debugging",
    "logical":           "debugging",
    "variable":          "concept_explanation",
    "code_explain":      "concept_explanation",
    "code_understanding":"concept_explanation",
    "usage":             "concept_explanation",
    "reasoning":         "concept_explanation",
    "algorithm":         "concept_explanation",
    "task":              "concept_explanation",
}


def fetch_cs1qa(max_items: int, topics_filter: list[str]) -> list[dict]:
    all_items = []
    for split in CS1QA_SPLITS:
        url = f"{CS1QA_BASE}/{split}.jsonl"
        try:
            raw = _fetch_text(url)
            items = [json.loads(l) for l in raw.strip().splitlines() if l.strip()]
            all_items.extend(items)
            print(f"  [cs1qa] {split}: {len(items)} items")
        except Exception as exc:
            print(f"  [cs1qa] {url} failed: {exc}")

    if not all_items:
        print("  [cs1qa] All downloads failed. Skipping.")
        return []

    rows = []
    seen: set[str] = set()
    for item in all_items:
        question = (item.get("question") or "").strip()
        if not question or not _is_quality_query(question, require_question_opener=True):
            continue

        topic = _classify_topic(question)
        if not topic:
            continue
        if topics_filter and topic not in topics_filter:
            continue

        qtype = str(item.get("questionType", "")).lower()
        category = CS1QA_CATEGORY_MAP.get(qtype, _classify_category(question))

        qid = _stable_id("cs1qa", question)
        if qid in seen:
            continue
        seen.add(qid)

        rows.append({"id": qid, "category": category, "topic": topic, "query": question})
        if len(rows) >= max_items:
            break

    print(f"  [cs1qa] Converted {len(rows)} queries.")
    return rows


# ── MBPP ─────────────────────────────────────────────────────────────────────
MBPP_URLS = [
    "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
]


def fetch_mbpp(max_items: int, topics_filter: list[str]) -> list[dict]:
    raw = None
    for url in MBPP_URLS:
        try:
            raw = _fetch_text(url)
            print(f"  [mbpp] Downloaded from {url}")
            break
        except Exception as exc:
            print(f"  [mbpp] {url} failed: {exc}")

    if raw is None:
        print("  [mbpp] Download failed. Trying HuggingFace parquet fallback...")
        # Fall back to HuggingFace datasets library if available
        try:
            import datasets as hf_datasets  # type: ignore
            ds = hf_datasets.load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
            raw_items = [{"text": row["text"], "task_id": row["task_id"]} for row in ds]
        except Exception as exc2:
            print(f"  [mbpp] HuggingFace fallback also failed: {exc2}")
            return []
        items = raw_items
    else:
        items = [json.loads(line) for line in raw.strip().splitlines() if line.strip()]

    rows = []
    seen: set[str] = set()
    for item in items:
        text = (item.get("text") or item.get("prompt") or "").strip()
        if not text or len(text) < 10:
            continue
        # MBPP descriptions are task instructions — wrap as a concept_explanation query
        query = text if text.endswith("?") else f"Write a Python function to: {text}"

        topic = _classify_topic(query)
        if not topic:
            # MBPP problems often don't mention the concept explicitly; skip unclassifiable
            continue
        if topics_filter and topic not in topics_filter:
            continue

        qid = _stable_id("mbpp", text)
        if qid in seen:
            continue
        seen.add(qid)

        rows.append({
            "id": qid,
            "category": "concept_explanation",
            "topic": topic,
            "query": query,
        })
        if len(rows) >= max_items:
            break

    print(f"  [mbpp] Converted {len(rows)} queries.")
    return rows


# ── StaQC / Stack Overflow Python ────────────────────────────────────────────
# Uses the HuggingFace `pacovaldez/stackoverflow-questions` dataset (streamed).
# We keep only Python-tagged questions that start with a question opener and
# mention at least one Python concept term — these are genuine "how does X
# work" conceptual questions, not code-write requests.

_SO_QUESTION_RE = _re.compile(
    r"^(how|what|why|when|which|can|is|are|does|do|should|could|will|would)\b",
    _re.IGNORECASE,
)
_CODE_WRITE_RE = _re.compile(
    r"\b(write|create|implement|build|make|generate|produce|output|develop)\b",
    _re.IGNORECASE,
)


def _is_so_conceptual(title: str) -> bool:
    """True if the SO question title looks like a conceptual Python question."""
    if len(title) < 20:
        return False
    if not _QUESTION_OPENERS.match(title.strip()):
        return False
    if _CODE_WRITE_RE.search(title):
        return False
    lower = title.lower()
    if not any(term in lower for term in _PYTHON_TERMS):
        return False
    return True


def fetch_staqc(max_items: int, topics_filter: list[str]) -> list[dict]:
    """Fetch conceptual Python questions from Stack Overflow via HuggingFace."""
    try:
        import datasets as hf_datasets  # type: ignore
    except ImportError:
        print("  [staqc] HuggingFace `datasets` not installed. Run: pip install datasets")
        return []

    rows: list[dict] = []
    seen: set[str] = set()

    # koutch/stackoverflow_python: all questions are Python-tagged (pre-filtered).
    # Fields: title, question_id, question_body, question_score, tags
    # One row per answer → deduplicate by question_id.
    # pacovaldez/stackoverflow-questions: all SO questions (not Python-only),
    # needs tag filtering.
    candidates = [
        {
            "name":       "koutch/stackoverflow_python",
            "split":      "train",
            "text_col":   "title",
            "id_col":     "question_id",
            "tags_col":   "tags",
            "tag_filter": None,   # already Python-only
        },
        {
            "name":       "pacovaldez/stackoverflow-questions",
            "split":      "train",
            "text_col":   "title",
            "id_col":     None,
            "tags_col":   "tags",
            "tag_filter": "python",
        },
    ]

    for cand in candidates:
        if len(rows) >= max_items:
            break
        ds_name    = cand["name"]
        split      = cand["split"]
        text_col   = cand["text_col"]
        id_col     = cand["id_col"]
        tags_col   = cand["tags_col"]
        tag_filter = cand["tag_filter"]
        try:
            print(f"  [staqc] Trying {ds_name} (streaming)...")
            ds = hf_datasets.load_dataset(ds_name, split=split, streaming=True)
            seen_qids: set[str] = set()
            for item in ds:
                if len(rows) >= max_items:
                    break
                title = (item.get(text_col) or "").strip()
                if not title:
                    continue
                # Deduplicate by question_id when available (avoids multi-answer rows)
                if id_col:
                    raw_qid = str(item.get(id_col, ""))
                    if raw_qid in seen_qids:
                        continue
                    seen_qids.add(raw_qid)
                # Tag filter (only needed for non-Python-specific datasets)
                if tag_filter:
                    tags = str(item.get(tags_col, "")).lower()
                    if tag_filter not in tags:
                        continue
                if not _is_so_conceptual(title):
                    continue
                topic = _classify_topic(title)
                if not topic:
                    continue
                if topics_filter and topic not in topics_filter:
                    continue
                qid = _stable_id("staqc", title)
                if qid in seen:
                    continue
                seen.add(qid)
                rows.append({
                    "id": qid,
                    "category": _classify_category(title),
                    "topic": topic,
                    "query": title,
                })
            if rows:
                print(f"  [staqc] Got {len(rows)} queries from {ds_name}")
                break
        except Exception as exc:
            print(f"  [staqc] {ds_name} failed: {exc}")

    if not rows:
        print("  [staqc] All sources failed. Skipping.")
    else:
        print(f"  [staqc] Converted {len(rows)} queries.")
    return rows


# ── CoNaLa ───────────────────────────────────────────────────────────────────
CONALA_URLS = [
    "https://raw.githubusercontent.com/conala-corpus/conala-benchmark/master/conala-train.json",
    "https://raw.githubusercontent.com/conala-corpus/conala-benchmark/master/conala-test.json",
]
CONALA_FALLBACK_URLS = [
    "https://conala-corpus.github.io/conala-train.json",
    "https://conala-corpus.github.io/conala-test.json",
]


def fetch_conala(max_items: int, topics_filter: list[str]) -> list[dict]:
    all_items = []
    for url_group in [CONALA_URLS, CONALA_FALLBACK_URLS]:
        if all_items:
            break
        for url in url_group:
            try:
                data = _fetch_json(url)
                if isinstance(data, list):
                    all_items.extend(data)
                print(f"  [conala] Downloaded {len(data)} items from {url}")
            except Exception as exc:
                print(f"  [conala] {url} failed: {exc}")

    if not all_items:
        print("  [conala] Trying HuggingFace datasets library...")
        try:
            import datasets as hf_datasets  # type: ignore
            for split in ("train", "test"):
                ds = hf_datasets.load_dataset("neulab/conala", split=split, trust_remote_code=True)
                all_items.extend(list(ds))
            print(f"  [conala] HuggingFace: {len(all_items)} total items")
        except Exception as exc:
            print(f"  [conala] HuggingFace fallback failed: {exc}")
            return []

    rows = []
    seen: set[str] = set()
    for item in all_items:
        # Prefer rewritten_intent (cleaner), fall back to intent
        intent = (
            item.get("rewritten_intent")
            or item.get("intent")
            or ""
        ).strip()
        if not intent or len(intent) < 8:
            continue

        # Make it a proper question if it isn't already
        if not intent.endswith("?"):
            query = f"How do I {intent[0].lower()}{intent[1:]}?" if not intent.lower().startswith("how") else intent
        else:
            query = intent

        topic = _classify_topic(query)
        if not topic:
            continue
        if topics_filter and topic not in topics_filter:
            continue

        qid = _stable_id("conala", intent)
        if qid in seen:
            continue
        seen.add(qid)

        rows.append({
            "id": qid,
            "category": _classify_category(query),
            "topic": topic,
            "query": query,
        })
        if len(rows) >= max_items:
            break

    print(f"  [conala] Converted {len(rows)} queries.")
    return rows


# ── Writer ────────────────────────────────────────────────────────────────────
def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows)} rows -> {path}")


def topic_stats(rows: list[dict]) -> str:
    from collections import Counter
    t = Counter(r["topic"] for r in rows)
    c = Counter(r["category"] for r in rows)
    return f"topics={dict(t)}  categories={dict(c)}"


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Download external datasets and convert to AdapTeach query format")
    parser.add_argument("--sources", nargs="+", default=["cs1qa", "mbpp", "staqc"],
                        choices=["cs1qa", "mbpp", "staqc", "conala"],
                        help="Which datasets to download (default: cs1qa + mbpp + staqc; conala currently unavailable upstream)")
    parser.add_argument("--max-per-source", type=int, default=500,
                        help="Max queries per source (default: 500)")
    parser.add_argument("--topics", nargs="+", default=[],
                        choices=CORE_TOPICS,
                        help="Only keep queries matching these topics (default: all 4)")
    parser.add_argument("--out-dir", default="bench",
                        help="Output directory (default: bench/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    topics_filter = args.topics  # empty = no filter

    fetchers = {
        "cs1qa":  fetch_cs1qa,
        "mbpp":   fetch_mbpp,
        "staqc":  fetch_staqc,
        "conala": fetch_conala,
    }

    all_rows: list[dict] = []
    seen_queries: set[str] = set()

    for source in args.sources:
        print(f"\n[{source.upper()}] Fetching...")
        rows = fetchers[source](args.max_per_source, topics_filter)
        if rows:
            write_jsonl(rows, out_dir / f"queries_{source}.jsonl")
            print(f"  Stats: {topic_stats(rows)}")
        # Merge into combined, deduplicating by query text
        for row in rows:
            norm = re.sub(r"\s+", " ", row["query"].lower().strip())
            if norm not in seen_queries:
                seen_queries.add(norm)
                all_rows.append(row)

    if all_rows:
        combined_path = out_dir / "queries_external.jsonl"
        write_jsonl(all_rows, combined_path)
        print(f"\n[COMBINED] {len(all_rows)} unique queries")
        print(f"  Stats: {topic_stats(all_rows)}")
        print(f"\nRun benchmark with:")
        print(f"  py bench/run_benchmark_suite.py --query-set {combined_path} --configs A,B,C,D,E,F --dry-run retrieval --out-dir bench/runs/")
    else:
        print("\nNo queries collected. Check network access and try again.")


if __name__ == "__main__":
    main()
