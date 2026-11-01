"""Documentation checks: relative links + anchors, Mermaid structure, tables, disclaimer."""

import re
from pathlib import Path

import pytest

from rehabpose import DISCLAIMER

ROOT = Path(__file__).resolve().parents[2]
MD_FILES = sorted(
    [*ROOT.glob("docs/**/*.md"), ROOT / "README.md", ROOT / "CLAUDE.md", ROOT / "PROGRESS.md"]
)
REQUIRED_DOCS = [
    "01_overview.md",
    "02_srs.md",
    "03_architecture.md",
    "04_exercise_spec.md",
    "05_data_plan.md",
    "06_ml_eval_plan.md",
    "07_testing_plan.md",
    "08_deployment_plan.md",
    "09_ethics_privacy.md",
    "10_model_card.md",
    "11_datasheet.md",
    "adr/0001-landmarks-over-websocket.md",
]
MERMAID_TYPES = ("flowchart", "graph", "sequenceDiagram", "erDiagram", "stateDiagram-v2")

FENCE_RE = re.compile(r"^```(\w[\w-]*)?\s*\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(ROOT)) for p in paths]


def _strip_code(text: str) -> str:
    text = FENCE_RE.sub("", text)
    return re.sub(r"`[^`\n]*`", "", text)


def _slug(heading: str) -> str:
    """GitHub-style heading anchor."""
    s = heading.strip().lower().replace("`", "")
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    text = FENCE_RE.sub("", path.read_text())
    seen: dict[str, int] = {}
    out: set[str] = set()
    for h in HEADING_RE.findall(text):
        base = _slug(h)
        n = seen.get(base, 0)
        out.add(base if n == 0 else f"{base}-{n}")
        seen[base] = n + 1
    return out


def _mermaid_blocks() -> list[tuple[Path, str]]:
    return [
        (p, m.group(2))
        for p in MD_FILES
        for m in FENCE_RE.finditer(p.read_text())
        if m.group(1) == "mermaid"
    ]


@pytest.mark.parametrize("name", REQUIRED_DOCS)
def test_required_docs_exist(name: str) -> None:
    assert (ROOT / "docs" / name).is_file()


@pytest.mark.parametrize("path", MD_FILES, ids=_ids(MD_FILES))
def test_relative_links_resolve(path: Path) -> None:
    broken = []
    for target in LINK_RE.findall(_strip_code(path.read_text())):
        if re.match(r"^[a-z]+:", target):
            continue
        file_part, _, anchor = target.partition("#")
        dest = (path.parent / file_part).resolve() if file_part else path
        if not dest.exists():
            broken.append(f"{target} (missing file)")
        elif anchor and dest.suffix == ".md" and anchor not in _anchors(dest):
            broken.append(f"{target} (missing anchor)")
    assert not broken, broken


@pytest.mark.parametrize("path", MD_FILES, ids=_ids(MD_FILES))
def test_table_rows_match_header(path: Path) -> None:
    lines = FENCE_RE.sub("", path.read_text()).splitlines()
    bad = []
    width: int | None = None
    for i, line in enumerate(lines, 1):
        if not line.startswith("|"):
            width = None
            continue
        cells = len(re.split(r"(?<!\\)\|", line.strip().strip("|")))
        if width is None:
            width = cells
        elif cells != width:
            bad.append(f"line {i}: {cells} cells, header has {width}")
    assert not bad, bad


@pytest.mark.parametrize(
    ("path", "block"), _mermaid_blocks(), ids=lambda v: v.name if isinstance(v, Path) else ""
)
def test_mermaid_block_structure(path: Path, block: str) -> None:
    lines = [ln for ln in block.splitlines() if ln.strip() and not ln.strip().startswith("%%")]
    kind = lines[0].split()[0]
    assert kind in MERMAID_TYPES, f"unknown diagram type {kind!r}"
    assert "\t" not in block
    for ln in lines:
        assert ln.count('"') % 2 == 0, f"unbalanced quotes: {ln}"
    if kind == "erDiagram":
        # Entity bodies span lines; cardinality markers such as ||--o{ are not brackets.
        bare = re.sub(r'"[^"]*"', "", block)
        bare = re.sub(r"[|o}{]{2}(--|\.\.)[|o}{]{2}", "", bare)
        assert bare.count("{") == bare.count("}"), "unbalanced entity braces"
    else:
        for ln in lines:
            bare = re.sub(r'"[^"]*"', "", ln)
            for open_, close in ("[]", "()", "{}"):
                assert bare.count(open_) == bare.count(close), f"unbalanced {open_}{close}: {ln}"
    if kind == "sequenceDiagram":
        declared = set(re.findall(r"^\s*(?:participant|actor)\s+(\w+)", block, re.MULTILINE))
        used = set(re.findall(r"^\s*(\w+)\s*-{1,2}>>?\+?-?\s*(\w+)\s*:", block, re.MULTILINE))
        undeclared = {a for pair in used for a in pair} - declared
        assert not undeclared, f"undeclared participants {undeclared}"
    if kind == "flowchart":
        assert block.count("subgraph") == len(re.findall(r"^\s*end\s*$", block, re.MULTILINE))


def test_mermaid_blocks_present() -> None:
    kinds = {b.split()[0] for _, b in _mermaid_blocks()}
    assert {"flowchart", "sequenceDiagram", "erDiagram"} <= kinds


@pytest.mark.parametrize(
    "name", ["README.md", "docs/09_ethics_privacy.md", "docs/10_model_card.md"]
)
def test_disclaimer_verbatim(name: str) -> None:
    text = (ROOT / name).read_text().replace(">", " ").replace("**Disclaimer:**", " ")
    assert " ".join(DISCLAIMER.split()) in " ".join(text.split())
