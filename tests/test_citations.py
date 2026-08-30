import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_citation_keys_exist_and_expected_set_is_preserved():
    tex = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "paper/main/sections").glob("*.tex"))
    cited = set()
    for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", tex):
        cited.update(key.strip() for key in group.split(","))
    bibliography = (ROOT / "paper/shared/references.bib").read_text(encoding="utf-8")
    defined = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bibliography))
    assert cited <= defined
    assert len(defined) == 26
