"""Web/PDF -> KB: nạp khuyến nghị từ knowledge_docs/*.md và dữ liệu có quan hệ từ JSON."""
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
DOCS = HERE.parent.parent / "knowledge_docs"
CYPHER_FILE = HERE.parent.parent / "exports" / "kb_neo4j.cypher"
CYPHER_HEADER = "// diabetes_mas KB export v1"


def fold(s: str) -> str:
    """Bỏ dấu, viết thường: 'Tụt đường' -> 'tut duong'."""
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def parse_docs(text: str) -> list[dict]:
    out = []
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        title, *rest = block.strip().splitlines()
        meta = {k: v.strip() for k, v in (ln.split(":", 1) for ln in rest[:3])}
        out.append({"title": title.strip(), "source": meta["source"], "url": meta["url"], "topic": meta["topic"],
                    "text": " ".join(rest[3:]).strip()})
    return out


@dataclass
class KnowledgeBase:
    recommendations: list[dict]
    foods: list[dict]
    drugs: list[dict]
    drug_interactions: list[dict]
    substitutes: list[dict]
    meta: dict

    def match_drug(self, text: str) -> dict | None:
        f = fold(text)
        for d in self.drugs:
            for name in [d["name"], *d["aliases"]]:
                if re.search(rf"\b{re.escape(fold(name))}\b", f):
                    return d
        return None

    def interactions(self, names: list[str]) -> list[dict]:
        s = set(names)
        return [i for i in self.drug_interactions if i["a"] in s and i["b"] in s]

    def alternatives(self, drug: str, condition: str) -> list[dict]:
        return [x for x in self.substitutes if x["drug"] == drug and x["condition"] == condition]


def load_kb(docs_dir: Path = DOCS) -> KnowledgeBase:
    recs = [r for p in sorted(Path(docs_dir).glob("*.md")) for r in parse_docs(p.read_text(encoding="utf-8"))]
    d = json.loads((HERE / "drugs.json").read_text(encoding="utf-8"))
    foods = json.loads((HERE / "foods.json").read_text(encoding="utf-8"))
    return KnowledgeBase(recs, foods, d["drugs"], d["interactions"], d["substitutes"],
                         {"backend": "bộ nhớ", "reviewed": False, "note": "Khuyến nghị tóm tắt từ bài viết Vinmec và trích từ PDF QĐ 5481/QĐ-BYT của Bộ Y tế, chưa được bác sĩ/dược sĩ thẩm định; "
                                                     "thực phẩm, tương tác và thuốc thay thế là ước lượng riêng của đồ án."})
