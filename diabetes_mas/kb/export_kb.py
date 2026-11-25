"""Đọc KB từ Neo4j, xuất file dựng lại được đồ thị và ảnh đồ thị.

python -m diabetes_mas.kb.export_kb [--out exports]      tạo kb_neo4j.cypher, kb_graph.png, kb_graph.svg
python -m diabetes_mas.kb.export_kb --restore FILE       dựng lại KB trong Neo4j từ file đã xuất
"""
import argparse
import json
import os
import textwrap
from pathlib import Path

from .neo4j_kb import LABELS, _connect, _rows
from .store import CYPHER_HEADER as HEADER
PANELS = [("Khuyến nghị và nguồn", {"Source": 0, "Recommendation": 1, "Topic": 2}),
          ("Thực phẩm", {"Food": 0, "FoodCategory": 1}),
          ("Thuốc và điều kiện lâm sàng", {"Drug": 0, "Condition": 1}),
          ("Bệnh nhân mẫu, chỉ số, nguy cơ, bệnh, triệu chứng", {"Patient": 0, "RiskFactor": 1, "Measurement": 2, "Disease": 3, "Symptom": 4})]
# Màu theo thứ tự cố định của bảng màu đã kiểm tra; nét đứt phân biệt thêm khi in đen trắng.
EDGES = {"FROM": ("Nguồn", "#2a78d6", "-"), "ABOUT": ("Chủ đề", "#eb6834", "-"), "IN_CATEGORY": ("Nhóm", "#1baf7a", "-"),
         "CONTRAINDICATED_IN": ("Chống chỉ định", "#e34948", "-"), "CAUTION_IN": ("Thận trọng", "#eda100", "--"),
         "INTERACTS_WITH": ("Tương tác", "#6250d6", "-"), "SUBSTITUTE_WHEN": ("Thay thế", "#008300", ":"),
         "HAS_MEASUREMENT": ("Có chỉ số", "#2a78d6", "--"), "HAS_RISK_FACTOR": ("Có yếu tố nguy cơ", "#eda100", "-"),
         "INCREASES_RISK_OF": ("Tăng nguy cơ", "#e34948", "--"), "INDICATES": ("Gợi ý (ngưỡng)", "#6250d6", "--"),
         "DIAGNOSED_WITH": ("Chẩn đoán", "#0b0b0b", "-"), "SUBTYPE_OF": ("Thuộc nhóm", "#52514e", ":"),
         "MAY_LEAD_TO": ("Có thể dẫn tới", "#eb6834", "-"), "HAS_SYMPTOM": ("Triệu chứng", "#1baf7a", "-")}


def literal(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def _props(p: dict) -> str:
    return "{" + ", ".join(f"`{k}`: {literal(v)}" for k, v in sorted(p.items())) + "}"


def read_graph(driver) -> dict:
    """Chỉ đọc các nhãn của KB, theo thứ tự ổn định để hai lần xuất cho cùng một file."""
    nodes = _rows(driver, """MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels)
                             RETURN elementId(n) AS id, labels(n)[0] AS label, properties(n) AS props""", labels=LABELS)
    if not nodes:
        raise RuntimeError("Đồ thị chưa có dữ liệu. Hãy chạy: python -m diabetes_mas.kb.neo4j_kb")
    nodes.sort(key=lambda n: (LABELS.index(n["label"]), n["props"].get("seq", 0), str(n["props"].get("name") or n["props"].get("title"))))
    edges = _rows(driver, """MATCH (a)-[r]->(b)
                             WHERE any(l IN labels(a) WHERE l IN $labels) AND any(l IN labels(b) WHERE l IN $labels)
                             RETURN elementId(a) AS from, elementId(b) AS to, type(r) AS type, properties(r) AS props""", labels=LABELS)
    order = {n["id"]: i for i, n in enumerate(nodes)}
    edges.sort(key=lambda e: (e["type"], order[e["from"]], order[e["to"]], e["props"].get("seq", 0)))
    return {"nodes": nodes, "edges": edges}


def export_cypher(driver, path) -> Path:
    g = read_graph(driver)
    kid = {n["id"]: i for i, n in enumerate(g["nodes"])}
    lines = [HEADER, "MATCH (n) WHERE any(l IN labels(n) WHERE l IN " + literal(LABELS) + ") DETACH DELETE n;"]
    lines += [f"CREATE (:`{n['label']}` {_props({**n['props'], '_kid': kid[n['id']]})});" for n in g["nodes"]]
    lines += [f"MATCH (a {{_kid: {kid[e['from']]}}}), (b {{_kid: {kid[e['to']]}}}) CREATE (a)-[:`{e['type']}` {_props(e['props'])}]->(b);"
              for e in g["edges"]]
    lines.append("MATCH (n) WHERE n._kid IS NOT NULL REMOVE n._kid;")
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)  # không để lại file dở dang nếu lỗi giữa chừng
    return path


def restore(driver, path) -> int:
    """Chạy trong một giao dịch: file lỗi thì đồ thị giữ nguyên."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != HEADER:
        raise ValueError(f"{path} không phải file KB do lệnh này xuất ra.")
    stmts = [ln.rstrip(";") for ln in lines[1:] if ln.strip()]

    def work(tx):
        for s in stmts:
            tx.run(s).consume()

    with driver.session() as session:
        session.execute_write(work)
    return len(stmts)


def draw(g: dict, out_dir) -> tuple[Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans"})  # giữ chữ trong SVG để tìm và sửa được
    ink, surface = "#0b0b0b", "#fcfcfb"
    fig, axes = plt.subplots(1, len(PANELS), figsize=(38, 24), width_ratios=[3.2, 1.6, 2.2, 5.0], facecolor=surface)
    for ax, (title, cols) in zip(axes, PANELS):
        ax.set_facecolor(surface)
        ax.axis("off")
        ax.set_title(title, fontsize=17, fontweight="bold", color=ink, pad=14)
        pos = {}
        for label, col in cols.items():
            members = [n for n in g["nodes"] if n["label"] == label]
            for i, n in enumerate(members):
                pos[n["id"]] = (col, 1 - (i + 0.5) / max(len(members), 1))
                name = n["props"].get("name") or n["props"].get("title") or ""
                text = textwrap.fill(name, 50 if label == "Recommendation" else 24 if len(cols) > 3 else 30)
                ax.text(col, pos[n["id"]][1] + 0.002, text, ha="center", va="bottom", fontsize=9.5, color=ink, zorder=4,
                        bbox={"facecolor": surface, "edgecolor": "none", "pad": 1, "alpha": 0.85})
        if not pos:  # đồ thị thiếu cả một nhóm nhãn thì để trống ô đó, không lỗi
            continue
        used = []
        for e in g["edges"]:
            if e["from"] not in pos or e["to"] not in pos:
                continue
            legend, color, style = EDGES.get(e["type"], (e["type"], "#52514e", "-"))
            if e["type"] not in used:
                used.append(e["type"])
            same_col = pos[e["from"]][0] == pos[e["to"]][0]
            ax.add_patch(FancyArrowPatch(pos[e["from"]], pos[e["to"]], arrowstyle="-|>", mutation_scale=9, lw=1.1, color=color,
                                         linestyle=style, alpha=0.8, zorder=2, shrinkA=4, shrinkB=4,
                                         connectionstyle="arc3,rad=0.25" if same_col else "arc3,rad=0"))
        ax.scatter(*zip(*pos.values()), s=34, color=ink, edgecolors=surface, linewidths=1.5, zorder=3)
        xs = [p[0] for p in pos.values()]
        # chừa chỗ bên trái cho các cung nối hai thuốc cùng cột
        arcs = any(pos[e["from"]][0] == pos[e["to"]][0] for e in g["edges"] if e["from"] in pos and e["to"] in pos)
        ax.set_xlim(min(xs) - (1.0 if arcs else 0.6), max(xs) + 0.6)
        ax.set_ylim(-0.03, 1.02)
        ax.legend(handles=[Line2D([0], [0], color=EDGES.get(t, (t, "#52514e", "-"))[1], ls=EDGES.get(t, (t, "#52514e", "-"))[2], lw=2,
                                  label=EDGES.get(t, (t, "", ""))[0]) for t in used],
                  loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=min(max(len(used), 1), 5), frameon=False, fontsize=11, labelcolor=ink)
    fig.suptitle(f"Đồ thị tri thức trong Neo4j: {len(g['nodes'])} nút, {len(g['edges'])} quan hệ", fontsize=20, fontweight="bold", color=ink)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = Path(out_dir)
    png, svg = out / "kb_graph.png", out / "kb_graph.svg"
    fig.savefig(png, dpi=150, facecolor=surface)
    fig.savefig(svg, facecolor=surface)
    plt.close(fig)
    return png, svg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="exports")
    ap.add_argument("--restore", metavar="FILE")
    args = ap.parse_args()
    drv = _connect()
    if args.restore:
        print(f"Đã dựng lại KB từ {args.restore} ({restore(drv, args.restore)} câu lệnh). Hãy khởi động lại server.")
    else:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        cy = export_cypher(drv, out_dir / "kb_neo4j.cypher")
        graph = read_graph(drv)
        png, svg = draw(graph, out_dir)
        print(f"Đã đọc từ Neo4j {len(graph['nodes'])} nút, {len(graph['edges'])} quan hệ.\nĐã tạo: {cy}, {png}, {svg}")
    drv.close()
