#!/usr/bin/env python3
"""Build an HTML case study of the 30K GQA step-grounding training data.

For each selected training row it renders the image with per-step bbox
overlays (V1 / V2 / V3), and embeds the full prompt, answer, GQA functional
program, per-step bbox values, and the textual CoT (thought) for inspection.

Usage (any python with PIL):
  python scripts/lkl_8gpu/tools/build_gqa_case_study_html.py
"""

from __future__ import annotations

import html
import json
import os
import re
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


REPO = "/home/dataset-local/lkl/CoLT-reproduction"
DATA_30K = "/home/dataset-local/lkl/datasets/CoLT_Train_Dataset/colt_sft_gqa_step_grounding_30k.json"
SRC_MERGED = "/home/dataset-local/lkl/datasets/LVR_Train_Dataset/visualcot_98k/lvr_gqa_merged_98k.json"
SCENE_GRAPH = "/home/dataset-local/lkl/tmp/train_sceneGraphs.json"
OUT_DIR = os.path.join(REPO, "Markdown", "experiments")
ASSET_DIR = os.path.join(OUT_DIR, "assets", "gqa_30k_cases")
MAX_W = 720

STEP_COLORS = {"V1": "#2563eb", "V2": "#ea580c", "V3": "#16a34a"}
TEMPLATE_MARKER = "Please answer this question based on the visual content."

# (index-in-30k, category, short title)
CASES = [
    (0, "select→relate→query 经典例", "锚点对象 bbox 紧凑"),
    (2, "空间关系 in front of", "锚点 + 关联对象逐步累积"),
    (127, "空间关系 to the left of", "locate(man) → relate → identify(vehicle)"),
    (1, "大 bbox 边界案例", "GQA 对 man 的标注接近整图 (area=0.77)"),
    (16, "大 bbox 边界案例", "GQA 对 sand 的标注很大 (area=0.59)"),
    (85, "细粒度对象命名", "左侧家具的类别命名"),
    (11, "比较/选择型", "Which is younger, the man or the boy?"),
    (372, "定位/类别型", "What kind of dessert is on the countertop?"),
]


def extract_question(sample: dict) -> str:
    content = sample["messages"][0]["content"]
    return content.split("<image>", 1)[1].split(TEMPLATE_MARKER, 1)[0].strip()


def extract_assistant(sample: dict) -> tuple[Optional[str], Optional[str]]:
    content = sample["messages"][1]["content"]
    think = re.search(r"<think>(.*?)</think>", content, re.S)
    answer = re.search(r"<answer>(.*?)</answer>", content, re.S)
    return (
        think.group(1).strip() if think else None,
        answer.group(1).strip() if answer else None,
    )


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate(image_path: str, boxes: list[list[float]], color: str, label: str, out_path: str) -> dict:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_W / w)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = load_font(18)
    for bb in boxes:
        x1, y1, x2, y2 = (int(bb[0] * img.width), int(bb[1] * img.height),
                          int(bb[2] * img.width), int(bb[3] * img.height))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 3, y1 + 3), label, fill=color, font=font)
    img.save(out_path)
    return {"path": out_path, "width": img.width, "height": img.height}


def object_names(scene: dict, img_id: str, arg: str) -> list[str]:
    im = scene.get(img_id)
    if im is None:
        return []
    names = []
    for obj_id in re.findall(r"\((\d+)\)", arg):
        obj = im["objects"].get(obj_id)
        names.append(f"{obj['name']}#{obj_id}" if obj else f"#{obj_id}")
    return names


def px_bbox(bb: list[float], w: int, h: int) -> list[int]:
    return [int(bb[0] * w), int(bb[1] * h), int(bb[2] * w), int(bb[3] * h)]


def main() -> None:
    data = json.load(open(DATA_30K, encoding="utf-8"))
    src = json.load(open(SRC_MERGED, encoding="utf-8"))
    scene = json.load(open(SCENE_GRAPH, encoding="utf-8"))
    src_map = {(os.path.basename(d["image"]), d["question"]): d for d in src}
    os.makedirs(ASSET_DIR, exist_ok=True)

    cards = []
    for idx, category, note in CASES:
        sample = data[idx]
        question = extract_question(sample)
        thought, answer = extract_assistant(sample)
        prompt = sample["messages"][0]["content"]
        step_bboxes = sample["step_bboxes"]
        image_path = sample["images"][0]
        src_row = src_map.get((os.path.basename(image_path), question), {})
        reasoning = src_row.get("reasoning") or []
        img_id = os.path.basename(image_path).replace(".jpg", "").replace(".png", "")

        im = Image.open(image_path)
        iw, ih = im.size
        im.close()

        base = f"case{idx:05d}"
        per_step = []
        for k, color in (("V1", STEP_COLORS["V1"]), ("V2", STEP_COLORS["V2"]), ("V3", STEP_COLORS["V3"])):
            out = os.path.join(ASSET_DIR, f"{base}_{k}.png")
            per_step.append(
                annotate(image_path, step_bboxes[int(k[1]) - 1], color, k, out)
            )
        combined = os.path.join(ASSET_DIR, f"{base}_all.png")
        all_boxes = [
            (bb, STEP_COLORS[k], k)
            for k, boxes in zip(("V1", "V2", "V3"), step_bboxes)
            for bb in boxes
        ]
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        scale = min(1.0, MAX_W / w)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        font = load_font(18)
        for bb, color, label in all_boxes:
            x1, y1, x2, y2 = (int(bb[0] * img.width), int(bb[1] * img.height),
                              int(bb[2] * img.width), int(bb[3] * img.height))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 3, y1 + 3), label, fill=color, font=font)
        img.save(combined)

        program_rows = []
        for i, st in enumerate(reasoning):
            program_rows.append(
                (i + 1, st.get("operation"), st.get("dependencies"), st.get("argument"),
                 "、".join(object_names(scene, img_id, st.get("argument", ""))) or "-")
            )

        bbox_rows = []
        for k, boxes in zip(("V1", "V2", "V3"), step_bboxes):
            for bb in boxes:
                area = (bb[2] - bb[0]) * (bb[3] - bb[1])
                bbox_rows.append((k, bb, px_bbox(bb, iw, ih), area))

        cards.append(
            {
                "idx": idx,
                "category": category,
                "note": note,
                "question": question,
                "prompt": prompt,
                "answer": answer or "",
                "thought": thought or "",
                "image_path": image_path,
                "program_rows": program_rows,
                "bbox_rows": bbox_rows,
                "per_step": per_step,
                "combined": combined,
            }
        )

    build_html(cards)
    print(f"HTML written; assets in {ASSET_DIR}")


def build_html(cards: list[dict]) -> None:
    css = """
    body { font-family: -apple-system, "Segoe UI", "Noto Sans SC", sans-serif;
           max-width: 1080px; margin: 0 auto; padding: 24px; color: #1f2937;
           background: #f9fafb; line-height: 1.6; }
    h1 { font-size: 26px; border-bottom: 2px solid #2563eb; padding-bottom: 8px; }
    h2 { font-size: 20px; margin-top: 40px; }
    h3 { font-size: 16px; margin: 8px 0; }
    .meta { color: #6b7280; font-size: 14px; }
    .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
            padding: 18px 20px; margin: 22px 0; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
    .tag { display: inline-block; background: #eff6ff; color: #1d4ed8;
           border-radius: 999px; padding: 2px 10px; font-size: 13px; margin-right: 8px; }
    .note { color: #92400e; background: #fffbeb; border-radius: 6px;
            padding: 4px 10px; font-size: 13px; display: inline-block; }
    pre { background: #f3f4f6; border-radius: 8px; padding: 12px; overflow-x: auto;
          font-size: 13px; white-space: pre-wrap; word-break: break-word; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
    th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }
    th { background: #f3f4f6; }
    .imgpanel { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; }
    .imgpanel figure { margin: 0; flex: 1 1 300px; }
    .imgpanel img { width: 100%; border: 1px solid #d1d5db; border-radius: 8px; }
    .imgpanel figcaption { font-size: 12px; color: #4b5563; text-align: center; margin-top: 4px; }
    .answer { font-size: 15px; font-weight: 600; color: #065f46; }
    .swatch { display: inline-block; width: 12px; height: 12px; border-radius: 3px;
              margin-right: 4px; vertical-align: -1px; }
    .statgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 12px; margin: 14px 0; }
    .stat { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px 12px; }
    .stat b { font-size: 20px; color: #1d4ed8; }
    """
    parts = ["<!DOCTYPE html>", '<html lang="zh-CN">', "<head>", '<meta charset="utf-8">',
             "<title>GQA 30K Step-Grounding 数据典型案例</title>", f"<style>{css}</style>",
             "</head>", "<body>"]
    parts.append("<h1>GQA 30K Step-Grounding 数据典型案例</h1>")
    parts.append('<p class="meta">数据：<code>colt_sft_gqa_step_grounding_30k.json</code>（30,000 条，'
                 '全部为 3 步 select→relate/filter→query，每步 bbox 固定 (1,2,1)）· '
                 '生成时间 2026-08-16 · 图片为本地路径引用，需用浏览器直接打开本文件查看。</p>')

    parts.append("<h2>数据画像（30K 全量统计）</h2>")
    parts.append("""<div class="statgrid">
    <div class="stat"><b>30,000</b><br>样本数，100% 为 3 步程序</div>
    <div class="stat"><b>(1,2,1)</b><br>每步 bbox 数：V1 锚点 / V2 锚点+关联 / V3 答案对象</div>
    <div class="stat"><b>100%</b><br>V1⊆V2 且 V3⊆V2（证据逐步累积）</div>
    <div class="stat"><b>~9-10%</b><br>bbox 面积中位数；p75 ≈ 29%</div>
    </div>""")
    parts.append("""<p>过滤漏斗：88,294 条源数据 → 剔除非 3 步程序（17,627）→ 剔除对象无法解析/bbox 非法
    （16,220）→ 候选 ≈ 54,447 → 固定 seed 随机抽样 30K。</p>
    <p>GQA 操作类型分布（30K 全量核验）：<b>select→relate→query 98.3%</b> ·
    select→relate→choose name 0.7% · select→select→choose（比较题）≈ 1.0%。</p>
    <p>问题类型（按问句形态）：what/who/which VQA ≈ 65% · spatial/relational ≈ 27% ·
    attribute/kind ≈ 7% · 比较/选择型 ≈ 1%。</p>
    <p><b>质量 caveat：</b>22.4% 的样本存在面积 &gt; 0.5 的 step bbox（如人物接近整图），
    过滤只保证 ≤ 0.95、不保证语义紧凑；这正是采用 per-step contrastive 而非硬 h_k→patch 匹配的原因。</p>
    <p><b>结构 caveat：</b>过滤只要求程序长度为 3，未强制操作类型。≈ 1%（281 条）
    select→select→choose 比较题中，V3 取的是第二步对象，约 52% 的这类样本 V3 指向非答案对象
    （答案常在 V1 侧）；标准 select→relate→query（98.3%）不受影响。后续可考虑剔除或修正。</p>""")

    for c in cards:
        parts.append('<div class="card">')
        parts.append(f'<h3><span class="tag">{html.escape(c["category"])}</span>'
                     f'Case #{c["idx"]}（30K 索引）<span class="note">{html.escape(c["note"])}</span></h3>')
        if c["program_rows"] and c["program_rows"][1][1] == "select" and c["program_rows"][2][1].startswith("choose"):
            parts.append('<p class="note">⚠️ 比较型程序（select→select→choose）：'
                         "V3 由第二步对象决定，可能指向非答案对象；本例答案在 V1 侧。</p>")
        parts.append(f'<p><b>Question：</b>{html.escape(c["question"])}</p>')
        parts.append(f'<p><b>Answer：</b><span class="answer">{html.escape(c["answer"])}</span></p>')
        parts.append("<p><b>完整 Prompt（训练输入）：</b></p>")
        parts.append(f"<pre>{html.escape(c['prompt'])}</pre>")
        parts.append("<p><b>文本 CoT（thought，来源为 Visual-CoT 98K；训练时被 visual_only mask，不参与 CE）：</b></p>")
        parts.append(f"<pre>{html.escape(c['thought'])}</pre>")

        if c["program_rows"]:
            parts.append("<p><b>GQA Functional Program（step bbox 的来源）：</b></p>")
            parts.append("<table><tr><th>Step</th><th>Operation</th><th>Deps</th>"
                         "<th>Argument</th><th>Scene-graph 对象</th></tr>")
            for row in c["program_rows"]:
                parts.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>")
            parts.append("</table>")

        parts.append("<p><b>Step bbox（归一化 xyxy / 像素 / 面积）：</b></p>")
        parts.append("<table><tr><th>Step</th><th>归一化 bbox</th><th>像素 bbox</th><th>面积占比</th></tr>")
        for k, bb, px, area in c["bbox_rows"]:
            norm = "[" + ", ".join(f"{v:.4f}" for v in bb) + "]"
            pixel = "[" + ", ".join(str(v) for v in px) + "]"
            parts.append(f'<tr><td><span class="swatch" style="background:{STEP_COLORS[k]}"></span>{k}</td>'
                         f"<td>{norm}</td><td>{pixel}</td><td>{area:.4f}</td></tr>")
        parts.append("</table>")

        parts.append("<p><b>逐步骤标注图（图例：</b>"
                     + "".join(f'<span class="swatch" style="background:{STEP_COLORS[k]}"></span>{k} '
                               for k in ("V1", "V2", "V3"))
                     + "）：</p>")
        parts.append('<div class="imgpanel">')
        for k, panel in zip(("V1", "V2", "V3"), c["per_step"]):
            rel = os.path.relpath(panel["path"], OUT_DIR)
            parts.append(f'<figure><img src="{rel}" alt="{k}">'
                         f"<figcaption>{k} —— 该步 bbox 覆盖区域</figcaption></figure>")
        rel_all = os.path.relpath(c["combined"], OUT_DIR)
        parts.append(f'<figure><img src="{rel_all}" alt="all">'
                     "<figcaption>V1+V2+V3 组合视图</figcaption></figure>")
        parts.append("</div>")
        parts.append(f'<p class="meta">原图：<code>{html.escape(c["image_path"])}</code></p>')
        parts.append("</div>")

    parts.append("</body></html>")
    out_path = os.path.join(OUT_DIR, "gqa_30k_step_grounding_典型案例.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
