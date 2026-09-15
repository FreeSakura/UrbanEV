#!/usr/bin/env python3
"""Build or check public research indexes, result summaries and Markdown links.

This standard-library tool reads public artifacts only. It never starts an
experiment, downloads data, or modifies frozen scientific evidence.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIRS = (".github", "artifacts", "configs", "docs", "licenses", "models",
               "paper", "release-assets", "results", "scripts", "src", "tests")
SKIP = {"__pycache__", ".pytest_cache", "build", "editable", "tmp"}
TEXT = {".md", ".csv", ".json", ".txt", ".py", ".yml", ".yaml", ".toml", ".tex", ".bib", ".cff"}
CORE = "artifacts/summaries/comprehensive_development_comparison_v1/"
MODEL_ORDER = ("RIDGE_O", "RIDGE_OD", "OD_PRODUCT", "OD_SEPARABLE",
               "OD_CONCAT_MLP", "TIMEXER_LOCAL_OD", "TIMEXER_GLOBAL_O")
SEEDS = {"20260915", "20260916", "20260917"}
CATEGORIES = {
    "benchmark": "基准与比较", "forecasting": "预测方法", "calibration": "校准与风险",
    "information": "观测信息", "mechanisms": "机制与理论", "audit": "历史审计", "history": "阶段历史",
}
KINDS = {"registered": "注册待完整结果", "development": "开发证据", "calibration": "校准证据",
         "synthetic": "合成检查", "protocol": "协议登记", "historical": "历史证据", "reference": "资料与核查"}
FOLDER_INTROS = {
    "docs/reports": ("实验与综合报告", "综合报告串联项目认识；各子目录保存原阶段记录。具体运行完成状态以公开回执为依据。"),
    "docs/reports/benchmark": ("基准与比较报告", "区分上游合同核对、已完成开发比较以及注册后的完整六折。"),
    "docs/reports/forecasting": ("预测方法报告", "短状态、长历史、交互、正则化和纠错研究；不同训练 cohort 不混排。"),
    "docs/reports/calibration": ("校准与风险报告", "分位输出、连续步长和双风险研究。原 NO_GO 按对应协议解释。"),
    "docs/reports/information": ("观测信息报告", "空间、时长、电量与字段可得性研究。特定模型失败不等于信息普遍无效。"),
    "docs/reports/mechanisms": ("机制与方法资料", "人工机制、理论来源及方法阅读，明确与真实预测证据的区别。"),
    "docs/reports/audit": ("配对事件审计", "历史开发数据上的事件界研究，与当前占用率预测分数分开。"),
    "docs/theory": ("理论与推导", "V1/V2/V3 是历史演进版本；主题推导和阅读记录各有假设与适用范围。"),
    "docs/protocols": ("阶段协议", "这里保留文字设计、登记与修订；机器配置仍位于 configs。旧阶段限制不覆盖后来正式合同。"),
    "docs/reviews": ("评审与来源快照", "评审记录适用于写作时的证据，后续结果不倒改原判断。"),
    "docs/history/proposals": ("历史方案", "初始、最终、实验计划等名称属于各自阶段；当前方案从项目状态与研究路线进入。"),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalized_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if path.suffix.lower() in TEXT else data


def public_files(root: Path) -> list[Path]:
    files = [p for p in root.iterdir() if p.is_file()]
    for name in PUBLIC_DIRS:
        files.extend(p for p in (root / name).rglob("*") if p.is_file()
                     and not (set(p.relative_to(root).parts) & SKIP)
                     and not any(x.endswith(".egg-info") for x in p.parts))
    return sorted(set(files), key=lambda p: p.relative_to(root).as_posix())


def relative_link(target: str, source: str) -> str:
    return os.path.relpath(target, str(Path(source).parent)).replace("\\", "/")


def title(path: Path) -> str:
    if path.suffix == ".md":
        match = re.search(r"(?m)^# (.+)$", path.read_text(encoding="utf-8-sig"))
        if match:
            return match[1].strip().replace("|", "／")
    return path.stem


def csv_text(rows: list[dict], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def sorted_paths(paths):
    """Use the same ordering on case-insensitive Windows and case-sensitive POSIX."""
    return sorted(paths, key=lambda path: path.as_posix())


def catalog_records(root: Path) -> list[dict]:
    catalog = read_json(root / "results/studies.json")
    if catalog.get("schema_version") != "urbanev-study-catalog/v1":
        raise ValueError("Unsupported study catalog schema")
    records = catalog["studies"]
    ids = [r["id"] for r in records]
    directories = {p.name for p in (root / "artifacts/summaries").iterdir() if p.is_dir()}
    if len(ids) != len(set(ids)) or set(ids) != directories:
        raise ValueError("Study catalog must cover every evidence directory exactly once")
    for row in records:
        if row["category"] not in CATEGORIES or row["evidence_kind"] not in KINDS:
            raise ValueError(f"Unknown evidence classification: {row['id']}")
        if row["evidence_dir"] != "artifacts/summaries/" + row["id"] or not row["reports"]:
            raise ValueError(f"Invalid evidence association: {row['id']}")
        for key in ("reports", "configs", "entrypoints", "status_sources"):
            for name in row[key]:
                path = root / name
                if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
                    raise ValueError(f"Missing or unsafe {key} link: {name}")
    return records


def aggregate_core(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Require complete fixed identities; average H within seed, then seeds.

    The input is a specific published development table, not a general-purpose
    leaderboard. Cached two-horizon references never enter its four-H averages.
    """
    selected = defaultdict(dict)
    for row in rows:
        if row["target_scope"] != "terminal_H" or row["postprocess"] != "raw":
            continue
        if row["cohort"] == "CACHED_NATIVE_REFERENCE":
            if row["model"] != "NATIVE":
                raise ValueError("Unexpected cached reference identity")
            continue
        if row["cohort"] != "NEW_MATCHED_CORE" or row["model"] not in MODEL_ORDER:
            raise ValueError("Unexpected model or mixed training cohort")
        if row["status"] != "AVAILABLE" or int(row["scored_values"]) != 3850:
            raise ValueError("Incomplete or mismatched development support")
        expected_track = "GLOBAL_O_168" if row["model"] == "TIMEXER_GLOBAL_O" else (
            "LOCAL_O_168" if row["model"] == "RIDGE_O" else "LOCAL_OD_168")
        if row["information_track"] != expected_track:
            raise ValueError("Mismatched information track")
        key = row["model"], row["seed"]
        horizon = int(row["horizon"])
        if horizon in selected[key]:
            raise ValueError("Duplicate model/seed/horizon")
        values = tuple(float(row[m]) for m in ("rmse", "mae"))
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Non-finite or negative metric")
        selected[key][horizon] = values
    if {key[0] for key in selected} != set(MODEL_ORDER):
        raise ValueError("Missing registered core model")
    instances, models = [], []
    for model in MODEL_ORDER:
        seeds = {seed for name, seed in selected if name == model}
        expected = {"fixed"} if model.startswith("RIDGE_") else SEEDS
        if seeds != expected:
            raise ValueError(f"Incomplete seed coverage: {model}")
        group = []
        for seed in sorted(seeds):
            values = selected[model, seed]
            if set(values) != {3, 6, 9, 12}:
                raise ValueError(f"Incomplete horizon coverage: {model}/{seed}")
            item = {"model": model, "seed": seed,
                    "rmse": mean(v[0] for v in values.values()), "mae": mean(v[1] for v in values.values())}
            instances.append(item)
            group.append(item)
        models.append({"model": model, "instances": len(group), "horizons": "3/6/9/12",
                       "cohort": "NEW_MATCHED_CORE", "target_scope": "terminal_H", "postprocess": "raw",
                       "rmse": mean(g["rmse"] for g in group), "mae": mean(g["mae"] for g in group)})
    return models, instances


def development_table(root: Path) -> list[dict]:
    with (root / CORE / "matched_core_scores.csv").open(encoding="utf-8-sig", newline="") as handle:
        models, instances = aggregate_core(list(csv.DictReader(handle)))
    frozen = read_json(root / CORE / "model_summary.json")["per_instance"]
    for item in instances:
        matches = [r for r in frozen if r["model"] == item["model"] and str(r["seed"]) == item["seed"]
                   and r["support"] == "FOUR_H" and r["postprocess"] == "raw"]
        if len(matches) != 1 or any(not math.isclose(item[m], matches[0]["macro_" + m], rel_tol=0, abs_tol=1e-12)
                                    for m in ("rmse", "mae")):
            raise ValueError(f"Derived metric differs from frozen summary: {item['model']}/{item['seed']}")
    return models


def generate(root: Path) -> dict[str, str]:
    studies = catalog_records(root)
    models = development_table(root)
    outputs = {}
    inventory = []
    for study in studies:
        for path in sorted_paths((root / study["evidence_dir"]).rglob("*")):
            if path.is_file():
                data = normalized_bytes(path)
                inventory.append({"study_id": study["id"], "path": path.relative_to(root).as_posix(),
                                  "format": path.suffix.lstrip("."), "bytes_lf": len(data),
                                  "sha256_lf": hashlib.sha256(data).hexdigest()})
    outputs["results/inventory.csv"] = csv_text(inventory, ["study_id", "path", "format", "bytes_lf", "sha256_lf"])
    outputs["results/development-comparison.csv"] = csv_text(models, list(models[0]))
    registration = read_json(root / "artifacts/summaries/urbanev_matched_six_fold_v1/registration.json")
    lines = ["# 结果总览", "", "<!-- Generated by scripts/repository.py; edit the sources, then build. -->", "",
             "本页由原始公开汇总生成。不同窗口、训练协议及数据集分别解释；登记预算与合成检查不作为真实预测成绩。",
             "", "[综合报告](../docs/reports/PROJECT_REPORT.md) · [研究登记](studies.json) · [逐文件证据清单](inventory.csv) · [项目状态](../docs/PROJECT_STATUS.md)",
             "", "## 开发集核心比较", "",
             "范围：275 区域占用率，1392—1548 的 14 个已曝光开发原点，stride12；联合预测12步。"
             "主表固定 raw 末点，先平均 H3/H6/H9/H12，再平均种子。神经模型为三个种子，ridge 为一次确定性拟合。",
             "", "| 系统 | 实例数 | 四 H 平均 RMSE | 四 H 平均 MAE |", "|---|---:|---:|---:|"]
    for row in models:
        lines.append(f"| {row['model']} | {row['instances']} | {row['rmse']:.9f} | {row['mae']:.9f} |")
    lines += ["", "这张表按方法类别排列。LOCAL-O、LOCAL-OD 与 GLOBAL-O 的输入信息不同，残差底座与直接预测的训练形式也不同；"
              "分数可比较系统，不能单独归因结构。epoch0 别名及选择记录保留在原证据中。此处不是完整六折或官方论文排行榜。",
              "", "数据：[逐视野/种子原表](../" + CORE + "matched_core_scores.csv)、[冻结汇总](../" + CORE +
              "model_summary.json)、[派生 CSV](development-comparison.csv)、[完整解释](../docs/reports/benchmark/COMPREHENSIVE_COMPARISON_REPORT_20260913.md)。",
              "", "缓存 Chronos native 只有 H3/H12；其共同支持比较保留在原表，不能填进四 H 宏平均。"
              "七个历史 cohort 的共同目标重算保留在[历史分组表](../" + CORE + "archive_family_summary.csv)，不与新核心合并种子或排行。",
              "", "## 完整六折：注册与结果分开", "",
              f"公开注册状态：`{registration['status']}`。登记预期 {registration['expected']['ridge_fits']} 次 ridge 拟合、"
              f"{registration['expected']['neural_training_runs']} 次神经训练、{registration['expected']['foundation_origin_tasks']:,} 个基础模型原点任务。",
              "", "该目录当前只有注册材料，没有全量执行及评分回执；本页不据此推断本地任务实时进度。"
              "[注册回执](../artifacts/summaries/urbanev_matched_six_fold_v1/registration.json) · "
              "[任务清单](../artifacts/summaries/urbanev_matched_six_fold_v1/registered_tasks.json) · "
              "[正式报告](../docs/reports/benchmark/MATCHED_SIX_FOLD_REPORT_20260914.md)",
              "", "## 所有研究与证据", "", f"共登记 **{len(studies)} 个证据目录、{len(inventory)} 个原公开文件**。类型只表示证据阶段，不代表方法成功。"]
    for category, label in CATEGORIES.items():
        lines += ["", f"### {label}", "", "| 研究 | 类型 | 已有认识与范围 |", "|---|---|---|"]
        for study in studies:
            if study["category"] == category:
                link = "../" + study["reports"][0]
                evidence = "../" + study["evidence_dir"]
                lines.append(f"| [{study['title']}]({link}) · [证据]({evidence}) | {KINDS[study['evidence_kind']]} | {study['summary']} |")
    lines += ["", "## 重建与来源", "", "```bash", "python scripts/repository.py build", "python scripts/repository.py check", "```", "",
              "生成器检查模型、种子、四 H 覆盖与原冻结均值一致，拒绝缺格或跨 cohort 混排。"
              "证据清单对文本采用 LF 换行规范化后的 SHA256，二进制采用原字节。"
              "历史实验的原始哈希、原协议决定及源码清单继续保存在对应证据目录中。", ""]
    outputs["results/README.md"] = "\n".join(lines)
    # Each curated folder has a readable landing page, including nested groups.
    for folder, (heading, intro) in FOLDER_INTROS.items():
        source = folder + "/README.md"
        lines = [f"# {heading}", "", "<!-- Generated by scripts/repository.py. -->", "", intro, "",
                 f"[项目状态]({relative_link('docs/PROJECT_STATUS.md', source)}) · "
                 f"[结果总览]({relative_link('results/README.md', source)})", ""]
        for nested, (child_heading, _) in FOLDER_INTROS.items():
            if Path(nested).parent.as_posix() == folder:
                lines.append(f"- [{child_heading}]({relative_link(nested + '/README.md', source)})")
        for path in sorted_paths((root / folder).glob("*.md")):
            if path.name != "README.md":
                lines.append(f"- [{title(path)}]({path.name})")
        outputs[source] = "\n".join(lines) + "\n"
    # Inventory all readable documents, including generated landing pages.
    document_paths = {p.relative_to(root).as_posix() for p in public_files(root) if p.suffix.lower() in {".md", ".pdf"}}
    document_paths.update(k for k in outputs if k.endswith(".md"))
    document_paths.update({"docs/catalog.md", "scripts/README.md"})
    lines = ["# 完整资料目录", "", "<!-- Generated by scripts/repository.py. -->", "",
             "覆盖公开树中的全部 Markdown 与 PDF。结果文件的逐项索引另见[证据清单](../results/inventory.csv)。", ""]
    groups = defaultdict(list)
    for name in sorted(document_paths):
        groups[str(Path(name).parent).replace("\\", "/")].append(name)
    for folder, names in sorted(groups.items()):
        lines += [f"## {folder}", "", "| 资料 | 格式 |", "|---|---|"]
        for name in names:
            heading = outputs.get(name, "").splitlines()
            label = heading[0][2:] if heading else (title(root / name) if (root / name).exists() else Path(name).stem)
            if name == "docs/catalog.md":
                label = "完整资料目录（本页）"
            elif name == "scripts/README.md":
                label = "脚本目录"
            lines.append(f"| [{label}]({relative_link(name, 'docs/catalog.md')}) | {Path(name).suffix[1:]} |")
        lines.append("")
    outputs["docs/catalog.md"] = "\n".join(lines)
    lines = ["# 脚本目录", "", "<!-- Generated by scripts/repository.py. -->", "",
             "全部命令从仓库根目录执行。以下按源码文档列出入口；实验脚本需要相应版本的配置、数据与环境，"
             "不会因为运行资料构建器而被调用。", "", "[安装与复现](../docs/guides/reproducibility.md) · [架构](../docs/guides/architecture.md)", ""]
    for folder in ("scripts", "scripts/research"):
        lines += [f"## {folder}", "", "| 脚本 | 用途（源码说明） |", "|---|---|"]
        for path in sorted_paths((root / folder).glob("*.py")):
            description = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8-sig"))) or "参见脚本内入口与对应研究配置。"
            description = description.splitlines()[0].replace("|", "／").replace("`", "")
            name = path.relative_to(root).as_posix()
            lines.append(f"| [{path.name}]({relative_link(name, 'scripts/README.md')}) | {description} |")
        lines.append("")
    outputs["scripts/README.md"] = "\n".join(lines)
    return outputs


def prose(text: str) -> str:
    """Remove fenced and inline code so example link syntax is not checked."""
    lines, fence = [], None
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if match:
            run = match[1]
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence):
                fence = None
            continue
        if fence is None:
            lines.append(line)
    return re.sub(r"(`+).*?\1", "", "\n".join(lines))


def markdown_links(text: str) -> list[str]:
    text = prose(text)
    urls = []
    # Balanced destinations handle images, badges and parentheses in filenames.
    for match in re.finditer(r"\]\(\s*", text):
        start = match.end()
        if start >= len(text):
            continue
        if text[start] == "<":
            end = text.find(">", start + 1)
            if end != -1:
                urls.append(text[start + 1:end])
            continue
        depth, end = 0, start
        while end < len(text):
            ch = text[end]
            if ch == "\\" and end + 1 < len(text):
                end += 2
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            elif ch.isspace() and depth == 0:
                break
            end += 1
        urls.append(re.sub(r"\\([()])", r"\1", text[start:end]))
    for match in re.finditer(r"(?m)^\s{0,3}\[[^\]]+\]:\s*(<[^>]+>|\S+)", text):
        urls.append(match[1].strip("<>"))
    urls.extend(m[1] for m in re.finditer(r'(?:href|src)=["\']([^"\']+)["\']', text))
    return urls


def anchors(text: str) -> set[str]:
    values = set(re.findall(r'(?:id|name)=["\']([^"\']+)["\']', text))
    seen = defaultdict(int)
    # Preserve inline-code heading text while omitting fenced code.
    fenced = re.sub(r"(?ms)^\s{0,3}(`{3,}|~{3,})[^\n]*\n.*?^\s{0,3}\1\s*$", "", text)
    for match in re.finditer(r"(?m)^#{1,6}\s+(.+?)(?:\s+#+)?\s*$", fenced):
        heading = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", match[1])
        heading = html.unescape(re.sub(r"<[^>]*>", "", heading)).lower()
        slug = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        count = seen[slug]
        seen[slug] += 1
        values.add(slug + (f"-{count}" if count else ""))
    return values


def link_errors(root: Path) -> list[str]:
    errors, heading_cache = [], {}
    for path in public_files(root):
        if path.suffix != ".md":
            continue
        name = path.relative_to(root).as_posix()
        for url in markdown_links(path.read_text(encoding="utf-8-sig")):
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc:
                continue
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path.resolve()
            if not target.is_relative_to(root.resolve()) or not target.exists():
                errors.append(f"{name}: missing/unsafe link {url}")
                continue
            # Case-exact paths also work on Linux and GitHub.
            cursor = root.resolve()
            for part in target.relative_to(root.resolve()).parts:
                if part not in {p.name for p in cursor.iterdir()}:
                    errors.append(f"{name}: path case mismatch {url}")
                    break
                cursor /= part
            fragment = unquote(parsed.fragment)
            if fragment and target.suffix == ".md" and target.is_file():
                if target not in heading_cache:
                    heading_cache[target] = anchors(target.read_text(encoding="utf-8-sig"))
                if fragment not in heading_cache[target]:
                    errors.append(f"{name}: missing heading {url}")
    return sorted(set(errors))


def check(root: Path) -> list[str]:
    errors = []
    try:
        for name, content in generate(root).items():
            path = root / name
            if not path.is_file() or path.read_text(encoding="utf-8-sig") != content:
                errors.append(f"stale generated file: {name}")
        for move in read_json(root / "docs/maintenance/path-map.json")["moves"]:
            if not (root / move["to"]).is_file():
                errors.append(f"missing migrated destination: {move['to']}")
    except (KeyError, ValueError, OSError) as exc:
        errors.append(str(exc))
    errors.extend(link_errors(root))
    return errors


def manifest_errors(root: Path) -> list[str]:
    """Check the committed tree manifest without requiring external ZIP assets."""
    manifest = read_json(root / "artifacts/manifests/FULL_RELEASE_MANIFEST.json")
    if manifest.get("schema_version") != "urbanev-full-release-manifest/v1":
        raise ValueError("Unsupported full-tree manifest schema")
    errors, seen = [], set()
    for record in manifest["tracked_files"]:
        name = record["path"]
        path = root / name
        if name in seen or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            errors.append(f"missing, duplicate or unsafe manifest path: {name}")
            continue
        seen.add(name)
        data = normalized_bytes(path)
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            errors.append(f"manifest content mismatch: {name}")
    if not seen:
        errors.append("empty full-tree manifest")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "verify-manifest"))
    args = parser.parse_args()
    if args.command == "build":
        outputs = generate(ROOT)
        for name, content in outputs.items():
            path = ROOT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
        print(f"Built {len(outputs)} public indexes and result files.")
    else:
        errors = manifest_errors(ROOT) if args.command == "verify-manifest" else check(ROOT)
        if errors:
            print("Repository check FAILED\n" + "\n".join(errors))
            raise SystemExit(1)
        if args.command == "verify-manifest":
            print("Repository manifest PASS: committed public file hashes; external Release assets are a separate audit.")
        else:
            print("Repository check PASS: study coverage, derived results, generated files and local Markdown links.")


if __name__ == "__main__":
    main()
