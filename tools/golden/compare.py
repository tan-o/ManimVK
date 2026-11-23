"""Compare two render runs of the collected scenes and write a report.

Each image pair gets:

* ``rgb_mae`` / ``alpha_mae``: mean absolute difference (0-255);
* ``psnr``: RGB peak signal-to-noise ratio in dB;
* ``changed``: fraction of pixels differing by more than 2/255 in any channel;
* ``coarse_mae``: RGB MAE after 8x8 area downsampling. Anti-aliasing and glyph
  rasterization differences average out at this scale; a large value means the
  drawing itself differs (missing object, wrong colour, wrong position).

Verdicts (thresholds are options):

* FAIL: sizes differ, ``coarse_mae`` > 3, or ``alpha_mae`` > 2;
* REVIEW: drawing matches coarsely but ``rgb_mae`` > 1 or ``changed`` > 10%;
* PASS: otherwise.

A scene's verdict is its worst image. A scene that errors with only one
renderer is a regression to investigate; one that errors with both is reported
separately (usually a missing tool such as LaTeX)::

    python tools/golden/compare.py --reference golden/cairo --candidate golden/vulkan --out golden/report
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMAGES = ("eval_first", "eval_mid", "eval_last", "final")
ORDER = {"FAIL": 0, "ERROR_CANDIDATE": 1, "REVIEW": 2, "ERROR_REFERENCE": 3, "ERROR_BOTH": 4, "PASS": 5}


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64)


def coarse(image: np.ndarray, factor: int = 8) -> np.ndarray:
    h, w = (image.shape[0] // factor) * factor, (image.shape[1] // factor) * factor
    image = image[:h, :w]
    return image.reshape(h // factor, factor, w // factor, factor, -1).mean(axis=(1, 3))


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        return {"size_mismatch": True}
    diff = np.abs(reference - candidate)
    rgb = diff[..., :3]
    mse = float((rgb**2).mean())
    return {
        "rgb_mae": float(rgb.mean()),
        "alpha_mae": float(diff[..., 3].mean()),
        "psnr": float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse)),
        "changed": float((diff > 2).any(axis=-1).mean()),
        "coarse_mae": float(np.abs(coarse(reference)[..., :3] - coarse(candidate)[..., :3]).mean()),
    }


def verdict(m: dict, args) -> str:
    if m.get("size_mismatch") or m["coarse_mae"] > args.coarse_fail or m["alpha_mae"] > args.alpha_fail:
        return "FAIL"
    if m["rgb_mae"] > args.mae_review or m["changed"] > args.changed_review:
        return "REVIEW"
    return "PASS"


def heatmap(reference: np.ndarray, candidate: np.ndarray, path: Path) -> None:
    diff = np.abs(reference - candidate).max(axis=-1)
    scaled = np.clip(diff * 4, 0, 255).astype(np.uint8)
    Image.fromarray(np.stack([scaled, np.zeros_like(scaled), 255 - scaled], axis=-1)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coarse-fail", type=float, default=3.0)
    parser.add_argument("--alpha-fail", type=float, default=2.0)
    parser.add_argument("--mae-review", type=float, default=1.0)
    parser.add_argument("--changed-review", type=float, default=0.10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "diff").mkdir(exist_ok=True)

    ids = sorted({p.parent.name for p in args.reference.glob("*/status.json")}
                 | {p.parent.name for p in args.candidate.glob("*/status.json")})
    rows = []
    for scene_id in ids:
        statuses = []
        for root in (args.reference, args.candidate):
            path = root / scene_id / "status.json"
            statuses.append(json.loads(path.read_text(encoding="utf-8")) if path.exists()
                            else {"ok": False, "error": "not rendered"})
        ref_ok, cand_ok = statuses[0]["ok"], statuses[1]["ok"]
        row = {"id": scene_id, "images": {}, "ref_error": statuses[0].get("error"),
               "cand_error": statuses[1].get("error")}
        if not ref_ok or not cand_ok:
            row["verdict"] = ("ERROR_BOTH" if not ref_ok and not cand_ok
                              else "ERROR_CANDIDATE" if not cand_ok else "ERROR_REFERENCE")
            rows.append(row)
            continue
        worst = "PASS"
        for name in IMAGES:
            ref_path, cand_path = args.reference / scene_id / f"{name}.png", args.candidate / scene_id / f"{name}.png"
            if not ref_path.exists() and not cand_path.exists():
                continue
            if ref_path.exists() != cand_path.exists():
                m, v = {"missing": True}, "FAIL"
            else:
                reference, candidate = load(ref_path), load(cand_path)
                m = metrics(reference, candidate)
                v = verdict(m, args)
                if v != "PASS" and not m.get("size_mismatch"):
                    heatmap(reference, candidate, args.out / "diff" / f"{scene_id}__{name}.png")
            m["verdict"] = v
            row["images"][name] = m
            worst = min(worst, v, key=ORDER.__getitem__)
        row["verdict"] = worst
        rows.append(row)

    rows.sort(key=lambda r: (ORDER[r["verdict"]], r["id"]))
    counts = {k: sum(r["verdict"] == k for r in rows) for k in ORDER}
    (args.out / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    def fmt(m: dict, key: str, spec: str) -> str:
        return format(m[key], spec) if key in m else "-"

    lines = ["# Golden comparison", "", f"Reference: `{args.reference}`  ", f"Candidate: `{args.candidate}`", "",
             "| Verdict | Scenes |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in counts.items()]
    lines += ["", "| Scene | Verdict | Worst image | coarse MAE | RGB MAE | alpha MAE | PSNR | changed | Error |",
              "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        if r["verdict"] == "PASS":
            continue
        worst_name, worst = None, {}
        for name, m in r["images"].items():
            if worst_name is None or ORDER[m["verdict"]] < ORDER[worst.get("verdict", "PASS")]:
                worst_name, worst = name, m
        error = (r["cand_error"] or r["ref_error"] or "").replace("|", "\\|").replace("\n", " ")[:160]
        lines.append(f"| {r['id']} | {r['verdict']} | {worst_name or '-'} | {fmt(worst, 'coarse_mae', '.2f')} | "
                     f"{fmt(worst, 'rgb_mae', '.2f')} | {fmt(worst, 'alpha_mae', '.2f')} | "
                     f"{fmt(worst, 'psnr', '.1f')} | {fmt(worst, 'changed', '.1%')} | {error} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cells = []
    for r in rows:
        if r["verdict"] not in ("FAIL", "REVIEW"):
            continue
        for name, m in r["images"].items():
            if m["verdict"] == "PASS":
                continue
            ref = (args.reference / r["id"] / f"{name}.png").resolve().as_uri()
            cand = (args.candidate / r["id"] / f"{name}.png").resolve().as_uri()
            diff = (args.out / "diff" / f"{r['id']}__{name}.png").resolve().as_uri()
            cells.append(f"<h3>{html.escape(r['id'])} / {name}: {m['verdict']} "
                         f"(coarse {m.get('coarse_mae', 0):.2f}, MAE {m.get('rgb_mae', 0):.2f})</h3>"
                         f"<img src='{ref}' width=420><img src='{cand}' width=420><img src='{diff}' width=420>")
    (args.out / "report.html").write_text(
        "<html><body><p>reference | candidate | difference (x4)</p>" + "".join(cells) + "</body></html>",
        encoding="utf-8")
    print(" ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"report: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
