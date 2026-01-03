#!/usr/bin/env python3

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


def _require_pycocotools() -> Any:
    try:
        import pycocotools.mask as mask_utils  # type: ignore

        return mask_utils
    except Exception as e:
        raise RuntimeError(
            "pycocotools is required to decode COCO compressed RLE. "
            "Install it with: pip install pycocotools"
        ) from e


@dataclass(frozen=True)
class MatchedTrack:
    video_id: int
    category_id: int
    noun_phrase: str
    gt_index: int
    pred_index: int
    pred_score: float


def _iter_non_none_frames(
    gt_rles: List[Optional[Dict[str, Any]]],
    pred_rles: List[Optional[Dict[str, Any]]],
    *,
    frame_start: int,
    frame_end: Optional[int],
) -> Iterable[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    n = min(len(gt_rles), len(pred_rles))
    end = n if frame_end is None else min(n, frame_end)
    for frame_idx in range(max(0, frame_start), end):
        gt_rle = gt_rles[frame_idx]
        pred_rle = pred_rles[frame_idx]
        if gt_rle is None or pred_rle is None:
            continue
        yield frame_idx, gt_rle, pred_rle


def _decode_rle_to_u8(mask_utils: Any, rle: Dict[str, Any]) -> np.ndarray:
    decoded = mask_utils.decode(rle)
    # pycocotools may return HxW or HxWx1
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    decoded = decoded.astype(np.uint8)
    return decoded


def _write_mask_png(mask01_u8: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(mask01_u8 * 255, mode="L")
    img.save(out_path)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _match_tracks(
    gt: Dict[str, Any],
    pred: List[Dict[str, Any]],
    *,
    match_by: str,
) -> List[MatchedTrack]:
    gt_anns: List[Dict[str, Any]] = gt["annotations"]

    if match_by != "category_id":
        raise ValueError("Only match_by=category_id is supported for now")

    pred_by_cat: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
    for i, p in enumerate(pred):
        key = (int(p["video_id"]), int(p["category_id"]))
        score = float(p.get("score", 0.0))
        pred_by_cat.setdefault(key, []).append((i, score))

    # Prefer highest-score prediction per (video_id, category_id)
    for k in list(pred_by_cat.keys()):
        pred_by_cat[k].sort(key=lambda t: t[1], reverse=True)

    matched: List[MatchedTrack] = []
    for gi, g in enumerate(gt_anns):
        key = (int(g["video_id"]), int(g["category_id"]))
        choices = pred_by_cat.get(key)
        if not choices:
            continue
        pi, score = choices[0]
        matched.append(
            MatchedTrack(
                video_id=key[0],
                category_id=key[1],
                noun_phrase=str(g.get("noun_phrase", "")),
                gt_index=gi,
                pred_index=pi,
                pred_score=score,
            )
        )

    return matched


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-frame binary PNG masks from veval toy GT/pred JSONs "
            "(COCO compressed RLE in `segmentations`). Writes paired filenames "
            "into --gt-out and --pred-out for OrdoCli eval-iou."
        )
    )
    parser.add_argument("--gt", required=True, type=Path, help="Path to *_gt.json")
    parser.add_argument("--pred", required=True, type=Path, help="Path to *_pred.json")
    parser.add_argument("--gt-out", required=True, type=Path, help="Output dir for GT PNG masks")
    parser.add_argument(
        "--pred-out", required=True, type=Path, help="Output dir for predicted PNG masks"
    )
    parser.add_argument(
        "--match-by",
        default="category_id",
        choices=["category_id"],
        help="How to align GT tracks to predicted tracks",
    )
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Limit number of matched tracks exported (debug convenience)",
    )

    args = parser.parse_args()

    mask_utils = _require_pycocotools()

    gt = _load_json(args.gt)
    pred = _load_json(args.pred)
    if not isinstance(gt, dict):
        raise TypeError("GT JSON must be a dict")
    if not isinstance(pred, list):
        raise TypeError("Pred JSON must be a list")

    matched = _match_tracks(gt, pred, match_by=args.match_by)
    if args.max_tracks is not None:
        matched = matched[: max(0, args.max_tracks)]

    gt_anns: List[Dict[str, Any]] = gt["annotations"]

    total_pairs = 0
    for m in matched:
        g = gt_anns[m.gt_index]
        p = pred[m.pred_index]
        gt_rles: List[Optional[Dict[str, Any]]] = g["segmentations"]
        pred_rles: List[Optional[Dict[str, Any]]] = p["segmentations"]

        for frame_idx, gt_rle, pred_rle in _iter_non_none_frames(
            gt_rles,
            pred_rles,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
        ):
            gt_mask = _decode_rle_to_u8(mask_utils, gt_rle)
            pred_mask = _decode_rle_to_u8(mask_utils, pred_rle)

            # Ensure same shape (H, W)
            if gt_mask.shape != pred_mask.shape:
                raise ValueError(
                    f"Shape mismatch for video={m.video_id} cat={m.category_id} frame={frame_idx}: "
                    f"gt={gt_mask.shape} pred={pred_mask.shape}"
                )

            gt_id = g.get("id", m.gt_index)
            stem = f"video{m.video_id}_cat{m.category_id}_gt{gt_id}_pred{m.pred_index}_frame{frame_idx:04d}.png"
            _write_mask_png(gt_mask, args.gt_out / stem)
            _write_mask_png(pred_mask, args.pred_out / stem)
            total_pairs += 1

    manifest = {
        "gt": str(args.gt),
        "pred": str(args.pred),
        "gt_out": str(args.gt_out),
        "pred_out": str(args.pred_out),
        "matched_tracks": [
            {
                "video_id": m.video_id,
                "category_id": m.category_id,
                "noun_phrase": m.noun_phrase,
                "pred_score": m.pred_score,
                "gt_index": m.gt_index,
                "pred_index": m.pred_index,
            }
            for m in matched
        ],
        "total_exported_pairs": total_pairs,
    }

    (args.gt_out.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Matched tracks: {len(matched)}")
    print(f"Exported paired frames: {total_pairs}")
    print(f"Wrote manifest: {(args.gt_out.parent / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
