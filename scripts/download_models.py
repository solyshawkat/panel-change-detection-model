#!/usr/bin/env python3
"""
Download ML models on first run.

Downloads CLIP (ViT-B/32) and LightGlue (SuperPoint) weights from
HuggingFace / GitHub so that large .pt/.bin files are never stored
in the git repository.

Usage:
    python scripts/download_models.py          # download all
    python scripts/download_models.py --clip    # CLIP only
    python scripts/download_models.py --lightglue  # LightGlue only

The script is idempotent — re-running skips models that are already
cached by HuggingFace / torch.hub.
"""
from __future__ import annotations

import argparse
import sys
import os

def download_clip():
    """Download CLIP ViT-B/32 weights via HuggingFace transformers."""
    print("=" * 60)
    print("  Downloading CLIP ViT-B/32 ...")
    print("=" * 60)
    try:
        from transformers import CLIPModel, CLIPProcessor

        model_name = "openai/clip-vit-base-patch32"
        print(f"  Model: {model_name}")
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        print(f"  CLIP downloaded successfully ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
    except Exception as e:
        print(f"  ERROR downloading CLIP: {e}")
        return False
    return True


def download_lightglue():
    """Download SuperPoint + LightGlue weights."""
    print("=" * 60)
    print("  Downloading SuperPoint + LightGlue ...")
    print("=" * 60)
    try:
        import torch
        from lightglue import LightGlue, SuperPoint

        device = "cpu"
        print(f"  Loading SuperPoint (max_num_keypoints=2048) ...")
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        print(f"  Loading LightGlue (features='superpoint') ...")
        matcher = LightGlue(features="superpoint").eval().to(device)
        print("  SuperPoint + LightGlue downloaded successfully")
    except Exception as e:
        print(f"  ERROR downloading LightGlue: {e}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Download ML models for PCD-AI service")
    parser.add_argument("--clip", action="store_true", help="Download CLIP only")
    parser.add_argument("--lightglue", action="store_true", help="Download LightGlue only")
    args = parser.parse_args()

    download_all = not args.clip and not args.lightglue

    results = {}

    if download_all or args.clip:
        results["CLIP"] = download_clip()

    if download_all or args.lightglue:
        results["LightGlue"] = download_lightglue()

    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\n  All models downloaded. Ready to run the pipeline.")
    else:
        print("\n  Some downloads failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
