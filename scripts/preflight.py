#!/usr/bin/env python
"""
preflight.py — Verify required files and pre-generate caches.

Run this *before* launching the dashboard to make sure every data file is
in place and all cache artefacts are ready.  The script will:

  1.  Check that every required file exists.
  2.  Check optional / on-demand files and report their status.
  3.  Generate missing edge caches  (edges_cache_*.pkl).
  4.  Generate missing UMAP position caches  (umap_*.npy).
  5.  Print a final summary.

Usage
-----
    python scripts/preflight.py              # uses ./config.yaml
    python scripts/preflight.py --config /path/to/config.yaml
    python scripts/preflight.py --skip-umap  # skip slow UMAP computation
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import textwrap
import time

import numpy as np
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import squareform

# ── Ensure project root is importable ────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import umap_utils  # noqa: E402  (project module)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OK   = "[  OK  ]"
_WARN = "[ WARN ]"
_FAIL = "[ FAIL ]"
_SKIP = "[ SKIP ]"
_MAKE = "[ MAKE ]"


def _status(tag: str, msg: str) -> None:
    print(f"  {tag}  {msg}")


def _load_config(path: str) -> dict:
    """Load config.yaml (returns empty dict when missing)."""
    if os.path.exists(path):
        import yaml  # optional at module-level; fine here
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        _status(_OK, f"config.yaml loaded from {path}")
        return cfg
    _status(_WARN, f"config.yaml not found at {path} — using built-in defaults")
    return {}


# ---------------------------------------------------------------------------
# 1. File existence checks
# ---------------------------------------------------------------------------

def check_required_files(data_cfg: dict, cache_dir: str) -> list[str]:
    """Return a list of *missing* required file paths."""
    required = {
        "n1hat_it_matrix":  data_cfg.get("n1hat_it_matrix",  "./data/n1hat_it_matrix_ordered.npy"),
        "n1hat_rm_matrix":  data_cfg.get("n1hat_rm_matrix",  "./data/n1hat_rm_matrix_ordered.npy"),
        "books":            data_cfg.get("books",            "./data/books_dashboard_ordered.npy"),
        "impr_names":       data_cfg.get("impr_names",       "./data/impr_names_dashboard_ordered.npy"),
        "symbs":            data_cfg.get("symbs",            "./data/symbs_dashboard.npy"),
        "images_cache_meta": os.path.join(cache_dir, "images_cache_meta.pkl"),
    }

    missing = []
    for label, path in required.items():
        if os.path.exists(path):
            _status(_OK, f"Required   {label:25s} → {path}")
        else:
            _status(_FAIL, f"Required   {label:25s} → {path}  ** MISSING **")
            missing.append(path)
    return missing


def check_optional_files(data_cfg: dict, umap_cfg: dict) -> None:
    """Report status of conditional / optional files (non-blocking)."""
    optional = {
        "w_rm_matrix":  data_cfg.get("w_rm_matrix",  "./data/w_rm_matrix_ordered.npy"),
        "w_it_matrix":  data_cfg.get("w_it_matrix",  "./data/w_it_matrix_ordered.npy"),
    }

    n_neighbors = umap_cfg.get("n_neighbors", 50)
    min_dist    = umap_cfg.get("min_dist", 0.5)
    for font in ("combined", "roman", "italic"):
        key = f"umap_{font}"
        optional[key] = umap_utils.umap_filename(font, n_neighbors, min_dist)

    for label, path in optional.items():
        if os.path.exists(path):
            _status(_OK, f"Optional   {label:25s} → {path}")
        else:
            _status(_WARN, f"Optional   {label:25s} → {path}  (missing)")


# ---------------------------------------------------------------------------
# 2. Edge cache generation
# ---------------------------------------------------------------------------

def _edge_cache_path(cache_dir: str, font_type: str, top_k: int, n_bins: int) -> str:
    return os.path.join(cache_dir, f"edges_cache_{font_type}_top{top_k}_bins{n_bins}.pkl")


def _compute_edges_for_font(font_type: str, w_rm, w_it, top_k: int, n_bins: int):
    """Pure-function replica of BookSimilarityDashboard._compute_edges_for_font."""
    n = w_rm.shape[0]
    i_idx, j_idx = np.triu_indices(n, k=1)

    if font_type == "roman":
        ew = w_rm[i_idx, j_idx]
    elif font_type == "italic":
        ew = w_it[i_idx, j_idx]
    else:  # combined
        ew = (w_rm[i_idx, j_idx] + w_it[i_idx, j_idx]) / 2.0

    if ew.size == 0:
        return [], {i: {"edges": [], "avg_w": 0} for i in range(n_bins)}

    k = min(top_k, ew.size)
    idx_top = np.argpartition(-ew, k - 1)[:k]
    idx_top_sorted = idx_top[np.argsort(ew[idx_top])[::-1]]

    top_edges = [(int(i_idx[idx]), int(j_idx[idx])) for idx in idx_top_sorted]
    top_w = ew[idx_top_sorted]

    if top_w.size == 0 or top_w.max() == top_w.min():
        binned = {i: {"edges": [], "avg_w": 0} for i in range(n_bins)}
        return top_edges, binned

    bins = np.linspace(top_w.min(), top_w.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(top_w, bins) - 1, 0, n_bins - 1)

    binned = {}
    for b in range(n_bins):
        sel = np.where(bin_idx == b)[0]
        edges_in_bin = [top_edges[i] for i in sel]
        if edges_in_bin:
            s = 0.0
            for (ii, jj) in edges_in_bin:
                if font_type == "roman":
                    s += float(w_rm[ii, jj])
                elif font_type == "italic":
                    s += float(w_it[ii, jj])
                else:
                    s += float((w_rm[ii, jj] + w_it[ii, jj]) / 2.0)
            avg_w = s / len(edges_in_bin)
        else:
            avg_w = 0
        binned[b] = {"edges": edges_in_bin, "avg_w": avg_w}

    return top_edges, binned


def generate_edge_caches(data_cfg: dict, net_cfg: dict, cache_dir: str) -> bool:
    """Generate any missing edge-cache .pkl files. Returns True if all OK."""
    top_k  = net_cfg.get("top_k", 5000)
    n_bins = net_cfg.get("n_bins", 10)
    fonts  = ["roman", "italic", "combined"]

    missing_fonts = [f for f in fonts if not os.path.exists(_edge_cache_path(cache_dir, f, top_k, n_bins))]
    if not missing_fonts:
        _status(_OK, "All edge caches present.")
        return True

    # We need weight matrices to compute
    w_rm_path = data_cfg.get("w_rm_matrix", "./data/w_rm_matrix_ordered.npy")
    w_it_path = data_cfg.get("w_it_matrix", "./data/w_it_matrix_ordered.npy")

    if not os.path.exists(w_rm_path) or not os.path.exists(w_it_path):
        _status(_FAIL, f"Cannot generate edge caches — weight matrices missing:")
        if not os.path.exists(w_rm_path):
            _status(_FAIL, f"  {w_rm_path}")
        if not os.path.exists(w_it_path):
            _status(_FAIL, f"  {w_it_path}")
        return False

    w_rm = np.load(w_rm_path, mmap_mode="r")
    w_it = np.load(w_it_path, mmap_mode="r")
    _status(_OK, f"Loaded weight matrices ({w_rm.shape})")

    os.makedirs(cache_dir, exist_ok=True)

    all_ok = True
    for font in missing_fonts:
        cache_path = _edge_cache_path(cache_dir, font, top_k, n_bins)
        _status(_MAKE, f"Computing edge cache: {font} (top_k={top_k}, n_bins={n_bins}) ...")
        t0 = time.perf_counter()
        try:
            top_edges, binned_edges = _compute_edges_for_font(font, w_rm, w_it, top_k, n_bins)
            with open(cache_path, "wb") as f:
                pickle.dump({"top_edges": top_edges, "binned_edges": binned_edges},
                            f, protocol=pickle.HIGHEST_PROTOCOL)
            dt = time.perf_counter() - t0
            _status(_OK, f"Saved {cache_path}  ({dt:.1f}s)")
        except Exception as e:
            _status(_FAIL, f"Edge cache {font} failed: {e}")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# 3. UMAP position cache generation
# ---------------------------------------------------------------------------

def generate_umap_caches(data_cfg: dict, umap_cfg: dict, *, skip: bool = False) -> bool:
    """Generate missing UMAP .npy caches. Returns True if all OK or skipped."""
    n_neighbors = umap_cfg.get("n_neighbors", 50)
    min_dist    = umap_cfg.get("min_dist", 0.5)

    fonts_and_matrices = {
        "roman":    data_cfg.get("n1hat_rm_matrix", "./data/n1hat_rm_matrix_ordered.npy"),
        "italic":   data_cfg.get("n1hat_it_matrix", "./data/n1hat_it_matrix_ordered.npy"),
        "combined": None,  # built from rm + it
    }

    missing = []
    for font in fonts_and_matrices:
        path = umap_utils.umap_filename(font, n_neighbors, min_dist)
        if not os.path.exists(path):
            missing.append(font)

    if not missing:
        _status(_OK, "All UMAP position caches present.")
        return True

    if skip:
        for font in missing:
            path = umap_utils.umap_filename(font, n_neighbors, min_dist)
            _status(_SKIP, f"UMAP {font:10s} → {path}  (--skip-umap)")
        return True

    # We need n1hat matrices to build distance matrices
    rm_path = data_cfg.get("n1hat_rm_matrix", "./data/n1hat_rm_matrix_ordered.npy")
    it_path = data_cfg.get("n1hat_it_matrix", "./data/n1hat_it_matrix_ordered.npy")

    if not os.path.exists(rm_path) or not os.path.exists(it_path):
        _status(_FAIL, "Cannot compute UMAP — n1hat matrices missing.")
        return False

    n1hat_rm = np.load(rm_path, mmap_mode="r")
    n1hat_it = np.load(it_path, mmap_mode="r")

    def _strength_to_distance(mat):
        mat_f = mat.astype(float)
        maxv = float(np.max(mat_f))
        dist = maxv - mat_f
        np.fill_diagonal(dist, 0.0)
        return dist

    all_ok = True
    for font in missing:
        _status(_MAKE, f"Computing UMAP positions: {font} (n_neighbors={n_neighbors}, min_dist={min_dist}) ...")
        t0 = time.perf_counter()
        try:
            if font == "roman":
                dist = _strength_to_distance(n1hat_rm)
            elif font == "italic":
                dist = _strength_to_distance(n1hat_it)
            else:
                combined = (n1hat_rm.astype(float) + n1hat_it.astype(float))
                dist = _strength_to_distance(combined)

            positions = umap_utils.load_positions(
                font_type=font,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                compute_if_missing=True,
                distance_matrix=dist,
            )
            dt = time.perf_counter() - t0
            if positions is not None:
                _status(_OK, f"UMAP {font:10s} → shape {positions.shape}  ({dt:.1f}s)")
            else:
                _status(_FAIL, f"UMAP {font} returned None (is umap-learn installed?)")
                all_ok = False
        except Exception as e:
            _status(_FAIL, f"UMAP {font} failed: {e}")
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# 4. Image directory sanity check
# ---------------------------------------------------------------------------

def check_images(cache_dir: str, books_path: str) -> None:
    """Quick sanity check: do book subdirectories exist in images_cache?"""
    if not os.path.isdir(cache_dir):
        _status(_WARN, f"Image cache dir does not exist: {cache_dir}")
        return

    if not os.path.exists(books_path):
        _status(_SKIP, "Cannot check image dirs — books array missing.")
        return

    books = np.load(books_path, allow_pickle=True)
    total = len(books)
    present = sum(1 for b in books if os.path.isdir(os.path.join(cache_dir, str(b))))
    if present == total:
        _status(_OK, f"All {total} book image directories found in {cache_dir}")
    else:
        _status(_WARN, f"{present}/{total} book image directories found in {cache_dir}")
        # Show first few missing
        missing = [str(b) for b in books if not os.path.isdir(os.path.join(cache_dir, str(b)))]
        for m in missing[:5]:
            _status(_WARN, f"  missing dir: {cache_dir}/{m}/")
        if len(missing) > 5:
            _status(_WARN, f"  ... and {len(missing) - 5} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify required files and pre-generate caches for sueltas-app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Exit codes:
              0  Everything OK (or only warnings).
              1  One or more required files are missing.
              2  Cache generation failed (app may still work, but slower on first use).
        """),
    )
    parser.add_argument("--config", default=os.environ.get("SUELTAS_CONFIG", "./config.yaml"),
                        help="Path to config.yaml  [default: ./config.yaml or $SUELTAS_CONFIG]")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Skip UMAP computation (can be slow for large datasets).")
    args = parser.parse_args()

    print("=" * 64)
    print("  sueltas-app preflight check")
    print("=" * 64)
    print()

    # ── Load config ──────────────────────────────────────────────────
    cfg       = _load_config(args.config)
    data_cfg  = cfg.get("data", {})
    umap_cfg  = cfg.get("umap", {})
    net_cfg   = cfg.get("network", {})
    img_cfg   = cfg.get("images", {})
    cache_dir = img_cfg.get("cache_dir", "./data/images/")

    # ── 1. Required files ────────────────────────────────────────────
    print()
    print("── Required files ──────────────────────────────────────────")
    missing = check_required_files(data_cfg, cache_dir)

    if missing:
        print()
        _status(_FAIL, f"{len(missing)} required file(s) missing — the app cannot start.")
        _status(_FAIL, "Please make sure these files are present and try again.")
        return 1

    # ── 2. Optional / on-demand files ────────────────────────────────
    print()
    print("── Optional files ──────────────────────────────────────────")
    check_optional_files(data_cfg, umap_cfg)

    # ── 3. Image directories ─────────────────────────────────────────
    print()
    print("── Image directories ───────────────────────────────────────")
    books_path = data_cfg.get("books", "./data/books_dashboard_ordered.npy")
    check_images(cache_dir, books_path)

    # ── 4. Edge caches ───────────────────────────────────────────────
    print()
    print("── Edge caches ─────────────────────────────────────────────")
    edges_ok = generate_edge_caches(data_cfg, net_cfg, cache_dir)

    # ── 5. UMAP caches ──────────────────────────────────────────────
    print()
    print("── UMAP position caches ────────────────────────────────────")
    umap_ok = generate_umap_caches(data_cfg, umap_cfg, skip=args.skip_umap)

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 64)
    if edges_ok and umap_ok:
        print("  All checks passed. The dashboard is ready to launch.")
    else:
        problems = []
        if not edges_ok:
            problems.append("edge caches")
        if not umap_ok:
            problems.append("UMAP caches")
        print(f"  Some caches could not be generated: {', '.join(problems)}.")
        print("  The app may still work but will be slower on first use.")
    print("=" * 64)

    return 0 if (edges_ok and umap_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
