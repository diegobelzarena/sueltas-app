# Sueltas App — File Dependencies

This document lists every data file the application reads or generates, its format, purpose, and whether it's required.

All data files live under the `data/` directory by default. Paths are configurable in `config.yaml`.

---

## Configuration

| File | Format | Required | Description |
|------|--------|----------|-------------|
| `config.yaml` | YAML | Optional | Central configuration for server, data paths, UMAP params, network graph, dendrogram, cache limits, and image loading strategy. If missing, built-in defaults are used. Path can be overridden via `SUELTAS_CONFIG` env var. |

---

## Core Data Matrices (NumPy `.npy`)

All loaded in `book_similarity_dashboard.py → set_data()` with `mmap_mode='r'` (read-only memory mapping).  
Paths are configurable via `config.yaml` under `data:`.

| File | Config key | Variable | Shape | Required | Description |
|------|------------|----------|-------|----------|-------------|
| `books_dashboard_ordered.npy` | `data.books` | `self.books` | `(n_books,)` | **Yes** | Array of book identifiers (strings). |
| `impr_names_dashboard_ordered.npy` | `data.impr_names` | `self.impr_names` | `(n_books,)` | **Yes** | Printer name for each book. |
| `symbs_dashboard.npy` | `data.symbs` | `self.symbs` | varies | **Yes** | Symbol/glyph metadata. |
| `n1hat_it_matrix_ordered.npy` | `data.n1hat_it_matrix` | `self.n1hat_it` | `(n_books, n_books)` | **Yes** | Italic font similarity matrix. |
| `n1hat_rm_matrix_ordered.npy` | `data.n1hat_rm_matrix` | `self.n1hat_rm` | `(n_books, n_books)` | **Yes** | Roman font similarity matrix. |
| `w_it_matrix_ordered.npy` | `data.w_it_matrix` | `self._w_it_mmap` | `(n_books, n_books)` | Conditional | Italic weight matrix. Only loaded if edge caches (`edges_cache_*.pkl`) are missing. |
| `w_rm_matrix_ordered.npy` | `data.w_rm_matrix` | `self._w_rm_mmap` | `(n_books, n_books)` | Conditional | Roman weight matrix. Only loaded if edge caches are missing. |

---

## UMAP Position Caches (NumPy `.npy`)

Loaded on-demand via `umap_utils.load_positions()`. If missing and `compute_if_missing=True`, positions are computed from the similarity matrices and saved.

Filename pattern: `umap_{font_type}_{n_neighbors}_{min_dist}.npy`  
Default params from config: `n_neighbors=50`, `min_dist=0.5`.

| File | Variable | Shape | Required | Description |
|------|----------|-------|----------|-------------|
| `umap_combined_50_0.5.npy` | return of `load_positions()` | `(n_books, 2)` | Optional | Combined-font UMAP 2D coordinates. |
| `umap_roman_50_0.5.npy` | return of `load_positions()` | `(n_books, 2)` | Optional | Roman-font UMAP 2D coordinates. |
| `umap_italic_50_0.5.npy` | return of `load_positions()` | `(n_books, 2)` | Optional | Italic-font UMAP 2D coordinates. |

---

## Edge Caches (Pickle `.pkl`)

Generated automatically on first use by `_ensure_precomputed_edges()` and cached to disk.  
Location: `data/images/` (configurable via `images.cache_dir`).  
Old cache files are LRU-pruned (default max 3 files).

Filename pattern: `edges_cache_{font_type}_top{top_k}_bins{n_bins}.pkl`  
Default params: `top_k=5000`, `n_bins=10`.

| File (example) | Contents | Required |
|-----------------|----------|----------|
| `edges_cache_roman_top5000_bins10.pkl` | Dict with precomputed top-k edges and binned edges for the network graph. | Auto-generated |
| `edges_cache_italic_top5000_bins10.pkl` | Same for italic. | Auto-generated |
| `edges_cache_combined_top5000_bins10.pkl` | Same for combined. | Auto-generated |

---

## Image Metadata (Pickle `.pkl`)

| File | Path | Variable | Required | Description |
|------|------|----------|----------|-------------|
| `images_cache_meta.pkl` | `{cache_dir}/images_cache_meta.pkl` | `self.meta` | **Yes** | Dict with keys `"all_letters"` (list of available letter labels) and `"book_index"` (mapping each book to its available `(font, letter)` tuples). |

---

## Glyph Images (WebP `.webp`)

Individual letter/glyph images stored per book in subdirectories of `data/images/`.

### Directory structure

```
data/images/
├── images_cache_meta.pkl
├── BNE_1000_891_T-55340-18/
│   ├── roman_lower-a_001.webp
│   ├── italic_lower-a_001.webp
│   ├── roman_upper-A_001.webp
│   └── ...
├── BNE_1001_615_T-55281-18/
│   └── ...
└── ...
```

### Naming conventions

| Pattern | Example | Description |
|---------|---------|-------------|
| `{font}_{case}-{letter}_{idx}.webp` | `roman_lower-a_001.webp` | Preferred format (font-prefixed). |
| `{case}-{letter}_{idx}.webp` | `lower-a_001.webp` | Legacy format (no font prefix). |
| `{letter}_{idx}.webp` | `a_001.webp` | Unprefixed legacy format. |

### Loading strategy (configurable in `config.yaml` → `images.loading_strategy`)

| Strategy | Behavior | Memory cost |
|----------|----------|-------------|
| `"preload"` | All WebP images read at startup, converted to base64 data URLs, held in RAM. O(1) lookup. | ~1.37× raw file size (base64 overhead). |
| `"lazy"` (default) | Directory index scanned at startup; images served on-demand via Flask route `/images_cache/<path>`. | Minimal at startup. |

---

## Static Assets (CSS / JS)

Loaded automatically by Dash from the `assets/` directory.

| File | Format | Description |
|------|--------|-------------|
| `assets/style.css` | CSS | Dashboard styling. |
| `assets/legend-click-helpers.js` | JavaScript | Client-side legend interaction helpers. |

---

## Unused / Legacy Files

These files exist in the workspace but are **not loaded by any current application code**:

| File | Format | Notes |
|------|--------|-------|
| `Corpus_600.csv` | CSV | Not referenced by any Python file. Likely external reference data. |
| `components_lineage.npy` | NumPy | Listed in `.dockerignore` comments. Not loaded. |
| `components_nodes_gte.npy` | NumPy | Listed in `.dockerignore` comments. Not loaded. |
| `idxs_order.npy` | NumPy | Listed in `.dockerignore` comments. Not loaded. |
| `n1hat_linkage.npy` | NumPy | Linkage is computed at runtime, not loaded from file. |
| `n1hat_linkage_desc.npy` | NumPy | Not loaded. |
| `n1hat_linkage_desc_dist.npy` | NumPy | Not loaded. |
| `images_cache/combined_for_umap_combined_50_0.5.npy` | NumPy | Not loaded by the app. |

---

## Summary

### Required at startup

| # | File | Format |
|---|------|--------|
| 1 | `books_dashboard_ordered.npy` | `.npy` |
| 2 | `impr_names_dashboard_ordered.npy` | `.npy` |
| 3 | `symbs_dashboard.npy` | `.npy` |
| 4 | `n1hat_it_matrix_ordered.npy` | `.npy` |
| 5 | `n1hat_rm_matrix_ordered.npy` | `.npy` |
| 6 | `data/images/images_cache_meta.pkl` | `.pkl` |

### Loaded on demand / auto-generated

| # | File pattern | Format |
|---|--------------|--------|
| 7 | `w_rm_matrix_ordered.npy` | `.npy` |
| 8 | `w_it_matrix_ordered.npy` | `.npy` |
| 9 | `umap_{font}_{n}_{d}.npy` | `.npy` |
| 10 | `edges_cache_{font}_top{k}_bins{b}.pkl` | `.pkl` |
| 11 | `data/images/{book}/*.webp` | `.webp` |

### Optional

| # | File | Format |
|---|------|--------|
| 12 | `config.yaml` | `.yaml` |
