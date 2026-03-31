# sueltas-app

Interactive dashboard for exploring typographic similarity across a corpus of historical books (BNE *sueltas* collection). Built with **Plotly Dash**.

## Quick start

```bash
# 1. Create / activate a Python 3.11 environment
conda create -n dash python=3.11 -y
conda activate dash

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the preflight check (verifies data files & builds caches)
python scripts/preflight.py

# 4. Launch
python launch_dashboard.py
# → open http://localhost:8050
```

## Docker

```bash
docker build -t sueltas-app .
docker run -p 8050:8050 sueltas-app python launch_dashboard.py
```

To serve under a URL prefix (reverse proxy):

```bash
docker run -e REQUESTS_PATHNAME_PREFIX=/sueltas/ -p 8050:8050 sueltas-app python launch_dashboard.py
```

## Preflight check

Before launching the dashboard for the first time, run the **preflight script** to verify that all required data files are present and to pre-generate any missing caches:

```bash
python scripts/preflight.py
```

The script will:

1. **Check required files** — the 5 core `.npy` matrices and `images_cache/images_cache_meta.pkl`. If any are missing the script stops with a clear error message.
2. **Check optional files** — weight matrices (`w_*.npy`) and UMAP caches.
3. **Verify image directories** — confirms each book listed in `books_dashboard_ordered.npy` has a matching subdirectory under `images_cache/`.
4. **Generate edge caches** — computes `edges_cache_*.pkl` from the weight matrices if they don't already exist. Requires `w_rm_matrix_ordered.npy` and `w_it_matrix_ordered.npy`.
5. **Generate UMAP caches** — computes `umap_*.npy` position files from the similarity matrices if missing (requires `umap-learn`). Use `--skip-umap` to skip this step if `umap-learn` is not installed or the dataset is very large.

### Minimum files needed

To run the preflight (and the app) you **must** provide at least these files:

| File | What it is |
|------|------------|
| `n1hat_it_matrix_ordered.npy` | Italic similarity matrix (books × books) |
| `n1hat_rm_matrix_ordered.npy` | Roman similarity matrix (books × books) |
| `books_dashboard_ordered.npy` | Array of book identifiers |
| `impr_names_dashboard_ordered.npy` | Array of printer names per book |
| `symbs_dashboard.npy` | Symbol / character labels |
| `images_cache/images_cache_meta.pkl` | Image metadata (pickled dict) |

Everything else (edge caches, UMAP positions) is generated automatically by the preflight script, provided you also supply the weight matrices (`w_rm_matrix_ordered.npy`, `w_it_matrix_ordered.npy`) for edge caches and have `umap-learn` installed for UMAP positions.

### Options

```bash
python scripts/preflight.py                 # full check + generate all caches
python scripts/preflight.py --skip-umap      # skip UMAP computation
python scripts/preflight.py --config other.yaml  # use a different config file
```

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | All checks passed. |
| `1` | Required files are missing — the app cannot start. |
| `2` | Cache generation failed (app may still work but will be slower on first use). |

> **Tip:** For a detailed breakdown of every file the app uses, see [`FILE_DEPENDENCIES.md`](FILE_DEPENDENCIES.md).

## Required data files

Place these in the repo root (same directory as `launch_dashboard.py`):

| File | Description |
|------|-------------|
| `n1hat_it_matrix_ordered.npy` | N1-hat italic similarity matrix (books × books) |
| `n1hat_rm_matrix_ordered.npy` | N1-hat roman similarity matrix (books × books) |
| `w_rm_matrix_ordered.npy` | Weight matrix — roman |
| `w_it_matrix_ordered.npy` | Weight matrix — italic |
| `books_dashboard_ordered.npy` | Array of book identifiers |
| `impr_names_dashboard_ordered.npy` | Array of imprinter (printer) names per book |
| `symbs_dashboard.npy` | Symbol / character labels |
| `umap_combined_50_0.5.npy` | Pre-computed UMAP positions (combined) |
| `umap_roman_50_0.5.npy` | Pre-computed UMAP positions (roman) |
| `umap_italic_50_0.5.npy` | Pre-computed UMAP positions (italic) |
| `images_cache/` | Folder of per-book WebP letter images (see below) |

Weight matrices (`w_*`) are only loaded when the edge cache hasn't been pre-computed yet. After the first run, they can be removed to save space.

### Image directory layout

```
images_cache/
├── images_cache_meta.pkl            # pickled dict: {"all_letters": ["a", "b", …]}
├── BNE_1000_891_T-55340-18/         # one folder per book (name = book id)
│   ├── a_001.webp                   # <letter>_<index>.webp
│   ├── e_002.webp
│   └── o_003.webp
└── BNE_1001_615_T-55281-18/
    ├── a_001.webp
    └── s_002.webp
```

- **Folder names** must match the identifiers in `books_dashboard_ordered.npy` exactly.
- **Images** must be WebP format. The filename convention is `<letter>_<index>.webp`.
- **`images_cache_meta.pkl`** is a pickled Python dict with at least the key `"all_letters"` containing the list of letter labels present in the corpus.
- In `config.yaml` you can choose `loading_strategy: preload` (all images loaded into RAM at startup) or `lazy` (images served on demand — faster startup, less memory).

## Configuration

All tuneable parameters live in [`config.yaml`](config.yaml). Key sections:

- **`server`** — port, host, debug mode, URL prefix.
- **`data`** — paths to every `.npy` file (change if your files are elsewhere).
- **`images`** — cache directory and loading strategy (`preload` vs `lazy`).
- **`network`** — `top_k` edges, `n_bins` opacity bins, default slider values.
- **`umap`** — `n_neighbors` and `min_dist` used in UMAP filenames.
- **`cache`** — figure/edge cache size limits.

## Project structure

```
├── book_similarity_dashboard.py   # Main dashboard class (~3 600 lines)
├── launch_dashboard.py            # Entry point — creates app & loads data
├── config.yaml                    # All tuneable parameters
├── requirements.txt               # Pinned Python dependencies
├── Dockerfile                     # Production container image
├── .dockerignore
├── .gitignore
├── assets/
│   ├── style.css                  # Custom slider & UI styling
│   └── legend-click-helpers.js    # Legend double-click isolation
├── images_cache/                  # Pre-extracted letter images (WebP)
│   └── BNE_<id>/                  # One folder per book
├── scripts/
│   ├── preflight.py               # Verify data files & pre-generate caches
│   ├── export_static_site.py      # Export dashboard as static HTML
│   ├── remove_pngs.py            # Cleanup utility (PNG → WebP migration)
│   └── verify_figs.py            # Quick smoke test for figure generation
└── *.npy                          # Data matrices (see table above)
```

## How it works

1. **`launch_dashboard.py`** reads `config.yaml`, creates a `BookSimilarityDashboard` instance, and calls `set_data()` to load the numpy matrices.
2. **Network graph** — the top-k most similar book pairs are shown as edges. Node positions come from pre-computed UMAP embeddings. Edges are binned by weight for opacity rendering.
3. **Heatmap** — the full N1-hat similarity matrix is shown, ordered by hierarchical clustering (single-linkage dendrogram).
4. **Letter comparison** — clicking two books shows their extracted letter images side-by-side.
5. **Dendrogram** — cuts the linkage tree at a user-chosen level to show printer-coloured groups.

## Where to change things

| Want to… | Look at… |
|----------|----------|
| Change default slider values | `config.yaml` → `network` section |
| Use different data files | `config.yaml` → `data` section |
| Adjust UMAP layout | `config.yaml` → `umap` section, then recompute `.npy` files |
| Change colour palette | `_get_printer_colors_dict()` in `book_similarity_dashboard.py` |
| Modify the UI layout | `_setup_layout()` in `book_similarity_dashboard.py` |
| Add/change callbacks | `_setup_callbacks()` in `book_similarity_dashboard.py` |
| Serve on a different port | `config.yaml` → `server.port` |
