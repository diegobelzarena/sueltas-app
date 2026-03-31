# sueltas-app

Interactive dashboard for exploring typographic similarity across a corpus of historical books (BNE *sueltas* collection). Built with **Plotly Dash**.

## Quick start

```bash
# 1. Create / activate a Python 3.11 environment
conda create -n dash python=3.11 -y
conda activate dash

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your data files in the data/ directory (see "Data files" below)

# 4. Run the preflight check (verifies files & builds caches)
python scripts/preflight.py

# 5. Launch
python launch_dashboard.py
# → open http://localhost:8050
```

## Data files

All data lives under the `data/` directory (git-ignored). Place your files there before running the app:

```
data/
├── n1hat_it_matrix_ordered.npy     # Italic similarity matrix (books × books)    [required]
├── n1hat_rm_matrix_ordered.npy     # Roman similarity matrix (books × books)     [required]
├── books_dashboard_ordered.npy     # Array of book identifiers                   [required]
├── impr_names_dashboard_ordered.npy# Printer names per book                      [required]
├── symbs_dashboard.npy             # Symbol / character labels                   [required]
├── w_rm_matrix_ordered.npy         # Weight matrix — roman                       [optional*]
├── w_it_matrix_ordered.npy         # Weight matrix — italic                      [optional*]
├── umap_combined_50_0.5.npy        # UMAP positions — combined                   [auto-generated]
├── umap_roman_50_0.5.npy           # UMAP positions — roman                      [auto-generated]
├── umap_italic_50_0.5.npy          # UMAP positions — italic                     [auto-generated]
└── images/                         # Per-book WebP letter images
    ├── images_cache_meta.pkl       # Image metadata (pickled dict)               [required]
    ├── edges_cache_*.pkl           #                                             [auto-generated]
    ├── BNE_1000_891_T-55340-18/
    │   ├── roman_lower-a_001.webp
    │   └── ...
    └── BNE_1001_615_T-55281-18/
        └── ...
```

\* Weight matrices are only needed if edge caches haven't been generated yet. After the first `preflight.py` run they can be removed to save space.

### Image conventions

- **Folder names** must match the identifiers in `books_dashboard_ordered.npy` exactly.
- **Images** must be WebP format: `<font>_<case>-<letter>_<index>.webp` (e.g. `roman_lower-a_001.webp`).
- **`images_cache_meta.pkl`** is a pickled dict with at least `{"all_letters": [...]}`.
- In `config.yaml` you can choose `loading_strategy: preload` (all images in RAM at startup) or `lazy` (served on demand — faster startup, less memory).

## Preflight check

Before the first launch, run the preflight script to verify data and build caches:

```bash
python scripts/preflight.py              # full check + generate all caches
python scripts/preflight.py --skip-umap  # skip UMAP computation (slow)
python scripts/preflight.py --config path/to/config.yaml
```

The script will:

1. **Check required files** — the 5 core `.npy` matrices and `images_cache_meta.pkl`. Exits with an error if any are missing.
2. **Check optional files** — weight matrices and UMAP caches.
3. **Verify image directories** — confirms each book has a matching subdirectory under `data/images/`.
4. **Generate edge caches** (`edges_cache_*.pkl`) from the weight matrices.
5. **Generate UMAP caches** (`umap_*.npy`) from the similarity matrices (requires `umap-learn`).

| Exit code | Meaning |
|-----------|---------|
| `0` | All checks passed. |
| `1` | Required files missing — the app cannot start. |
| `2` | Cache generation failed (app may still work but slower on first use). |

> For a detailed breakdown of every file dependency, see [`docs/FILE_DEPENDENCIES.md`](docs/FILE_DEPENDENCIES.md).

## Docker

```bash
# Prep: compress images for baking into the container
tar czf images_cache.tar.gz -C data/images .

# Build & run
docker build -t sueltas-app .
docker run -p 8050:8050 sueltas-app
```

To serve under a URL prefix (reverse proxy):

```bash
docker run -e REQUESTS_PATHNAME_PREFIX=/sueltas/ -p 8050:8050 sueltas-app
```

## Configuration

All tuneable parameters live in [`config.yaml`](config.yaml). Key sections:

| Section | Controls |
|---------|----------|
| `server` | Port, host, debug mode, URL prefix |
| `data` | Paths to all `.npy` files |
| `images` | Image cache directory, loading strategy (`preload` / `lazy`) |
| `network` | `top_k` edges, `n_bins` opacity bins, default slider values |
| `umap` | `n_neighbors` and `min_dist` hyper-parameters |
| `dendrogram` | Default cut level, truncation |
| `cache` | Figure/edge cache size limits |

## Project structure

```
sueltas-app/
├── launch_dashboard.py            # Entry point — loads config & starts the app
├── book_similarity_dashboard.py   # Main dashboard class
├── umap_utils.py                  # UMAP loading / computation helpers
├── config.yaml                    # All tuneable parameters
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Production container image
├── docker-compose.yml
├── LICENSE                        # MIT License
├── assets/
│   ├── style.css                  # Dashboard styling
│   └── legend-click-helpers.js    # Legend interaction helpers
├── data/                          # Data files (git-ignored, user-provided)
│   ├── *.npy                      # Similarity / weight / UMAP matrices
│   └── images/                    # Per-book WebP letter images + metadata
├── scripts/
│   ├── preflight.py               # Verify data & pre-generate caches
│   ├── export_static_site.py      # Export as static HTML site
│   ├── remove_pngs.py            # PNG → WebP cleanup utility
│   └── verify_figs.py            # Smoke test for figure generation
└── docs/
    └── FILE_DEPENDENCIES.md       # Detailed file dependency documentation
```

## How it works

1. **`launch_dashboard.py`** reads `config.yaml`, creates a `BookSimilarityDashboard` instance, and calls `set_data()` to load the numpy matrices.
2. **Network graph** — the top-k most similar book pairs are shown as edges. Node positions come from UMAP embeddings. Edges are binned by weight for opacity rendering.
3. **Heatmap** — the full similarity matrix, ordered by hierarchical clustering (single-linkage dendrogram).
4. **Letter comparison** — clicking two books shows their extracted letter images side-by-side.
5. **Dendrogram** — cuts the linkage tree at a user-chosen level to show printer-coloured groups.

## Where to change things

| Want to… | Look at… |
|----------|----------|
| Change default slider values | `config.yaml` → `network` section |
| Use different data files | `config.yaml` → `data` section |
| Adjust UMAP layout | `config.yaml` → `umap` section, then re-run `scripts/preflight.py` |
| Change colour palette | `_get_printer_colors_dict()` in `book_similarity_dashboard.py` |
| Modify the UI layout | `_setup_layout()` in `book_similarity_dashboard.py` |
| Add/change callbacks | `_setup_callbacks()` in `book_similarity_dashboard.py` |
| Serve on a different port | `config.yaml` → `server.port` |

## License

[MIT](LICENSE)
