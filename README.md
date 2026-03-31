# sueltas-app

Interactive dashboard for exploring typographic similarity across a corpus of historical books (BNE *sueltas* collection). Built with **Plotly Dash**.

## Quick start

```bash
# 1. Clone the repository (matrices & CSV are included)
git clone https://github.com/<owner>/sueltas-app.git
cd sueltas-app

# 2. Download the image archive and extract it
#    → see "Image data" below for the download link
tar xzf images_cache.tar.gz -C data/images

# 3. Create / activate a Python 3.11 environment
conda create -n dash python=3.11 -y
conda activate dash

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the preflight check (verifies files & builds caches)
python scripts/preflight.py

# 6. Launch
python launch_dashboard.py
# → open http://localhost:8050
```

### Quick start with Docker

```bash
git clone https://github.com/<owner>/sueltas-app.git
cd sueltas-app
# download images_cache.tar.gz into the repo root (see below)
docker build -t sueltas-app .
docker run -d -p 8050:8050 sueltas-app
# → open http://localhost:8050
```

## Data files

The repository ships with all numerical data needed to run the dashboard:

```
data/                                  ← checked into git
├── n1hat_it_matrix_ordered.npy        # Italic similarity matrix (books × books)
├── n1hat_rm_matrix_ordered.npy        # Roman similarity matrix (books × books)
├── books_dashboard_ordered.npy        # Array of book identifiers
├── impr_names_dashboard_ordered.npy   # Printer names per book
├── symbs_dashboard.npy                # Symbol / character labels
├── w_rm_matrix_ordered.npy            # Weight matrix — roman
├── w_it_matrix_ordered.npy            # Weight matrix — italic
├── Corpus_600.csv                     # Corpus metadata
├── umap_combined_50_0.5.npy           # UMAP positions — combined      [auto-generated]
├── umap_roman_50_0.5.npy              # UMAP positions — roman          [auto-generated]
├── umap_italic_50_0.5.npy             # UMAP positions — italic         [auto-generated]
└── images/                            ← NOT in git — see below
    ├── images_cache_meta.pkl          # Image metadata (pickled dict)
    ├── edges_cache_*.pkl              #                                 [auto-generated]
    ├── BNE_1000_891_T-55340-18/
    │   ├── roman_lower-a_001.webp
    │   └── ...
    └── BNE_1001_615_T-55281-18/
        └── ...
```

### Image data

The `data/images/` directory (~610 book folders, 69 000 WebP images, ~10 MB compressed) is **not** tracked in git. Download the archive and extract it:

| Download | Link |
|----------|------|
| `images_cache.tar.gz` | **[TODO: add Zenodo / Google Drive link]** |

```bash
# Create the directory and extract
mkdir -p data/images          # (PowerShell: New-Item -ItemType Directory -Force data/images)
tar xzf images_cache.tar.gz -C data/images
```

The archive already includes pre-computed edge caches (`edges_cache_*.pkl`) and image metadata (`images_cache_meta.pkl`), so `preflight.py` will not need to regenerate them.

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

The Docker image bundles everything into a single portable container. Since the `.npy` matrices and CSV are already in the repo, you only need the `images_cache.tar.gz` archive in the repo root before building.

### Clone → Build → Run

```bash
git clone https://github.com/<owner>/sueltas-app.git
cd sueltas-app

# Download images_cache.tar.gz into the repo root
# (see "Image data" above for the download link)

docker build -t sueltas-app .
docker run -d -p 8050:8050 sueltas-app
# → open http://localhost:8050
```

Or with Docker Compose:

```bash
docker compose up -d
```

### What goes into the image

| Layer | Source | How |
|-------|--------|-----|
| Python code, `config.yaml`, `assets/` | repo | `COPY . ./` |
| `.npy` matrices & CSV (`data/`) | repo | `COPY . ./` |
| Letter images + caches (`data/images/`) | `images_cache.tar.gz` | `ADD` (auto-extracts) |

The `.dockerignore` excludes `data/images/` (raw folder), `scripts/`, `docs/`, `.git/`, and other dev files — only the archive is used for images.

### Serving behind a reverse proxy

```bash
docker run -d -p 8050:8050 -e REQUESTS_PATHNAME_PREFIX=/sueltas/ sueltas-app
```

### Rebuilding the image archive

If you modify images locally, recreate the archive before rebuilding:

```bash
# (optional) regenerate edge caches first so they're baked in
python scripts/preflight.py

tar czf images_cache.tar.gz -C data/images .
docker build -t sueltas-app .
```

| What changed | What to do |
|--------------|------------|
| Images in `data/images/` | Re-create `images_cache.tar.gz`, then `docker build` |
| `.npy` matrices in `data/` | Just `docker build` |
| Python code or config | Just `docker build` |

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
├── data/                          # Matrices & CSV (tracked); images (git-ignored)
│   ├── *.npy                      # Similarity / weight / UMAP matrices
│   ├── Corpus_600.csv             # Corpus metadata
│   └── images/                    # Per-book WebP letter images + caches (download separately)
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
