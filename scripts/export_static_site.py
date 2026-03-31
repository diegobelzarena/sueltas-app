"""
Export a lightweight static site (HTML + assets) from the dashboard data.

- Does NOT modify the running dashboard code.
- Avoids computing UMAPs: only includes existing cached UMAP .npy files.
- Exports Plotly figure JSONs for heatmap and networks using cached edges and UMAP positions.
- Extracts per-book letter images from the image cache pickles into `static_site/images/`.

Usage:
    conda activate dash
    python scripts/export_static_site.py --outdir static_site

Notes:
- Keep an eye on output size when many images are embedded.
- The generated site can be served by any static host (GitHub Pages, Netlify, S3).
"""

import os
import json
import glob
import pickle
import base64
import argparse
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import io

ROOT = Path(__file__).resolve().parents[1]
IMAGES_CACHE_DIR = ROOT / 'data' / 'images'
OUT_DEFAULT = ROOT / 'static_site'

FONTS = ['combined', 'roman', 'italic']


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)


def load_npy_if_exists(path):
    try:
        return np.load(path)
    except Exception:
        return None


def build_heatmap_fig(n1hat_matrix, books, impr_names):
    """Build heatmap figure that matches the dashboard layout exactly."""
    fig = go.Figure()
    
    # Main heatmap trace
    fig.add_trace(go.Heatmap(
        z=n1hat_matrix.tolist(), 
        x=books.tolist(), 
        y=books.tolist(), 
        colorscale='viridis', 
        showscale=False,
        hoverongaps=False,
        hovertemplate='Book 1: %{y}<br>Book 2: %{x}<br>Shared Types: %{z:.0f}<extra></extra>'
    ))

    # Add printer marker traces (one per printer, matching dashboard)
    unique_imprs = [impr for impr in np.unique(impr_names) if impr not in ['n. nan', 'm. missing', 'Unknown']]
    colors = ["#C93232", "#34B5AC", "#1D4C57", "#21754E", "#755D11", "#772777", "#DFB13D", "#C9459F", "#F09D55", "#487BA0", "#6A3A3A", "#5AAE54", "#B15928"]
    markers = ['circle', 'square', 'triangle-up', 'diamond']
    
    for i, impr in enumerate(unique_imprs):
        mask = (impr_names == impr)
        if not np.any(mask):
            continue
        fig.add_trace(go.Scatter(
            x=books[mask].tolist(), 
            y=books[mask].tolist(), 
            mode='markers', 
            marker=dict(
                symbol=markers[i % len(markers)], 
                color=colors[i % len(colors)], 
                size=6,
                line=dict(color='white', width=1)
            ), 
            name=impr, 
            showlegend=True,
            legendgroup=impr
        ))

    # Layout matching dashboard exactly
    fig.update_layout(
        title=None,
        autosize=True,
        uirevision='constant',
        margin=dict(l=25, r=25, t=15, b=15),
        plot_bgcolor='#F8F5EC',
        paper_bgcolor='#F8F5EC',
        xaxis=dict(
            title="",
            side="bottom",
            showgrid=False,
            showticklabels=False,
            automargin=False,
            fixedrange=False,
            constrain='domain',
            categoryorder='array',
            categoryarray=books.tolist()
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            showticklabels=False,
            autorange='reversed',
            constrain='domain',
            automargin=False,
            fixedrange=False,
            categoryorder='array',
            categoryarray=books.tolist(),
            scaleanchor='x',
            scaleratio=1
        ),
        legend=dict(
            title=dict(
                text="<b>Printers:</b>",
                font=dict(size=10, family="Inter, Arial, sans-serif", color="#887C57"),
                side="left"
            ),
            orientation="h",
            xanchor='center',
            x=0.5,
            y=0.02,
            yanchor='top',
            bgcolor="rgba(248,245,236,0.9)",
            borderwidth=0,
            font=dict(size=9, family="Inter, Arial, sans-serif", color="#374151"),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            tracegroupgap=1
        )
    )
    return fig


def build_network_fig(umap_positions, binned_edges, books, impr_names):
    """Build network graph figure matching dashboard style exactly."""
    fig = go.Figure()

    # Add edges binned by bin_idx
    for bin_idx in sorted(map(int, binned_edges.keys())):
        bin_data = binned_edges.get(bin_idx, {'edges': [], 'avg_w': 0})
        edges = bin_data['edges']
        avg_w = bin_data.get('avg_w', 0)
        if not edges:
            continue
        x_all = []
        y_all = []
        for i, j in edges:
            x0, y0 = float(umap_positions[i, 0]), float(umap_positions[i, 1])
            x1, y1 = float(umap_positions[j, 0]), float(umap_positions[j, 1])
            x_all.extend([x0, x1, None])
            y_all.extend([y0, y1, None])
        opacity = min(1.0, avg_w)
        fig.add_trace(go.Scatter(
            x=x_all, 
            y=y_all, 
            mode='lines', 
            line=dict(width=1, color=f'rgba(100,100,100,{opacity})'), 
            showlegend=False, 
            hoverinfo='skip', 
            name=f'bin_{bin_idx}'
        ))

    # Add nodes colored by printer (matching dashboard)
    unique_imprs = [impr for impr in np.unique(impr_names) if impr not in ['n. nan', 'm. missing', 'Unknown']]
    colors = ["#C93232", "#34B5AC", "#1D4C57", "#21754E", "#755D11", "#772777", "#DFB13D", "#C9459F", "#F09D55", "#487BA0", "#6A3A3A", "#5AAE54", "#B15928"]
    markers = ['circle', 'square', 'triangle-up']
    
    for i, impr in enumerate(unique_imprs):
        mask = (impr_names == impr)
        if not np.any(mask):
            continue
        idxs = np.where(mask)[0]
        xs = umap_positions[mask][:, 0].tolist()
        ys = umap_positions[mask][:, 1].tolist()
        texts = [books[int(j)] for j in idxs]
        hover_texts = [f"Printer: {impr}<br>Book: {books[int(j)]}" for j in idxs]
        
        fig.add_trace(go.Scatter(
            x=xs, 
            y=ys, 
            mode='markers+text', 
            marker=dict(
                symbol=markers[i % len(markers)], 
                size=12, 
                color=colors[i % len(colors)],
                line=dict(color='white', width=1)
            ), 
            text=texts, 
            textposition='top center',
            textfont=dict(size=8, family="Inter, Arial, sans-serif"),
            name=impr, 
            showlegend=True,
            legendgroup=impr,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        ))

    # Add unknown/missing printers in gray if present
    unknown_mask = np.isin(impr_names, ['n. nan', 'm. missing', 'Unknown'])
    if np.any(unknown_mask):
        idxs = np.where(unknown_mask)[0]
        xs = umap_positions[unknown_mask][:, 0].tolist()
        ys = umap_positions[unknown_mask][:, 1].tolist()
        texts = [books[int(j)] for j in idxs]
        fig.add_trace(go.Scatter(
            x=xs, 
            y=ys, 
            mode='markers', 
            marker=dict(symbol='x', size=10, color='rgba(128,128,128,0.5)'), 
            text=texts,
            name='Unknown/Missing', 
            showlegend=True,
            hovertemplate='Book: %{text}<extra></extra>'
        ))

    # Layout matching dashboard
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            fixedrange=False
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            fixedrange=False
        ),
        legend=dict(
            title=dict(
                text="<b>Printers:</b>",
                font=dict(size=10, family="Inter, Arial, sans-serif", color="#887C57"),
                side="left"
            ),
            orientation="h",
            xanchor='center',
            x=0.5,
            y=-0.02,
            yanchor='top',
            bgcolor="rgba(255,255,255,0.9)",
            borderwidth=0,
            font=dict(size=9, family="Inter, Arial, sans-serif", color="#374151")
        )
    )
    return fig


def export_images(out_images_dir):
    # images_cache_meta.pkl contains index metadata
    meta_pkl = IMAGES_CACHE_DIR / 'images_cache_meta.pkl'
    index = {}
    if not meta_pkl.exists():
        print('Warning: images_cache_meta.pkl not found, skipping images export')
        return index

    with open(meta_pkl, 'rb') as f:
        meta = pickle.load(f)

    # meta contains 'book_index' mapping and 'all_letters'
    book_index = meta.get('book_index', {})

    ensure_dir(out_images_dir)

    # Iterate per-book pickle files images_{book}.pkl
    for book, entries in book_index.items():
        # expected filename
        book_pkl = IMAGES_CACHE_DIR / f'images_{book}.pkl'
        if not book_pkl.exists():
            # skip
            continue
        with open(book_pkl, 'rb') as f:
            book_data = pickle.load(f)
        out_book_dir = IMAGES_CACHE_DIR / book
        ensure_dir(out_book_dir)
        index[book] = {}
        for (ft, letter), images in book_data.items():
            # images: list of (path, base64)
            index[book].setdefault(letter, [])
            for i, (img_path, encoded) in enumerate(images):
                # Save as WebP to reduce size and improve web performance
                # Use `upper-` for uppercase and `lower-` for lowercase letters to avoid case-insensitive path collisions on Windows
                # Include font prefix in filename to indicate font type (e.g., 'italic_upper-A_1.webp')
                case = 'upper' if str(letter).isupper() else 'lower'
                filename = f"{ft}_{case}-{letter}_{i}.webp"
                out_path = out_book_dir / filename
                try:
                    img_bytes = base64.b64decode(encoded)
                    im = Image.open(io.BytesIO(img_bytes))
                    # Ensure a compatible mode for WebP (preserve alpha if present)
                    if im.mode not in ("RGB", "RGBA"):
                        im = im.convert("RGBA")
                    # Use reasonable default quality; tune if needed (lossy). For lossless use lossless=True
                    im.save(out_path, format='WEBP', quality=80, method=6)
                    index[book][letter].append(str(Path(IMAGES_CACHE_DIR) / book / filename))
                except Exception as e:
                    print(f"Warning: failed to write image {out_path}: {e}")
    # Write index json
    return index


def find_umap_files():
    # Returns dict font -> list of (filename, n_neighbors, min_dist)
    res = {}
    for font in FONTS:
        pattern = str(ROOT / f'umap_{font}_*.npy')
        files = glob.glob(pattern)
        res[font] = files
    return res


def find_edge_cache_files():
    # Search images_cache for edges cache files
    res = {}
    for font in FONTS:
        pattern = str(IMAGES_CACHE_DIR / f'edges_cache_{font}_*.pkl')
        files = glob.glob(pattern)
        res[font] = files
    return res


def main(outdir=OUT_DEFAULT):
    outdir = Path(outdir)
    ensure_dir(outdir)
    ensure_dir(outdir / 'plots')
    ensure_dir(outdir / 'umap')
    ensure_dir(outdir / 'edges')
    ensure_dir(outdir / 'images')
    ensure_dir(outdir / 'js')

    # Copy existing dashboard stylesheet if present to keep visual parity
    try:
        import shutil
        src_css = ROOT / 'assets' / 'style.css'
        if src_css.exists():
            shutil.copy(src_css, outdir / 'style.css')
            print(f'Copied dashboard stylesheet to {outdir / "style.css"}')
    except Exception as e:
        print(f'Warning: could not copy style.css: {e}')

    # Load basic arrays
    books = load_npy_if_exists(ROOT / 'books_dashboard_ordered.npy')
    impr_names = load_npy_if_exists(ROOT / 'impr_names_dashboard_ordered.npy')
    n1hat_rm = load_npy_if_exists(ROOT / 'n1hat_rm_matrix_ordered.npy')
    n1hat_it = load_npy_if_exists(ROOT / 'n1hat_it_matrix_ordered.npy')
    if books is None or impr_names is None:
        print('ERROR: books or impr_names not found - aborting static export')
        return

    # Export heatmaps for fonts
    for font in FONTS:
        if font == 'roman':
            n1hat = n1hat_rm
        elif font == 'italic':
            n1hat = n1hat_it
        else:
            n1hat = (n1hat_rm + n1hat_it) / 2.0
        if n1hat is None:
            continue
        fig = build_heatmap_fig(n1hat, books, impr_names)
        path = outdir / 'plots' / f'heatmap_{font}.json'
        write_json(path, fig.to_plotly_json())
        print(f'Wrote heatmap json for {font}: {path}')

    # Write metadata (books, printers, letters) to help client-side filtering and comparison
    try:
        unique_printers = sorted([p for p in np.unique(impr_names) if p not in ['n. nan', 'm. missing']])
        all_letters = []
        # attempt to get all_letters from images_cache meta if present
        meta_pkl = IMAGES_CACHE_DIR / 'images_cache_meta.pkl'
        if meta_pkl.exists():
            try:
                with open(meta_pkl, 'rb') as f:
                    meta = pickle.load(f)
                all_letters = meta.get('all_letters', [])
            except Exception:
                all_letters = []
        meta_obj = {'books': books.tolist(), 'impr_names': [str(x) for x in impr_names.tolist()], 'unique_printers': unique_printers, 'all_letters': all_letters}
        write_json(outdir / 'meta.json', meta_obj)
        print(f'Wrote meta.json with {len(meta_obj["books"])} books and {len(unique_printers)} printers')
    except Exception as e:
        print(f'Warning: failed to write meta.json: {e}')

    # UMAP files - copy numpy into JSON (list) WITHOUT recomputing
    umap_files = find_umap_files()
    available_umap_sources = set()
    for font, files in umap_files.items():
        for f in files:
            try:
                arr = np.load(f)
                filename = Path(f).stem + '.json'
                write_json(outdir / 'umap' / filename, arr.tolist())
                print(f'Wrote UMAP positions {f} -> {outdir / "umap" / filename}')
                available_umap_sources.add(Path(f).stem)  # substring like umap_combined_50_0.5
            except Exception as e:
                print(f'Warning: failed to include umap file {f}: {e}')

    # Edge caches: dump binned_edges (keep small)
    edge_files = find_edge_cache_files()
    for font, files in edge_files.items():
        for f in files:
            try:
                with open(f, 'rb') as fh:
                    data = pickle.load(fh)
                binned = data.get('binned_edges', {})
                # compress binned to list-of-dicts with edge indices only
                simple_binned = {}
                for k, v in binned.items():
                    simple_binned[int(k)] = {'avg_w': float(v.get('avg_w', 0)), 'edges': [(int(a), int(b)) for (a, b) in v.get('edges', [])]}
                outname = Path(f).stem + '.json'
                write_json(outdir / 'edges' / outname, simple_binned)
                print(f'Wrote edges {f} -> {outdir / "edges" / outname}')
            except Exception as e:
                print(f'Warning: failed to export edges from {f}: {e}')

    # Build network figures for every combination of edge font and available UMAP source
    # For each edge font, pick edge cache file if exists (first one)
    for edge_font in FONTS:
        efiles = edge_files.get(edge_font, [])
        binned = {}
        if efiles:
            try:
                with open(efiles[0], 'rb') as fh:
                    data = pickle.load(fh)
                binned = data.get('binned_edges', {})
            except Exception as e:
                print(f'Warning: failed to load edge cache {efiles[0]}: {e}')
        # For each umap source file available (any font)
        for genome in available_umap_sources:
            # genome string like umap_combined_50_0.5
            # try to load corresponding JSON we wrote
            umap_json = outdir / 'umap' / (genome + '.json')
            if not umap_json.exists():
                continue
            try:
                umap_arr = np.array(json.load(open(umap_json)))
                if umap_arr.size == 0:
                    continue
                fig = build_network_fig(umap_arr, binned, books, impr_names)
                name = f'network_edges_{edge_font}_pos_{genome}.json'
                write_json(outdir / 'plots' / name, fig.to_plotly_json())
                print(f'Wrote network plot: {name}')
            except Exception as e:
                print(f'Warning: failed to build network {edge_font} with umap {genome}: {e}')

    # Export images
    images_index = export_images(outdir / 'images')
    write_json(outdir / 'images' / 'images_index.json', images_index)
    print(f'Wrote images index with {len(images_index)} books')

    # Write a minimal index.html and JS loader
    index_html = OUT_HTML_TEMPLATE
    with open(outdir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)
    print(f'Wrote index.html to {outdir / "index.html"}')
    # Write app JS
    with open(outdir / 'js' / 'site.js', 'w', encoding='utf-8') as f:
        f.write(INDEX_JS)
    print(f'Wrote JS to {outdir / "js/site.js"}')

    # Also copy meta and images index into top-level static folder for client to fetch
    try:
        if (outdir / 'images' / 'images_index.json').exists():
            # already written earlier
            pass
        print('Static site assets prepared')
    except Exception as e:
        print('Warning preparing final assets:', e)

    # Summary
    print('\nStatic site build complete. Serve the directory', outdir)


# HTML template matching the dashboard layout exactly
OUT_HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Theatre Chapbooks At Scale</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * { box-sizing: border-box; }
    body { font-family: Inter, Arial, sans-serif; background: #f5f3ed; margin: 0; padding: 20px; }
    
    /* Button styles matching dashboard */
    .btn-active {
      margin-right: 5px; padding: 8px 16px; font-size: 12px; font-weight: 500;
      font-family: Inter, Arial, sans-serif; background-color: #2f4a84; color: white;
      border: none; border-radius: 6px; cursor: pointer;
      display: inline-flex; align-items: center; justify-content: center;
      box-shadow: 0 1px 3px rgba(0,0,0,0.2); width: 100px;
    }
    .btn-inactive {
      margin-right: 5px; padding: 8px 16px; font-size: 12px; font-weight: 500;
      font-family: Inter, Arial, sans-serif; background-color: #DBD1B5; color: #5a5040;
      border: none; border-radius: 6px; cursor: pointer;
      display: inline-flex; align-items: center; justify-content: center;
      box-shadow: 0 1px 3px rgba(0,0,0,0.2); width: 100px;
    }
    .btn-small-active {
      padding: 4px 8px; font-size: 10px; font-weight: 500; font-family: Inter, Arial, sans-serif;
      background-color: #2f4a84; color: white; border: none; border-radius: 4px; cursor: pointer;
    }
    .btn-small-inactive {
      padding: 4px 8px; font-size: 10px; font-weight: 500; font-family: Inter, Arial, sans-serif;
      background-color: #DBD1B5; color: #5a5040; border: none; border-radius: 4px; cursor: pointer;
    }
    .btn-medium-active {
      padding: 8px 16px; font-size: 12px; font-weight: 500; font-family: Inter, Arial, sans-serif;
      background-color: #2f4a84; color: white; border: none; border-radius: 6px; cursor: pointer;
      display: inline-flex; align-items: center; justify-content: center;
      box-shadow: 0 1px 3px rgba(0,0,0,0.2); width: 110px;
    }
    .btn-medium-inactive {
      padding: 8px 16px; font-size: 12px; font-weight: 500; font-family: Inter, Arial, sans-serif;
      background-color: #DBD1B5; color: #5a5040; border: none; border-radius: 6px; cursor: pointer;
      display: inline-flex; align-items: center; justify-content: center;
      box-shadow: 0 1px 3px rgba(0,0,0,0.2); width: 110px;
    }
    
    /* Section styles */
    .section-header {
      text-align: center; padding: 6px 0; background-color: #F8F5EC;
      border-radius: 6px; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.15);
    }
    .section-header h3 {
      margin: 0; font-family: Inter, Arial, sans-serif; font-weight: 600;
      letter-spacing: 0.5px; color: #887C57;
    }
    .panel-bg { background-color: #DBD1B5; border-radius: 8px; padding: 8px; margin-bottom: 8px; }
    .inner-panel { background-color: #F8F5EC; border-radius: 8px; padding: 10px; }
    
    /* Dropdown styles */
    select {
      width: 100%; font-size: 11px; padding: 6px; border: 1px solid #d1c7ad;
      border-radius: 4px; background: white;
    }
    
    /* Label styles */
    .label-small {
      font-weight: 500; font-size: 11px; color: #5a5040; font-family: Inter, Arial, sans-serif;
    }
    .label-section {
      text-align: center; margin-bottom: 8px; font-family: Inter, Arial, sans-serif;
      font-weight: 600; font-size: 13px; color: #887C57;
    }
  </style>
</head>
<body>
  <!-- Page Title -->
  <div style="margin-bottom: 20px; padding: 20px 0;">
    <h1 style="text-align: center; margin: 0; font-family: Inter, Arial, sans-serif; font-weight: 700; font-size: 2.2rem; color: #374151; letter-spacing: -0.5px;">
      Theatre Chapbooks At Scale
    </h1>
    <p style="text-align: center; margin: 5px 0 0 0; font-family: Inter, Arial, sans-serif; font-weight: 400; font-size: 1rem; color: #887C57; letter-spacing: 0.5px;">
      A Statistical Comparative Analysis of Typography
    </p>
  </div>

  <!-- Font Type Control Panel -->
  <div style="margin-bottom: 15px; padding: 12px 20px; background-color: #DBD1B5; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.15);">
    <div style="position: relative; display: flex; align-items: center; justify-content: center;">
      <div style="position: absolute; left: 20px; top: 50%; transform: translateY(-50%);">
        <label style="font-size: 13px; font-weight: 600; color: #887C57; font-family: Inter, Arial, sans-serif;">Font type</label>
      </div>
      <div style="display: flex; justify-content: center; width: 100%;">
        <button id="font-combined-btn" class="btn-active">Combined</button>
        <button id="font-roman-btn" class="btn-inactive">Roman</button>
        <button id="font-italic-btn" class="btn-inactive" style="margin-right:0">Italic</button>
      </div>
    </div>
  </div>

  <!-- Main Content: Typographic Similarity Analysis -->
  <div style="width: 100%; background-color: #DBD1B5; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.15); padding: 10px; margin-bottom: 20px;">
    <div class="section-header">
      <h3>Typographic Similarity Analysis</h3>
    </div>
    
    <div style="display: flex; gap: 2%; align-items: flex-start; justify-content: space-between;">
      <!-- Left Column: Similarity Matrix (45%) -->
      <div style="flex: 1 1 45%; min-width: 0; max-width: 48%; box-sizing: border-box; background-color: #F8F5EC; border-radius: 8px; padding: 2px; overflow: hidden;">
        <div class="label-section">Similarity Matrix</div>
        <div style="margin-bottom: 8px; text-align: center;">
          <button id="hide-all-printers-btn" class="btn-medium-inactive" style="margin-right:5px">Hide Printers</button>
          <button id="show-all-printers-btn" class="btn-medium-active">Show Printers</button>
        </div>
        <div id="heatmap" style="width: 100%; aspect-ratio: 1 / 1; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.2); margin: 0 auto;"></div>
      </div>

      <!-- Right Column: Letter Comparison (45%) -->
      <div style="flex: 1 1 45%; min-width: 0; max-width: 48%; box-sizing: border-box; background-color: #F8F5EC; border-radius: 8px; padding: 10px; overflow: hidden;">
        <div class="label-section">Letter Comparison</div>
        
        <!-- Printer Filter -->
        <div class="panel-bg">
          <label class="label-small" style="margin-right: 10px;">Filter by printer:</label>
          <select id="printer-filter-dropdown" style="margin-top:4px"></select>
          <button id="select-all-printer-books-btn" class="btn-small-active" style="display:none; margin-top:5px;">Select all from this printer</button>
        </div>

        <!-- Book Selector -->
        <div class="panel-bg">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <label class="label-small">Select books:</label>
            <button id="clear-comparison-btn" class="btn-small-inactive">Clear</button>
          </div>
          <select id="additional-books-dropdown" multiple style="width:100%;min-height:80px;margin-top:4px"></select>
        </div>

        <!-- Letter Filter -->
        <div class="panel-bg">
          <label class="label-small" style="margin-right:5px">Filter:</label>
          <button id="select-all-letters" class="btn-small-active" style="margin-right:3px">All</button>
          <button id="select-no-letters" class="btn-small-inactive" style="margin-right:8px">None</button>
          <button id="select-lowercase" class="btn-small-active" style="margin-right:3px">a-z</button>
          <button id="select-uppercase" class="btn-small-active" style="margin-right:8px">A-Z</button>
          <div id="letter-filter" style="display:inline-block;font-size:11px;margin-top:4px"></div>
        </div>

        <!-- Letter Comparison Panel -->
        <div id="letter-comparison-panel" style="border: 1px solid #d1c7ad; border-radius: 6px; padding: 15px; background-color: #F8F5EC; min-height: 500px; max-height: 800px; overflow-y: auto;">
          <p style="text-align: center; color: #6b7280; margin-top: 200px; font-size: 13px; font-family: Inter, Arial, sans-serif;">
            Click on a cell in the similarity matrix or a node in the network graph
          </p>
        </div>
      </div>
    </div>
  </div>

  <!-- Network Graph Section -->
  <div style="width: 100%; background-color: #DBD1B5; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.15); padding: 5px;">
    <div class="section-header">
      <h3>Graph of Typographic Similarity</h3>
    </div>

    <!-- UMAP Position Source Selector -->
    <div style="text-align: center; margin-bottom: 10px;">
      <span style="font-size: 13px; font-weight: 600; color: #887C57; font-family: Inter, Arial, sans-serif; margin-right: 10px;">Node positions source:</span>
      <button id="umap-pos-combined-btn" class="btn-active">Combined</button>
      <button id="umap-pos-roman-btn" class="btn-inactive">Roman</button>
      <button id="umap-pos-italic-btn" class="btn-inactive" style="margin-right:0">Italic</button>
    </div>

    <!-- Network Controls Row -->
    <div style="margin-bottom: 5px; padding: 10px; background-color: #DBD1B5; border-radius: 8px; display: flex; flex-wrap: nowrap; align-items: stretch; justify-content: space-between;">
      <!-- Column 1: Hide/Show Buttons -->
      <div style="width: 30%; flex-shrink: 0; display: flex; align-items: center; justify-content: center;">
        <div style="display: inline-block; text-align: center;">
          <button id="hide-all-labels-btn" class="btn-medium-inactive" style="margin-right:5px;margin-bottom:8px;width:120px">Hide Labels</button>
          <button id="show-all-labels-btn" class="btn-medium-active" style="margin-bottom:8px;width:120px">Show Labels</button>
          <br>
          <button id="hide-all-markers-btn" class="btn-medium-inactive" style="margin-right:5px;margin-bottom:8px;width:120px">Hide Markers</button>
          <button id="show-all-markers-btn" class="btn-medium-active" style="margin-bottom:8px;width:120px">Show Markers</button>
          <br>
          <button id="hide-all-network-printers-btn" class="btn-medium-inactive" style="margin-right:5px;margin-bottom:8px;width:120px">Hide Printers</button>
          <button id="show-all-network-printers-btn" class="btn-medium-active" style="margin-bottom:8px;width:120px">Show Printers</button>
          <br>
          <button id="hide-selected-books-btn" class="btn-medium-active" style="margin-right:5px;margin-bottom:8px;width:120px">Hide Selected</button>
          <button id="show-selected-books-btn" class="btn-medium-inactive" style="margin-bottom:8px;width:120px">Show Selected</button>
        </div>
      </div>

      <!-- Column 2: Sliders -->
      <div style="width: 30%; flex-shrink: 0; display: flex; flex-direction: column; justify-content: center; align-items: center;">
        <div style="width: 85%;">
          <div style="margin-bottom: 8px;">
            <label style="font-size: 11px; font-weight: 500; color: dimgray; font-family: Inter, Arial, sans-serif;">Edge Opacity:</label>
            <input type="range" id="edge-opacity-slider" min="0" max="2" step="0.1" value="1" style="width:100%">
          </div>
          <div style="margin-bottom: 8px;">
            <label style="font-size: 11px; font-weight: 500; color: dimgray; font-family: Inter, Arial, sans-serif;">Node Size:</label>
            <input type="range" id="node-size-slider" min="6" max="24" step="1" value="12" style="width:100%">
          </div>
          <div>
            <label style="font-size: 11px; font-weight: 500; color: dimgray; font-family: Inter, Arial, sans-serif;">Label Size:</label>
            <input type="range" id="label-size-slider" min="6" max="24" step="1" value="8" style="width:100%">
          </div>
        </div>
      </div>

      <!-- Column 3: Placeholder for UMAP params (skipped per user request) -->
      <div style="width: 30%; flex-shrink: 0; display: flex; align-items: center; justify-content: center;">
        <div style="background-color: #F8F5EC; border-radius: 8px; padding: 15px; text-align: center; width: 100%;">
          <div style="font-weight: 600; font-size: 11px; color: #887C57; font-family: Inter, Arial, sans-serif;">
            Static Export
          </div>
          <p style="font-size: 10px; color: #5a5040; margin: 8px 0 0 0;">
            UMAP recalculation is not available in static mode.
          </p>
        </div>
      </div>
    </div>

    <div id="network" style="height: 800px; margin-top: 8px; border-radius: 8px; overflow: hidden; background: white;"></div>
  </div>

  <script src="js/site.js"></script>
</body>
</html>"""


# Client-side JavaScript matching dashboard functionality
INDEX_JS = """
(async function(){
  // Directories for precomputed assets
  const plotsDir = 'plots';
  const umapDir = 'umap';
  const imagesIndexPath = 'images/images_index.json';

  // DOM elements
  const heatmapDiv = document.getElementById('heatmap');
  const networkDiv = document.getElementById('network');
  const letterPanel = document.getElementById('letter-comparison-panel');

  // State
  let currentFont = 'combined';
  let currentUmapSource = 'combined';
  let selectedBooks = [];
  let meta = null;
  let imagesIndex = null;
  let selectedLetters = [];

  // Button references
  const fontButtons = {
    combined: document.getElementById('font-combined-btn'),
    roman: document.getElementById('font-roman-btn'),
    italic: document.getElementById('font-italic-btn')
  };
  const umapButtons = {
    combined: document.getElementById('umap-pos-combined-btn'),
    roman: document.getElementById('umap-pos-roman-btn'),
    italic: document.getElementById('umap-pos-italic-btn')
  };

  // Helper to fetch JSON or return null
  async function fetchJSON(path) {
    try {
      const r = await fetch(path);
      if (r.ok) return await r.json();
    } catch(e) {}
    return null;
  }

  // Load all heatmaps
  const heatmaps = {
    combined: await fetchJSON(plotsDir + '/heatmap_combined.json'),
    roman: await fetchJSON(plotsDir + '/heatmap_roman.json'),
    italic: await fetchJSON(plotsDir + '/heatmap_italic.json')
  };

  // Load metadata
  meta = await fetchJSON('meta.json') || { books: [], impr_names: [], unique_printers: [], all_letters: [] };
  imagesIndex = await fetchJSON(imagesIndexPath) || {};

  // Update button styles
  function setButtonActive(btn, isActive) {
    if (isActive) {
      btn.className = 'btn-active';
    } else {
      btn.className = 'btn-inactive';
    }
  }

  function updateFontButtons() {
    Object.keys(fontButtons).forEach(k => setButtonActive(fontButtons[k], k === currentFont));
  }

  function updateUmapButtons() {
    Object.keys(umapButtons).forEach(k => setButtonActive(umapButtons[k], k === currentUmapSource));
  }

  // Render heatmap
  function renderHeatmap(font) {
    const data = heatmaps[font];
    if (data) {
      Plotly.react(heatmapDiv, data.data, data.layout, {responsive: true});
    }
  }

  // Load and render network
  async function renderNetwork(edgeFont, umapSource) {
    const umapFile = 'umap_' + umapSource + '_50_0.5';
    const plotFile = plotsDir + '/network_edges_' + edgeFont + '_pos_' + umapFile + '.json';
    const data = await fetchJSON(plotFile);
    if (data) {
      // Apply slider values
      const opacity = parseFloat(document.getElementById('edge-opacity-slider').value);
      const nodeSize = parseInt(document.getElementById('node-size-slider').value);
      const labelSize = parseInt(document.getElementById('label-size-slider').value);
      
      // Modify traces based on slider values
      data.data.forEach(trace => {
        if (trace.mode && trace.mode.includes('markers')) {
          if (trace.marker) trace.marker.size = nodeSize;
          if (trace.textfont) trace.textfont.size = labelSize;
        }
        if (trace.mode === 'lines' && trace.line) {
          // Adjust opacity in line color
          const color = trace.line.color || 'rgba(100,100,100,0.5)';
          const match = color.match(/rgba?\\(([^)]+)\\)/);
          if (match) {
            const parts = match[1].split(',').map(s => s.trim());
            if (parts.length >= 3) {
              const newOpacity = Math.min(1, (parseFloat(parts[3] || 0.5) * opacity));
              trace.line.color = 'rgba(' + parts.slice(0,3).join(',') + ',' + newOpacity + ')';
            }
          }
        }
      });
      
      Plotly.react(networkDiv, data.data, data.layout, {responsive: true});
    } else {
      networkDiv.innerHTML = '<div style="padding:40px;color:#777;text-align:center">No precomputed network for this combination</div>';
    }
  }

  // Populate UI controls
  function populateControls() {
    // Printer dropdown
    const printerDropdown = document.getElementById('printer-filter-dropdown');
    printerDropdown.innerHTML = '<option value="">All printers</option>';
    meta.unique_printers.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p;
      opt.textContent = p;
      printerDropdown.appendChild(opt);
    });

    // Books dropdown
    const booksDropdown = document.getElementById('additional-books-dropdown');
    booksDropdown.innerHTML = '';
    meta.books.forEach(b => {
      const opt = document.createElement('option');
      opt.value = b;
      opt.textContent = b;
      booksDropdown.appendChild(opt);
    });

    // Letter filter checkboxes
    const letterFilter = document.getElementById('letter-filter');
    letterFilter.innerHTML = '';
    meta.all_letters.forEach(l => {
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.id = 'letter_' + l;
      cb.value = l;
      cb.checked = true;
      cb.style.marginRight = '2px';
      cb.style.marginLeft = '6px';
      const lbl = document.createElement('label');
      lbl.htmlFor = cb.id;
      lbl.textContent = l;
      lbl.style.marginRight = '4px';
      letterFilter.appendChild(cb);
      letterFilter.appendChild(lbl);
    });
    // Initialize selected letters
    selectedLetters = [...meta.all_letters];
  }

  // Get currently selected letters from checkboxes
  function getSelectedLetters() {
    const cbs = document.querySelectorAll('#letter-filter input[type=checkbox]:checked');
    return Array.from(cbs).map(cb => cb.value);
  }

  // Render letter images for selected books
  function renderLetterImages() {
    letterPanel.innerHTML = '';
    if (selectedBooks.length === 0) {
      letterPanel.innerHTML = '<p style="text-align: center; color: #6b7280; margin-top: 200px; font-size: 13px;">Click on a cell in the similarity matrix or a node in the network graph</p>';
      return;
    }

    const letters = getSelectedLetters();
    if (letters.length === 0) {
      letterPanel.innerHTML = '<p style="text-align:center;color:#6b7280;margin-top:40px;">No letters selected</p>';
      return;
    }

    // Group images by letter
    letters.forEach(letter => {
      const letterDiv = document.createElement('div');
      letterDiv.style.marginBottom = '20px';
      
      const letterHeader = document.createElement('h4');
      letterHeader.textContent = letter;
      letterHeader.style.margin = '0 0 8px 0';
      letterHeader.style.color = '#374151';
      letterHeader.style.fontFamily = 'Inter, Arial, sans-serif';
      letterDiv.appendChild(letterHeader);

      const imagesRow = document.createElement('div');
      imagesRow.style.display = 'flex';
      imagesRow.style.flexWrap = 'wrap';
      imagesRow.style.gap = '10px';

      selectedBooks.forEach(book => {
        const bookImages = imagesIndex[book];
        if (bookImages && bookImages[letter]) {
          const bookDiv = document.createElement('div');
          bookDiv.style.textAlign = 'center';
          
          const bookLabel = document.createElement('div');
          bookLabel.textContent = book;
          bookLabel.style.fontSize = '10px';
          bookLabel.style.color = '#5a5040';
          bookLabel.style.marginBottom = '4px';
          bookDiv.appendChild(bookLabel);

          bookImages[letter].forEach(imgPath => {
            const img = document.createElement('img');
            img.src = imgPath;
            img.style.height = '60px';
            img.style.margin = '2px';
            img.style.border = '1px solid #d1c7ad';
            img.style.borderRadius = '4px';
            bookDiv.appendChild(img);
          });

          imagesRow.appendChild(bookDiv);
        }
      });

      letterDiv.appendChild(imagesRow);
      letterPanel.appendChild(letterDiv);
    });
  }

  // Event handlers
  // Font type buttons
  fontButtons.combined.addEventListener('click', () => { currentFont = 'combined'; updateFontButtons(); renderHeatmap(currentFont); renderNetwork(currentFont, currentUmapSource); });
  fontButtons.roman.addEventListener('click', () => { currentFont = 'roman'; updateFontButtons(); renderHeatmap(currentFont); renderNetwork(currentFont, currentUmapSource); });
  fontButtons.italic.addEventListener('click', () => { currentFont = 'italic'; updateFontButtons(); renderHeatmap(currentFont); renderNetwork(currentFont, currentUmapSource); });

  // UMAP position source buttons
  umapButtons.combined.addEventListener('click', () => { currentUmapSource = 'combined'; updateUmapButtons(); renderNetwork(currentFont, currentUmapSource); });
  umapButtons.roman.addEventListener('click', () => { currentUmapSource = 'roman'; updateUmapButtons(); renderNetwork(currentFont, currentUmapSource); });
  umapButtons.italic.addEventListener('click', () => { currentUmapSource = 'italic'; updateUmapButtons(); renderNetwork(currentFont, currentUmapSource); });

  // Heatmap printer visibility
  document.getElementById('hide-all-printers-btn').addEventListener('click', () => {
    const traces = heatmapDiv.data || [];
    const visibility = traces.map((t, i) => i === 0 ? true : 'legendonly');
    Plotly.restyle(heatmapDiv, {visible: visibility});
  });
  document.getElementById('show-all-printers-btn').addEventListener('click', () => {
    Plotly.restyle(heatmapDiv, {visible: true});
  });

  // Network controls
  document.getElementById('hide-all-labels-btn').addEventListener('click', () => {
    Plotly.restyle(networkDiv, {mode: 'markers'}, networkDiv.data.map((t, i) => t.mode && t.mode.includes('text') ? i : null).filter(x => x !== null));
  });
  document.getElementById('show-all-labels-btn').addEventListener('click', () => {
    Plotly.restyle(networkDiv, {mode: 'markers+text'}, networkDiv.data.map((t, i) => t.mode && t.mode.includes('markers') ? i : null).filter(x => x !== null));
  });
  document.getElementById('hide-all-markers-btn').addEventListener('click', () => {
    const indices = networkDiv.data.map((t, i) => t.mode && t.mode.includes('markers') ? i : null).filter(x => x !== null);
    Plotly.restyle(networkDiv, {visible: 'legendonly'}, indices);
  });
  document.getElementById('show-all-markers-btn').addEventListener('click', () => {
    const indices = networkDiv.data.map((t, i) => t.mode && t.mode.includes('markers') ? i : null).filter(x => x !== null);
    Plotly.restyle(networkDiv, {visible: true}, indices);
  });
  document.getElementById('hide-all-network-printers-btn').addEventListener('click', () => {
    const indices = networkDiv.data.map((t, i) => t.showlegend === true ? i : null).filter(x => x !== null);
    Plotly.restyle(networkDiv, {visible: 'legendonly'}, indices);
  });
  document.getElementById('show-all-network-printers-btn').addEventListener('click', () => {
    const indices = networkDiv.data.map((t, i) => t.showlegend === true ? i : null).filter(x => x !== null);
    Plotly.restyle(networkDiv, {visible: true}, indices);
  });
  document.getElementById('hide-selected-books-btn').addEventListener('click', () => {
    const indices = networkDiv.data.map((t, i) => t.name && t.name.startsWith('selected_') ? i : null).filter(x => x !== null);
    Plotly.restyle(networkDiv, {visible: 'legendonly'}, indices);
  });
  document.getElementById('show-selected-books-btn').addEventListener('click', () => {
    const indices = networkDiv.data.map((t, i) => t.name && t.name.startsWith('selected_') ? i : null).filter(x => x !== null);
    Plotly.restyle(networkDiv, {visible: true}, indices);
  });

  // Sliders
  document.getElementById('edge-opacity-slider').addEventListener('input', () => renderNetwork(currentFont, currentUmapSource));
  document.getElementById('node-size-slider').addEventListener('input', () => {
    const size = parseInt(document.getElementById('node-size-slider').value);
    const indices = networkDiv.data.map((t, i) => t.mode && t.mode.includes('markers') ? i : null).filter(x => x !== null);
    Plotly.restyle(networkDiv, {'marker.size': size}, indices);
  });
  document.getElementById('label-size-slider').addEventListener('input', () => {
    const size = parseInt(document.getElementById('label-size-slider').value);
    const indices = networkDiv.data.map((t, i) => t.mode && t.mode.includes('text') ? i : null).filter(x => x !== null);
    Plotly.restyle(networkDiv, {'textfont.size': size}, indices);
  });

  // Printer filter dropdown
  document.getElementById('printer-filter-dropdown').addEventListener('change', (e) => {
    const printer = e.target.value;
    const booksDropdown = document.getElementById('additional-books-dropdown');
    const selectAllBtn = document.getElementById('select-all-printer-books-btn');
    
    booksDropdown.innerHTML = '';
    let filteredBooks = meta.books;
    if (printer) {
      filteredBooks = meta.books.filter((b, i) => meta.impr_names[i] === printer);
      selectAllBtn.style.display = 'inline-block';
    } else {
      selectAllBtn.style.display = 'none';
    }
    filteredBooks.forEach(b => {
      const opt = document.createElement('option');
      opt.value = b;
      opt.textContent = b;
      booksDropdown.appendChild(opt);
    });
  });

  // Select all printer books button
  document.getElementById('select-all-printer-books-btn').addEventListener('click', () => {
    const booksDropdown = document.getElementById('additional-books-dropdown');
    Array.from(booksDropdown.options).forEach(opt => opt.selected = true);
    selectedBooks = Array.from(booksDropdown.selectedOptions).map(o => o.value);
    renderLetterImages();
  });

  // Books dropdown selection
  document.getElementById('additional-books-dropdown').addEventListener('change', () => {
    const booksDropdown = document.getElementById('additional-books-dropdown');
    selectedBooks = Array.from(booksDropdown.selectedOptions).map(o => o.value);
    renderLetterImages();
  });

  // Clear button
  document.getElementById('clear-comparison-btn').addEventListener('click', () => {
    document.getElementById('additional-books-dropdown').selectedIndex = -1;
    selectedBooks = [];
    renderLetterImages();
  });

  // Letter filter buttons
  document.getElementById('select-all-letters').addEventListener('click', () => {
    document.querySelectorAll('#letter-filter input[type=checkbox]').forEach(cb => cb.checked = true);
    renderLetterImages();
  });
  document.getElementById('select-no-letters').addEventListener('click', () => {
    document.querySelectorAll('#letter-filter input[type=checkbox]').forEach(cb => cb.checked = false);
    renderLetterImages();
  });
  document.getElementById('select-lowercase').addEventListener('click', () => {
    document.querySelectorAll('#letter-filter input[type=checkbox]').forEach(cb => {
      cb.checked = cb.value === cb.value.toLowerCase() && cb.value !== cb.value.toUpperCase();
    });
    renderLetterImages();
  });
  document.getElementById('select-uppercase').addEventListener('click', () => {
    document.querySelectorAll('#letter-filter input[type=checkbox]').forEach(cb => {
      cb.checked = cb.value === cb.value.toUpperCase() && cb.value !== cb.value.toLowerCase();
    });
    renderLetterImages();
  });

  // Letter checkbox changes
  document.getElementById('letter-filter').addEventListener('change', () => {
    renderLetterImages();
  });

  // Heatmap click - select books
  heatmapDiv.addEventListener('plotly_click', function(data) {
    try {
      const pt = data.detail.points[0];
      if (pt && pt.x && pt.y) {
        const book1 = pt.x;
        const book2 = pt.y;
        // Add both books if not already selected
        if (!selectedBooks.includes(book1)) selectedBooks.push(book1);
        if (book1 !== book2 && !selectedBooks.includes(book2)) selectedBooks.push(book2);
        // Update dropdown
        const booksDropdown = document.getElementById('additional-books-dropdown');
        Array.from(booksDropdown.options).forEach(opt => {
          opt.selected = selectedBooks.includes(opt.value);
        });
        renderLetterImages();
      }
    } catch(e) { console.warn('heatmap click error', e); }
  });

  // Network click - select books
  networkDiv.addEventListener('plotly_click', function(data) {
    try {
      const pt = data.detail.points[0];
      if (pt) {
        let book = pt.text || (pt.customdata && pt.customdata.split('<br>')[0]) || null;
        if (book && !selectedBooks.includes(book)) {
          selectedBooks.push(book);
          const booksDropdown = document.getElementById('additional-books-dropdown');
          Array.from(booksDropdown.options).forEach(opt => {
            opt.selected = selectedBooks.includes(opt.value);
          });
          renderLetterImages();
        }
      }
    } catch(e) { console.warn('network click error', e); }
  });

  // Initialize
  populateControls();
  updateFontButtons();
  updateUmapButtons();
  renderHeatmap(currentFont);
  await renderNetwork(currentFont, currentUmapSource);
})();
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a static site from dashboard data')
    parser.add_argument('--outdir', default=str(OUT_DEFAULT), help='Output directory for static site')
    args = parser.parse_args()
    main(args.outdir)
