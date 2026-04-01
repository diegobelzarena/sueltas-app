import dash
from dash import dcc, html, callback, Input, Output, State, ALL
import plotly.graph_objects as go
import numpy as np
import pandas as pd
# SciPy imports used for dendrogram/linkage
from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram as scipy_dendrogram
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import json
import base64
import os
import glob
import pickle
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import tempfile
from flask import send_from_directory, abort
import time
import umap_utils

class BookSimilarityDashboard:
    # Cache size limits for memory optimization
    MAX_FIGURE_CACHE_SIZE = 10  # Limit total cached figure/aux entries
    MAX_UMAP_CACHE_SIZE = 5     # Limit cached UMAP parameter combinations

    # Default settings for edge cache behavior
    EDGE_CACHE_MAX_FILES = 3  # Keep at most 3 edge cache files by default (LRU)

    def __init__(self, requests_pathname_prefix='', config=None):
        """Initialize dashboard.

        requests_pathname_prefix: optional URL prefix where the app is mounted
        (e.g. '/investigacion/grupos/gti/sueltas/'). When provided, pass it
        to Dash so URLs and `_dash-config` are generated correctly.
        config: optional dict loaded from config.yaml (see config.yaml for keys).
        """
        cfg = config or {}
        net_cfg = cfg.get("network", {})
        cache_cfg = cfg.get("cache", {})
        img_cfg = cfg.get("images", {})
        data_cfg = cfg.get("data", {})

        self.config = cfg
        self.books = None
        self.impr_names = None
        self.symbs = None
        self.n1hat_rm = None
        self.n1hat_it = None
        self.top_k = net_cfg.get("top_k", 5000)
        self.n_bins = net_cfg.get("n_bins", 10)

        self.MAX_FIGURE_CACHE_SIZE = cache_cfg.get("max_figure_entries", 10)
        self.MAX_UMAP_CACHE_SIZE = cache_cfg.get("max_umap_entries", 5)
        self.EDGE_CACHE_MAX_FILES = cache_cfg.get("edge_cache_max_files", 3)

        # Resolve the data directory from config
        self._data_dir = data_cfg.get("dir", "./data")

        # Serve images from static folder
        self.letter_images_path = os.path.join(self._data_dir, 'images')
        self._cache_dir = img_cfg.get("cache_dir", os.path.join(self._data_dir, 'images'))
        self._image_loading_strategy = img_cfg.get("loading_strategy", "preload")
        os.makedirs(self._cache_dir, exist_ok=True)

        # Button styles for toggle buttons (network panel)
        self._active_btn_style = {
            'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
            'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
            'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
            'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
            'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1',
            'width': '120px', 'minWidth': '120px', 'maxWidth': '120px'
        }
        self._inactive_btn_style = {
            'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
            'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
            'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
            'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
            'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1',
            'width': '120px', 'minWidth': '120px', 'maxWidth': '120px'
        }

        # Track last selected font to know when to reload edge caches
        self._last_font_type = None

        # Lock to protect concurrent cache writes
        self._edge_cache_lock = threading.Lock()

        # Option: keep all caches in memory if True (useful for debugging)
        self.KEEP_ALL_EDGE_CACHES = False

        # Pass the prefix into the Dash constructor (allowed at init time)
        dash_kwargs = {"suppress_callback_exceptions": True}
        if requests_pathname_prefix:
            dash_kwargs["requests_pathname_prefix"] = requests_pathname_prefix
            dash_kwargs["routes_pathname_prefix"] = requests_pathname_prefix
        self.app = dash.Dash(__name__, **dash_kwargs)
        self.app.layout = html.Div("Loading data, please wait…")
        # Register route to serve pre-extracted images from the images_cache folder
        self._register_image_route()
        
    def _resolve_data_path(self, key, default_name):
        """Resolve a data file path: explicit override or default under _data_dir."""
        data_cfg = self.config.get("data", {})
        override = data_cfg.get(key)
        if override and os.path.isabs(override):
            return override
        if override:
            # If override looks like a full relative path (contains os.sep or /), use as-is
            if os.sep in override or '/' in override:
                return override
            # Otherwise treat as filename relative to data dir
            return os.path.join(self._data_dir, override)
        return os.path.join(self._data_dir, default_name)

    def set_data(self):
        """
        Initialize the dashboard with your data.
        """
        umap_cfg = self.config.get("umap", {})
        self._umap_n_neighbors = umap_cfg.get("n_neighbors", 50)
        self._umap_min_dist = umap_cfg.get("min_dist", 0.5)

        self.n1hat_it = np.load(self._resolve_data_path("n1hat_it_matrix", "n1hat_it_matrix_ordered.npy"), mmap_mode='r')
        self.n1hat_rm = np.load(self._resolve_data_path("n1hat_rm_matrix", "n1hat_rm_matrix_ordered.npy"), mmap_mode='r')
        self.books = np.load(self._resolve_data_path("books", "books_dashboard_ordered.npy"), mmap_mode='r')
        self.impr_names = np.load(self._resolve_data_path("impr_names", "impr_names_dashboard_ordered.npy"), mmap_mode='r')
        self.symbs = np.load(self._resolve_data_path("symbs", "symbs_dashboard.npy"), mmap_mode='r')
        print(" Loaded ordered .npy files")

        # Decide whether to load weight memmaps: only if edge caches are missing
        fonts = ['roman', 'italic', 'combined']
        need_edges = any(not os.path.exists(self._edge_cache_path(font, self.top_k, self.n_bins)) for font in fonts)
        # Check for per-font combined UMAP file (we only load combined positions at startup if present)
        combined_umap_file = umap_utils.umap_filename('combined', self._umap_n_neighbors, self._umap_min_dist, data_dir=self._data_dir)
        combined_umap_present = os.path.exists(combined_umap_file)

        # Check the dtype of n1hat_rm, n1hat_it
        print(f"n1hat_rm dtype: {self.n1hat_rm.dtype if self.n1hat_rm is not None else 'None'}, n1hat_it dtype: {self.n1hat_it.dtype if self.n1hat_it is not None else 'None'}")

        # Precompute SciPy linkages (single-link) for dendrograms (roman, italic, combined).
        # Convert strength -> distance by subtracting from max and use single linkage to mimic level-set connectivity.
        try:
            max_rm = float(np.max(self.n1hat_rm))
            max_it = float(np.max(self.n1hat_it))
            combined = (self.n1hat_rm + self.n1hat_it).astype(float)
            max_combined = float(np.max(combined))

            def _compute_linkage_from_strength(mat):
                # Convert strengths to distances
                maxv = float(np.max(mat))
                dist = maxv - mat.astype(float)
                np.fill_diagonal(dist, 0.0)
                condensed = squareform(dist)
                lk = scipy_linkage(condensed, method='single')
                return lk

            self.n1hat_linkage_rm = _compute_linkage_from_strength(self.n1hat_rm)
            self.n1hat_linkage_it = _compute_linkage_from_strength(self.n1hat_it)
            self.n1hat_linkage_combined = _compute_linkage_from_strength(combined)
            self.n1hat_levels_max = int(max(max_rm, max_it, max_combined))
            print(f"Precomputed linkages (rm/it/combined). levels_max={self.n1hat_levels_max}")
        except Exception as e:
            # If SciPy linkage fails (memory/time), fall back to None and compute on demand later
            self.n1hat_linkage_rm = None
            self.n1hat_linkage_it = None
            self.n1hat_linkage_combined = None
            self.n1hat_levels_max = 22  # sensible default
            print("Warning: failed to precompute linkages:", e)

        # Load weight matrices as memory-mapped (cheap — needed for edges and UMAP)
        try:
            self._w_rm_mmap = np.load(self._resolve_data_path("w_rm_matrix", "w_rm_matrix_ordered.npy"), mmap_mode='r')
            self._w_it_mmap = np.load(self._resolve_data_path("w_it_matrix", "w_it_matrix_ordered.npy"), mmap_mode='r')
            print("Loaded weight matrices as memory-mapped.")
        except Exception as e:
            self._w_rm_mmap = None
            self._w_it_mmap = None
            print(f"Warning: could not load weight matrices: {e}")

        if need_edges:
            print("Edge caches missing — will compute from weight matrices.")
        else:
            print("Edge caches present.")

        # Initialize printer markers dictionary (lightweight, no trace objects)
        self._printer_markers = self._get_printer_colors_dict()
                                
        # Image cache removed for memory efficiency
        # Images will be loaded on-demand without caching
        self._load_metadata_for_letter_images()  # 3 images per letter

        # Setup layout. Load combined UMAP positions (computed via umap-learn if missing).
        self._setup_layout()
        self._setup_callbacks()
        
    def _precompute_edges(self, w_rm, w_it, top_k=10000, n_bins=10):
        """Backward-compatible wrapper — compute edges for all fonts (kept for explicit use).

        Prefer using _ensure_precomputed_edges(font_type) which will lazily load or compute and cache
        per-font edges without computing all fonts at startup.
        """
        # Compute edges for all font_types if explicitly requested
        for font in ['roman', 'italic', 'combined']:
            self._compute_edges_for_font(font, w_rm, w_it, top_k=top_k, n_bins=n_bins)

    def _edge_cache_path(self, font_type, top_k, n_bins):
        """Return cache path for given font/top_k/n_bins."""
        safe_name = f"edges_cache_{font_type}_top{top_k}_bins{n_bins}.pkl"
        return os.path.join(self._cache_dir, safe_name)

    def _ensure_precomputed_edges(self, font_type='combined', top_k=None, n_bins=None):
        """Ensure _top_edges and _binned_edges exist for a given font_type.

        Loads from cache if present, otherwise computes and saves to cache. Keeps only one font in memory
        by default to limit memory use. Returns (top_edges, binned_edges).
        """
        if top_k is None:
            top_k = self.top_k
        if n_bins is None:
            n_bins = self.n_bins

        if getattr(self, '_top_edges', None) is None:
            self._top_edges = {}
        if getattr(self, '_binned_edges', None) is None:
            self._binned_edges = {}

        # If already have it in memory, return
        if font_type in self._top_edges and len(self._top_edges[font_type]) > 0:
            return self._top_edges[font_type], self._binned_edges.get(font_type, {})

        cache_path = self._edge_cache_path(font_type, top_k, n_bins)
        # Try load from cache atomically
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                self._top_edges[font_type] = data.get('top_edges', [])
                self._binned_edges[font_type] = data.get('binned_edges', {})
                print(f"Loaded edge cache for font '{font_type}' from {cache_path}")
                # Optionally prune other fonts from memory to keep memory low
                if not getattr(self, 'KEEP_ALL_EDGE_CACHES', False):
                    for k in list(self._top_edges.keys()):
                        if k != font_type:
                            self._top_edges.pop(k, None)
                            self._binned_edges.pop(k, None)
                return self._top_edges[font_type], self._binned_edges[font_type]
            except Exception as e:
                print(f"Warning: Failed to load edge cache {cache_path}: {e}")
                # Continue to compute if cache load fails
                pass

        # Compute and cache (with atomic write and LRU prunning)
        print(f"Computing edge cache for font '{font_type}' (top_k={top_k}, n_bins={n_bins})...")
        # Ensure weight memmaps are loaded (lazy load on demand)
        if getattr(self, '_w_rm_mmap', None) is None or getattr(self, '_w_it_mmap', None) is None:
            try:
                self._w_rm_mmap = np.load(self._resolve_data_path('w_rm_matrix', 'w_rm_matrix_ordered.npy'), mmap_mode='r')
                self._w_it_mmap = np.load(self._resolve_data_path('w_it_matrix', 'w_it_matrix_ordered.npy'), mmap_mode='r')
                print("Loaded weight matrices as memory-mapped (needed for edge computation).")
            except Exception as e:
                print(f"Warning: Failed to load weight memmaps for edges: {e}")
                raise

        top_edges, binned_edges = self._compute_edges_for_font(font_type, self._w_rm_mmap, self._w_it_mmap, top_k=top_k, n_bins=n_bins)
        # Write atomically with lock
        try:
            tmp_name = None
            with self._edge_cache_lock:
                fd, tmp_name = tempfile.mkstemp(prefix='edges_cache_', dir=self._cache_dir)
                os.close(fd)
                with open(tmp_name, 'wb') as f:
                    pickle.dump({'top_edges': top_edges, 'binned_edges': binned_edges}, f, protocol=pickle.HIGHEST_PROTOCOL)
                # Atomic replace
                os.replace(tmp_name, cache_path)
                print(f"Saved edge cache to {cache_path}")
                # Prune old cache files if above limit (use LRU by mtime)
                if not getattr(self, 'KEEP_ALL_EDGE_CACHES', False):
                    try:
                        self._prune_edge_caches(self.EDGE_CACHE_MAX_FILES)
                    except Exception as e:
                        print(f"Warning: prune edge caches failed: {e}")
        except Exception as e:
            print(f"Warning: Could not write edge cache {cache_path}: {e}")
            try:
                if tmp_name and os.path.exists(tmp_name):
                    os.remove(tmp_name)
            except Exception:
                pass

        return top_edges, binned_edges

    def _compute_edges_for_font(self, font_type, w_rm, w_it, top_k=10000, n_bins=10):
        """Compute (and return) top_edges/binned_edges for a given font without creating a full combined matrix.

        This function uses upper-triangular indices and np.argpartition to find top-k efficiently and
        avoids creating large temporary full (n,n) arrays by operating on the upper triangle directly.
        """
        n = w_rm.shape[0]
        i_indices, j_indices = np.triu_indices(n, k=1)

        if font_type == 'roman':
            edge_weights = w_rm[i_indices, j_indices]
        elif font_type == 'italic':
            edge_weights = w_it[i_indices, j_indices]
        else:  # combined
            # Avoid materializing (w_rm + w_it) as a full matrix; compute only upper-triangle combined weights
            edge_weights = (w_rm[i_indices, j_indices] + w_it[i_indices, j_indices]) / 2.0

        if edge_weights.size == 0:
            self._top_edges[font_type] = []
            self._binned_edges[font_type] = {i: {'edges': [], 'avg_w': 0} for i in range(n_bins)}
            return self._top_edges[font_type], self._binned_edges[font_type]

        k = min(top_k, edge_weights.size)
        # Use argpartition to get unsorted top-k indices, then sort those descending
        idx_top = np.argpartition(-edge_weights, k - 1)[:k]
        idx_top_sorted = idx_top[np.argsort(edge_weights[idx_top])[::-1]]

        top_edges = [(int(i_indices[idx]), int(j_indices[idx])) for idx in idx_top_sorted]
        self._top_edges[font_type] = top_edges

        # Compute weights for top edges to bin them
        top_edge_weights = edge_weights[idx_top_sorted]
        if top_edge_weights.size == 0 or top_edge_weights.max() == top_edge_weights.min():
            binned = {i: {'edges': [], 'avg_w': 0} for i in range(n_bins)}
            self._binned_edges[font_type] = binned
            return top_edges, binned

        bins = np.linspace(top_edge_weights.min(), top_edge_weights.max(), n_bins + 1)
        bin_indices = np.digitize(top_edge_weights, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        binned = {}
        for bin_idx in range(n_bins):
            sel = np.where(bin_indices == bin_idx)[0]
            edges_in_bin = [top_edges[i] for i in sel]
            if edges_in_bin:
                # compute average weight from memmaps for these edges
                # Avoid large temporaries: compute weights in a small loop
                s = 0.0
                for (ii, jj) in edges_in_bin:
                    if font_type == 'roman':
                        s += float(w_rm[ii, jj])
                    elif font_type == 'italic':
                        s += float(w_it[ii, jj])
                    else:
                        s += float((w_rm[ii, jj] + w_it[ii, jj]) / 2.0)
                avg_w = float(s / len(edges_in_bin))
            else:
                avg_w = 0
            binned[bin_idx] = {'edges': edges_in_bin, 'avg_w': avg_w}

        self._binned_edges[font_type] = binned
        return top_edges, binned

    def _prune_edge_caches(self, max_files):
        """Prune oldest edge cache files so only `max_files` remain.

        Uses file modification time as LRU heuristic and removes oldest cache files first.
        """
        # Find all edge cache files
        pattern = os.path.join(self._cache_dir, 'edges_cache_*.pkl')
        files = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p))
        if len(files) <= max_files:
            return
        to_remove = files[:len(files) - max_files]
        for f in to_remove:
            try:
                os.remove(f)
                print(f"Pruned old edge cache: {f}")
            except Exception as e:
                print(f"Warning: failed to remove cache file {f}: {e}")
    
    def _get_printer_colors_dict(self):
        """Get lightweight printer color/marker dictionary (no trace objects)."""
        impr_to_color, impr_to_marker, unique_imprs = self._get_printer_colors()
        return {
            'colors': impr_to_color,
            'markers': impr_to_marker,
            'unique_imprinters': unique_imprs
        }
    
    def _build_printer_marker_traces(self, n1hat):
        """Build printer marker traces on-demand for heatmap."""
        traces = []
        types_diag = np.diagonal(n1hat)
        
        for impr in self._printer_markers['unique_imprinters']:
            mask = self.impr_names == impr
            if not np.any(mask):
                continue
            
            trace = go.Scatter(
                x=self.books[mask],
                y=self.books[mask],
                mode='markers',
                marker=dict(
                    symbol=self._printer_markers['markers'][impr],
                    size=6,
                    color=self._printer_markers['colors'][impr],
                    line=dict(color='white', width=1)
                ),
                showlegend=True,
                legendgroup=impr,
                name=impr,
                customdata=types_diag[mask][:, None],
                hovertemplate=f'Printer: {impr}<br>Book: %{{x}}<br>Types: %{{customdata[0]}}<extra></extra>'
            )
            traces.append(trace)
        return traces
    
                
    def _load_umap_positions(self, font_type='combined', w_rm=None, w_it=None, w_combined=None, n_neighbors=None, min_dist=None, compute_if_missing=True):
        """Load (or compute) UMAP positions.  Delegates to umap_utils.

        When the cached file is missing and *compute_if_missing* is True,
        a distance matrix is built from the weight matrices and passed to
        ``umap_utils.load_positions`` so UMAP can be computed on the fly
        (requires ``umap-learn``).
        """
        if n_neighbors is None:
            n_neighbors = getattr(self, '_umap_n_neighbors', 50)
        if min_dist is None:
            min_dist = getattr(self, '_umap_min_dist', 0.5)

        # Build distance matrix from weight matrices when computation may be needed
        distance_matrix = None
        if compute_if_missing:
            if w_rm is None:
                w_rm = getattr(self, '_w_rm_mmap', None)
            if w_it is None:
                w_it = getattr(self, '_w_it_mmap', None)
            if font_type == 'roman' and w_rm is not None:
                distance_matrix = np.clip(1 - np.asarray(w_rm, dtype=np.float32), 0, 1)
            elif font_type == 'italic' and w_it is not None:
                distance_matrix = np.clip(1 - np.asarray(w_it, dtype=np.float32), 0, 1)
            elif w_rm is not None and w_it is not None:
                avg = (np.asarray(w_rm, dtype=np.float32) + np.asarray(w_it, dtype=np.float32)) / 2
                distance_matrix = np.clip(1 - avg, 0, 1)
            elif w_rm is not None:
                distance_matrix = np.clip(1 - np.asarray(w_rm, dtype=np.float32), 0, 1)
            elif w_it is not None:
                distance_matrix = np.clip(1 - np.asarray(w_it, dtype=np.float32), 0, 1)

        return umap_utils.load_positions(
            font_type=font_type,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            compute_if_missing=compute_if_missing,
            distance_matrix=distance_matrix,
            data_dir=getattr(self, '_data_dir', './data'),
        )

    def _register_image_route(self):
        """Register a Flask route to serve images from the images_cache directory.

        The route will serve files at /images_cache/<book>/<filename>, where the
        files are read from self._cache_dir.
        """
        try:
            cache_dir_abs = os.path.abspath(self._cache_dir)

            @self.app.server.route('/images_cache/<path:filename>')
            def _serve_image(filename):
                # Security: ensure the requested file is inside the cache dir
                target = os.path.abspath(os.path.join(cache_dir_abs, filename))
                if not target.startswith(cache_dir_abs):
                    abort(404)
                # send_from_directory will set appropriate headers
                return send_from_directory(cache_dir_abs, filename)
        except Exception as e:
            print(f"Warning: Could not register image route: {e}")
        
    def _create_heatmap(self):
        """Create similarity matrix heatmap matching notebook style"""     
           
        n1hat = self.n1hat_rm + self.n1hat_it # Combined n1hat for hover info
        n_books = len(self.books)
        
        # Create the main heatmap
        # To avoid creating a large object-typed customdata array (which causes many small Python allocations),
        # we use a lightweight hovertemplate that shows only the book names and the shared types count.
        # Printer information remains available via diagonal marker traces created separately.

        fig = go.Figure(data=go.Heatmap(
            z=n1hat,
            x=self.books,
            y=self.books,
            colorscale='viridis',  # Match notebook colormap
            reversescale=False,    # Don't reverse (notebook doesn't)
            hoverongaps=False,
            hovertemplate=(
                'Book 1: %{y}<br>'
                'Book 2: %{x}<br>'
                'Shared Types: %{z:.0f}<extra></extra>'
            ),
            showscale=False  # Hide colorbar to avoid legend overlap
        ))
        

        # Add printer marker traces (built on-demand for memory efficiency)
        marker_traces = self._build_printer_marker_traces(n1hat)
        for trace in marker_traces:
            fig.add_trace(trace)
        
        # Clean layout - responsive sizing, no tick labels
        fig.update_layout(
            title=None,
            autosize=True,
            uirevision='constant',  # Preserve zoom/pan state and prevent layout recalculation
            xaxis=dict(
                title="",
                side="bottom",
                showgrid=False,
                showticklabels=False,  # Hide tick labels
                automargin=False,
                fixedrange=False,  # Allow zooming
                constrain='domain',
                categoryorder='array',
                categoryarray=self.books
            ),
            yaxis=dict(
                title="",
                showgrid=False,
                showticklabels=False,  # Hide tick labels
                autorange='reversed',  # Match notebook orientation
                constrain="domain",
                automargin=False,
                fixedrange=False,  # Allow zooming
                categoryorder='array',
                categoryarray=self.books,
                scaleanchor="x",  # Lock aspect ratio to x
                scaleratio=1     # 1:1 aspect ratio (square)
            ),
            margin=dict(l=25, r=25, t=15, b=15),  # Minimal margins, small space for legend
            plot_bgcolor='#F8F5EC',
            paper_bgcolor='#F8F5EC',
            # Legend configuration - horizontal at bottom, compact
            legend=dict(
                title=dict(
                    text="<b>Printers:</b>",
                    font=dict(size=10, family="Inter, Arial, sans-serif", color="#887C57"),
                    side="left"
                ),
                orientation="h",  # Horizontal layout
                xanchor='center',
                x=0.5,
                y=0.02,
                yanchor='top',
                bgcolor="rgba(248,245,236,0.9)",
                borderwidth=0,
                font=dict(size=9, family="Inter, Arial, sans-serif", color="#374151"),
                itemclick="toggle",
                itemdoubleclick="toggleothers",
                tracegroupgap=1,
            )
        )
        
        return fig
    
    
    def _load_metadata_for_letter_images(self):
        """
        Load global metadata (all_letters) and prepare image access.

        Respects self._image_loading_strategy from config.yaml:
          - "preload" (default): read every WebP into RAM as base64 for instant display.
          - "lazy": only build the directory index; images are served on-demand via Flask.
        """
        with open(f'{self._cache_dir}/images_cache_meta.pkl', 'rb') as f:
            self.meta = pickle.load(f)
        self._all_letters = self.meta.get('all_letters', [])
        print(f"Loaded metadata from images_cache_meta.pkl ({len(self._all_letters)} letters)")

        if self._image_loading_strategy == 'preload':
            self._preload_all_images()
        else:
            self._build_image_index()

    def _get_linkage_subtree_leaves(self, lk, node_id, n):
        """Return sorted list of original leaf indices under linkage node `node_id`."""
        if node_id < n:
            return [node_id]
        row = node_id - n
        if row < 0 or row >= len(lk):
            return []
        left, right = int(lk[row, 0]), int(lk[row, 1])
        return sorted(self._get_linkage_subtree_leaves(lk, left, n) +
                       self._get_linkage_subtree_leaves(lk, right, n))

    def _build_image_index(self):
        """Scan image directories to build _image_index without loading file contents.

        Used by the 'lazy' loading strategy.  Images are served on demand via
        the Flask route registered in _register_image_route().
        """
        self._image_cache = {}
        self._image_index = {}
        t0 = time.time()
        try:
            for entry in os.scandir(self._cache_dir):
                if not entry.is_dir():
                    continue
                book_files = [
                    fentry.name
                    for fentry in os.scandir(entry.path)
                    if fentry.is_file() and fentry.name.lower().endswith('.webp')
                ]
                self._image_index[entry.name] = book_files
            elapsed = time.time() - t0
            n_files = sum(len(v) for v in self._image_index.values())
            print(f"Built image index ({len(self._image_index)} books, {n_files} files) in {elapsed:.2f}s [lazy mode]")
        except Exception as e:
            print(f"Warning: Image index build failed: {e}")
            self._image_index = {}

    def _preload_all_images(self):
        """Preload all WebP images as base64 data URLs.
        
        Uses threaded parallel I/O to speed up reading many small files.
        For ~10MB of WebP files, this uses ~13-14MB of RAM (base64 overhead)
        but provides instant image display with no HTTP requests.
        """
        self._image_cache = {}  # {(book_name, filename): "data:image/webp;base64,..."}
        self._image_index = {}  # {book_name: [list of filenames]}
        
        t0 = time.time()
        
        try:
            # 1. Collect all (book_name, filepath, filename) using scandir (faster than listdir)
            all_tasks = []  # [(book_name, fpath, fname), ...]
            for entry in os.scandir(self._cache_dir):
                if not entry.is_dir():
                    continue
                book_name = entry.name
                book_files = []
                for fentry in os.scandir(entry.path):
                    if fentry.is_file() and fentry.name.lower().endswith('.webp'):
                        book_files.append(fentry.name)
                        all_tasks.append((book_name, fentry.path, fentry.name))
                self._image_index[book_name] = book_files
            
            n_books = len(self._image_index)
            n_files = len(all_tasks)
            print(f"Preloading {n_files} images from {n_books} book directories...")
            
            # 2. Read + encode in parallel threads (I/O-bound, threads work well)
            def _read_and_encode(task):
                book_name, fpath, fname = task
                with open(fpath, 'rb') as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode('ascii')
                return book_name, fname, len(data), f"data:image/webp;base64,{b64}"
            
            total_size = 0
            total_files = 0
            n_workers = min(16, max(4, os.cpu_count() or 4))
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                for result in executor.map(_read_and_encode, all_tasks):
                    book_name, fname, size, data_url = result
                    self._image_cache[(book_name, fname)] = data_url
                    total_size += size
                    total_files += 1
                    # Progress log every 500 files
                    if total_files % 500 == 0:
                        elapsed = time.time() - t0
                        print(f"  {total_files}/{n_files} files, {total_size/(1024*1024):.1f} MB ({elapsed:.1f}s)")
            
            elapsed = time.time() - t0
            mb_size = total_size / (1024 * 1024)
            print(f"Preloaded {total_files} images ({mb_size:.1f} MB raw, ~{mb_size * 1.37:.1f} MB in memory) in {elapsed:.2f}s [{n_workers} threads]")
        except Exception as e:
            print(f"Warning: Image preloading failed: {e}")
            self._image_cache = {}
            self._image_index = {}


    
    def _get_printer_colors(self):
        """Get color mapping for printers - matching heatmap colors/markers"""
        unique_imprs = [impr for impr in np.unique(self.impr_names) 
                       if impr not in ['n. nan', 'm. missing', 'Unknown']]
        
        # Use SAME colors and markers as heatmap diagonal markers
        markers = ['circle', 'square', 'triangle-up']
        colors = ["#C93232", "#34B5AC", "#1D4C57", "#21754E", "#755D11", "#772777", "#DFB13D", "#C9459F", "#F09D55", "#487BA0", "#6A3A3A", "#5AAE54", "#B15928"]
        
        impr_to_color = {}
        impr_to_marker = {}
        for i, impr in enumerate(unique_imprs):
            impr_to_color[impr] = colors[i % len(colors)]
            impr_to_marker[impr] = markers[i % len(markers)]
        
        # Gray for unknown/missing with 0.5 alpha
        impr_to_color['n. nan'] = 'rgba(128, 128, 128, 0.5)'
        impr_to_color['m. missing'] = 'rgba(128, 128, 128, 0.5)'
        impr_to_color['Unknown'] = 'rgba(128, 128, 128, 0.5)'
        impr_to_marker['n. nan'] = 'x'
        impr_to_marker['m. missing'] = 'x'
        impr_to_marker['Unknown'] = 'x'
        
        return impr_to_color, impr_to_marker, unique_imprs
    
    def _get_book_color(self, book_name):
        """Get color for a specific book from predefined color list"""
        colors = ["#C93232", "#34B5AC", "#1D4C57", "#21754E", "#755D11", "#772777", "#DFB13D", "#C9459F", "#F09D55", "#487BA0", "#6A3A3A", "#5AAE54", "#B15928"]
        book_list = list(self.books)
        if book_name in book_list:
            book_idx = book_list.index(book_name)
            return colors[book_idx % len(colors)]
        return colors[0]  # Default to first color

    def _create_network_graph(self, umap_positions=None, edge_opacity=1.0,marker_size=12,
                              label_size=8, font_type='combined'):
        """Create network graph from weight matrix with UMAP positioning and printer colors
        
        Args:
            umap_positions: numpy array or list of UMAP coordinates (will be converted if needed)
        """
        # Ensure edges are precomputed (lazy load/compute)
        try:
            self._ensure_precomputed_edges(font_type, top_k=self.top_k, n_bins=self.n_bins)
        except Exception as e:
            print(f"Warning: Edge precomputation failed: {e}")

        # Use precomputed top edges
        edges = self._top_edges.get(font_type, [])

        # If umap_positions is None, compute a fallback using networkx spring layout from edges
        if umap_positions is None:            
            print("Warning: no UMAP positions available, using zero positions")
            umap_positions = np.zeros((len(self.books), 2), dtype=np.float32)
        else:
            # Use asarray to avoid copying if already an array
            umap_positions = np.asarray(umap_positions, dtype=np.float32)
        
        if not edges:
            print(f"Warning: No edges found.")
            fig = go.Figure()
            fig.add_annotation(
                text=f"No connections<br>",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(title="No Network Connections Found")
            return fig
        
                
        # Get printer colors and markers (matching heatmap)
        impr_to_color = self._printer_markers['colors']
        impr_to_marker = self._printer_markers['markers']
        unique_imprs = self._printer_markers['unique_imprinters']
                
        
        # Create figure
        fig = go.Figure()
        
        # Add edges in binned traces for efficient opacity control
        bins = self._binned_edges.get(font_type, {})
        for bin_idx in range(10):  # n_bins
            bin_data = bins.get(bin_idx, {'edges': [], 'avg_w': 0})
            edges_in_bin = bin_data['edges']
            avg_w = bin_data['avg_w']
            if not edges_in_bin:
                continue
            x_all = []
            y_all = []
            for i, j in edges_in_bin:
                x0, y0 = umap_positions[i]
                x1, y1 = umap_positions[j]
                x_all.extend([x0, x1, None])
                y_all.extend([y0, y1, None])
            opacity = min(1.0, avg_w * edge_opacity)
            color = f'rgba(100,100,100,{opacity})'
            fig.add_trace(go.Scatter(
                x=x_all,
                y=y_all,
                mode='lines',
                line=dict(width=1, color=color),
                showlegend=False,
                name=f'bin_{bin_idx}',
                customdata=[avg_w],  # Store average weight for updating
                hoverinfo='skip'
            ))
        
        # Add nodes for unknown/missing printers with 0.5 alpha and no label
        unknown_mask = np.isin(self.impr_names, ['n. nan', 'm. missing', 'Unknown'])
        if np.any(unknown_mask):
            node_x = umap_positions[unknown_mask][:, 0].tolist()
            node_y = umap_positions[unknown_mask][:, 1].tolist()
            node_text = [f"{bk}<br>Printer: Unknown" 
                        for bk in self.books[unknown_mask]]
            # No label for unknown/missing printers
            node_labels = ['' for _ in range(unknown_mask.sum())]
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    symbol='circle',  # Circle marker for unknown
                    size=int(marker_size * 1 / 2),
                    color='rgba(128, 128, 128, 0.5)',  # 0.5 alpha
                    line=dict(width=1, color='white')
                ),
                text=node_labels,
                textposition='top center',
                textfont=dict(size=int(label_size * 1 / 2), color='rgba(128, 128, 128, 0.5)'),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=node_text,
                name='Unknown',
                legendgroup='Unknown',
                showlegend=True
            ))

        # Add nodes colored by printer, one trace per printer for legend
        # Using same colors and markers as heatmap diagonal
        for impr in unique_imprs:
            node_mask = self.impr_names == impr
            if not np.any(node_mask):
                continue
            
            node_x = umap_positions[node_mask][:, 0].tolist()
            node_y = umap_positions[node_mask][:, 1].tolist()
            node_text = [f"{self.books[i]}<br>Printer: {self.impr_names[i]}" 
                        for i in np.where(node_mask)[0]]
            # Show printer name as label on top of nodes
            node_labels = [impr for _ in np.where(node_mask)[0]]
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    symbol=impr_to_marker.get(impr, 'circle'),  # Match heatmap marker
                    size=marker_size,
                    color=impr_to_color[impr],
                    line=dict(width=1, color='white')
                ),
                text=node_labels,
                textposition='top center',
                textfont=dict(
                    size=label_size, 
                    color=impr_to_color[impr],
                    family='Arial, bold'  # Bold font
                ),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=node_text,
                name=impr,
                legendgroup=impr,
                showlegend=True
            ))
        
        
        fig.update_layout(
            title=None,
            showlegend=True,
            legend=dict(
                title=dict(
                    text="<span style='font-weight:600'>  Printers  </span>",
                    font=dict(size=13, family="Inter, Arial, sans-serif", color="#887C57")
                ),
                x=1.02,
                y=1,
                bgcolor="#F8F5EC",
                bordercolor="#d1c7ad",
                borderwidth=1,
                font=dict(size=11, family="Inter, Arial, sans-serif", color="#374151"),
                itemclick="toggle",
                itemdoubleclick="toggleothers",
                tracegroupgap=1,
                itemsizing="constant"

            ),
            hovermode="closest",
            margin=dict(b=20, l=5, r=160, t=20),
            annotations=[
                dict(
                    text="Node color = printer · Edge opacity controlled by slider",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.02,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(
                        color="#6b7280",
                        size=10,
                        family="Inter, Arial, sans-serif"
                    ),
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#F8F5EC",
            paper_bgcolor="#F8F5EC",
        )

        
        return fig

    def _orient_umap_positions(self, umap_positions):
        """Delegate to umap_utils.orient_positions."""
        return umap_utils.orient_positions(umap_positions)

    def _update_network_edges(self, current_fig, edge_opacity, umap_pos_array, font_type, selected_books=None):
        """Update edge traces in the network figure for new weight_matrix and threshold, keeping node traces.
        
        Args:
            umap_pos_array: numpy array of UMAP positions (not list)
            selected_books: list of selected book names to recreate edges for (default None)
        """
        # Ensure edges are computed/loaded for this font
        try:
            self._ensure_precomputed_edges(font_type, top_k=self.top_k, n_bins=self.n_bins)
        except Exception as e:
            print(f"Warning: _ensure_precomputed_edges failed: {e}")

        # Use binned edges for the font type
        bins = self._binned_edges.get(font_type, {})
        
        # Use Patch for efficient update
        patched = dash.Patch()
        
        # Use provided selected_books or default to empty list
        if selected_books is None:
            selected_books = []
        
        # Update bin traces and remove selected_edge_ traces
        for i, trace in enumerate(current_fig['data']):
            if trace['name'].startswith('bin_'):
                bin_idx = int(trace['name'].split('_')[1])
                bin_data = bins.get(bin_idx, {'edges': [], 'avg_w': 0})
                edges_in_bin = bin_data['edges']
                avg_w = bin_data['avg_w']
                if not edges_in_bin:
                    patched['data'][i]['x'] = []
                    patched['data'][i]['y'] = []
                    patched['data'][i]['line']['color'] = 'rgba(100,100,100,0)'
                else:
                    x_all = []
                    y_all = []
                    for edge_i, edge_j in edges_in_bin:
                        x0, y0 = umap_pos_array[edge_i]
                        x1, y1 = umap_pos_array[edge_j]
                        x_all.extend([x0, x1, None])
                        y_all.extend([y0, y1, None])
                    opacity = min(1.0, avg_w * edge_opacity)
                    color = f'rgba(100,100,100,{opacity})'
                    patched['data'][i]['x'] = x_all
                    patched['data'][i]['y'] = y_all
                    patched['data'][i]['line']['color'] = color
            elif trace['name'].startswith('selected_edge_'):
                # Remove selected edges - they will be recreated
                patched['data'][i]['x'] = []
                patched['data'][i]['y'] = []
        
        # Recreate selected book edges with new font type
        if selected_books:
            book_list = list(self.books)
            for book in selected_books:
                if book in book_list:
                    book_idx = book_list.index(book)
                    book_color = self._get_book_color(book)
                    
                    # Convert hex to RGB
                    r = int(book_color[1:3], 16)
                    g = int(book_color[3:5], 16)
                    b = int(book_color[5:7], 16)
                    
                    # Add edges for this book from binned edges
                    for bin_idx in range(10):
                        bin_data = bins.get(bin_idx, {'edges': [], 'avg_w': 0})
                        edges_in_bin = bin_data['edges']
                        avg_w = bin_data['avg_w']
                        
                        if not edges_in_bin:
                            continue
                        
                        # Vectorize edge filtering
                        edges_array = np.array(edges_in_bin)
                        mask = (edges_array[:, 0] == book_idx) | (edges_array[:, 1] == book_idx)
                        matching_edges = edges_array[mask]
                        
                        if len(matching_edges) > 0:
                            edges_x = []
                            edges_y = []
                            for i, j in matching_edges:
                                edges_x.extend([umap_pos_array[i, 0], umap_pos_array[j, 0], None])
                                edges_y.extend([umap_pos_array[i, 1], umap_pos_array[j, 1], None])
                            
                            # Selected edges use avg_w directly for opacity (not affected by edge_opacity slider)
                            opacity = min(1.0, avg_w)
                            color = f'rgba({r},{g},{b},{opacity})'
                            patched['data'].append({
                                'type': 'scatter',
                                'x': edges_x,
                                'y': edges_y,
                                'mode': 'lines',
                                'line': {'width': 1, 'color': color},
                                'showlegend': False,
                                'customdata': [avg_w],
                                'name': f'selected_edge_{book}_bin{bin_idx}',
                                'hoverinfo': 'skip'
                            })
        
        return patched
    
    def _get_available_letters_for_books(self, book1, book2, font_type='roman', letter_images_path='./letter_images/'):
        """Get list of letters that have images available for either of the two specified books using the metadata index."""
        letters = set()
        if font_type == 'combined':
            font_types_to_search = ['roman', 'italic']
        else:
            font_types_to_search = [font_type]

        # Use self.meta['book_index'] to get available (font, letter) pairs for each book
        for book in [book1, book2]:
            available = self.meta.get('book_index', {}).get(book, [])
            for font, letter in available:
                if font in font_types_to_search:
                    letters.add(letter)

        result = sorted(list(letters))
        return result
    
    def _get_available_letters_for_single_book(self, book, font_type='roman'):
        """Get list of letters that have images available for a single book using the metadata index."""
        letters = set()
        if font_type == 'combined':
            font_types_to_search = ['roman', 'italic']
        else:
            font_types_to_search = [font_type]

        available = self.meta.get('book_index', {}).get(book, [])
        for font, letter in available:
            if font in font_types_to_search:
                letters.add(letter)

        return sorted(list(letters))
    
    def _get_available_images(self, book_name, letter, font_type='roman'):
        """Get available images for a specific book and letter.

        Behavior (streamlined):
          - Uses preloaded image cache for instant access (no filesystem calls).
          - Returns base64 data URLs for immediate display.
          - Enforces strict font+case matching.
        """
        # Use preloaded index if available
        if hasattr(self, '_image_index') and book_name in self._image_index:
            files = self._image_index[book_name]
        else:
            # Fallback to directory scan if preloading failed
            book_dir = os.path.join(self._cache_dir, book_name)
            if not os.path.isdir(book_dir):
                return []
            try:
                files = [f for f in os.listdir(book_dir) if f.lower().endswith('.webp')]
            except Exception:
                return []

        if not files:
            return []

        # Determine font types to consider
        if font_type == 'combined':
            font_types_to_search = ['roman', 'italic']
        else:
            font_types_to_search = [font_type]

        # First, prefer font-prefixed files like 'italic_upper-A_1.webp' / 'roman_lower-a_0.webp'
        prefixed_candidates = [f for f in files if any(f.startswith(f"{ft}_") for ft in font_types_to_search)]
        selected = []

        def _matches_case_and_letter_after(prefix, fn, expected_letter):
            # prefix is like 'italic_upper-' or 'roman_lower-'
            try:
                ch = fn[len(prefix)]
            except Exception:
                return False
            if prefix.endswith('upper-'):
                return ch == expected_letter and expected_letter.isupper()
            else:
                return ch == expected_letter and expected_letter.islower()

        if prefixed_candidates:
            for fn in prefixed_candidates:
                for ft in font_types_to_search:
                    up_pref = f"{ft}_upper-"
                    low_pref = f"{ft}_lower-"
                    if fn.startswith(up_pref) and _matches_case_and_letter_after(up_pref, fn, letter):
                        selected.append(fn)
                        break
                    if fn.startswith(low_pref) and _matches_case_and_letter_after(low_pref, fn, letter):
                        selected.append(fn)
                        break
        else:
            # No font-prefixed files found: accept legacy naming conventions or unprefixed files
            for fn in files:
                if fn.startswith('upper-'):
                    try:
                        ch = fn.split('-', 1)[1][0]
                    except Exception:
                        continue
                    if ch == letter and letter.isupper():
                        selected.append(fn)
                elif fn.startswith('lower-'):
                    try:
                        ch = fn.split('-', 1)[1][0]
                    except Exception:
                        continue
                    if ch == letter and letter.islower():
                        selected.append(fn)
                else:
                    # Unprefixed names, e.g., 'A_0.webp' or 'a_1.webp'
                    if fn and fn[0] == letter:
                        selected.append(fn)

        # Sort by trailing numeric suffix if present (stable fallback to filename order)
        def _extract_num_from_name(fn):
            stem = fn.rsplit('.', 1)[0]
            try:
                return int(stem.split('_')[-1])
            except Exception:
                return 0

        selected = sorted(selected, key=_extract_num_from_name)
        results = []
        for fname in selected:
            p = os.path.join(self._cache_dir, book_name, fname)
            # Use preloaded base64 if available, otherwise fall back to URL
            if hasattr(self, '_image_cache') and (book_name, fname) in self._image_cache:
                data_url = self._image_cache[(book_name, fname)]
            else:
                data_url = f"/images_cache/{book_name}/{fname}"
            results.append((p, data_url))
        return results
            
    
    
    # ------------------------------------------------------------------
    # Layout sub-methods — each returns a Dash component tree
    # ------------------------------------------------------------------

    def _build_page_header(self):
        """Page title and subtitle."""
        return html.Div([
            html.H1("Theatre Chapbooks At Scale",
                     style={'textAlign': 'center', 'margin': '0', 'fontFamily': 'Inter, Arial, sans-serif',
                            'fontWeight': '700', 'fontSize': '2.2rem', 'color': '#374151',
                            'letterSpacing': '-0.5px'}),
            html.P("A Statistical Comparative Analysis of Typography",
                    style={'textAlign': 'center', 'margin': '5px 0 0 0', 'fontFamily': 'Inter, Arial, sans-serif',
                           'fontWeight': '400', 'fontSize': '1rem', 'color': '#887C57',
                           'letterSpacing': '0.5px'}),
        ], style={'marginBottom': '20px', 'padding': '20px 0'})

    def _build_font_selector(self):
        """Global font-type selector bar (Combined / Roman / Italic)."""
        active = {'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                  'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                  'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                  'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                  'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'width': '100px'}
        inactive = {**active, 'backgroundColor': '#DBD1B5', 'color': '#5a5040'}
        inactive_last = {**inactive, 'marginRight': '0'}

        return html.Div([
            html.Div([
                html.Div(
                    html.Label("Font type",
                               style={'fontSize': '13px', 'fontWeight': '600', 'color': '#887C57',
                                      'fontFamily': 'Inter, Arial, sans-serif'}),
                    style={'position': 'absolute', 'left': '20px', 'top': '50%', 'transform': 'translateY(-50%)'}
                ),
                html.Div([
                    html.Button("Combined", id='font-combined-btn', n_clicks=0, style=active),
                    html.Button("Roman", id='font-roman-btn', n_clicks=0, style=inactive),
                    html.Button("Italic", id='font-italic-btn', n_clicks=0, style=inactive_last),
                ], style={'display': 'flex', 'justifyContent': 'center', 'width': '100%'}),
                dcc.Store(id='font-type-store', data='combined'),
            ], style={'position': 'relative', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
        ], style={'marginBottom': '15px', 'padding': '12px 20px', 'backgroundColor': '#DBD1B5',
                  'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.15)'})

    def _build_data_stores(self, initial_umap_list):
        """Hidden dcc.Store components that persist application state."""
        return [
            dcc.Store(id='umap-positions-store', data=initial_umap_list),
            dcc.Store(id='network-selected-books-store', data=[]),
            dcc.Store(id='network-selected-books-visibility-store', data=False),
            dcc.Store(id='network-legend-visibility-store', data={}),
            dcc.Store(id='heatmap-legend-visibility-store', data={}),
        ]

    def _build_heatmap_and_comparison(self, initial_heatmap_fig):
        """Similarity matrix (left) + letter comparison panel (right)."""
        section_header = html.Div(
            html.H3("Typographic Similarity Analysis",
                     style={"margin": "0", "fontFamily": "Inter, Arial, sans-serif",
                            "fontWeight": "600", "letterSpacing": "0.5px", "color": "#887C57"}),
            style={"textAlign": "center", "padding": "6px 0", "backgroundColor": "#F8F5EC",
                   "borderRadius": "6px", "marginBottom": "10px",
                   "boxShadow": "0 1px 2px rgba(0,0,0,0.15)"})

        heatmap_col = html.Div([
            html.Div("Similarity Matrix",
                      style={'textAlign': 'center', 'marginBottom': '8px', 'fontFamily': 'Inter, Arial, sans-serif',
                             'fontWeight': '600', 'fontSize': '13px', 'color': '#887C57'}),
            html.Div([
                html.Button("Hide Printers", id='hide-all-printers-btn', n_clicks=0,
                             style={'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                    'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                                    'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                                    'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                                    'width': '110px', 'minWidth': '110px'}),
                html.Button("Show Printers", id='show-all-printers-btn', n_clicks=0,
                             style={'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                    'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                    'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                                    'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                                    'width': '110px', 'minWidth': '110px'}),
            ], style={'marginBottom': '8px', 'textAlign': 'center'}),
            dcc.Graph(id='similarity-heatmap', figure=initial_heatmap_fig,
                      style={'width': '100%', 'aspectRatio': '1 / 1', 'borderRadius': '8px',
                             'boxShadow': '0 1px 3px rgba(0,0,0,0.2)s', "margin": "0 auto"},
                      config={'responsive': True})
        ], style={'flex': '1 1 45%', 'minWidth': '0', 'maxWidth': '48%', 'boxSizing': 'border-box',
                  'backgroundColor': '#F8F5EC', 'borderRadius': '8px', 'padding': '2px', 'overflow': 'hidden'})

        comparison_col = html.Div([
            html.Div("Letter Comparison",
                      style={'textAlign': 'center', 'marginBottom': '8px', 'fontFamily': 'Inter, Arial, sans-serif',
                             'fontWeight': '600', 'fontSize': '13px', 'color': '#887C57'}),
            # Printer filter
            html.Div([
                html.Label("Filter by printer: ",
                            style={'fontWeight': '500', 'marginRight': '10px', 'fontSize': '11px',
                                   'color': '#5a5040', 'fontFamily': 'Inter, Arial, sans-serif'}),
                dcc.Dropdown(id='printer-filter-dropdown', options=[], value=None, multi=False,
                             placeholder='All printers', clearable=True, style={'width': '100%', 'fontSize': '11px'}),
                html.Button("Select all from this printer", id='select-all-printer-books-btn', n_clicks=0,
                             style={'marginTop': '5px', 'padding': '6px 12px', 'fontSize': '11px', 'fontWeight': '500',
                                    'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                    'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'none',
                                    'boxShadow': '0 1px 3px rgba(0,0,0,0.2)'}),
            ], style={'marginBottom': '8px', 'padding': '8px', 'backgroundColor': '#DBD1B5', 'borderRadius': '6px'}),
            # Book selector
            html.Div([
                html.Label("Select books: ",
                            style={'fontWeight': '500', 'marginRight': '10px', 'fontSize': '11px',
                                   'color': '#5a5040', 'fontFamily': 'Inter, Arial, sans-serif'}),
                html.Button("Clear", id='clear-comparison-btn', n_clicks=0,
                             style={'float': 'right', 'padding': '4px 10px', 'fontSize': '10px', 'fontWeight': '500',
                                    'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                                    'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                                    'boxShadow': '0 1px 3px rgba(0,0,0,0.2)'}),
                dcc.Dropdown(id='additional-books-dropdown', options=[], value=[], multi=True,
                             placeholder='+ Click matrix or select books here...', style={'width': '100%', 'fontSize': '11px'}),
                dcc.Store(id='clicked-books-store', data=[]),
            ], style={'marginBottom': '8px', 'padding': '8px', 'backgroundColor': '#DBD1B5', 'borderRadius': '6px'}),
            # Letter filter
            html.Div([
                html.Label("Filter: ", style={'fontWeight': '500', 'marginRight': '5px', 'fontSize': '11px',
                                               'color': '#5a5040', 'fontFamily': 'Inter, Arial, sans-serif'}),
                html.Button("All", id='select-all-letters', n_clicks=0,
                             style={'marginRight': '3px', 'padding': '4px 8px', 'fontSize': '10px', 'fontWeight': '500',
                                    'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                    'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                html.Button("None", id='select-no-letters', n_clicks=0,
                             style={'marginRight': '8px', 'padding': '4px 8px', 'fontSize': '10px', 'fontWeight': '500',
                                    'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                                    'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                html.Button("a-z", id='select-lowercase', n_clicks=0,
                             style={'marginRight': '3px', 'padding': '4px 8px', 'fontSize': '10px', 'fontWeight': '500',
                                    'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                    'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                html.Button("A-Z", id='select-uppercase', n_clicks=0,
                             style={'marginRight': '8px', 'padding': '4px 8px', 'fontSize': '10px', 'fontWeight': '500',
                                    'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                    'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'}),
                dcc.Checklist(id='letter-filter', options=[], value=[], inline=True,
                              style={'display': 'inline-block', 'fontSize': '11px'},
                              inputStyle={'marginRight': '2px', 'marginLeft': '6px'})
            ], style={'marginBottom': '8px', 'padding': '8px', 'backgroundColor': '#DBD1B5', 'borderRadius': '6px'}),
            # Comparison output
            html.Div(id='letter-comparison-panel',
                      style={'border': '1px solid #d1c7ad', 'borderRadius': '6px', 'padding': '15px',
                             'backgroundColor': '#F8F5EC', 'minHeight': '500px', 'maxHeight': '800px', 'overflowY': 'auto'},
                      children=[html.P("Click on a cell in the similarity matrix or a node in the network graph",
                                        style={'textAlign': 'center', 'color': '#6b7280', 'marginTop': '200px',
                                               'fontSize': '13px', 'fontFamily': 'Inter, Arial, sans-serif'})])
        ], style={'flex': '1 1 45%', 'minWidth': '0', 'maxWidth': '48%', 'boxSizing': 'border-box',
                  'backgroundColor': '#F8F5EC', 'borderRadius': '8px', 'padding': '10px', 'overflow': 'hidden'})

        return html.Div([
            section_header,
            html.Div([heatmap_col, comparison_col],
                      style={'display': 'flex', 'gap': '2%', 'alignItems': 'flex-start', 'justifyContent': 'space-between'}),
        ], style={"width": "100%", "backgroundColor": "#DBD1B5", "borderRadius": "8px",
                  "boxShadow": "0 2px 4px rgba(0,0,0,0.15)", "padding": "10px", "marginBottom": "20px"})

    def _build_network_section(self, initial_network_fig):
        """Network graph with controls (visibility toggles, sliders, UMAP source)."""
        net_cfg = self.config.get("network", {})
        edge_opacity_val = net_cfg.get("edge_opacity", 1.0)
        node_size_val = net_cfg.get("node_size", 12)
        label_size_val = net_cfg.get("label_size", 8)

        section_header = html.Div(
            html.H3("Graph of Typographic Similarity",
                     style={"margin": "0", "fontFamily": "Inter, Arial, sans-serif",
                            "fontWeight": "600", "letterSpacing": "0.5px", "color": "#887C57"}),
            style={"textAlign": "center", "padding": "6px 0", "backgroundColor": "#F8F5EC",
                   "borderRadius": "6px", "marginBottom": "10px",
                   "boxShadow": "0 1px 2px rgba(0,0,0,0.15)"})

        # --- Reusable button style helpers ---
        _toggle_base = {'marginBottom': '8px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                        'fontFamily': 'Inter, Arial, sans-serif', 'border': 'none', 'borderRadius': '6px',
                        'cursor': 'pointer', 'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                        'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'transition': 'background-color 0.15s ease, transform 0.05s ease',
                        'lineHeight': '1', 'width': '120px', 'minWidth': '120px'}
        _active = {**_toggle_base, 'backgroundColor': '#2f4a84', 'color': 'white'}
        _inactive = {**_toggle_base, 'backgroundColor': '#DBD1B5', 'color': '#5a5040'}
        _active_mr = {**_active, 'marginRight': '5px'}
        _inactive_mr = {**_inactive, 'marginRight': '5px'}

        col1 = html.Div([
            html.Div([
                html.Button("Hide Labels", id='hide-all-labels-btn', n_clicks=0, style=_inactive_mr),
                html.Button("Show Labels", id='show-all-labels-btn', n_clicks=0, style=_active),
                html.Br(),
                html.Button("Hide Markers", id='hide-all-markers-btn', n_clicks=0, style=_inactive_mr),
                html.Button("Show Markers", id='show-all-markers-btn', n_clicks=0, style=_active),
                html.Br(),
                html.Button("Hide Printers", id='hide-all-network-printers-btn', n_clicks=0, style=_inactive_mr),
                html.Button("Show Printers", id='show-all-network-printers-btn', n_clicks=0, style=_active),
                html.Br(),
                html.Button("Hide Selected", id='hide-selected-books-btn', n_clicks=0, style=_active_mr),
                html.Button("Show Selected", id='show-selected-books-btn', n_clicks=0, style=_inactive),
            ], style={'display': 'inline-block', 'textAlign': 'center'})
        ], style={'width': '30%', 'flexShrink': '0', 'flexGrow': '0', 'display': 'flex',
                  'alignItems': 'center', 'justifyContent': 'center'})

        slider_label = {'fontSize': '11px', 'fontWeight': '500', 'color': 'dimgray',
                        'fontFamily': 'Inter, Arial, sans-serif', 'marginBottom': '2px'}
        col2 = html.Div([
            html.Div([
                html.Div([
                    html.Label("Edge Opacity:", style=slider_label),
                    dcc.Slider(id='edge-opacity-slider', min=0, max=2, step=0.1, value=edge_opacity_val,
                               marks={0: {'label': '0', 'style': {'fontSize': '10px'}},
                                      1: {'label': '1', 'style': {'fontSize': '10px'}},
                                      2: {'label': '2', 'style': {'fontSize': '10px'}}},
                               tooltip={"placement": "bottom", "always_visible": False}, className="compact-slider")
                ], style={'marginBottom': '-10px'}),
                html.Div([
                    html.Label("Node Size:", style=slider_label),
                    dcc.Slider(id='node-size-slider', min=6, max=24, step=1, value=node_size_val,
                               marks={6: {'label': '6', 'style': {'fontSize': '10px'}},
                                      15: {'label': '15', 'style': {'fontSize': '10px'}},
                                      24: {'label': '24', 'style': {'fontSize': '10px'}}},
                               tooltip={"placement": "bottom", "always_visible": False}, className="compact-slider"),
                ], style={'marginBottom': '-10px'}),
                html.Div([
                    html.Label("Label Size:", style=slider_label),
                    dcc.Slider(id='label-size-slider', min=6, max=24, step=1, value=label_size_val,
                               marks={6: {'label': '6', 'style': {'fontSize': '10px'}},
                                      15: {'label': '15', 'style': {'fontSize': '10px'}},
                                      24: {'label': '24', 'style': {'fontSize': '10px'}}},
                               tooltip={"placement": "bottom", "always_visible": False}, className="compact-slider"),
                ]),
            ], style={'width': '85%'})
        ], style={'width': '30%', 'flexShrink': '0', 'flexGrow': '0', 'display': 'flex',
                  'flexDirection': 'column', 'justifyContent': 'center'})

        umap_active = {'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                       'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                       'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                       'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                       'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'width': '100px'}
        umap_inactive = {**umap_active, 'backgroundColor': '#DBD1B5', 'color': '#5a5040'}
        umap_inactive_last = {**umap_inactive, 'marginRight': '0'}

        col3 = html.Div([
            html.Div([
                html.Div("Node positions source",
                          style={'fontWeight': '600', 'fontSize': '11px', 'color': '#887C57',
                                 'marginBottom': '8px', 'textAlign': 'center', 'fontFamily': 'Inter, Arial, sans-serif'}),
                html.Div([
                    html.Button("Combined", id='umap-pos-combined-btn', n_clicks=0, style=umap_active),
                    html.Button("Roman", id='umap-pos-roman-btn', n_clicks=0, style=umap_inactive),
                    html.Button("Italic", id='umap-pos-italic-btn', n_clicks=0, style=umap_inactive_last),
                ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '2px'})
            ], style={'backgroundColor': '#DBD1B5', 'borderRadius': '8px', 'padding': '8px 10px', 'width': '100%'})
        ], style={'width': '30%', 'flexShrink': '0', 'flexGrow': '0', 'display': 'flex',
                  'alignItems': 'center', 'justifyContent': 'center'})

        controls_row = html.Div([col1, col2, col3],
                                 style={'marginBottom': '5px', 'padding': '10px', 'backgroundColor': "#DBD1B5",
                                        'borderRadius': '8px', 'display': 'flex', 'flexWrap': 'nowrap',
                                        'alignItems': 'stretch', 'justifyContent': 'space-between'})

        return html.Div([
            section_header,
            dcc.Store(id='umap-pos-source-store', data='combined'),
            controls_row,
            dcc.Graph(id='network-graph', figure=initial_network_fig,
                      style={'height': '800px', 'marginTop': '8px', 'borderRadius': '8px', 'overflow': 'hidden'})
        ], style={"width": "100%", "backgroundColor": "#DBD1B5", "borderRadius": "8px",
                  "boxShadow": "0 2px 4px rgba(0,0,0,0.15)", "padding": "5px"})

    def _build_dendrogram_section(self, initial_dendro_fig, level_marks):
        """Typographic Dendrogram panel — font selector, cut-level slider, truncation, drill-down, groups."""
        dendro_cfg = self.config.get("dendrogram", {})
        default_level = dendro_cfg.get("default_cut_level", 6)
        default_p = dendro_cfg.get("default_truncation", 30)

        _font = {'fontSize': '12px', 'fontFamily': 'Inter, Arial, sans-serif'}
        _label = {'fontSize': '12px', 'fontWeight': '600', 'color': '#5a5040', 'fontFamily': 'Inter, Arial, sans-serif'}

        section_header = html.Div(
            html.H3("Typographic Dendrogram",
                     style={"margin": "0", "fontFamily": "Inter, Arial, sans-serif",
                            "fontWeight": "600", "letterSpacing": "0.5px", "color": "#887C57"}),
            style={"textAlign": "center", "padding": "6px 0", "backgroundColor": "#F8F5EC",
                   "borderRadius": "6px", "marginBottom": "10px",
                   "boxShadow": "0 1px 2px rgba(0,0,0,0.15)"})

        font_active = {'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                       'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                       'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                       'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                       'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'width': '100px'}
        font_inactive = {**font_active, 'backgroundColor': '#DBD1B5', 'color': '#5a5040', 'marginRight': '0'}

        _toggle_base = {'marginBottom': '8px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                        'fontFamily': 'Inter, Arial, sans-serif', 'border': 'none', 'borderRadius': '6px',
                        'cursor': 'pointer', 'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                        'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'transition': 'background-color 0.15s ease, transform 0.05s ease',
                        'lineHeight': '1', 'width': '120px', 'minWidth': '120px'}
        _active = {**_toggle_base, 'backgroundColor': '#2f4a84', 'color': 'white'}
        _inactive = {**_toggle_base, 'backgroundColor': '#DBD1B5', 'color': '#5a5040'}

        # Truncation controls: how many leaf-clusters to show
        n_books = len(self.books)
        p_marks = {}
        for v in [10, 20, 30, 50, 100, n_books]:
            if v <= n_books:
                p_marks[v] = {'label': str(v) if v < n_books else 'All', 'style': {'fontSize': '10px'}}

        # --- Collapsible user guide ---
        _help_item = {'marginBottom': '6px', 'lineHeight': '1.45'}
        _help_bold = {'fontWeight': '600', 'color': '#2f4a84'}
        help_guide = html.Details([
            html.Summary("📖 How to use this dendrogram (click to collapse)",
                         style={'cursor': 'pointer', 'fontWeight': '600', 'fontSize': '12.5px',
                                'color': '#2f4a84', 'fontFamily': 'Inter, Arial, sans-serif',
                                'padding': '6px 0', 'userSelect': 'none'}),
            html.Div([
                html.P("This dendrogram clusters books by typographic similarity. "
                       "Books sharing more letter-shapes are linked by shorter branches.",
                       style={'marginTop': '6px', 'marginBottom': '8px', 'fontSize': '11.5px',
                              'color': '#5a5040', 'fontFamily': 'Inter, Arial, sans-serif'}),
                html.Ul([
                    html.Li([html.Span("Roman / Italic", style=_help_bold),
                             " — Switch between similarity matrices computed from roman or italic letterforms."],
                            style=_help_item),
                    html.Li([html.Span("Cut Level", style=_help_bold),
                             " — Controls how groups are formed. A higher threshold means books must share "
                             "more letter-shapes to belong to the same group, producing fewer, tighter clusters. "
                             "A lower threshold yields larger, more inclusive groups."],
                            style=_help_item),
                    html.Li([html.Span("Detail Level", style=_help_bold),
                             " — Controls how many leaf nodes are visible. At low values, nearby leaves are "
                             "collapsed into diamond-shaped cluster nodes showing a book count. Slide right "
                             "toward \"All\" to see every individual book. Works in both the full tree and "
                             "drill-down subtrees."],
                            style=_help_item),
                    html.Li([html.Span("Click a book (●)", style=_help_bold),
                             " — Selects that book and adds it to the network graph selection. "
                             "The book will be highlighted across all views."],
                            style=_help_item),
                    html.Li([html.Span("Click a cluster (◆)", style=_help_bold),
                             " — Drills down into that cluster, showing only its sub-tree. "
                             "Use the \"← Back to full tree\" button to return to the complete view."],
                            style=_help_item),
                    html.Li([html.Span("Search book", style=_help_bold),
                             " — Type a book name in the search box to instantly drill down into "
                             "the group that contains it at the current cut level. "
                             "Clear the search to return to the full tree."],
                            style=_help_item),
                    html.Li([html.Span("Hide Singletons", style=_help_bold),
                             " — Removes books that form their own solitary group (size\u00a0=\u00a01) from the "
                             "dendrogram entirely, including their branches, so you can focus on the books "
                             "that cluster together."],
                            style=_help_item),
                    html.Li([html.Span("Group cards (below the chart)", style=_help_bold),
                             " — Each card shows a group's size, its most common printers, and sample book names. "
                             "Use ", html.Em("Select"), " to add all books in that group to your selection, ",
                             html.Em("Highlight"), " to visually emphasize them in the dendrogram "
                             "(click Highlight again to clear), or ",
                             html.Em("Drill Down"), " to zoom into only that group's sub-tree."],
                            style=_help_item),
                    html.Li([html.Span("Branch colors", style=_help_bold),
                             " — Branches are colored by group membership. A solid-colored branch means all "
                             "descending books belong to the same group; gray branches span multiple groups."],
                            style=_help_item),
                    html.Li([html.Span("Reading the tree", style=_help_bold),
                             " — The root (leftmost point) splits into branches that end at individual books "
                             "or clusters on the right. Shorter horizontal distance between two books means "
                             "higher typographic similarity."],
                            style=_help_item),
                ], style={'paddingLeft': '18px', 'margin': '0', 'fontSize': '11.5px',
                          'color': '#5a5040', 'fontFamily': 'Inter, Arial, sans-serif'}),
            ], style={'padding': '4px 10px 8px 10px'}),
        ], open=True,
           style={'margin': '0 6px 10px 6px', 'padding': '6px 10px',
                  'backgroundColor': '#F8F5EC', 'borderRadius': '6px',
                  'border': '1px solid rgba(47,74,132,0.25)',
                  'boxShadow': '0 1px 3px rgba(0,0,0,0.08)'})

        return html.Div([
            section_header,
            help_guide,
            # Font selector row
            html.Div([
                html.Button("Roman", id='dendro-roman-btn', n_clicks=0, style=font_active),
                html.Button("Italic", id='dendro-italic-btn', n_clicks=0, style=font_inactive),
                dcc.Store(id='dendro-font-store', data='roman'),
                dcc.Store(id='dendro-highlight-store', data=None),
                # Store: which subtree to drill into (None = full tree, int = linkage node id)
                dcc.Store(id='dendro-drilldown-store', data=None),
            ], style={'textAlign': 'center', 'marginBottom': '8px'}),
            # Controls row: two columns
            html.Div([
                # Left column — cut level
                html.Div([
                    html.Label('Cut Level (threshold to form groups):', style=_label),
                    dcc.Slider(id='dendro-level-slider', min=0, max=self.n1hat_levels_max, step=1,
                               value=default_level, marks=level_marks,
                               tooltip={"placement": "bottom", "always_visible": False}, className='compact-slider'),
                ], style={'flex': '1', 'padding': '0 8px'}),
                # Right column — truncation p
                html.Div([
                    html.Label('Detail level (visible clusters):', style=_label),
                    dcc.Slider(id='dendro-truncation-slider', min=5, max=n_books, step=1,
                               value=min(default_p, n_books), marks=p_marks,
                               tooltip={"placement": "bottom", "always_visible": False}, className='compact-slider'),
                ], style={'flex': '1', 'padding': '0 8px'}),
            ], style={'display': 'flex', 'padding': '6px 0'}),
            # Status row: level label + breadcrumb + show/hide buttons
            html.Div([
                html.Div(id='dendro-level-label', style={'textAlign': 'center', 'color': '#5a5040', **_font}),
                html.Div(id='dendro-breadcrumb', children=[
                    html.Button("← Back to full tree", id='dendro-back-btn', n_clicks=0,
                                style={'display': 'none'})
                ], style={'textAlign': 'center', 'marginTop': '4px'}),
                html.Div([
                    html.Div([
                        html.Label('Search book:', style={**_label, 'marginRight': '6px', 'display': 'inline-block', 'verticalAlign': 'middle'}),
                        dcc.Dropdown(
                            id='dendro-search-book-dropdown',
                            options=[{'label': str(b), 'value': str(b)} for b in sorted(self.books)],
                            placeholder='Type a book name...',
                            searchable=True,
                            clearable=True,
                            style={'width': '350px', 'display': 'inline-block', 'verticalAlign': 'middle',
                                   'fontSize': '11px', 'fontFamily': 'Inter, Arial, sans-serif'},
                        ),
                    ], style={'display': 'inline-flex', 'alignItems': 'center', 'marginRight': '16px'}),
                    html.Button('Hide Singletons', id='dendro-hide-singletons-btn', n_clicks=0,
                                 style=_inactive),
                ], style={'textAlign': 'center', 'marginTop': '8px', 'display': 'flex',
                          'justifyContent': 'center', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '6px'}),
                dcc.Store(id='dendro-hide-singletons', data=False),
            ], style={'padding': '4px 12px'}),
            # Graph — dynamic height set by callback
            dcc.Graph(id='dendrogram-graph', figure=initial_dendro_fig,
                      style={'marginTop': '8px', 'borderRadius': '8px', 'overflow': 'hidden'}),
            # Group summaries
            html.Div(id='dendro-groups-container',
                      style={'marginTop': '10px', 'maxHeight': '300px', 'overflowY': 'auto',
                             'padding': '8px', 'backgroundColor': '#F8F5EC', 'borderRadius': '6px'}),
        ], style={'marginTop': '20px', 'padding': '10px', 'backgroundColor': '#DBD1B5',
                  'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.15)'})

    def _build_export_section(self):
        """Export controls and download component."""
        return html.Div([
            html.Div(
                html.H3("Export",
                         style={"margin": "0", "fontFamily": "Inter, Arial, sans-serif",
                                "fontWeight": "600", "letterSpacing": "0.5px", "color": "#887C57"}),
                style={"textAlign": "center", "padding": "6px 0", "backgroundColor": "#F8F5EC",
                       "borderRadius": "6px", "marginBottom": "10px",
                       "boxShadow": "0 1px 2px rgba(0,0,0,0.15)"}),
            html.Div([
                html.Button("Download HTML", id="export-html-btn", n_clicks=0,
                             style={'padding': '10px 20px', 'backgroundColor': '#2f4a84', 'color': 'white',
                                    'border': 'none', 'borderRadius': '6px', 'fontSize': '13px',
                                    'cursor': 'pointer', 'fontWeight': '500', 'fontFamily': 'Inter, Arial, sans-serif',
                                    'boxShadow': '0 1px 3px rgba(0,0,0,0.2)'}),
            ], style={'textAlign': 'center'}),
            html.Div(id="export-status",
                      style={'textAlign': 'center', 'marginTop': '10px', 'fontFamily': 'Inter, Arial, sans-serif',
                             'color': '#5a5040'})
        ], style={'marginTop': '20px', 'padding': '10px', 'backgroundColor': '#DBD1B5',
                  'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.15)'})

    # ------------------------------------------------------------------
    # Layout compositor
    # ------------------------------------------------------------------

    def _setup_layout(self, w_rm=None, w_it=None):
        """Compose the dashboard layout from sub-method components."""
        # Load cached UMAP positions, computing via umap-learn if missing
        initial_umap = self._load_umap_positions(font_type='combined', w_rm=w_rm, w_it=w_it)
        initial_umap_list = initial_umap.tolist() if initial_umap is not None else None
        print("UMAP positions loaded (if cached), figures will be created on first render")

        # Empty placeholder figures (actual content created on first callback)
        initial_network_fig = go.Figure()
        initial_heatmap_fig = go.Figure()
        initial_dendro_fig = go.Figure()

        # Dendrogram slider marks
        try:
            step = max(1, self.n1hat_levels_max // 6)
            level_marks = {i: {'label': str(i)} for i in range(0, self.n1hat_levels_max + 1, step)}
        except Exception:
            level_marks = {0: {'label': '0'}, 6: {'label': '6'}}

        self.app.layout = html.Div([
            self._build_page_header(),
            self._build_font_selector(),
            *self._build_data_stores(initial_umap_list),
            self._build_heatmap_and_comparison(initial_heatmap_fig),
            self._build_network_section(initial_network_fig),
            self._build_dendrogram_section(initial_dendro_fig, level_marks),
            self._build_export_section(),
            dcc.Download(id="download-html"),
        ])
    
    def _register_font_callbacks(self):
        """Font type toggle (Combined/Roman/Italic)."""
        
        # Font type button callback - updates store and button styles
        @self.app.callback(
            [Output('font-type-store', 'data'),
             Output('font-combined-btn', 'style'),
             Output('font-roman-btn', 'style'),
             Output('font-italic-btn', 'style')],
            [Input('font-combined-btn', 'n_clicks'),
             Input('font-roman-btn', 'n_clicks'),
             Input('font-italic-btn', 'n_clicks')],
            [State('font-type-store', 'data')],
            prevent_initial_call=True
        )
        def update_font_type(combined_clicks, roman_clicks, italic_clicks, current_font_type):
            # Button styles
            active_style = {'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                           'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                           'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                           'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                           'width': '100px', 'minWidth': '100px'}
            inactive_style = {'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                             'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                             'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                             'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                             'width': '100px', 'minWidth': '100px'}
            inactive_style_last = {**inactive_style, 'marginRight': '0'}  # Last button no margin
            active_style_last = {**active_style, 'marginRight': '0'}
            
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'font-combined-btn':
                return 'combined', active_style, inactive_style, inactive_style_last
            elif trigger_id == 'font-roman-btn':
                return 'roman', inactive_style, active_style, inactive_style_last
            elif trigger_id == 'font-italic-btn':
                return 'italic', inactive_style, inactive_style, active_style_last
            
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    def _register_dendrogram_callbacks(self):
        """Dendrogram: font selector, figure, book/group selection, highlight, show/hide."""

        # Dendrogram font selector callback (Roman / Italic buttons)
        @self.app.callback(
            [Output('dendro-font-store', 'data'),
             Output('dendro-roman-btn', 'style'),
             Output('dendro-italic-btn', 'style')],
            [Input('dendro-roman-btn', 'n_clicks'), Input('dendro-italic-btn', 'n_clicks')],
            [State('dendro-font-store', 'data')],
            prevent_initial_call=True
        )
        def update_dendro_font(roman_clicks, italic_clicks, current_dendro_font):
            active = {'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                      'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                      'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                      'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                      'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'width': '100px'}
            inactive = {'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                        'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                        'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                        'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                        'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'width': '100px'}
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger_id == 'dendro-roman-btn':
                return 'roman', active, inactive
            elif trigger_id == 'dendro-italic-btn':
                return 'italic', inactive, active
            return dash.no_update, dash.no_update, dash.no_update

        # Update dendrogram figure and level label based on selected level and font
        @self.app.callback(
            [Output('dendrogram-graph', 'figure'), Output('dendro-level-label', 'children'), Output('dendro-groups-container', 'children'),
             Output('dendro-level-slider', 'max'), Output('dendro-level-slider', 'marks'),
             Output('dendro-breadcrumb', 'children')],
            [Input('dendro-level-slider', 'value'), Input('dendro-font-store', 'data'), Input('dendro-highlight-store', 'data'),
             Input('dendro-truncation-slider', 'value'), Input('dendro-drilldown-store', 'data'),
             Input('dendro-hide-singletons', 'data')]
        )
        def update_dendrogram_figure(level, font, highlight_label, truncation_p, drilldown_node, hide_singletons):
            try:
                import plotly.express as px
                import plotly.graph_objects as go

                # Choose the n1hat matrix
                if font == 'italic':
                    n1hat = self.n1hat_it
                elif font == 'roman':
                    n1hat = self.n1hat_rm
                else:
                    n1hat = self.n1hat_rm + self.n1hat_it

                # Dynamic slider max & marks
                slider_max = int(np.max(n1hat))
                slider_step = max(1, slider_max // 6)
                slider_marks = {i: {'label': str(i)} for i in range(0, slider_max + 1, slider_step)}

                # Connected components at threshold
                adj = (n1hat >= level).astype(int)
                np.fill_diagonal(adj, 0)
                G_sparse = csr_matrix(adj)
                n_comp, labels = connected_components(G_sparse, directed=False)

                # Color palette & mapping
                palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Safe + px.colors.qualitative.Vivid
                unique_labels = np.unique(labels)
                color_map = {int(l): palette[i % len(palette)] for i, l in enumerate(unique_labels)}

                # Group size info for rank labels
                _grp_sizes = {int(l): int(np.sum(labels == l)) for l in unique_labels}
                _sorted_labels = sorted(_grp_sizes.keys(), key=lambda x: -_grp_sizes[x])
                _label_to_rank = {l: (i + 1) for i, l in enumerate(_sorted_labels)}

                # Validate highlight_label
                try:
                    if highlight_label is not None:
                        h_try = int(highlight_label)
                        if h_try not in set(unique_labels.astype(int)):
                            highlight_label = None
                except Exception:
                    highlight_label = None

                # Prepare linkage
                lk = None
                if font == 'italic':
                    lk = self.n1hat_linkage_it
                elif font == 'roman':
                    lk = self.n1hat_linkage_rm
                else:
                    lk = self.n1hat_linkage_combined
                if lk is None:
                    maxv = float(np.max(n1hat))
                    dist = maxv - n1hat.astype(float)
                    np.fill_diagonal(dist, 0.0)
                    lk = scipy_linkage(squareform(dist), method='single')

                # --- Truncation & drill-down ---
                n_total = len(self.books)
                truncation_p = truncation_p or n_total
                use_truncation = truncation_p < n_total

                # If drilling into a specific subtree, build a sub-linkage
                breadcrumb = []
                active_lk = lk
                active_n = n_total
                # Mapping from sub-leaf index back to original book index
                sub_leaf_map = None  # None means identity (full tree)

                if drilldown_node is not None:
                    try:
                        # drilldown_node can be:
                        # - a list of original book indices (from group drill-down)
                        # - an int or string int (linkage node ID from cluster click)
                        if isinstance(drilldown_node, list):
                            sub_leaves = sorted(int(x) for x in drilldown_node)
                        else:
                            node_id = int(drilldown_node)
                            sub_leaves = self._get_linkage_subtree_leaves(lk, node_id, n_total)
                        if sub_leaves and len(sub_leaves) > 2:
                            sub_leaf_map = sorted(sub_leaves)
                            sub_n = len(sub_leaf_map)
                            # Build sub-distance matrix
                            maxv = float(np.max(n1hat))
                            full_dist = maxv - n1hat.astype(float)
                            np.fill_diagonal(full_dist, 0.0)
                            sub_dist = full_dist[np.ix_(sub_leaf_map, sub_leaf_map)]
                            active_lk = scipy_linkage(squareform(sub_dist), method='single')
                            active_n = sub_n
                            # Recompute labels for sub-tree
                            sub_n1hat = n1hat[np.ix_(sub_leaf_map, sub_leaf_map)]
                            sub_adj = (sub_n1hat >= level).astype(int)
                            np.fill_diagonal(sub_adj, 0)
                            _, sub_labels = connected_components(csr_matrix(sub_adj), directed=False)
                            # Override labels/unique_labels/color_map for sub-tree
                            labels = np.zeros(n_total, dtype=int)
                            for si, orig_idx in enumerate(sub_leaf_map):
                                labels[orig_idx] = sub_labels[si]
                            unique_labels = np.unique(sub_labels)
                            color_map = {int(l): palette[i % len(palette)] for i, l in enumerate(unique_labels)}
                            _grp_sizes = {int(l): int(np.sum(sub_labels == l)) for l in unique_labels}
                            _sorted_labels = sorted(_grp_sizes.keys(), key=lambda x: -_grp_sizes[x])
                            _label_to_rank = {l: (i + 1) for i, l in enumerate(_sorted_labels)}

                            breadcrumb.append(
                                html.Span([
                                    html.Button("← Back to full tree", id='dendro-back-btn', n_clicks=0,
                                                style={'padding': '4px 12px', 'fontSize': '11px', 'fontWeight': '500',
                                                       'fontFamily': 'Inter, Arial, sans-serif',
                                                       'backgroundColor': '#2f4a84', 'color': 'white',
                                                       'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                                                       'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'marginRight': '8px'}),
                                    html.Span(f"Viewing subtree: {sub_n} books",
                                              style={'fontSize': '12px', 'color': '#5a5040',
                                                     'fontFamily': 'Inter, Arial, sans-serif'}),
                                ])
                            )
                        else:
                            drilldown_node = None
                    except Exception as e:
                        import traceback
                        print(f"Drill-down error: {e}")
                        print(traceback.format_exc())
                        drilldown_node = None

                # Compute maxv_linkage/maxv_offdiag for the active matrix
                # (full matrix when no drill-down, sub-matrix when drilled down)
                if sub_leaf_map is not None:
                    active_n1hat = n1hat[np.ix_(sub_leaf_map, sub_leaf_map)]
                else:
                    active_n1hat = n1hat
                maxv_linkage = float(np.max(active_n1hat))
                active_offdiag = active_n1hat.copy().astype(float)
                np.fill_diagonal(active_offdiag, 0.0)
                maxv_offdiag = float(np.max(active_offdiag))

                # --- If hiding singletons, rebuild linkage from non-singleton leaves only ---
                # This ensures branch lines for singletons are removed entirely.
                def _orig_idx_base(slm, i):
                    return slm[i] if slm is not None else i

                singleton_leaf_map = None  # maps new sub-index → active-tree leaf index
                if hide_singletons:
                    # Identify which active-tree leaves are NOT singletons
                    if sub_leaf_map is not None:
                        active_indices = list(range(active_n))
                    else:
                        active_indices = list(range(n_total))
                    non_sing = [i for i in active_indices
                                if _grp_sizes.get(int(labels[_orig_idx_base(sub_leaf_map, i)]), 1) > 1]
                    if len(non_sing) >= 2 and len(non_sing) < active_n:
                        singleton_leaf_map = non_sing
                        sing_n = len(singleton_leaf_map)
                        # Build sub-distance matrix for non-singletons
                        if sub_leaf_map is not None:
                            orig_indices = [sub_leaf_map[i] for i in singleton_leaf_map]
                        else:
                            orig_indices = singleton_leaf_map
                        maxv2 = float(np.max(n1hat))
                        full_dist2 = maxv2 - n1hat.astype(float)
                        np.fill_diagonal(full_dist2, 0.0)
                        sing_dist = full_dist2[np.ix_(orig_indices, orig_indices)]
                        active_lk = scipy_linkage(squareform(sing_dist), method='single')
                        active_n = sing_n
                        # Update sub_leaf_map to chain: sing index → orig index
                        sub_leaf_map = orig_indices

                # Build dendrogram from active linkage, with truncation if requested
                effective_truncation = use_truncation and truncation_p < active_n
                if effective_truncation:
                    dend = scipy_dendrogram(active_lk, no_plot=True, truncate_mode='lastp', p=truncation_p)
                else:
                    dend = scipy_dendrogram(active_lk, no_plot=True)

                icoord = np.array(dend['icoord'])
                dcoord = np.array(dend['dcoord'])
                leaves = dend['leaves']

                # Helper: map a leaf id from the dendrogram back to original book index
                def _orig_idx(leaf_id):
                    if sub_leaf_map is not None and leaf_id < active_n:
                        return sub_leaf_map[leaf_id]
                    return leaf_id

                fig = go.Figure()

                # --- HORIZONTAL dendrogram (left-to-right): swap x↔y ---
                # scipy dendrogram: icoord = leaf positions, dcoord = distances
                # Horizontal: x = distance (reversed so 0 on right), y = leaf position

                # Color branches by group membership
                def _branch_color(xs_orig, ys_orig):
                    """Determine color for a U-shaped branch."""
                    leaf_positions = set()
                    for x_val in [xs_orig[0], xs_orig[3]]:
                        li = round((x_val - 5) / 10)
                        if 0 <= li < len(leaves):
                            leaf_positions.add(li)
                    if not leaf_positions:
                        return 'rgba(60,60,60,0.5)'
                    group_set = set()
                    for li in leaf_positions:
                        leaf_id = leaves[li]
                        if leaf_id < active_n:
                            oi = _orig_idx(leaf_id)
                            group_set.add(int(labels[oi]))
                        else:
                            orig_leaves_sub = self._get_linkage_subtree_leaves(active_lk, leaf_id, active_n)
                            if orig_leaves_sub:
                                for ol in orig_leaves_sub:
                                    oi = _orig_idx(ol)
                                    group_set.add(int(labels[oi]))
                    if len(group_set) == 1:
                        gid = next(iter(group_set))
                        c = color_map.get(gid, 'rgba(60,60,60,0.5)')
                        if c.startswith('#') and len(c) == 7:
                            r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
                            return f'rgba({r},{g},{b},0.7)'
                        return c
                    return 'rgba(120,120,120,0.4)'

                # Draw branches: swap coords for horizontal layout
                # Transform x from distance to shared types: shared = maxv_linkage - dist
                # Store branch data for potential fading (applied after highlight logic)
                # All leaves placed at maxv_offdiag + 1 (right edge, beyond any branch)
                leaf_x_val = maxv_offdiag + 1

                branch_traces = []
                branch_leaf_sets = []  # which leaf indices each branch connects to
                for xs_orig, ys_orig in zip(icoord, dcoord):
                    bc = _branch_color(xs_orig, ys_orig)
                    # icoord values → y (leaf space), dcoord values → x (distance → shared types)
                    ys_new = list(xs_orig)
                    xs_new = [min(maxv_linkage - d, leaf_x_val) for d in ys_orig]
                    # Track which leaf indices this branch touches
                    branch_leaves = set()
                    for x_val in [xs_orig[0], xs_orig[3]]:
                        li = round((x_val - 5) / 10)
                        if 0 <= li < len(leaves):
                            leaf_id = leaves[li]
                            if leaf_id < active_n:
                                branch_leaves.add(_orig_idx(leaf_id))
                            else:
                                sub = self._get_linkage_subtree_leaves(active_lk, leaf_id, active_n)
                                if sub:
                                    for ol in sub:
                                        branch_leaves.add(_orig_idx(ol))
                    branch_traces.append((xs_new, ys_new, bc))
                    branch_leaf_sets.append(branch_leaves)

                # --- Leaf markers with rich tooltips ---
                n_leaves = len(leaves)
                leaf_y_pos = np.array([(i * 10) + 5 for i in range(n_leaves)])
                leaf_x_pos = np.full(n_leaves, leaf_x_val)

                leaf_colors = []
                leaf_sizes = []
                leaf_texts = []
                hover_texts = []
                leaf_is_cluster = []

                book_to_idx = {str(b): int(i) for i, b in enumerate(self.books)}

                for i, leaf_id in enumerate(leaves):
                    if leaf_id < active_n:
                        oi = _orig_idx(leaf_id)
                        book_name = str(self.books[oi])
                        grp = int(labels[oi])
                        rank = _label_to_rank.get(grp, '?')
                        printer = self.impr_names[oi] if oi < len(self.impr_names) else 'Unknown'
                        if printer in ['n. nan', 'm. missing']:
                            printer = 'Unknown'
                        hover = (f"<b>{book_name}</b><br>"
                                 f"Printer: {printer}<br>"
                                 f"Group #{rank} ({_grp_sizes.get(grp, '?')} books)<br>"
                                 f"<i>Click to select</i>")
                        hover_texts.append(hover)
                        leaf_texts.append(book_name)
                        leaf_colors.append(color_map.get(grp, '#999'))
                        leaf_sizes.append(8)
                        leaf_is_cluster.append(False)
                    else:
                        orig_leaves_sub = self._get_linkage_subtree_leaves(active_lk, leaf_id, active_n)
                        count = len(orig_leaves_sub) if orig_leaves_sub else 0
                        grp_counts = {}
                        for ol in (orig_leaves_sub or []):
                            oi = _orig_idx(ol)
                            g = int(labels[oi])
                            grp_counts[g] = grp_counts.get(g, 0) + 1
                        if grp_counts:
                            dominant_grp = max(grp_counts, key=grp_counts.get)
                            n_groups = len(grp_counts)
                            dominant_pct = grp_counts[dominant_grp] / count * 100 if count > 0 else 0
                            rank = _label_to_rank.get(dominant_grp, '?')
                        else:
                            dominant_grp = -1
                            n_groups = 0
                            dominant_pct = 0
                            rank = '?'
                        sample = [str(self.books[_orig_idx(ol)]) for ol in (orig_leaves_sub or [])[:3]]
                        sample_str = ', '.join(sample)
                        if count > 3:
                            sample_str += f', ... (+{count - 3})'
                        hover = (f"<b>Cluster ({count} books)</b><br>"
                                 f"Dominant: Group #{rank} ({dominant_pct:.0f}%)"
                                 + (f" + {n_groups - 1} other group{'s' if n_groups > 2 else ''}" if n_groups > 1 else '')
                                 + f"<br>{sample_str}<br>"
                                 f"<i>Click to drill down</i>")
                        hover_texts.append(hover)
                        leaf_texts.append(f"({count} books)")
                        if n_groups == 1:
                            leaf_colors.append(color_map.get(dominant_grp, '#999'))
                        else:
                            leaf_colors.append('#888')
                        leaf_sizes.append(min(18, max(10, 8 + count // 10)))
                        leaf_is_cluster.append(True)

                # --- Apply highlight rendering ---
                # Determine which original book indices should be emphasized
                if highlight_label is not None:
                    h = int(highlight_label)
                    for i, leaf_id in enumerate(leaves):
                        if leaf_id < active_n:
                            oi = _orig_idx(leaf_id)
                            if int(labels[oi]) != h:
                                leaf_colors[i] = 'rgba(180,180,180,0.25)'
                                leaf_sizes[i] = 6
                            else:
                                leaf_sizes[i] = 10
                        else:
                            orig_leaves_sub = self._get_linkage_subtree_leaves(active_lk, leaf_id, active_n)
                            has_match = any(int(labels[_orig_idx(ol)]) == h for ol in (orig_leaves_sub or []))
                            if not has_match:
                                leaf_colors[i] = 'rgba(180,180,180,0.25)'
                                leaf_sizes[i] = 6

                # --- Add branch traces ---
                for (xs_new, ys_new, bc), bl_set in zip(branch_traces, branch_leaf_sets):
                    fig.add_trace(go.Scatter(
                        x=xs_new, y=ys_new, mode='lines',
                        line=dict(color=bc, width=1.5),
                        hoverinfo='none', showlegend=False))

                # Add leaf markers — individual traces for click customdata
                for i, leaf_id in enumerate(leaves):
                    is_clust = leaf_is_cluster[i]
                    if is_clust:
                        # Encode the original book indices for drill-down
                        orig_leaves_sub = self._get_linkage_subtree_leaves(active_lk, leaf_id, active_n)
                        orig_book_idxs = [_orig_idx(ol) for ol in (orig_leaves_sub or [])]
                        cdata = "cluster_" + ",".join(str(x) for x in orig_book_idxs)
                    else:
                        cdata = str(self.books[_orig_idx(leaf_id)])
                    fig.add_trace(go.Scatter(
                        x=[leaf_x_pos[i]], y=[leaf_y_pos[i]],
                        mode='markers+text',
                        marker=dict(
                            color=leaf_colors[i],
                            size=leaf_sizes[i],
                            symbol='diamond' if is_clust else 'circle',
                            line=dict(color='white' if not is_clust else '#555', width=1)),
                        text=[leaf_texts[i]],
                        textposition='middle right',
                        textfont=dict(size=9, color='#3b3b3b', family='Inter, Arial, sans-serif'),
                        hovertemplate=hover_texts[i] + '<extra></extra>',
                        customdata=[cdata],
                        showlegend=False,
                        name=f'leaf_{leaf_id}'
                    ))

                # Dynamic height based on visible leaves
                n_visible = len(leaves)
                fig_height = max(400, n_visible * 18)

                title_text = f"Typographic Dendrogram ({font.capitalize()}) — groups at level ≥ {level}"
                if effective_truncation:
                    title_text += f"  ·  showing {n_visible} clusters"
                if hide_singletons:
                    title_text += "  ·  singletons hidden"

                fig.update_layout(
                    title=dict(text=title_text, font=dict(size=13, family='Inter, Arial, sans-serif', color='#5a5040')),
                    xaxis=dict(title='Shared letter types', showgrid=True, gridcolor='rgba(0,0,0,0.06)',
                               zeroline=True, zerolinecolor='rgba(0,0,0,0.1)', side='top',
                               range=[-0.5, leaf_x_val + 0.5],
                               dtick=max(1, int(maxv_offdiag / 8))),
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    height=fig_height,
                    margin=dict(t=60, b=20, l=40, r=200),
                    plot_bgcolor='rgba(248,245,236,0.5)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hoverlabel=dict(bgcolor='white', font_size=12, font_family='Inter, Arial, sans-serif'),
                )

                # Vertical reference line at cut level
                fig.add_vline(x=level, line_width=1.5, line_dash='dash',
                              line_color='rgba(180,60,60,0.6)',
                              annotation_text=f'cut = {level}',
                              annotation_position='top',
                              annotation_font=dict(size=10, color='rgba(180,60,60,0.8)',
                                                   family='Inter, Arial, sans-serif'))

                level_label = f"Groups: {len(unique_labels)} — largest group size: {max(_grp_sizes.values()) if _grp_sizes else 0}"

                # Build group summaries — use the active node set
                if sub_leaf_map is not None:
                    active_node_indices = sub_leaf_map
                else:
                    active_node_indices = list(range(n_total))
                total_nodes = len(active_node_indices)
                groups = []
                for lab in unique_labels:
                    nodes = [idx for idx in active_node_indices if int(labels[idx]) == int(lab)]
                    size = len(nodes)
                    printers_raw = self.impr_names[nodes]
                    known_mask = ~np.isin(printers_raw, ['n. nan', 'm. missing', 'Unknown'])
                    if np.any(known_mask):
                        printers, pcounts = np.unique(printers_raw[known_mask], return_counts=True)
                        order = np.argsort(-pcounts)
                        all_printers = [(str(printers[i]), int(pcounts[i]), float(pcounts[i]) / size) for i in order]
                        printers_unknown = False
                    else:
                        all_printers = []
                        printers_unknown = True
                    sample_books = [self.books[int(idx)] for idx in nodes[:3]]
                    groups.append({'label': int(lab), 'size': int(size), 'pct': float(size) / total_nodes,
                                   'printers': all_printers, 'printers_unknown': printers_unknown,
                                   'sample_books': sample_books, 'color': color_map.get(int(lab), '#999')})
                groups.sort(key=lambda x: -x['size'])

                # HTML group summaries
                children = []
                children.append(html.Div(f"Top groups (level ≥ {level}) — total nodes: {total_nodes}",
                                         style={'fontWeight': '600', 'color': '#5a5040', 'marginBottom': '6px',
                                                'fontSize': '12px', 'fontFamily': 'Inter, Arial, sans-serif'}))
                max_show = min(15, len(groups))
                btn_style_active = {'marginRight': '5px', 'padding': '4px 10px', 'fontSize': '11px', 'fontWeight': '500',
                                    'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                    'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                                    'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                                    'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1'}
                btn_style_secondary = {'marginRight': '5px', 'padding': '4px 10px', 'fontSize': '11px', 'fontWeight': '500',
                                       'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                                       'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                                       'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                                       'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1'}

                for i in range(max_show):
                    g = groups[i]
                    sample_str = ', '.join([str(b) for b in g['sample_books']])
                    if g.get('printers_unknown'):
                        printers_str = 'Unknown'
                        printers_count_display = 'Unknown'
                    else:
                        printers_str = ', '.join([f"{p} ({cnt}, {pct*100:.0f}%)" for p, cnt, pct in g['printers']])
                        printers_count_display = str(len(g['printers']))

                    select_btn = html.Button('Select', id={'type': 'dendro-select-group-btn', 'index': g['label']}, n_clicks=0,
                                             style=btn_style_active)
                    highlight_btn = html.Button('Highlight', id={'type': 'dendro-highlight-btn', 'index': g['label']}, n_clicks=0,
                                                style=btn_style_secondary)
                    drilldown_btn = html.Button('Drill Down', id={'type': 'dendro-drilldown-group-btn', 'index': g['label']}, n_clicks=0,
                                                style={**btn_style_secondary, 'backgroundColor': '#5a7a3a', 'color': 'white'}) if g['size'] > 2 else None
                    # Color swatch for group
                    swatch = html.Span('', style={'display': 'inline-block', 'width': '12px', 'height': '12px',
                                                   'backgroundColor': g['color'], 'borderRadius': '3px',
                                                   'marginRight': '6px', 'verticalAlign': 'middle',
                                                   'border': '1px solid rgba(0,0,0,0.15)'})

                    row_style = {'padding': '6px', 'borderBottom': '1px solid #e6e2d6'}
                    try:
                        if highlight_label is not None and int(g['label']) == int(highlight_label):
                            row_style.update({'backgroundColor': '#EAF6FF', 'borderLeft': '3px solid #2f4a84'})
                    except Exception:
                        pass

                    btn_row = [swatch,
                                  html.Span(f"Group #{i+1}: {g['size']} nodes ({g['pct']*100:.1f}%) — {printers_count_display} printers",
                                            style={'fontWeight': '600', 'color': '#3b3b3b', 'fontSize': '12px',
                                                   'fontFamily': 'Inter, Arial, sans-serif'}),
                                  html.Span(select_btn, style={'marginLeft': '4px'}),
                                  html.Span(highlight_btn, style={'marginLeft': '4px'})]
                    if drilldown_btn is not None:
                        btn_row.append(html.Span(drilldown_btn, style={'marginLeft': '4px'}))

                    children.append(html.Div([
                        html.Div(btn_row),
                        html.Div(f"Printers: {printers_str}", style={'fontSize': '11px', 'color': '#5a5040',
                                                                       'fontFamily': 'Inter, Arial, sans-serif',
                                                                       'marginLeft': '18px'}),
                        html.Div(f"Sample: {sample_str}", style={'fontSize': '11px', 'color': '#5a5040',
                                                                    'marginBottom': '6px', 'fontFamily': 'Inter, Arial, sans-serif',
                                                                    'marginLeft': '18px'}),
                        html.Div(id={'type': 'dendro-group-content', 'index': g['label']}, children=[], style={'marginTop': '6px'})
                    ], style=row_style))

                if len(groups) > max_show:
                    children.append(html.Div(f"And {len(groups)-max_show} more groups...",
                                             style={'fontStyle': 'italic', 'color': '#6b6b6b'}))

                return fig, level_label, children, slider_max, slider_marks, breadcrumb
            except Exception as e:
                import traceback
                print('Error building dendrogram figure:', e)
                print(traceback.format_exc())
                return go.Figure(), 'Error building dendrogram', html.Div('Error'), dash.no_update, dash.no_update, []

        # When a dendrogram leaf is clicked, add that book to the global selected books store
        # (or drill down if it's a cluster node)
        @self.app.callback(
            [Output('additional-books-dropdown', 'value', allow_duplicate=True),
             Output('network-selected-books-visibility-store', 'data', allow_duplicate=True),
             Output('dendro-drilldown-store', 'data', allow_duplicate=True)],
            Input('dendrogram-graph', 'clickData'),
            [State('additional-books-dropdown', 'value'), State('network-selected-books-visibility-store', 'data')],
            prevent_initial_call=True
        )
        def select_book_from_dendro(clickData, dropdown_val, current_vis):
            if not clickData:
                return dash.no_update, dash.no_update, dash.no_update
            try:
                pt = clickData['points'][0]
                custom = pt.get('customdata', '')
                if isinstance(custom, list):
                    custom = custom[0] if custom else ''
                custom = str(custom)
                if custom.startswith('cluster_'):
                    # Drill-down into this cluster — customdata contains original book indices
                    indices_str = custom.replace('cluster_', '')
                    book_indices = [int(x) for x in indices_str.split(',') if x]
                    if len(book_indices) > 2:
                        return dash.no_update, dash.no_update, book_indices
                    return dash.no_update, dash.no_update, dash.no_update
                # Single book — add to dropdown
                book_name = custom
                if not book_name:
                    return dash.no_update, dash.no_update, dash.no_update
                dropdown_val = dropdown_val or []
                if book_name in dropdown_val:
                    return dropdown_val, True, dash.no_update
                return dropdown_val + [book_name], True, dash.no_update
            except Exception as e:
                print('Error selecting book from dendrogram:', e)
                return dash.no_update, dash.no_update, dash.no_update

        # Back button resets drill-down, clears search, and clears highlight
        @self.app.callback(
            [Output('dendro-drilldown-store', 'data', allow_duplicate=True),
             Output('dendro-search-book-dropdown', 'value', allow_duplicate=True),
             Output('dendro-highlight-store', 'data', allow_duplicate=True)],
            Input('dendro-back-btn', 'n_clicks'),
            prevent_initial_call=True
        )
        def reset_dendro_drilldown(n_clicks):
            if n_clicks:
                return None, None, None
            return dash.no_update, dash.no_update, dash.no_update

        # When a group 'Select' button clicked, add group's books to the global selected store
        @self.app.callback(
            [Output('additional-books-dropdown', 'value', allow_duplicate=True), Output('network-selected-books-visibility-store', 'data', allow_duplicate=True)],
            [Input({'type':'dendro-select-group-btn', 'index': ALL}, 'n_clicks')],
            [State('dendro-level-slider', 'value'), State('dendro-font-store','data'),
             State('additional-books-dropdown', 'value'), State('dendro-drilldown-store', 'data')],
            prevent_initial_call=True
        )
        def select_group_from_dendro(n_clicks_list, level, font, dropdown_val, drilldown_node):
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update
            # Guard: if ALL n_clicks are 0 or None, this is initial creation — skip
            if not any(n_clicks_list):
                return dash.no_update, dash.no_update
            trig = ctx.triggered[0]['prop_id'].split('.')[0]
            try:
                import json
                idd = json.loads(trig)
                label = int(idd['index'])
            except Exception as e:
                print('Error parsing triggered id for group select:', e)
                return dash.no_update, dash.no_update
            # compute components for level & font
            try:
                if font == 'italic':
                    n1hat = self.n1hat_it
                elif font == 'roman':
                    n1hat = self.n1hat_rm
                else:
                    n1hat = self.n1hat_rm + self.n1hat_it
                # If drilled down, compute labels on the sub-matrix
                if drilldown_node is not None and isinstance(drilldown_node, list):
                    sub_leaves = sorted(int(x) for x in drilldown_node)
                    sub_n1hat = n1hat[np.ix_(sub_leaves, sub_leaves)]
                    sub_adj = (sub_n1hat >= level).astype(int)
                    np.fill_diagonal(sub_adj, 0)
                    _, sub_labels = connected_components(csr_matrix(sub_adj), directed=False)
                    matched = np.where(sub_labels == label)[0]
                    nodes = [sub_leaves[int(i)] for i in matched]
                else:
                    adj = (n1hat >= level).astype(int)
                    np.fill_diagonal(adj, 0)
                    _, labels = connected_components(csr_matrix(adj), directed=False)
                    nodes = list(np.where(labels == label)[0])
                books_to_add = [self.books[int(i)] for i in nodes]
            except Exception as e:
                print('Error computing group nodes for select:', e)
                return dash.no_update, dash.no_update
            dropdown_val = dropdown_val or []
            # preserve order: add books in ascending index order, skip duplicates
            to_add = [b for b in books_to_add if b not in dropdown_val]
            if not to_add:
                # ensure visibility on even when nothing new added
                return dropdown_val, True
            return dropdown_val + to_add, True

        # Highlight group callback — toggles the highlighted group id in a store
        @self.app.callback(
            Output('dendro-highlight-store', 'data'),
            [Input({'type':'dendro-highlight-btn', 'index': ALL}, 'n_clicks')],
            [State('dendro-highlight-store', 'data')],
            prevent_initial_call=True
        )
        def toggle_dendro_highlight(n_clicks_list, current_highlight):
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update
            # Guard: if ALL n_clicks are 0 or None, this is initial creation — skip
            if not any(n_clicks_list):
                return dash.no_update
            trig = ctx.triggered[0]['prop_id'].split('.')[0]
            try:
                import json
                idd = json.loads(trig)
                label = int(idd['index'])
            except Exception as e:
                print('Error parsing highlight trigger:', e)
                return dash.no_update
            # toggle: if same as current, clear
            if current_highlight == label:
                return None
            return label

        # Drill down into a specific group from its card button
        @self.app.callback(
            Output('dendro-drilldown-store', 'data', allow_duplicate=True),
            [Input({'type': 'dendro-drilldown-group-btn', 'index': ALL}, 'n_clicks')],
            [State('dendro-level-slider', 'value'), State('dendro-font-store', 'data'),
             State('dendro-drilldown-store', 'data')],
            prevent_initial_call=True
        )
        def drilldown_into_group(n_clicks_list, level, font, drilldown_node):
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update
            if not any(n_clicks_list):
                return dash.no_update
            trig = ctx.triggered[0]['prop_id'].split('.')[0]
            try:
                import json
                idd = json.loads(trig)
                label = int(idd['index'])
            except Exception as e:
                print('Error parsing drill-down trigger:', e)
                return dash.no_update
            try:
                if font == 'italic':
                    n1hat = self.n1hat_it
                elif font == 'roman':
                    n1hat = self.n1hat_rm
                else:
                    n1hat = self.n1hat_rm + self.n1hat_it
                # If already drilled down, compute labels on the sub-matrix
                if drilldown_node is not None and isinstance(drilldown_node, list):
                    sub_leaves = sorted(int(x) for x in drilldown_node)
                    sub_n1hat = n1hat[np.ix_(sub_leaves, sub_leaves)]
                    sub_adj = (sub_n1hat >= level).astype(int)
                    np.fill_diagonal(sub_adj, 0)
                    _, sub_labels = connected_components(csr_matrix(sub_adj), directed=False)
                    matched = np.where(sub_labels == label)[0]
                    nodes = sorted(sub_leaves[int(i)] for i in matched)
                else:
                    adj = (n1hat >= level).astype(int)
                    np.fill_diagonal(adj, 0)
                    _, labels = connected_components(csr_matrix(adj), directed=False)
                    nodes = sorted(int(x) for x in np.where(labels == label)[0])
                if len(nodes) > 2:
                    return nodes
            except Exception as e:
                print('Error computing group nodes for drill-down:', e)
            return dash.no_update

        # Search book → drill down into its group (or highlight if singleton)
        @self.app.callback(
            [Output('dendro-drilldown-store', 'data', allow_duplicate=True),
             Output('dendro-highlight-store', 'data', allow_duplicate=True)],
            Input('dendro-search-book-dropdown', 'value'),
            [State('dendro-level-slider', 'value'),
             State('dendro-font-store', 'data')],
            prevent_initial_call=True
        )
        def search_book_drilldown(book_name, level, font):
            """When user selects a book from the search dropdown, drill down into
            the group that contains it at the current cut level.
            For small groups (≤2 books), highlight the group in the full tree instead."""
            if not book_name:
                # Cleared → back to full tree, clear highlight
                return None, None
            book_list = list(self.books)
            if book_name not in book_list:
                return dash.no_update, dash.no_update
            if font == 'italic':
                n1hat = self.n1hat_it
            elif font == 'roman':
                n1hat = self.n1hat_rm
            else:
                n1hat = self.n1hat_rm + self.n1hat_it
            adj = (n1hat >= level).astype(int)
            np.fill_diagonal(adj, 0)
            _, comp_labels = connected_components(csr_matrix(adj), directed=False)
            book_idx = book_list.index(book_name)
            group_label = int(comp_labels[book_idx])
            group_nodes = sorted(int(x) for x in np.where(comp_labels == group_label)[0])
            if len(group_nodes) <= 2:
                # Group too small for a subtree → highlight it in the full tree
                return None, group_label
            # Large enough → drill down, clear any previous highlight
            return group_nodes, None

        # Hide singletons toggle
        @self.app.callback(
            [Output('dendro-hide-singletons', 'data'),
             Output('dendro-hide-singletons-btn', 'style')],
            Input('dendro-hide-singletons-btn', 'n_clicks'),
            State('dendro-hide-singletons', 'data'),
            prevent_initial_call=True
        )
        def toggle_hide_singletons(n_clicks, current_val):
            _base = {'marginRight': '5px', 'marginBottom': '8px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                     'fontFamily': 'Inter, Arial, sans-serif', 'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                     'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                     'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'transition': 'background-color 0.15s ease, transform 0.05s ease',
                     'lineHeight': '1', 'width': '120px', 'minWidth': '120px'}
            _active = {**_base, 'backgroundColor': '#2f4a84', 'color': 'white'}
            _inactive = {**_base, 'backgroundColor': '#DBD1B5', 'color': '#5a5040'}
            new_val = not current_val
            return new_val, _active if new_val else _inactive

    def _register_core_visualization_callbacks(self):
        """Central visualization update + UMAP position source."""

        # Font type update callback - updates network graph and heatmap
        @self.app.callback(
            [Output('network-graph', 'figure'),
             Output('similarity-heatmap', 'figure'),
             Output('umap-positions-store', 'data')],
            [Input('font-type-store', 'data')],
            [State('umap-positions-store', 'data'),
             State('network-graph', 'figure'),
             State('similarity-heatmap', 'figure'),
             State('edge-opacity-slider', 'value'),
             State('additional-books-dropdown', 'value'),
             State('node-size-slider', 'value'),
             State('label-size-slider', 'value'),
             State('network-legend-visibility-store', 'data'),
             State('heatmap-legend-visibility-store', 'data'),
             State('network-selected-books-store', 'data'),
             State('network-selected-books-visibility-store', 'data')],
            prevent_initial_call=False
        )
        def update_visualizations(font_type, umap_pos_array, current_network_fig, current_heatmap_fig, edge_opacity, 
                                  selected_books, node_size, label_size, network_legend_vis, heatmap_legend_vis,
                                  stored_selected_books, stored_selected_visibility):
            try:
                print(f"DEBUG: update_visualizations called - font_type={font_type}, umap_pos_array is None? {umap_pos_array is None}, last_font={self._last_font_type}")
                ctx = dash.callback_context
                print(f"DEBUG: ctx.triggered={ctx.triggered}")
                # Handle None case for UMAP positions (load cached combined if available, but do not abort if missing)
                store_update = dash.no_update
                if umap_pos_array is None:
                    umap_pos_array = self._load_umap_positions(font_type='combined')
                    if umap_pos_array is not None:
                        store_update = umap_pos_array.tolist()  # Update store with list for JSON
                # Convert list to numpy array ONCE at start (from dcc.Store JSON) if available
                # Use asarray (not array) to avoid copying if already an array
                if umap_pos_array is not None:
                    umap_array = np.asarray(umap_pos_array, dtype=np.float32)
                else:
                    umap_array = None
                
                # Select appropriate n1hat matrix
                if font_type == 'roman':
                    n1hat_matrix = self.n1hat_rm
                elif font_type == 'italic':
                    n1hat_matrix = self.n1hat_it
                else:
                    n1hat_matrix = (self.n1hat_rm + self.n1hat_it) / 2

                # If font_type changed, ensure edges are precomputed and force a full redraw
                if font_type != self._last_font_type:
                    try:
                        self._ensure_precomputed_edges(font_type, top_k=self.top_k, n_bins=self.n_bins)
                    except Exception as e:
                        print(f"Warning: Could not ensure edges for font {font_type}: {e}")

                    # DON'T change UMAP positions here — those are controlled solely by the
                    # "Node positions source" selector in the network block.
                    # Just keep whatever UMAP positions are already in the store.
                    print(f"Font type changed to '{font_type}', keeping current UMAP positions")

                    # Full redraw ensures traces match the newly loaded binned edges
                    network_fig = self._create_network_graph(umap_array, edge_opacity or 1.0, marker_size=node_size, label_size=label_size, font_type=font_type)
                    # Apply persisted legend visibilities (if any)
                    if network_legend_vis:
                        nf = network_fig.to_plotly_json() if isinstance(network_fig, go.Figure) else network_fig
                        for tr in nf.get('data', []):
                            name = tr.get('name', '')
                            if name in network_legend_vis:
                                tr['visible'] = network_legend_vis[name]
                        network_fig = nf
                    # Restore selected-book overlays if the store indicates they should be shown
                    try:
                        if stored_selected_books and stored_selected_visibility and umap_array is not None:
                            nf2 = network_fig if isinstance(network_fig, dict) else network_fig.to_plotly_json()
                            umap_arr = np.asarray(umap_array, dtype=np.float32)
                            book_list = list(self.books)
                            for book in stored_selected_books:
                                if book in book_list:
                                    book_idx = book_list.index(book)
                                    book_color = self._get_book_color(book)
                                    printer_name = self.impr_names[book_idx] if book_idx < len(self.impr_names) else 'Unknown'
                                    if printer_name in ['n. nan', 'm. missing']:
                                        printer_name = 'Unknown'
                                    nf2['data'].append({
                                        'type': 'scatter',
                                        'x': [umap_arr[book_idx, 0]],
                                        'y': [umap_arr[book_idx, 1]],
                                        'mode': 'markers+text',
                                        'marker': {'symbol': 'star', 'size': node_size * 1.3 if node_size else 18, 'color': book_color, 'line': {'width': 2, 'color': 'white'}},
                                        'text': [book], 'textposition': 'top center', 'textfont': {'size': label_size if label_size else 8, 'color': book_color, 'family': 'Arial, bold'},
                                        'hovertemplate': f'{book}<br>Printer: {printer_name}<extra></extra>', 'name': f'selected_book_{book}', 'showlegend': False
                                    })
                            # Add selected edges
                            bins = self._binned_edges.get(font_type, {})
                            for book in stored_selected_books:
                                if book in book_list:
                                    book_idx = book_list.index(book)
                                    book_color = self._get_book_color(book)
                                    r = int(book_color[1:3], 16); g = int(book_color[3:5], 16); b = int(book_color[5:7], 16)
                                    for bin_idx in range(10):
                                        bin_data = bins.get(bin_idx, {'edges': [], 'avg_w': 0})
                                        edges_in_bin = bin_data['edges']; avg_w = bin_data['avg_w']
                                        if not edges_in_bin:
                                            continue
                                        edges_array = np.array(edges_in_bin)
                                        mask = (edges_array[:, 0] == book_idx) | (edges_array[:, 1] == book_idx)
                                        matching_edges = edges_array[mask]
                                        if len(matching_edges) > 0:
                                            edges_x = []; edges_y = []
                                            for i, j in matching_edges:
                                                edges_x.extend([umap_arr[i, 0], umap_arr[j, 0], None])
                                                edges_y.extend([umap_arr[i, 1], umap_arr[j, 1], None])
                                            opacity = min(1.0, avg_w)
                                            color = f'rgba({r},{g},{b},{opacity})'
                                            nf2['data'].append({'type': 'scatter', 'x': edges_x, 'y': edges_y, 'mode': 'lines', 'line': {'width': 2, 'color': color}, 'showlegend': False, 'customdata': [avg_w], 'name': f'selected_edge_{book}_bin{bin_idx}', 'hoverinfo': 'skip'})
                            network_fig = nf2
                    except Exception as e:
                        print(f"Warning: restoring selected books on redraw failed: {e}")
                    self._last_font_type = font_type
                else:
                    # Check if figures are empty (initial state) or need full redraw
                    if not current_network_fig.get('data'):
                        # Full redraw - pass array directly
                        network_fig = self._create_network_graph(umap_array, edge_opacity or 1.0, marker_size=node_size, label_size=label_size, font_type=font_type)
                        # Apply persisted legend visibilities (if any)
                        if network_legend_vis:
                            nf = network_fig.to_plotly_json() if isinstance(network_fig, go.Figure) else network_fig
                            for tr in nf.get('data', []):
                                name = tr.get('name', '')
                                if name in network_legend_vis:
                                    tr['visible'] = network_legend_vis[name]
                            network_fig = nf
                        # Restore selected-book overlays if the store indicates they should be shown
                        try:
                            if stored_selected_books and stored_selected_visibility and umap_array is not None:
                                nf2 = network_fig if isinstance(network_fig, dict) else network_fig.to_plotly_json()
                                umap_arr = np.asarray(umap_array, dtype=np.float32)
                                book_list = list(self.books)
                                for book in stored_selected_books:
                                    if book in book_list:
                                        book_idx = book_list.index(book)
                                        book_color = self._get_book_color(book)
                                        printer_name = self.impr_names[book_idx] if book_idx < len(self.impr_names) else 'Unknown'
                                        if printer_name in ['n. nan', 'm. missing']:
                                            printer_name = 'Unknown'
                                        nf2['data'].append({
                                            'type': 'scatter',
                                            'x': [umap_arr[book_idx, 0]],
                                            'y': [umap_arr[book_idx, 1]],
                                            'mode': 'markers+text',
                                            'marker': {'symbol': 'star', 'size': node_size * 1.3 if node_size else 18, 'color': book_color, 'line': {'width': 2, 'color': 'white'}},
                                            'text': [book], 'textposition': 'top center', 'textfont': {'size': label_size if label_size else 8, 'color': book_color, 'family': 'Arial, bold'},
                                            'hovertemplate': f'{book}<br>Printer: {printer_name}<extra></extra>', 'name': f'selected_book_{book}', 'showlegend': False
                                        })
                                # Add selected edges
                                bins = self._binned_edges.get(font_type, {})
                                for book in stored_selected_books:
                                    if book in book_list:
                                        book_idx = book_list.index(book)
                                        book_color = self._get_book_color(book)
                                        r = int(book_color[1:3], 16); g = int(book_color[3:5], 16); b = int(book_color[5:7], 16)
                                        for bin_idx in range(10):
                                            bin_data = bins.get(bin_idx, {'edges': [], 'avg_w': 0})
                                            edges_in_bin = bin_data['edges']; avg_w = bin_data['avg_w']
                                            if not edges_in_bin:
                                                continue
                                            edges_array = np.array(edges_in_bin)
                                            mask = (edges_array[:, 0] == book_idx) | (edges_array[:, 1] == book_idx)
                                            matching_edges = edges_array[mask]
                                            if len(matching_edges) > 0:
                                                edges_x = []; edges_y = []
                                                for i, j in matching_edges:
                                                    edges_x.extend([umap_arr[i, 0], umap_arr[j, 0], None])
                                                    edges_y.extend([umap_arr[i, 1], umap_arr[j, 1], None])
                                                opacity = min(1.0, avg_w)
                                                color = f'rgba({r},{g},{b},{opacity})'
                                                nf2['data'].append({'type': 'scatter', 'x': edges_x, 'y': edges_y, 'mode': 'lines', 'line': {'width': 2, 'color': color}, 'showlegend': False, 'customdata': [avg_w], 'name': f'selected_edge_{book}_bin{bin_idx}', 'hoverinfo': 'skip'})
                                network_fig = nf2
                        except Exception as e:
                            print(f"Warning: restoring selected books on redraw failed: {e}")
                    else:
                        # Minimal update: patch network edges - pass array directly
                        # This will also handle updating selected book edges with new font type
                        network_fig = self._update_network_edges(current_network_fig, edge_opacity or 1.0, umap_array, font_type, selected_books)

                if not current_heatmap_fig.get('data'):
                    # Full redraw
                    heatmap_fig = self._create_heatmap()
                    # Apply persisted heatmap legend visibilities (if any)
                    if heatmap_legend_vis:
                        hf = heatmap_fig.to_plotly_json() if isinstance(heatmap_fig, go.Figure) else heatmap_fig
                        for tr in hf.get('data', []):
                            name = tr.get('name', '')
                            if name in heatmap_legend_vis:
                                tr['visible'] = heatmap_legend_vis[name]
                        heatmap_fig = hf
                else:
                    # Minimal update: patch heatmap z/title and network edges
                    heatmap_fig = dash.Patch()
                    heatmap_fig['data'][0]['z'] = n1hat_matrix
                    
                    # Update customdata for marker traces
                    types_diag = np.diagonal(n1hat_matrix)
                    for i in range(1, len(current_heatmap_fig['data'])):
                        trace = current_heatmap_fig['data'][i]
                        impr = trace['name']
                        mask = self.impr_names == impr  # boolean mask
                        if np.any(mask):
                            heatmap_fig['data'][i]['customdata'] = types_diag[mask][:, None]
                            
                print(f"DEBUG: update_visualizations returning network_fig type={type(network_fig)}, heatmap_fig type={type(heatmap_fig)}, store_update={'set' if store_update is not dash.no_update else 'no_update'}")
                return network_fig, heatmap_fig, store_update
            except Exception as e:
                import traceback
                print("ERROR in update_visualizations:", e)
                print(traceback.format_exc())
                return dash.no_update, dash.no_update, dash.no_update
        @self.app.callback(
            [Output('umap-pos-source-store', 'data'), Output('umap-pos-combined-btn', 'style'), Output('umap-pos-roman-btn', 'style'), Output('umap-pos-italic-btn', 'style')],
            [Input('umap-pos-combined-btn', 'n_clicks'), Input('umap-pos-roman-btn', 'n_clicks'), Input('umap-pos-italic-btn', 'n_clicks')],
            [State('umap-pos-source-store', 'data')],
            prevent_initial_call=True
        )
        def update_umap_pos_source(combined_clicks, roman_clicks, italic_clicks, current_source):
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            active = {'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                        'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                        'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                                        'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                        'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'width': '100px'}
            inactive = {'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                        'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                        'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                        'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                        'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'width': '100px', 'minWidth': '100px'}
            if trigger_id == 'umap-pos-combined-btn':
                return 'combined', active, inactive, inactive
            elif trigger_id == 'umap-pos-roman-btn':
                return 'roman', inactive, active, inactive
            elif trigger_id == 'umap-pos-italic-btn':
                return 'italic', inactive, inactive, active
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        @self.app.callback(
            [Output('umap-positions-store', 'data', allow_duplicate=True),
            Output('network-graph', 'figure', allow_duplicate=True)],
            [Input('umap-pos-source-store', 'data')],
            [State('network-graph', 'figure'), State('edge-opacity-slider', 'value'),
            State('node-size-slider', 'value'), State('label-size-slider', 'value'),
            State('font-type-store', 'data'), State('network-legend-visibility-store', 'data')],
            prevent_initial_call=True
        )
        def handle_umap_pos_source_change(pos_source, current_network_fig, edge_opacity, node_size, label_size, current_edge_font, network_legend_vis):
            # Try to load cached UMAP positions for selected source; do not compute automatically
            umap_positions = self._load_umap_positions(font_type=pos_source)
            if umap_positions is None:
                print(f"No UMAP positions for source '{pos_source}' (umap-learn may not be installed).")
                return dash.no_update, dash.no_update
            umap_array = np.asarray(umap_positions, dtype=np.float32)
            # Recreate network using existing edge font selection (edges unaffected by position source)
            network_fig = self._create_network_graph(umap_array, edge_opacity or 1.0, marker_size=node_size, label_size=label_size, font_type=current_edge_font)
            # Apply persisted visibilities if present
            if network_legend_vis:
                nf = network_fig.to_plotly_json() if isinstance(network_fig, go.Figure) else network_fig
                for tr in nf.get('data', []):
                    name = tr.get('name', '')
                    if name in network_legend_vis:
                        tr['visible'] = network_legend_vis[name]
                network_fig = nf
            print(f"DEBUG: handle_umap_pos_source_change returning network_fig type={type(network_fig)}")
            return umap_array.tolist(), network_fig     

    def _register_filter_callbacks(self):
        """Filter setup, printer/book selectors, matrix click, overlays, letter quick-select."""
           
        # Initialize letter filter and printer dropdown on load
        @self.app.callback(
            [Output('letter-filter', 'options'),
             Output('letter-filter', 'value'),
             Output('printer-filter-dropdown', 'options')],
            [Input('font-type-store', 'data')],  # Trigger on font-type change and on load
            [State('letter-filter', 'value')],
            prevent_initial_call=False
        )
        def init_filters(_, current_selected):
            # Build options for all letters
            letter_options = [{'label': f' {l}', 'value': l} for l in self._all_letters]
            # Preserve previously selected letters when possible (persist through font changes)
            if current_selected and isinstance(current_selected, list):
                # Keep only letters that still exist in master list (defensive)
                preserved = [l for l in current_selected if l in self._all_letters]
                selected = preserved if preserved else self._all_letters
            else:
                selected = self._all_letters

            # Get unique printers
            unique_printers = sorted(set(self.impr_names))
            printer_options = [{'label': p, 'value': p} for p in unique_printers if p not in ['n. nan', 'm. missing']]
            return letter_options, selected, printer_options
        
        # Filter books by printer
        @self.app.callback(
            Output('additional-books-dropdown', 'options'),
            [Input('printer-filter-dropdown', 'value')],
            prevent_initial_call=False
        )
        def filter_books_by_printer(selected_printer):
            if selected_printer is None:
                # Show all books
                book_options = [{'label': f"{b} ({self.impr_names[i] if self.impr_names[i] not in ['n. nan', 'm. missing'] else 'Unknown'})", 'value': b} 
                               for i, b in enumerate(self.books)]
            else:
                # Filter books by printer
                book_options = [{'label': f"{b} ({self.impr_names[i] if self.impr_names[i] not in ['n. nan', 'm. missing'] else 'Unknown'})", 'value': b} 
                               for i, b in enumerate(self.books) 
                               if self.impr_names[i] == selected_printer]
            return sorted(book_options, key=lambda x: x['label'])
        
        # Show/hide "Select all from printer" button
        @self.app.callback(
            Output('select-all-printer-books-btn', 'style'),
            [Input('printer-filter-dropdown', 'value')],
            prevent_initial_call=False
        )
        def toggle_select_all_printer_btn(selected_printer):
            base_style = {'marginTop': '5px', 'padding': '3px 8px', 'fontSize': '10px', 'backgroundColor': '#ffe8cc', 'border': '1px solid #cc8800', 'borderRadius': '3px', 'cursor': 'pointer'}
            if selected_printer is not None and selected_printer not in ['n. nan', 'm. missing', 'Unknown']:
                return {**base_style, 'display': 'block'}
            return {**base_style, 'display': 'none'}
        
        # Select all books from filtered printer
        @self.app.callback(
            Output('additional-books-dropdown', 'value', allow_duplicate=True),
            [Input('select-all-printer-books-btn', 'n_clicks')],
            [State('printer-filter-dropdown', 'value')],
            prevent_initial_call=True
        )
        def select_all_printer_books(n_clicks, selected_printer):
            if n_clicks and selected_printer:
                # Get all books from this printer
                books_from_printer = [b for i, b in enumerate(self.books) 
                                      if (self.impr_names[i] if self.impr_names[i] not in ['n. nan', 'm. missing'] else 'Unknown') == selected_printer]
                return books_from_printer
            return dash.no_update
        
        # Store clicked books from matrix and handle clear button
        @self.app.callback(
            [Output('clicked-books-store', 'data', allow_duplicate=True),
             Output('additional-books-dropdown', 'value', allow_duplicate=True)],
            [Input('similarity-heatmap', 'clickData'),
             Input('clear-comparison-btn', 'n_clicks')],
            [State('clicked-books-store', 'data'),
             State('additional-books-dropdown', 'value')],
            prevent_initial_call=True
        )
        def handle_matrix_click_and_clear(click_data, clear_clicks, stored_books, dropdown_books):
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Clear button pressed
            if trigger_id == 'clear-comparison-btn':
                return [], []
            
            # Matrix clicked
            if trigger_id == 'similarity-heatmap' and click_data is not None:
                x_label = click_data['points'][0]['x']
                y_label = click_data['points'][0]['y']
                
                book1 = x_label.split(' | ')[0] if ' | ' in x_label else x_label
                book2 = y_label.split(' | ')[0] if ' | ' in y_label else y_label
                
                # Check if diagonal click (same book on x and y)
                is_diagonal = (book1 == book2)
                
                if is_diagonal:
                    # Diagonal click: ADD this book to existing selection
                    current_books = list(dropdown_books) if dropdown_books else []
                    if book1 not in current_books:
                        current_books.append(book1)
                    return current_books, current_books
                else:
                    # Off-diagonal click: REPLACE selection with the two books
                    new_books = [book1, book2]
                    return new_books, new_books
            
            return dash.no_update, dash.no_update
        
        # Add row/column overlays for selected books
        @self.app.callback(
            Output('similarity-heatmap', 'figure', allow_duplicate=True),
            [Input('additional-books-dropdown', 'value')],
            [State('similarity-heatmap', 'figure')],
            prevent_initial_call=True
        )
        def update_matrix_overlays(selected_books, current_fig):
            if current_fig is None:
                return dash.no_update
            
            patched_fig = dash.Patch()
            
            # Remove existing overlay traces (traces with name starting with 'overlay_')
            data_to_keep = []
            for i, trace in enumerate(current_fig.get('data', [])):
                if not trace.get('name', '').startswith('overlay_'):
                    data_to_keep.append(trace)
            
            # Set the data to only non-overlay traces
            patched_fig['data'] = data_to_keep
            
            # Add new overlays for selected books
            if selected_books:
                n_books = len(self.books)
                book_list = list(self.books)
                
                for book in selected_books:
                    if book in book_list:
                        book_idx = book_list.index(book)
                        
                        # Get book-specific color with 0.35 alpha
                        book_color = self._get_book_color(book)
                        # Convert hex to rgba with 0.35 alpha
                        r = int(book_color[1:3], 16)
                        g = int(book_color[3:5], 16)
                        b = int(book_color[5:7], 16)
                        overlay_color = f'rgba({r}, {g}, {b}, 0.35)'
                        
                        # Horizontal line (row)
                        patched_fig['data'].append({
                            'type': 'scatter',
                            'x': [book_list[0], book_list[-1]],
                            'y': [book, book],
                            'mode': 'lines',
                            'line': {'color': overlay_color, 'width': 3},
                            'showlegend': False,
                            'hoverinfo': 'skip',
                            'name': f'overlay_row_{book}'
                        })
                        
                        # Vertical line (column)
                        patched_fig['data'].append({
                            'type': 'scatter',
                            'x': [book, book],
                            'y': [book_list[0], book_list[-1]],
                            'mode': 'lines',
                            'line': {'color': overlay_color, 'width': 3},
                            'showlegend': False,
                            'hoverinfo': 'skip',
                            'name': f'overlay_col_{book}'
                        })
            
            return patched_fig
            
        # Quick select buttons for letter filter
        @self.app.callback(
            Output('letter-filter', 'value', allow_duplicate=True),
            [Input('select-all-letters', 'n_clicks'),
             Input('select-no-letters', 'n_clicks'),
             Input('select-lowercase', 'n_clicks'),
             Input('select-uppercase', 'n_clicks')],
            prevent_initial_call=True
        )
        def quick_select_letters(all_clicks, none_clicks, lower_clicks, upper_clicks):
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'select-all-letters':
                return self._all_letters
            elif button_id == 'select-no-letters':
                return []
            elif button_id == 'select-lowercase':
                return [l for l in self._all_letters if l.islower()]
            elif button_id == 'select-uppercase':
                return [l for l in self._all_letters if l.isupper()]
            return dash.no_update
        
        # Button styles are defined in self._active_btn_style / self._inactive_btn_style

    def _register_network_callbacks(self):
        """Network graph controls: labels, markers, printers, node click, sliders, selected books, legend sync."""
        
        # Show/Hide all labels on network graph
        @self.app.callback(
            [Output('network-graph', 'figure', allow_duplicate=True),
             Output('show-all-labels-btn', 'style'),
             Output('hide-all-labels-btn', 'style')],
            [Input('show-all-labels-btn', 'n_clicks'),
             Input('hide-all-labels-btn', 'n_clicks')],
            [State('network-graph', 'figure')],
            prevent_initial_call=True
        )
        def toggle_network_labels(show_clicks, hide_clicks, current_fig):
            ctx = dash.callback_context
            # Defensive default in case ctx.triggered has unexpected shape
            show_labels = False
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            show_labels = (trigger_id == 'show-all-labels-btn')
            
            # Set button styles based on which is active (consistent margins)
            show_style = {**self._active_btn_style, 'marginBottom': '8px'} if show_labels else {**self._inactive_btn_style, 'marginRight': '5px', 'marginBottom': '8px'}
            hide_style = {**self._active_btn_style, 'marginRight': '5px', 'marginBottom': '8px'} if not show_labels else {**self._inactive_btn_style, 'marginBottom': '8px'}
            
            if not current_fig.get('data'):
                return dash.no_update, show_style, hide_style
            
            # Use Patch for efficient update
            patched_fig = dash.Patch()
            
            # Update text visibility for all node traces (skip bin_ traces which are edges)
            for i, trace in enumerate(current_fig['data']):
                name = trace.get('name', '')
                # Skip edge traces (named bin_0, bin_1, etc.)
                if name.startswith('bin_') or name.startswith('selected_'):
                    continue
                current_mode = trace.get('mode', 'markers')
                has_markers = 'markers' in current_mode
                if show_labels:
                    patched_fig['data'][i]['mode'] = 'markers+text' if has_markers else 'text'
                else:
                    patched_fig['data'][i]['mode'] = 'markers' if has_markers else 'none'
            
            return patched_fig, show_style, hide_style
        
        # Show/Hide all markers on network graph
        @self.app.callback(
            [Output('network-graph', 'figure', allow_duplicate=True),
             Output('show-all-markers-btn', 'style'),
             Output('hide-all-markers-btn', 'style')],
            [Input('show-all-markers-btn', 'n_clicks'),
             Input('hide-all-markers-btn', 'n_clicks')],
            [State('network-graph', 'figure')],
            prevent_initial_call=True
        )
        def toggle_network_markers(show_clicks, hide_clicks, current_fig):
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            show_markers = trigger_id == 'show-all-markers-btn'
            
            # Set button styles based on which is active
            show_style = {**self._active_btn_style} if show_markers else {**self._inactive_btn_style}
            hide_style = {**self._active_btn_style, 'marginRight': '5px', 'marginBottom': '8px'} if not show_markers else {**self._inactive_btn_style, 'marginRight': '5px', 'marginBottom': '8px'}
            show_style['marginBottom'] = '8px'
            
            if not current_fig.get('data'):
                return dash.no_update, show_style, hide_style
            
            # Use Patch for efficient update
            patched_fig = dash.Patch()
            
            # Update marker visibility for all node traces (skip bin_ traces which are edges)
            for i, trace in enumerate(current_fig['data']):
                # Skip edge traces (named bin_0, bin_1, etc.)
                name = trace.get('name', '')
                if name.startswith('bin_') or name.startswith('selected_'):
                    continue
                current_mode = trace.get('mode', 'markers')
                has_text = 'text' in current_mode
                if show_markers:
                    patched_fig['data'][i]['mode'] = 'markers+text' if has_text else 'markers'
                else:
                    patched_fig['data'][i]['mode'] = 'text' if has_text else 'none'
            
            return patched_fig, show_style, hide_style
        
        # Click on network graph node to add book to comparison
        @self.app.callback(
            [Output('clicked-books-store', 'data', allow_duplicate=True),
             Output('additional-books-dropdown', 'value', allow_duplicate=True)],
            [Input('network-graph', 'clickData')],
            [State('clicked-books-store', 'data'),
             State('additional-books-dropdown', 'value')],
            prevent_initial_call=True
        )
        def handle_network_click(click_data, stored_books, dropdown_books):
            """Toggle a book in the selected-books list when a node is clicked.
            If the clicked book is already selected, remove it; otherwise add it."""
            if click_data is None:
                return dash.no_update, dash.no_update

            # Get the clicked point's custom data (contains book name)
            point = click_data['points'][0]

            # customdata contains "book_name<br>Printer: ..." - extract book name
            if 'customdata' in point:
                custom = point['customdata']
                # Extract book name from the custom data string
                book_name = custom.split('<br>')[0] if '<br>' in custom else custom
            else:
                # Fallback: might be clicking on edge, ignore
                return dash.no_update, dash.no_update

            # Toggle book in existing selection
            current_books = list(dropdown_books) if dropdown_books else []
            if book_name in current_books:
                # If already selected, remove it
                current_books = [b for b in current_books if b != book_name]
            else:
                # Otherwise add it
                current_books.append(book_name)

            # Update both the clicked-books store and the dropdown value
            return current_books, current_books
                
        # Separate callback for edge opacity - uses Patch for instant updates
        @self.app.callback(
            Output('network-graph', 'figure', allow_duplicate=True),
            [Input('edge-opacity-slider', 'value')],
            [State('network-graph', 'figure')],
            prevent_initial_call=True
        )
        def update_edge_opacity_only(edge_opacity, current_fig):
            if not current_fig.get('data'):
                return dash.no_update
            if edge_opacity is None:
                return dash.no_update

            patched = dash.Patch()
            
            # Update bin traces by iterating through data and checking names
            for i, trace in enumerate(current_fig['data']):
                name = trace.get('name', '')
                if name.startswith('bin_') or name.startswith('selected_edge_'):
                    customdata = trace.get('customdata', [])
                    if customdata and len(customdata) > 0:
                        w = customdata[0]
                        if name.startswith('selected_edge_'):
                            new_a = min(1.0, w)
                        else:
                            new_a = min(1.0, w * float(edge_opacity))
                        color = f'rgba(100,100,100,{new_a})'
                        patched['data'][i]['line']['color'] = color

            return patched
        
        @self.app.callback(
            Output('network-graph', 'figure', allow_duplicate=True),
            [Input('node-size-slider', 'value'),
             Input('label-size-slider', 'value')],
            [State('network-graph', 'figure')],
            prevent_initial_call=True
        )
        def update_node_label_size(node_size, label_size, current_fig):
            """Update only node and label sizes using Patch, preserving zoom and layout."""
            if not current_fig.get('data'):
                return dash.no_update
            patched_fig = dash.Patch()
            # Node traces start at index 1 (0 is edges)
            for i in range(1, len(current_fig['data'])):
                trace = current_fig['data'][i]
                # Update marker size
                if 'marker' in trace:
                    # Unknown/missing printers are 5/6 size
                    if 'rgba(128, 128, 128' in str(trace.get('marker', {}).get('color', '')):
                        patched_fig['data'][i]['marker']['size'] = int(node_size * 1 / 2)
                    else:
                        patched_fig['data'][i]['marker']['size'] = node_size
                # Update label size
                if 'textfont' in trace:
                    # Unknown/missing printers are 5/6 size
                    if 'rgba(128, 128, 128' in str(trace.get('textfont', {}).get('color', '')):
                        patched_fig['data'][i]['textfont']['size'] = int(label_size * 1 / 2)
                    else:
                        patched_fig['data'][i]['textfont']['size'] = label_size
            return patched_fig


        
        # Show/hide all printers in network graph legend
        @self.app.callback(
            [Output('network-graph', 'figure', allow_duplicate=True),
             Output('show-all-network-printers-btn', 'style', allow_duplicate=True),
             Output('hide-all-network-printers-btn', 'style', allow_duplicate=True),
             Output('network-legend-visibility-store', 'data', allow_duplicate=True)],
            [Input('show-all-network-printers-btn', 'n_clicks'),
             Input('hide-all-network-printers-btn', 'n_clicks')],
            [State('network-graph', 'figure')],
            prevent_initial_call=True
        )
        def toggle_all_network_printers(show_clicks, hide_clicks, current_fig):
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            show_all = trigger_id == 'show-all-network-printers-btn'
            
            
            # Keep margins consistent with other toggle callbacks
            # Set button styles based on which is active
            show_style = {**self._active_btn_style, 'marginBottom': '8px'} if show_all else {**self._inactive_btn_style, 'marginRight': '5px', 'marginBottom': '8px'}
            hide_style = {**self._active_btn_style, 'marginRight': '5px', 'marginBottom': '8px'} if not show_all else {**self._inactive_btn_style, 'marginBottom': '8px'}
                        
            if not current_fig.get('data'):
                return dash.no_update, show_style, hide_style, dash.no_update
            
            # Use dash.Patch for efficient update
            patched_fig = dash.Patch()
            
            # Update visibility for all printer traces (skip bin_ edge traces and selected_book_ traces)
            if current_fig.get('data'):
                for i, trace in enumerate(current_fig['data']):
                    # Skip edge traces (named bin_0, bin_1, etc.) and selected book traces
                    if trace.get('name', '').startswith('bin_') or trace.get('name', '').startswith('selected_'):
                        continue
                    # Use 'visible' so legend interactions still work (legendonly / True)
                    patched_fig['data'][i]['visible'] = True if show_all else 'legendonly'

            # Build visibility mapping for network legend store
            store_update = {}
            for i, trace in enumerate(current_fig.get('data', [])):
                if trace.get('name', '').startswith('bin_') or trace.get('name', '').startswith('selected_'):
                    continue
                name = trace.get('name', '')
                store_update[name] = True if show_all else 'legendonly'

            return patched_fig, show_style, hide_style, store_update

        # Update network legend visibility store when user toggles legend items in the network graph
        @self.app.callback(
            Output('network-legend-visibility-store', 'data', allow_duplicate=True),
            [Input('network-graph', 'restyleData')],
            [State('network-graph', 'figure'), State('network-legend-visibility-store', 'data')],
            prevent_initial_call=True
        )
        def sync_network_legend_visibility(restyleData, current_fig, store):
            if restyleData is None:
                return dash.no_update
            try:
                changes = restyleData[0]
                idxs = restyleData[1] if len(restyleData) > 1 else None
                new_store = dict(store) if store else {}
                if 'visible' in changes:
                    vals = changes['visible']
                    if idxs:
                        for i, idx in enumerate(idxs):
                            val = vals[i] if isinstance(vals, list) and len(vals) > i else vals[0]
                            name = current_fig['data'][idx].get('name', '')
                            new_store[name] = val
                    else:
                        arr = vals[0] if isinstance(vals, list) and len(vals) == 1 and isinstance(vals[0], list) else vals
                        for idx, tr in enumerate(current_fig.get('data', [])):
                            name = tr.get('name', '')
                            if idx < len(arr):
                                new_store[name] = arr[idx]
                return new_store
            except Exception as e:
                print(f"Warning: sync_network_legend_visibility failed: {e}")
                return dash.no_update

        # Update heatmap legend visibility store when user toggles legend items in the heatmap
        @self.app.callback(
            Output('heatmap-legend-visibility-store', 'data', allow_duplicate=True),
            [Input('similarity-heatmap', 'restyleData')],
            [State('similarity-heatmap', 'figure'), State('heatmap-legend-visibility-store', 'data')],
            prevent_initial_call=True
        )
        def sync_heatmap_legend_visibility(restyleData, current_fig, store):
            if restyleData is None:
                return dash.no_update
            try:
                changes = restyleData[0]
                idxs = restyleData[1] if len(restyleData) > 1 else None
                new_store = dict(store) if store else {}
                if 'visible' in changes:
                    vals = changes['visible']
                    if idxs:
                        for i, idx in enumerate(idxs):
                            val = vals[i] if isinstance(vals, list) and len(vals) > i else vals[0]
                            name = current_fig['data'][idx].get('name', '')
                            new_store[name] = val
                    else:
                        arr = vals[0] if isinstance(vals, list) and len(vals) == 1 and isinstance(vals[0], list) else vals
                        for idx, tr in enumerate(current_fig.get('data', [])):
                            name = tr.get('name', '')
                            if idx < len(arr):
                                new_store[name] = arr[idx]
                return new_store
            except Exception as e:
                print(f"Warning: sync_heatmap_legend_visibility failed: {e}")
                return dash.no_update

        # Keep heatmap show/hide button styles in sync with legend visibility store
        @self.app.callback(
            [Output('show-all-printers-btn', 'style'), Output('hide-all-printers-btn', 'style')],
            [Input('heatmap-legend-visibility-store', 'data')],
            prevent_initial_call=False
        )
        def sync_heatmap_printer_buttons(store):
            # Recreate the active/inactive style dicts to match layout
            active_style = {'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                           'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                           'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                           'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                           'width': '120px', 'minWidth': '120px', 'maxWidth': '120px'}
            inactive_style = {'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                             'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                             'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'width': '120px', 'minWidth': '120px', 'maxWidth': '120px'}
            if not store:
                return dash.no_update, dash.no_update
            vals = list(store.values())
            # Determine if all visible, all hidden, or mixed
            all_visible = all(v is True for v in vals)
            all_hidden = all(v == 'legendonly' for v in vals)
            if all_visible:
                return active_style, inactive_style
            if all_hidden:
                return inactive_style, active_style
            return inactive_style, inactive_style

        # Keep network show/hide printer button styles in sync with legend visibility store
        @self.app.callback(
            [Output('show-all-network-printers-btn', 'style'), Output('hide-all-network-printers-btn', 'style')],
            [Input('network-legend-visibility-store', 'data')],
            prevent_initial_call=False
        )
        def sync_network_printer_buttons(store):
            active_style = {'marginBottom': '8px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                                  'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                                  'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                                                  'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                                                  'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1',
                                                  'width': '120px', 'minWidth': '120px', 'maxWidth': '120px'}
            inactive_style = {'marginRight': '5px', 'marginBottom': '8px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                                  'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                                                  'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                                                  'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                                                  'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1',
                                                  'width': '120px', 'minWidth': '120px', 'maxWidth': '120px'}
            
            if not store:
                return dash.no_update, dash.no_update
            vals = list(store.values())
            all_visible = all(v is True for v in vals)
            all_hidden = all(v == 'legendonly' for v in vals)
            if all_visible:
                return active_style, inactive_style
            if all_hidden:
                return inactive_style, active_style
            return inactive_style, inactive_style

        # Sync selected books from dropdown to network graph and handle show/hide
        @self.app.callback(
            [Output('network-graph', 'figure', allow_duplicate=True),
             Output('network-selected-books-store', 'data'),
             Output('show-selected-books-btn', 'style'),
             Output('hide-selected-books-btn', 'style'),
             Output('network-selected-books-visibility-store', 'data')],
            [Input('additional-books-dropdown', 'value'),
             Input('show-selected-books-btn', 'n_clicks'),
             Input('hide-selected-books-btn', 'n_clicks')],
            [State('network-graph', 'figure'),
             State('network-selected-books-store', 'data'),
             State('umap-positions-store', 'data'),
             State('node-size-slider', 'value'),
             State('label-size-slider', 'value'),
             State('font-type-store', 'data'),
             State('edge-opacity-slider', 'value'),
             State('show-selected-books-btn', 'style'),
             State('hide-selected-books-btn', 'style'),
             State('network-selected-books-visibility-store', 'data')],
            prevent_initial_call=True
        )
        def handle_selected_books_in_network(selected_books, show_clicks, hide_clicks, current_fig, 
                                              stored_books, umap_positions, node_size, label_size, font_type, edge_opacity,
                                              show_btn_style, hide_btn_style, stored_visibility):
            if current_fig is None or not current_fig.get('data'):
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Button styles
            active_style = {**self._active_btn_style, 'marginBottom': '8px'}
            inactive_style = {**self._inactive_btn_style, 'marginBottom': '8px'}
            active_style_mr = {**self._active_btn_style, 'marginRight': '5px', 'marginBottom': '8px'}
            inactive_style_mr = {**self._inactive_btn_style, 'marginRight': '5px', 'marginBottom': '8px'}
            
            selected_books = selected_books or []
            show_selected = trigger_id == 'show-selected-books-btn'
            hide_selected = trigger_id == 'hide-selected-books-btn'
            
            patched_fig = dash.Patch()
            
            # Remove existing selected book traces and edges
            data_to_keep = []
            for trace in current_fig.get('data', []):
                if not trace.get('name', '').startswith('selected_book_') and not trace.get('name', '').startswith('selected_edge_'):
                    data_to_keep.append(trace)
            patched_fig['data'] = data_to_keep
            
            # Determine current visibility state from existing traces
            currently_showing = False
            for trace in current_fig.get('data', []):
                if trace.get('name', '').startswith('selected_book_'):
                    currently_showing = True
                    break
            
            # Determine desired visibility state based on trigger
            if show_selected:
                should_show = True
            elif hide_selected:
                should_show = False
            elif trigger_id == 'additional-books-dropdown':
                # Check which button is active by checking button styles
                show_is_active = show_btn_style.get('backgroundColor') == '#2f4a84'
                hide_is_active = hide_btn_style.get('backgroundColor') == '#2f4a84'
                
                # If "Show selected" is active and not currently showing, show them
                if show_is_active and not currently_showing:
                    should_show = True
                # If "Hide selected" is active and currently showing, hide them
                elif hide_is_active and currently_showing:
                    should_show = False
                else:
                    # Otherwise preserve current state
                    should_show = currently_showing
            else:
                # Default to hidden on first load
                should_show = False
            
            # Add new selected book traces if showing and there are books to show
            if selected_books and should_show:
                umap_array = np.asarray(umap_positions, dtype=np.float32)
                book_list = list(self.books)
                
                for book in selected_books:
                    if book in book_list:
                        book_idx = book_list.index(book)
                        book_color = self._get_book_color(book)
                        
                        # Get printer name for hover
                        printer_name = self.impr_names[book_idx] if book_idx < len(self.impr_names) else 'Unknown'
                        if printer_name in ['n. nan', 'm. missing']:
                            printer_name = 'Unknown'
                        
                        patched_fig['data'].append({
                            'type': 'scatter',
                            'x': [umap_array[book_idx, 0]],
                            'y': [umap_array[book_idx, 1]],
                            'mode': 'markers+text',
                            'marker': {
                                'symbol': 'star',
                                'size': node_size * 1.3 if node_size else 18,
                                'color': book_color,
                                'line': {'width': 2, 'color': 'white'}
                            },
                            'text': [book],
                            'textposition': 'top center',
                            'textfont': {
                                'size': label_size if label_size else 8,
                                'color': book_color,
                                'family': 'Arial, bold'
                            },
                            'hovertemplate': f'{book}<br>Printer: {printer_name}<extra></extra>',
                            'name': f'selected_book_{book}',
                            'showlegend': False
                        })
                
                # Add colored edges for selected books using binned edges
                bins = self._binned_edges.get(font_type, {})
                book_list = list(self.books)
                
                # Use edge opacity from slider state
                edge_opacity_val = edge_opacity if edge_opacity is not None else 1.0
                
                for book in selected_books:
                    if book in book_list:
                        book_idx = book_list.index(book)
                        book_color = self._get_book_color(book)
                        
                        # Convert hex to RGB
                        r = int(book_color[1:3], 16)
                        g = int(book_color[3:5], 16)
                        b = int(book_color[5:7], 16)
                        
                        # Find edges connected to this book from binned edges
                        for bin_idx in range(10):
                            bin_data = bins.get(bin_idx, {'edges': [], 'avg_w': 0})
                            edges_in_bin = bin_data['edges']
                            avg_w = bin_data['avg_w']
                            
                            if not edges_in_bin:
                                continue
                            
                            # Vectorize edge filtering
                            edges_array = np.array(edges_in_bin)
                            mask = (edges_array[:, 0] == book_idx) | (edges_array[:, 1] == book_idx)
                            matching_edges = edges_array[mask]
                            
                            if len(matching_edges) > 0:
                                edges_x = []
                                edges_y = []
                                for i, j in matching_edges:
                                    edges_x.extend([umap_array[i, 0], umap_array[j, 0], None])
                                    edges_y.extend([umap_array[i, 1], umap_array[j, 1], None])
                                
                                # Use same opacity formula as bin edges
                                opacity = min(1.0, avg_w)
                                color = f'rgba({r},{g},{b},{opacity})'
                                patched_fig['data'].append({
                                    'type': 'scatter',
                                    'x': edges_x,
                                    'y': edges_y,
                                    'mode': 'lines',
                                    'line': {'width': 2, 'color': color},
                                    'showlegend': False,
                                    'customdata': [avg_w],
                                    'name': f'selected_edge_{book}_bin{bin_idx}',
                                    'hoverinfo': 'skip'
                                })
                
            # Update button styles based on desired visibility state (independent of whether there are books)
            # Don't update button styles when dropdown changes (clicking nodes)
            if trigger_id == 'additional-books-dropdown':
                # Return no_update for button styles to preserve current state
                return patched_fig, selected_books, dash.no_update, dash.no_update, stored_visibility
            elif should_show:
                show_sel_style = active_style
                hide_sel_style = inactive_style_mr
                store_update = True
            else:
                show_sel_style = inactive_style
                hide_sel_style = active_style_mr
                store_update = False
            
            return patched_fig, selected_books, show_sel_style, hide_sel_style, store_update

    def _register_heatmap_callbacks(self):
        """Heatmap printer toggle and dynamic tick font on zoom."""
        
        # Toggle heatmap printer visibility
        @self.app.callback(
            [Output('similarity-heatmap', 'figure', allow_duplicate=True),
             Output('show-all-printers-btn', 'style', allow_duplicate=True),
             Output('hide-all-printers-btn', 'style', allow_duplicate=True),
             Output('heatmap-legend-visibility-store', 'data', allow_duplicate=True)],
            [Input('show-all-printers-btn', 'n_clicks'),
            Input('hide-all-printers-btn', 'n_clicks')],
            [State('similarity-heatmap', 'figure')],
            prevent_initial_call=True
        )
        def toggle_all_printers(show_clicks, hide_clicks, current_fig):
            # Button styles matching layout
            active_style = {'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                           'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                           'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                           'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                           'width': '130px', 'minWidth': '130px'}
            inactive_style = {'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                             'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                             'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                             'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                             'width': '130px', 'minWidth': '130px'}
            
            if current_fig is None:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update

            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update

            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            show_all = trigger_id == 'show-all-printers-btn'

            patched_fig = dash.Patch()
            for i in range(1, len(current_fig['data'])):
                # Skip overlay traces (they should always be visible)
                if current_fig['data'][i].get('name', '').startswith('overlay_'):
                    continue
                # Use 'visible' so legend interactions still work (legendonly / True)
                patched_fig['data'][i]['visible'] = True if show_all else 'legendonly'

            # Preserve layout settings
            patched_fig['layout']['uirevision'] = 'constant'
            patched_fig['layout']['xaxis']['automargin'] = False
            patched_fig['layout']['yaxis']['automargin'] = False

            # Set button styles based on action
            if show_all:
                show_style = active_style
                hide_style = inactive_style
            else:
                show_style = inactive_style
                hide_style = active_style

            # Build visibility mapping for heatmap legend store
            store_update = {}
            for i in range(1, len(current_fig['data'])):
                if current_fig['data'][i].get('name', '').startswith('overlay_'):
                    continue
                name = current_fig['data'][i].get('name', '')
                store_update[name] = True if show_all else 'legendonly'

            return patched_fig, show_style, hide_style, store_update
        
        # Dynamically adjust tick font size based on zoom level
        @self.app.callback(
            Output('similarity-heatmap', 'figure', allow_duplicate=True),
            [Input('similarity-heatmap', 'relayoutData')],
            [State('similarity-heatmap', 'figure')],
            prevent_initial_call=True
        )
        def adjust_tick_font_on_zoom(relayout_data, current_fig):
            if relayout_data is None or current_fig is None:
                return dash.no_update

            # Ignore autosize events
            if 'autosize' in relayout_data or relayout_data == {}:
                return dash.no_update

            try:
                n_books = len(self.books)
                default_size = max(6, min(12, 350/n_books))

                # Determine visible range
                x_range = None
                y_range = None

                # Check for zoom (range set)
                if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
                    x_range = (relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]'])
                if 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
                    y_range = (relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]'])

                patched_fig = dash.Patch()

                # Check for reset (double-click to reset zoom)
                if 'xaxis.autorange' in relayout_data or 'yaxis.autorange' in relayout_data:
                    # Reset to default font size and hide labels
                    patched_fig['layout']['xaxis']['tickfont'] = {'size': default_size}
                    patched_fig['layout']['yaxis']['tickfont'] = {'size': default_size}
                    patched_fig['layout']['xaxis']['automargin'] = False
                    patched_fig['layout']['yaxis']['automargin'] = False
                    patched_fig['layout']['yaxis']['showticklabels'] = False
                    patched_fig['layout']['margin']['l'] = 25  # Reset to original margin
                    return patched_fig

                # Calculate visible items
                if x_range:
                    visible_x = max(1, abs(x_range[1] - x_range[0]))
                else:
                    visible_x = n_books

                if y_range:
                    visible_y = max(1, abs(y_range[1] - y_range[0]))
                else:
                    visible_y = n_books

                visible_items = min(visible_x, visible_y)

                # Scale font size - larger when zoomed in
                if visible_items <= 5:
                    font_size = 10
                elif visible_items <= 10:
                    font_size = 10
                elif visible_items <= 20:
                    font_size = 10
                elif visible_items <= 35:
                    font_size = 10
                else:
                    font_size = default_size

                # Update font sizes
                patched_fig['layout']['xaxis']['tickfont'] = {'size': font_size, 'family': 'Arial Narrow, Arial, sans-serif'}
                patched_fig['layout']['yaxis']['tickfont'] = {'size': font_size, 'family': 'Arial Narrow, Arial, sans-serif'}
                patched_fig['layout']['xaxis']['automargin'] = False
                patched_fig['layout']['yaxis']['automargin'] = False
                
                # Show y-axis labels when zoomed in to 35 or fewer books
                if visible_items <= 35:
                    patched_fig['layout']['yaxis']['showticklabels'] = True
                    patched_fig['layout']['margin']['l'] = 120  # Add left margin for labels
                else:
                    patched_fig['layout']['yaxis']['showticklabels'] = False
                    patched_fig['layout']['margin']['l'] = 25  # Reset to original margin

                return patched_fig

            except Exception as e:
                print(f"Error adjusting font size: {e}")
                return dash.no_update

    def _register_export_and_comparison_callbacks(self):
        """HTML export and letter comparison panel."""
        
        @self.app.callback(
            [Output('export-status', 'children'),
             Output('download-html', 'data')],
            [Input('export-html-btn', 'n_clicks')],
            [State('font-type-store', 'data'),
             State('similarity-heatmap', 'figure'),
             State('network-graph', 'figure')]
        )
        def export_data(html_clicks, font_type, heatmap_fig, network_fig):
            if not html_clicks:
                return "", None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export figures as interactive HTML for download
            try:
                import plotly.io as pio
                
                threshold = threshold if threshold is not None else 0.1
                
                # Compute stats for export
                if font_type == 'roman':
                    n1hat = self.n1hat_rm
                elif font_type == 'italic':
                    n1hat = self.n1hat_it
                else:
                    n1hat = (self.n1hat_rm + self.n1hat_it) / 2
                
                total_connections = np.sum(n1hat > threshold)
                connected_books = len(np.where(np.sum(n1hat > threshold, axis=0) > 0)[0])
                
                # Create combined HTML with both figures
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Book Typography Similarity Analysis - {timestamp}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ text-align: center; }}
        .figure-container {{ margin: 20px 0; }}
        .info {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .stats {{ background: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Book Typography Similarity Analysis</h1>
    <div class="info">
        <p><strong>Font Type:</strong> {font_type.capitalize()}</p>
        <p><strong>Similarity Threshold:</strong> {threshold}</p>
        <p><strong>Number of Books:</strong> {len(self.books)}</p>
        <p><strong>Exported:</strong> {timestamp}</p>
    </div>
    <div class="stats">
        <h3>Analysis Statistics</h3>
        <p><strong>Total Connections (above threshold):</strong> {total_connections}</p>
        <p><strong>Connected Books:</strong> {connected_books}</p>
        <p><strong>Symbols Analyzed:</strong> {len(self.symbs)}</p>
    </div>
    <h2>Similarity Matrix</h2>
    <div class="figure-container" id="heatmap"></div>
    <h2>Network Graph</h2>
    <div class="figure-container" id="network"></div>
    <script>
        var heatmapData = {json.dumps(heatmap_fig)};
        var networkData = {json.dumps(network_fig)};
        Plotly.newPlot('heatmap', heatmapData.data, heatmapData.layout);
        Plotly.newPlot('network', networkData.data, networkData.layout);
    </script>
</body>
</html>"""
                    
                filename = f'dashboard_export_{font_type}_{timestamp}.html'
                
                return html.P(f"✅ HTML export ready - download will start automatically!", style={'color': 'green', 'fontWeight': 'bold'}), dict(content=html_content, filename=filename)
                
            except Exception as e:
                return html.P(f"❌ Export failed: {str(e)}", style={'color': 'red'}), None
        
        # Callback to show letter comparisons - shows ALL cached images instantly
        @self.app.callback(
            Output('letter-comparison-panel', 'children'),
            [Input('font-type-store', 'data'),
             Input('letter-filter', 'value'),
             Input('additional-books-dropdown', 'value')],
            prevent_initial_call=True
        )
        def update_letter_comparison(font_type, selected_letters, selected_books):
            if not selected_books:
                return html.P("Click on a cell in the similarity matrix or select books above", 
                             style={'textAlign': 'center', 'color': 'gray', 'marginTop': '150px'})
            
            try:
                # Use books from dropdown (which includes clicked books)
                books_to_compare = list(selected_books) if selected_books else []
                
                if not books_to_compare:
                    return html.P("Select books to compare (click matrix or use dropdown)", 
                                 style={'textAlign': 'center', 'color': 'gray', 'marginTop': '150px'})
                
                # Get available letters across all books
                available_letters = set()
                if len(books_to_compare) == 1:
                    # Single book - get letters just for that book
                    book = books_to_compare[0]
                    letters = self._get_available_letters_for_single_book(book, font_type=font_type)
                    available_letters.update(letters)
                else:
                    # Multiple books - get letters across pairs
                    for i, b1 in enumerate(books_to_compare):
                        for b2 in books_to_compare[i+1:]:
                            letters = self._get_available_letters_for_books(b1, b2, font_type=font_type)
                            available_letters.update(letters)
                available_letters = sorted(list(available_letters))
                
                # Calculate column width based on number of books
                n_books = len(books_to_compare)
                col_width = f'{100 // n_books}%'
                
                # Create header with all book names
                header_items = []
                for book in books_to_compare:
                    # Get printer name for this book
                    book_idx = np.where(self.books == book)[0]
                    printer_name = self.impr_names[book_idx[0]] if len(book_idx) > 0 else ''
                    if printer_name in ['n. nan', 'm. missing']:
                        printer_name = 'Unknown'
                    
                    # Get book color with light alpha for background
                    book_color = self._get_book_color(book)
                    r = int(book_color[1:3], 16)
                    g = int(book_color[3:5], 16)
                    b = int(book_color[5:7], 16)
                    bg_color = f'rgba({r}, {g}, {b}, 0.35)'
                    
                    header_items.append(
                        html.Div([
                            html.P(f"{book}", style={'fontSize': '10px', 'fontWeight': 'bold', 'margin': '0', 'wordWrap': 'break-word', 'color': '#333'}),
                            html.P(f"{printer_name}", style={'fontSize': '9px', 'color': '#666', 'margin': '0'})
                        ], style={'display': 'inline-block', 'width': col_width, 'textAlign': 'center', 'verticalAlign': 'top', 'backgroundColor': bg_color, 'padding': '5px', 'borderRadius': '4px', 'boxSizing': 'border-box'})
                    )
                
                comparison_content = [
                    html.Div(header_items, style={'marginBottom': '10px', 'display': 'flex', 'gap': '2px', 'justifyContent': 'center'}),
                    html.P(f"Viewing {n_books} book{'s' if n_books > 1 else ''}", style={'textAlign': 'center', 'fontSize': '11px', 'marginBottom': '10px', 'color': '#666'}),
                ]
                
                if not available_letters:
                    comparison_content.append(
                        html.P(f"No {font_type} letter images found for these books", 
                               style={'textAlign': 'center', 'color': 'gray', 'marginTop': '30px'})
                    )
                else:
                    # Filter to only selected letters
                    letters_to_show = [l for l in available_letters if l in (selected_letters or [])]
                    
                    if not letters_to_show:
                        comparison_content.append(
                            html.P("Select letters above to compare", 
                                   style={'textAlign': 'center', 'color': 'gray', 'marginTop': '30px'})
                        )
                    else:
                        letters_with_images = 0
                        # Show filtered letters instantly from cache
                        for letter in letters_to_show:
                            # Get images for all books using the updated cache-aware function
                            all_book_images = []
                            has_any_images = False
                            for book in books_to_compare:
                                images = self._get_available_images(book, letter, font_type)
                                all_book_images.append(images)
                                if images:
                                    has_any_images = True

                            # Skip if no book has images for this letter
                            if not has_any_images:
                                continue

                            letters_with_images += 1

                            # Build image elements for each book
                            book_columns = []
                            for idx, book_images in enumerate(all_book_images):
                                book = books_to_compare[idx]
                                
                                # Get book color with light alpha for background
                                book_color = self._get_book_color(book)
                                r = int(book_color[1:3], 16)
                                g = int(book_color[3:5], 16)
                                b = int(book_color[5:7], 16)
                                bg_color = f'rgba({r}, {g}, {b}, 0.35)'
                                
                                img_elements = []
                                for img_path, encoded_or_url in book_images:
                                    # If we received a URL for a pre-extracted image, use it directly
                                    src = encoded_or_url

                                    img_elements.append(
                                        html.Img(
                                            src=src,
                                            style={'height': '70px', 'margin': '2px', 'border': '2px solid #ccc', 'borderRadius': '4px'}
                                        )
                                    )
                                if not img_elements:
                                    img_elements.append(html.Span("—", style={'color': '#999', 'fontSize': '24px'}))

                                book_columns.append(
                                    html.Div(img_elements, style={'width': col_width, 'display': 'inline-block', 'textAlign': 'center', 'verticalAlign': 'middle', 'backgroundColor': bg_color, 'padding': '5px', 'borderRadius': '4px', 'boxSizing': 'border-box'})
                                )

                            comparison_content.append(
                                html.Div([
                                    html.H5(f"'{letter}'", style={'marginBottom': '5px', 'textAlign': 'center', 'fontSize': '14px', 'fontWeight': 'bold'}),
                                    html.Div(book_columns, style={'marginBottom': '8px', 'paddingBottom': '8px', 'borderBottom': '1px solid #eee', 'display': 'flex', 'gap': '2px', 'justifyContent': 'center'})
                                ])
                            )
                        
                        if letters_with_images == 0:
                            comparison_content.append(
                                html.P(f"Selected letters have no images for these books.", 
                                       style={'textAlign': 'center', 'color': 'orange', 'marginTop': '30px'})
                            )
                
                return comparison_content
                
            except Exception as e:
                # Enhanced error handling with more details
                import traceback
                error_details = traceback.format_exc()
                print(f"ERROR in update_letter_comparison: {str(e)}")
                print(f"ERROR traceback: {error_details}")
                return html.Div([
                    html.P(f"Error loading comparison: {str(e)}", 
                           style={'textAlign': 'center', 'color': 'red'}),
                    html.P(f"Selected books: {selected_books}", 
                           style={'textAlign': 'center', 'color': 'gray', 'fontSize': '10px'}),
                    html.P(f"Font type: {font_type}", 
                           style={'textAlign': 'center', 'color': 'gray', 'fontSize': '10px'})
                ])
        

    def _setup_callbacks(self):
        """Setup dashboard callbacks — delegates to sub-methods."""
        self._register_font_callbacks()
        self._register_dendrogram_callbacks()
        self._register_core_visualization_callbacks()
        self._register_filter_callbacks()
        self._register_network_callbacks()
        self._register_heatmap_callbacks()
        self._register_export_and_comparison_callbacks()

        # Debug: confirm callback setup finished
        try:
            print(f"DEBUG: _setup_callbacks completed - {len(self.app.callback_map)} callbacks registered; keys sample: {list(self.app.callback_map.keys())[:8]}")
        except Exception as e:
            print("DEBUG: _setup_callbacks completed - callback_map inspect failed:", e)
    

    def _encode_image(self, image_path):
        """Load image directly from disk without caching"""
        try:
            with open(image_path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode()
            return encoded
        except Exception:
            return ""
    
    def run_server(self, debug=True, port=8050, host='127.0.0.1'):
        """
        Run the dashboard server.
        
        Args:
            debug: Enable debug mode (auto-reload on changes)
            port: Port number to run on (default 8050)
            host: Host address. Use '0.0.0.0' to allow external access
        
        To share with others on your local network:
            dashboard.run_server(host='0.0.0.0', port=8050)
            Then share your IP address: http://<your-ip>:8050
        """
        try:
            print(f"DEBUG: Starting server on {host}:{port} with {len(self.app.callback_map)} callbacks registered")
        except Exception as e:
            print("DEBUG: Starting server (callback_map inspect failed):", e)
        self.app.run_server(debug=debug, port=port, host=host)