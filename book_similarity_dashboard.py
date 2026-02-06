import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import json
import base64
import os
import glob
import pickle
from datetime import datetime
import threading
import tempfile
from flask import send_from_directory, abort

class BookSimilarityDashboard:
    # Cache size limits for memory optimization
    MAX_FIGURE_CACHE_SIZE = 10  # Limit total cached figure/aux entries
    MAX_UMAP_CACHE_SIZE = 5     # Limit cached UMAP parameter combinations

    # Default settings for edge cache behavior
    EDGE_CACHE_MAX_FILES = 3  # Keep at most 3 edge cache files by default (LRU)

    def __init__(self):
        self.books = None
        self.impr_names = None
        self.symbs = None
        self.n1hat_rm = None
        self.n1hat_it = None
        self.top_k = 5000  # For network graphs (reduced for memory efficiency)
        self.n_bins = 10 
        
        # Serve images from static folder
        self.letter_images_path = './letter_images/'
        self._cache_dir = './images_cache/'  # Directory for per-book cache files
        os.makedirs(self._cache_dir, exist_ok=True)

        # Track last selected font to know when to reload edge caches
        self._last_font_type = None

        # Lock to protect concurrent cache writes
        self._edge_cache_lock = threading.Lock()

        # Option: keep all caches in memory if True (useful for debugging)
        self.KEEP_ALL_EDGE_CACHES = False

        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.app.layout = html.Div("Loading data, please wait…")
        # Register route to serve pre-extracted images from the images_cache folder
        self._register_image_route()
        
    def set_data(self):
        """
        Initialize the dashboard with your data.
        """
        
        self.n1hat_it = np.load('./n1hat_it_matrix_ordered.npy', mmap_mode='r')
        self.n1hat_rm = np.load('./n1hat_rm_matrix_ordered.npy', mmap_mode='r')
        self.books = np.load('./books_dashboard_ordered.npy', mmap_mode='r')
        self.impr_names = np.load('./impr_names_dashboard_ordered.npy', mmap_mode='r')
        self.symbs = np.load('./symbs_dashboard.npy', mmap_mode='r')
        print(" Loaded ordered .npy files")
        
        # Decide whether to load weight memmaps: only if edge caches are missing
        fonts = ['roman', 'italic', 'combined']
        need_edges = any(not os.path.exists(self._edge_cache_path(font, self.top_k, self.n_bins)) for font in fonts)
        # Check for per-font combined UMAP file (we only load combined positions at startup if present)
        combined_umap_file = f'./umap_combined_50_0.5.npy'
        combined_umap_present = os.path.exists(combined_umap_file)

        # Check the dtype of n1hat_rm, n1hat_it
        print(f"n1hat_rm dtype: {self.n1hat_rm.dtype if self.n1hat_rm is not None else 'None'}, n1hat_it dtype: {self.n1hat_it.dtype if self.n1hat_it is not None else 'None'}")

        if need_edges:
            # Only load weight memmaps when needed for missing caches
            w_rm_mmap = np.load('./w_rm_matrix_ordered.npy', mmap_mode='r')
            w_it_mmap = np.load('./w_it_matrix_ordered.npy', mmap_mode='r')
            self._w_rm_mmap = w_rm_mmap
            self._w_it_mmap = w_it_mmap
            print("Loaded weight matrices as memory-mapped (needed for missing caches).")
        else:
            # Keep attributes None to signal weights are not loaded
            self._w_rm_mmap = None
            self._w_it_mmap = None
            print("Weight matrices not loaded (edge caches present).")

        # Initialize printer markers dictionary (lightweight, no trace objects)
        self._printer_markers = self._get_printer_colors_dict()
                                
        # Image cache removed for memory efficiency
        # Images will be loaded on-demand without caching
        self._load_metadata_for_letter_images()  # 3 images per letter

        # Setup layout. Load combined UMAP positions only if the per-font combined UMAP file exists.
        # We do NOT compute UMAP at startup if the file is missing; computation is deferred to user action.
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
                self._w_rm_mmap = np.load('./w_rm_matrix_ordered.npy', mmap_mode='r')
                self._w_it_mmap = np.load('./w_it_matrix_ordered.npy', mmap_mode='r')
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
    
                
    def _load_umap_positions(self, font_type='combined', w_rm=None, w_it=None, w_combined=None, n_neighbors=50, min_dist=0.5, compute_if_missing=False):
        """Load or compute UMAP positions for a given font_type ('roman', 'italic', 'combined').

        This creates/uses per-font cached files named like: ./umap_{font_type}_{n_neighbors}_{min_dist}.npy
        By default, callers with no font_type supplied get 'combined' (keeps startup behavior).
        The 'compute_if_missing' flag controls whether missing UMAP files are computed or deferred.
        """
        # filename includes font type to allow per-font UMAP position caches
        umap_file = f'./umap_{font_type}_{n_neighbors}_{min_dist}.npy'

        # if file exists, load it
        if os.path.exists(umap_file):
            umap_positions = np.load(umap_file)
            print(f"Loaded UMAP positions from {umap_file}")
            print(f"UMAP positions shape: {umap_positions.shape}")
            return self._orient_umap_positions(umap_positions)

        if not compute_if_missing:
            print(f"UMAP file {umap_file} missing — computation deferred until requested.")
            return None
        
        return None

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
        Build per-book image pickle files if missing, and load global metadata (all_letters).
        """
        # Check if per-book pickle files exist for all books
        with open(f'{self._cache_dir}/images_cache_meta.pkl', 'rb') as f:
            self.meta = pickle.load(f)
        self._all_letters = self.meta.get('all_letters', [])
        print(f"Loaded metadata from images_cache_meta.pkl ({len(self._all_letters)} letters)")


    
    # def _compute_umap_positions(self, distance_matrix, n_neighbors=50, min_dist=0.5, random_state=42):
    #     """Compute UMAP positions from distance matrix"""
    #     if not UMAP_AVAILABLE:
    #         print("UMAP not available, falling back to spring layout")
    #         return None
        
    #     try:
    #         reducer = umap.UMAP(
    #             metric='precomputed',
    #             n_neighbors=n_neighbors,
    #             min_dist=min_dist,
    #             n_components=2,
    #             random_state=random_state,
    #         )
    #         # UMAP expects float32 input for best performance/compatibility
    #         dm = distance_matrix.astype(np.float32, copy=False)
    #         positions = reducer.fit_transform(dm)
    #         return positions
    #     except Exception as e:
    #         print(f"UMAP failed: {e}, falling back to spring layout")
    #         return None
    
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
            print(f"Warning: fallback layout failed ({e}), using zero positions")
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
        """Rotate (swap axes) so the axis with the larger range is placed on the x-axis.
        This helps the network occupy more horizontal space in wide layouts.
        """
        umap_positions = np.asarray(umap_positions, dtype=np.float32)
        # Guard for empty input or unexpected shapes
        if umap_positions.size == 0 or umap_positions.ndim != 2 or umap_positions.shape[1] < 2:
            return umap_positions
        x_range = np.nanmax(umap_positions[:, 0]) - np.nanmin(umap_positions[:, 0])
        y_range = np.nanmax(umap_positions[:, 1]) - np.nanmin(umap_positions[:, 1])
        if y_range > x_range:
            # Swap columns so the larger spread becomes x
            rotated = umap_positions[:, [1, 0]].copy()
            # Debug log (uncomment if needed)
            # print("Rotated UMAP positions: swapped x/y to maximize horizontal spread")
            return rotated
        return umap_positions

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
          - Assumes only WebP files on disk (no PNG/JPG fallback).
          - Does NOT fall back to per-book pickle data — returns [] if no files found.
          - Efficiently scans the book directory once and enforces strict font+case matching.
        """
        book_dir = os.path.join(self._cache_dir, book_name)
        if not os.path.isdir(book_dir):
            return []

        # Only consider .webp files (case-insensitive extension check)
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
            p = os.path.join(book_dir, fname)
            url = f"/images_cache/{book_name}/{fname}"
            results.append((p, url))
        return results
            
    
    
    def _setup_layout(self, w_rm=None, w_it=None):
        """Setup the dashboard layout"""
        
        # Load UMAP for store initialization (figures created on first callback for memory efficiency).
        # Do NOT compute missing UMAP at startup; only load cached combined positions if present.
        initial_umap = self._load_umap_positions(font_type='combined', w_rm=w_rm, w_it=w_it, n_neighbors=50, min_dist=0.5, compute_if_missing=False)
        initial_umap_list = initial_umap.tolist() if initial_umap is not None else None
        print("UMAP positions loaded (if cached), figures will be created on first render")
        
        # Create empty placeholder figures (actual figures created in callback to save memory)
        initial_network_fig = go.Figure()
        initial_heatmap_fig = go.Figure()
        self.app.layout = html.Div([            
            # Modern page title
            html.Div([
                html.H1("Theatre Chapbooks At Scale", 
                       style={'textAlign': 'center', 'margin': '0', 'fontFamily': 'Inter, Arial, sans-serif', 
                              'fontWeight': '700', 'fontSize': '2.2rem', 'color': '#374151',
                              'letterSpacing': '-0.5px'}),
                html.P("A Statistical Comparative Analysis of Typography",
                       style={'textAlign': 'center', 'margin': '5px 0 0 0', 'fontFamily': 'Inter, Arial, sans-serif',
                              'fontWeight': '400', 'fontSize': '1rem', 'color': '#887C57',
                              'letterSpacing': '0.5px'}),
            ], style={'marginBottom': '20px', 'padding': '20px 0'}),
            
            # Control panel - Font Type selector (centered, button style)
            html.Div([
                # Row container
                html.Div([

                    # Label (absolutely positioned)
                    html.Div(
                        html.Label(
                            "Font type",
                            style={
                                'fontSize': '13px',
                                'fontWeight': '600',
                                'color': '#887C57',
                                'fontFamily': 'Inter, Arial, sans-serif'
                            }
                        ),
                        style={'position': 'absolute', 'left': '20px', 'top': '50%', 'transform': 'translateY(-50%)'}
                    ),

                    # Button group (centered in full width)
                    html.Div([
                        html.Button("Combined", id='font-combined-btn', n_clicks=0,
                                    style={'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                        'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                        'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                                        'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                        'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'width': '100px'}),

                        html.Button("Roman", id='font-roman-btn', n_clicks=0,
                                    style={'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                        'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                                        'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                                        'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                        'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'width': '100px'}),

                        html.Button("Italic", id='font-italic-btn', n_clicks=0,
                                    style={'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                        'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                                        'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                                        'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                        'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'width': '100px'}),
                    ], style={
                        'display': 'flex',
                        'justifyContent': 'center',
                        'width': '100%'
                    }),

                    dcc.Store(id='font-type-store', data='combined'),

                ], style={
                    'position': 'relative',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center'
                })
            ],
            style={
                'marginBottom': '15px',
                'padding': '12px 20px',
                'backgroundColor': '#DBD1B5',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.15)'
            }),

            
            # Store for UMAP positions cache - initialize with loaded positions
            dcc.Store(id='umap-positions-store', data=initial_umap_list),
            
            # Store for selected books in network graph
            dcc.Store(id='network-selected-books-store', data=[]),
            # Store to persist whether selected books traces are shown (True) or hidden ('legendonly'/False)
            dcc.Store(id='network-selected-books-visibility-store', data=False),

            # Persist legend visibility across updates (mapping trace name -> True or 'legendonly')
            dcc.Store(id='network-legend-visibility-store', data={}),
            dcc.Store(id='heatmap-legend-visibility-store', data={}),
            
            # Main content with letter comparison panel - styled container like network graph
            html.Div([
                # Section header
                html.Div(
                    html.H3(
                        "Typographic Similarity Analysis",
                        style={
                            "margin": "0",
                            "fontFamily": "Inter, Arial, sans-serif",
                            "fontWeight": "600",
                            "letterSpacing": "0.5px",
                            "color": "#887C57",
                        },
                    ),
                    style={
                        "textAlign": "center",
                        "padding": "6px 0",
                        "backgroundColor": "#F8F5EC",
                        "borderRadius": "6px",
                        "marginBottom": "10px",
                        "boxShadow": "0 1px 2px rgba(0,0,0,0.15)",
                    },
                ),
                # Content row: matrix + letter comparison
                html.Div([
                    # Similarity matrix - left side (45%)
                    html.Div([
                        html.Div(
                            "Similarity Matrix",
                            style={
                                'textAlign': 'center',
                                'marginBottom': '8px',
                                'fontFamily': 'Inter, Arial, sans-serif',
                                'fontWeight': '600',
                                'fontSize': '13px',
                                'color': '#887C57'
                            }
                        ),
                        # Buttons to toggle all printer overlays
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
                                  style={'width': '100%', 'aspectRatio': '1 / 1', 'borderRadius': '8px', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)s', "margin": "0 auto"},
                                  config={'responsive': True})
                    ], style={'flex': '1 1 45%', 'minWidth': '0', 'maxWidth': '48%', 'boxSizing': 'border-box', 'backgroundColor': '#F8F5EC', 'borderRadius': '8px', 'padding': '2px', 'overflow': 'hidden'}),
                    
                    # Letter comparison panel - right side (45%)
                    html.Div([
                        html.Div(
                            "Letter Comparison",
                            style={
                                'textAlign': 'center',
                                'marginBottom': '8px',
                                'fontFamily': 'Inter, Arial, sans-serif',
                                'fontWeight': '600',
                                'fontSize': '13px',
                                'color': '#887C57'
                            }
                        ),
                        
                        # Printer filter
                        html.Div([
                            html.Label("Filter by printer: ", style={'fontWeight': '500', 'marginRight': '10px', 'fontSize': '11px', 'color': '#5a5040', 'fontFamily': 'Inter, Arial, sans-serif'}),
                            dcc.Dropdown(
                                id='printer-filter-dropdown',
                                options=[],  # Will be populated
                                value=None,
                                multi=False,
                                placeholder='All printers',
                                clearable=True,
                                style={'width': '100%', 'fontSize': '11px'}
                            ),
                            html.Button("Select all from this printer", id='select-all-printer-books-btn', n_clicks=0,
                                       style={'marginTop': '5px', 'padding': '6px 12px', 'fontSize': '11px', 'fontWeight': '500',
                                              'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                              'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'none',
                                              'boxShadow': '0 1px 3px rgba(0,0,0,0.2)'}),
                        ], style={'marginBottom': '8px', 'padding': '8px', 'backgroundColor': '#DBD1B5', 'borderRadius': '6px'}),
                        
                        # Book selector for multi-book comparison
                        html.Div([
                            html.Label("Select books: ", style={'fontWeight': '500', 'marginRight': '10px', 'fontSize': '11px', 'color': '#5a5040', 'fontFamily': 'Inter, Arial, sans-serif'}),
                            html.Button("Clear", id='clear-comparison-btn', n_clicks=0, 
                                       style={'float': 'right', 'padding': '4px 10px', 'fontSize': '10px', 'fontWeight': '500',
                                              'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                                              'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                                              'boxShadow': '0 1px 3px rgba(0,0,0,0.2)'}),
                            dcc.Dropdown(
                                id='additional-books-dropdown',
                                options=[],  # Will be populated
                                value=[],
                                multi=True,
                                placeholder='+ Click matrix or select books here...',
                                style={'width': '100%', 'fontSize': '11px'}
                            ),
                            # Store to track clicked books from matrix
                            dcc.Store(id='clicked-books-store', data=[]),
                        ], style={'marginBottom': '8px', 'padding': '8px', 'backgroundColor': '#DBD1B5', 'borderRadius': '6px'}),
                        
                        # Letter filter - now inside comparison panel
                        html.Div([
                            html.Label("Filter: ", style={'fontWeight': '500', 'marginRight': '5px', 'fontSize': '11px', 'color': '#5a5040', 'fontFamily': 'Inter, Arial, sans-serif'}),
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
                            dcc.Checklist(
                                id='letter-filter',
                                options=[],
                                value=[],
                                inline=True,
                                style={'display': 'inline-block', 'fontSize': '11px'},
                                inputStyle={'marginRight': '2px', 'marginLeft': '6px'}
                            )
                        ], style={'marginBottom': '8px', 'padding': '8px', 'backgroundColor': '#DBD1B5', 'borderRadius': '6px'}),
                        
                        html.Div(id='letter-comparison-panel', 
                                style={
                                    'border': '1px solid #d1c7ad', 
                                    'borderRadius': '6px',
                                    'padding': '15px',
                                    'backgroundColor': '#F8F5EC',
                                    'minHeight': '500px',
                                    'maxHeight': '800px',
                                    'overflowY': 'auto'
                                },
                                children=[
                                    html.P("Click on a cell in the similarity matrix or a node in the network graph", 
                                          style={'textAlign': 'center', 'color': '#6b7280', 'marginTop': '200px', 'fontSize': '13px', 'fontFamily': 'Inter, Arial, sans-serif'})
                                ])
                    ], style={'flex': '1 1 45%', 'minWidth': '0', 'maxWidth': '48%', 'boxSizing': 'border-box', 'backgroundColor': '#F8F5EC', 'borderRadius': '8px', 'padding': '10px', 'overflow': 'hidden'})
                ], style={'display': 'flex', 'gap': '2%', 'alignItems': 'flex-start', 'justifyContent': 'space-between'}),
            ], style={
                "width": "100%",
                "backgroundColor": "#DBD1B5",
                "borderRadius": "8px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.15)",
                "padding": "10px",
                "marginBottom": "20px"
            }),
                
                # Network graph section with controls
            html.Div([
                html.Div(
                    html.H3(
                        "Graph of Typographic Similarity",
                        style={
                            "margin": "0",
                            "fontFamily": "Inter, Arial, sans-serif",
                            "fontWeight": "600",
                            "letterSpacing": "0.5px",
                            "color": "#887C57",
                        },
                    ),
                    style={
                        "textAlign": "center",
                        "padding": "6px 0",
                        "backgroundColor": "#F8F5EC",
                        "borderRadius": "6px",
                        "marginBottom": "10px",
                        "boxShadow": "0 1px 2px rgba(0,0,0,0.15)",
                    },
                ),
                # UMAP positioning source selector (Combined / Roman / Italic) - buttons placed on top of the network graph
                html.Div([
                    html.Div("Node positions source:", style={'fontSize': '13px',
                                'fontWeight': '600',
                                'color': '#887C57',
                                'fontFamily': 'Inter, Arial, sans-serif'}),
                    html.Button("Combined", id='umap-pos-combined-btn', n_clicks=0,
                                style={'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                        'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                        'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                                        'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                        'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'width': '100px'}),
                    html.Button("Roman", id='umap-pos-roman-btn', n_clicks=0,
                                style={'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                        'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                                        'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                                        'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                        'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'width': '100px'}),
                    html.Button("Italic", id='umap-pos-italic-btn', n_clicks=0,
                                style={'marginRight': '5px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                        'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                                        'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer',
                                        'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                        'boxShadow': '0 1px 3px rgba(0,0,0,0.2)', 'width': '100px'}),
                    # Store to remember current UMAP position source (default 'combined')
                    dcc.Store(id='umap-pos-source-store', data='combined')
                ], style={'textAlign': 'center', 'marginBottom': '10px'}),

                # Network graph controls row - 3 columns with fixed 30% width each
                html.Div([
                    # Column 1: Hide/Show buttons (30% width, centered content)
                    html.Div([
                        html.Div([
                            html.Button("Hide Labels", id='hide-all-labels-btn', n_clicks=0,
                                           style={'marginRight': '5px', 'marginBottom': '8px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                                  'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                                                  'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                                                  'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                                                  'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1',
                                                  'width': '120px', 'minWidth': '120px'}),
                                html.Button("Show Labels", id='show-all-labels-btn', n_clicks=0,
                                           style={'marginBottom': '8px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                                  'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                                  'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                                                  'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                                                  'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1',
                                                  'width': '120px', 'minWidth': '120px'}),
                                html.Br(),
                                html.Button("Hide Markers", id='hide-all-markers-btn', n_clicks=0,
                                           style={'marginRight': '5px', 'marginBottom': '8px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                                  'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                                                  'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                                                  'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                                                  'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1',
                                                  'width': '120px', 'minWidth': '120px'}),
                                html.Button("Show Markers", id='show-all-markers-btn', n_clicks=0,
                                           style={'marginBottom': '8px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                                  'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                                  'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                                                  'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                                                  'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1',
                                                  'width': '120px', 'minWidth': '120px'}),
                                html.Br(),
                                html.Button("Hide Printers", id='hide-all-network-printers-btn', n_clicks=0,
                                           style={'marginRight': '5px', 'marginBottom': '8px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                                  'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                                                  'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                                                  'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                                                  'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1',
                                                  'width': '120px', 'minWidth': '120px'}),
                                html.Button("Show Printers", id='show-all-network-printers-btn', n_clicks=0,
                                           style={'marginBottom': '8px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                                  'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                                  'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                                                  'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                                                  'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1',
                                                  'width': '120px', 'minWidth': '120px'}),
                                html.Br(),
                                html.Button("Hide Selected", id='hide-selected-books-btn', n_clicks=0,
                                           style={'marginRight': '5px', 'marginBottom': '8px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                                  'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
                                                  'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                                                  'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                                                  'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1',
                                                  'width': '120px', 'minWidth': '120px'}),
                                html.Button("Show Selected", id='show-selected-books-btn', n_clicks=0,
                                           style={'marginBottom': '8px', 'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
                                                  'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
                                                  'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
                                                  'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
                                                  'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1',
                                                  'width': '120px', 'minWidth': '120px'}),
                            ], style={'display': 'inline-block', 'textAlign': 'center'})
                        ], style={'width': '30%', 'flexShrink': '0', 'flexGrow': '0', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                        
                        # Column 2: Sliders stacked vertically (30% width, centered content)
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.Label("Edge Opacity:", style={
                                        'fontSize': '11px',
                                        'fontWeight': '500',
                                        'color': 'dimgray',
                                        'fontFamily': 'Inter, Arial, sans-serif',
                                        'marginBottom': '2px'   # 👈 tighten label spacing
                                    }),
                                    dcc.Slider(
                                        id='edge-opacity-slider',
                                        min=0,
                                        max=2,
                                        step=0.1,
                                        value=1,
                                        marks={
                                            0: {'label': '0', 'style': {'fontSize': '10px'}},
                                            1: {'label': '1', 'style': {'fontSize': '10px'}},
                                            2: {'label': '2', 'style': {'fontSize': '10px'}}
                                        },
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="compact-slider"
                                    )
                                ], style={'marginBottom': '-10px'}),   # 👈 reduce gap between blocks

                                html.Div([
                                    html.Label("Node Size:", style={
                                        'fontSize': '11px',
                                        'fontWeight': '500',
                                        'color': 'dimgray',
                                        'fontFamily': 'Inter, Arial, sans-serif',
                                        'marginBottom': '2px'
                                    }),
                                    dcc.Slider(
                                        id='node-size-slider',
                                        min=6,
                                        max=24,
                                        step=1,
                                        value=12,
                                        marks={
                                            6: {'label': '6', 'style': {'fontSize': '10px'}},
                                            15: {'label': '15', 'style': {'fontSize': '10px'}},
                                            24: {'label': '24', 'style': {'fontSize': '10px'}}
                                        },
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="compact-slider"
                                    ),
                                ], style={'marginBottom': '-10px'}),

                                html.Div([
                                    html.Label("Label Size:", style={
                                        'fontSize': '11px',
                                        'fontWeight': '500',
                                        'color': 'dimgray',
                                        'fontFamily': 'Inter, Arial, sans-serif',
                                        'marginBottom': '2px'
                                    }),
                                    dcc.Slider(
                                        id='label-size-slider',
                                        min=6,
                                        max=24,
                                        step=1,
                                        value=8,
                                        marks={
                                            6: {'label': '6', 'style': {'fontSize': '10px'}},
                                            15: {'label': '15', 'style': {'fontSize': '10px'}},
                                            24: {'label': '24', 'style': {'fontSize': '10px'}}
                                        },
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="compact-slider"
                                    ),
                                ]),
                            ], style={'width': '85%'})
                        ], style={
                            'width': '30%',
                            'flexShrink': '0',
                            'flexGrow': '0',
                            'display': 'flex',
                            'flexDirection': 'column',
                            'justifyContent': 'center'
                        }),

                        # Column 3: UMAP Parameters box (30% width)
                        html.Div([
                            html.Div([
                                html.Div("UMAP Parameters", style={'fontWeight': '600', 'fontSize': '11px', 'color': "#887C57", 'marginBottom': '8px', 'textAlign': 'center', 'fontFamily': 'Inter, Arial, sans-serif'}),
                                html.Div([
                                    html.Div([
                                        html.Label("Number of Neighbors:", style={'fontSize': '10px', 'color': 'dimgray', 'fontFamily': 'Inter, Arial, sans-serif'}),
                                        dcc.Slider(
                                            id='umap-n-neighbors',
                                            min=5,
                                            max=100,
                                            step=5,
                                            value=50,
                                            marks={5: {'label': '5', 'style': {'fontSize': '8px'}}, 50: {'label': '50', 'style': {'fontSize': '8px'}}, 100: {'label': '100', 'style': {'fontSize': '8px'}}},
                                            tooltip={"placement": "bottom", "always_visible": False}
                                        )
                                    ], style={'width': '45%', 'display': 'inline-block', 'marginRight': '5%'}),
                                    
                                    html.Div([
                                        html.Label("Minimum Distance:", style={'fontSize': '10px', 'color': 'dimgray','fontFamily': 'Inter, Arial, sans-serif'}),
                                        dcc.Slider(
                                            id='umap-min-dist',
                                            min=0.0,
                                            max=1.0,
                                            step=0.05,
                                            value=0.5,
                                            marks={0: {'label': '0', 'style': {'fontSize': '8px'}}, 0.5: {'label': '0.5', 'style': {'fontSize': '8px'}}, 1: {'label': '1', 'style': {'fontSize': '8px'}}},
                                            tooltip={"placement": "bottom", "always_visible": False}
                                        )
                                    ], style={'width': '45%', 'display': 'inline-block'}),
                                    
                                    html.Div([
                                        html.Button(
                                                "Recalculate",
                                                id="recalculate-umap-btn",
                                                n_clicks=0,
                                                style={
                                                    "padding": "6px 14px",
                                                    "fontSize": "12px",
                                                    "fontWeight": "500",
                                                    "fontFamily": "Inter, Arial, sans-serif",
                                                    "backgroundColor": "#2f4a84",
                                                    "color": "white",
                                                    "border": "none",
                                                    "borderRadius": "6px",
                                                    "cursor": "pointer",
                                                    "display": "inline-flex",
                                                    "alignItems": "center",
                                                    "justifyContent": "center",
                                                    "boxShadow": "0 1px 3px rgba(0,0,0,0.2)",
                                                    "transition": "background-color 0.15s ease, transform 0.05s ease",
                                                    "lineHeight": "1",
                                                },
                                            )
                                    ], style={'width': '100%', 'textAlign': 'center', 'marginTop': '8px'})
                                ], style={'padding': '5px 0'})
                            ], style={'backgroundColor': "#F8F5EC", 'borderRadius': '8px', 'padding': '8px 10px', 'width': '100%'})
                        ], style={'width': '30%', 'flexShrink': '0', 'flexGrow': '0', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                ], style={'marginBottom': '5px', 'padding': '10px', 'backgroundColor': "#DBD1B5", 'borderRadius': '8px', 'display': 'flex', 'flexWrap': 'nowrap', 'alignItems': 'stretch', 'justifyContent': 'space-between'}),
                
                dcc.Graph(id='network-graph', figure=initial_network_fig, style={'height': '800px', 'marginTop': '8px', 'borderRadius': '8px', 'overflow': 'hidden'})
            ], style={
                "width": "100%",
                "backgroundColor": "#DBD1B5",
                "borderRadius": "8px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.15)",
                "padding": "5px",
            }),
                        
            # Export section - matching theme
            html.Div([
                html.Div(
                    html.H3(
                        "Export",
                        style={
                            "margin": "0",
                            "fontFamily": "Inter, Arial, sans-serif",
                            "fontWeight": "600",
                            "letterSpacing": "0.5px",
                            "color": "#887C57",
                        },
                    ),
                    style={
                        "textAlign": "center",
                        "padding": "6px 0",
                        "backgroundColor": "#F8F5EC",
                        "borderRadius": "6px",
                        "marginBottom": "10px",
                        "boxShadow": "0 1px 2px rgba(0,0,0,0.15)",
                    },
                ),
                html.Div([
                    html.Button("Download HTML", id="export-html-btn", n_clicks=0,
                               style={'padding': '10px 20px', 'backgroundColor': '#2f4a84', 'color': 'white', 
                                      'border': 'none', 'borderRadius': '6px', 'fontSize': '13px', 
                                      'cursor': 'pointer', 'fontWeight': '500', 'fontFamily': 'Inter, Arial, sans-serif',
                                      'boxShadow': '0 1px 3px rgba(0,0,0,0.2)'}),
                ], style={'textAlign': 'center'}),
                html.Div(id="export-status", style={'textAlign': 'center', 'marginTop': '10px', 'fontFamily': 'Inter, Arial, sans-serif', 'color': '#5a5040'})
            ], style={'marginTop': '20px', 'padding': '10px', 'backgroundColor': '#DBD1B5', 'borderRadius': '8px',
                      'boxShadow': '0 2px 4px rgba(0,0,0,0.15)'}),
            
            # Download component for HTML export
            dcc.Download(id="download-html")
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
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
                    umap_pos_array = self._load_umap_positions(font_type='combined', compute_if_missing=False)
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

                    # Try to load per-font UMAP positions (cached only) and use them as layout when available.
                    umap_for_font = self._load_umap_positions(font_type=font_type, compute_if_missing=False)
                    if umap_for_font is not None:
                        umap_array = np.asarray(umap_for_font, dtype=np.float32)
                        store_update = umap_array.tolist()
                        print(f"Loaded cached UMAP positions for font '{font_type}'")
                    else:
                        # Fall back to previously loaded UMAP (likely combined) and keep store_update as-is
                        print(f"No cached UMAP found for font '{font_type}', using existing UMAP positions")

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
        # # Recalculate UMAP positions for selected font (computes and caches per-font UMAP)
        # @self.app.callback(
        #     [Output('umap-positions-store', 'data'), Output('network-graph', 'figure')],
        #     [Input('recalculate-umap-btn', 'n_clicks')],
        #     [State('umap-n-neighbors', 'value'), State('umap-min-dist', 'value'), State('umap-pos-source-store', 'data'),
        #      State('edge-opacity-slider', 'value'), State('network-graph', 'figure'), State('node-size-slider', 'value'), State('label-size-slider', 'value'), State('font-type-store', 'data')],
        #     prevent_initial_call=True
        # )
        # def recalc_umap(n_clicks, n_neighbors, min_dist, umap_pos_source, edge_opacity, current_network_fig, node_size, label_size, current_edge_font):
        #     if not n_clicks:
        #         return dash.no_update, dash.no_update
        #     print(f"Recalculating UMAP for positions source '{umap_pos_source}' (n_neighbors={n_neighbors}, min_dist={min_dist})")
        #     # Compute and cache UMAP positions for this positions source (may take time)
        #     umap_positions = self._load_umap_positions(font_type=umap_pos_source, n_neighbors=n_neighbors, min_dist=min_dist, compute_if_missing=True)
        #     if umap_positions is None:
        #         print("UMAP recomputation failed or was aborted.")
        #         return dash.no_update, dash.no_update
        #     umap_array = np.asarray(umap_positions, dtype=np.float32)
        #     # Ensure edges for current selected edge font exist so graph can be built
        #     try:
        #         self._ensure_precomputed_edges(current_edge_font, top_k=self.top_k, n_bins=self.n_bins)
        #     except Exception as e:
        #         print(f"Warning: Edge ensure failed during UMAP recalc: {e}")
        #     # Recreate network graph using the *selected edge font* (edges unaffected by position source)
        #     network_fig = self._create_network_graph(umap_array, edge_opacity or 1.0, marker_size=node_size, label_size=label_size, font_type=current_edge_font)
        #     return umap_array.tolist(), network_fig

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
            umap_positions = self._load_umap_positions(font_type=pos_source, compute_if_missing=False)
            if umap_positions is None:
                print(f"No cached UMAP positions for source '{pos_source}', computation deferred. Use Recalculate to compute.")
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
        
        # Define button styles for active/inactive states
        active_btn_style = {
            'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
            'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#2f4a84', 'color': 'white',
            'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
            'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
            'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1',
            'width': '120px', 'minWidth': '120px', 'maxWidth': '120px'
        }
        inactive_btn_style = {
            'padding': '8px 16px', 'fontSize': '12px', 'fontWeight': '500',
            'fontFamily': 'Inter, Arial, sans-serif', 'backgroundColor': '#DBD1B5', 'color': '#5a5040',
            'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'display': 'inline-flex',
            'alignItems': 'center', 'justifyContent': 'center', 'boxShadow': '0 1px 3px rgba(0,0,0,0.2)',
            'transition': 'background-color 0.15s ease, transform 0.05s ease', 'lineHeight': '1',
            'width': '120px', 'minWidth': '120px', 'maxWidth': '120px'
        }
        
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
            show_style = {**active_btn_style, 'marginBottom': '8px'} if show_labels else {**inactive_btn_style, 'marginRight': '5px', 'marginBottom': '8px'}
            hide_style = {**active_btn_style, 'marginRight': '5px', 'marginBottom': '8px'} if not show_labels else {**inactive_btn_style, 'marginBottom': '8px'}
            
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
            show_style = {**active_btn_style} if show_markers else {**inactive_btn_style}
            hide_style = {**active_btn_style, 'marginRight': '5px', 'marginBottom': '8px'} if not show_markers else {**inactive_btn_style, 'marginRight': '5px', 'marginBottom': '8px'}
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
            show_style = {**active_btn_style, 'marginBottom': '8px'} if show_all else {**inactive_btn_style, 'marginRight': '5px', 'marginBottom': '8px'}
            hide_style = {**active_btn_style, 'marginRight': '5px', 'marginBottom': '8px'} if not show_all else {**inactive_btn_style, 'marginBottom': '8px'}
                        
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
            active_style = {**active_btn_style, 'marginBottom': '8px'}
            inactive_style = {**inactive_btn_style, 'marginBottom': '8px'}
            active_style_mr = {**active_btn_style, 'marginRight': '5px', 'marginBottom': '8px'}
            inactive_style_mr = {**inactive_btn_style, 'marginRight': '5px', 'marginBottom': '8px'}
            
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
                                    if isinstance(encoded_or_url, str) and (encoded_or_url.startswith('/') or encoded_or_url.startswith('http')):
                                        src = encoded_or_url
                                    else:
                                        # Legacy: base64 encoded image from pickle — assume PNG
                                        src = f"data:image/png;base64,{encoded_or_url}"

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


# Helper function to load your data and run dashboard
def create_dashboard_from_notebook():
    """
    Helper function to create dashboard using your notebook data.
    Call this after running your analysis in the notebook.
    """
    try:
        # Load your data (adjust paths as needed)
        books = np.load('./report_new/names_italics.npy')  # or however you load your books
        
        # You'll need to expose these variables from your notebook
        # For now, creating dummy data - replace with your actual variables
        n_books = len(books)
        w_rm = np.random.rand(n_books, n_books) * 0.5  # Replace with your w_rm
        w_it = np.random.rand(n_books, n_books) * 0.5  # Replace with your w_it
        
        # Make symmetric
        w_rm = (w_rm + w_rm.T) / 2
        w_it = (w_it + w_it.T) / 2
        np.fill_diagonal(w_rm, 0)
        np.fill_diagonal(w_it, 0)
        
        # Load imprinter names if available
        try:
            impr_names = np.load('./report_new/impr_names_rounds.npy')
            impr_names = np.array([name.split()[0][:1]+ '. ' + name.split()[-1] for name in impr_names])
        except:
            impr_names = None
        
        # Create and run dashboard
        dashboard = BookSimilarityDashboard(books, w_rm, w_it, impr_names)
        return dashboard
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please make sure your data files are available and variables are defined.")
        return None


if __name__ == "__main__":
    # Example usage
    print("Creating dashboard...")
    dashboard = create_dashboard_from_notebook()
    if dashboard:
        print("Starting server on http://localhost:8050")
        dashboard.run_server(debug=True)
    else:
        print("Failed to create dashboard. Check your data files.")