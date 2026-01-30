import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform
import json
import base64
import os
import glob
import pickle
from datetime import datetime
import time

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed. Install with 'pip install umap-learn' for UMAP layouts.")

class BookSimilarityDashboard:
    # Cache size limits for memory optimization
    MAX_FIGURE_CACHE_SIZE = 10  # Limit total cached figure/aux entries
    MAX_UMAP_CACHE_SIZE = 5     # Limit cached UMAP parameter combinations

    def __init__(self, books, w_rm, w_it, impr_names=None, symbs=None, n1hat_rm=None, n1hat_it=None, cached_order=None, figures_cache=None):
        """
        Initialize the dashboard with your data.
        
        Args:
            books: array of book names
            w_rm: weight matrix for romans
            w_it: weight matrix for italics  
            impr_names: array of imprinter names (optional)
            symbs: array of symbols analyzed (optional)
            n1hat_rm: n1hat matrix for romans (optional, for hover info)
            n1hat_it: n1hat matrix for italics (optional, for hover info)
            cached_order: pre-computed hierarchical ordering (optional)
            figures_cache: dict of pre-computed figures (optional)
        """
        import time
        init_start = time.time()
        
        self.books = books
        self.w_rm = w_rm
        self.w_it = w_it
        self.n1hat_rm = n1hat_rm
        self.n1hat_it = n1hat_it
        # Reduce precision to save memory (try float16 first)
        try:
            self.w_rm = self.w_rm.astype(np.float16)
            self.w_it = self.w_it.astype(np.float16)
        except:
            self.w_rm = self.w_rm.astype(np.float32)
            self.w_it = self.w_it.astype(np.float32)
        self.impr_names = impr_names if impr_names is not None else np.array([f"Unknown_{i}" for i in range(len(books))])
        self.symbs = symbs if symbs is not None else np.array([f"symbol_{i}" for i in range(w_rm.shape[0] if len(w_rm.shape) > 2 else 26)])
        
        # Remove isolated books
        self.idxs_connected = np.where(np.sum((self.w_rm + self.w_it) > 0, axis=0) > 1)[0]
        
        # Use cached ordering if available, otherwise compute
        if cached_order is not None:
            print("✓ Using cached hierarchical ordering")
            self.idxs_order = cached_order
        else:
            print("Computing hierarchical ordering...")
            metric = 1 - (self.w_rm.astype(np.float64) + self.w_it.astype(np.float64)) / 2
            self.idxs_order = self._hierarchical_order(metric)
        
        # Pre-compute figures cache
        self._figures_cache = figures_cache if figures_cache else {}
        self._figures_cache_file = './dashboard_figures_cache.pkl'
        self._precompute_figures()
        
        
        # Save figures cache to disk for next startup
        self._save_figures_cache()
        
        # Initialize app with static folder for serving images
        self.app = dash.Dash(__name__, 
                             assets_folder='letter_images',
                             suppress_callback_exceptions=True)
        
        # Serve images from static folder
        self.letter_images_path = './letter_images/'
        self._cache_dir = './images_cache/'  # Directory for per-book cache files
        os.makedirs(self._cache_dir, exist_ok=True)

        # Image cache removed for memory efficiency
        # Images will be loaded on-demand without caching
        self._load_or_create_per_book_files(max_per_letter=3)  # 3 images per letter
                
        self._setup_layout()
        self._setup_callbacks()
        
        # Report total init time
        init_elapsed = time.time() - init_start
        print(f"✓ Dashboard initialized in {init_elapsed:.2f} seconds")
    
    def _load_or_create_per_book_files(self, max_per_letter=3):
        """
        Build per-book image pickle files if missing, and load global metadata (all_letters).
        """
        start = time.time()

        # Check if per-book pickle files exist for all books
        missing_books = []
        for book in self.books:
            book_pkl = f"{self._cache_dir}/images_{book}.pkl"
            if not os.path.exists(book_pkl):
                missing_books.append(book)

        if missing_books:
            print(f"Building per-book image pickle files for {len(missing_books)} books...")
            # Build image per-book files
            self._build_image_cache(max_per_letter)
        else:
            print("✓ All per-book image caches found. Loading metadata...")
            # Load general metadata (book_index, etc.)
            try:
                with open(f'{self._cache_dir}/images_cache_meta.pkl', 'rb') as f:
                    self.meta = pickle.load(f)
                self._all_letters = self.meta.get('all_letters', [])
                print(f"  ✓ Loaded metadata from images_cache_meta.pkl ({len(self._all_letters)} letters)")
            except Exception as e:
                print(f"  ✗ Failed to load images_cache_meta.pkl: {e}")
    
        
    def _build_image_cache(self, max_per_letter=3):
        """Build per-book image pickle files from letter_images folder"""
        import time
        from concurrent.futures import ThreadPoolExecutor
        start = time.time()
        
        if not os.path.exists(self.letter_images_path):
            print(f"Warning: {self.letter_images_path} not found")
            return
        
        # Get all PNG files
        all_images = glob.glob(os.path.join(self.letter_images_path, '*.png'))
        print(f"Found {len(all_images)} total images")
        
        # Group by book_font_letter to limit per letter
        grouped = {}
        for img_path in all_images:
            filename = os.path.basename(img_path)
            parts = filename.split('_')
            if len(parts) >= 4:
                key = '_'.join(parts[:-1])
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(img_path)
        
        # Build per-book image cache and metadata
        # Group images by book, font, letter
        book_index = {}
        book_img_dict = {}
        for img_path in all_images:
            filename = os.path.basename(img_path)
            parts = filename.split('_')
            if len(parts) >= 4:
                number = parts[-1].split('.')[0]
                letter_part = parts[-2]
                font = parts[-3]
                book = '_'.join(parts[:-3])
                # Handle case-safe format
                if letter_part.startswith('upper-'):
                    letter = letter_part[6:]
                elif letter_part.startswith('lower-'):
                    letter = letter_part[6:]
                else:
                    letter = letter_part
                # Build book_img_dict for limiting per letter
                key = (book, font, letter)
                if key not in book_img_dict:
                    book_img_dict[key] = []
                book_img_dict[key].append(img_path)
        # For each book, collect images, encode, and save pickle
        books = set([k[0] for k in book_img_dict.keys()])
        all_letters = set()
        for book in books:
            book_data = {}
            available = set()
            # For each (font, letter) in this book
            for (b, font, letter), paths in book_img_dict.items():
                if b != book:
                    continue
                # Limit to max_per_letter
                selected_paths = sorted(paths)[:max_per_letter]
                # Encode images
                encoded_imgs = []
                for img_path in selected_paths:
                    try:
                        with open(img_path, 'rb') as f:
                            encoded = base64.b64encode(f.read()).decode()
                        encoded_imgs.append((img_path, encoded))
                    except:
                        continue
                if encoded_imgs:
                    book_data[(font, letter)] = encoded_imgs
                    available.add((font, letter))
                    all_letters.add(letter)
            # Save pickle for this book
            book_pkl = os.path.join(self._cache_dir, f"images_{book}.pkl")
            try:
                with open(book_pkl, 'wb') as f:
                    pickle.dump(book_data, f)
                print(f"  ✓ Saved {book_pkl} ({len(book_data)} keys)")
            except Exception as e:
                print(f"  ✗ Failed to save {book_pkl}: {e}")
            # Save available (font, letter) for metadata
            book_index[book] = sorted(list(available))
        # Save all unique letters with at least one image
        self._all_letters = sorted(all_letters, key=lambda x: (not x.isupper(), x.lower(), x))
        # Save general metadata file
        self.meta = {'book_index': book_index, 'all_letters': self._all_letters}
        meta_file = os.path.join(self._cache_dir, 'images_cache_meta.pkl')
        try:
            with open(meta_file, 'wb') as f:
                pickle.dump(self.meta, f)
            print(f"  ✓ Saved {meta_file} (metadata for {len(book_index)} books, {len(self._all_letters)} letters)")
        except Exception as e:
            print(f"  ✗ Failed to save {meta_file}: {e}")
        elapsed = time.time() - start
        print(f"Built and saved per-book caches and metadata in {elapsed:.2f} seconds")
    
    def _hierarchical_order(self, distance_matrix):
        """Create hierarchical ordering of books"""
        try:
            # Convert to condensed distance matrix for scipy
            mask = np.triu(np.ones_like(distance_matrix, dtype=bool), k=1)
            condensed = distance_matrix[mask]
            
            # Handle edge cases
            if len(condensed) == 0 or np.all(np.isnan(condensed)) or np.all(np.isinf(condensed)):
                return np.arange(len(distance_matrix))
            
            # Replace inf/nan with max finite value + 1
            max_finite = np.max(condensed[np.isfinite(condensed)]) if np.any(np.isfinite(condensed)) else 1.0
            condensed = np.where(np.isfinite(condensed), condensed, max_finite + 1)
            
            # Perform linkage and optimal leaf ordering
            linkage_matrix = linkage(condensed, method='average')
            optimal_order = optimal_leaf_ordering(linkage_matrix, condensed)
            
            # optimal_leaf_ordering returns the linkage matrix with optimal leaf ordering
            # We need to extract the leaf order from this
            from scipy.cluster.hierarchy import leaves_list
            leaf_order = leaves_list(optimal_order)
            
            print(f"Hierarchical order computed: {leaf_order}")  # Debugging print
            return leaf_order
        except Exception as e:
            print(f"Warning: Hierarchical ordering failed ({e}), using default order")
            return np.arange(len(distance_matrix))
    
    def _precompute_figures(self):
        """Pre-compute and cache essential data for heatmaps and network graphs."""
        import time

# Check if essential data or figures are already cached (support old and new formats)
        cached_available = (
            ('umap_positions_50_0.5' in self._figures_cache)
            )
        
        if cached_available:
            print("✓ Using cached figures/data (heatmaps + networks)")
            return

        print("Pre-computing essential data...")
        start = time.time()

        # Cache network graph data (adjacency matrix + positions)
        default_threshold = 0.1
        default_layout = 'umap'
        umap_positions = self._compute_umap_positions(1 - (self.w_rm.astype(np.float64) + self.w_it.astype(np.float64)) / 2)  # Default UMAP positions

        if 'umap_positions_50_0.5' not in self._figures_cache:
            print("  - Caching combined network data...")
            # Store default UMAP positions (ensure small memory footprint)
            self._figures_cache['umap_positions_50_0.5'] = umap_positions
            # Trim figures cache if it grows beyond MAX_FIGURE_CACHE_SIZE
            while len(self._figures_cache) > self.MAX_FIGURE_CACHE_SIZE:
                # Remove the oldest non-default entry
                for k in list(self._figures_cache.keys()):
                    if k != 'umap_positions_50_0.5':
                        del self._figures_cache[k]
                        break
            
        elapsed = time.time() - start
        print(f"✓ Essential data cached in {elapsed:.2f} seconds")
    
    def _rebuild_heatmap(self, font_type, title):
        """Rebuild a heatmap figure from cached data."""
        if font_type == 'combined':
            matrix = (self.w_rm + self.w_it) / 2
            n1hat = self.n1hat_rm + self.n1hat_it
        elif font_type == 'roman':
            matrix = self.w_rm
            n1hat = self.n1hat_rm
        else:
            matrix = self.w_it
            n1hat = self.n1hat_it

        # Apply hierarchical ordering
        # order = self.idxs_order        
        # print(f"Applying hierarchical order: {order}")  # Debugging print
        # # Apply the hierarchical order
        # ordered_matrix = matrix[np.ix_(order, order)]
        # ordered_n1hat = n1hat[np.ix_(order, order)] if n1hat is not None else None
        
        # print(f"Ordered matrix shape: {ordered_matrix.shape}")  # Debugging print
        # Use the existing _create_heatmap method with ordered data
        return self._create_heatmap(matrix, n1hat, title, reorder=True)

    
    def _save_figures_cache(self):
        """Save figures cache to disk"""
        import pickle
        from datetime import datetime
        try:
            # Trim cache before saving to keep disk cache small
            default_key = 'umap_positions_50_0.5'
            while len(self._figures_cache) > self.MAX_FIGURE_CACHE_SIZE:
                for k in list(self._figures_cache.keys()):
                    if k != default_key:
                        del self._figures_cache[k]
                        break

            cache_data = {
                'data': self._figures_cache,  # Now contains essential data, not full figures
                'cached_at': datetime.now().isoformat(),
                'version': '1.2'
            }
            with open(self._figures_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Saved figures cache ({len(self._figures_cache)} items)")
        except Exception as e:
            print(f"⚠ Failed to save figures cache: {e}")
    
    def _get_initial_umap_positions(self):
        """Get initial UMAP positions from cache for default parameters"""
        default_cache_key = 'umap_positions_50_0.5'
        if default_cache_key in self._figures_cache:
            positions = self._figures_cache[default_cache_key]
            print(f"✓ Initializing UMAP store from cache ({len(positions)} positions)")
            return positions.tolist() if hasattr(positions, 'tolist') else positions
        return None
    
    def _get_cached_heatmap(self, font_type):
        """Get heatmap by rebuilding from cached data or return cached figure"""
        
        title = f"Book Similarity Matrix ({font_type.capitalize()})"
        return self._rebuild_heatmap(font_type, title)

    
    def _compute_umap_positions(self, distance_matrix, n_neighbors=50, min_dist=0.5, random_state=42):
        """Compute UMAP positions from distance matrix"""
        if not UMAP_AVAILABLE:
            print("UMAP not available, falling back to spring layout")
            return None
        
        try:
            reducer = umap.UMAP(
                metric='precomputed',
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=2,
                random_state=random_state,
            )
            # UMAP expects float32 input for best performance/compatibility
            dm = distance_matrix.astype(np.float32, copy=False)
            positions = reducer.fit_transform(dm)
            return positions
        except Exception as e:
            print(f"UMAP failed: {e}, falling back to spring layout")
            return None
    
    def _get_printer_colors(self):
        """Get color mapping for printers - matching heatmap colors/markers"""
        unique_imprs = [impr for impr in np.unique(self.impr_names) 
                       if impr not in ['n. nan', 'm. missing', 'Unknown']]
        
        # Use SAME colors and markers as heatmap diagonal markers
        markers = ['circle', 'square', 'diamond']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        
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

    def _create_network_graph(self, weight_matrix, threshold=0.1, layout_type='umap',
                         n_neighbors=50, min_dist=0.5, umap_positions=None, edge_opacity=0.3, n1hat_matrix=None,
                         marker_size=12, label_size=8):
        """Create network graph from weight matrix with UMAP positioning and printer colors"""
        
        # Filter by threshold (vectorized for performance)
        mask = weight_matrix > threshold
        upper_mask = np.triu(mask, k=1)  # Upper triangle, exclude diagonal
        i_indices, j_indices = np.where(upper_mask)
        edges = list(zip(i_indices, j_indices))
        edge_weights = weight_matrix[upper_mask]
        
        edges_array = np.array(edges)  # For vectorized operations
        
        if not edges:
            print(f"Warning: No edges found with threshold {threshold}. Max similarity: {np.max(weight_matrix):.4f}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"No connections above threshold {threshold:.3f}<br>Max similarity: {np.max(weight_matrix):.4f}<br>Try lowering the threshold",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(title="No Network Connections Found")
            return fig
        
        
        # Calculate layout positions
        if layout_type == 'umap' and UMAP_AVAILABLE:
            if umap_positions is not None:
                # Use provided positions
                pos = {i: umap_positions[i] for i in range(len(self.books))}
            else:
                # Compute UMAP positions from distance matrix
                distance_matrix = 1 - ((self.w_rm + self.w_it) / 2)  # Convert similarity to distance
                np.fill_diagonal(distance_matrix, 0)
                print("Computing UMAP positions for network layout...", umap_positions)
                positions = self._compute_umap_positions(distance_matrix, n_neighbors, min_dist)
                if positions is not None:
                    pos = {i: positions[i] for i in range(len(self.books))}
                else:
                    pos = None
        else:
            pos = None  # Will use spring layout
        
        # Get printer colors and markers (matching heatmap)
        impr_to_color, impr_to_marker, unique_imprs = self._get_printer_colors()
        
        # Convert pos to array for vectorized operations
        pos_array = np.array([pos[i] for i in range(len(pos))])
        impr_array = np.array(self.impr_names)
        
        # Create figure
        fig = go.Figure()
        
        # Add ALL edges as a SINGLE trace using None separators (much faster for legend interactions)
        # This dramatically reduces the number of traces from hundreds to just one
        x0 = pos_array[edges_array[:, 0], 0]
        x1 = pos_array[edges_array[:, 1], 0]
        y0 = pos_array[edges_array[:, 0], 1]
        y1 = pos_array[edges_array[:, 1], 1]

        # Single edge trace with opacity varying by weight (edge_weight * edge_opacity)
        # try:
        # Parameters you can tweak
        bottom_pct = 0.90   # fraction of edges to send to zero (bulk)
        top_pct = 0.999     # top fraction that should become alpha ~1
        gamma = 2.0         # >1 makes mapping more aggressive (pushes values toward 0/1)
        n_bins = 8          # number of binned traces (performance vs fidelity)

        eps = 1e-12
        # percentile anchors
        p_low = np.quantile(edge_weights, bottom_pct)
        p_high = np.quantile(edge_weights, top_pct)

        # safety if distribution degenerate
        if p_high <= p_low:
            p_low = np.min(edge_weights)
            p_high = np.max(edge_weights) if np.max(edge_weights) > p_low else p_low + eps

        # log-scale mapping from weight -> [0,1]
        log_w = np.log10(edge_weights + eps)
        log_low = np.log10(p_low + eps)
        log_high = np.log10(p_high + eps)
        raw = (log_w - log_low) / (log_high - log_low)
        scaled = np.clip(raw, 0.0, 1.0) ** gamma   # raise to gamma for more contrast (optional)

        # bin scaled alphas and add one trace per bin (concatenated lines with None separators)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(scaled, bin_edges) - 1

        for b in range(n_bins):
            mask = bin_indices == b
            if not np.any(mask):
                continue
            # build None-separated lines for this bin
            xs = np.concatenate([x0[mask], x1[mask], np.full(np.sum(mask), None, dtype=object)])
            ys = np.concatenate([y0[mask], y1[mask], np.full(np.sum(mask), None, dtype=object)])
            # use midpoint of bin for display alpha
            alpha = ((bin_edges[b] + bin_edges[b+1]) / 2.0)
            color = f"rgba(100,100,100,{alpha})"
            print(color)
            fig.add_trace(go.Scatter(
                x=xs.tolist(), y=ys.tolist(),
                mode="lines",
                line=dict(width=1, color=color),
                showlegend=False,
                name=f"edges_{alpha:.2f}",
            ))  

            # # Prepare per-edge intensity values (0..1) from weights            
            # for (x0_, x1_, y0_, y1_, w) in zip(x0, x1, y0, y1, edge_weights):

            #     alpha = float(w) * edge_opacity
            #     color = f"rgba(100,100,100,{alpha})"

            #     fig.add_trace(go.Scatter(
            #         x=[x0_, x1_],
            #         y=[y0_, y1_],
            #         mode="lines",
            #         line=dict(width=2, color=color),
            #         showlegend=False,
            #         name='edges',
            #     ))
        # except Exception:
        #     print('Trace Fallback')
        #     # Fallback to uniform color if per-segment coloring is not supported by the renderer
        #     fig.add_trace(go.Scatter(
        #         x=edge_x, y=edge_y,
        #         mode='lines',
        #         line=dict(width=2, color=f'rgba(100, 100, 100, {edge_opacity})'),
        #         hoverinfo='text',
        #         hovertext=edge_text,
        #         showlegend=False,
        #         name='edges',
        #     ))
        
        # Add nodes colored by printer, one trace per printer for legend
        # Using same colors and markers as heatmap diagonal
        for impr in unique_imprs:
            node_indices = np.where(impr_array == impr)[0]
            if len(node_indices) == 0:
                continue
            
            node_x = pos_array[node_indices, 0].tolist()
            node_y = pos_array[node_indices, 1].tolist()
            node_text = [f"{self.books[i]}<br>Printer: {self.impr_names[i]}" 
                        for i in node_indices]
            # Show printer name as label on top of nodes
            node_labels = [impr for _ in node_indices]
            
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
                textfont=dict(size=label_size, color=impr_to_color[impr]),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=node_text,
                name=impr,
                legendgroup=impr,
                showlegend=True
            ))
        
        # Add nodes for unknown/missing printers with 0.5 alpha and no label
        unknown_mask = np.isin(impr_array, ['n. nan', 'm. missing', 'Unknown'])
        unknown_indices = np.where(unknown_mask)[0]
        if len(unknown_indices) > 0:
            node_x = pos_array[unknown_indices, 0].tolist()
            node_y = pos_array[unknown_indices, 1].tolist()
            node_text = [f"{self.books[i]}<br>Printer: Unknown" 
                        for i in unknown_indices]
            # No label for unknown/missing printers
            node_labels = ['' for _ in unknown_indices]
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    symbol='circle',  # Circle marker for unknown
                    size=int(marker_size * 5 / 6),
                    color='rgba(128, 128, 128, 0.5)',  # 0.5 alpha
                    line=dict(width=1, color='white')
                ),
                text=node_labels,
                textposition='top center',
                textfont=dict(size=int(label_size * 5 / 6), color='rgba(128, 128, 128, 0.5)'),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=node_text,
                name='Unknown',
                legendgroup='Unknown',
                showlegend=True
            ))
        
        fig.update_layout(
            title=dict(text='Book Similarity Network', x=0.5, font=dict(size=16)),
            showlegend=True,
            legend=dict(
                title=dict(text="Printers", font=dict(size=14)),
                x=1.02,
                y=1,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='gray',
                borderwidth=1,
                font=dict(size=11)
            ),
            hovermode='closest',
            margin=dict(b=20, l=5, r=150, t=40),
            annotations=[dict(
                text="Node color/marker = printer (matching matrix), Edge opacity controlled by slider",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="gray", size=10)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y'),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
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
        print(f"DEBUG: Found letters for {book1} and {book2}: {result}")
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
        """Get available encoded images for a specific book, letter, and font type from the per-book pickle cache."""
        
        # Determine which font types to search for
        if font_type == 'combined':
            font_types_to_search = ['roman', 'italic']
        else:
            font_types_to_search = [font_type]

        book_pkl = os.path.join(self._cache_dir, f"images_{book_name}.pkl")
        if not os.path.exists(book_pkl):
            print(f"Warning: Cache file not found for book {book_name}: {book_pkl}")
            return []

        try:
            with open(book_pkl, 'rb') as f:
                book_data = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache for {book_name}: {e}")
            return []

        images = []
        for ft in font_types_to_search:
            key = (ft, letter)
            if key in book_data:
                images.extend(book_data[key])  # Each entry is (img_path, encoded)

        # Optionally sort by image path number if needed
        def extract_number(item):
            path = item[0]
            filename = os.path.basename(path)
            try:
                return int(filename.split('_')[-1].split('.')[0])
            except ValueError:
                return 0

        return sorted(images, key=extract_number)
    
    def _create_heatmap(self, matrix, n1hat, title="Similarity Matrix", reorder=False):
        """Create similarity matrix heatmap matching notebook style"""
        
        try:
            if reorder and hasattr(self, 'idxs_order') and self.idxs_order is not None:
                # Ensure idxs_order is 1D and contains valid indices
                if self.idxs_order.ndim == 1 and np.all(self.idxs_order < len(self.books)):
                    ordered_matrix = matrix[np.ix_(self.idxs_order, self.idxs_order)]
                    ordered_books = self.books[self.idxs_order]
                    ordered_impr = self.impr_names[self.idxs_order] if self.impr_names is not None else None
                    ordered_n1hat = n1hat[np.ix_(self.idxs_order, self.idxs_order)] if n1hat is not None else None
                    print(f"Ordered books: {ordered_books}")  # Debugging print
                else:
                    print(f"Warning: Invalid idxs_order shape {self.idxs_order.shape}, using default order")
                    ordered_matrix = matrix
                    ordered_books = self.books
                    ordered_impr = self.impr_names
                    ordered_n1hat = n1hat
            else:
                ordered_matrix = matrix
                ordered_books = self.books
                ordered_impr = self.impr_names
                ordered_n1hat = n1hat

            # Create labels in notebook format: "book | imprinter"
            if ordered_impr is not None:
                labels = [f'{book}' for book in ordered_books]
            else:
                labels = ordered_books
            print(f"Final tick labels: {labels}")  # Debugging print
                
        except Exception as e:
            print(f"Warning: Error in reordering ({e}), using default order")
            ordered_matrix = matrix
            ordered_books = self.books
            ordered_impr = self.impr_names
            ordered_n1hat = n1hat
            if ordered_impr is not None:
                labels = [f'{book}' for book in ordered_books]
            else:
                labels = ordered_books

        # Create discrete colormap matching notebook style
        #vmin, vmax = ordered_matrix.min(), ordered_matrix.max()
        vmin = ordered_n1hat.min() if ordered_n1hat is not None else ordered_matrix.min()  # Fixed range for similarity
        vmax = ordered_n1hat.max() if ordered_n1hat is not None else ordered_matrix.max()  # Fixed range for similarity
        
        # Handle edge case where all values are the same
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
        
        n_books = len(labels)
        
        # Create the main heatmap
        # Prepare customdata: for each cell, [impr_y, impr_x] (vectorized)
        if ordered_impr is not None:
            impr_array = np.array(ordered_impr)
            missing_mask = np.isin(impr_array, ['n. nan', 'm. missing'])
            customdata = np.empty((len(labels), len(labels), 2), dtype=object)
            # For rows (y): repeat for each column
            customdata[:, :, 0] = np.where(missing_mask[:, None], '', impr_array[:, None])
            # For columns (x): repeat for each row
            customdata[:, :, 1] = np.where(missing_mask[None, :], '', impr_array[None, :])
        else:
            customdata = None

        fig = go.Figure(data=go.Heatmap(
            z=ordered_n1hat if ordered_n1hat is not None else ordered_matrix,
            x=labels,
            y=labels,
            colorscale='viridis',  # Match notebook colormap
            reversescale=False,    # Don't reverse (notebook doesn't)
            customdata=customdata,
            hoverongaps=False,
            hovertemplate=(
                'Book 1: %{y} (%{customdata[0]})<br>'
                'Book 2: %{x} (%{customdata[1]})<br>'
                'Shared Types: %{z:.0f}<extra></extra>'
            ),
            showscale=False  # Hide colorbar to avoid legend overlap
        ))
        
        # Add printer markers on diagonal and row/column overlays
        if ordered_impr is not None:
            # Create printer color/marker mapping like notebook
            unique_imprs = [impr for impr in np.unique(ordered_impr) if impr not in ['n. nan', 'm. missing']]
            
            # Use consistent colors and markers for printers
            markers = ['circle', 'square', 'diamond']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
            
            printer_markers = {}
            for i, impr in enumerate(unique_imprs):
                printer_markers[impr] = {
                    'marker': markers[i % len(markers)],
                    'color': colors[i % len(colors)]
                }
            
            # Get all indices for each printer (for individual row/column overlays)
            printer_indices = {}
            for impr in unique_imprs:
                indices = [i for i, p in enumerate(ordered_impr) if p == impr]
                if indices:
                    printer_indices[impr] = indices
            
            # Add row/column overlays as shapes with printer name embedded
            for impr, indices in printer_indices.items():
                if impr in printer_markers:
                    color = printer_markers[impr]['color']
                    # Convert hex to rgba with alpha
                    rgba_color = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)"
                    
                    # Skip overlays for performance - keep only diagonal markers
                    # for idx in indices:
                    #     # Horizontal band (row overlay) for this book
                    #     fig.add_shape(...)
                    #     # Vertical band (column overlay) for this book
                    #     fig.add_shape(...)
            
            # Add diagonal markers as scatter traces (respond to legend clicks)
            for impr, indices in printer_indices.items():
                if impr in printer_markers:
                    marker_info = printer_markers[impr]
                    # Use labels for categorical axis
                    x_positions = [labels[i] for i in indices]
                    y_positions = [labels[i] for i in indices]
                    types = [ordered_n1hat[i, i] if ordered_n1hat is not None else ordered_matrix[i, i] for i in indices]
                    
                    fig.add_trace(go.Scatter(
                        x=x_positions,
                        y=y_positions,
                        mode='markers',
                        marker=dict(
                            symbol=marker_info['marker'],
                            size=10,
                            color=marker_info['color'],
                            line=dict(color='white', width=1)
                        ),
                        showlegend=True,
                        legendgroup=impr,
                        name=impr,
                        customdata=np.array(types)[:, None],
                        hovertemplate=f'Printer: {impr}<br>Book: %{{x}}<br>Types: %{{customdata[0]}}<extra></extra>'
                    ))
        
        # Calculate figure size for web viewing - ensure square aspect ratio
        base_size = max(700, min(1000, n_books * 25))  # Scale with books, web-optimized
        fig_height = base_size
        fig_width = base_size  # Keep exactly square
        
        # Clean layout matching notebook style exactly
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            width=fig_width,
            height=fig_height,
            uirevision='constant',  # Preserve zoom/pan state and prevent layout recalculation
            xaxis=dict(
                title="",  # No axis title like notebook
                side="bottom",
                tickangle=90,  # Rotate labels like notebook 
                showgrid=False,
                tickfont=dict(size=max(6, min(12, 350/n_books))),  # Size based on number of books
                scaleanchor="y",  # Lock aspect ratio
                scaleratio=1,    # 1:1 aspect ratio (square)
                automargin=False,  # Prevent automatic margin adjustment
                fixedrange=False,  # Allow zooming
                constrain='domain',
                categoryorder='array',  # Preserve our label order
                categoryarray=labels    # Use the ordered labels
            ),
            yaxis=dict(
                title="",  # No axis title like notebook
                showgrid=False,
                autorange='reversed',  # Match notebook orientation
                tickfont=dict(size=max(6, min(12, 350/n_books))),  # Size based on number of books
                constrain="domain",  # Keep within plot area
                automargin=False,  # Prevent automatic margin adjustment
                fixedrange=False,  # Allow zooming
                categoryorder='array',  # Preserve our label order
                categoryarray=labels    # Use the ordered labels
            ),
            margin=dict(l=180, r=180, t=60, b=180),  # Fixed margins for labels
            plot_bgcolor='white',
            paper_bgcolor='white',
            # Legend configuration for printers - positioned above colorbar
            legend=dict(
                title=dict(text="Printers", font=dict(size=14, color='black')),
                xanchor='left',
                x=1.01,  # Position relative to plot area
                y=1,
                yanchor='top',
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='gray',
                borderwidth=1,
                font=dict(size=11, color='black'),
                itemsizing='constant',
                itemwidth=30,
                tracegroupgap=4
            ) if ordered_impr is not None else None
        )
        
        return fig
    
    def _setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Book Typography Similarity Analysis Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Control panel - Font Type selector (main control)
            html.Div([
                html.Div([
                    html.Label("Analysis Mode:", style={'fontSize': '16px', 'fontWeight': 'bold', 'marginRight': '15px', 'color': '#333'}),
                    dcc.RadioItems(
                        id='font-type-dropdown',
                        options=[
                            {'label': ' Combined (Roman + Italic)', 'value': 'combined'},
                            {'label': ' Roman only', 'value': 'roman'},
                            {'label': ' Italic only', 'value': 'italic'}
                        ],
                        value='combined',
                        inline=True,
                        inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                        labelStyle={'marginRight': '25px', 'fontSize': '14px', 'cursor': 'pointer', 'padding': '8px 12px', 
                                   'backgroundColor': '#fff', 'borderRadius': '5px', 'border': '1px solid #ddd'}
                    )
                ], style={'textAlign': 'center'}),
                # Hidden elements to maintain compatibility - threshold fixed at 0.1, layout always umap
                dcc.Store(id='threshold-slider', data=0.1),
                dcc.Store(id='layout-dropdown', data='umap'),
            ], style={'marginBottom': 10, 'padding': '15px 20px', 'backgroundColor': '#f0f8ff', 'borderRadius': '8px', 'border': '2px solid #4a90d9'}),
            
            # Store for UMAP positions cache - initialize with cached default positions if available
            dcc.Store(id='umap-positions-store', data=self._get_initial_umap_positions()),
            
            # Main content with letter comparison panel
            html.Div([
                html.Div([
                    # Similarity matrix - shown first
                    html.Div([
                           # Buttons to toggle all printer overlays and alpha overlays
                           html.Div([
                            html.Button("Show all printers", id='show-all-printers-btn', n_clicks=0,
                                    style={'marginRight': '5px', 'padding': '4px 10px', 'fontSize': '11px', 
                                        'backgroundColor': '#e8f4e8', 'border': '1px solid #4a4', 'borderRadius': '3px', 'cursor': 'pointer'}),
                            html.Button("Hide all printers", id='hide-all-printers-btn', n_clicks=0,
                                    style={'padding': '4px 10px', 'fontSize': '11px',
                                        'backgroundColor': '#f4e8e8', 'border': '1px solid #a44', 'borderRadius': '3px', 'cursor': 'pointer'}),
                           ], style={'marginBottom': '5px', 'textAlign': 'center'}),
                        dcc.Graph(id='similarity-heatmap')
                    ], style={'flex': '0 0 55%', 'boxSizing': 'border-box'}),
                    
                    # Letter comparison panel - wider
                    html.Div([
                        html.H4("Letter Comparison", style={'textAlign': 'center', 'marginBottom': '10px'}),
                        
                        # Printer filter
                        html.Div([
                            html.Label("Filter by printer: ", style={'fontWeight': 'bold', 'marginRight': '10px', 'fontSize': '12px'}),
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
                                       style={'marginTop': '5px', 'padding': '3px 8px', 'fontSize': '10px', 'backgroundColor': '#ffe8cc', 'border': '1px solid #cc8800', 'borderRadius': '3px', 'cursor': 'pointer', 'display': 'none'}),
                        ], style={'marginBottom': '8px', 'padding': '5px', 'backgroundColor': '#fff8e8', 'borderRadius': '5px'}),
                        
                        # Book selector for multi-book comparison
                        html.Div([
                            html.Label("Select books: ", style={'fontWeight': 'bold', 'marginRight': '10px', 'fontSize': '12px'}),
                            html.Button("Clear", id='clear-comparison-btn', n_clicks=0, 
                                       style={'float': 'right', 'padding': '2px 8px', 'fontSize': '10px', 'backgroundColor': '#ffcccc', 'border': '1px solid #cc0000', 'borderRadius': '3px', 'cursor': 'pointer'}),
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
                        ], style={'marginBottom': '10px', 'padding': '5px', 'backgroundColor': '#e8f4e8', 'borderRadius': '5px'}),
                        
                        # Letter filter - now inside comparison panel
                        html.Div([
                            html.Label("Filter: ", style={'fontWeight': 'bold', 'marginRight': '5px', 'fontSize': '12px'}),
                            html.Button("All", id='select-all-letters', n_clicks=0, style={'marginRight': '3px', 'padding': '2px 6px', 'fontSize': '11px'}),
                            html.Button("None", id='select-no-letters', n_clicks=0, style={'marginRight': '8px', 'padding': '2px 6px', 'fontSize': '11px'}),
                            html.Button("a-z", id='select-lowercase', n_clicks=0, style={'marginRight': '3px', 'padding': '2px 6px', 'fontSize': '11px'}),
                            html.Button("A-Z", id='select-uppercase', n_clicks=0, style={'marginRight': '8px', 'padding': '2px 6px', 'fontSize': '11px'}),
                            dcc.Checklist(
                                id='letter-filter',
                                options=[],
                                value=[],
                                inline=True,
                                style={'display': 'inline-block', 'fontSize': '12px'},
                                inputStyle={'marginRight': '2px', 'marginLeft': '6px'}
                            )
                        ], style={'marginBottom': '10px', 'padding': '5px 10px', 'backgroundColor': '#f0f8ff', 'borderRadius': '5px'}),
                        
                        html.Div(id='letter-comparison-panel', 
                                style={
                                    'border': '2px solid #ddd', 
                                    'borderRadius': '5px',
                                    'padding': '15px',
                                    'backgroundColor': '#fafafa',
                                    'minHeight': '500px',
                                    'maxHeight': '800px',
                                    'overflowY': 'auto'
                                },
                                children=[
                                    html.P("Click on a cell in the similarity matrix or a node in the network graph", 
                                          style={'textAlign': 'center', 'color': 'gray', 'marginTop': '200px', 'fontSize': '14px'})
                                ])
                    ], style={'flex': '0 0 43%', 'boxSizing': 'border-box'})
                ], style={'marginBottom': '30px', 'display': 'flex', 'gap': '2%', 'alignItems': 'flex-start', 'flexWrap': 'wrap'}),
                
                # Network graph section with controls
                html.Div([
                    html.H3("Network Graph", style={'textAlign': 'center', 'marginBottom': '10px'}),
                    
                    # Network graph controls row
                    html.Div([
                        # Show/hide labels buttons
                        html.Div([
                            html.Button("Show labels", id='show-all-labels-btn', n_clicks=0,
                                       style={'marginRight': '3px', 'padding': '3px 8px', 'fontSize': '10px', 
                                              'backgroundColor': '#e8f4e8', 'border': '1px solid #4a4', 'borderRadius': '3px', 'cursor': 'pointer'}),
                            html.Button("Hide labels", id='hide-all-labels-btn', n_clicks=0,
                                       style={'marginRight': '10px', 'padding': '3px 8px', 'fontSize': '10px',
                                              'backgroundColor': '#f4e8e8', 'border': '1px solid #a44', 'borderRadius': '3px', 'cursor': 'pointer'}),
                            html.Button("Show markers", id='show-all-markers-btn', n_clicks=0,
                                       style={'marginRight': '3px', 'padding': '3px 8px', 'fontSize': '10px', 
                                              'backgroundColor': '#e8e8f4', 'border': '1px solid #44a', 'borderRadius': '3px', 'cursor': 'pointer'}),
                            html.Button("Hide markers", id='hide-all-markers-btn', n_clicks=0,
                                       style={'marginRight': '15px', 'padding': '3px 8px', 'fontSize': '10px',
                                              'backgroundColor': '#f4e8f4', 'border': '1px solid #a4a', 'borderRadius': '3px', 'cursor': 'pointer'}),
                            html.Button("Show all printers", id='show-all-network-printers-btn', n_clicks=0,
                                       style={'marginRight': '3px', 'padding': '3px 8px', 'fontSize': '10px',
                                              'backgroundColor': '#e8f4e8', 'border': '1px solid #4a4', 'borderRadius': '3px', 'cursor': 'pointer'}),
                            html.Button("Hide all printers", id='hide-all-network-printers-btn', n_clicks=0,
                                       style={'padding': '3px 8px', 'fontSize': '10px',
                                              'backgroundColor': '#f4e8e8', 'border': '1px solid #a44', 'borderRadius': '3px', 'cursor': 'pointer'}),
                        ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),
                        
                        # Edge opacity slider - adjusted range for better visibility control
                        html.Div([
                            html.Label("Edge Opacity:", style={'fontSize': '11px', 'fontWeight': 'bold', 'marginRight': '5px'}),
                            dcc.Slider(
                                id='edge-opacity-slider',
                                min=0.0,
                                max=2.0,
                                step=0.1,
                                value=1.0,
                                marks={0: '0', 0.5: '0.5', 1.0: '1.0', 1.5: '1.5', 2.0: '2.0'},
                                tooltip={"placement": "bottom", "always_visible": False}
                            )
                        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'middle', 'marginLeft': '10px'}),
                        
                        # Node and label size sliders
                        html.Div([
                            html.Label("Node Size:", style={'fontSize': '11px', 'fontWeight': 'bold', 'marginRight': '5px'}),
                            dcc.Slider(
                                id='node-size-slider',
                                min=6,
                                max=36,
                                step=1,
                                value=12,
                                marks={i: str(i) for i in range(6, 37, 6)},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Label("Label Size:", style={'fontSize': '11px', 'fontWeight': 'bold', 'marginLeft': '10px', 'marginRight': '5px'}),
                            dcc.Slider(
                                id='label-size-slider',
                                min=6,
                                max=24,
                                step=1,
                                value=8,
                                marks={i: str(i) for i in range(6, 25, 6)},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'middle', 'marginLeft': '10px'}),
                        
                        # UMAP controls (collapsible)
                        html.Div([
                            html.Details([
                                html.Summary("UMAP Parameters", style={'cursor': 'pointer', 'fontWeight': 'bold', 'fontSize': '11px'}),
                                html.Div([
                                    html.Div([
                                        html.Label("n_neighbors:", style={'fontSize': '10px'}),
                                        dcc.Slider(
                                            id='umap-n-neighbors',
                                            min=5,
                                            max=100,
                                            step=5,
                                            value=50,
                                            marks={5: '5', 50: '50', 100: '100'},
                                            tooltip={"placement": "bottom", "always_visible": False}
                                        )
                                    ], style={'width': '35%', 'display': 'inline-block'}),
                                    
                                    html.Div([
                                        html.Label("min_dist:", style={'fontSize': '10px'}),
                                        dcc.Slider(
                                            id='umap-min-dist',
                                            min=0.0,
                                            max=1.0,
                                            step=0.05,
                                            value=0.5,
                                            marks={0: '0', 0.5: '0.5', 1: '1'},
                                            tooltip={"placement": "bottom", "always_visible": False}
                                        )
                                    ], style={'width': '35%', 'display': 'inline-block', 'marginLeft': '5%'}),
                                    
                                    html.Div([
                                        html.Button("Recalculate", id='recalculate-umap-btn', n_clicks=0,
                                                   style={'padding': '5px 10px', 'fontSize': '10px', 'backgroundColor': '#4a90d9', 
                                                          'color': 'white', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer'})
                                    ], style={'width': '15%', 'display': 'inline-block', 'marginLeft': '5%', 'verticalAlign': 'bottom'})
                                ], style={'padding': '5px 0'})
                            ], open=False)
                        ], style={'display': 'inline-block', 'width': '45%', 'marginLeft': '10px', 'verticalAlign': 'middle',
                                  'backgroundColor': '#f0f4f8', 'borderRadius': '5px', 'padding': '3px 10px'}),
                    ], style={'marginBottom': '5px', 'padding': '5px 10px', 'backgroundColor': '#f5f5f5', 'borderRadius': '5px', 'textAlign': 'center'}),
                    
                    dcc.Graph(id='network-graph', style={'height': '800px'})
                ], style={'width': '100%'})
            ]),
            
            # Statistics panel
            html.Div([
                html.H3("Analysis Statistics"),
                html.Div(id='stats-panel')
            ], style={'marginTop': 30, 'padding': 20, 'backgroundColor': '#f0f0f0'}),
            
            # Export section
            html.Div([
                html.H3("Export Options"),
                html.Div([
                    html.Button("Export Network Data (JSON)", id="export-network-btn", n_clicks=0, 
                               style={'marginRight': '10px', 'padding': '8px 15px'}),
                    html.Button("Export Similarity Matrix (CSV)", id="export-matrix-btn", n_clicks=0,
                               style={'marginRight': '10px', 'padding': '8px 15px'}),
                    html.Button("📄 Save as HTML", id="export-html-btn", n_clicks=0,
                               style={'marginRight': '10px', 'padding': '8px 15px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'borderRadius': '4px'}),
                ], style={'marginBottom': '10px'}),
                html.Div(id="export-status")
            ], style={'marginTop': 30, 'padding': 20, 'backgroundColor': '#f0f0f0'})
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        # Initialize letter filter and printer dropdown on load
        @self.app.callback(
            [Output('letter-filter', 'options'),
             Output('letter-filter', 'value'),
             Output('printer-filter-dropdown', 'options')],
            [Input('threshold-slider', 'value')],  # Just trigger on load
            prevent_initial_call=False
        )
        def init_filters(_):
            letter_options = [{'label': f' {l}', 'value': l} for l in self._all_letters]
            # Get unique printers
            unique_printers = sorted(set(self.impr_names))
            printer_options = [{'label': p, 'value': p} for p in unique_printers if p not in ['n. nan', 'm. missing']]
            return letter_options, self._all_letters, printer_options
        
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
            [Output('clicked-books-store', 'data'),
             Output('additional-books-dropdown', 'value')],
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
        
        # Show/Hide all labels on network graph
        @self.app.callback(
            Output('network-graph', 'figure', allow_duplicate=True),
            [Input('show-all-labels-btn', 'n_clicks'),
             Input('hide-all-labels-btn', 'n_clicks')],
            [State('network-graph', 'figure')],
            prevent_initial_call=True
        )
        def toggle_network_labels(show_clicks, hide_clicks, current_fig):
            if current_fig is None:
                return dash.no_update
            
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            show_labels = trigger_id == 'show-all-labels-btn'
            
            # Use Patch for efficient update
            patched_fig = dash.Patch()
            
            # Update text visibility for all node traces (skip first trace which is edges)
            for i in range(1, len(current_fig['data'])):
                current_mode = current_fig['data'][i].get('mode', 'markers')
                has_markers = 'markers' in current_mode
                if show_labels:
                    patched_fig['data'][i]['mode'] = 'markers+text' if has_markers else 'text'
                else:
                    patched_fig['data'][i]['mode'] = 'markers' if has_markers else 'none'
            
            return patched_fig
        
        # Show/Hide all markers on network graph
        @self.app.callback(
            Output('network-graph', 'figure', allow_duplicate=True),
            [Input('show-all-markers-btn', 'n_clicks'),
             Input('hide-all-markers-btn', 'n_clicks')],
            [State('network-graph', 'figure')],
            prevent_initial_call=True
        )
        def toggle_network_markers(show_clicks, hide_clicks, current_fig):
            if current_fig is None:
                return dash.no_update
            
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            show_markers = trigger_id == 'show-all-markers-btn'
            
            # Use Patch for efficient update
            patched_fig = dash.Patch()
            
            # Update marker visibility for all node traces (skip first trace which is edges)
            for i in range(1, len(current_fig['data'])):
                current_mode = current_fig['data'][i].get('mode', 'markers')
                has_text = 'text' in current_mode
                if show_markers:
                    patched_fig['data'][i]['mode'] = 'markers+text' if has_text else 'markers'
                else:
                    patched_fig['data'][i]['mode'] = 'text' if has_text else 'none'
            
            return patched_fig
        
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
            
            # Add book to existing selection
            current_books = list(dropdown_books) if dropdown_books else []
            if book_name not in current_books:
                current_books.append(book_name)
            
            return current_books, current_books
        
        # Callback to recalculate UMAP positions
        @self.app.callback(
            Output('umap-positions-store', 'data'),
            [Input('recalculate-umap-btn', 'n_clicks')],
            [State('umap-n-neighbors', 'value'),
             State('umap-min-dist', 'value')],
            prevent_initial_call=True
        )
        def recalculate_umap(n_clicks, n_neighbors, min_dist):
            if n_clicks is None or n_clicks == 0:
                return dash.no_update
            
            # Check cache first for these parameters
            cache_key = f'umap_positions_{n_neighbors}_{min_dist}'
            if cache_key in self._figures_cache:
                print(f"Using cached UMAP positions for n_neighbors={n_neighbors}, min_dist={min_dist}")
                cached = self._figures_cache[cache_key]
                return cached.tolist() if hasattr(cached, 'tolist') else cached
            
            # Compute new UMAP positions
            print(f"Computing UMAP positions for n_neighbors={n_neighbors}, min_dist={min_dist}...")
            distance_matrix = 1 - (self.w_rm + self.w_it) / 2  # Combined distance
            np.fill_diagonal(distance_matrix, 0)
            positions = self._compute_umap_positions(distance_matrix, n_neighbors, min_dist)
            
            if positions is not None:
                # Cache for future use (manage UMAP cache size)
                # Insert new entry
                self._figures_cache[cache_key] = positions

                # Trim only UMAP-related entries first to respect MAX_UMAP_CACHE_SIZE
                umap_keys = [k for k in self._figures_cache.keys() if k.startswith('umap_positions_')]
                # Keep default key if present
                default_key = 'umap_positions_50_0.5'
                # If too many UMAP entries, remove the oldest non-default ones
                if len(umap_keys) > self.MAX_UMAP_CACHE_SIZE:
                    removable = [k for k in umap_keys if k != default_key]
                    # Remove oldest until under limit
                    while len([k for k in self._figures_cache.keys() if k.startswith('umap_positions_')]) > self.MAX_UMAP_CACHE_SIZE and removable:
                        key_to_remove = removable.pop(0)
                        if key_to_remove in self._figures_cache:
                            del self._figures_cache[key_to_remove]

                # Global figure cache trim: keep total entries under MAX_FIGURE_CACHE_SIZE
                while len(self._figures_cache) > self.MAX_FIGURE_CACHE_SIZE:
                    # Remove oldest non-default entry
                    for k in list(self._figures_cache.keys()):
                        if k != default_key:
                            del self._figures_cache[k]
                            break

                return positions.tolist()  # Convert to list for JSON serialization
            return None
        
        # Separate callback for edge opacity - uses Patch for instant updates
        @self.app.callback(
            Output('network-graph', 'figure', allow_duplicate=True),
            [Input('edge-opacity-slider', 'value')],
            [State('network-graph', 'figure')],
            prevent_initial_call=True
        )
        def update_edge_opacity_only(edge_opacity, current_fig):
            if current_fig is None:
                return dash.no_update
            # Scale existing alpha values multiplicatively by `edge_opacity`.
            # Handles binned traces named like 'edges_{orig_alpha:.2f}'
            # traces with `line.colorscale`, numeric `line.color` arrays, and `rgba(...)` strings.
            if edge_opacity is None:
                return dash.no_update

            patched = dash.Patch()
            traces = current_fig.get('data', [])

            for i, trace in enumerate(traces):
                if not trace:
                    continue
                name = str(trace.get('name', '') or '')
                line = trace.get('line', {}) if isinstance(trace.get('line', {}), dict) else {}

                # Binned trace naming convention: 'edges_{orig_alpha:.2f}'
                if name.startswith('edges_'):
                    try:
                        orig_alpha = float(name.split('_')[-1])
                    except Exception:
                        orig_alpha = None
                    if orig_alpha is not None:
                        new_a = max(0.0, min(1.0, orig_alpha * float(edge_opacity)))
                        # Preserve existing RGB if possible
                        existing_color = line.get('color')
                        if isinstance(existing_color, str) and 'rgba' in existing_color:
                            parts = existing_color.replace('rgba(', '').replace(')', '').split(',')
                            if len(parts) >= 3:
                                r, g, b = parts[0].strip(), parts[1].strip(), parts[2].strip()
                                patched['data'][i]['line']['color'] = f'rgba({r},{g},{b},{new_a})'
                                continue
                        # Fallback color
                        patched['data'][i]['line']['color'] = f'rgba(100,100,100,{new_a})'
                        continue

                # If trace uses a colorscale for mapping numeric values -> update its opaque end
                if 'colorscale' in line:
                    try:
                        cs = line.get('colorscale') or []
                        # Try to extract RGB from last colorscale color
                        rgb = (100, 100, 100)
                        if cs and isinstance(cs, (list, tuple)) and len(cs) > 0:
                            last = cs[-1][1]
                            if isinstance(last, str) and 'rgba' in last:
                                parts = last.replace('rgba(', '').replace(')', '').split(',')
                                if len(parts) >= 3:
                                    rgb = (int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip()))
                        new_a = max(0.0, min(1.0, float(edge_opacity)))
                        patched['data'][i]['line']['colorscale'] = [[0.0, f'rgba({rgb[0]},{rgb[1]},{rgb[2]}, 0)'], [1.0, f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{new_a})']]
                    except Exception:
                        # safest fallback
                        patched['data'][i]['line']['colorscale'] = [[0.0, 'rgba(100,100,100,0)'], [1.0, f'rgba(100,100,100,{edge_opacity})']]
                    continue

                # If line.color is a numeric array (weights) -> set/update colorscale max alpha
                col = line.get('color', None)
                if isinstance(col, (list, tuple)):
                    patched['data'][i]['line']['colorscale'] = [[0.0, 'rgba(100,100,100,0)'], [1.0, f'rgba(100,100,100,{edge_opacity})']]
                    continue

                # If color is an rgba string, parse and scale its alpha
                if isinstance(col, str) and 'rgba' in col:
                    try:
                        parts = col.replace('rgba(', '').replace(')', '').split(',')
                        if len(parts) >= 3:
                            r, g, b = parts[0].strip(), parts[1].strip(), parts[2].strip()
                            orig_a = float(parts[3]) if len(parts) > 3 else 1.0
                            new_a = max(0.0, min(1.0, orig_a * float(edge_opacity)))
                            patched['data'][i]['line']['color'] = f'rgba({r},{g},{b},{new_a})'
                            continue
                    except Exception:
                        patched['data'][i]['line']['color'] = f'rgba(100,100,100,{edge_opacity})'
                        continue

                # Otherwise: skip non-edge traces or unknown formats
                # (we avoid touching node traces or unrelated items)

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
            if current_fig is None:
                return dash.no_update
            patched_fig = dash.Patch()
            # Node traces start at index 1 (0 is edges)
            for i in range(1, len(current_fig['data'])):
                trace = current_fig['data'][i]
                # Update marker size
                if 'marker' in trace:
                    # Unknown/missing printers are 5/6 size
                    if 'rgba(128, 128, 128' in str(trace.get('marker', {}).get('color', '')):
                        patched_fig['data'][i]['marker']['size'] = int(node_size * 5 / 6)
                    else:
                        patched_fig['data'][i]['marker']['size'] = node_size
                # Update label size
                if 'textfont' in trace:
                    # Unknown/missing printers are 5/6 size
                    if 'rgba(128, 128, 128' in str(trace.get('textfont', {}).get('color', '')):
                        patched_fig['data'][i]['textfont']['size'] = int(label_size * 5 / 6)
                    else:
                        patched_fig['data'][i]['textfont']['size'] = label_size
            return patched_fig

        # The original update_visualizations callback should NOT include node-size-slider or label-size-slider as Inputs anymore.
        @self.app.callback(
            [Output('network-graph', 'figure'),
             Output('similarity-heatmap', 'figure'),
             Output('stats-panel', 'children')],
            [Input('threshold-slider', 'data'),
             Input('layout-dropdown', 'data'),
             Input('font-type-dropdown', 'value'),
             Input('umap-positions-store', 'data')],
            [State('umap-n-neighbors', 'value'),
             State('umap-min-dist', 'value'),
             State('edge-opacity-slider', 'value'),
             State('node-size-slider', 'value'),
             State('label-size-slider', 'value')]
        )
        def update_visualizations(threshold, layout, font_type, umap_positions, n_neighbors, min_dist, edge_opacity, node_size, label_size):
            # ...existing code, but node_size and label_size come from State, not Input...
            threshold = threshold if threshold is not None else 0.1
            layout = layout if layout is not None else 'umap'
            if font_type == 'roman':
                weight_matrix = self.w_rm
                title_suffix = " (Roman)"
            elif font_type == 'italic':
                weight_matrix = self.w_it
                title_suffix = " (Italic)"
            else:
                weight_matrix = (self.w_rm + self.w_it) / 2
                title_suffix = " (Combined)"
            umap_pos_array = None
            if umap_positions is not None and len(umap_positions) > 0:
                umap_pos_array = np.array(umap_positions)
            elif layout == 'umap':
                cache_key = f'umap_positions_{int(n_neighbors)}_{float(min_dist)}'
                if cache_key in self._figures_cache:
                    umap_pos_array = self._figures_cache[cache_key]
            if font_type == 'roman':
                n1hat_matrix = self.n1hat_rm
            elif font_type == 'italic':
                n1hat_matrix = self.n1hat_it
            else:
                n1hat_matrix = self.n1hat_rm + self.n1hat_it
            network_fig = self._create_network_graph(
                weight_matrix, threshold, layout,
                n_neighbors=n_neighbors, min_dist=min_dist,
                umap_positions=umap_pos_array, edge_opacity=edge_opacity,
                n1hat_matrix=n1hat_matrix,
                marker_size=node_size, label_size=label_size
            )
            heatmap_fig = self._get_cached_heatmap(font_type)
            total_connections = np.sum(weight_matrix > threshold)
            avg_similarity = np.mean(weight_matrix[weight_matrix > threshold]) if total_connections > 0 else 0
            max_similarity = np.max(weight_matrix)
            connected_books = len(np.where(np.sum(weight_matrix > threshold, axis=0) > 0)[0])
            stats = [
                html.P(f"Total Books: {len(self.books)}"),
                html.P(f"Connected Books (threshold > {threshold}): {connected_books}"),
                html.P(f"Total Connections: {total_connections}"),
                html.P(f"Average Similarity (above threshold): {avg_similarity:.3f}"),
                html.P(f"Maximum Similarity: {max_similarity:.3f}"),
                html.P(f"Symbols Analyzed: {len(self.symbs)}")
            ]
            return network_fig, heatmap_fig, stats
        
        # Toggle shape overlays when legend items are clicked
        @self.app.callback(
            Output('similarity-heatmap', 'figure', allow_duplicate=True),
            [Input('similarity-heatmap', 'restyleData')],
            [State('similarity-heatmap', 'figure')],
            prevent_initial_call=True
        )
        def toggle_shape_overlays(restyle_data, current_fig):
            if restyle_data is None or current_fig is None:
                return dash.no_update
            
            # restyleData format: [{'visible': ['legendonly']}, [1]] means trace 1 was hidden
            # or [{'visible': [True]}, [1]] means trace 1 was shown
            try:
                visibility_change = restyle_data[0].get('visible', None)
                trace_indices = restyle_data[1] if len(restyle_data) > 1 else []
                
                if visibility_change is None or not trace_indices:
                    return dash.no_update
                
                # Get the trace that was toggled
                traces = current_fig.get('data', [])
                shapes = current_fig.get('layout', {}).get('shapes', [])
                
                if not shapes:
                    return dash.no_update
                
                # For each toggled trace, find its name (printer) and toggle corresponding shapes
                for i, trace_idx in enumerate(trace_indices):
                    if trace_idx < len(traces):
                        trace = traces[trace_idx]
                        printer_name = trace.get('name', '')
                        new_visible = visibility_change[i] if i < len(visibility_change) else visibility_change[0]
                        
                        # Check if visible - handle various formats Plotly might send
                        # 'legendonly' means hidden, anything else (True, true, 1) means visible
                        is_visible = new_visible not in ['legendonly', False, 'false', None]
                        
                        # Update shapes that belong to this printer
                        for shape in shapes:
                            shape_name = shape.get('name', '')
                            if shape_name == f"overlay_{printer_name}":
                                # Toggle visibility by setting opacity (preserve RGB)
                                current_fill = shape.get('fillcolor', '')
                                if 'rgba' in current_fill:
                                    parts = current_fill.replace('rgba(', '').replace(')', '').split(',')
                                    if len(parts) >= 3:
                                        if is_visible:
                                            # Restore alpha to 0.3
                                            shape['fillcolor'] = f"rgba({parts[0]},{parts[1]},{parts[2]}, 0.3)"
                                        else:
                                            # Set alpha to 0 (keep RGB for later restore)
                                            shape['fillcolor'] = f"rgba({parts[0]},{parts[1]},{parts[2]}, 0)"
                
                # Explicitly preserve layout settings to prevent margin shifts
                current_fig['layout']['uirevision'] = 'constant'
                current_fig['layout']['xaxis']['automargin'] = False
                current_fig['layout']['yaxis']['automargin'] = False
                
                return current_fig
                
            except Exception as e:
                print(f"Error toggling shapes: {e}")
                return dash.no_update
        
        # Show/hide all printers in network graph legend
        @self.app.callback(
            Output('network-graph', 'figure', allow_duplicate=True),
            [Input('show-all-network-printers-btn', 'n_clicks'),
             Input('hide-all-network-printers-btn', 'n_clicks')],
            [State('network-graph', 'figure')],
            prevent_initial_call=True
        )
        def toggle_all_network_printers(show_clicks, hide_clicks, current_fig):
            if current_fig is None:
                return dash.no_update
            
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            show_all = trigger_id == 'show-all-network-printers-btn'
            
            # Use dash.Patch for efficient update
            patched_fig = dash.Patch()
            
            # Update visibility for all printer traces (skip edge trace at index 0)
            if current_fig.get('data'):
                for i in range(1, len(current_fig['data'])):
                    patched_fig['data'][i]['visible'] = True if show_all else 'legendonly'
            
            return patched_fig
        
        # Show/hide all printer overlays at once
        @self.app.callback(
            Output('similarity-heatmap', 'figure', allow_duplicate=True),
            [Input('show-all-printers-btn', 'n_clicks'),
             Input('hide-all-printers-btn', 'n_clicks')],
            [State('similarity-heatmap', 'figure')],
            prevent_initial_call=True
        )
        def toggle_all_printers(show_clicks, hide_clicks, current_fig):
            if current_fig is None:
                return dash.no_update
            
            ctx = dash.callback_context
            if not ctx.triggered:
                return dash.no_update
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            show_all = trigger_id == 'show-all-printers-btn'
            
            try:
                traces = current_fig.get('data', [])
                shapes = current_fig.get('layout', {}).get('shapes', [])
                
                # Update all printer traces visibility (skip heatmap trace at index 0)
                for i, trace in enumerate(traces):
                    if i == 0:  # Skip heatmap
                        continue
                    trace['visible'] = True if show_all else 'legendonly'
                
                # Update all shape overlays
                for shape in shapes:
                    shape_name = shape.get('name', '')
                    if shape_name.startswith('overlay_'):
                        current_fill = shape.get('fillcolor', '')
                        if 'rgba' in current_fill:
                            parts = current_fill.replace('rgba(', '').replace(')', '').split(',')
                            if len(parts) >= 3:
                                if show_all:
                                    # Restore alpha to 0.3
                                    shape['fillcolor'] = f"rgba({parts[0]},{parts[1]},{parts[2]}, 0.3)"
                                else:
                                    # Set alpha to 0 (keep RGB for later restore)
                                    shape['fillcolor'] = f"rgba({parts[0]},{parts[1]},{parts[2]}, 0)"
                
                # Explicitly preserve layout settings to prevent margin shifts
                current_fig['layout']['uirevision'] = 'constant'
                current_fig['layout']['xaxis']['automargin'] = False
                current_fig['layout']['yaxis']['automargin'] = False
                
                return current_fig
                
            except Exception as e:
                print(f"Error toggling all printers: {e}")
                return dash.no_update
        
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
                
                # Check for reset (double-click to reset zoom)
                if 'xaxis.autorange' in relayout_data or 'yaxis.autorange' in relayout_data:
                    # Reset to default font size
                    current_fig['layout']['xaxis']['tickfont'] = {'size': default_size}
                    current_fig['layout']['yaxis']['tickfont'] = {'size': default_size}
                    current_fig['layout']['xaxis']['automargin'] = False
                    current_fig['layout']['yaxis']['automargin'] = False
                    return current_fig
                
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
                    font_size = 14
                elif visible_items <= 10:
                    font_size = 12
                elif visible_items <= 20:
                    font_size = 11
                elif visible_items <= 35:
                    font_size = 10
                else:
                    font_size = default_size
                
                # Update font sizes
                current_fig['layout']['xaxis']['tickfont'] = {'size': font_size}
                current_fig['layout']['yaxis']['tickfont'] = {'size': font_size}
                current_fig['layout']['xaxis']['automargin'] = False
                current_fig['layout']['yaxis']['automargin'] = False
                
                return current_fig
                
            except Exception as e:
                print(f"Error adjusting font size: {e}")
                return dash.no_update
        
        @self.app.callback(
            Output('export-status', 'children'),
            [Input('export-network-btn', 'n_clicks'),
             Input('export-matrix-btn', 'n_clicks'),
             Input('export-html-btn', 'n_clicks')],
            [State('font-type-dropdown', 'value'),
             State('similarity-heatmap', 'figure'),
             State('network-graph', 'figure')]
        )
        def export_data(network_clicks, matrix_clicks, html_clicks, font_type, heatmap_fig, network_fig):
            ctx = dash.callback_context
            if not ctx.triggered:
                return ""
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if button_id == 'export-network-btn':
                # Export network data as JSON
                if font_type == 'roman':
                    weight_matrix = self.w_rm
                elif font_type == 'italic':
                    weight_matrix = self.w_it
                else:
                    weight_matrix = (self.w_rm + self.w_it) / 2
                
                network_data = {
                    'books': self.books.tolist(),
                    'weights': weight_matrix.tolist(),
                    'imprinters': self.impr_names.tolist() if hasattr(self, 'impr_names') else [],
                    'export_time': timestamp,
                    'font_type': font_type
                }
                
                filename = f'network_data_{font_type}_{timestamp}.json'
                with open(filename, 'w') as f:
                    json.dump(network_data, f, indent=2)
                
                return html.P(f"Network data exported to {filename}", style={'color': 'green'})
            
            elif button_id == 'export-matrix-btn':
                # Export similarity matrix as CSV
                if font_type == 'roman':
                    weight_matrix = self.w_rm
                elif font_type == 'italic':
                    weight_matrix = self.w_it
                else:
                    weight_matrix = (self.w_rm + self.w_it) / 2
                
                df = pd.DataFrame(weight_matrix, index=self.books, columns=self.books)
                filename = f'similarity_matrix_{font_type}_{timestamp}.csv'
                df.to_csv(filename)
                
                return html.P(f"Similarity matrix exported to {filename}", style={'color': 'green'})
            
            elif button_id == 'export-html-btn':
                # Export figures as interactive HTML
                try:
                    import plotly.io as pio
                    
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
    </style>
</head>
<body>
    <h1>Book Typography Similarity Analysis</h1>
    <div class="info">
        <p><strong>Font Type:</strong> {font_type}</p>
        <p><strong>Number of Books:</strong> {len(self.books)}</p>
        <p><strong>Exported:</strong> {timestamp}</p>
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
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    
                    return html.P(f"✅ Dashboard exported to {filename} - open in any browser!", style={'color': 'green', 'fontWeight': 'bold'})
                    
                except Exception as e:
                    return html.P(f"❌ Export failed: {str(e)}", style={'color': 'red'})
            
            return ""
        
        # Callback to show letter comparisons - shows ALL cached images instantly
        @self.app.callback(
            Output('letter-comparison-panel', 'children'),
            [Input('font-type-dropdown', 'value'),
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
                    header_items.append(
                        html.Div([
                            html.P(f"{book}", style={'fontSize': '10px', 'fontWeight': 'bold', 'margin': '0', 'wordWrap': 'break-word'}),
                            html.P(f"{printer_name}", style={'fontSize': '9px', 'color': '#666', 'margin': '0'})
                        ], style={'display': 'inline-block', 'width': col_width, 'textAlign': 'center', 'verticalAlign': 'top'})
                    )
                
                comparison_content = [
                    html.Div(header_items, style={'marginBottom': '10px'}),
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
                            for book_images in all_book_images:
                                img_elements = []
                                for img_path, encoded in book_images:
                                    img_elements.append(
                                        html.Img(
                                            src=f"data:image/png;base64,{encoded}",
                                            style={'height': '70px', 'margin': '2px', 'border': '2px solid #ccc', 'borderRadius': '4px'}
                                        )
                                    )
                                if not img_elements:
                                    img_elements.append(html.Span("—", style={'color': '#999', 'fontSize': '24px'}))

                                book_columns.append(
                                    html.Div(img_elements, style={'width': col_width, 'display': 'inline-block', 'textAlign': 'center', 'verticalAlign': 'middle'})
                                )

                            comparison_content.append(
                                html.Div([
                                    html.H5(f"'{letter}'", style={'marginBottom': '5px', 'textAlign': 'center', 'fontSize': '14px', 'fontWeight': 'bold'}),
                                    html.Div(book_columns, style={'marginBottom': '8px', 'paddingBottom': '8px', 'borderBottom': '1px solid #eee'})
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