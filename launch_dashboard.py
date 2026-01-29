"""
Dashboard Launcher for Book Similarity Analysis
This script integrates with your acontrario.ipynb notebook data
"""

import sys
import os
import numpy as np
import pickle
from datetime import datetime
from book_similarity_dashboard import BookSimilarityDashboard

CACHE_FILE = './dashboard_data_cache.pkl'
FIGURES_CACHE_FILE = './dashboard_figures_cache.pkl'

def save_to_cache(books, w_rm, w_it, impr_names, symbs, n1hat_rm=None, n1hat_it=None, idxs_order=None):
    """Save all dashboard data to a single cache file"""
    cache_data = {
        'books': books,
        'w_rm': w_rm,
        'w_it': w_it,
        'impr_names': impr_names,
        'symbs': symbs,
        'n1hat_rm': n1hat_rm,
        'n1hat_it': n1hat_it,
        'idxs_order': idxs_order,
        'cached_at': datetime.now().isoformat(),
        'version': '2.0'
    }
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"✓ Saved data to cache: {CACHE_FILE}")

def save_figures_cache(figures_dict):
    """Save pre-computed figures to cache"""
    cache_data = {
        'figures': figures_dict,
        'cached_at': datetime.now().isoformat(),
        'version': '1.0'
    }
    with open(FIGURES_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"✓ Saved figures to cache: {FIGURES_CACHE_FILE}")

def load_figures_cache():
    """Load pre-computed figures from cache"""
    if not os.path.exists(FIGURES_CACHE_FILE):
        return None
    try:
        with open(FIGURES_CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"✓ Loaded figures from cache (saved: {cache_data.get('cached_at', 'unknown')})")
        return cache_data.get('figures')
    except Exception as e:
        print(f"⚠ Figures cache load failed: {e}")
        return None

def load_from_cache():
    """Load dashboard data from cache file"""
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
        
        books = cache_data['books']
        w_rm = cache_data['w_rm']
        w_it = cache_data['w_it']
        impr_names = cache_data.get('impr_names')
        symbs = cache_data.get('symbs')
        n1hat_rm = cache_data.get('n1hat_rm')
        n1hat_it = cache_data.get('n1hat_it')
        idxs_order = cache_data.get('idxs_order')
        cached_at = cache_data.get('cached_at', 'unknown')
        
        print(f"✓ Loaded data from cache (saved: {cached_at})")
        print(f"  - {len(books)} books, matrix shape: {w_rm.shape}")
        if idxs_order is not None:
            print(f"  - Hierarchical ordering: cached")
        
        return books, w_rm, w_it, impr_names, symbs, n1hat_rm, n1hat_it, idxs_order
        
    except Exception as e:
        print(f"⚠ Cache load failed: {e}")
        return None

def load_analysis_results(force_reload=False):
    """
    Load the results from your analysis.
    Uses cache for fast loading, falls back to .npy files if needed.
    Returns: books, w_rm, w_it, impr_names, symbs, n1hat_rm, n1hat_it, idxs_order, figures_cache
    """
    idxs_order = None
    figures_cache = None
    
    # Try cache first (unless force reload)
    if not force_reload:
        cached = load_from_cache()
        if cached:
            books, w_rm, w_it, impr_names, symbs, n1hat_rm, n1hat_it, idxs_order = cached
            figures_cache = load_figures_cache()
            return books, w_rm, w_it, impr_names, symbs, n1hat_rm, n1hat_it, idxs_order, figures_cache
    
    try:
        print("Loading from .npy files...")
        # Load the exported weight matrices and data
        w_rm = np.load('./w_rm_matrix.npy')
        w_it = np.load('./w_it_matrix.npy')
        n1hat_it = np.load('./n1hat_it_matrix.npy')
        n1hat_rm = np.load('./n1hat_rm_matrix.npy')
        books = np.load('./books_dashboard.npy')
        print("✓ Loaded exported data from notebook analysis")
        
        # Load additional data if available
        try:
            impr_names = np.load('./impr_names_dashboard.npy')
            print("✓ Loaded imprinter names")
        except:
            print("⚠ Imprinter names not found, using defaults")
            impr_names = None
            
        try:
            symbs = np.load('./symbs_dashboard.npy')
            print("✓ Loaded symbols data")
        except:
            print("⚠ Symbols not found, using defaults")
            symbs = None
            
        print(f"\nData Summary:")
        print(f"- Books: {len(books)}")
        print(f"- Roman matrix shape: {w_rm.shape}")
        print(f"- Italic matrix shape: {w_it.shape}")
        print(f"- Roman n1hat shape: {n1hat_rm.shape}")
        print(f"- Italic n1hat shape: {n1hat_it.shape}")
        print(f"- Roman similarity range: [{np.min(w_rm):.4f}, {np.max(w_rm):.4f}]")
        print(f"- Italic similarity range: [{np.min(w_it):.4f}, {np.max(w_it):.4f}]")
        print(f"- Roman connections (>0.01): {np.sum(w_rm > 0.01)}")
        print(f"- Italic connections (>0.01): {np.sum(w_it > 0.01)}")
        print(f"- Roman n1hat range: [{np.min(n1hat_rm):.4f}, {np.max(n1hat_rm):.4f}]")
        print(f"- Italic n1hat range: [{np.min(n1hat_it):.4f}, {np.max(n1hat_it):.4f}]")
        
        # Note: idxs_order will be computed by dashboard and saved later
        # Save to cache for next time (without ordering yet)
        save_to_cache(books, w_rm, w_it, impr_names, symbs, n1hat_rm=n1hat_rm, n1hat_it=n1hat_it, idxs_order=None)
        
        return books, w_rm, w_it, impr_names, symbs, n1hat_rm, n1hat_it, None, None
            
    except FileNotFoundError as e:
        print(f"✗ Data files not found: {e}")
        print("\nTo export your data from the notebook:")
        print("1. Run your acontrario.ipynb analysis completely")
        print("2. Execute the export cell at the bottom")
        print("3. Run this dashboard again")
        return None, None, None, None, None, None, None, None, None
        
    except Exception as e:
        print(f"✗ Error loading analysis results: {e}")
        return None, None, None, None, None, None, None, None, None
def create_sample_data():
    """Create sample data for testing the dashboard"""
    print("Creating sample data for testing...")
    
    # Generate sample book names
    books = np.array([f"Book_{i:03d}" for i in range(50)])
    
    # Create sample weight matrices with some structure
    n_books = len(books)
    
    # Roman weights - create clusters
    w_rm = np.random.exponential(0.1, (n_books, n_books))
    
    # Add some clustering structure
    cluster_size = 10
    for i in range(0, n_books, cluster_size):
        end_idx = min(i + cluster_size, n_books)
        w_rm[i:end_idx, i:end_idx] += 0.3
    
    # Make symmetric and zero diagonal
    w_rm = (w_rm + w_rm.T) / 2
    np.fill_diagonal(w_rm, 0)
    
    # Italic weights - similar structure but different values
    w_it = np.random.exponential(0.1, (n_books, n_books))
    for i in range(0, n_books, cluster_size):
        end_idx = min(i + cluster_size, n_books)
        w_it[i:end_idx, i:end_idx] += 0.2
    
    w_it = (w_it + w_it.T) / 2
    np.fill_diagonal(w_it, 0)
    
    # Create n1hat matrices
    n1hat_rm = np.where(w_rm > 0, w_rm * 20, 0).astype(int)
    n1hat_it = np.where(w_it > 0, w_it * 20, 0).astype(int)
    
    # Sample imprinter names
    imprinters = ["A. Smith", "B. Jones", "C. Wilson", "D. Brown", "E. Davis"]
    impr_names = np.array([imprinters[i % len(imprinters)] for i in range(n_books)])
    
    # Sample symbols
    symbs = np.array(list('abcdefghijklmnopqrstuvwxyz'))
    
    return books, w_rm, w_it, impr_names, symbs, n1hat_rm, n1hat_it, None, None  # No cached order/figures for sample

def main():
    """Main launcher function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Book Typography Similarity Dashboard')
    parser.add_argument('--host', default='127.0.0.1', 
                       help='Host address. Use 0.0.0.0 to allow remote access')
    parser.add_argument('--port', type=int, default=8050, 
                       help='Port number (default: 8050)')
    parser.add_argument('--remote', action='store_true',
                       help='Enable remote access (shortcut for --host 0.0.0.0)')
    parser.add_argument('--threaded', action='store_true',
                       help='Enable multi-threaded mode for better multi-user performance')
    parser.add_argument('--reload', action='store_true',
                       help='Force reload from .npy files (ignore cache)')
    args = parser.parse_args()
    
    # For Render: always use 0.0.0.0 and get port from environment
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', args.port))
    
    print("=" * 60)
    print("Book Typography Similarity Dashboard")
    print("=" * 60)
    
    # Try to load real data first
    result = load_analysis_results(force_reload=args.reload)
    books, w_rm, w_it, impr_names, symbs, n1hat_rm, n1hat_it, cached_order, figures_cache = result
    
    if books is None:
        print("\nReal data not available. Using sample data for Render deployment.")
        books, w_rm, w_it, impr_names, symbs, n1hat_rm, n1hat_it, cached_order, figures_cache = create_sample_data()
    
    # Create and launch dashboard
    print(f"\nCreating dashboard with {len(books)} books...")
    print(f"Roman connections: {np.sum(w_rm > 0.1)}")
    print(f"Italic connections: {np.sum(w_it > 0.1)}")
    
    dashboard = BookSimilarityDashboard(
        books, w_rm, w_it, impr_names, symbs, n1hat_rm, n1hat_it,
        cached_order=cached_order, figures_cache=figures_cache
    )
    
    # Save computed data back to cache if it was freshly calculated
    if cached_order is None and dashboard.idxs_order is not None:
        print("Saving computed ordering to cache...")
        save_to_cache(books, w_rm, w_it, impr_names, symbs, n1hat_rm, n1hat_it, dashboard.idxs_order)
    
    # Save figures cache if it was freshly computed
    if figures_cache is None and hasattr(dashboard, '_figures_cache') and dashboard._figures_cache:
        print("Saving computed figures to cache...")
        save_figures_cache(dashboard._figures_cache)
    
    # Get local IP for remote access info
    local_ip = "localhost"
    if host == '0.0.0.0':
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except:
            local_ip = "your-ip-address"
    
    print("\n" + "=" * 60)
    print(f"Dashboard starting at http://{host}:{port}")
    print("If running on Render, your public URL will be provided by the Render dashboard.")
    print("=" * 60)
    
    
    server = dashboard.app.server  # Flask server
    try:
        dashboard.app.run_server(debug=False, port=port, host=host)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"\nError running dashboard: {e}")

if __name__ == "__main__":
    main()