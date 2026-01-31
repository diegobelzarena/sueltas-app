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
    # Downcast large numeric arrays to float16 before saving to reduce cache size
    try:
        w_rm_save = w_rm.astype(np.float16) if isinstance(w_rm, np.ndarray) else w_rm
        w_it_save = w_it.astype(np.float16) if isinstance(w_it, np.ndarray) else w_it
        n1hat_rm_save = n1hat_rm.astype(np.float16) if isinstance(n1hat_rm, np.ndarray) else n1hat_rm
        n1hat_it_save = n1hat_it.astype(np.float16) if isinstance(n1hat_it, np.ndarray) else n1hat_it
    except Exception:
        w_rm_save, w_it_save, n1hat_rm_save, n1hat_it_save = w_rm, w_it, n1hat_rm, n1hat_it

    cache_data = {
        'books': books,
        'w_rm': w_rm_save,
        'w_it': w_it_save,
        'impr_names': impr_names,
        'symbs': symbs,
        'n1hat_rm': n1hat_rm_save,
        'n1hat_it': n1hat_it_save,
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
        version = cache_data.get('version', '1.0')
        if version >= '1.2':
            # New format: 'data' contains essential data
            figures_cache = cache_data.get('data')
        else:
            # Old format: 'figures' contains full figures
            figures_cache = cache_data.get('figures')
        print(f"✓ Loaded figures from cache (version: {version}, saved: {cache_data.get('cached_at', 'unknown')})")
        return figures_cache
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
        # Downcast large matrices to float16 to reduce memory footprint immediately
        try:
            import gc
            if isinstance(w_rm, np.ndarray):
                w_rm = w_rm.astype(np.float16)
            if isinstance(w_it, np.ndarray):
                w_it = w_it.astype(np.float16)
            if isinstance(n1hat_rm, np.ndarray):
                n1hat_rm = n1hat_rm.astype(np.float16)
            if isinstance(n1hat_it, np.ndarray):
                n1hat_it = n1hat_it.astype(np.float16)
            gc.collect()
        except Exception:
            pass

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
        print("Loading from .npy files... (using memmap where possible)")
        # Load the exported weight matrices and data using memory-mapping to reduce peak RAM
        w_rm = np.load('./w_rm_matrix.npy', mmap_mode='r')
        w_it = np.load('./w_it_matrix.npy', mmap_mode='r')
        n1hat_it = np.load('./n1hat_it_matrix.npy', mmap_mode='r')
        n1hat_rm = np.load('./n1hat_rm_matrix.npy', mmap_mode='r')
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

def load_real_data_into_dashboard(dashboard):
    print("Lazy-loading analysis data...")

    result = load_analysis_results(force_reload=False)
    books, w_rm, w_it, impr_names, symbs, n1hat_rm, n1hat_it, cached_order, figures_cache = result

    if books is None:
        print("Falling back to sample data")
        books, w_rm, w_it, impr_names, symbs, n1hat_rm, n1hat_it, cached_order, figures_cache = create_sample_data()

    dashboard.set_data(
        books, w_rm, w_it, impr_names, symbs, n1hat_rm, n1hat_it,
        cached_order=cached_order, figures_cache=figures_cache
    )

    print("✓ Data loaded into dashboard")



def main():
    """Main launcher function"""
    
    dashboard = BookSimilarityDashboard()

    app = dashboard.app
    server = app.server

    with server.app_context():
        load_real_data_into_dashboard(dashboard)

    
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)
    

if __name__ == "__main__":
    main()