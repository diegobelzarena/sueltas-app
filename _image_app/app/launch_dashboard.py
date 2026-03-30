"""
Dashboard Launcher for Book Similarity Analysis
This script integrates with your acontrario.ipynb notebook data
"""

import os
import pickle
from datetime import datetime
from book_similarity_dashboard import BookSimilarityDashboard
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.wrappers import Response
from werkzeug.middleware.dispatcher import DispatcherMiddleware
# from memory_profile import get_object_memory, get_dashboard_memory

FIGURES_CACHE_FILE = './dashboard_figures_cache.pkl'

def save_figures_cache(figures_dict):
    """Save pre-computed figures to cache"""
    cache_data = {
        'figures': figures_dict,
        'cached_at': datetime.now().isoformat(),
        'version': '1.0'
    }
    with open(FIGURES_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"Saved figures cache to: {FIGURES_CACHE_FILE}")

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
        print(f"Loaded figures from cache (version: {version}, saved: {cache_data.get('cached_at', 'unknown')})")
        return figures_cache
    except Exception as e:
        print(f"WARNING: Figures cache load failed: {e}")
        return None
    
def load_real_data_into_dashboard(dashboard):
    print("Lazy-loading analysis data...")
        
    dashboard.set_data()

    print("✓ Data loaded into dashboard")
    
    # # Profile memory usage
    # print("\nProfiling memory usage...")
    # get_dashboard_memory(dashboard)
    # get_object_memory()


# Allow serving the app under a subpath. Configure via env var:
#   REQUESTS_PATHNAME_PREFIX=/investigacion/grupos/gti/sueltas/
# When set, pass the prefix into the dashboard constructor so Dash sets
# the prefix values at creation time (which is required).
prefix = os.getenv("REQUESTS_PATHNAME_PREFIX", "").strip()
if prefix:
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    if not prefix.endswith("/"):
        prefix = prefix + "/"

# Instantiate dashboard with prefix (if any)
dashboard = BookSimilarityDashboard(requests_pathname_prefix=prefix if prefix else "")
app = dashboard.app
flask_server = app.server

# Trust proxy headers
flask_server.wsgi_app = ProxyFix(flask_server.wsgi_app, x_prefix=1)

# DO NOT wrap with DispatcherMiddleware
server = flask_server

with flask_server.app_context():
    load_real_data_into_dashboard(dashboard)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)
