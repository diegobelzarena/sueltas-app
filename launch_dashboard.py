"""
Dashboard Launcher for Book Similarity Analysis
"""

import os
import yaml
from book_similarity_dashboard import BookSimilarityDashboard
from werkzeug.middleware.proxy_fix import ProxyFix

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------
CONFIG_PATH = os.environ.get("SUELTAS_CONFIG", "./config.yaml")

def load_config(path=CONFIG_PATH):
    """Load config.yaml and return the dict. Missing file → empty dict (all defaults)."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    print(f"Warning: config file {path} not found — using built-in defaults.")
    return {}

cfg = load_config()


def load_real_data_into_dashboard(dashboard):
    print("Loading analysis data...")
    dashboard.set_data()
    print("Data loaded into dashboard")


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
dashboard = BookSimilarityDashboard(
    requests_pathname_prefix=prefix if prefix else "",
    config=cfg,
)
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
