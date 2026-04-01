"""
Utility functions for loading and computing UMAP positions.

Keeps UMAP logic out of the main dashboard class.  The ``umap-learn``
package is imported lazily so it remains an optional dependency.
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def orient_positions(umap_positions):
    """Swap axes so the larger range is on x (better for wide layouts)."""
    umap_positions = np.asarray(umap_positions, dtype=np.float32)
    if umap_positions.size == 0 or umap_positions.ndim != 2 or umap_positions.shape[1] < 2:
        return umap_positions
    x_range = np.nanmax(umap_positions[:, 0]) - np.nanmin(umap_positions[:, 0])
    y_range = np.nanmax(umap_positions[:, 1]) - np.nanmin(umap_positions[:, 1])
    if y_range > x_range:
        return umap_positions[:, [1, 0]].copy()
    return umap_positions


def umap_filename(font_type, n_neighbors, min_dist, data_dir="./data"):
    """Return the expected .npy filename for a UMAP cache."""
    return os.path.join(data_dir, f"umap_{font_type}_{n_neighbors}_{min_dist}.npy")


def load_positions(font_type="combined", n_neighbors=50, min_dist=0.5,
                   compute_if_missing=False, distance_matrix=None,
                   random_state=42, data_dir="./data"):
    """Load cached UMAP positions, optionally computing if missing.

    Parameters
    ----------
    font_type : str
        One of 'roman', 'italic', 'combined'.
    n_neighbors, min_dist : float
        UMAP hyper-parameters (also used to locate the cache file).
    compute_if_missing : bool
        If True and the cache file doesn't exist, compute UMAP positions
        (requires ``umap-learn``).  A *distance_matrix* must be provided.
    distance_matrix : array-like or None
        Pre-computed distance matrix (symmetric, 0-diagonal).  Only needed
        when *compute_if_missing* is True.
    random_state : int
        Seed for reproducibility.
    data_dir : str
        Directory where UMAP cache files are stored.

    Returns
    -------
    np.ndarray of shape (n_books, 2) or None if unavailable.
    """
    path = umap_filename(font_type, n_neighbors, min_dist, data_dir=data_dir)

    if os.path.exists(path):
        positions = np.load(path)
        print(f"Loaded UMAP positions from {path}  (shape {positions.shape})")
        return orient_positions(positions)

    if not compute_if_missing:
        print(f"UMAP file {path} missing — computation deferred.")
        return None

    # --- compute ---------------------------------------------------------
    if distance_matrix is None:
        print("Cannot compute UMAP: no distance_matrix provided.")
        return None

    positions = compute_positions(distance_matrix, n_neighbors=n_neighbors,
                                  min_dist=min_dist, random_state=random_state)
    if positions is None:
        return None

    # Cache for next time
    np.save(path, positions)
    print(f"Saved UMAP positions to {path}")
    return orient_positions(positions)


def compute_positions(distance_matrix, n_neighbors=50, min_dist=0.5,
                      random_state=42):
    """Compute 2-D UMAP embedding from a pre-computed distance matrix.

    Requires ``umap-learn`` (optional dependency).  Returns None if the
    package is not installed or computation fails.
    """
    try:
        import umap  # optional dependency
    except ImportError:
        print("umap-learn is not installed.  Install it with:  pip install umap-learn")
        return None

    try:
        reducer = umap.UMAP(
            metric="precomputed",
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
        )
        dm = np.asarray(distance_matrix, dtype=np.float32)
        positions = reducer.fit_transform(dm)
        return positions
    except Exception as e:
        print(f"UMAP computation failed: {e}")
        return None
