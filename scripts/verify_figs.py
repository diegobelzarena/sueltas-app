import sys
import os
# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from book_similarity_dashboard import BookSimilarityDashboard
import numpy as np

b = BookSimilarityDashboard()
print('Calling set_data()')
b.set_data()
print('Loading combined UMAP (cached)')
umap_combined = b._load_umap_positions(font_type='combined', compute_if_missing=False)
print('combined loaded:', umap_combined is not None)
if umap_combined is None:
    umap_combined = np.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'umap_combined_50_0.5.npy'))

print('Building network figure...')
fig_net = b._create_network_graph(umap_combined, edge_opacity=1.0, font_type='combined')
print('network traces:', len(fig_net.data))

print('Building heatmap figure...')
fig_heat = b._create_heatmap()
print('heatmap traces:', len(fig_heat.data))
print('heatmap z shape:', np.array(fig_heat.data[0].z).shape)

print('Done')
