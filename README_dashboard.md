# Book Typography Similarity Dashboard

An interactive web dashboard for visualizing and analyzing book typography similarities based on your a-contrario analysis.

## Features

### 🌐 Interactive Network Visualization
- Dynamic network graphs showing book similarities
- Adjustable similarity thresholds
- Multiple layout algorithms (spring, circular, random)
- Hover effects with book details and imprinter information
- Color-coded nodes based on total similarity scores

### 🔥 Similarity Matrix Heatmap
- Interactive heatmaps of pairwise book similarities
- Hierarchical ordering for pattern discovery
- Separate views for Roman, Italic, and Combined analyses
- Hover details showing exact similarity values

### 📊 Real-time Analytics
- Live statistics updates based on filter settings
- Connection counts and similarity metrics
- Connected vs isolated book analysis

### 💾 Export Capabilities
- Export network data as JSON
- Export similarity matrices as CSV
- Timestamped exports for reproducibility

## Quick Start

### 1. Setup (First time only)
```bash
python setup_dashboard.py
```

### 2. Export Your Analysis Data
In your `acontrario.ipynb` notebook, run the new export cell at the bottom:
```python
# This will save your weight matrices
np.save('../../w_rm_matrix.npy', w_rm)
np.save('../../w_it_matrix.npy', w_it)
```

### 3. Launch Dashboard
```bash
python launch_dashboard.py
```

### 4. Open Browser
Navigate to `http://localhost:8050`

## Dashboard Components

### Control Panel
- **Similarity Threshold**: Filter connections by minimum similarity score
- **Network Layout**: Choose visualization algorithm
- **Font Type**: Switch between Roman, Italic, or Combined analysis

### Network Graph (Left Panel)
- Nodes represent books
- Edges represent similarities above threshold
- Node size/color indicates total connectivity
- Hover for book and imprinter details

### Similarity Matrix (Right Panel)
- Heatmap of all pairwise similarities
- Hierarchically ordered for pattern detection
- Color scale from low (blue) to high (red) similarity

### Statistics Panel
- Total and connected book counts
- Connection statistics
- Average and maximum similarities

### Export Section
- Download network data and matrices
- JSON format for networks, CSV for matrices
- Includes metadata and timestamps

## File Structure

```
├── book_similarity_dashboard.py    # Main dashboard class
├── launch_dashboard.py            # Launcher script
├── setup_dashboard.py             # Setup and installation
├── dashboard_requirements.txt     # Python dependencies
├── README.md                      # This file
└── extract_chars/
    └── acontrario.ipynb           # Your analysis notebook
```

## Data Requirements

The dashboard expects these data files:
- `./report_new/names_rounds.npy` - Book names (roman)
- `./report_new/names_italics.npy` - Book names (italic)
- `./report_new/impr_names_rounds.npy` - Imprinter names
- `./w_rm_matrix.npy` - Roman weight matrix (exported from notebook)
- `./w_it_matrix.npy` - Italic weight matrix (exported from notebook)

## Customization

### Modify Network Layouts
Edit the `_create_network_graph` method in `book_similarity_dashboard.py`:
```python
# Add custom layout
elif layout_type == 'custom':
    pos = your_custom_layout_function(G)
```

### Add New Statistics
Extend the statistics calculation in the `update_visualizations` callback:
```python
# Add new metrics
custom_metric = calculate_your_metric(weight_matrix)
stats.append(html.P(f"Custom Metric: {custom_metric:.3f}"))
```

### Modify Color Schemes
Change color palettes in the plotting functions:
```python
# Network node colors
colorscale='Viridis'  # Change to 'Plasma', 'Cividis', etc.

# Heatmap colors  
colorscale='RdBu'     # Change to 'RdYlBu', 'Spectral', etc.
```

## Troubleshooting

### Dashboard won't start
1. Check Python package installation: `pip install -r dashboard_requirements.txt`
2. Ensure port 8050 is available
3. Check for error messages in the console

### No data showing
1. Verify data files exist and are accessible
2. Run the export cell in your notebook
3. Check file paths match your directory structure

### Network graph empty
1. Lower the similarity threshold
2. Check if your weight matrices have values > 0
3. Verify matrix dimensions match number of books

### Performance issues
1. Reduce number of books by filtering isolated ones
2. Increase similarity threshold to reduce connections
3. Use circular layout for faster rendering

## Advanced Usage

### Running on Different Port
```python
dashboard.run_server(debug=False, port=8051)
```

### Deploying Online
The dashboard can be deployed to:
- **Heroku**: Follow Plotly Dash deployment guide
- **Railway**: Easy Python app deployment
- **DigitalOcean**: App platform deployment

### Adding Authentication
For shared deployments, add Dash authentication:
```python
import dash_auth

auth = dash_auth.BasicAuth(app, {'username': 'password'})
```

## Dependencies

- dash==2.16.1
- plotly==5.18.0
- pandas==2.1.4
- numpy==1.24.3
- networkx==3.2.1
- scipy==1.11.4
- dash-bootstrap-components==1.5.0

## License

This dashboard is designed for academic research use. Please cite your original analysis when sharing results.

## Support

For issues or questions about the dashboard:
1. Check the troubleshooting section above
2. Verify your data format matches expectations
3. Test with sample data first

Enjoy exploring your book typography similarities! 📚✨