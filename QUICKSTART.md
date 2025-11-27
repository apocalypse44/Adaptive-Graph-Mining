# Quick Start Guide

## 🚀 Getting Started

This guide will help you set up and run the Graph-Based Recommendation System.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Option 1: Demo Mode (No Data Required)

The project includes a demo mode that creates a sample graph for testing:

```bash
python src/main.py
```

This will:
- Create a sample product graph
- Compute PageRank scores
- Generate visualizations
- Demonstrate adaptive updates

### Option 2: With Real Amazon Data

1. **Download the dataset:**
   - Visit: https://amazon-reviews-2023.github.io/
   - Download a category (e.g., "All_Beauty" for smaller dataset)
   - Download both:
     - Review file: `review_All_Beauty.jsonl.gz`
     - Metadata file: `meta_All_Beauty.jsonl.gz`

2. **Place files in data directory:**
   ```
   data/
   ├── review_All_Beauty.jsonl.gz
   └── meta_All_Beauty.jsonl.gz
   ```

3. **Update file paths in `src/main.py`:**
   ```python
   REVIEW_FILE = DATA_DIR / "review_All_Beauty.jsonl.gz"
   META_FILE = DATA_DIR / "meta_All_Beauty.jsonl.gz"
   ```

4. **Run the script:**
   ```bash
   python src/main.py
   ```

## Project Structure

```
AGM/
├── data/                   # Place your dataset files here
├── src/
│   ├── data_loader.py      # Loads Amazon review data
│   ├── graph_builder.py    # Builds product-product graph
│   ├── pagerank.py         # PageRank implementation
│   ├── adaptive_update.py  # Incremental PageRank updates
│   ├── visualization.py    # Graph visualization
│   └── main.py             # Main execution script
├── requirements.txt
├── README.md
└── QUICKSTART.md
```

## Key Features

### 1. Data Loading (`data_loader.py`)
- Loads reviews and metadata from JSONL.gz files
- Extracts co-purchase relationships (products reviewed by same user)
- Extracts co-view relationships (from "bought_together" metadata)

### 2. Graph Construction (`graph_builder.py`)
- Builds directed product-product graph
- Supports weighted edges
- Can add/remove products dynamically

### 3. PageRank (`pagerank.py`)
- Standard PageRank algorithm
- Custom iterative implementation
- Get top products and recommendations

### 4. Adaptive Updates (`adaptive_update.py`)
- Incremental PageRank updates when graph changes
- Efficient recomputation for affected nodes only
- Supports adding/removing nodes and edges

### 5. Visualization (`visualization.py`)
- Interactive network graphs with Plotly
- PageRank score distributions
- Top products bar charts
- Highlight recommendations for selected products

## Usage Examples

### Basic Usage

```python
from data_loader import AmazonDataLoader
from graph_builder import ProductGraphBuilder
from pagerank import PageRankCalculator

# Load data
loader = AmazonDataLoader()
reviews = loader.load_reviews("data/reviews.jsonl.gz", max_reviews=10000)
metadata = loader.load_metadata("data/meta.jsonl.gz")

# Build graph
co_purchases = loader.extract_co_purchase_relationships()
builder = ProductGraphBuilder()
graph = builder.build_graph(co_purchases=co_purchases)

# Compute PageRank
calculator = PageRankCalculator(graph)
scores = calculator.compute_pagerank()

# Get recommendations
recommendations = calculator.get_recommendations("B001234567", k=10)
```

### Adaptive Updates

```python
from adaptive_update import AdaptivePageRank

# Initialize with existing scores
adaptive = AdaptivePageRank(graph, scores)

# Add new product
new_edges = [("P001", "out"), ("P002", "out")]
updated = adaptive.add_node("P999", new_edges)

# Get updated scores
new_scores = adaptive.get_scores()
```

### Visualization

```python
from visualization import GraphVisualizer

visualizer = GraphVisualizer(graph, pagerank_scores)
fig = visualizer.create_network_plot(selected_node="B001234567", top_k=10)
fig.write_html("output.html")
```

## Output Files

After running, you'll get:
- `visualization_network.html` - Interactive network graph
- `visualization_top_products.html` - Top products bar chart

## Tips

1. **Start Small**: Use `max_reviews` parameter to limit data size for testing
2. **Category Selection**: Start with smaller categories like "All_Beauty" or "Baby_Products"
3. **Performance**: For large graphs, visualization limits to 500 nodes by default
4. **Memory**: Large datasets may require significant RAM

## Troubleshooting

**Issue**: "File not found" error
- **Solution**: Ensure data files are in the `data/` directory with correct names

**Issue**: Out of memory
- **Solution**: Reduce `max_reviews` parameter or use a smaller category

**Issue**: Visualization is slow
- **Solution**: Reduce `max_nodes` parameter in visualization functions

## Next Steps

1. Experiment with different damping factors (default: 0.85)
2. Try different edge weights for co-purchase vs co-view
3. Implement custom recommendation strategies
4. Add evaluation metrics (precision, recall, etc.)

## Support

For issues or questions, refer to:
- Project PDF: `Group 2_Adaptive Graph Mining-1.pdf`
- Dataset docs: https://amazon-reviews-2023.github.io/

