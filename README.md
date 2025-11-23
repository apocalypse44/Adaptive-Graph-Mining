# Graph-Based Recommendation System using PageRank

## Project Overview

This project implements a PageRank-based recommendation system for Amazon products using graph data structures. The system builds a product-product graph from co-purchase and co-view relationships and uses PageRank to rank products for recommendations.

## Features

- **Graph Construction**: Build product-product graphs from Amazon review data
- **PageRank Algorithm**: Compute product rankings using iterative PageRank
- **Adaptive Updates**: Incremental PageRank updates when new products/reviews are added
- **Interactive Visualization**: Visualize graph structure and top recommendations

## Project Structure

```
AGM/
├── data/                   # Data storage directory
├── src/
│   ├── data_loader.py      # Load and preprocess Amazon review data
│   ├── graph_builder.py    # Construct product-product graph
│   ├── pagerank.py         # PageRank implementation
│   ├── adaptive_update.py  # Incremental PageRank updates
│   ├── visualization.py    # Graph visualization with Plotly
│   └── main.py             # Main execution script
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Download Amazon Reviews dataset from [https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/)
2. Place data files in the `data/` directory
3. Run the main script:

```bash
python src/main.py
```

## Dataset

The project uses the Amazon Reviews 2023 dataset by McAuley Lab, UCSD.
Download from: https://amazon-reviews-2023.github.io/

