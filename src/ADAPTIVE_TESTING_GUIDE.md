# Adaptive Graph Mining Testing Guide

This guide explains how to test the adaptive graph mining functionality by adding products to your graph.

## Quick Start

### Option 1: Run the Test Script (Recommended)

The easiest way to test adaptive updates is to run the provided test script:

```bash
cd src
python test_adaptive.py
```

This script will:
1. Load your existing graph
2. Show top products before adding a new product
3. Add a new test product with connections to top products
4. Show updated PageRank scores after adaptive update
5. Verify the results by comparing with full recomputation

### Option 2: Interactive Testing in Python

You can also test interactively in a Python shell or script:

```python
from main import *  # After running main.py, you'll have the adaptive object

# Add a new product
new_product_id = "TEST_PRODUCT_001"
new_edges = [
    ("B085BB7B1M", "out"),      # New product → existing product
    ("B012Q9NGE4", "bidirectional"),  # Bidirectional connection
    ("B0BM4GX6TT", "in"),       # Existing product → new product
]
edge_weights = [1.5, 2.0, 1.0]

# Add the product
updated_scores = adaptive.add_node(new_product_id, new_edges, edge_weights)

# Check the new product's score
print(f"New product score: {adaptive.get_scores()[new_product_id]}")
```

## Understanding Edge Directions

When adding a product, you specify edges as `(target_product, direction)`:

- **`'out'`**: New product → Target product
  - Meaning: "People who buy the new product also buy the target product"
  - Example: `("B085BB7B1M", "out")`

- **`'in'`**: Target product → New product
  - Meaning: "People who buy the target product also buy the new product"
  - Example: `("B085BB7B1M", "in")`

- **`'bidirectional'`**: New product ↔ Target product
  - Meaning: Both products are frequently bought together
  - Example: `("B085BB7B1M", "bidirectional")`

## Step-by-Step Manual Testing

### Step 1: Run the Main Pipeline

First, ensure your main pipeline has run successfully:

```bash
python main.py
```

This creates the initial graph and PageRank scores.

### Step 2: Access the Adaptive System

At the end of `main.py`, an `AdaptivePageRank` object is created. You can:

**Option A**: Modify `main.py` to keep the adaptive object accessible:

```python
# At the end of main.py, add:
if __name__ == "__main__":
    adaptive, graph, loader = main()  # Return these objects
    # Now you can use them interactively
```

**Option B**: Create a separate test script (recommended - see `test_adaptive.py`)

### Step 3: Add a Product

```python
# Example: Add a new beauty product
new_product = "NEW_BEAUTY_001"

# Connect it to top 3 products
top_products = ["B085BB7B1M", "B012Q9NGE4", "B0BM4GX6TT"]

edges = [
    (top_products[0], "out"),           # New product recommends top product 1
    (top_products[1], "bidirectional"), # Strong co-purchase relationship
    (top_products[2], "in"),            # Top product 3 recommends new product
]

weights = [1.5, 2.0, 1.0]  # Higher weight = stronger relationship

# Add the product
updated = adaptive.add_node(new_product, edges, weights)
print(f"Updated {len(updated)} nodes")
```

### Step 4: Check Results

```python
# Get all scores
all_scores = adaptive.get_scores()

# Check new product's rank
new_score = all_scores[new_product]
print(f"New product PageRank: {new_score}")

# Get top products
from pagerank import PageRankCalculator
calc = PageRankCalculator(graph)
calc.scores = all_scores
top_10 = calc.get_top_products(k=10)

# Check if new product is in top 10
for rank, (product, score) in enumerate(top_10, 1):
    marker = " ⭐ NEW" if product == new_product else ""
    print(f"{rank}. {product}: {score:.6f}{marker}")
```

### Step 5: Verify with Full Recompute

To verify the adaptive update is correct:

```python
# Full recomputation
from pagerank import PageRankCalculator
full_calc = PageRankCalculator(graph)
full_scores = full_calc.compute_pagerank()

# Compare
adaptive_score = adaptive.get_scores()[new_product]
full_score = full_scores[new_product]
print(f"Adaptive: {adaptive_score:.8f}")
print(f"Full:     {full_score:.8f}")
print(f"Diff:     {abs(adaptive_score - full_score):.8f}")
```

## Other Operations

### Add an Edge Between Existing Products

```python
# Add edge from product A to product B
updated = adaptive.add_edge("PRODUCT_A", "PRODUCT_B", weight=1.5)
```

### Remove a Product

```python
# Remove a product and update scores
updated = adaptive.remove_node("PRODUCT_ID")
```

### Get Current State

```python
# Get all current PageRank scores
scores = adaptive.get_scores()

# Get graph statistics
print(f"Nodes: {graph.number_of_nodes()}")
print(f"Edges: {graph.number_of_edges()}")
```

## Expected Results

When you add a product:

1. **Graph size increases**: +1 node, +N edges (where N = number of connections)
2. **PageRank scores update**: Only affected nodes' scores change
3. **New product gets a score**: Based on its connections
4. **Top products may shift**: If the new product is highly connected

## Troubleshooting

### Product ID not found
- Make sure you're using valid ASINs from your dataset
- Check that the product exists in the graph: `"PRODUCT_ID" in graph`

### No score change
- The new product might not be well-connected
- Try connecting to more popular products
- Use bidirectional edges for stronger relationships

### Large difference from full recompute
- This is normal for the first few iterations
- The adaptive update uses iterative refinement
- Differences should be < 0.0001 after convergence

## Example: Complete Test Session

```python
# 1. Load and setup (from main.py)
from data_loader import AmazonDataLoader
from graph_builder import ProductGraphBuilder
from pagerank import PageRankCalculator
from adaptive_update import AdaptivePageRank

loader = AmazonDataLoader()
# ... load data and build graph ...

calculator = PageRankCalculator(graph)
initial_scores = calculator.compute_pagerank()

# 2. Initialize adaptive system
adaptive = AdaptivePageRank(graph, initial_scores)

# 3. Show before
print("Before:")
top_before = calculator.get_top_products(k=5)
for p, s in top_before:
    print(f"  {p}: {s:.6f}")

# 4. Add product
new_product = "TEST_001"
edges = [(top_before[0][0], "bidirectional")]
updated = adaptive.add_node(new_product, edges)

# 5. Show after
print("\nAfter:")
final_scores = adaptive.get_scores()
calc_after = PageRankCalculator(graph)
calc_after.scores = final_scores
top_after = calc_after.get_top_products(k=5)
for p, s in top_after:
    marker = " ⭐" if p == new_product else ""
    print(f"  {p}: {s:.6f}{marker}")

# 6. Check new product
print(f"\nNew product score: {final_scores[new_product]:.6f}")
```

## Next Steps

1. Try adding multiple products in sequence
2. Test removing products
3. Test adding edges between existing products
4. Compare performance: adaptive update vs full recompute
5. Visualize the changes in the network graph

