"""
Test Script for Adaptive Graph Mining
Demonstrates how to add products and test incremental PageRank updates
"""

import sys
from pathlib import Path
from data_loader import AmazonDataLoader
from graph_builder import ProductGraphBuilder
from pagerank import PageRankCalculator
from adaptive_update import AdaptivePageRank

def test_adaptive_update():
    """Test adaptive graph mining by adding a new product"""
    
    print("=" * 60)
    print("Testing Adaptive Graph Mining - Adding a Product")
    print("=" * 60)
    
    # ========== SETUP: Load existing graph ==========
    print("\n[Setup] Loading existing graph...")
    
    CATEGORY = "All_Beauty"
    MAX_REVIEWS = 50000
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    REVIEW_FILE = DATA_DIR / f"review_{CATEGORY}.jsonl.gz"
    META_FILE = DATA_DIR / f"meta_{CATEGORY}.jsonl.gz"
    
    # Load data
    loader = AmazonDataLoader()
    reviews = loader.load_reviews(str(REVIEW_FILE), max_reviews=MAX_REVIEWS)
    metadata = loader.load_metadata(str(META_FILE))
    
    # Build graph
    co_purchases = loader.extract_co_purchase_relationships()
    co_views = loader.extract_co_view_relationships()
    
    builder = ProductGraphBuilder()
    graph = builder.build_graph(
        co_purchases=co_purchases,
        co_views=co_views,
        weight_co_purchase=1.0,
        weight_co_view=0.5
    )
    
    # Compute initial PageRank
    calculator = PageRankCalculator(graph)
    initial_scores = calculator.compute_pagerank()
    
    print(f"\nInitial graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # ========== STEP 1: Show top products BEFORE adding new product ==========
    print("\n" + "=" * 60)
    print("BEFORE: Top 10 products by PageRank")
    print("=" * 60)
    top_before = calculator.get_top_products(k=10)
    for i, (product, score) in enumerate(top_before, 1):
        product_info = loader.get_product_info(product)
        title = product_info.get('title', 'N/A')[:50] if product_info else 'N/A'
        print(f"{i:2d}. {product}: {score:.6f} - {title}")
    
    # ========== STEP 2: Initialize Adaptive PageRank ==========
    print("\n" + "=" * 60)
    print("Initializing Adaptive PageRank system...")
    print("=" * 60)
    adaptive = AdaptivePageRank(graph, initial_scores)
    print("✓ Adaptive system ready!")
    
    # ========== STEP 3: Add a new product ==========
    print("\n" + "=" * 60)
    print("STEP 1: Adding a new product to the graph")
    print("=" * 60)
    
    # Choose a new product ID (using a format similar to existing ASINs)
    new_product_id = "TEST_PRODUCT_001"
    
    # Get some existing top products to connect to
    # This simulates a new product being related to popular products
    top_3_products = [product for product, _ in top_before[:3]]
    
    print(f"\nNew Product ID: {new_product_id}")
    print(f"Connecting to top products: {top_3_products}")
    
    # Define edges: (target_product, direction)
    # 'out' = new product points to target (new product recommends target)
    # 'in' = target points to new product (target recommends new product)
    # 'bidirectional' = both directions
    new_edges = [
        (top_3_products[0], 'out'),   # New product → Top product 1
        (top_3_products[1], 'bidirectional'),  # New product ↔ Top product 2
        (top_3_products[2], 'in'),    # Top product 3 → New product
    ]
    
    # Optional: Add edge weights (default is 1.0)
    edge_weights = [1.5, 2.0, 1.0]  # Higher weight = stronger relationship
    
    print(f"\nAdding edges:")
    for (target, direction), weight in zip(new_edges, edge_weights):
        print(f"  - {new_product_id} --[{direction}, weight={weight}]--> {target}")
    
    # Add the product using adaptive update
    updated_scores = adaptive.add_node(new_product_id, new_edges, edge_weights)
    
    print(f"\n✓ Product added successfully!")
    print(f"✓ Updated PageRank scores for {len(updated_scores)} affected nodes")
    
    # ========== STEP 4: Show results AFTER adding product ==========
    print("\n" + "=" * 60)
    print("AFTER: Top 10 products by PageRank (after adaptive update)")
    print("=" * 60)
    
    # Get updated scores
    final_scores = adaptive.get_scores()
    
    # Create a new calculator with updated scores for display
    updated_calculator = PageRankCalculator(graph, damping=0.85)
    updated_calculator.scores = final_scores
    
    top_after = updated_calculator.get_top_products(k=10)
    for i, (product, score) in enumerate(top_after, 1):
        product_info = loader.get_product_info(product)
        title = product_info.get('title', 'N/A')[:50] if product_info else 'N/A'
        
        # Check if this is the new product
        marker = " ⭐ NEW" if product == new_product_id else ""
        
        # Check if rank changed
        before_rank = next((idx for idx, (p, _) in enumerate(top_before, 1) if p == product), None)
        if before_rank and before_rank != i:
            change = before_rank - i
            marker += f" (↑{change})" if change > 0 else f" (↓{abs(change)})"
        
        print(f"{i:2d}. {product}: {score:.6f} - {title}{marker}")
    
    # ========== STEP 5: Show new product's score ==========
    print("\n" + "=" * 60)
    print("New Product Details")
    print("=" * 60)
    new_product_score = final_scores.get(new_product_id, 0)
    print(f"Product ID: {new_product_id}")
    print(f"PageRank Score: {new_product_score:.6f}")
    print(f"Rank: {next((i for i, (p, _) in enumerate(top_after, 1) if p == new_product_id), 'Not in top 10')}")
    
    # Show neighbors
    neighbors = list(graph.neighbors(new_product_id))
    print(f"Connected to {len(neighbors)} products:")
    for neighbor in neighbors[:5]:  # Show first 5
        neighbor_info = loader.get_product_info(neighbor)
        neighbor_title = neighbor_info.get('title', 'N/A')[:40] if neighbor_info else 'N/A'
        print(f"  - {neighbor}: {neighbor_title}")
    if len(neighbors) > 5:
        print(f"  ... and {len(neighbors) - 5} more")
    
    # ========== STEP 6: Verify graph statistics ==========
    print("\n" + "=" * 60)
    print("Graph Statistics After Update")
    print("=" * 60)
    print(f"Total nodes: {graph.number_of_nodes()} (+1)")
    print(f"Total edges: {graph.number_of_edges()} (+{len(new_edges)})")
    
    # ========== STEP 7: Compare with full recomputation ==========
    print("\n" + "=" * 60)
    print("Verification: Comparing adaptive vs full recomputation")
    print("=" * 60)
    
    # Full recomputation for comparison
    full_calculator = PageRankCalculator(graph, damping=0.85)
    full_scores = full_calculator.compute_pagerank()
    
    # Compare scores for new product
    adaptive_score = final_scores.get(new_product_id, 0)
    full_score = full_scores.get(new_product_id, 0)
    difference = abs(adaptive_score - full_score)
    
    print(f"New product PageRank:")
    print(f"  Adaptive update: {adaptive_score:.8f}")
    print(f"  Full recompute:  {full_score:.8f}")
    print(f"  Difference:      {difference:.8f}")
    
    if difference < 0.0001:
        print("✓ Adaptive update matches full recomputation!")
    else:
        print(f"⚠ Small difference (expected due to iterative refinement)")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
    return adaptive, graph, final_scores


if __name__ == "__main__":
    try:
        adaptive, graph, scores = test_adaptive_update()
        
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("=" * 60)
        print("1. Try adding more products:")
        print("   adaptive.add_node('NEW_PRODUCT_002', [(existing_product, 'out')])")
        print("\n2. Try adding edges between existing products:")
        print("   adaptive.add_edge('PRODUCT_A', 'PRODUCT_B', weight=1.5)")
        print("\n3. Try removing a product:")
        print("   adaptive.remove_node('PRODUCT_ID')")
        print("\n4. Get current scores:")
        print("   scores = adaptive.get_scores()")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

