"""
Test Script for Adaptive Graph Mining
Demonstrates how to add products and test incremental PageRank updates
WITH BEFORE/AFTER VISUALIZATIONS
"""

import sys
from pathlib import Path
from data_loader import AmazonDataLoader
from graph_builder import ProductGraphBuilder
from pagerank import PageRankCalculator
from adaptive_update import AdaptivePageRank
from visualization import GraphVisualizer

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
    reviews = loader.load_reviews(str(REVIEW_FILE), max_reviews=MAX_REVIEWS, sample_ratio=0.05)
    metadata = loader.load_metadata(str(META_FILE))
    
    # Build graph
    co_purchases = loader.extract_co_purchase_relationships()
    co_reviews = loader.extract_co_review_relationships()
    builder = ProductGraphBuilder()
    graph = builder.build_graph(
        co_purchases=co_purchases,
        co_reviews=co_reviews,
        weight_co_purchase=1.0,
        weight_co_review=0.8
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
    
    # ========== VISUALIZE BEFORE STATE ==========
    print("\n" + "=" * 60)
    print("Creating BEFORE visualizations...")
    print("=" * 60)
    
    visualizer_before = GraphVisualizer(graph, initial_scores)
    selected_product = top_before[0][0] if top_before else None
    
    # Network plot - BEFORE
    fig_network_before = visualizer_before.create_network_plot(
        selected_node=selected_product, 
        top_k=5,
        max_nodes=100
    )
    fig_network_before.write_html("adaptive_before_network.html")
    print("✓ Saved: adaptive_before_network.html")
    
    # Top products - BEFORE
    fig_top_before = visualizer_before.plot_top_products(k=15, highlight_product=selected_product)
    fig_top_before.write_html("adaptive_before_top_products.html")
    print("✓ Saved: adaptive_before_top_products.html")
    
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
    
    # Choose a new product ID
    new_product_id = "TEST_PRODUCT_001"
    
    # Get some existing top products to connect to
    top_3_products = [product for product, _ in top_before[:3]]
    
    print(f"\nNew Product ID: {new_product_id}")
    print(f"Connecting to top products: {top_3_products}")
    
    # Define edges: (target_product, direction)
    # Note: adaptive_update.py expects 2-tuples, not 3-tuples with edge_type
    new_edges = [
        (top_3_products[0], 'out'),
        (top_3_products[1], 'bidirectional'),
        (top_3_products[2], 'in'),
    ]
    
    # Edge weights
    edge_weights = [1.5, 2.0, 1.0]
    
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
    
    # ========== VISUALIZE AFTER STATE ==========
    print("\n" + "=" * 60)
    print("Creating AFTER visualizations...")
    print("=" * 60)
    
    visualizer_after = GraphVisualizer(graph, final_scores)
    
    # Network plot - AFTER (showing new product)
    fig_network_after = visualizer_after.create_network_plot(
        selected_node=new_product_id, 
        top_k=5,
        max_nodes=100
    )
    fig_network_after.write_html("adaptive_after_network.html")
    print("✓ Saved: adaptive_after_network.html")
    
    # Top products - AFTER
    fig_top_after = visualizer_after.plot_top_products(k=15, highlight_product=new_product_id)
    fig_top_after.write_html("adaptive_after_top_products.html")
    print("✓ Saved: adaptive_after_top_products.html")
    
    # Comparison chart for new product
    fig_comparison = visualizer_after.plot_comparison_chart(new_product_id, top_k=8)
    fig_comparison.write_html("adaptive_new_product_comparison.html")
    print("✓ Saved: adaptive_new_product_comparison.html")
    
    # ========== CREATE SIDE-BY-SIDE COMPARISON ==========
    print("\n" + "=" * 60)
    print("Creating side-by-side comparison visualization...")
    print("=" * 60)
    
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Create side-by-side comparison of top products
    fig_comparison_side = make_subplots(
        rows=1, cols=2,
        subplot_titles=('BEFORE: Top 10 Products', 'AFTER: Top 10 Products'),
        horizontal_spacing=0.15
    )
    
    # BEFORE data
    products_before = [p[0][:12] for p in top_before[:10]]
    scores_before = [p[1] for p in top_before[:10]]
    
    fig_comparison_side.add_trace(
        go.Bar(
            x=products_before,
            y=scores_before,
            marker_color='#4169E1',
            name='Before',
            text=[f'{s:.4f}' for s in scores_before],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # AFTER data
    products_after = [p[0][:12] for p in top_after[:10]]
    scores_after = [p[1] for p in top_after[:10]]
    colors_after = ['red' if p[0] == new_product_id else '#32CD32' for p in top_after[:10]]
    
    fig_comparison_side.add_trace(
        go.Bar(
            x=products_after,
            y=scores_after,
            marker_color=colors_after,
            name='After',
            text=[f'{s:.4f}' for s in scores_after],
            textposition='outside'
        ),
        row=1, col=2
    )
    
    fig_comparison_side.update_xaxes(tickangle=-45)
    fig_comparison_side.update_layout(
        title_text=f"PageRank Comparison: Before vs After Adding {new_product_id}",
        showlegend=False,
        height=600
    )
    
    fig_comparison_side.write_html("adaptive_before_after_comparison.html")
    print("✓ Saved: adaptive_before_after_comparison.html")
    
    # ========== STEP 5: Show new product's score ==========
    print("\n" + "=" * 60)
    print("New Product Details")
    print("=" * 60)
    new_product_score = final_scores.get(new_product_id, 0)
    print(f"Product ID: {new_product_id}")
    print(f"PageRank Score: {new_product_score:.6f}")
    
    rank_position = next((i for i, (p, _) in enumerate(top_after, 1) if p == new_product_id), None)
    if rank_position:
        print(f"Rank: #{rank_position}")
    else:
        all_ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        rank_position = next((i for i, (p, _) in enumerate(all_ranked, 1) if p == new_product_id), None)
        print(f"Rank: #{rank_position} (not in top 10)")
    
    # Show neighbors
    neighbors = list(graph.neighbors(new_product_id))
    print(f"Connected to {len(neighbors)} products:")
    for neighbor in neighbors[:5]:
        neighbor_info = loader.get_product_info(neighbor)
        neighbor_title = neighbor_info.get('title', 'N/A')[:40] if neighbor_info else 'N/A'
        print(f"  - {neighbor}: {neighbor_title}")
    if len(neighbors) > 5:
        print(f"  ... and {len(neighbors) - 5} more")
    
    # ========== STEP 6: Show score changes for affected products ==========
    print("\n" + "=" * 60)
    print("PageRank Score Changes for Connected Products")
    print("=" * 60)
    
    for product in top_3_products:
        before_score = initial_scores.get(product, 0)
        after_score = final_scores.get(product, 0)
        change = after_score - before_score
        change_pct = (change / before_score * 100) if before_score > 0 else 0
        
        product_info = loader.get_product_info(product)
        title = product_info.get('title', 'N/A')[:40] if product_info else 'N/A'
        
        print(f"\n{product}: {title}")
        print(f"  Before: {before_score:.6f}")
        print(f"  After:  {after_score:.6f}")
        print(f"  Change: {change:+.6f} ({change_pct:+.2f}%)")
    
    # ========== STEP 7: Verify graph statistics ==========
    print("\n" + "=" * 60)
    print("Graph Statistics After Update")
    print("=" * 60)
    print(f"Total nodes: {graph.number_of_nodes()} (+1)")
    print(f"Total edges: {graph.number_of_edges()} (+{len(new_edges)})")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
    print("\n📊 Generated Visualizations:")
    print("  1. adaptive_before_network.html - Network BEFORE adding product")
    print("  2. adaptive_after_network.html - Network AFTER adding product")
    print("  3. adaptive_before_top_products.html - Top products BEFORE")
    print("  4. adaptive_after_top_products.html - Top products AFTER")
    print("  5. adaptive_new_product_comparison.html - New product vs its neighbors")
    print("  6. adaptive_before_after_comparison.html - Side-by-side comparison")
    
    return adaptive, graph, final_scores


if __name__ == "__main__":
    try:
        adaptive, graph, scores = test_adaptive_update()
        
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("=" * 60)
        print("1. Open the HTML files in your browser to see visualizations")
        print("2. Try adding more products:")
        print("   adaptive.add_node('NEW_PRODUCT_002', [('existing_product', 'out', 'co_purchase')])")
        print("\n3. Try adding edges between existing products:")
        print("   adaptive.add_edge('PRODUCT_A', 'PRODUCT_B', weight=1.5)")
        print("\n4. Try removing a product:")
        print("   adaptive.remove_node('PRODUCT_ID')")
        print("\n5. Get current scores:")
        print("   scores = adaptive.get_scores()")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)