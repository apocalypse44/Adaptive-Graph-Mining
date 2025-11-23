"""
Main Execution Script
Orchestrates the graph-based recommendation system
"""

import os
import sys
from pathlib import Path
from data_loader import AmazonDataLoader
from graph_builder import ProductGraphBuilder
from pagerank import PageRankCalculator
from adaptive_update import AdaptivePageRank
from visualization import GraphVisualizer


def main():
    """Main execution function"""
    
    print("=" * 60)
    print("Graph-Based Recommendation System using PageRank")
    print("=" * 60)
    
    # Configuration
    DATA_DIR = Path("data")
    REVIEW_FILE = DATA_DIR / "reviews.jsonl.gz"  # Update with actual file path
    META_FILE = DATA_DIR / "meta.jsonl.gz"      # Update with actual file path
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"\nCreating data directory: {DATA_DIR}")
        DATA_DIR.mkdir(exist_ok=True)
        print("\nPlease download Amazon Reviews dataset and place files in data/ directory")
        print("Download from: https://amazon-reviews-2023.github.io/")
        return
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    loader = AmazonDataLoader()
    
    # Check if files exist
    if not REVIEW_FILE.exists():
        print(f"\nReview file not found: {REVIEW_FILE}")
        print("Please download and place the review file in the data directory")
        print("\nFor testing, you can use a small sample. Creating sample workflow...")
        demo_mode = True
    else:
        demo_mode = False
        # Load actual data (limit for demo)
        print(f"Loading reviews from {REVIEW_FILE}...")
        reviews = loader.load_reviews(str(REVIEW_FILE), max_reviews=50000)
        
        if META_FILE.exists():
            print(f"Loading metadata from {META_FILE}...")
            metadata = loader.load_metadata(str(META_FILE))
        else:
            print(f"Metadata file not found: {META_FILE}")
            metadata = {}
    
    if demo_mode:
        print("\n" + "=" * 60)
        print("DEMO MODE: Creating sample graph for demonstration")
        print("=" * 60)
        
        # Create sample graph for demonstration
        from src.graph_builder import ProductGraphBuilder
        from src.pagerank import PageRankCalculator
        from src.visualization import GraphVisualizer
        import networkx as nx
        
        # Create a sample graph
        sample_graph = nx.DiGraph()
        products = [f"P{i:03d}" for i in range(20)]
        
        # Add nodes
        for product in products:
            sample_graph.add_node(product)
        
        # Add edges (simulating co-purchase relationships)
        import random
        random.seed(42)
        for i in range(30):
            u = random.choice(products)
            v = random.choice(products)
            if u != v:
                sample_graph.add_edge(u, v, weight=random.uniform(0.5, 2.0))
        
        print(f"\nSample graph created: {sample_graph.number_of_nodes()} nodes, "
              f"{sample_graph.number_of_edges()} edges")
        
        # Step 2: Compute PageRank
        print("\n[Step 2] Computing PageRank...")
        calculator = PageRankCalculator(sample_graph)
        pagerank_scores = calculator.compute_pagerank()
        
        print("\nTop 10 products by PageRank:")
        top_products = calculator.get_top_products(k=10)
        for i, (product, score) in enumerate(top_products, 1):
            print(f"{i:2d}. {product}: {score:.6f}")
        
        # Step 3: Visualization
        print("\n[Step 3] Creating visualizations...")
        visualizer = GraphVisualizer(sample_graph, pagerank_scores)
        
        # Create network plot
        fig = visualizer.create_network_plot(selected_node="P001", top_k=5)
        output_file = "visualization_network.html"
        fig.write_html(output_file)
        print(f"Network visualization saved to: {output_file}")
        
        # Create top products plot
        fig2 = visualizer.plot_top_products(k=10)
        output_file2 = "visualization_top_products.html"
        fig2.write_html(output_file2)
        print(f"Top products visualization saved to: {output_file2}")
        
        # Step 4: Adaptive updates demo
        print("\n[Step 4] Demonstrating adaptive updates...")
        adaptive = AdaptivePageRank(sample_graph, pagerank_scores)
        
        # Add a new product
        new_product = "P999"
        new_edges = [("P001", "out"), ("P002", "out"), ("P003", "in")]
        updated_scores = adaptive.add_node(new_product, new_edges)
        print(f"Added product {new_product} with {len(new_edges)} connections")
        print(f"Updated scores for {len(updated_scores)} affected nodes")
        
        # Show updated top products
        updated_pagerank = adaptive.get_scores()
        updated_calculator = PageRankCalculator(sample_graph, damping=0.85)
        updated_calculator.scores = updated_pagerank
        print("\nTop 10 products after adaptive update:")
        top_after = updated_calculator.get_top_products(k=10)
        for i, (product, score) in enumerate(top_after, 1):
            print(f"{i:2d}. {product}: {score:.6f}")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        print("\nTo use with real data:")
        print("1. Download Amazon Reviews dataset from https://amazon-reviews-2023.github.io/")
        print("2. Place review and metadata files in the data/ directory")
        print("3. Update file paths in main.py")
        print("4. Run the script again")
        
        return
    
    # Step 2: Build graph
    print("\n[Step 2] Building product graph...")
    co_purchases = loader.extract_co_purchase_relationships()
    co_views = loader.extract_co_view_relationships()
    
    builder = ProductGraphBuilder()
    graph = builder.build_graph(
        co_purchases=co_purchases,
        co_views=co_views,
        weight_co_purchase=1.0,
        weight_co_view=0.5
    )
    
    stats = builder.get_graph_stats()
    print("\nGraph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Step 3: Compute PageRank
    print("\n[Step 3] Computing PageRank...")
    calculator = PageRankCalculator(graph)
    pagerank_scores = calculator.compute_pagerank()
    
    print("\nTop 10 products by PageRank:")
    top_products = calculator.get_top_products(k=10)
    for i, (product, score) in enumerate(top_products, 1):
        product_info = loader.get_product_info(product)
        title = product_info.get('title', 'N/A')[:50] if product_info else 'N/A'
        print(f"{i:2d}. {product}: {score:.6f} - {title}")
    
    # Step 4: Visualization
    print("\n[Step 4] Creating visualizations...")
    visualizer = GraphVisualizer(graph, pagerank_scores)
    
    # Create network plot
    fig = visualizer.create_network_plot(selected_node=top_products[0][0] if top_products else None)
    output_file = "visualization_network.html"
    fig.write_html(output_file)
    print(f"Network visualization saved to: {output_file}")
    
    # Create top products plot
    fig2 = visualizer.plot_top_products(k=20)
    output_file2 = "visualization_top_products.html"
    fig2.write_html(output_file2)
    print(f"Top products visualization saved to: {output_file2}")
    
    # Step 5: Adaptive updates
    print("\n[Step 5] Setting up adaptive update system...")
    adaptive = AdaptivePageRank(graph, pagerank_scores)
    print("Adaptive update system ready!")
    print("You can now add/remove products and edges incrementally.")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

