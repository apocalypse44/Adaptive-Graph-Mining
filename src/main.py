import os
import sys
import networkx as nx
import random
from pathlib import Path
from data_loader import AmazonDataLoader
from graph_builder import ProductGraphBuilder
from pagerank import PageRankCalculator
from adaptive_update import AdaptivePageRank
from visualization import GraphVisualizer


def main():    
    print("=" * 60)
    print("Graph-Based Recommendation System using PageRank")
    print("=" * 60)
    CATEGORY = "All_Beauty"  # Change this to your chosen category
    
    MAX_REVIEWS = 50000  # Start with 50K, increase as needed

    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    # Auto-generate file paths based on category
    if CATEGORY:
        REVIEW_FILE = DATA_DIR / f"review_{CATEGORY}.jsonl.gz"
        META_FILE = DATA_DIR / f"meta_{CATEGORY}.jsonl.gz"
        print(f"\n Selected Category: {CATEGORY}")
        print(f"   Review file: {REVIEW_FILE.name}")
        print(f"   Metadata file: {META_FILE.name}")
    else:
        # Fallback to generic names
        REVIEW_FILE = DATA_DIR / "reviews.jsonl.gz"
        META_FILE = DATA_DIR / "meta.jsonl.gz"
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"\nCreating data directory: {DATA_DIR}")
        DATA_DIR.mkdir(exist_ok=True)
        print("\nPlease download Amazon Reviews dataset and place files in data/ directory")
        print("Download from: https://amazon-reviews-2023.github.io/")
        return
    
    print("\n[Step 1] Loading data...")
    loader = AmazonDataLoader()
    
    # Resolve paths to absolute paths for reliable checking
    REVIEW_FILE = REVIEW_FILE.resolve()
    META_FILE = META_FILE.resolve()
    
    print(f"Review file: {REVIEW_FILE}")
    print(f"File exists: {REVIEW_FILE.exists()}")
    
    # Check if files exist
    if not REVIEW_FILE.exists():
        print(f"\nReview file not found: {REVIEW_FILE}")
        print("Please download and place the review file in the data directory")
        print("\nFor testing, you can use a small sample. Creating sample workflow...")
        demo_mode = True
    else:
        demo_mode = False
        # Load actual data (limit for faster processing)
        print(f"Loading reviews from {REVIEW_FILE}...")
        reviews = loader.load_reviews(str(REVIEW_FILE), max_reviews=MAX_REVIEWS, sample_ratio=0.05)
        
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
        
        # Create a sample multi-edge graph
        sample_graph = nx.MultiDiGraph()
        products = [f"P{i:03d}" for i in range(20)]
        
        # Add nodes
        for product in products:
            sample_graph.add_node(product)
        
        # Add edges with different types
        import random
        random.seed(42)
        
        # Add co-purchase edges
        for i in range(20):
            u = random.choice(products)
            v = random.choice(products)
            if u != v:
                sample_graph.add_edge(u, v, weight=random.uniform(0.8, 2.0), edge_type='co_purchase')
        
        # Add co-review edges
        for i in range(15):
            u = random.choice(products)
            v = random.choice(products)
            if u != v:
                sample_graph.add_edge(u, v, weight=random.uniform(0.5, 1.5), edge_type='co_review')
        
        co_purchase_count = sum(1 for u, v, d in sample_graph.edges(data=True) if d.get('edge_type') == 'co_purchase')
        co_review_count = sum(1 for u, v, d in sample_graph.edges(data=True) if d.get('edge_type') == 'co_review')
        
        print(f"\nSample graph created: {sample_graph.number_of_nodes()} nodes, "
              f"{sample_graph.number_of_edges()} edges")
        print(f"  Co-purchase edges: {co_purchase_count}")
        print(f"  Co-review edges: {co_review_count}")
        
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
        
        selected_product = "P001"
        
        # Create network plot
        fig = visualizer.create_network_plot(selected_node=selected_product, top_k=5)
        output_file = "visualization_network.html"
        fig.write_html(output_file)
        print(f"Network visualization saved to: {output_file}")
        
        # Create top products plot with highlighting
        fig2 = visualizer.plot_top_products(k=10, highlight_product=selected_product)
        output_file2 = "visualization_top_products.html"
        fig2.write_html(output_file2)
        print(f"Top products visualization saved to: {output_file2}")
        
        # Create comparison chart
        fig3 = visualizer.plot_comparison_chart(selected_product, top_k=5)
        output_file3 = "visualization_comparison.html"
        fig3.write_html(output_file3)
        print(f"Comparison chart saved to: {output_file3}")
        
        # Create edge type distribution
        fig4 = visualizer.plot_edge_type_distribution()
        output_file4 = "visualization_edge_types.html"
        fig4.write_html(output_file4)
        print(f"Edge type distribution saved to: {output_file4}")
        
        # Step 4: Adaptive updates demo
        print("\n[Step 4] Demonstrating adaptive updates...")
        adaptive = AdaptivePageRank(sample_graph, pagerank_scores)
        
        # Add a new product
        new_product = "P999"
        new_edges = [("P001", "out", "co_purchase"), ("P002", "out", "co_review"), ("P003", "in", "co_purchase")]
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
        print("2. RECOMMENDED: Start with 'All_Beauty' category (smallest, easiest)")
        print("3. Place files in data/ directory:")
        print("   - review_All_Beauty.jsonl.gz")
        print("   - meta_All_Beauty.jsonl.gz")
        print("4. Update CATEGORY variable in main.py (already set to 'All_Beauty')")
        print("5. Run the script again")
        print("\n💡 See CATEGORY_GUIDE.md for category recommendations!")
        
        return
    
    # Step 2: Build graph (without co-view)
    print("\n[Step 2] Building product graph...")
    co_purchases = loader.extract_co_purchase_relationships()
    co_reviews = loader.extract_co_review_relationships()
    
    builder = ProductGraphBuilder()
    graph = builder.build_graph(
        co_purchases=co_purchases,
        co_reviews=co_reviews,
        weight_co_purchase=1.0,
        weight_co_review=0.8
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
    
    selected_product = top_products[0][0] if top_products else None
    
    # Create network plot
    fig = visualizer.create_network_plot(selected_node=selected_product)
    output_file = "visualization_network.html"
    fig.write_html(output_file)
    print(f"Network visualization saved to: {output_file}")
    
    # Create top products plot with highlighting
    fig2 = visualizer.plot_top_products(k=20, highlight_product=selected_product)
    output_file2 = "visualization_top_products.html"
    fig2.write_html(output_file2)
    print(f"Top products visualization saved to: {output_file2}")
    
    # Create comparison chart
    if selected_product:
        fig3 = visualizer.plot_comparison_chart(selected_product, top_k=10)
        output_file3 = "visualization_comparison.html"
        fig3.write_html(output_file3)
        print(f"Comparison chart saved to: {output_file3}")
    
    # Create edge type distribution
    fig4 = visualizer.plot_edge_type_distribution()
    output_file4 = "visualization_edge_types.html"
    fig4.write_html(output_file4)
    print(f"Edge type distribution saved to: {output_file4}")
    
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