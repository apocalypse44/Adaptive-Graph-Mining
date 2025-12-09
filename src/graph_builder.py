import networkx as nx
from typing import List, Tuple, Dict, Set
from collections import Counter


class ProductGraphBuilder:    
    def __init__(self):
        # Use MultiDiGraph to support multiple edge types between same nodes
        self.graph = nx.MultiDiGraph()
        
    def build_graph(self, 
                   co_purchases: List[Tuple[str, str]] = None,
                   co_reviews: List[Tuple[str, str]] = None,
                   weight_co_purchase: float = 1.0,
                   weight_co_review: float = 0.8) -> nx.MultiDiGraph:
        """
        Build a multi-edge graph with separate edges for co-purchase and co-review.
        Co-view edges are excluded as they're always 0.
        """
        self.graph = nx.MultiDiGraph()
        
        # Add co-purchase edges
        if co_purchases:
            print(f"Adding {len(co_purchases)} co-purchase edges...")
            edge_weights = Counter()
            for u, v in co_purchases:
                # Add bidirectional edges for co-purchases
                edge_weights[(u, v)] += weight_co_purchase
                edge_weights[(v, u)] += weight_co_purchase
            
            for (u, v), weight in edge_weights.items():
                self.graph.add_edge(u, v, weight=weight, edge_type='co_purchase')
        
        # Add co-review edges (separate from co-purchase)
        if co_reviews:
            print(f"Adding {len(co_reviews)} co-review edges...")
            edge_weights = Counter()
            for u, v in co_reviews:
                edge_weights[(u, v)] += weight_co_review
                edge_weights[(v, u)] += weight_co_review
            
            for (u, v), weight in edge_weights.items():
                # Add as separate edge even if co-purchase edge exists
                self.graph.add_edge(u, v, weight=weight, edge_type='co_review')
        
        print(f"Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        print(f"  Co-purchase edges: {sum(1 for u, v, d in self.graph.edges(data=True) if d.get('edge_type') == 'co_purchase')}")
        print(f"  Co-review edges: {sum(1 for u, v, d in self.graph.edges(data=True) if d.get('edge_type') == 'co_review')}")
        
        return self.graph
    
    def add_product(self, product_id: str, relationships: List[Tuple[str, str, str]], 
                    weights: List[float] = None):
        """
        Add a product with relationships.
        relationships: List of (related_product, direction, edge_type) tuples
        """
        if weights is None:
            weights = [1.0] * len(relationships)
        
        for (related_product, direction, edge_type), weight in zip(relationships, weights):
            if direction == 'out':
                self.graph.add_edge(product_id, related_product, weight=weight, edge_type=edge_type)
            elif direction == 'in':
                self.graph.add_edge(related_product, product_id, weight=weight, edge_type=edge_type)
            else:  # bidirectional
                self.graph.add_edge(product_id, related_product, weight=weight, edge_type=edge_type)
                self.graph.add_edge(related_product, product_id, weight=weight, edge_type=edge_type)
    
    def remove_product(self, product_id: str):
        if product_id in self.graph:
            self.graph.remove_node(product_id)
    
    def get_graph_stats(self) -> Dict:
        if self.graph.number_of_nodes() == 0:
            return {}
        
        co_purchase_count = sum(1 for u, v, d in self.graph.edges(data=True) if d.get('edge_type') == 'co_purchase')
        co_review_count = sum(1 for u, v, d in self.graph.edges(data=True) if d.get('edge_type') == 'co_review')
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'co_purchase_edges': co_purchase_count,
            'co_review_edges': co_review_count,
            'density': nx.density(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'is_strongly_connected': nx.is_strongly_connected(self.graph),
            'num_weakly_connected_components': nx.number_weakly_connected_components(self.graph)
        }
    
    def get_neighbors(self, product_id: str, edge_type: str = None) -> Set[str]:
        """
        Get neighbors, optionally filtered by edge type.
        """
        if product_id not in self.graph:
            return set()
        
        if edge_type is None:
            return set(self.graph.neighbors(product_id))
        
        # Filter by edge type
        neighbors = set()
        for neighbor in self.graph.neighbors(product_id):
            # Check if any edge to this neighbor has the specified type
            for key, data in self.graph[product_id][neighbor].items():
                if data.get('edge_type') == edge_type:
                    neighbors.add(neighbor)
                    break
        
        return neighbors