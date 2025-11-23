"""
Graph Builder Module
Constructs product-product graph from relationships
"""

import networkx as nx
from typing import List, Tuple, Dict, Set
from collections import Counter


class ProductGraphBuilder:
    """Build product-product graph from relationships"""
    
    def __init__(self):
        """Initialize graph builder"""
        self.graph = nx.DiGraph()
        
    def build_graph(self, 
                   co_purchases: List[Tuple[str, str]] = None,
                   co_views: List[Tuple[str, str]] = None,
                   weight_co_purchase: float = 1.0,
                   weight_co_view: float = 0.5) -> nx.DiGraph:
        """
        Build directed graph from relationships
        
        Args:
            co_purchases: List of co-purchase pairs
            co_views: List of co-view pairs
            weight_co_purchase: Weight for co-purchase edges
            weight_co_view: Weight for co-view edges
            
        Returns:
            NetworkX directed graph
        """
        self.graph = nx.DiGraph()
        
        # Add co-purchase edges
        if co_purchases:
            print(f"Adding {len(co_purchases)} co-purchase edges...")
            edge_weights = Counter()
            for u, v in co_purchases:
                # Add bidirectional edges for co-purchases
                edge_weights[(u, v)] += weight_co_purchase
                edge_weights[(v, u)] += weight_co_purchase
            
            for (u, v), weight in edge_weights.items():
                self.graph.add_edge(u, v, weight=weight, type='co_purchase')
        
        # Add co-view edges
        if co_views:
            print(f"Adding {len(co_views)} co-view edges...")
            edge_weights = Counter()
            for u, v in co_views:
                edge_weights[(u, v)] += weight_co_view
                edge_weights[(v, u)] += weight_co_view
            
            for (u, v), weight in edge_weights.items():
                if self.graph.has_edge(u, v):
                    # Combine weights if edge already exists
                    self.graph[u][v]['weight'] += weight
                else:
                    self.graph.add_edge(u, v, weight=weight, type='co_view')
        
        print(f"Graph built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def add_product(self, product_id: str, relationships: List[Tuple[str, str]], 
                    weights: List[float] = None):
        """
        Add a new product and its relationships to the graph
        
        Args:
            product_id: ID of the product to add
            relationships: List of (related_product, direction) tuples
            weights: Optional list of edge weights
        """
        if weights is None:
            weights = [1.0] * len(relationships)
        
        for (related_product, direction), weight in zip(relationships, weights):
            if direction == 'out':
                self.graph.add_edge(product_id, related_product, weight=weight)
            elif direction == 'in':
                self.graph.add_edge(related_product, product_id, weight=weight)
            else:  # bidirectional
                self.graph.add_edge(product_id, related_product, weight=weight)
                self.graph.add_edge(related_product, product_id, weight=weight)
    
    def remove_product(self, product_id: str):
        """Remove a product and all its edges from the graph"""
        if product_id in self.graph:
            self.graph.remove_node(product_id)
    
    def get_graph_stats(self) -> Dict:
        """Get statistics about the graph"""
        if self.graph.number_of_nodes() == 0:
            return {}
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            'is_strongly_connected': nx.is_strongly_connected(self.graph),
            'num_weakly_connected_components': nx.number_weakly_connected_components(self.graph)
        }
    
    def get_neighbors(self, product_id: str) -> Set[str]:
        """Get all neighbor products of a given product"""
        if product_id not in self.graph:
            return set()
        return set(self.graph.neighbors(product_id))

