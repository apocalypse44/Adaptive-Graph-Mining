import networkx as nx
from typing import List, Tuple, Dict, Set
from collections import Counter


class ProductGraphBuilder:    
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def build_graph(self, 
                   co_purchases: List[Tuple[str, str]] = None,
                   co_views: List[Tuple[str, str]] = None,
                   co_reviews: List[Tuple[str, str]] = None,
                   weight_co_purchase: float = 1.0,
                   weight_co_view: float = 0.5,
                   weight_co_review: float = 0.8) -> nx.DiGraph:
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
        

        if co_reviews:
            print(f"Adding {len(co_reviews)} co-review edges...")
            edge_weights = Counter()
            for u, v in co_reviews:
                edge_weights[(u, v)] += weight_co_review
                edge_weights[(v, u)] += weight_co_review
            
            for (u, v), weight in edge_weights.items():
                if self.graph.has_edge(u, v):
                    self.graph[u][v]['weight'] += weight
                    # preserve existing type, but could store multiple types
                else:
                    self.graph.add_edge(u, v, weight=weight, type='co_review')

        return self.graph
    
    def add_product(self, product_id: str, relationships: List[Tuple[str, str]], 
                    weights: List[float] = None):
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
        if product_id in self.graph:
            self.graph.remove_node(product_id)
    
    def get_graph_stats(self) -> Dict:
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
        if product_id not in self.graph:
            return set()
        return set(self.graph.neighbors(product_id))

