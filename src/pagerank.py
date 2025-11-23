"""
PageRank Implementation Module
Computes PageRank scores for products in the graph
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.sparse import csr_matrix


class PageRankCalculator:
    """Calculate PageRank scores for products"""
    
    def __init__(self, graph: nx.DiGraph, damping: float = 0.85, 
                 max_iter: int = 100, tol: float = 1e-6):
        """
        Initialize PageRank calculator
        
        Args:
            graph: NetworkX directed graph
            damping: Damping factor (default 0.85)
            max_iter: Maximum iterations
            tol: Convergence tolerance
        """
        self.graph = graph
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.scores = {}
        
    def compute_pagerank(self, personalization: Optional[Dict] = None) -> Dict[str, float]:
        """
        Compute PageRank scores for all nodes
        
        Args:
            personalization: Optional personalization vector
            
        Returns:
            Dictionary mapping node IDs to PageRank scores
        """
        # Use NetworkX PageRank implementation
        self.scores = nx.pagerank(
            self.graph,
            alpha=self.damping,
            max_iter=self.max_iter,
            tol=self.tol,
            personalization=personalization,
            weight='weight'
        )
        
        return self.scores
    
    def compute_pagerank_custom(self, personalization: Optional[Dict] = None) -> Dict[str, float]:
        """
        Custom iterative PageRank implementation
        
        Args:
            personalization: Optional personalization vector
            
        Returns:
            Dictionary mapping node IDs to PageRank scores
        """
        nodes = list(self.graph.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        if n == 0:
            return {}
        
        # Build transition matrix
        M = np.zeros((n, n))
        out_degrees = dict(self.graph.out_degree(weight='weight'))
        
        for u, v, data in self.graph.edges(data=True):
            weight = data.get('weight', 1.0)
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            if out_degrees[u] > 0:
                M[v_idx, u_idx] = weight / out_degrees[u]
        
        # Handle dangling nodes (nodes with no outgoing edges)
        dangling = np.where(M.sum(axis=0) == 0)[0]
        if len(dangling) > 0:
            M[:, dangling] = 1.0 / n
        
        # Initialize PageRank vector
        if personalization:
            p = np.array([personalization.get(node, 0) for node in nodes])
            p = p / p.sum()
        else:
            p = np.ones(n) / n
        
        pr = np.ones(n) / n
        
        # Iterative computation
        for iteration in range(self.max_iter):
            pr_new = (1 - self.damping) * p + self.damping * M @ pr
            
            # Check convergence
            if np.linalg.norm(pr_new - pr, ord=1) < self.tol:
                print(f"PageRank converged after {iteration + 1} iterations")
                break
            
            pr = pr_new
        
        # Convert to dictionary
        self.scores = {nodes[i]: float(pr[i]) for i in range(n)}
        return self.scores
    
    def get_top_products(self, k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top k products by PageRank score
        
        Args:
            k: Number of top products to return
            
        Returns:
            List of (product_id, score) tuples
        """
        if not self.scores:
            return []
        
        sorted_products = sorted(
            self.scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_products[:k]
    
    def get_recommendations(self, product_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top k recommendations for a given product
        Based on neighbors' PageRank scores
        
        Args:
            product_id: ID of the product
            k: Number of recommendations
            
        Returns:
            List of (recommended_product_id, score) tuples
        """
        if product_id not in self.graph:
            return []
        
        if not self.scores:
            self.compute_pagerank()
        
        # Get neighbors and their scores
        neighbors = list(self.graph.neighbors(product_id))
        neighbor_scores = [
            (neighbor, self.scores.get(neighbor, 0))
            for neighbor in neighbors
        ]
        
        # Sort by score and return top k
        neighbor_scores.sort(key=lambda x: x[1], reverse=True)
        return neighbor_scores[:k]
    
    def update_scores(self, new_scores: Dict[str, float]):
        """Update PageRank scores (for adaptive updates)"""
        self.scores.update(new_scores)

