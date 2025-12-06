"""
Adaptive Update Module
Implements incremental PageRank updates when graph changes
"""

import networkx as nx
from networkx.exception import PowerIterationFailedConvergence
import numpy as np
from typing import Dict, Set, List, Tuple
from pagerank import PageRankCalculator


class AdaptivePageRank:
    """Incremental PageRank updates for graph changes"""
    
    def __init__(self, graph: nx.DiGraph, initial_scores: Dict[str, float],
                 damping: float = 0.85):
        """
        Initialize adaptive PageRank
        
        Args:
            graph: NetworkX directed graph
            initial_scores: Initial PageRank scores
            damping: Damping factor
        """
        self.graph = graph
        self.scores = initial_scores.copy()
        self.damping = damping
        
    def add_node(self, node: str, edges: List[Tuple[str, str]], 
                 edge_weights: List[float] = None) -> Dict[str, float]:
        """
        Add a new node and update PageRank incrementally
        
        Args:
            node: New node to add
            edges: List of (target_node, direction) tuples
            edge_weights: Optional weights for edges
            
        Returns:
            Updated scores for affected nodes
        """
        if edge_weights is None:
            edge_weights = [1.0] * len(edges)
        
        # Add node and edges to graph
        for (target, direction), weight in zip(edges, edge_weights):
            if direction == 'out':
                self.graph.add_edge(node, target, weight=weight)
            elif direction == 'in':
                self.graph.add_edge(target, node, weight=weight)
            else:  # bidirectional
                self.graph.add_edge(node, target, weight=weight)
                self.graph.add_edge(target, node, weight=weight)
        
        # Initialize score for new node
        if node not in self.scores:
            self.scores[node] = (1 - self.damping) / len(self.graph)
        
        # Update scores for affected nodes
        affected_nodes = {node}
        for target, _ in edges:
            affected_nodes.add(target)
            # Add neighbors of affected nodes
            affected_nodes.update(self.graph.predecessors(target))
            affected_nodes.update(self.graph.successors(target))
        
        # Recompute PageRank for affected subgraph
        return self._update_affected_nodes(affected_nodes)
    
    def remove_node(self, node: str) -> Dict[str, float]:
        """
        Remove a node and update PageRank incrementally
        
        Args:
            node: Node to remove
            
        Returns:
            Updated scores for affected nodes
        """
        if node not in self.graph:
            return {}
        
        # Get affected nodes before removal
        affected_nodes = {node}
        affected_nodes.update(self.graph.predecessors(node))
        affected_nodes.update(self.graph.successors(node))
        
        # Remove node from graph
        self.graph.remove_node(node)
        if node in self.scores:
            del self.scores[node]
        
        # Redistribute removed node's score
        n = len(self.graph)
        if n > 0:
            redistribution = self.scores.get(node, 0) / n
            for affected_node in affected_nodes:
                if affected_node in self.scores:
                    self.scores[affected_node] += redistribution
        
        # Update affected nodes
        return self._update_affected_nodes(affected_nodes)
    
    def add_edge(self, source: str, target: str, weight: float = 1.0) -> Dict[str, float]:
        """
        Add an edge and update PageRank incrementally
        
        Args:
            source: Source node
            target: Target node
            weight: Edge weight
            
        Returns:
            Updated scores for affected nodes
        """
        self.graph.add_edge(source, target, weight=weight)
        
        # Get affected nodes
        affected_nodes = {source, target}
        affected_nodes.update(self.graph.predecessors(target))
        affected_nodes.update(self.graph.successors(source))
        
        return self._update_affected_nodes(affected_nodes)
    
    def _update_affected_nodes(self, affected_nodes: Set[str], 
                              max_iter: int = 100) -> Dict[str, float]:
        """
        Update PageRank scores for affected nodes using iterative refinement
        
        Args:
            affected_nodes: Set of nodes to update
            max_iter: Maximum iterations for refinement (increased from 20 to 100)
            
        Returns:
            Updated scores for affected nodes
        """
        if not affected_nodes:
            return {}
        
        # Create subgraph of affected nodes and their neighbors
        subgraph_nodes = set(affected_nodes)
        for node in affected_nodes:
            subgraph_nodes.update(self.graph.predecessors(node))
            subgraph_nodes.update(self.graph.successors(node))
        
        # If subgraph is too large (>50% of graph), use simpler update or full recompute
        if len(subgraph_nodes) > len(self.graph) * 0.5:
            # Too many nodes affected, use full recomputation for better accuracy
            print(f"  Warning: Large subgraph ({len(subgraph_nodes)}/{len(self.graph)} nodes). Using full recomputation...")
            return self._full_recompute_affected(affected_nodes)
        
        # Use PageRank calculator for subgraph
        subgraph = self.graph.subgraph(subgraph_nodes).copy()
        
        # Try adaptive update with error handling
        try:
            calculator = PageRankCalculator(subgraph, damping=self.damping, max_iter=max_iter, tol=1e-6)
            
            # Compute PageRank with personalization from current scores
            personalization = {node: self.scores.get(node, 0) for node in subgraph_nodes}
            if sum(personalization.values()) > 0:
                total = sum(personalization.values())
                personalization = {k: v / total for k, v in personalization.items()}
            else:
                personalization = None
            
            new_scores = calculator.compute_pagerank(personalization=personalization)
            
            # Update scores
            for node in affected_nodes:
                if node in new_scores:
                    self.scores[node] = new_scores[node]
            
            return {node: self.scores[node] for node in affected_nodes if node in self.scores}
            
        except (PowerIterationFailedConvergence, Exception) as e:
            # If convergence fails, fall back to simpler incremental update
            print(f"  Warning: Adaptive update convergence failed. Using incremental update method...")
            return self._incremental_update_fallback(affected_nodes)
    
    def _incremental_update_fallback(self, affected_nodes: Set[str]) -> Dict[str, float]:
        """
        Fallback method: Simple incremental update when subgraph PageRank fails
        Uses a simplified update based on local graph structure
        """
        updated = {}
        
        for node in affected_nodes:
            if node not in self.graph:
                continue
                
            # Simple update: redistribute based on in-degree and current scores
            in_degree = self.graph.in_degree(node, weight='weight')
            out_degree = self.graph.out_degree(node, weight='weight')
            
            if in_degree > 0 or out_degree > 0:
                # Update score based on neighbors' scores
                neighbor_contrib = 0.0
                for predecessor in self.graph.predecessors(node):
                    pred_out = self.graph.out_degree(predecessor, weight='weight')
                    if pred_out > 0:
                        edge_weight = self.graph[predecessor][node].get('weight', 1.0)
                        neighbor_contrib += (self.scores.get(predecessor, 0) * edge_weight / pred_out)
                
                # Update with damping
                new_score = (1 - self.damping) / len(self.graph) + self.damping * neighbor_contrib
                self.scores[node] = new_score
                updated[node] = new_score
        
        return updated
    
    def _full_recompute_affected(self, affected_nodes: Set[str]) -> Dict[str, float]:
        """
        Full recomputation for affected nodes (fallback for large subgraphs)
        """
        # Recompute PageRank for entire graph
        calculator = PageRankCalculator(self.graph, damping=self.damping)
        all_scores = calculator.compute_pagerank()
        
        # Update all scores
        self.scores = all_scores
        
        # Return only affected nodes
        return {node: all_scores[node] for node in affected_nodes if node in all_scores}
    
    def full_recompute(self) -> Dict[str, float]:
        """
        Perform full PageRank recomputation (fallback)
        
        Returns:
            All updated scores
        """
        calculator = PageRankCalculator(self.graph, damping=self.damping)
        self.scores = calculator.compute_pagerank()
        return self.scores
    
    def get_scores(self) -> Dict[str, float]:
        """Get current PageRank scores"""
        return self.scores.copy()

