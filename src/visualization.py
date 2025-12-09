import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import numpy as np


class GraphVisualizer:
    """Visualize product graph and PageRank results with multi-edge support"""
    
    def __init__(self, graph: nx.MultiDiGraph, pagerank_scores: Dict[str, float] = None):
        self.graph = graph
        self.pagerank_scores = pagerank_scores or {}
        
    def create_network_plot(self, 
                        selected_node: Optional[str] = None,
                        top_k: int = 10,
                        layout: str = 'spring',
                        node_size_factor: float = 1000,
                        max_nodes: int = 500) -> go.Figure:
        # Sample nodes if graph is too large
        if self.graph.number_of_nodes() > max_nodes:
            # Keep top nodes by PageRank and selected node
            if self.pagerank_scores:
                top_nodes = sorted(
                    self.pagerank_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:max_nodes - 1]
                nodes_to_keep = {node for node, _ in top_nodes}
            else:
                nodes_to_keep = set(list(self.graph.nodes())[:max_nodes])
            
            if selected_node and selected_node in self.graph:
                nodes_to_keep.add(selected_node)
                # Add neighbors of selected node
                nodes_to_keep.update(self.graph.neighbors(selected_node))
            
            subgraph = self.graph.subgraph(nodes_to_keep).copy()
        else:
            subgraph = self.graph
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(subgraph)
        elif layout == 'kamada_kawai':
            try:
                pos = nx.kamada_kawai_layout(subgraph)
            except:
                pos = nx.spring_layout(subgraph)
        else:
            pos = nx.spring_layout(subgraph)
        
        # Prepare edge traces by type
        edge_traces = []
        
        # Co-purchase edges (blue)
        edge_x_purchase = []
        edge_y_purchase = []
        for u, v, data in subgraph.edges(data=True):
            if data.get('edge_type') == 'co_purchase':
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x_purchase.extend([x0, x1, None])
                edge_y_purchase.extend([y0, y1, None])
        
        if edge_x_purchase:
            edge_traces.append(go.Scatter(
                x=edge_x_purchase, y=edge_y_purchase,
                line=dict(width=1.5, color='#4169E1'),  # Royal blue
                hoverinfo='none',
                mode='lines',
                name='Co-purchase',
                showlegend=True,
                legendgroup='edges'
            ))
        
        # Co-review edges (green)
        edge_x_review = []
        edge_y_review = []
        for u, v, data in subgraph.edges(data=True):
            if data.get('edge_type') == 'co_review':
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x_review.extend([x0, x1, None])
                edge_y_review.extend([y0, y1, None])
        
        if edge_x_review:
            edge_traces.append(go.Scatter(
                x=edge_x_review, y=edge_y_review,
                line=dict(width=1.5, color='#32CD32', dash='dash'),  # Lime green, dashed
                hoverinfo='none',
                mode='lines',
                name='Co-review',
                showlegend=True,
                legendgroup='edges'
            ))
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_colors = []
        node_symbols = []
        
        recommendations = set()
        if selected_node and selected_node in subgraph:
            # Get top recommendations
            neighbors = list(subgraph.neighbors(selected_node))
            if self.pagerank_scores:
                neighbor_scores = [
                    (n, self.pagerank_scores.get(n, 0))
                    for n in neighbors
                ]
                neighbor_scores.sort(key=lambda x: x[1], reverse=True)
                recommendations = {n for n, _ in neighbor_scores[:top_k]}
            else:
                recommendations = set(neighbors[:top_k])
        
        # Separate traces for different node types
        selected_x, selected_y, selected_text, selected_size = [], [], [], []
        rec_x, rec_y, rec_text, rec_sizes = [], [], [], []
        regular_x, regular_y, regular_text, regular_sizes = [], [], [], []
        
        for node in subgraph.nodes():
            x, y = pos[node]
            
            # Node size based on PageRank
            score = self.pagerank_scores.get(node, 0)
            size = max(12, score * node_size_factor) if score > 0 else 15
            
            # Count edge types
            purchase_count = sum(1 for _, _, d in subgraph.out_edges(node, data=True) if d.get('edge_type') == 'co_purchase')
            review_count = sum(1 for _, _, d in subgraph.out_edges(node, data=True) if d.get('edge_type') == 'co_review')
            
            # Node text
            text = (
                f"Product: {node}<br>"
                f"PageRank: {score:.6f}<br>"
                f"Co-purchase edges: {purchase_count}<br>"
                f"Co-review edges: {review_count}"
            )
            
            # Categorize nodes
            if node == selected_node:
                selected_x.append(x)
                selected_y.append(y)
                selected_text.append(text)
                selected_size.append(size * 1.5)
            elif node in recommendations:
                rec_x.append(x)
                rec_y.append(y)
                rec_text.append(text)
                rec_sizes.append(size * 1.2)
            else:
                regular_x.append(x)
                regular_y.append(y)
                regular_text.append(text)
                regular_sizes.append(size)
        
        node_traces = []
        
        # Regular nodes
        if regular_x:
            node_traces.append(go.Scatter(
                x=regular_x, y=regular_y,
                mode='markers',
                hoverinfo='text',
                hovertext=regular_text,
                marker=dict(
                    color='lightblue',
                    size=regular_sizes,
                    line=dict(width=1, color='white'),
                    symbol='circle'
                ),
                name='Other Products',
                showlegend=True,
                legendgroup='nodes'
            ))
        
        # Recommendation nodes
        if rec_x:
            node_traces.append(go.Scatter(
                x=rec_x, y=rec_y,
                mode='markers',
                hoverinfo='text',
                hovertext=rec_text,
                marker=dict(
                    color='orange',
                    size=rec_sizes,
                    line=dict(width=2, color='darkorange'),
                    symbol='diamond'
                ),
                name=f'Top {top_k} Recommendations',
                showlegend=True,
                legendgroup='nodes'
            ))
        
        # Selected node
        if selected_x:
            node_traces.append(go.Scatter(
                x=selected_x, y=selected_y,
                mode='markers',
                hoverinfo='text',
                hovertext=selected_text,
                marker=dict(
                    color='red',
                    size=selected_size,
                    line=dict(width=3, color='darkred'),
                    symbol='star'
                ),
                name=f'Selected: {selected_node}',
                showlegend=True,
                legendgroup='nodes'
            ))
        
        # Build title
        if selected_node:
            selected_score = self.pagerank_scores.get(selected_node, 0)
            title = f'Product Recommendation Graph<br><sub>Showing recommendations for: <b>{selected_node}</b> (PageRank: {selected_score:.6f})</sub>'
        else:
            title = 'Product Recommendation Graph'
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + node_traces,
            layout=go.Layout(
                title=title,
                showlegend=True,
                height=600,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=80),
                annotations=[
                    dict(
                        text="🔴 Star = Selected | 🟠 Diamond = Recommendations | 🔵 Circle = Other Products",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.05,
                        xanchor="center", yanchor="bottom",
                        font=dict(color="#555", size=11)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                legend=dict(
                    x=1.02, y=1.0,
                    xanchor='left', yanchor='top',
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='gray',
                    borderwidth=1
                )
            )
        )
        
        return fig
    
    def plot_edge_type_distribution(self) -> go.Figure:
        """Plot distribution of edge types"""
        co_purchase_count = sum(1 for u, v, d in self.graph.edges(data=True) if d.get('edge_type') == 'co_purchase')
        co_review_count = sum(1 for u, v, d in self.graph.edges(data=True) if d.get('edge_type') == 'co_review')
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Co-purchase', 'Co-review'],
                y=[co_purchase_count, co_review_count],
                marker_color=['#4169E1', '#32CD32'],
                text=[co_purchase_count, co_review_count],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title='Edge Type Distribution',
            xaxis_title='Edge Type',
            yaxis_title='Count'
        )
        
        return fig
    
    def plot_pagerank_distribution(self) -> go.Figure:
        if not self.pagerank_scores:
            return go.Figure()
        
        scores = list(self.pagerank_scores.values())
        
        fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=50)])
        fig.update_layout(
            title='PageRank Score Distribution',
            xaxis_title='PageRank Score',
            yaxis_title='Frequency'
        )
        
        return fig
    
    def plot_top_products(self, k: int = 20, highlight_product: Optional[str] = None) -> go.Figure:
        """Plot top products with optional highlighting of a specific product"""
        if not self.pagerank_scores:
            return go.Figure()
        
        top_products = sorted(
            self.pagerank_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        products = [p[0][:15] for p in top_products]
        scores = [p[1] for p in top_products]
        
        # Color bars based on whether they match highlight_product
        colors = []
        for p, _ in top_products:
            if highlight_product and p == highlight_product:
                colors.append('red')
            else:
                colors.append('#4169E1')
        
        fig = go.Figure(data=[
            go.Bar(
                x=products, 
                y=scores, 
                marker_color=colors,
                text=[f'{s:.6f}' for s in scores],
                textposition='outside'
            )
        ])
        
        title_text = f'Top {k} Products by PageRank'
        if highlight_product:
            selected_score = self.pagerank_scores.get(highlight_product, 0)
            # Find rank
            all_ranked = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)
            rank = next((i+1 for i, (p, _) in enumerate(all_ranked) if p == highlight_product), None)
            
            if rank and rank <= k:
                title_text += f'<br><sub>Selected product <b>{highlight_product}</b> is highlighted in red (Rank #{rank})</sub>'
            elif rank:
                title_text += f'<br><sub>Selected product <b>{highlight_product}</b> is ranked #{rank} (Score: {selected_score:.6f}, not in top {k})</sub>'
        
        fig.update_layout(
            title=title_text,
            xaxis_title='Product ID',
            yaxis_title='PageRank Score',
            xaxis_tickangle=-45,
            height=500
        )
        
        return fig
    
    def plot_comparison_chart(self, selected_node: str, top_k: int = 10) -> go.Figure:
        """Create a comparison chart showing selected product vs its recommendations"""
        if selected_node not in self.graph:
            return go.Figure()
        
        # Get recommendations
        recommendations = self._get_recommendations(selected_node, k=top_k)
        
        if not recommendations:
            return go.Figure()
        
        # Build comparison data
        products = [selected_node] + [r[0] for r in recommendations]
        scores = [self.pagerank_scores.get(selected_node, 0)] + [r[1] for r in recommendations]
        colors = ['red'] + ['orange'] * len(recommendations)
        labels = ['SELECTED'] + [f'Rec #{i+1}' for i in range(len(recommendations))]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[p[:15] for p in products],
                y=scores,
                marker_color=colors,
                text=labels,
                textposition='outside',
                hovertext=[f'{p}<br>PageRank: {s:.6f}' for p, s in zip(products, scores)],
                hoverinfo='text'
            )
        ])
        
        fig.update_layout(
            title=f'PageRank Comparison: Selected Product vs Top {top_k} Recommendations<br><sub>Comparing <b>{selected_node}</b> with its recommended products</sub>',
            xaxis_title='Product ID',
            yaxis_title='PageRank Score',
            xaxis_tickangle=-45,
            height=500
        )
        
        return fig
    
    def create_interactive_dashboard(self, selected_node: Optional[str] = None) -> go.Figure:
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Network Graph', 'Top Products', 
                          'Edge Type Distribution', 'PageRank Distribution'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Network graph
        network_fig = self.create_network_plot(selected_node=selected_node)
        for trace in network_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # Top products with highlighting
        top_fig = self.plot_top_products(k=10, highlight_product=selected_node)
        fig.add_trace(top_fig.data[0], row=1, col=2)
        
        # Edge type distribution
        edge_fig = self.plot_edge_type_distribution()
        fig.add_trace(edge_fig.data[0], row=2, col=1)
        
        # PageRank distribution
        dist_fig = self.plot_pagerank_distribution()
        fig.add_trace(dist_fig.data[0], row=2, col=2)
        
        dashboard_title = "Product Recommendation Dashboard"
        if selected_node:
            dashboard_title += f" - Analysis for {selected_node}"
        
        fig.update_layout(height=700, showlegend=False, title_text=dashboard_title)
        
        return fig
    
    def _get_recommendations(self, product_id: str, k: int = 10, edge_type: str = None) -> List[Tuple[str, float]]:
        """Get recommendations, optionally filtered by edge type"""
        if product_id not in self.graph:
            return []
        
        neighbors = []
        if edge_type:
            # Filter by edge type
            for neighbor in self.graph.neighbors(product_id):
                for key, data in self.graph[product_id][neighbor].items():
                    if data.get('edge_type') == edge_type:
                        neighbors.append(neighbor)
                        break
        else:
            neighbors = list(self.graph.neighbors(product_id))
        
        neighbor_scores = [
            (neighbor, self.pagerank_scores.get(neighbor, 0))
            for neighbor in neighbors
        ]
        neighbor_scores.sort(key=lambda x: x[1], reverse=True)
        return neighbor_scores[:k]