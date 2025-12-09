import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import numpy as np


class GraphVisualizer:
    """Visualize product graph and PageRank results"""
    
    def __init__(self, graph: nx.DiGraph, pagerank_scores: Dict[str, float] = None):
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
            
            subgraph = self.graph.subgraph(nodes_to_keep)
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
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_colors = []
        
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
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node size based on PageRank
            score = self.pagerank_scores.get(node, 0)
            size = max(5, score * node_size_factor) if score > 0 else 10
            node_sizes.append(size)
            
            # Node color
            if node == selected_node:
                color = 'red'
            elif node in recommendations:
                color = 'orange'
            else:
                color = 'lightblue'
            node_colors.append(color)
            
            # Node text
            node_text.append(f"Product: {node[:10]}...<br>PageRank: {score:.4f}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node[:8] for node in subgraph.nodes()],
            textposition="middle center",
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=node_colors,
                size=node_sizes,
                colorbar=dict(
                    thickness=15,
                    title="Node Type",
                    xanchor="left",
                    # titleside="right"
                ),
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Product Recommendation Graph',
                # titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Node size represents PageRank score. Red = selected, Orange = recommendations",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor="left", yanchor="bottom",
                        font=dict(color="#888", size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
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
    
    def plot_top_products(self, k: int = 20) -> go.Figure:
        if not self.pagerank_scores:
            return go.Figure()
        
        top_products = sorted(
            self.pagerank_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        products = [p[0][:15] for p in top_products]
        scores = [p[1] for p in top_products]
        
        fig = go.Figure(data=[
            go.Bar(x=products, y=scores)
        ])
        fig.update_layout(
            title=f'Top {k} Products by PageRank',
            xaxis_title='Product ID',
            yaxis_title='PageRank Score',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_interactive_dashboard(self, selected_node: Optional[str] = None) -> go.Figure:
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Network Graph', 'Top Products', 
                          'PageRank Distribution', 'Recommendations'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Network graph
        network_fig = self.create_network_plot(selected_node=selected_node)
        fig.add_trace(network_fig.data[0], row=1, col=1)
        fig.add_trace(network_fig.data[1], row=1, col=1)
        
        # Top products
        top_fig = self.plot_top_products(k=10)
        fig.add_trace(top_fig.data[0], row=1, col=2)
        
        # Distribution
        dist_fig = self.plot_pagerank_distribution()
        fig.add_trace(dist_fig.data[0], row=2, col=1)
        
        # Recommendations
        if selected_node and self.pagerank_scores:
            recommendations = self._get_recommendations(selected_node, k=10)
            if recommendations:
                rec_products = [r[0][:15] for r in recommendations]
                rec_scores = [r[1] for r in recommendations]
                fig.add_trace(
                    go.Bar(x=rec_products, y=rec_scores),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, showlegend=False, title_text="Product Recommendation Dashboard")
        
        return fig
    
    def _get_recommendations(self, product_id: str, k: int = 10) -> List[Tuple[str, float]]:
        if product_id not in self.graph:
            return []
        
        neighbors = list(self.graph.neighbors(product_id))
        neighbor_scores = [
            (neighbor, self.pagerank_scores.get(neighbor, 0))
            for neighbor in neighbors
        ]
        neighbor_scores.sort(key=lambda x: x[1], reverse=True)
        return neighbor_scores[:k]


# """
# Visualization Module
# Interactive 3D and 2D graph visualization
# """

# import networkx as nx
# import plotly.graph_objects as go
# from typing import Dict, List, Optional, Tuple
# import numpy as np


# class GraphVisualizer:
#     """Visualize product graph and PageRank results"""
    
#     def __init__(self, graph: nx.DiGraph, pagerank_scores: Dict[str, float] = None):
#         """
#         Initialize visualizer
        
#         Args:
#             graph: NetworkX graph to visualize
#             pagerank_scores: Optional PageRank scores for node sizing
#         """
#         self.graph = graph
#         self.pagerank_scores = pagerank_scores or {}
    
#     def create_3d_network_plot(self,
#                                selected_node: Optional[str] = None,
#                                top_k: int = 10,
#                                node_size_factor: float = 20,
#                                max_nodes: int = 500) -> go.Figure:
#         """
#         Create interactive 3D network visualization
        
#         Args:
#             selected_node: Highlight this node and its recommendations
#             top_k: Number of top recommendations to highlight
#             node_size_factor: Factor to scale node sizes
#             max_nodes: Maximum nodes to display (for performance)
            
#         Returns:
#             Plotly 3D figure object
#         """
#         # Sample nodes if graph is too large
#         if self.graph.number_of_nodes() > max_nodes:
#             if self.pagerank_scores:
#                 top_nodes = sorted(
#                     self.pagerank_scores.items(),
#                     key=lambda x: x[1],
#                     reverse=True
#                 )[:max_nodes - 1]
#                 nodes_to_keep = {node for node, _ in top_nodes}
#             else:
#                 nodes_to_keep = set(list(self.graph.nodes())[:max_nodes])
            
#             if selected_node and selected_node in self.graph:
#                 nodes_to_keep.add(selected_node)
#                 nodes_to_keep.update(self.graph.neighbors(selected_node))
            
#             subgraph = self.graph.subgraph(nodes_to_keep)
#         else:
#             subgraph = self.graph
        
#         # Compute 3D spring layout
#         pos_3d = nx.spring_layout(subgraph, dim=3, k=1, iterations=50, seed=42)
        
#         # Get recommendations
#         recommendations = set()
#         if selected_node and selected_node in subgraph:
#             neighbors = list(subgraph.neighbors(selected_node))
#             if self.pagerank_scores:
#                 neighbor_scores = [
#                     (n, self.pagerank_scores.get(n, 0))
#                     for n in neighbors
#                 ]
#                 neighbor_scores.sort(key=lambda x: x[1], reverse=True)
#                 recommendations = {n for n, _ in neighbor_scores[:top_k]}
#             else:
#                 recommendations = set(neighbors[:top_k])
        
#         # Prepare edge traces
#         edge_x = []
#         edge_y = []
#         edge_z = []
        
#         for edge in subgraph.edges():
#             x0, y0, z0 = pos_3d[edge[0]]
#             x1, y1, z1 = pos_3d[edge[1]]
#             edge_x.extend([x0, x1, None])
#             edge_y.extend([y0, y1, None])
#             edge_z.extend([z0, z1, None])
        
#         edge_trace = go.Scatter3d(
#             x=edge_x, y=edge_y, z=edge_z,
#             mode='lines',
#             line=dict(color='rgba(125,125,125,0.3)', width=1),
#             hoverinfo='none',
#             showlegend=False
#         )
        
#         # Prepare node traces
#         node_x = []
#         node_y = []
#         node_z = []
#         node_text = []
#         node_sizes = []
#         node_colors = []
#         node_color_values = []
        
#         for node in subgraph.nodes():
#             x, y, z = pos_3d[node]
#             node_x.append(x)
#             node_y.append(y)
#             node_z.append(z)
            
#             # Node size based on PageRank
#             score = self.pagerank_scores.get(node, 0)
#             size = max(5, score * node_size_factor) if score > 0 else 5
#             node_sizes.append(size)
            
#             # Node color and hover text
#             if node == selected_node:
#                 node_colors.append('red')
#                 node_color_values.append(3)
#                 node_text.append(f"<b>SELECTED</b><br>Product: {node}<br>PageRank: {score:.6f}")
#             elif node in recommendations:
#                 node_colors.append('orange')
#                 node_color_values.append(2)
#                 node_text.append(f"<b>RECOMMENDED</b><br>Product: {node}<br>PageRank: {score:.6f}")
#             else:
#                 node_colors.append('lightblue')
#                 node_color_values.append(score)
#                 node_text.append(f"Product: {node}<br>PageRank: {score:.6f}")
        
#         node_trace = go.Scatter3d(
#             x=node_x, y=node_y, z=node_z,
#             mode='markers+text',
#             marker=dict(
#                 size=node_sizes,
#                 color=node_color_values,
#                 colorscale='Viridis',
#                 showscale=True,
#                 colorbar=dict(
#                     title="PageRank<br>Score",
#                     thickness=15,
#                     len=0.7
#                 ),
#                 line=dict(color='white', width=0.5)
#             ),
#             text=[node[:8] for node in subgraph.nodes()],
#             textposition="middle center",
#             textfont=dict(size=8),
#             hovertext=node_text,
#             hoverinfo='text',
#             showlegend=False
#         )
        
#         # Create figure
#         fig = go.Figure(data=[edge_trace, node_trace])
        
#         fig.update_layout(
#             title=dict(
#                 text='Interactive 3D Product Recommendation Network',
#                 x=0.5,
#                 xanchor='center',
#                 font=dict(size=18)
#             ),
#             scene=dict(
#                 xaxis=dict(showbackground=False, showticklabels=False, title=''),
#                 yaxis=dict(showbackground=False, showticklabels=False, title=''),
#                 zaxis=dict(showbackground=False, showticklabels=False, title=''),
#                 bgcolor='rgba(240,240,240,0.9)'
#             ),
#             showlegend=False,
#             hovermode='closest',
#             margin=dict(l=0, r=0, b=0, t=40),
#             annotations=[
#                 dict(
#                     text="🖱️ Click and drag to rotate | Scroll to zoom | Hover over nodes for details",
#                     showarrow=False,
#                     xref="paper", yref="paper",
#                     x=0.5, y=0.02,
#                     xanchor="center",
#                     font=dict(size=12, color="gray")
#                 )
#             ]
#         )
        
#         return fig
    
#     def create_interactive_network_pyvis(self,
#                                         selected_node: Optional[str] = None,
#                                         top_k: int = 10,
#                                         max_nodes: int = 500,
#                                         output_file: str = "network_graph.html") -> None:
#         """
#         Create interactive network using PyVis (physics-based)
        
#         Args:
#             selected_node: Highlight this node and its recommendations
#             top_k: Number of top recommendations to highlight
#             max_nodes: Maximum nodes to display
#             output_file: Output HTML filename
#         """
#         try:
#             from pyvis.network import Network
#         except ImportError:
#             print("PyVis not installed. Install with: pip install pyvis")
#             return
        
#         # Sample nodes if graph is too large
#         if self.graph.number_of_nodes() > max_nodes:
#             if self.pagerank_scores:
#                 top_nodes = sorted(
#                     self.pagerank_scores.items(),
#                     key=lambda x: x[1],
#                     reverse=True
#                 )[:max_nodes - 1]
#                 nodes_to_keep = {node for node, _ in top_nodes}
#             else:
#                 nodes_to_keep = set(list(self.graph.nodes())[:max_nodes])
            
#             if selected_node and selected_node in self.graph:
#                 nodes_to_keep.add(selected_node)
#                 nodes_to_keep.update(self.graph.neighbors(selected_node))
            
#             subgraph = self.graph.subgraph(nodes_to_keep)
#         else:
#             subgraph = self.graph
        
#         # Get recommendations
#         recommendations = set()
#         if selected_node and selected_node in subgraph:
#             neighbors = list(subgraph.neighbors(selected_node))
#             if self.pagerank_scores:
#                 neighbor_scores = [
#                     (n, self.pagerank_scores.get(n, 0))
#                     for n in neighbors
#                 ]
#                 neighbor_scores.sort(key=lambda x: x[1], reverse=True)
#                 recommendations = {n for n, _ in neighbor_scores[:top_k]}
#             else:
#                 recommendations = set(neighbors[:top_k])
        
#         # Create PyVis network
#         net = Network(height="800px", width="100%", bgcolor="#ffffff", 
#                      font_color="black", directed=True)
        
#         # Configure physics for better interaction
#         net.set_options("""
#         {
#           "physics": {
#             "forceAtlas2Based": {
#               "gravitationalConstant": -50,
#               "centralGravity": 0.01,
#               "springLength": 100,
#               "springConstant": 0.08
#             },
#             "maxVelocity": 50,
#             "solver": "forceAtlas2Based",
#             "timestep": 0.35,
#             "stabilization": {"iterations": 150}
#           }
#         }
#         """)
        
#         # Add nodes with styling
#         for node in subgraph.nodes():
#             score = self.pagerank_scores.get(node, 0)
#             size = max(10, score * 1000)
            
#             if node == selected_node:
#                 color = '#ff0000'
#                 title = f"<b>SELECTED</b><br>Product: {node}<br>PageRank: {score:.6f}"
#             elif node in recommendations:
#                 color = '#ffa500'
#                 title = f"<b>RECOMMENDED</b><br>Product: {node}<br>PageRank: {score:.6f}"
#             else:
#                 color = '#4a90e2'
#                 title = f"Product: {node}<br>PageRank: {score:.6f}"
            
#             net.add_node(node, 
#                         label=node[:8],
#                         title=title,
#                         size=size,
#                         color=color)
        
#         # Add edges
#         for edge in subgraph.edges():
#             weight = subgraph[edge[0]][edge[1]].get('weight', 1)
#             net.add_edge(edge[0], edge[1], value=weight, arrows='to')
        
#         # Save to HTML
#         net.save_graph(output_file)
#         print(f"Interactive network saved to: {output_file}")
#         print(f"Open {output_file} in your browser to interact with the graph!")
    
#     def plot_top_products(self, k: int = 20) -> go.Figure:
#         """
#         Plot bar chart of top k products by PageRank
        
#         Args:
#             k: Number of top products to show
            
#         Returns:
#             Plotly figure object
#         """
#         if not self.pagerank_scores:
#             return go.Figure()
        
#         top_products = sorted(
#             self.pagerank_scores.items(),
#             key=lambda x: x[1],
#             reverse=True
#         )[:k]
        
#         products = [p[0][:15] for p in top_products]
#         scores = [p[1] for p in top_products]
        
#         fig = go.Figure(data=[
#             go.Bar(x=products, y=scores, marker_color='steelblue')
#         ])
#         fig.update_layout(
#             title=f'Top {k} Products by PageRank',
#             xaxis_title='Product ID',
#             yaxis_title='PageRank Score',
#             xaxis_tickangle=-45,
#             hovermode='x'
#         )
        
#         return fig
    
#     def plot_pagerank_distribution(self) -> go.Figure:
#         """
#         Plot distribution of PageRank scores
        
#         Returns:
#             Plotly figure object
#         """
#         if not self.pagerank_scores:
#             return go.Figure()
        
#         scores = list(self.pagerank_scores.values())
        
#         fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=50, marker_color='steelblue')])
#         fig.update_layout(
#             title='PageRank Score Distribution',
#             xaxis_title='PageRank Score',
#             yaxis_title='Frequency',
#             hovermode='x'
#         )
        
#         return fig
    
#     def _get_recommendations(self, product_id: str, k: int = 10) -> List[Tuple[str, float]]:
#         """Get recommendations for a product"""
#         if product_id not in self.graph:
#             return []
        
#         neighbors = list(self.graph.neighbors(product_id))
#         neighbor_scores = [
#             (neighbor, self.pagerank_scores.get(neighbor, 0))
#             for neighbor in neighbors
#         ]
#         neighbor_scores.sort(key=lambda x: x[1], reverse=True)
#         return neighbor_scores[:k]