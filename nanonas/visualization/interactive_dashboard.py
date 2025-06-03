"""
Interactive NAS Dashboard
========================

A professional web-based dashboard for visualizing and analyzing Neural Architecture 
Search results in real-time. Built with Dash and Plotly for publication-quality 
interactive visualizations.

Key Features:
- Real-time search progress monitoring
- Interactive architecture visualization  
- Performance analysis and comparisons
- Multi-objective optimization plots
- Architecture evolution tracking
- Search landscape exploration
- Export functionality for results

Example Usage:
    >>> from nanonas.visualization.interactive_dashboard import NASDashboard
    >>> 
    >>> dashboard = NASDashboard()
    >>> dashboard.add_search_results(results)
    >>> dashboard.run_server(debug=True, port=8050)
"""

import dash
from dash import dcc, html, Input, Output, State, callback_table
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import dash_cytoscape as cyto
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import base64
import io
from datetime import datetime
import threading
import time

from ..core.architecture import Architecture
from ..utils.metrics import compute_model_stats
from .architecture_viz import ArchitectureVisualizer
from .search_viz import SearchProgressTracker


class NASDashboard:
    """
    Interactive web dashboard for Neural Architecture Search visualization.
    
    This dashboard provides a comprehensive interface for monitoring NAS experiments,
    visualizing architectures, analyzing performance, and exploring search dynamics.
    
    Args:
        title: Dashboard title
        theme: Color theme ('plotly', 'plotly_white', 'plotly_dark')  
        update_interval: Real-time update interval in milliseconds
        max_data_points: Maximum data points to display in time series
        port: Server port number
        debug: Enable debug mode
    """
    
    def __init__(
        self,
        title: str = "nanoNAS: Neural Architecture Search Dashboard",
        theme: str = "plotly_white",
        update_interval: int = 5000,
        max_data_points: int = 1000,
        port: int = 8050,
        debug: bool = False
    ):
        self.title = title
        self.theme = theme
        self.update_interval = update_interval
        self.max_data_points = max_data_points
        self.port = port
        self.debug = debug
        
        # Data storage
        self.search_results = {}
        self.current_experiment = None
        self.architecture_data = []
        self.performance_data = []
        self.search_progress = []
        
        # Components
        self.app = None
        self.arch_visualizer = ArchitectureVisualizer()
        
        # Setup
        self._setup_logging()
        self._initialize_app()
        self._setup_callbacks()
    
    def _initialize_app(self) -> None:
        """Initialize the Dash application."""
        self.app = dash.Dash(
            __name__,
            title=self.title,
            suppress_callback_exceptions=True,
            external_stylesheets=[
                'https://codepen.io/chriddyp/pen/bWLwgP.css',
                'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
            ]
        )
        
        # Custom CSS
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    body {
                        font-family: 'Inter', sans-serif;
                        margin: 0;
                        background-color: #f8f9fa;
                    }
                    .dashboard-header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        margin-bottom: 20px;
                        border-radius: 8px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }
                    .metric-card {
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                        margin-bottom: 20px;
                    }
                    .architecture-node {
                        border-radius: 8px;
                        border: 2px solid #e1e5e9;
                        background: white;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        # Layout
        self.app.layout = self._create_layout()
    
    def _create_layout(self) -> html.Div:
        """Create the main dashboard layout."""
        return html.Div([
            # Header
            html.Div([
                html.H1("🧠 nanoNAS Dashboard", className="dashboard-title"),
                html.P("Real-time Neural Architecture Search Monitoring & Analysis", 
                       style={"margin": "10px 0", "opacity": "0.9"}),
                html.Div([
                    html.Span("🔄 Auto-refresh: "),
                    dcc.Interval(
                        id='interval-component',
                        interval=self.update_interval,
                        n_intervals=0
                    ),
                    html.Span("📊 Last updated: "),
                    html.Span(id='last-updated', children=datetime.now().strftime("%H:%M:%S"))
                ], style={"display": "flex", "align-items": "center", "gap": "10px"})
            ], className="dashboard-header"),
            
            # Main content
            dcc.Tabs(id="main-tabs", value="overview", children=[
                
                # Overview Tab
                dcc.Tab(label="📊 Overview", value="overview", children=[
                    html.Div([
                        # Key metrics row
                        html.Div([
                            html.Div([
                                html.H3("🎯 Best Accuracy"),
                                html.H2(id="best-accuracy", children="--"),
                                html.P("Current best performance")
                            ], className="metric-card", style={"width": "23%", "display": "inline-block"}),
                            
                            html.Div([
                                html.H3("🔍 Architectures Evaluated"),
                                html.H2(id="total-architectures", children="--"),
                                html.P("Total candidates tested")
                            ], className="metric-card", style={"width": "23%", "display": "inline-block", "margin-left": "2%"}),
                            
                            html.Div([
                                html.H3("⏱️ Search Time"),
                                html.H2(id="search-time", children="--"),
                                html.P("Elapsed time")
                            ], className="metric-card", style={"width": "23%", "display": "inline-block", "margin-left": "2%"}),
                            
                            html.Div([
                                html.H3("🏆 Strategy"),
                                html.H2(id="best-strategy", children="--"),
                                html.P("Best performing method")
                            ], className="metric-card", style={"width": "23%", "display": "inline-block", "margin-left": "2%"}),
                        ]),
                        
                        # Charts row
                        html.Div([
                            html.Div([
                                dcc.Graph(id="accuracy-progress-chart")
                            ], style={"width": "48%", "display": "inline-block"}),
                            
                            html.Div([
                                dcc.Graph(id="strategy-comparison-chart")
                            ], style={"width": "48%", "display": "inline-block", "margin-left": "4%"}),
                        ]),
                        
                        # Architecture evolution
                        html.Div([
                            dcc.Graph(id="architecture-evolution-chart")
                        ], style={"margin-top": "20px"})
                        
                    ], style={"padding": "20px"})
                ]),
                
                # Architecture Tab
                dcc.Tab(label="🏗️ Architecture", value="architecture", children=[
                    html.Div([
                        html.Div([
                            html.H3("Architecture Selector"),
                            dcc.Dropdown(
                                id="architecture-dropdown",
                                placeholder="Select an architecture to visualize",
                                style={"margin-bottom": "20px"}
                            ),
                            
                            html.Div([
                                html.Div([
                                    cyto.Cytoscape(
                                        id='architecture-graph',
                                        layout={'name': 'dagre', 'rankDir': 'TB'},
                                        style={'width': '100%', 'height': '500px'},
                                        elements=[]
                                    )
                                ], style={"width": "65%", "display": "inline-block"}),
                                
                                html.Div([
                                    html.H4("Architecture Details"),
                                    html.Div(id="architecture-details"),
                                    
                                    html.H4("Performance Metrics", style={"margin-top": "30px"}),
                                    html.Div(id="architecture-metrics"),
                                    
                                    html.H4("Export Options", style={"margin-top": "30px"}),
                                    html.Button("📥 Download Architecture", id="download-arch-btn", 
                                              className="button", style={"margin": "5px"}),
                                    html.Button("📊 Download Metrics", id="download-metrics-btn", 
                                              className="button", style={"margin": "5px"}),
                                    dcc.Download(id="download-architecture"),
                                    dcc.Download(id="download-metrics")
                                ], style={"width": "30%", "display": "inline-block", "margin-left": "5%", 
                                         "vertical-align": "top", "padding": "20px", 
                                         "background": "white", "border-radius": "8px"})
                            ])
                        ], className="metric-card")
                    ], style={"padding": "20px"})
                ]),
                
                # Analysis Tab
                dcc.Tab(label="📈 Analysis", value="analysis", children=[
                    html.Div([
                        html.Div([
                            html.H3("Performance Analysis"),
                            
                            html.Div([
                                html.Div([
                                    dcc.Graph(id="accuracy-vs-params-scatter")
                                ], style={"width": "48%", "display": "inline-block"}),
                                
                                html.Div([
                                    dcc.Graph(id="accuracy-vs-flops-scatter")
                                ], style={"width": "48%", "display": "inline-block", "margin-left": "4%"}),
                            ]),
                            
                            html.Div([
                                html.Div([
                                    dcc.Graph(id="pareto-front-chart")
                                ], style={"width": "48%", "display": "inline-block"}),
                                
                                html.Div([
                                    dcc.Graph(id="operation-distribution-chart")
                                ], style={"width": "48%", "display": "inline-block", "margin-left": "4%"}),
                            ], style={"margin-top": "20px"}),
                            
                        ], className="metric-card"),
                        
                        html.Div([
                            html.H3("Statistical Analysis"),
                            dcc.Graph(id="performance-distribution-chart")
                        ], className="metric-card", style={"margin-top": "20px"}),
                        
                    ], style={"padding": "20px"})
                ]),
                
                # Search Landscape Tab
                dcc.Tab(label="🗺️ Search Landscape", value="landscape", children=[
                    html.Div([
                        html.Div([
                            html.H3("Search Space Exploration"),
                            
                            html.Div([
                                html.Div([
                                    html.H4("Search Progress"),
                                    dcc.Graph(id="search-landscape-3d")
                                ], style={"width": "48%", "display": "inline-block"}),
                                
                                html.Div([
                                    html.H4("Convergence Analysis"),
                                    dcc.Graph(id="convergence-chart")
                                ], style={"width": "48%", "display": "inline-block", "margin-left": "4%"}),
                            ]),
                            
                            html.Div([
                                html.H4("Strategy Comparison"),
                                dcc.Graph(id="strategy-performance-box")
                            ], style={"margin-top": "20px"})
                            
                        ], className="metric-card")
                    ], style={"padding": "20px"})
                ]),
                
                # Data Management Tab
                dcc.Tab(label="💾 Data", value="data", children=[
                    html.Div([
                        html.Div([
                            html.H3("Data Management"),
                            
                            html.Div([
                                html.Div([
                                    html.H4("Load Experiment"),
                                    dcc.Upload(
                                        id='upload-data',
                                        children=html.Div([
                                            'Drag and Drop or ',
                                            html.A('Select Files')
                                        ]),
                                        style={
                                            'width': '100%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px'
                                        },
                                        multiple=False
                                    ),
                                    html.Div(id='upload-status')
                                ], style={"width": "48%", "display": "inline-block"}),
                                
                                html.Div([
                                    html.H4("Export Results"),
                                    html.Button("📊 Export All Data", id="export-all-btn", 
                                              className="button", style={"margin": "5px", "width": "100%"}),
                                    html.Button("📈 Export Charts", id="export-charts-btn", 
                                              className="button", style={"margin": "5px", "width": "100%"}),
                                    html.Button("🏗️ Export Architectures", id="export-arch-btn", 
                                              className="button", style={"margin": "5px", "width": "100%"}),
                                    dcc.Download(id="download-all-data"),
                                    dcc.Download(id="download-charts"),
                                    dcc.Download(id="download-all-architectures")
                                ], style={"width": "48%", "display": "inline-block", "margin-left": "4%"}),
                            ]),
                            
                            html.Div([
                                html.H4("Experiment Summary"),
                                html.Div(id="experiment-summary")
                            ], style={"margin-top": "30px"})
                            
                        ], className="metric-card")
                    ], style={"padding": "20px"})
                ])
            ])
        ])
    
    def _setup_callbacks(self) -> None:
        """Setup all dashboard callbacks."""
        
        # Auto-refresh callback
        @self.app.callback(
            [Output('last-updated', 'children'),
             Output('best-accuracy', 'children'),
             Output('total-architectures', 'children'),
             Output('search-time', 'children'),
             Output('best-strategy', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_metrics(n):
            current_time = datetime.now().strftime("%H:%M:%S")
            
            if not self.performance_data:
                return current_time, "--", "--", "--", "--"
            
            best_acc = max(self.performance_data, key=lambda x: x.get('accuracy', 0))
            total_archs = len(self.performance_data)
            
            # Calculate total search time
            if self.search_progress:
                search_time = max(p.get('elapsed_time', 0) for p in self.search_progress)
                search_time_str = f"{search_time:.1f}s"
            else:
                search_time_str = "--"
            
            best_strategy = best_acc.get('strategy', '--')
            
            return (
                current_time,
                f"{best_acc.get('accuracy', 0):.3f}",
                str(total_archs),
                search_time_str,
                best_strategy
            )
        
        # Accuracy progress chart
        @self.app.callback(
            Output('accuracy-progress-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_accuracy_progress(n):
            if not self.search_progress:
                return self._empty_figure("No search progress data available")
            
            df = pd.DataFrame(self.search_progress)
            
            fig = px.line(
                df, 
                x='epoch', 
                y='best_accuracy',
                color='strategy' if 'strategy' in df.columns else None,
                title="Search Progress: Best Accuracy Over Time",
                template=self.theme
            )
            fig.update_layout(
                xaxis_title="Epoch",
                yaxis_title="Accuracy",
                hovermode='x unified'
            )
            return fig
        
        # Strategy comparison chart
        @self.app.callback(
            Output('strategy-comparison-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_strategy_comparison(n):
            if not self.performance_data:
                return self._empty_figure("No performance data available")
            
            df = pd.DataFrame(self.performance_data)
            if 'strategy' not in df.columns:
                return self._empty_figure("No strategy information available")
            
            fig = px.box(
                df,
                x='strategy',
                y='accuracy',
                title="Strategy Performance Comparison",
                template=self.theme
            )
            fig.update_layout(
                xaxis_title="Search Strategy",
                yaxis_title="Accuracy"
            )
            return fig
        
        # Architecture dropdown update
        @self.app.callback(
            Output('architecture-dropdown', 'options'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_architecture_dropdown(n):
            if not self.architecture_data:
                return []
            
            options = []
            for i, arch_data in enumerate(self.architecture_data):
                accuracy = arch_data.get('accuracy', 0)
                strategy = arch_data.get('strategy', 'unknown')
                options.append({
                    'label': f"Architecture {i+1} (acc: {accuracy:.3f}, {strategy})",
                    'value': i
                })
            
            return options
        
        # Architecture visualization
        @self.app.callback(
            [Output('architecture-graph', 'elements'),
             Output('architecture-details', 'children'),
             Output('architecture-metrics', 'children')],
            [Input('architecture-dropdown', 'value')]
        )
        def update_architecture_viz(selected_arch):
            if selected_arch is None or not self.architecture_data:
                return [], "No architecture selected", "No metrics available"
            
            arch_data = self.architecture_data[selected_arch]
            architecture = arch_data.get('architecture')
            
            if architecture is None:
                return [], "Architecture data not available", "No metrics available"
            
            # Create graph elements
            elements = self._create_architecture_graph(architecture)
            
            # Create details
            details = html.Div([
                html.P(f"Operations: {len(architecture.operations)}"),
                html.P(f"Search Space: {architecture.search_space.name if architecture.search_space else 'Unknown'}"),
                html.P(f"Method: {arch_data.get('strategy', 'Unknown')}"),
            ])
            
            # Create metrics
            metrics = html.Div([
                html.P(f"Accuracy: {arch_data.get('accuracy', 0):.3f}"),
                html.P(f"Parameters: {arch_data.get('parameters', 0):,}"),
                html.P(f"FLOPs: {arch_data.get('flops', 0):,}"),
            ])
            
            return elements, details, metrics
        
        # Additional callbacks for analysis charts
        self._setup_analysis_callbacks()
        self._setup_export_callbacks()
    
    def _setup_analysis_callbacks(self) -> None:
        """Setup callbacks for analysis charts."""
        
        @self.app.callback(
            Output('accuracy-vs-params-scatter', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_accuracy_vs_params(n):
            if not self.performance_data:
                return self._empty_figure("No performance data available")
            
            df = pd.DataFrame(self.performance_data)
            if 'parameters' not in df.columns:
                return self._empty_figure("No parameter data available")
            
            fig = px.scatter(
                df,
                x='parameters',
                y='accuracy',
                color='strategy' if 'strategy' in df.columns else None,
                title="Accuracy vs Model Parameters",
                template=self.theme
            )
            fig.update_layout(
                xaxis_title="Parameters",
                yaxis_title="Accuracy"
            )
            return fig
        
        @self.app.callback(
            Output('pareto-front-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_pareto_front(n):
            if not self.performance_data:
                return self._empty_figure("No performance data available")
            
            df = pd.DataFrame(self.performance_data)
            if 'parameters' not in df.columns or 'accuracy' not in df.columns:
                return self._empty_figure("Insufficient data for Pareto analysis")
            
            # Find Pareto front
            pareto_indices = self._find_pareto_front(df)
            
            fig = go.Figure()
            
            # All points
            fig.add_trace(go.Scatter(
                x=df['parameters'],
                y=df['accuracy'],
                mode='markers',
                name='All Architectures',
                marker=dict(color='lightblue', size=8),
                text=[f"Strategy: {s}" for s in df.get('strategy', ['Unknown'] * len(df))]
            ))
            
            # Pareto front
            if pareto_indices:
                pareto_df = df.iloc[pareto_indices]
                fig.add_trace(go.Scatter(
                    x=pareto_df['parameters'],
                    y=pareto_df['accuracy'],
                    mode='markers+lines',
                    name='Pareto Front',
                    marker=dict(color='red', size=12),
                    line=dict(color='red', dash='dash')
                ))
            
            fig.update_layout(
                title="Pareto Front: Accuracy vs Parameters",
                xaxis_title="Parameters",
                yaxis_title="Accuracy",
                template=self.theme
            )
            
            return fig
    
    def _setup_export_callbacks(self) -> None:
        """Setup callbacks for data export functionality."""
        
        @self.app.callback(
            Output('download-all-data', 'data'),
            [Input('export-all-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def export_all_data(n_clicks):
            if n_clicks is None:
                return dash.no_update
            
            export_data = {
                'performance_data': self.performance_data,
                'search_progress': self.search_progress,
                'architecture_data': [
                    {**arch, 'architecture': arch['architecture'].to_dict() if arch.get('architecture') else None}
                    for arch in self.architecture_data
                ],
                'export_timestamp': datetime.now().isoformat()
            }
            
            return dict(
                content=json.dumps(export_data, indent=2),
                filename=f"nanonas_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
    
    def _create_architecture_graph(self, architecture: Architecture) -> List[Dict]:
        """Create Cytoscape graph elements for architecture visualization."""
        elements = []
        
        # Add nodes for each operation
        for i, op in enumerate(architecture.operations):
            elements.append({
                'data': {
                    'id': f'op_{i}',
                    'label': f'Op {i}: {op}',
                    'type': 'operation'
                },
                'classes': 'architecture-node'
            })
        
        # Add edges (connections between operations)
        for i in range(len(architecture.operations) - 1):
            elements.append({
                'data': {
                    'source': f'op_{i}',
                    'target': f'op_{i+1}',
                    'id': f'edge_{i}'
                }
            })
        
        return elements
    
    def _find_pareto_front(self, df: pd.DataFrame) -> List[int]:
        """Find Pareto-optimal points in the dataset."""
        pareto_indices = []
        
        for i, row_i in df.iterrows():
            is_dominated = False
            
            for j, row_j in df.iterrows():
                if i == j:
                    continue
                
                # Check if point i is dominated by point j
                # (j has higher accuracy and lower parameters)
                if (row_j['accuracy'] >= row_i['accuracy'] and 
                    row_j['parameters'] <= row_i['parameters'] and
                    (row_j['accuracy'] > row_i['accuracy'] or row_j['parameters'] < row_i['parameters'])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def _empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            template=self.theme,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig
    
    def add_search_results(self, results: Dict[str, Any]) -> None:
        """Add search results to the dashboard."""
        experiment_id = results.get('experiment_id', f'exp_{len(self.search_results)}')
        self.search_results[experiment_id] = results
        self.current_experiment = experiment_id
        
        # Update data stores
        self._update_data_from_results(results)
        
        self.logger.info(f"Added search results for experiment: {experiment_id}")
    
    def _update_data_from_results(self, results: Dict[str, Any]) -> None:
        """Update internal data structures from search results."""
        
        # Update performance data
        if 'architectures' in results:
            for arch_data in results['architectures']:
                self.performance_data.append({
                    'accuracy': arch_data.get('accuracy', 0),
                    'parameters': arch_data.get('parameters', 0),
                    'flops': arch_data.get('flops', 0),
                    'strategy': arch_data.get('strategy', 'unknown')
                })
                
                self.architecture_data.append(arch_data)
        
        # Update search progress
        if 'search_progress' in results:
            self.search_progress.extend(results['search_progress'])
        
        # Limit data points for performance
        if len(self.performance_data) > self.max_data_points:
            self.performance_data = self.performance_data[-self.max_data_points:]
        
        if len(self.search_progress) > self.max_data_points:
            self.search_progress = self.search_progress[-self.max_data_points:]
    
    def run_server(self, host: str = '127.0.0.1', debug: bool = None, 
                   threaded: bool = True, **kwargs) -> None:
        """Run the dashboard server."""
        if debug is None:
            debug = self.debug
        
        self.logger.info(f"🚀 Starting NAS Dashboard on http://{host}:{self.port}")
        
        self.app.run_server(
            host=host,
            port=self.port,
            debug=debug,
            threaded=threaded,
            **kwargs
        )
    
    def stop_server(self) -> None:
        """Stop the dashboard server."""
        self.logger.info("🛑 Stopping NAS Dashboard")
        # Note: Dash doesn't have a built-in stop method
        # In production, this would be handled by the WSGI server
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.logger = logging.getLogger(f"{__name__}.NASDashboard")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)


def run_dashboard_server(results_file: Optional[str] = None, 
                        port: int = 8050, 
                        debug: bool = False) -> None:
    """
    Convenience function to run dashboard server.
    
    Args:
        results_file: Optional JSON file with search results to load
        port: Server port number
        debug: Enable debug mode
    """
    dashboard = NASDashboard(port=port, debug=debug)
    
    if results_file:
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            dashboard.add_search_results(results)
            print(f"✅ Loaded results from {results_file}")
        except Exception as e:
            print(f"❌ Failed to load results: {e}")
    
    try:
        dashboard.run_server()
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run nanoNAS Interactive Dashboard")
    parser.add_argument("--results", "-r", type=str, 
                       help="JSON file with search results to load")
    parser.add_argument("--port", "-p", type=int, default=8050,
                       help="Server port number")
    parser.add_argument("--debug", "-d", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    run_dashboard_server(
        results_file=args.results,
        port=args.port,
        debug=args.debug
    ) 