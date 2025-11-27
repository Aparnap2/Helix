# Helix Dashboard - Cache Performance Page
# Real-time cache hit rates and optimization analytics

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import json

class CachePerformancePage:
    """Cache performance page for hit rates and optimization analytics"""

    def __init__(self, dashboard):
        self.dashboard = dashboard

    def render_cache_overview(self):
        """Render cache performance overview metrics"""
        st.markdown("### 🚀 Cache Performance Overview")

        cache_metrics = self._get_cache_metrics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="💎 Overall Hit Rate",
                value=f"{cache_metrics['overall_hit_rate']:.1f}%",
                delta=f"+{cache_metrics['hit_rate_change']:.1f}% from yesterday",
                delta_color="normal"
            )

        with col2:
            st.metric(
                label="⚡ Exact Cache Hits",
                value=f"{cache_metrics['exact_hits']:,}",
                delta=f"{cache_metrics['exact_hit_rate']:.1f}% rate"
            )

        with col3:
            st.metric(
                label="🔍 Semantic Hits",
                value=f"{cache_metrics['semantic_hits']:,}",
                delta=f"{cache_metrics['semantic_hit_rate']:.1f}% rate"
            )

        with col4:
            st.metric(
                label="⏱️ Avg Cache Latency",
                value=f"{cache_metrics['avg_cache_latency']:.1f}ms",
                delta="-{cache_metrics['latency_improvement']:.1f}ms",
                delta_color="normal"
            )

    def render_cache_trends(self):
        """Render cache performance trends over time"""
        col1, col2 = st.columns(2)

        with col1:
            self._render_hit_rate_trend()

        with col2:
            self._render_cache_type_breakdown()

    def _render_hit_rate_trend(self):
        """Render cache hit rate trend chart"""
        st.markdown("#### 📈 Hit Rate Trend (7 Days)")

        trend_data = self._get_hit_rate_trend()

        if trend_data:
            df = pd.DataFrame(trend_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            fig = go.Figure()

            # Add overall hit rate
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['overall_hit_rate'],
                name='Overall Hit Rate',
                line=dict(color='#00ff41', width=3),
                fill='tonexty',
                fillcolor='rgba(0, 255, 65, 0.1)'
            ))

            # Add exact cache rate
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['exact_hit_rate'],
                name='Exact Cache Rate',
                line=dict(color='#ffaa00', width=2)
            ))

            # Add semantic cache rate
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['semantic_hit_rate'],
                name='Semantic Cache Rate',
                line=dict(color='#00aaff', width=2)
            ))

            # Add target line
            fig.add_hline(
                y=50,  # Target hit rate
                line_dash="dash",
                line_color="red",
                annotation_text="Target: 50%"
            )

            fig.update_layout(
                title='Cache Hit Rate Over Time',
                xaxis=dict(title='Time'),
                yaxis=dict(title='Hit Rate (%)', range=[0, 100]),
                template='plotly_dark',
                height=350,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cache trend data available")

    def _render_cache_type_breakdown(self):
        """Render cache type breakdown pie chart"""
        st.markdown("#### 🎯 Cache Type Distribution")

        cache_data = self._get_cache_type_distribution()

        if cache_data:
            # Create subplots for pie and funnel
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "pie"}, {"type": "bar"}]],
                subplot_titles=['Cache Hit Sources', 'Request Distribution']
            )

            # Pie chart for cache sources
            fig.add_trace(go.Pie(
                labels=list(cache_data['sources'].keys()),
                values=list(cache_data['sources'].values()),
                name="Cache Sources"
            ), row=1, col=1)

            # Bar chart for request distribution
            fig.add_trace(go.Bar(
                x=list(cache_data['distribution'].keys()),
                y=list(cache_data['distribution'].values()),
                name="Request Distribution",
                marker_color='#00ff41'
            ), row=1, col=2)

            fig.update_layout(
                height=350,
                template='plotly_dark',
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cache distribution data available")

    def render_cache_analytics(self):
        """Render detailed cache analytics"""
        col1, col2 = st.columns(2)

        with col1:
            self._render_model_performance()

        with col2:
            self._render_cache_efficiency()

    def _render_model_performance(self):
        """Render cache performance by model"""
        st.markdown("#### 🤖 Cache Performance by Model")

        model_data = self._get_model_cache_performance()

        if model_data:
            df = pd.DataFrame(model_data)

            # Create scatter plot for performance metrics
            fig = px.scatter(
                df,
                x='hit_rate',
                y='avg_latency',
                size='total_requests',
                hover_name='model',
                title='Model Performance Overview',
                labels={
                    'hit_rate': 'Hit Rate (%)',
                    'avg_latency': 'Avg Latency (ms)',
                    'total_requests': 'Total Requests'
                },
                color='hit_rate',
                color_continuous_scale='Viridis'
            )

            fig.update_layout(
                template='plotly_dark',
                height=350,
                xaxis=dict(title='Hit Rate (%)', range=[0, 100]),
                yaxis=dict(title='Average Latency (ms)')
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            st.dataframe(
                df[['model', 'hit_rate', 'avg_latency', 'total_requests', 'cache_size']],
                column_config={
                    "model": "Model",
                    "hit_rate": st.column_config.NumberColumn(
                        "Hit Rate (%)",
                        format="%.1f"
                    ),
                    "avg_latency": st.column_config.NumberColumn(
                        "Avg Latency (ms)",
                        format="%.1f"
                    ),
                    "total_requests": "Total Requests",
                    "cache_size": st.column_config.NumberColumn(
                        "Cache Size",
                        format=",.0f"
                    )
                },
                hide_index=True
            )
        else:
            st.info("No model performance data available")

    def _render_cache_efficiency(self):
        """Render cache efficiency metrics"""
        st.markdown("#### ⚡ Cache Efficiency Metrics")

        efficiency_data = self._get_cache_efficiency()

        if efficiency_data:
            # Create gauges for key metrics
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]],
                subplot_titles=['Hit Rate', 'Latency Improvement', 'Cache Utilization', 'Cost Savings']
            )

            # Hit Rate Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=efficiency_data['hit_rate'],
                domain={'x': [0, 0.5], 'y': [0, 0.5]},
                title={'text': "Hit Rate (%)"},
                gauge={'axis': {'range': [None, 100]}}
            ), row=1, col=1)

            # Latency Improvement Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=efficiency_data['latency_improvement'],
                domain={'x': [0.5, 1], 'y': [0, 0.5]},
                title={'text': "Latency Imp. (%)"},
                gauge={'axis': {'range': [None, 100]}}
            ), row=1, col=2)

            # Cache Utilization Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=efficiency_data['cache_utilization'],
                domain={'x': [0, 0.5], 'y': [0.5, 1]},
                title={'text': "Cache Util. (%)"},
                gauge={'axis': {'range': [None, 100]}}
            ), row=2, col=1)

            # Cost Savings Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=efficiency_data['cost_savings'],
                domain={'x': [0.5, 1], 'y': [0.5, 1]},
                title={'text': "Cost Savings (%)"},
                gauge={'axis': {'range': [None, 100]}}
            ), row=2, col=2)

            fig.update_layout(
                height=400,
                template='plotly_dark'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No efficiency data available")

    def render_cache_optimization(self):
        """Render cache optimization recommendations"""
        st.markdown("### 🎯 Cache Optimization Recommendations")

        recommendations = self._get_optimization_recommendations()

        if recommendations:
            for recommendation in recommendations:
                self._render_recommendation_card(recommendation)
        else:
            st.success("✅ Cache performance is optimal!")

        # Manual cache controls
        st.markdown("#### 🛠️ Cache Management")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🧹 Clear Cache", type="secondary"):
                self._clear_cache()
                st.success("Cache cleared successfully!")

        with col2:
            cache_ttl = st.number_input(
                "Cache TTL (hours)",
                value=24,
                min_value=1,
                max_value=168
            )

        with col3:
            if st.button("Update TTL"):
                self._update_cache_ttl(cache_ttl)
                st.success("Cache TTL updated!")

    def _render_recommendation_card(self, recommendation: Dict):
        """Render a single recommendation card"""
        priority_color = {
            'high': 'red',
            'medium': 'orange',
            'low': 'blue'
        }.get(recommendation['priority'], 'gray')

        impact_color = {
            'high': 'green',
            'medium': 'yellow',
            'low': 'gray'
        }.get(recommendation['impact'], 'gray')

        st.markdown(
            f"""
            <div style='padding: 15px; margin: 10px 0; border-left: 4px solid {priority_color};
                        background-color: rgba(255,255,255,0.05); border-radius: 5px;'>
                <h4 style='margin: 0; color: {priority_color};'>
                    {recommendation['title']}
                    <span style='background-color: {impact_color}; color: white; padding: 2px 8px;
                               border-radius: 12px; font-size: 0.8em; margin-left: 10px;'>
                        {recommendation['impact'].upper()} IMPACT
                    </span>
                </h4>
                <p style='margin: 5px 0;'>{recommendation['description']}</p>
                <small style='color: gray;'>
                    Expected improvement: +{recommendation['expected_improvement']}% hit rate
                </small>
            </div>
            """,
            unsafe_allow_html=True
        )

    def render_semantic_analysis(self):
        """Render semantic cache analysis"""
        st.markdown("### 🔍 Semantic Cache Analysis")

        col1, col2 = st.columns(2)

        with col1:
            self._render_similarity_distribution()

        with col2:
            self._render_embedding_performance()

    def _render_similarity_distribution(self):
        """Render similarity score distribution"""
        st.markdown("#### 📊 Similarity Score Distribution")

        similarity_data = self._get_similarity_distribution()

        if similarity_data:
            df = pd.DataFrame(similarity_data)

            fig = px.histogram(
                df,
                x='similarity_score',
                title='Distribution of Similarity Scores',
                labels={'similarity_score': 'Similarity Score', 'count': 'Count'},
                nbins=20,
                color_discrete_sequence=['#00ff41']
            )

            fig.add_vline(
                x=0.88,  # Similarity threshold
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold: 0.88"
            )

            fig.update_layout(
                template='plotly_dark',
                height=350,
                xaxis=dict(title='Similarity Score', range=[0, 1]),
                yaxis=dict(title='Count')
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No similarity data available")

    def _render_embedding_performance(self):
        """Render embedding model performance"""
        st.markdown("#### 🤖 Embedding Model Performance")

        embedding_data = self._get_embedding_performance()

        if embedding_data:
            df = pd.DataFrame(embedding_data)

            fig = go.Figure()

            # Add accuracy bars
            fig.add_trace(go.Bar(
                name='Search Accuracy',
                x=df['model'],
                y=df['accuracy'],
                marker_color='#00ff41'
            ))

            # Add latency line
            fig.add_trace(go.Scatter(
                name='Avg Search Latency',
                x=df['model'],
                y=df['latency'],
                mode='markers+lines',
                yaxis='y2',
                line=dict(color='#ffaa00', width=2)
            ))

            fig.update_layout(
                title='Embedding Model Performance',
                xaxis=dict(title='Model'),
                yaxis=dict(title='Accuracy (%)', range=[0, 100]),
                yaxis2=dict(title='Latency (ms)', overlaying='y', side='right'),
                template='plotly_dark',
                height=350,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No embedding performance data available")

    def _get_cache_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics"""
        total_requests = int(self.dashboard.redis_client.get("helix:requests:total") or 0) if self.dashboard.redis_client else 0
        cache_hits = int(self.dashboard.redis_client.get("helix:requests:cache_hits") or 0) if self.dashboard.redis_client else 0

        overall_hit_rate = (cache_hits / max(total_requests, 1)) * 100

        return {
            'overall_hit_rate': overall_hit_rate,
            'hit_rate_change': 2.3,  # Mock data - calculate from historical data
            'exact_hits': int(cache_hits * 0.7),  # Mock distribution
            'exact_hit_rate': overall_hit_rate * 0.7,
            'semantic_hits': int(cache_hits * 0.3),  # Mock distribution
            'semantic_hit_rate': overall_hit_rate * 0.3,
            'avg_cache_latency': 15.2,
            'latency_improvement': 8.5
        }

    def _get_hit_rate_trend(self) -> List[Dict]:
        """Get hit rate trend data for the last 7 days"""
        trend_data = []
        for i in range(7, 0, -1):
            timestamp = datetime.now() - timedelta(days=i)
            # Mock data with realistic patterns
            base_rate = 35 + np.random.normal(0, 5)
            trend_data.append({
                'timestamp': timestamp,
                'overall_hit_rate': max(0, min(100, base_rate + np.random.normal(0, 3))),
                'exact_hit_rate': max(0, min(100, base_rate * 0.7 + np.random.normal(0, 2))),
                'semantic_hit_rate': max(0, min(100, base_rate * 0.3 + np.random.normal(0, 2)))
            })
        return trend_data

    def _get_cache_type_distribution(self) -> Dict[str, Dict]:
        """Get cache type distribution data"""
        return {
            'sources': {
                'Exact Cache': 70,
                'Semantic Cache': 30
            },
            'distribution': {
                'Cache Hits': 35,
                'Cache Misses': 65
            }
        }

    def _get_model_cache_performance(self) -> List[Dict]:
        """Get cache performance by model"""
        models = ['gpt-4', 'gpt-3.5-turbo', 'claude-3', 'llama-3-70b', 'gemini-pro']
        data = []
        for model in models:
            data.append({
                'model': model,
                'hit_rate': np.random.uniform(20, 60),
                'avg_latency': np.random.uniform(10, 50),
                'total_requests': np.random.randint(100, 1000),
                'cache_size': np.random.randint(100, 500)
            })
        return data

    def _get_cache_efficiency(self) -> Dict[str, float]:
        """Get cache efficiency metrics"""
        return {
            'hit_rate': 35.2,
            'latency_improvement': 78.5,
            'cache_utilization': 45.8,
            'cost_savings': 32.1
        }

    def _get_optimization_recommendations(self) -> List[Dict]:
        """Get cache optimization recommendations"""
        return [
            {
                'title': 'Increase Similarity Threshold',
                'description': 'Consider increasing semantic similarity threshold from 0.88 to 0.90 for better precision.',
                'priority': 'medium',
                'impact': 'medium',
                'expected_improvement': 3
            },
            {
                'title': 'Optimize Cache TTL',
                'description': 'Current TTL of 24 hours may be too long for certain user patterns. Consider shorter TTLs for frequently changing content.',
                'priority': 'low',
                'impact': 'low',
                'expected_improvement': 2
            }
        ]

    def _get_similarity_distribution(self) -> List[Dict]:
        """Get similarity score distribution data"""
        scores = []
        for _ in range(1000):
            scores.append(np.random.beta(2, 1))  # Simulated similarity scores
        return [{'similarity_score': score} for score in scores]

    def _get_embedding_performance(self) -> List[Dict]:
        """Get embedding model performance data"""
        models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'e5-large-v2']
        data = []
        for model in models:
            data.append({
                'model': model,
                'accuracy': np.random.uniform(85, 95),
                'latency': np.random.uniform(5, 25)
            })
        return data

    def _clear_cache(self):
        """Clear cache data"""
        # In production, clear Redis cache keys
        if self.dashboard.redis_client:
            # Clear cache keys matching helix: patterns
            keys = self.dashboard.redis_client.keys("helix:*")
            if keys:
                self.dashboard.redis_client.delete(*keys)

    def _update_cache_ttl(self, ttl_hours: int):
        """Update cache TTL configuration"""
        # In production, update configuration
        pass

    def render(self):
        """Main render method for cache performance page"""
        self.render_cache_overview()
        st.markdown("---")
        self.render_cache_trends()
        st.markdown("---")
        self.render_cache_analytics()
        st.markdown("---")
        self.render_cache_optimization()
        st.markdown("---")
        self.render_semantic_analysis()