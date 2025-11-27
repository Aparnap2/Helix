# Helix Dashboard - Overview Page
# Main metrics and KPIs with real-time updates

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncio

class OverviewPage:
    """Overview page for main dashboard metrics"""

    def __init__(self, dashboard):
        self.dashboard = dashboard

    def render_key_metrics(self):
        """Render key overview metrics with enhanced styling"""
        st.markdown("### 🎯 Real-time Metrics")

        # Get metrics with caching
        metrics = self._get_cached_metrics()

        # Create metrics grid with custom styling
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            delta_color = "normal" if metrics['cost_savings']['total_savings'] > 0 else "inverse"
            st.metric(
                label="💰 Total Cost Savings",
                value=f"${metrics['cost_savings']['total_savings']:,.2f}",
                delta=f"{metrics['cost_savings']['cache_savings_rate']:.1f}% from cache",
                delta_color=delta_color,
                help="Total savings from cache hits and optimizations"
            )

        with col2:
            cache_color = "normal" if metrics['cache_hit_rate'] > 50 else "inverse"
            st.metric(
                label="🚀 Cache Hit Rate",
                value=f"{metrics['cache_hit_rate']:.1f}%",
                delta=f"{metrics['total_cache_hits']:,} hits",
                delta_color=cache_color,
                help="Percentage of requests served from cache"
            )

        with col3:
            latency_color = "normal" if metrics['latency']['p99'] < 1000 else "inverse"
            st.metric(
                label="⚡ P99 Latency",
                value=f"{metrics['latency']['p99']:.0f}ms",
                delta="-{metrics['latency_improvement']:.0f}ms from yesterday",
                delta_color=latency_color,
                help="99th percentile response time"
            )

        with col4:
            st.metric(
                label="📊 Total Requests",
                value=f"{metrics['total_requests']:,}",
                delta=f"+{metrics['requests_today']:,} today",
                help="Total number of requests processed"
            )

        # Additional metrics row
        st.markdown("### 📈 Performance Trends")

        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                label="🔒 PII Incidents",
                value=metrics['pii_incidents']['today'],
                delta=f"Last 7d: {metrics['pii_incidents']['week']}",
                delta_color="inverse" if metrics['pii_incidents']['today'] > 0 else "normal"
            )

        with col6:
            st.metric(
                label="💎 Uptime",
                value=f"{metrics['uptime']:.3f}%",
                delta="24h average",
                help="System uptime percentage"
            )

        with col7:
            st.metric(
                label="🔥 Active Users",
                value=metrics['active_users'],
                delta=f"+{metrics['new_users']} new",
                help="Number of active users"
            )

        with col8:
            st.metric(
                label="🎯 Success Rate",
                value=f"{metrics['success_rate']:.1f}%",
                delta="Last hour",
                help="Percentage of successful requests"
            )

    def render_realtime_charts(self):
        """Render real-time charts and visualizations"""
        col1, col2 = st.columns(2)

        with col1:
            self._render_request_volume_chart()

        with col2:
            self._render_cost_savings_chart()

        col3, col4 = st.columns(2)

        with col3:
            self._render_latency_chart()

        with col4:
            self._render_cache_performance_chart()

    def _render_request_volume_chart(self):
        """Render request volume over time"""
        st.markdown("#### 📊 Request Volume Trend")

        # Get time series data
        hourly_data = self._get_hourly_request_data()

        if hourly_data:
            df = pd.DataFrame(hourly_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            fig = px.line(
                df,
                x='timestamp',
                y='requests',
                title='Requests per Hour (Last 24h)',
                labels={'requests': 'Number of Requests', 'timestamp': 'Time'},
                line_shape='linear',
                color_discrete_sequence=['#00ff41']
            )

            fig.update_layout(
                height=300,
                showlegend=False,
                template='plotly_dark',
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                )
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No request volume data available")

    def _render_cost_savings_chart(self):
        """Render cost savings visualization"""
        st.markdown("#### 💰 Cost Breakdown")

        cost_data = self._get_cost_breakdown()

        if cost_data['total_savings'] > 0:
            fig = go.Figure(data=[
                go.Bar(
                    name='Actual Spend',
                    x=['Cost Analysis'],
                    y=[cost_data['actual_spend']],
                    marker_color='#ff6b6b'
                ),
                go.Bar(
                    name='Cache Savings',
                    x=['Cost Analysis'],
                    y=[cost_data['cache_savings']],
                    marker_color='#00ff41'
                )
            ])

            fig.update_layout(
                title='Daily Cost Analysis',
                barmode='stack',
                height=300,
                template='plotly_dark',
                yaxis=dict(title='Cost ($)')
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cost data available")

    def _render_latency_chart(self):
        """Render latency distribution chart"""
        st.markdown("#### ⚡ Latency Distribution")

        latency_data = self._get_latency_distribution()

        if latency_data:
            fig = go.Figure()

            # Add percentile lines
            percentiles = ['p50', 'p75', 'p90', 'p95', 'p99']
            colors = ['#00ff41', '#00ff41', '#ffaa00', '#ff6b6b', '#ff0000']

            for i, p in enumerate(percentiles):
                if p in latency_data:
                    fig.add_hline(
                        y=latency_data[p],
                        line_dash="dash",
                        line_color=colors[i],
                        annotation_text=f"{p.upper()}: {latency_data[p]:.0f}ms"
                    )

            fig.update_layout(
                title='Latency Percentiles',
                height=300,
                template='plotly_dark',
                yaxis=dict(title='Latency (ms)'),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No latency data available")

    def _render_cache_performance_chart(self):
        """Render cache performance gauge"""
        st.markdown("#### 🚀 Cache Performance")

        cache_data = self._get_cache_performance()

        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = cache_data['hit_rate'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Cache Hit Rate (%)"},
            delta = {'reference': cache_data['hit_rate_yesterday']},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        fig.update_layout(
            height=300,
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

    def _get_cached_metrics(self) -> Dict[str, Any]:
        """Get cached metrics data"""
        def fetch_metrics():
            cost_savings = self.dashboard.get_cost_savings()
            cache_hit_rate = self.dashboard.calculate_cache_hit_rate()
            latency_metrics = self.dashboard.get_latency_metrics()

            total_requests = int(self.dashboard.redis_client.get("helix:requests:total") or 0) if self.dashboard.redis_client else 0
            cache_hits = int(self.dashboard.redis_client.get("helix:requests:cache_hits") or 0) if self.dashboard.redis_client else 0

            return {
                'cost_savings': cost_savings,
                'cache_hit_rate': cache_hit_rate,
                'total_cache_hits': cache_hits,
                'latency': latency_metrics,
                'latency_improvement': 15,  # Mock data - calculate from historical data
                'total_requests': total_requests,
                'requests_today': int(total_requests * 0.1),  # Estimate
                'pii_incidents': self.dashboard.get_pii_incidents_summary(),
                'uptime': 99.9,  # Mock data - calculate from system health
                'active_users': 156,  # Mock data - get from user analytics
                'new_users': 12,
                'success_rate': 99.5
            }

        return self.dashboard._get_cached_data("overview_metrics", fetch_metrics, ttl=60)

    def _get_hourly_request_data(self) -> List[Dict]:
        """Get hourly request data for the last 24 hours"""
        # Mock data - in production, this would come from Redis time series
        hours = []
        for i in range(24):
            hour_time = datetime.now() - timedelta(hours=i)
            hours.append({
                'timestamp': hour_time,
                'requests': np.random.randint(50, 200)
            })
        return hours

    def _get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown data"""
        # Mock data - in production, this would come from Redis analytics
        return {
            'actual_spend': 125.50,
            'cache_savings': 45.30,
            'total_savings': 45.30
        }

    def _get_latency_distribution(self) -> Dict[str, float]:
        """Get latency distribution percentiles"""
        return self.dashboard.get_latency_metrics()

    def _get_cache_performance(self) -> Dict[str, float]:
        """Get cache performance metrics"""
        hit_rate = self.dashboard.calculate_cache_hit_rate()
        return {
            'hit_rate': hit_rate,
            'hit_rate_yesterday': max(0, hit_rate - 5)  # Mock improvement
        }

    def render_alerts_section(self):
        """Render alerts and notifications section"""
        st.markdown("### 🚨 Alerts & Notifications")

        alerts = self._get_active_alerts()

        if alerts:
            for alert in alerts[:5]:  # Show top 5 alerts
                severity_color = {
                    'critical': 'red',
                    'warning': 'orange',
                    'info': 'blue'
                }.get(alert['severity'], 'gray')

                st.markdown(
                    f"""
                    <div style='padding: 10px; margin: 5px 0; border-left: 4px solid {severity_color};
                                background-color: rgba(255,255,255,0.05); border-radius: 5px;'>
                        <strong>{alert['title']}</strong><br>
                        <small>{alert['message']}</small><br>
                        <small style='color: gray;'>{alert['timestamp']}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.success("✅ No active alerts - All systems operational")

    def _get_active_alerts(self) -> List[Dict]:
        """Get active alerts from system"""
        # Mock alerts - in production, this would come from alerting system
        return [
            {
                'title': 'High Latency Detected',
                'message': 'P99 latency exceeded 1000ms for the last 5 minutes',
                'severity': 'warning',
                'timestamp': '2 minutes ago'
            },
            {
                'title': 'Cache Hit Rate Below Target',
                'message': 'Current cache hit rate is 35%, below target of 50%',
                'severity': 'info',
                'timestamp': '15 minutes ago'
            }
        ]

    def render(self):
        """Main render method for the overview page"""
        # Auto-refresh logic
        if st.session_state.get('auto_refresh', True):
            time.sleep(st.session_state.get('refresh_interval', 5))
            st.rerun()

        # Render components
        self.render_key_metrics()
        st.markdown("---")
        self.render_realtime_charts()
        st.markdown("---")
        self.render_alerts_section()