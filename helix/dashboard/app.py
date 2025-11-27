"""
Helix AI Gateway Dashboard
Streamlit-based monitoring and management interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
import logging

from helix.core.config import get_config, HelixConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Helix AI Gateway Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    margin: 0.5rem 0;
}
.cache-hit {
    background: linear-gradient(135deg, #00c896 0%, #00a67e 100%);
}
.pii-detected {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
}
.cost-saved {
    background: linear-gradient(135deg, #4834d4 0%, #3742fa 100%);
}
.sidebar-section {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


class HelixDashboard:
    """Main Helix Dashboard application"""

    def __init__(self):
        self.config = get_config()
        self.api_base = os.getenv("HELIX_API_BASE", "http://localhost:8000")
        self.init_session_state()

    def init_session_state(self):
        """Initialize Streamlit session state"""
        if 'page' not in st.session_state:
            st.session_state.page = 'overview'
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 30
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True

    def run(self):
        """Run the dashboard application"""
        # Sidebar
        self.render_sidebar()

        # Header
        self.render_header()

        # Main content based on selected page
        if st.session_state.page == 'overview':
            self.render_overview_page()
        elif st.session_state.page == 'caching':
            self.render_caching_page()
        elif st.session_state.page == 'costs':
            self.render_costs_page()
        elif st.session_state.page == 'pii':
            self.render_pii_page()
        elif st.session_state.page == 'settings':
            self.render_settings_page()
        elif st.session_state.page == 'api':
            self.render_api_page()

        # Auto-refresh
        if st.session_state.auto_refresh:
            time.sleep(1)
            st.rerun()

    def render_sidebar(self):
        """Render the sidebar navigation"""
        with st.sidebar:
            st.title("🧬 Helix Dashboard")

            # Navigation
            st.header("Navigation")

            pages = [
                ("📊 Overview", "overview"),
                ("🔄 Caching", "caching"),
                ("💰 Costs", "costs"),
                ("🔒 PII", "pii"),
                ("⚙️ Settings", "settings"),
                ("🔌 API", "api")
            ]

            for page_name, page_key in pages:
                if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                    st.session_state.page = page_key
                    st.rerun()

            # Configuration Status
            st.header("Configuration Status")
            self.render_config_status()

            # Auto-refresh controls
            st.header("Auto Refresh")
            auto_refresh = st.checkbox("Enable Auto Refresh", value=st.session_state.auto_refresh)
            st.session_state.auto_refresh = auto_refresh

            if auto_refresh:
                refresh_interval = st.slider(
                    "Refresh Interval (seconds)",
                    min_value=5,
                    max_value=300,
                    value=st.session_state.refresh_interval,
                    step=5
                )
                st.session_state.refresh_interval = refresh_interval

            # Manual refresh
            if st.button("🔄 Refresh Now", use_container_width=True):
                st.session_state.last_refresh = datetime.now()
                st.rerun()

            # Last refresh info
            st.text(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

    def render_header(self):
        """Render the dashboard header"""
        st.title("🧬 Helix AI Gateway Dashboard")

        # Status indicators
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status_color = "🟢" if self.config.enabled else "🔴"
            st.metric("Helix Status", f"{status_color} {'Enabled' if self.config.enabled else 'Disabled'}")

        with col2:
            uptime = self.get_uptime()
            st.metric("Uptime", uptime)

        with col3:
            last_refresh = st.session_state.last_refresh.strftime('%H:%M:%S')
            st.metric("Last Refresh", last_refresh)

        with col4:
            current_time = datetime.now().strftime('%H:%M:%S')
            st.metric("Current Time", current_time)

    def render_config_status(self):
        """Render configuration status in sidebar"""
        config_data = {
            "Caching": self.config.caching.enabled,
            "PII Detection": self.config.pii.enabled,
            "Cost Tracking": self.config.cost.enabled,
            "Monitoring": self.config.monitoring.enabled
        }

        for feature, enabled in config_data.items():
            status = "✅" if enabled else "❌"
            st.write(f"{status} {feature}")

        # Additional config info
        if self.config.caching.enabled:
            cache_type = self.config.caching.cache_type
            st.write(f"🔄 Cache Type: {cache_type}")

        if self.config.pii.enabled:
            pii_mode = self.config.pii.mode
            st.write(f"🔒 PII Mode: {pii_mode}")

    def render_overview_page(self):
        """Render the overview page"""
        st.header("📊 System Overview")

        # Key metrics row
        self.render_key_metrics()

        # Charts section
        st.subheader("Performance Metrics")

        # Create two columns for charts
        col1, col2 = st.columns(2)

        with col1:
            self.render_request_volume_chart()

        with col2:
            self.render_success_rate_chart()

        # Second row of charts
        col1, col2 = st.columns(2)

        with col1:
            self.render_cache_performance_chart()

        with col2:
            self.render_cost_trend_chart()

        # Recent activity
        st.subheader("Recent Activity")
        self.render_recent_activity()

    def render_key_metrics(self):
        """Render key system metrics"""
        metrics = self.get_key_metrics()

        # Create metric cards in a grid
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Requests</h3>
                <h2>{}</h2>
                <p>Last 24 hours</p>
            </div>
            """.format(metrics.get('total_requests', 0)), unsafe_allow_html=True)

        with col2:
            cache_hit_rate = metrics.get('cache_hit_rate', 0)
            st.markdown(f"""
            <div class="metric-card cache-hit">
                <h3>Cache Hit Rate</h3>
                <h2>{cache_hit_rate:.1f}%</h2>
                <p>Last 24 hours</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            pii_detection_rate = metrics.get('pii_detection_rate', 0)
            st.markdown(f"""
            <div class="metric-card pii-detected">
                <h3>PII Detection Rate</h3>
                <h2>{pii_detection_rate:.1f}%</h2>
                <p>Last 24 hours</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            cost_savings = metrics.get('cost_savings', 0)
            st.markdown(f"""
            <div class="metric-card cost-saved">
                <h3>Cost Savings</h3>
                <h2>${cost_savings:.2f}</h2>
                <p>Last 24 hours</p>
            </div>
            """, unsafe_allow_html=True)

    def render_request_volume_chart(self):
        """Render request volume chart"""
        st.subheader("Request Volume")

        # Get request volume data
        data = self.get_request_volume_data()

        if data:
            df = pd.DataFrame(data)

            fig = px.line(
                df,
                x='timestamp',
                y='requests',
                title='Requests per Hour',
                labels={'timestamp': 'Time', 'requests': 'Number of Requests'}
            )

            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Requests",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No request volume data available")

    def render_success_rate_chart(self):
        """Render success rate chart"""
        st.subheader("Success Rate")

        # Get success rate data
        data = self.get_success_rate_data()

        if data:
            df = pd.DataFrame(data)

            fig = go.Figure()

            # Add success rate line
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['success_rate'],
                mode='lines+markers',
                name='Success Rate (%)',
                line=dict(color='green', width=2)
            ))

            # Add error rate line
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['error_rate'],
                mode='lines+markers',
                name='Error Rate (%)',
                line=dict(color='red', width=2)
            ))

            fig.update_layout(
                title='Success and Error Rates',
                xaxis_title='Time',
                yaxis_title='Rate (%)',
                height=400,
                yaxis=dict(range=[0, 100])
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No success rate data available")

    def render_cache_performance_chart(self):
        """Render cache performance chart"""
        st.subheader("Cache Performance")

        # Get cache performance data
        data = self.get_cache_performance_data()

        if data:
            df = pd.DataFrame(data)

            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Cache Hit Rate (%)', 'Cache Lookup Time (ms)'),
                vertical_spacing=0.1
            )

            # Cache hit rate
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['hit_rate'], name='Hit Rate'),
                row=1, col=1
            )

            # Cache lookup time
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['lookup_time'], name='Lookup Time'),
                row=2, col=1
            )

            fig.update_layout(
                height=600,
                title_text="Cache Performance Metrics"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cache performance data available")

    def render_cost_trend_chart(self):
        """Render cost trend chart"""
        st.subheader("Cost Trends")

        # Get cost data
        data = self.get_cost_data()

        if data:
            df = pd.DataFrame(data)

            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Daily Costs ($)', 'Cumulative Savings ($)'),
                vertical_spacing=0.1
            )

            # Daily costs
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['daily_cost'], name='Daily Cost'),
                row=1, col=1
            )

            # Cumulative savings
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['cumulative_savings'], name='Cumulative Savings'),
                row=2, col=1
            )

            fig.update_layout(
                height=600,
                title_text="Cost Analysis"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cost data available")

    def render_recent_activity(self):
        """Render recent activity table"""
        # Get recent activity data
        data = self.get_recent_activity()

        if data:
            df = pd.DataFrame(data)

            # Format timestamp
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Display table
            st.dataframe(
                df,
                column_config={
                    "timestamp": "Time",
                    "model": "Model",
                    "user": "User",
                    "cache_hit": st.column_config.CheckboxColumn("Cache Hit"),
                    "pii_detected": st.column_config.CheckboxColumn("PII Detected"),
                    "cost": st.column_config.NumberColumn("Cost ($)", format="$%.4f"),
                    "latency_ms": st.column_config.NumberColumn("Latency (ms)")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No recent activity data available")

    def render_caching_page(self):
        """Render the caching page"""
        st.header("🔄 Caching Analytics")

        if not self.config.caching.enabled:
            st.warning("Caching is not enabled in the current configuration")
            return

        # Cache metrics summary
        col1, col2, col3, col4 = st.columns(4)

        cache_metrics = self.get_cache_metrics()

        with col1:
            st.metric("Total Cache Hits", cache_metrics.get('total_hits', 0))

        with col2:
            st.metric("Total Cache Misses", cache_metrics.get('total_misses', 0))

        with col3:
            hit_rate = cache_metrics.get('hit_rate', 0)
            st.metric("Hit Rate", f"{hit_rate:.1f}%")

        with col4:
            total_savings = cache_metrics.get('total_savings', 0)
            st.metric("Total Savings", f"${total_savings:.2f}")

        # Cache performance charts
        col1, col2 = st.columns(2)

        with col1:
            self.render_cache_type_distribution()

        with col2:
            self.render_cache_efficiency_chart()

        # Cache entries table
        st.subheader("Cache Entries")
        self.render_cache_entries_table()

    def render_cache_type_distribution(self):
        """Render cache type distribution chart"""
        st.subheader("Cache Type Distribution")

        data = self.get_cache_type_data()

        if data:
            df = pd.DataFrame(data)

            fig = px.pie(
                df,
                values='count',
                names='cache_type',
                title='Cache Hits by Type'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cache type data available")

    def render_cache_efficiency_chart(self):
        """Render cache efficiency chart"""
        st.subheader("Cache Efficiency Over Time")

        data = self.get_cache_efficiency_data()

        if data:
            df = pd.DataFrame(data)

            fig = px.line(
                df,
                x='timestamp',
                y='efficiency',
                title='Cache Efficiency (%)',
                labels={'timestamp': 'Time', 'efficiency': 'Efficiency (%)'}
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cache efficiency data available")

    def render_cache_entries_table(self):
        """Render cache entries table"""
        data = self.get_cache_entries()

        if data:
            df = pd.DataFrame(data)

            # Format data
            if not df.empty:
                df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                df['hit_rate'] = df['hit_rate'].round(2)

            st.dataframe(
                df,
                column_config={
                    "cache_key": "Cache Key",
                    "cache_type": "Type",
                    "hit_count": "Hits",
                    "hit_rate": st.column_config.NumberColumn("Hit Rate (%)", format="%.2f"),
                    "created_at": "Created",
                    "last_access": "Last Access"
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No cache entries available")

    def render_costs_page(self):
        """Render the costs page"""
        st.header("💰 Cost Analysis")

        if not self.config.cost.enabled:
            st.warning("Cost tracking is not enabled in the current configuration")
            return

        # Cost summary metrics
        col1, col2, col3, col4 = st.columns(4)

        cost_metrics = self.get_cost_metrics()

        with col1:
            daily_spend = cost_metrics.get('daily_spend', 0)
            st.metric("Daily Spend", f"${daily_spend:.2f}")

        with col2:
            monthly_spend = cost_metrics.get('monthly_spend', 0)
            st.metric("Monthly Spend", f"${monthly_spend:.2f}")

        with col3:
            total_savings = cost_metrics.get('total_savings', 0)
            st.metric("Total Savings", f"${total_savings:.2f}")

        with col4:
            budget_usage = cost_metrics.get('budget_usage', 0)
            st.metric("Budget Usage", f"{budget_usage:.1f}%")

        # Cost breakdown charts
        col1, col2 = st.columns(2)

        with col1:
            self.render_cost_by_model_chart()

        with col2:
            self.render_cost_by_provider_chart()

        # Budget tracking
        st.subheader("Budget Tracking")
        self.render_budget_tracking()

    def render_cost_by_model_chart(self):
        """Render cost breakdown by model chart"""
        st.subheader("Cost by Model")

        data = self.get_cost_by_model_data()

        if data:
            df = pd.DataFrame(data)

            fig = px.bar(
                df,
                x='model',
                y='cost',
                title='Cost Distribution by Model',
                labels={'model': 'Model', 'cost': 'Cost ($)'}
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cost by model data available")

    def render_cost_by_provider_chart(self):
        """Render cost breakdown by provider chart"""
        st.subheader("Cost by Provider")

        data = self.get_cost_by_provider_data()

        if data:
            df = pd.DataFrame(data)

            fig = px.pie(
                df,
                values='cost',
                names='provider',
                title='Cost Distribution by Provider'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cost by provider data available")

    def render_budget_tracking(self):
        """Render budget tracking section"""
        # Budget progress bars
        budgets = self.get_budget_data()

        for budget in budgets:
            with st.expander(f"Budget: {budget['name']}"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    # Progress bar
                    st.progress(budget['usage_percentage'])
                    st.write(f"Usage: {budget['usage_percentage']:.1f}%")
                    st.write(f"Spent: ${budget['current_spend']:.2f} / ${budget['max_budget']:.2f}")

                with col2:
                    # Status indicator
                    if budget['usage_percentage'] >= 100:
                        st.error("Over Budget")
                    elif budget['usage_percentage'] >= 90:
                        st.warning("Critical")
                    elif budget['usage_percentage'] >= 70:
                        st.info("Warning")
                    else:
                        st.success("Healthy")

    def render_pii_page(self):
        """Render the PII page"""
        st.header("🔒 PII Detection & Redaction")

        if not self.config.pii.enabled:
            st.warning("PII detection is not enabled in the current configuration")
            return

        # PII summary metrics
        col1, col2, col3, col4 = st.columns(4)

        pii_metrics = self.get_pii_metrics()

        with col1:
            st.metric("Total Requests", pii_metrics.get('total_requests', 0))

        with col2:
            detected_count = pii_metrics.get('pii_detected_count', 0)
            st.metric("PII Detected", detected_count)

        with col3:
            detection_rate = pii_metrics.get('detection_rate', 0)
            st.metric("Detection Rate", f"{detection_rate:.1f}%")

        with col4:
            redaction_rate = pii_metrics.get('redaction_rate', 0)
            st.metric("Redaction Rate", f"{redaction_rate:.1f}%")

        # PII type distribution
        col1, col2 = st.columns(2)

        with col1:
            self.render_pii_type_distribution()

        with col2:
            self.render_pii_confidence_distribution()

        # Recent PII detections
        st.subheader("Recent PII Detections")
        self.render_pii_detections_table()

    def render_pii_type_distribution(self):
        """Render PII type distribution chart"""
        st.subheader("PII Type Distribution")

        data = self.get_pii_type_data()

        if data:
            df = pd.DataFrame(data)

            fig = px.bar(
                df,
                x='pii_type',
                y='count',
                title='PII Detections by Type',
                labels={'pii_type': 'PII Type', 'count': 'Count'}
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No PII type data available")

    def render_pii_confidence_distribution(self):
        """Render PII confidence distribution chart"""
        st.subheader("PII Confidence Distribution")

        data = self.get_pii_confidence_data()

        if data:
            df = pd.DataFrame(data)

            fig = px.histogram(
                df,
                x='confidence',
                title='PII Detection Confidence Scores',
                labels={'confidence': 'Confidence Score', 'count': 'Count'},
                nbins=20
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No PII confidence data available")

    def render_pii_detections_table(self):
        """Render PII detections table"""
        data = self.get_pii_detections()

        if data:
            df = pd.DataFrame(data)

            # Format data
            if not df.empty:
                df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                df['confidence_score'] = df['confidence_score'].round(2)

            st.dataframe(
                df,
                column_config={
                    "created_at": "Time",
                    "pii_type": "PII Type",
                    "confidence_score": st.column_config.NumberColumn("Confidence", format="%.2f"),
                    "redaction_applied": st.column_config.CheckboxColumn("Redacted"),
                    "processing_time_ms": st.column_config.NumberColumn("Processing Time (ms)")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No PII detections available")

    def render_settings_page(self):
        """Render the settings page"""
        st.header("⚙️ Helix Settings")

        # Current configuration display
        st.subheader("Current Configuration")

        config_summary = self.get_config_summary()

        # Display configuration in a structured way
        col1, col2 = st.columns(2)

        with col1:
            st.write("**General Settings**")
            st.json({
                "Helix Enabled": config_summary['enabled'],
                "Debug Mode": config_summary['debug']
            })

            st.write("**Caching Settings**")
            st.json({
                "Enabled": config_summary['caching']['enabled'],
                "Type": config_summary['caching']['cache_type'],
                "Vector Search": config_summary['caching']['vector_search']
            })

        with col2:
            st.write("**PII Settings**")
            st.json({
                "Enabled": config_summary['pii']['enabled'],
                "Mode": config_summary['pii']['mode'],
                "Recognizers": config_summary['pii']['recognizers_count']
            })

            st.write("**Cost Settings**")
            st.json({
                "Enabled": config_summary['cost']['enabled'],
                "Real-time Tracking": config_summary['cost']['real_time_tracking'],
                "Budget Management": config_summary['cost']['budget_management']
            })

        # Configuration management
        st.subheader("Configuration Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Reload Configuration", use_container_width=True):
                self.reload_configuration()
                st.success("Configuration reloaded successfully!")
                st.rerun()

        with col2:
            if st.button("📋 Export Configuration", use_container_width=True):
                config_json = json.dumps(config_summary, indent=2)
                st.download_button(
                    label="Download Configuration",
                    data=config_json,
                    file_name="helix_config.json",
                    mime="application/json"
                )

        # Health checks
        st.subheader("Health Checks")
        self.render_health_checks()

    def render_api_page(self):
        """Render the API documentation page"""
        st.header("🔌 API Documentation")

        # API endpoints
        st.subheader("Available Endpoints")

        endpoints = [
            {
                "Method": "GET",
                "Endpoint": "/health",
                "Description": "Health check endpoint",
                "Example": "curl http://localhost:8000/health"
            },
            {
                "Method": "GET",
                "Endpoint": "/metrics",
                "Description": "Prometheus metrics endpoint",
                "Example": "curl http://localhost:8000/metrics"
            },
            {
                "Method": "GET",
                "Endpoint": "/helix/config",
                "Description": "Get current Helix configuration",
                "Example": "curl http://localhost:8000/helix/config"
            },
            {
                "Method": "GET",
                "Endpoint": "/helix/cache/stats",
                "Description": "Get cache statistics",
                "Example": "curl http://localhost:8000/helix/cache/stats"
            },
            {
                "Method": "GET",
                "Endpoint": "/helix/cost/stats",
                "Description": "Get cost statistics",
                "Example": "curl http://localhost:8000/helix/cost/stats"
            },
            {
                "Method": "POST",
                "Endpoint": "/helix/cache/clear",
                "Description": "Clear cache (admin only)",
                "Example": "curl -X POST http://localhost:8000/helix/cache/clear"
            }
        ]

        df = pd.DataFrame(endpoints)
        st.dataframe(df, hide_index=True, use_container_width=True)

        # API usage examples
        st.subheader("Usage Examples")

        example_code = '''
# Get health status
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())

# Get cache statistics
response = requests.get("http://localhost:8000/helix/cache/stats")
print(response.json())

# Get cost statistics
response = requests.get("http://localhost:8000/helix/cost/stats")
print(response.json())
        '''

        st.code(example_code, language="python")

    def render_health_checks(self):
        """Render health checks section"""
        health_data = self.get_health_status()

        for component, status in health_data.items():
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                status_emoji = "🟢" if status['healthy'] else "🔴"
                st.write(f"{status_emoji} **{component.replace('_', ' ').title()}**")

            with col2:
                if status['healthy']:
                    st.success("Healthy")
                else:
                    st.error(f"Error: {status.get('error', 'Unknown error')}")

            with col3:
                if status['response_time']:
                    st.write(f"{status['response_time']}ms")

    # Data fetching methods (these would connect to your actual data sources)

    def get_key_metrics(self) -> Dict[str, Any]:
        """Get key system metrics"""
        # This would connect to your metrics database or API
        # For now, return mock data
        return {
            'total_requests': 15420,
            'cache_hit_rate': 67.5,
            'pii_detection_rate': 23.8,
            'cost_savings': 342.67
        }

    def get_request_volume_data(self) -> List[Dict]:
        """Get request volume data for the last 24 hours"""
        # Mock data - replace with actual API call
        data = []
        for i in range(24):
            timestamp = datetime.now() - timedelta(hours=i)
            requests = 500 + (i * 50) + (hash(str(i)) % 200)
            data.append({
                'timestamp': timestamp,
                'requests': requests
            })
        return list(reversed(data))

    def get_success_rate_data(self) -> List[Dict]:
        """Get success rate data"""
        # Mock data - replace with actual API call
        data = []
        for i in range(24):
            timestamp = datetime.now() - timedelta(hours=i)
            success_rate = 95 + (hash(str(i)) % 5)
            error_rate = 100 - success_rate
            data.append({
                'timestamp': timestamp,
                'success_rate': success_rate,
                'error_rate': error_rate
            })
        return list(reversed(data))

    def get_cache_performance_data(self) -> List[Dict]:
        """Get cache performance data"""
        # Mock data - replace with actual API call
        data = []
        for i in range(24):
            timestamp = datetime.now() - timedelta(hours=i)
            hit_rate = 60 + (hash(str(i)) % 30)
            lookup_time = 10 + (hash(str(i)) % 40)
            data.append({
                'timestamp': timestamp,
                'hit_rate': hit_rate,
                'lookup_time': lookup_time
            })
        return list(reversed(data))

    def get_cost_data(self) -> List[Dict]:
        """Get cost data"""
        # Mock data - replace with actual API call
        data = []
        cumulative_savings = 0
        for i in range(30):
            timestamp = datetime.now() - timedelta(days=i)
            daily_cost = 50 + (hash(str(i)) % 100)
            savings = 10 + (hash(str(i*2)) % 20)
            cumulative_savings += savings
            data.append({
                'timestamp': timestamp,
                'daily_cost': daily_cost,
                'cumulative_savings': cumulative_savings
            })
        return list(reversed(data))

    def get_recent_activity(self) -> List[Dict]:
        """Get recent activity data"""
        # Mock data - replace with actual API call
        data = []
        for i in range(10):
            timestamp = datetime.now() - timedelta(minutes=i*5)
            data.append({
                'timestamp': timestamp,
                'model': ['gpt-4', 'claude-3-sonnet', 'gemini-pro'][i % 3],
                'user': f'user_{i}',
                'cache_hit': i % 3 != 0,
                'pii_detected': i % 5 == 0,
                'cost': 0.001 + (i * 0.0001),
                'latency_ms': 1200 + (i * 100)
            })
        return data

    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        return {
            'total_hits': 8423,
            'total_misses': 4051,
            'hit_rate': 67.5,
            'total_savings': 342.67
        }

    def get_cache_type_data(self) -> List[Dict]:
        """Get cache type distribution data"""
        return [
            {'cache_type': 'Exact Match', 'count': 5201},
            {'cache_type': 'Semantic Match', 'count': 2176},
            {'cache_type': 'Hybrid', 'count': 1046}
        ]

    def get_cache_efficiency_data(self) -> List[Dict]:
        """Get cache efficiency data"""
        data = []
        for i in range(24):
            timestamp = datetime.now() - timedelta(hours=i)
            efficiency = 60 + (hash(str(i)) % 30)
            data.append({
                'timestamp': timestamp,
                'efficiency': efficiency
            })
        return list(reversed(data))

    def get_cache_entries(self) -> List[Dict]:
        """Get cache entries"""
        # Mock data - replace with actual API call
        return [
            {
                'cache_key': 'helix:abc123',
                'cache_type': 'Semantic',
                'hit_count': 42,
                'hit_rate': 78.5,
                'created_at': datetime.now() - timedelta(hours=2),
                'last_access': datetime.now() - timedelta(minutes=15)
            },
            {
                'cache_key': 'helix:def456',
                'cache_type': 'Exact',
                'hit_count': 128,
                'hit_rate': 92.1,
                'created_at': datetime.now() - timedelta(hours=5),
                'last_access': datetime.now() - timedelta(minutes=3)
            }
        ]

    def get_cost_metrics(self) -> Dict[str, Any]:
        """Get cost metrics"""
        return {
            'daily_spend': 87.43,
            'monthly_spend': 2543.12,
            'total_savings': 342.67,
            'budget_usage': 58.2
        }

    def get_cost_by_model_data(self) -> List[Dict]:
        """Get cost breakdown by model"""
        return [
            {'model': 'gpt-4', 'cost': 1234.56},
            {'model': 'claude-3-sonnet', 'cost': 876.32},
            {'model': 'gemini-pro', 'cost': 432.24}
        ]

    def get_cost_by_provider_data(self) -> List[Dict]:
        """Get cost breakdown by provider"""
        return [
            {'provider': 'OpenAI', 'cost': 1567.89},
            {'provider': 'Anthropic', 'cost': 876.32},
            {'provider': 'Google', 'cost': 98.91}
        ]

    def get_budget_data(self) -> List[Dict]:
        """Get budget data"""
        return [
            {
                'name': 'Daily Budget',
                'current_spend': 87.43,
                'max_budget': 150.0,
                'usage_percentage': 58.2
            },
            {
                'name': 'Monthly Budget',
                'current_spend': 2543.12,
                'max_budget': 5000.0,
                'usage_percentage': 50.9
            }
        ]

    def get_pii_metrics(self) -> Dict[str, Any]:
        """Get PII metrics"""
        return {
            'total_requests': 15420,
            'pii_detected_count': 3668,
            'detection_rate': 23.8,
            'redaction_rate': 95.2
        }

    def get_pii_type_data(self) -> List[Dict]:
        """Get PII type distribution data"""
        return [
            {'pii_type': 'EMAIL_ADDRESS', 'count': 1234},
            {'pii_type': 'PHONE_NUMBER', 'count': 892},
            {'pii_type': 'US_SSN', 'count': 234},
            {'pii_type': 'CREDIT_CARD', 'count': 567},
            {'pii_type': 'URL', 'count': 741}
        ]

    def get_pii_confidence_data(self) -> List[Dict]:
        """Get PII confidence distribution data"""
        data = []
        for i in range(100):
            confidence = 0.5 + (hash(str(i)) % 50) / 100
            data.append({'confidence': confidence})
        return data

    def get_pii_detections(self) -> List[Dict]:
        """Get recent PII detections"""
        # Mock data - replace with actual API call
        return [
            {
                'created_at': datetime.now() - timedelta(minutes=5),
                'pii_type': 'EMAIL_ADDRESS',
                'confidence_score': 0.95,
                'redaction_applied': True,
                'processing_time_ms': 23
            },
            {
                'created_at': datetime.now() - timedelta(minutes=12),
                'pii_type': 'PHONE_NUMBER',
                'confidence_score': 0.87,
                'redaction_applied': True,
                'processing_time_ms': 18
            }
        ]

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'enabled': self.config.enabled,
            'debug': self.config.general.debug,
            'caching': {
                'enabled': self.config.caching.enabled,
                'cache_type': self.config.caching.cache_type,
                'vector_search': self.config.caching.vector_search.enabled
            },
            'pii': {
                'enabled': self.config.pii.enabled,
                'mode': self.config.pii.mode,
                'recognizers_count': len(self.config.pii.presidio.recognizers)
            },
            'cost': {
                'enabled': self.config.cost.enabled,
                'real_time_tracking': self.config.cost.real_time_tracking,
                'budget_management': self.config.cost.budget_management.enabled
            }
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all components"""
        # Mock data - replace with actual health checks
        return {
            'redis': {'healthy': True, 'response_time': 5},
            'database': {'healthy': True, 'response_time': 12},
            'pii_processor': {'healthy': True, 'response_time': 23},
            'cache': {'healthy': True, 'response_time': 8},
            'cost_tracker': {'healthy': False, 'error': 'Database connection timeout', 'response_time': None}
        }

    def get_uptime(self) -> str:
        """Get system uptime"""
        # Mock uptime - replace with actual uptime calculation
        return "3d 14h 27m"

    def reload_configuration(self):
        """Reload Helix configuration"""
        try:
            # This would call the actual configuration reload function
            from helix.core.config import reload_config
            reload_config()
            st.success("Configuration reloaded successfully!")
        except Exception as e:
            st.error(f"Failed to reload configuration: {e}")


def main():
    """Main function to run the dashboard"""
    dashboard = HelixDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()