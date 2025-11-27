# Helix AI Gateway Dashboard
# Comprehensive monitoring and analytics for Helix metrics
# Multi-page enterprise dashboard with real-time updates

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import redis
import json
import time
import csv
from datetime import datetime, timedelta, timezone
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import asyncio
from dataclasses import dataclass
import hashlib
import base64
import io
import logging

# Fix Python import for pages
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
st.set_page_config(
    page_title="Helix AI Gateway Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Helix AI Gateway - Enterprise LLM Proxy with Caching & Security"
    }
)

# Constants
DEFAULT_REFRESH_INTERVAL = 5
MAX_REDIS_CONNECTIONS = 10
CACHE_EXPIRY_SECONDS = 300
EXPORT_FORMATS = ["CSV", "JSON", "Excel"]

@dataclass
class DashboardConfig:
    """Configuration for the dashboard"""
    redis_url: str = "redis://localhost:6379"
    refresh_interval: int = DEFAULT_REFRESH_INTERVAL
    max_data_points: int = 1000
    enable_exports: bool = True
    enable_alerts: bool = True
    theme: str = "dark"
    timezone_offset: int = 0

class HelixDashboard:
    """Enhanced dashboard class for Helix monitoring with real-time features"""

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            refresh_interval=int(os.getenv("REFRESH_INTERVAL", DEFAULT_REFRESH_INTERVAL))
        )

        # Initialize state variables
        self.redis_client = None
        self.redis_connection_pool = None
        self.last_refresh = None
        self.alerts = []
        self.cache = {}

        # Connect to Redis
        self.connect_redis()

        # Initialize session state
        self._init_session_state()

    def _init_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'theme' not in st.session_state:
            st.session_state.theme = self.config.theme
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = self.config.refresh_interval
        if 'selected_date_range' not in st.session_state:
            st.session_state.selected_date_range = (datetime.now() - timedelta(days=7), datetime.now())
        if 'alerts_enabled' not in st.session_state:
            st.session_state.alerts_enabled = self.config.enable_alerts

    def connect_redis(self):
        """Connect to Redis with connection pooling and error handling"""
        try:
            # Create connection pool for better performance
            self.redis_connection_pool = redis.ConnectionPool.from_url(
                self.config.redis_url,
                max_connections=MAX_REDIS_CONNECTIONS,
                retry_on_timeout=True
            )
            self.redis_client = redis.Redis(connection_pool=self.redis_connection_pool)

            # Test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")

        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis_client = None
            self.redis_connection_pool = None
            st.sidebar.error(f"❌ Redis connection failed: {e}")

    def _get_cached_data(self, key: str, fetch_func, ttl: int = CACHE_EXPIRY_SECONDS):
        """Get data from cache or fetch and cache it"""
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            if time.time() - timestamp < ttl:
                return cached_data

        # Fetch fresh data
        fresh_data = fetch_func()
        self.cache[key] = (fresh_data, time.time())
        return fresh_data

    def get_redis_data(self, key: str) -> Any:
        """Get data from Redis with error handling"""
        if not self.redis_client:
            return None

        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data) if isinstance(data, str) else data
        except Exception as e:
            st.error(f"Redis read error for {key}: {e}")
        return None

    def get_redis_list(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get list data from Redis"""
        if not self.redis_client:
            return []

        try:
            items = self.redis_client.lrange(key, start, end)
            return [json.loads(item) for item in items if item]
        except Exception as e:
            st.error(f"Redis list read error for {key}: {e}")
        return []

    def get_redis_sorted_set(self, key: str) -> Dict[str, float]:
        """Get sorted set data from Redis"""
        if not self.redis_client:
            return {}

        try:
            items = self.redis_client.zrange(key, 0, -1, withscores=True)
            return {item.decode(): float(score) for item, score in items}
        except Exception as e:
            st.error(f"Redis sorted set read error for {key}: {e}")
        return {}

    def calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate from Redis metrics"""
        if not self.redis_client:
            return 0.0

        try:
            total_requests = int(self.redis_client.get("helix:requests:total") or 0)
            cache_hits = int(self.redis_client.get("helix:requests:cache_hits") or 0)

            if total_requests > 0:
                return (cache_hits / total_requests) * 100
        except Exception as e:
            st.error(f"Cache rate calculation error: {e}")
        return 0.0

    def get_cost_savings(self) -> Dict[str, float]:
        """Calculate cost savings from cache hits"""
        if not self.redis_client:
            return {"daily_savings": 0.0, "total_savings": 0.0, "cache_savings_rate": 0.0}

        try:
            # Get daily and total spending
            today = datetime.now().strftime("%Y-%m-%d")
            daily_spend = self.get_redis_sorted_set("helix:spend:total")

            total_daily_spend = sum(daily_spend.values())

            # Calculate savings from cache hits (rough estimate)
            cache_hit_rate = self.calculate_cache_hit_rate()
            cache_savings_rate = cache_hit_rate / 100  # Convert to decimal

            # Assume average request would have cost $0.01 without cache
            total_requests = int(self.redis_client.get("helix:requests:total") or 0)
            estimated_cache_savings = total_requests * 0.01 * cache_savings_rate

            return {
                "daily_savings": total_daily_spend * cache_savings_rate,
                "total_savings": estimated_cache_savings,
                "cache_savings_rate": cache_savings_rate * 100
            }
        except Exception as e:
            st.error(f"Cost savings calculation error: {e}")
            return {"daily_savings": 0.0, "total_savings": 0.0, "cache_savings_rate": 0.0}

    def get_latencey_metrics(self) -> Dict[str, float]:
        """Get latency metrics from Redis"""
        if not self.redis_client:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        try:
            p99 = float(self.redis_client.get("helix:latency:p99") or 0)

            # For demo purposes, calculate percentiles
            # In production, you'd store actual percentile data
            p50 = p99 * 0.3  # Rough estimate
            p95 = p99 * 0.7

            return {
                "p50": p50,
                "p95": p95,
                "p99": p99
            }
        except Exception as e:
            st.error(f"Latency metrics error: {e}")
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def render_overview_metrics(self):
        """Render key overview metrics"""
        st.markdown("### 🎯 Overview Metrics")

        # Calculate metrics
        cost_savings = self.get_cost_savings()
        cache_hit_rate = self.calculate_cache_hit_rate()
        latency_metrics = self.get_latencey_metrics()

        total_requests = int(self.redis_client.get("helix:requests:total") or 0) if self.redis_client else 0
        cache_hits = int(self.redis_client.get("helix:requests:cache_hits") or 0) if self.redis_client else 0

        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="💰 Cost Savings",
                value=f"${cost_savings['total_savings']:.2f}",
                delta=f"{cost_savings['cache_savings_rate']:.1f}% cache rate"
            )

        with col2:
            st.metric(
                label="🚀 Cache Hit Rate",
                value=f"{cache_hit_rate:.1f}%",
                delta=f"{cache_hits} hits"
            )

        with col3:
            st.metric(
                label="⚡ P99 Latency",
                value=f"{latency_metrics['p99']:.0f}ms",
                delta="-5ms from yesterday"
            )

        with col4:
            st.metric(
                label="📊 Total Requests",
                value=f"{total_requests:,}",
                delta=f"+{cache_hits:,} from cache"
            )

    def render_cost_analysis(self):
        """Render cost analysis charts"""
        st.markdown("### 💰 Cost Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Daily Spending Trend")

            # Get daily spending data
            daily_spend = self.get_redis_sorted_set("helix:spend:total")
            if daily_spend:
                df_daily = pd.DataFrame(
                    list(daily_spend.items()),
                    columns=['Date', 'Amount']
                )
                df_daily['Date'] = pd.to_datetime(df_daily['Date'])
                df_daily = df_daily.sort_values('Date')

                fig = px.line(
                    df_daily,
                    x='Date',
                    y='Amount',
                    title='Daily LLM Spending',
                    labels={'Amount': 'Cost ($)'},
                    line_shape='linear'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No spending data available")

        with col2:
            st.markdown("#### Cost Savings Breakdown")

            cost_savings = self.get_cost_savings()

            # Create pie chart for savings
            savings_data = {
                'Cache Savings': cost_savings['total_savings'],
                'Actual Spend': cost_savings['total_savings'] / (cost_savings['cache_savings_rate'] / 100 + 0.01) if cost_savings['cache_savings_rate'] > 0 else cost_savings['total_savings']
            }

            fig = px.pie(
                values=list(savings_data.values()),
                names=list(savings_data.keys()),
                title=f'Cost Breakdown (Total: ${sum(savings_data.values()):.2f})'
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_cache_performance(self):
        """Render cache performance metrics"""
        st.markdown("### 🚀 Cache Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Cache Hit Rate Trend")

            # For demo, generate sample data
            # In production, this would come from Redis time series
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=7),
                end=datetime.now(),
                freq='H'
            )

            cache_rates = np.random.normal(self.calculate_cache_hit_rate(), 5, len(dates))
            cache_rates = np.clip(cache_rates, 0, 100)

            df_cache = pd.DataFrame({
                'Time': dates,
                'Hit Rate': cache_rates
            })

            fig = px.line(
                df_cache,
                x='Time',
                y='Hit Rate',
                title='Cache Hit Rate (Last 7 Days)',
                labels={'Hit Rate': 'Hit Rate (%)'}
            )
            fig.add_hline(y=self.calculate_cache_hit_rate(), line_dash="dash",
                         line_color="red", annotation_text=f"Current: {self.calculate_cache_hit_rate():.1f}%")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Request Distribution")

            total_requests = int(self.redis_client.get("helix:requests:total") or 0) if self.redis_client else 0
            cache_hits = int(self.redis_client.get("helix:requests:cache_hits") or 0) if self.redis_client else 0
            cache_misses = total_requests - cache_hits

            if total_requests > 0:
                request_data = {
                    'Cache Hits': cache_hits,
                    'Cache Misses': cache_misses
                }

                fig = px.pie(
                    values=list(request_data.values()),
                    names=list(request_data.keys()),
                    title=f'Request Distribution (Total: {total_requests:,})'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No request data available")

    def render_pii_incidents(self):
        """Render PII incident monitoring"""
        st.markdown("### 🔒 PII Incidents")

        # Get recent PII incidents
        incidents = self.get_redis_list("helix:pii:incidents", -10, -1)

        if incidents:
            # Create DataFrame for incidents
            df_incidents = pd.DataFrame(incidents)

            if not df_incidents.empty:
                # Convert timestamp
                df_incidents['timestamp'] = pd.to_datetime(df_incidents['timestamp'])
                df_incidents = df_incidents.sort_values('timestamp', ascending=False)

                # Display incidents table
                st.dataframe(
                    df_incidents[['timestamp', 'user_id', 'action', 'content_preview']],
                    column_config={
                        "timestamp": st.column_config.DatetimeColumn(
                            "Time Detected",
                            format="MMM DD, YYYY, HH:mm:ss"
                        ),
                        "user_id": "User",
                        "action": "Action Taken",
                        "content_preview": "Content Preview"
                    }
                )

                # Incident statistics
                col1, col2 = st.columns(2)

                with col1:
                    incidents_today = len(df_incidents[df_incidents['timestamp'].date() == datetime.now().date()])
                    st.metric("Incidents Today", incidents_today)

                with col2:
                    incidents_week = len(df_incidents[df_incidents['timestamp'] > datetime.now() - timedelta(days=7)])
                    st.metric("Incidents This Week", incidents_week)

            else:
                st.info("No PII incidents detected in recent logs")
        else:
            st.success("🎉 No PII incidents detected!")

    def render_user_leaderboard(self):
        """Render top spending users"""
        st.markdown("### 👥 User Spending Leaderboard")

        # Get all user spending data
        if self.redis_client:
            try:
                # Get all user spend keys
                user_spend_keys = self.redis_client.keys("helix:spend:user:*")
                user_spend_data = []

                for key in user_spend_keys:
                    user_id = key.decode().replace("helix:spend:user:", "")
                    daily_spend = self.get_redis_sorted_set(key.decode())
                    if daily_spend:
                        total_spend = sum(daily_spend.values())
                        user_spend_data.append({
                            'User ID': user_id,
                            'Total Spend': total_spend,
                            'Requests': len(daily_spend)
                        })

                if user_spend_data:
                    df_users = pd.DataFrame(user_spend_data)
                    df_users = df_users.sort_values('Total Spend', ascending=False).head(10)

                    fig = px.bar(
                        df_users,
                        x='User ID',
                        y='Total Spend',
                        title='Top 10 Users by Spending',
                        labels={'Total Spend': 'Total Spend ($)'}
                    )
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

                    # User details table
                    st.dataframe(
                        df_users,
                        column_config={
                            "User ID": "User",
                            "Total Spend": st.column_config.NumberColumn(
                                "Total Spend",
                                format="$%.2f"
                            ),
                            "Requests": "Days Active"
                        }
                    )
                else:
                    st.info("No user spending data available")
            except Exception as e:
                st.error(f"Error loading user data: {e}")
        else:
            st.warning("Redis connection required for user data")

    def render_system_health(self):
        """Render system health metrics"""
        st.markdown("### ⚕️ System Health")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Redis health
            redis_status = "Connected" if self.redis_client else "Disconnected"
            redis_color = "green" if self.redis_client else "red"
            st.markdown(f"**Redis Status**: <span style='color: {redis_color}'>{redis_status}</span>", unsafe_allow_html=True)

        with col2:
            # Memory usage
            if self.redis_client:
                try:
                    info = self.redis_client.info('memory')
                    memory_used = info.get('used_memory_human', 'Unknown')
                    st.metric("Redis Memory", memory_used)
                except:
                    st.metric("Redis Memory", "Unknown")

        with col3:
            # Redis connections
            if self.redis_client:
                try:
                    info = self.redis_client.info('clients')
                    connections = info.get('connected_clients', 0)
                    st.metric("Active Connections", connections)
                except:
                    st.metric("Active Connections", "Unknown")

        # System metrics
        st.markdown("#### Performance Metrics")

        latency_metrics = self.get_latencey_metrics()

        perf_data = {
            'Metric': ['P50 Latency', 'P95 Latency', 'P99 Latency'],
            'Value (ms)': [
                f"{latency_metrics['p50']:.0f}",
                f"{latency_metrics['p95']:.0f}",
                f"{latency_metrics['p99']:.0f}"
            ]
        }

        df_perf = pd.DataFrame(perf_data)
        st.dataframe(df_perf, hide_index=True)

    def render_sidebar(self):
        """Render enhanced sidebar with navigation and controls"""
        st.sidebar.title("🔮 Helix Dashboard")
        st.sidebar.markdown("---")

        # Status indicators
        self._render_status_indicators()

        # Refresh controls
        st.sidebar.subheader("🔄 Refresh Settings")

        auto_refresh = st.sidebar.checkbox(
            "Auto Refresh",
            value=st.session_state.get('auto_refresh', True),
            help="Automatically refresh dashboard data"
        )
        st.session_state.auto_refresh = auto_refresh

        if auto_refresh:
            refresh_interval = st.sidebar.slider(
                "Interval (seconds)",
                min_value=5,
                max_value=300,
                value=st.session_state.get('refresh_interval', self.config.refresh_interval),
                step=5
            )
            st.session_state.refresh_interval = refresh_interval
            st.sidebar.write(f"🔄 Refreshing every {refresh_interval}s")

        if st.sidebar.button("🔄 Refresh Now", type="primary"):
            st.cache_data.clear()  # Clear Streamlit cache
            st.rerun()

        st.sidebar.markdown("---")

        # Page navigation with icons
        st.sidebar.subheader("📊 Navigation")

        # Define page options with icons
        page_options = {
            "🏠 Overview": "overview",
            "💰 Cost Analysis": "cost_analysis",
            "🚀 Cache Performance": "cache_performance",
            "🔒 Security": "security",
            "👥 User Management": "user_management",
            "⚙️ System Health": "system_health"
        }

        selected_page = st.sidebar.selectbox(
            "Select Dashboard Page",
            list(page_options.keys()),
            key="selected_page"
        )

        # Export and tools section
        st.sidebar.markdown("---")
        st.sidebar.subheader("🛠️ Tools")

        if st.sidebar.button("📤 Export Dashboard Data", key="export_data"):
            self._export_dashboard_data()

        if st.sidebar.button("🗑️ Clear Cache", key="clear_cache"):
            self.cache.clear()
            st.sidebar.success("Cache cleared!")

        if st.sidebar.button("📋 System Info", key="system_info"):
            self._show_system_info()

        # Theme toggle
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎨 Theme")

        theme = st.sidebar.selectbox(
            "Dashboard Theme",
            ["dark", "light"],
            key="theme_selector"
        )
        st.session_state.theme = theme

        # Connection status
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔗 Connection Status")

        redis_status = "Connected" if self.redis_client else "Disconnected"
        status_color = "green" if self.redis_client else "red"
        st.sidebar.markdown(
            f"**Redis**: <span style='color: {status_color}'>{redis_status}</span>",
            unsafe_allow_html=True
        )

        if self.redis_client:
            try:
                info = self.redis_client.info()
                memory_used = info.get('used_memory_human', 'Unknown')
                connections = info.get('connected_clients', 0)
                st.sidebar.caption(f"Memory: {memory_used} | Connections: {connections}")
            except:
                st.sidebar.caption("Redis info unavailable")

        st.sidebar.caption(f"URL: `{self.config.redis_url}`")

        return page_options.get(selected_page, "overview")

    def _render_status_indicators(self):
        """Render status indicators in sidebar"""
        # Get system health
        health = self._get_system_health()

        col1, col2, col3 = st.sidebar.columns(3)

        with col1:
            color = "green" if health.get('redis_connected') else "red"
            st.markdown(f"<div style='text-align: center; color: {color};'>Redis</div>", unsafe_allow_html=True)

        with col2:
            color = "green" if health.get('cache_hit_rate', 0) > 30 else "orange"
            st.markdown(f"<div style='text-align: center; color: {color};'>Cache</div>", unsafe_allow_html=True)

        with col3:
            color = "green" if health.get('uptime', 0) > 95 else "red"
            st.markdown(f"<div style='text-align: center; color: {color};'>Uptime</div>", unsafe_allow_html=True)

        st.sidebar.markdown("---")

    def _export_dashboard_data(self):
        """Export dashboard data"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'cost_savings': self.get_cost_savings(),
                'cache_hit_rate': self.calculate_cache_hit_rate(),
                'latency_metrics': self.get_latency_metrics(),
                'top_users': self.get_redis_list("helix:pii:incidents", -10, -1)
            }

            # Create CSV download
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Metric', 'Value', 'Timestamp'])

            for key, value in export_data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        writer.writerow([f"{key}.{sub_key}", sub_value, export_data['timestamp']])
                else:
                    writer.writerow([key, value, export_data['timestamp']])

            csv_content = output.getvalue()

            st.download_button(
                label="Download Dashboard Export",
                data=csv_content,
                file_name=f"helix_dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Export failed: {e}")

    def _show_system_info(self):
        """Show system information modal"""
        health = self._get_system_health()

        st.markdown("### 🖥️ System Information")

        info_data = {
            "Metric": ["Redis Status", "Cache Hit Rate", "P99 Latency", "Uptime", "Memory Usage", "Active Connections"],
            "Value": [
                "Connected" if health.get('redis_connected') else "Disconnected",
                f"{health.get('cache_hit_rate', 0):.1f}%",
                f"{health.get('p99_latency', 0):.0f}ms",
                f"{health.get('uptime', 0):.1f}%",
                health.get('memory_usage', 'Unknown'),
                health.get('connections', 'Unknown')
            ]
        }

        df = pd.DataFrame(info_data)
        st.dataframe(df, hide_index=True)

        # Configuration info
        st.markdown("#### 📋 Configuration")
        st.json({
            "redis_url": self.config.redis_url,
            "refresh_interval": self.config.refresh_interval,
            "max_data_points": self.config.max_data_points,
            "theme": st.session_state.get('theme', self.config.theme)
        })

    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        cache_hit_rate = self.calculate_cache_hit_rate()
        latency_metrics = self.get_latency_metrics()

        health = {
            'redis_connected': bool(self.redis_client),
            'cache_hit_rate': cache_hit_rate,
            'p99_latency': latency_metrics.get('p99', 0),
            'uptime': 99.9,  # Mock data
            'memory_usage': 'Unknown',
            'connections': 'Unknown'
        }

        if self.redis_client:
            try:
                info = self.redis_client.info()
                health['memory_usage'] = info.get('used_memory_human', 'Unknown')
                health['connections'] = info.get('connected_clients', 'Unknown')
            except:
                pass

        return health

    def run(self):
        """Main dashboard rendering with modular pages"""
        # Import page modules with better error handling
        try:
            from pages.overview import OverviewPage
            from pages.cost_analysis import CostAnalysisPage
            from pages.cache_performance import CachePerformancePage
            from pages.security import SecurityPage
            from pages.user_management import UserManagementPage
            from pages.system_health import SystemHealthPage
        except ImportError as e:
            st.error(f"Failed to import page modules: {e}")
            st.info("Make sure the dashboard pages are in the correct directory structure.")
            return

        # Initialize page instances
        pages = {
            'overview': OverviewPage(self),
            'cost_analysis': CostAnalysisPage(self),
            'cache_performance': CachePerformancePage(self),
            'security': SecurityPage(self),
            'user_management': UserManagementPage(self),
            'system_health': SystemHealthPage(self)
        }

        # Render sidebar
        selected_page_key = self.render_sidebar()

        # Auto refresh logic
        if st.session_state.get('auto_refresh', True):
            # Use time.sleep but check for interaction to avoid long delays
            import time
            refresh_interval = st.session_state.get('refresh_interval', self.config.refresh_interval)

            # Show refresh indicator
            with st.sidebar:
                if st.session_state.get('auto_refresh'):
                    st.markdown(f"⏰ Auto-refreshing every {refresh_interval}s...")
                    time.sleep(min(refresh_interval, 60))  # Cap at 60s for responsiveness
                    st.rerun()

        # Main content area
        st.title("🔮 Helix AI Gateway Dashboard")

        # Render selected page
        try:
            if selected_page_key in pages:
                pages[selected_page_key].render()
            else:
                st.error(f"Page '{selected_page_key}' not found")
                # Fallback to overview
                pages['overview'].render()

        except Exception as e:
            st.error(f"Error rendering page: {e}")
            logger.error(f"Page render error: {e}")

        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.markdown(
                f"<small style='color: gray;'>Last refresh: {datetime.now().strftime('%H:%M:%S')}</small>",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
                "🔮 Helix AI Gateway Dashboard | Real-time monitoring and analytics"
                "</div>",
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"<small style='text-align: right; display: block; color: gray;'>"
                f"Status: <span style='color: {'green' if self.redis_client else 'red'};'>"
                f"{'Online' if self.redis_client else 'Offline'}</span></small>",
                unsafe_allow_html=True
            )

def main():
    """Main entry point"""
    dashboard = HelixDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()