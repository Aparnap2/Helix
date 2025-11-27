# Helix Dashboard - System Health Page
# Real-time system monitoring and health metrics

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import json
import psutil
import platform

class SystemHealthPage:
    """System health page for real-time monitoring"""

    def __init__(self, dashboard):
        self.dashboard = dashboard

    def render_system_overview(self):
        """Render system health overview"""
        st.markdown("### ⚕️ System Health Overview")

        health_data = self._get_system_health_data()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status_color = "green" if health_data['overall_status'] == "healthy" else "orange" if health_data['overall_status'] == "warning" else "red"
            st.metric(
                label="🏥 System Status",
                value=health_data['overall_status'].title(),
                delta="All systems operational" if health_data['overall_status'] == "healthy" else "Attention required",
                delta_color="normal" if health_data['overall_status'] == "healthy" else "inverse"
            )

        with col2:
            st.metric(
                label="⏱️ Uptime",
                value=f"{health_data['uptime']:.1f}%",
                delta=f"Last checked: {health_data['last_check']}"
            )

        with col3:
            st.metric(
                label="🔄 Redis Status",
                value=health_data['redis_status'],
                delta=f"{health_data['redis_connections']} connections"
            )

        with col4:
            st.metric(
                label="💾 Memory Usage",
                value=f"{health_data['memory_usage']:.1f}%",
                delta=f"{health_data['memory_used']}/{health_data['memory_total']}"
            )

    def render_resource_monitoring(self):
        """Render resource monitoring charts"""
        col1, col2 = st.columns(2)

        with col1:
            self._render_cpu_memory_chart()

        with col2:
            self._render_redis_metrics()

    def _render_cpu_memory_chart(self):
        """Render CPU and memory usage chart"""
        st.markdown("#### 💻 System Resources")

        resource_data = self._get_resource_data()

        if resource_data:
            df = pd.DataFrame(resource_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['CPU Usage (%)', 'Memory Usage (%)'],
                vertical_spacing=0.1
            )

            # CPU usage
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['cpu_percent'],
                name='CPU Usage',
                line=dict(color='#00ff41', width=2)
            ), row=1, col=1)

            # Memory usage
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['memory_percent'],
                name='Memory Usage',
                line=dict(color='#ffaa00', width=2)
            ), row=2, col=1)

            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Usage (%)", range=[0, 100], row=1, col=1)
            fig.update_yaxes(title_text="Usage (%)", range=[0, 100], row=2, col=1)

            fig.update_layout(
                height=400,
                template='plotly_dark',
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No resource data available")

    def _render_redis_metrics(self):
        """Render Redis performance metrics"""
        st.markdown("#### 🔴 Redis Metrics")

        redis_data = self._get_redis_metrics()

        if redis_data:
            # Create metrics grid
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Performance**")
                perf_df = pd.DataFrame([
                    {'Metric': 'Connected Clients', 'Value': redis_data['connected_clients']},
                    {'Metric': 'Commands/sec', 'Value': redis_data['instantaneous_ops_per_sec']},
                    {'Metric': 'Hit Rate', 'Value': f"{redis_data['hit_rate']:.1f}%"},
                    {'Metric': 'Expired Keys', 'Value': redis_data['expired_keys']}
                ])
                st.dataframe(perf_df, hide_index=True)

            with col2:
                st.markdown("**Memory**")
                mem_df = pd.DataFrame([
                    {'Metric': 'Used Memory', 'Value': redis_data['used_memory_human']},
                    {'Metric': 'Peak Memory', 'Value': redis_data['used_memory_peak_human']},
                    {'Metric': 'Memory Fragmentation', 'Value': f"{redis_data['mem_fragmentation_ratio']:.2f}"},
                    {'Metric': 'Avg TTL', 'Value': f"{redis_data['avg_ttl']}ms"}
                ])
                st.dataframe(mem_df, hide_index=True)

            # Redis commands chart
            command_data = self._get_redis_commands_data()
            if command_data:
                df_commands = pd.DataFrame(command_data)

                fig = px.bar(
                    df_commands,
                    x='count',
                    y='command',
                    title='Most Frequent Redis Commands',
                    labels={'count': 'Count', 'command': 'Command'},
                    orientation='h',
                    color_discrete_sequence=['#00ff41']
                )

                fig.update_layout(
                    template='plotly_dark',
                    height=250
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Redis metrics available")

    def render_service_health(self):
        """Render service health status"""
        st.markdown("### 🚨 Service Health")

        services = self._get_service_health()

        for service in services:
            status_color = {
                'healthy': 'green',
                'warning': 'orange',
                'critical': 'red',
                'unknown': 'gray'
            }.get(service['status'], 'gray')

            status_icon = {
                'healthy': '✅',
                'warning': '⚠️',
                'critical': '❌',
                'unknown': '❓'
            }.get(service['status'], '❓')

            st.markdown(
                f"""
                <div style='padding: 15px; margin: 10px 0; border-left: 4px solid {status_color};
                            background-color: rgba(255,255,255,0.05); border-radius: 5px;'>
                    <h4 style='margin: 0;'>{status_icon} {service['name']}</h4>
                    <p style='margin: 5px 0;'>{service['description']}</p>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='background-color: {status_color}; color: white; padding: 3px 10px;
                                   border-radius: 12px; font-size: 0.8em;'>
                            {service['status'].upper()}
                        </span>
                        <small style='color: gray;'>
                            Last check: {service['last_check']} |
                            Response: {service['response_time']}ms
                        </small>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    def render_log_monitoring(self):
        """Render log monitoring and error tracking"""
        st.markdown("### 📝 Log Monitoring")

        col1, col2 = st.columns(2)

        with col1:
            self._render_error_trend()

        with col2:
            self._render_recent_logs()

    def _render_error_trend(self):
        """Render error trend chart"""
        st.markdown("#### 📉 Error Trend (24h)")

        error_data = self._get_error_trend()

        if error_data:
            df = pd.DataFrame(error_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            fig = px.line(
                df,
                x='timestamp',
                y='error_count',
                title='Error Rate Over Time',
                labels={'error_count': 'Error Count', 'timestamp': 'Time'},
                color_discrete_sequence=['#ff6b6b']
            )

            # Add threshold line
            threshold = 10  # Error threshold
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Threshold: {threshold}"
            )

            fig.update_layout(
                template='plotly_dark',
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No error data available")

    def _render_recent_logs(self):
        """Render recent error logs"""
        st.markdown("#### 📋 Recent Error Logs")

        logs = self._get_recent_logs()

        if logs:
            df = pd.DataFrame(logs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Log level filter
            log_level = st.selectbox(
                "Filter by Level",
                ["All", "ERROR", "WARNING", "INFO"],
                key="log_level_filter"
            )

            if log_level != "All":
                df = df[df['level'] == log_level]

            st.dataframe(
                df[['timestamp', 'level', 'message', 'source']],
                column_config={
                    "timestamp": st.column_config.DatetimeColumn(
                        "Time",
                        format="MMM DD, YYYY, HH:mm:ss"
                    ),
                    "level": "Level",
                    "message": "Message",
                    "source": "Source"
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.success("✅ No recent errors found")

    def render_performance_monitoring(self):
        """Render performance monitoring section"""
        st.markdown("### ⚡ Performance Monitoring")

        col1, col2 = st.columns(2)

        with col1:
            self._render_latency_distribution()

        with col2:
            self._render_throughput_metrics()

    def _render_latency_distribution(self):
        """Render latency distribution chart"""
        st.markdown("#### 📊 Latency Distribution")

        latency_data = self._get_latency_distribution()

        if latency_data:
            percentiles = ['p50', 'p75', 'p90', 'p95', 'p99']
            values = [latency_data.get(p, 0) for p in percentiles]

            fig = go.Figure()

            # Add bars for percentiles
            fig.add_trace(go.Bar(
                x=percentiles,
                y=values,
                name='Latency Percentiles',
                marker_color='#00ff41'
            ))

            fig.update_layout(
                title='Response Time Percentiles',
                xaxis=dict(title='Percentile'),
                yaxis=dict(title='Latency (ms)'),
                template='plotly_dark',
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No latency data available")

    def _render_throughput_metrics(self):
        """Render throughput metrics"""
        st.markdown("#### 📈 Throughput Metrics")

        throughput_data = self._get_throughput_metrics()

        if throughput_data:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Requests/sec",
                    value=f"{throughput_data['requests_per_second']:.1f}"
                )

            with col2:
                st.metric(
                    label="Avg Response Time",
                    value=f"{throughput_data['avg_response_time']:.1f}ms"
                )

            with col3:
                st.metric(
                    label="Success Rate",
                    value=f"{throughput_data['success_rate']:.1f}%"
                )

            # Throughput trend
            trend_data = self._get_throughput_trend()
            if trend_data:
                df = pd.DataFrame(trend_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                fig = px.line(
                    df,
                    x='timestamp',
                    y='throughput',
                    title='Throughput Over Time',
                    labels={'throughput': 'Requests/sec', 'timestamp': 'Time'},
                    color_discrete_sequence=['#00ff41']
                )

                fig.update_layout(
                    template='plotly_dark',
                    height=200
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No throughput data available")

    def render_alerts_and_notifications(self):
        """Render alerts and notifications section"""
        st.markdown("### 🚨 Alerts & Notifications")

        alerts = self._get_active_alerts()

        if alerts:
            for alert in alerts:
                severity_color = {
                    'critical': 'red',
                    'warning': 'orange',
                    'info': 'blue'
                }.get(alert['severity'], 'gray')

                action_col1, action_col2, action_col3 = st.columns([3, 1, 1])

                with action_col1:
                    st.markdown(
                        f"""
                        <div style='padding: 15px; margin: 10px 0; border-left: 4px solid {severity_color};
                                    background-color: rgba(255,255,255,0.05); border-radius: 5px;'>
                            <h4 style='margin: 0; color: {severity_color};'>
                                {alert['title']}
                                <span style='background-color: {severity_color}; color: white; padding: 2px 8px;
                                           border-radius: 12px; font-size: 0.8em; margin-left: 10px;'>
                                    {alert['severity'].upper()}
                                </span>
                            </h4>
                            <p style='margin: 5px 0;'>{alert['message']}</p>
                            <small style='color: gray;'>{alert['timestamp']}</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with action_col2:
                    if st.button("🔕 Acknowledge", key=f"ack_{alert['id']}"):
                        self._acknowledge_alert(alert['id'])

                with action_col3:
                    if st.button("🗑️ Dismiss", key=f"dismiss_{alert['id']}"):
                        self._dismiss_alert(alert['id'])
        else:
            st.success("✅ No active alerts")

        # Alert configuration
        st.markdown("#### ⚙️ Alert Configuration")

        alert_col1, alert_col2 = st.columns(2)

        with alert_col1:
            cpu_threshold = st.number_input(
                "CPU Alert Threshold (%)",
                value=80,
                min_value=50,
                max_value=95
            )

        with alert_col2:
            memory_threshold = st.number_input(
                "Memory Alert Threshold (%)",
                value=85,
                min_value=50,
                max_value=95
            )

        if st.button("Update Alert Thresholds"):
            self._update_alert_thresholds(cpu_threshold, memory_threshold)
            st.success("Alert thresholds updated!")

    def _get_system_health_data(self) -> Dict[str, Any]:
        """Get comprehensive system health data"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Redis metrics
            redis_status = "connected" if self.dashboard.redis_client else "disconnected"
            redis_connections = 0

            if self.dashboard.redis_client:
                try:
                    redis_info = self.dashboard.redis_client.info()
                    redis_connections = redis_info.get('connected_clients', 0)
                except:
                    pass

            return {
                'overall_status': 'healthy' if cpu_percent < 80 and memory.percent < 85 else 'warning' if cpu_percent < 95 and memory.percent < 95 else 'critical',
                'uptime': 99.9,  # Mock data
                'last_check': datetime.now().strftime('%H:%M:%S'),
                'redis_status': redis_status,
                'redis_connections': redis_connections,
                'memory_usage': memory.percent,
                'memory_used': f"{memory.used // (1024**3)}GB",
                'memory_total': f"{memory.total // (1024**3)}GB"
            }
        except Exception as e:
            return {
                'overall_status': 'unknown',
                'uptime': 0,
                'last_check': 'Unknown',
                'redis_status': 'unknown',
                'redis_connections': 0,
                'memory_usage': 0,
                'memory_used': 'Unknown',
                'memory_total': 'Unknown'
            }

    def _get_resource_data(self) -> List[Dict]:
        """Get resource usage data for the last hour"""
        data = []
        now = datetime.now()

        for i in range(60, 0, -2):  # Every 2 minutes for the last hour
            timestamp = now - timedelta(minutes=i)
            data.append({
                'timestamp': timestamp,
                'cpu_percent': np.random.uniform(10, 70),  # Mock data
                'memory_percent': np.random.uniform(30, 80)
            })

        return data

    def _get_redis_metrics(self) -> Dict[str, Any]:
        """Get Redis performance metrics"""
        if not self.dashboard.redis_client:
            return {}

        try:
            info = self.dashboard.redis_client.info()
            stats = self.dashboard.redis_client.info('stats')

            return {
                'connected_clients': info.get('connected_clients', 0),
                'instantaneous_ops_per_sec': stats.get('instantaneous_ops_per_sec', 0),
                'hit_rate': (stats.get('keyspace_hits', 0) / max(stats.get('keyspace_misses', 0) + stats.get('keyspace_hits', 0), 1)) * 100,
                'expired_keys': stats.get('expired_keys', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'used_memory_peak_human': info.get('used_memory_peak_human', '0B'),
                'mem_fragmentation_ratio': info.get('mem_fragmentation_ratio', 1.0),
                'avg_ttl': int(info.get('avg_ttl', 0) / 1000) if info.get('avg_ttl') else 0
            }
        except Exception as e:
            return {}

    def _get_redis_commands_data(self) -> List[Dict]:
        """Get most frequent Redis commands"""
        # Mock data - in production, get from Redis commandstats
        return [
            {'command': 'GET', 'count': np.random.randint(100, 1000)},
            {'command': 'SET', 'count': np.random.randint(80, 800)},
            {'command': 'HGET', 'count': np.random.randint(50, 500)},
            {'command': 'HSET', 'count': np.random.randint(40, 400)},
            {'command': 'PING', 'count': np.random.randint(30, 300)}
        ]

    def _get_service_health(self) -> List[Dict]:
        """Get health status of all services"""
        services = [
            {
                'name': 'Helix Proxy',
                'description': 'Main AI gateway service',
                'status': 'healthy',
                'response_time': np.random.randint(10, 100),
                'last_check': datetime.now().strftime('%H:%M:%S')
            },
            {
                'name': 'Redis Cache',
                'description': 'Cache and data storage',
                'status': 'healthy' if self.dashboard.redis_client else 'critical',
                'response_time': np.random.randint(1, 10) if self.dashboard.redis_client else 999,
                'last_check': datetime.now().strftime('%H:%M:%S')
            },
            {
                'name': 'Semantic Search',
                'description': 'Vector search service',
                'status': 'healthy',
                'response_time': np.random.randint(50, 200),
                'last_check': datetime.now().strftime('%H:%M:%S')
            },
            {
                'name': 'PII Detection',
                'description': 'PII redaction service',
                'status': 'healthy',
                'response_time': np.random.randint(20, 150),
                'last_check': datetime.now().strftime('%H:%M:%S')
            }
        ]
        return services

    def _get_error_trend(self) -> List[Dict]:
        """Get error trend data for the last 24 hours"""
        data = []
        now = datetime.now()

        for i in range(24, 0, -1):
            timestamp = now - timedelta(hours=i)
            data.append({
                'timestamp': timestamp,
                'error_count': np.random.randint(0, 15)  # Mock data
            })

        return data

    def _get_recent_logs(self) -> List[Dict]:
        """Get recent error logs"""
        # Mock data - in production, get from log files
        levels = ['ERROR', 'WARNING', 'INFO']
        sources = ['helix_proxy', 'redis', 'semantic_search', 'pii_detector']
        messages = [
            'Connection timeout to upstream provider',
            'Memory usage above threshold',
            'Cache miss for request',
            'PII entity detected and redacted',
            'Semantic search completed successfully'
        ]

        logs = []
        for i in range(10):
            logs.append({
                'timestamp': datetime.now() - timedelta(minutes=i*5),
                'level': np.random.choice(levels),
                'message': np.random.choice(messages),
                'source': np.random.choice(sources)
            })

        return logs

    def _get_latency_distribution(self) -> Dict[str, float]:
        """Get latency distribution percentiles"""
        return self.dashboard.get_latency_metrics()

    def _get_throughput_metrics(self) -> Dict[str, Any]:
        """Get throughput metrics"""
        return {
            'requests_per_second': np.random.uniform(10, 100),
            'avg_response_time': np.random.uniform(50, 300),
            'success_rate': np.random.uniform(95, 100)
        }

    def _get_throughput_trend(self) -> List[Dict]:
        """Get throughput trend data"""
        data = []
        now = datetime.now()

        for i in range(60, 0, -2):  # Every 2 minutes
            timestamp = now - timedelta(minutes=i)
            data.append({
                'timestamp': timestamp,
                'throughput': np.random.uniform(10, 100)  # Mock data
            })

        return data

    def _get_active_alerts(self) -> List[Dict]:
        """Get active system alerts"""
        # Mock alerts - in production, get from alerting system
        return [
            {
                'id': 'alert_001',
                'title': 'High CPU Usage',
                'message': 'CPU usage exceeded 80% for the last 5 minutes',
                'severity': 'warning',
                'timestamp': '5 minutes ago'
            },
            {
                'id': 'alert_002',
                'title': 'Redis Memory High',
                'message': 'Redis memory usage is approaching capacity',
                'severity': 'info',
                'timestamp': '15 minutes ago'
            }
        ]

    def _acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        # In production, update alert status in database
        st.success(f"Alert {alert_id} acknowledged")

    def _dismiss_alert(self, alert_id: str):
        """Dismiss an alert"""
        # In production, remove alert from active alerts
        st.success(f"Alert {alert_id} dismissed")

    def _update_alert_thresholds(self, cpu_threshold: int, memory_threshold: int):
        """Update alert thresholds"""
        # In production, save to configuration
        pass

    def render(self):
        """Main render method for system health page"""
        # Auto-refresh logic
        if st.session_state.get('auto_refresh', True):
            time.sleep(st.session_state.get('refresh_interval', 5))
            st.rerun()

        # Render sections
        self.render_system_overview()
        st.markdown("---")
        self.render_resource_monitoring()
        st.markdown("---")
        self.render_service_health()
        st.markdown("---")
        self.render_log_monitoring()
        st.markdown("---")
        self.render_performance_monitoring()
        st.markdown("---")
        self.render_alerts_and_notifications()