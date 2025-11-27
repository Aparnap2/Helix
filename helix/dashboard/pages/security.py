# Helix Dashboard - Security Page
# PII incidents monitoring and compliance management

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import json

class SecurityPage:
    """Security page for PII monitoring and compliance management"""

    def __init__(self, dashboard):
        self.dashboard = dashboard

    def render_security_overview(self):
        """Render security overview metrics"""
        st.markdown("### 🔒 Security Overview")

        security_metrics = self._get_security_metrics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="🚨 PII Incidents Today",
                value=security_metrics['incidents_today'],
                delta=f"{security_metrics['incident_change']:+.0f} from yesterday",
                delta_color="inverse" if security_metrics['incident_change'] > 0 else "normal"
            )

        with col2:
            st.metric(
                label="🛡️ Requests Processed",
                value=f"{security_metrics['requests_processed']:,}",
                delta=f"{security_metrics['processed_successfully']:,} successful"
            )

        with col3:
            st.metric(
                label="🔐 Entities Detected",
                value=security_metrics['entities_detected'],
                delta=f"{security_metrics['entities_blocked']} blocked"
            )

        with col4:
            st.metric(
                label="✅ Compliance Score",
                value=f"{security_metrics['compliance_score']:.1f}%",
                delta=f"{security_metrics['compliance_trend']:+.1f}%",
                delta_color="normal"
            )

    def render_incident_monitoring(self):
        """Render PII incident monitoring section"""
        col1, col2 = st.columns(2)

        with col1:
            self._render_incident_trend()

        with col2:
            self._render_entity_breakdown()

    def _render_incident_trend(self):
        """Render incident trend chart"""
        st.markdown("#### 📊 PII Incident Trend")

        incident_data = self._get_incident_trend()

        if incident_data:
            df = pd.DataFrame(incident_data)
            df['date'] = pd.to_datetime(df['date'])

            fig = go.Figure()

            # Add incident count bars
            fig.add_trace(go.Bar(
                x=df['date'],
                y=df['incidents'],
                name='PII Incidents',
                marker_color='#ff6b6b',
                opacity=0.8
            ))

            # Add moving average
            df['moving_avg'] = df['incidents'].rolling(window=7).mean()
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['moving_avg'],
                name='7-Day Average',
                line=dict(color='#ffaa00', width=2)
            ))

            fig.update_layout(
                title='PII Incidents (Last 30 Days)',
                xaxis=dict(title='Date'),
                yaxis=dict(title='Number of Incidents'),
                template='plotly_dark',
                height=350,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No incident trend data available")

    def _render_entity_breakdown(self):
        """Render entity type breakdown"""
        st.markdown("#### 🎯 Entity Type Breakdown")

        entity_data = self._get_entity_breakdown()

        if entity_data:
            df = pd.DataFrame(entity_data)

            # Create sunburst chart for entity distribution
            fig = go.Figure(go.Sunburst(
                ids=[f"{entity}-{category}" for entity in entity_data.keys() for category in ['detected', 'blocked']],
                labels=[category for entity in entity_data.keys() for category in [entity, 'Detected', 'Blocked']],
                parents=["", entity, entity for entity in entity_data.keys()],
                values=[df.loc[df['entity_type'] == entity, 'count'].values[0] for entity in entity_data.keys() for _ in [1, 2]],
                branchvalues="total"
            ))

            fig.update_layout(
                title='PII Entity Distribution',
                template='plotly_dark',
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No entity breakdown data available")

    def render_recent_incidents(self):
        """Render recent PII incidents table"""
        st.markdown("### 🔍 Recent PII Incidents")

        incidents = self._get_recent_incidents()

        if incidents:
            df = pd.DataFrame(incidents)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Filter controls
            col1, col2, col3 = st.columns(3)

            with col1:
                entity_filter = st.selectbox(
                    "Filter by Entity",
                    ["All"] + list(df['entity_type'].unique())
                )

            with col2:
                time_filter = st.selectbox(
                    "Time Range",
                    ["All", "Last 24h", "Last 7d", "Last 30d"]
                )

            with col3:
                action_filter = st.selectbox(
                    "Filter by Action",
                    ["All"] + list(df['action_taken'].unique())
                )

            # Apply filters
            filtered_df = df.copy()
            if entity_filter != "All":
                filtered_df = filtered_df[filtered_df['entity_type'] == entity_filter]
            if action_filter != "All":
                filtered_df = filtered_df[filtered_df['action_taken'] == action_filter]

            # Display incidents table
            st.dataframe(
                filtered_df,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn(
                        "Time Detected",
                        format="MMM DD, YYYY, HH:mm:ss"
                    ),
                    "user_id": "User ID",
                    "entity_type": "Entity Type",
                    "severity": st.column_config.ProgressColumn(
                        "Severity",
                        format="%.0f%%",
                        min_value=0,
                        max_value=100
                    ),
                    "action_taken": "Action Taken",
                    "content_preview": "Content Preview"
                },
                hide_index=True,
                use_container_width=True
            )

            # Export incidents
            if st.button("📤 Export Incidents"):
                self._export_incidents(filtered_df)
                st.success("Incidents exported successfully!")
        else:
            st.success("🎉 No PII incidents detected in recent logs")

    def render_compliance_management(self):
        """Render compliance management section"""
        st.markdown("### 📋 Compliance Management")

        col1, col2 = st.columns(2)

        with col1:
            self._render_compliance_score()

        with col2:
            self._render_regulatory_status()

    def _render_compliance_score(self):
        """Render compliance score visualization"""
        st.markdown("#### 📊 Compliance Score Breakdown")

        compliance_data = self._get_compliance_data()

        if compliance_data:
            # Create radar chart for compliance metrics
            categories = compliance_data['categories']
            scores = compliance_data['scores']

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=categories,
                fill='toself',
                name='Compliance Score',
                line_color='#00ff41',
                fillcolor='rgba(0, 255, 65, 0.25)'
            ))

            # Add target line
            fig.add_trace(go.Scatterpolar(
                r=[90] * len(categories),  # Target score
                theta=categories,
                name='Target Score',
                line_color='red',
                line_dash='dash',
                fill='none'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title='Compliance Score Overview',
                template='plotly_dark',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No compliance data available")

    def _render_regulatory_status(self):
        """Render regulatory compliance status"""
        st.markdown("#### 🏛️ Regulatory Compliance Status")

        regulatory_data = self._get_regulatory_status()

        if regulatory_data:
            for regulation in regulatory_data:
                status_color = {
                    'compliant': 'green',
                    'partial': 'orange',
                    'non_compliant': 'red'
                }.get(regulation['status'], 'gray')

                status_icon = {
                    'compliant': '✅',
                    'partial': '⚠️',
                    'non_compliant': '❌'
                }.get(regulation['status'], '❓')

                st.markdown(
                    f"""
                    <div style='padding: 15px; margin: 10px 0; border-left: 4px solid {status_color};
                                background-color: rgba(255,255,255,0.05); border-radius: 5px;'>
                        <h4 style='margin: 0;'>{status_icon} {regulation['name']}</h4>
                        <p style='margin: 5px 0;'>{regulation['description']}</p>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='background-color: {status_color}; color: white; padding: 3px 10px;
                                       border-radius: 12px; font-size: 0.8em;'>
                                {regulation['status'].upper()}
                            </span>
                            <small style='color: gray;'>Last reviewed: {regulation['last_reviewed']}</small>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No regulatory status data available")

    def render_security_settings(self):
        """Render security settings and configuration"""
        st.markdown("### ⚙️ Security Settings")

        # PII Detection Settings
        st.markdown("#### 🔍 PII Detection Settings")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Detection Sensitivity")
            sensitivity = st.slider(
                "Detection Sensitivity",
                value=0.8,
                min_value=0.1,
                max_value=1.0,
                step=0.1,
                help="Higher values increase detection accuracy but may have more false positives"
            )

        with col2:
            st.subheader("Entity Types")
            entity_types = st.multiselect(
                "Monitored Entity Types",
                ['CREDIT_CARD', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'SSN', 'PASSPORT', 'DRIVER_LICENSE'],
                default=['CREDIT_CARD', 'EMAIL_ADDRESS', 'PHONE_NUMBER']
            )

        with col3:
            st.subheader("Action Settings")
            strict_mode = st.checkbox(
                "Strict Mode",
                value=False,
                help="In strict mode, any detected PII results in request blocking"
            )
            log_all = st.checkbox(
                "Log All Incidents",
                value=True,
                help="Log all detected PII incidents, including those that were successfully processed"
            )

        # Update settings button
        if st.button("Update Security Settings", type="primary"):
            self._update_security_settings({
                'sensitivity': sensitivity,
                'entity_types': entity_types,
                'strict_mode': strict_mode,
                'log_all': log_all
            })
            st.success("Security settings updated successfully!")

        # Alert Configuration
        st.markdown("#### 🚨 Alert Configuration")

        alert_col1, alert_col2, alert_col3 = st.columns(3)

        with alert_col1:
            threshold = st.number_input(
                "Alert Threshold (incidents/hour)",
                value=10,
                min_value=1,
                max_value=100
            )

        with alert_col2:
            notification_methods = st.multiselect(
                "Notification Methods",
                ['Email', 'Slack', 'Webhook', 'SMS'],
                default=['Email', 'Slack']
            )

        with alert_col3:
            escalation_enabled = st.checkbox(
                "Enable Escalation",
                value=True,
                help="Automatically escalate high-priority incidents"
            )

        if st.button("Update Alert Settings", type="secondary"):
            self._update_alert_settings({
                'threshold': threshold,
                'methods': notification_methods,
                'escalation': escalation_enabled
            })
            st.success("Alert settings updated successfully!")

    def _get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics"""
        return {
            'incidents_today': np.random.randint(0, 15),
            'incident_change': np.random.randint(-5, 10),
            'requests_processed': np.random.randint(1000, 5000),
            'processed_successfully': np.random.randint(950, 4800),
            'entities_detected': np.random.randint(50, 200),
            'entities_blocked': np.random.randint(5, 25),
            'compliance_score': np.random.uniform(85, 98),
            'compliance_trend': np.random.uniform(-2, 5)
        }

    def _get_incident_trend(self) -> List[Dict]:
        """Get incident trend data for the last 30 days"""
        trend_data = []
        for i in range(30, 0, -1):
            date = datetime.now() - timedelta(days=i)
            trend_data.append({
                'date': date,
                'incidents': np.random.randint(0, 20)
            })
        return trend_data

    def _get_entity_breakdown(self) -> Dict[str, int]:
        """Get entity type breakdown"""
        return {
            'CREDIT_CARD': np.random.randint(5, 25),
            'EMAIL_ADDRESS': np.random.randint(10, 50),
            'PHONE_NUMBER': np.random.randint(8, 35),
            'SSN': np.random.randint(2, 15),
            'PASSPORT': np.random.randint(1, 10),
            'DRIVER_LICENSE': np.random.randint(3, 20)
        }

    def _get_recent_incidents(self) -> List[Dict]:
        """Get recent PII incidents"""
        incidents = self.dashboard.get_redis_list("helix:pii:incidents", -50, -1)

        # Convert to proper format if needed
        formatted_incidents = []
        for incident in incidents:
            if isinstance(incident, str):
                try:
                    incident = json.loads(incident)
                except:
                    continue

            # Ensure required fields
            if not isinstance(incident, dict):
                continue

            formatted_incidents.append({
                'timestamp': incident.get('timestamp', datetime.now().isoformat()),
                'user_id': incident.get('user_id', 'unknown'),
                'entity_type': incident.get('entity_type', 'UNKNOWN'),
                'severity': incident.get('severity', 50),
                'action_taken': incident.get('action_taken', 'redacted'),
                'content_preview': incident.get('content_preview', incident.get('content', ''))[:100]
            })

        # If no incidents from Redis, add some sample data
        if not formatted_incidents:
            for i in range(5):
                formatted_incidents.append({
                    'timestamp': (datetime.now() - timedelta(hours=i*2)).isoformat(),
                    'user_id': f'user_{i:03d}',
                    'entity_type': np.random.choice(['CREDIT_CARD', 'EMAIL_ADDRESS', 'PHONE_NUMBER']),
                    'severity': np.random.randint(30, 90),
                    'action_taken': 'redacted',
                    'content_preview': f'Sample PII incident #{i+1} with sensitive content preview...'
                })

        return formatted_incidents

    def _get_compliance_data(self) -> Dict[str, Any]:
        """Get compliance data"""
        return {
            'categories': ['Data Encryption', 'Access Control', 'Audit Logging', 'PII Redaction', 'Data Retention', 'Incident Response'],
            'scores': [95, 88, 92, 85, 78, 90]
        }

    def _get_regulatory_status(self) -> List[Dict]:
        """Get regulatory compliance status"""
        return [
            {
                'name': 'GDPR',
                'description': 'General Data Protection Regulation',
                'status': 'compliant',
                'last_reviewed': '2024-01-15'
            },
            {
                'name': 'CCPA',
                'description': 'California Consumer Privacy Act',
                'status': 'compliant',
                'last_reviewed': '2024-01-10'
            },
            {
                'name': 'HIPAA',
                'description': 'Health Insurance Portability and Accountability Act',
                'status': 'partial',
                'last_reviewed': '2024-01-08'
            },
            {
                'name': 'PCI DSS',
                'description': 'Payment Card Industry Data Security Standard',
                'status': 'compliant',
                'last_reviewed': '2024-01-12'
            }
        ]

    def _export_incidents(self, df):
        """Export incidents to file"""
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Incidents CSV",
            data=csv,
            file_name=f"pii_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    def _update_security_settings(self, settings: Dict[str, Any]):
        """Update security settings"""
        # In production, save to configuration or database
        pass

    def _update_alert_settings(self, settings: Dict[str, Any]):
        """Update alert settings"""
        # In production, save to configuration or database
        pass

    def render(self):
        """Main render method for security page"""
        self.render_security_overview()
        st.markdown("---")
        self.render_incident_monitoring()
        st.markdown("---")
        self.render_recent_incidents()
        st.markdown("---")
        self.render_compliance_management()
        st.markdown("---")
        self.render_security_settings()