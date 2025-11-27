# Helix Dashboard - Cost Analysis Page
# Comprehensive cost tracking and budget management

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import io

class CostAnalysisPage:
    """Cost analysis page for spending trends and budget tracking"""

    def __init__(self, dashboard):
        self.dashboard = dashboard

    def render_cost_overview(self):
        """Render cost overview metrics"""
        st.markdown("### 💰 Cost Overview")

        # Get cost data
        cost_data = self._get_cost_overview()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="📈 Total Spend (MTD)",
                value=f"${cost_data['mtd_spend']:,.2f}",
                delta=f"{cost_data['mtd_growth']:.1f}% from last month",
                delta_color="inverse" if cost_data['mtd_growth'] > 0 else "normal"
            )

        with col2:
            st.metric(
                label="💎 Cache Savings",
                value=f"${cost_data['cache_savings']:,.2f}",
                delta=f"{cost_data['cache_savings_rate']:.1f}% of total",
                delta_color="normal"
            )

        with col3:
            st.metric(
                label="📊 Average Daily",
                value=f"${cost_data['daily_average']:,.2f}",
                delta=f"${cost_data['daily_average_change']:+,.2f} from yesterday"
            )

        with col4:
            st.metric(
                label="🎯 Budget Remaining",
                value=f"${cost_data['budget_remaining']:,.2f}",
                delta=f"{cost_data['budget_usage']:.1f}% used",
                delta_color="inverse" if cost_data['budget_usage'] > 80 else "normal"
            )

    def render_spending_trends(self):
        """Render spending trends charts"""
        col1, col2 = st.columns(2)

        with col1:
            self._render_daily_spending_chart()

        with col2:
            self._render_cumulative_spending_chart()

    def _render_daily_spending_chart(self):
        """Render daily spending trend chart"""
        st.markdown("#### 📊 Daily Spending Trend")

        spending_data = self._get_daily_spending_data()

        if spending_data:
            df = pd.DataFrame(spending_data)
            df['date'] = pd.to_datetime(df['date'])
            df['rolling_avg'] = df['amount'].rolling(window=7).mean()

            fig = go.Figure()

            # Add daily spending bars
            fig.add_trace(go.Bar(
                x=df['date'],
                y=df['amount'],
                name='Daily Spending',
                marker_color='#00ff41',
                opacity=0.7
            ))

            # Add rolling average line
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['rolling_avg'],
                name='7-Day Average',
                line=dict(color='#ffaa00', width=2)
            ))

            fig.update_layout(
                title='Daily LLM Spending (Last 30 Days)',
                xaxis=dict(title='Date'),
                yaxis=dict(title='Cost ($)'),
                template='plotly_dark',
                height=350,
                hovermode='x unified',
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
            st.info("No spending data available")

    def _render_cumulative_spending_chart(self):
        """Render cumulative spending chart"""
        st.markdown("#### 📈 Cumulative Spending")

        spending_data = self._get_monthly_spending_data()

        if spending_data:
            df = pd.DataFrame(spending_data)
            df['month'] = pd.to_datetime(df['month'])
            df['cumulative'] = df['amount'].cumsum()

            fig = px.area(
                df,
                x='month',
                y='cumulative',
                title='Cumulative Spending (YTD)',
                labels={'cumulative': 'Cumulative Cost ($)', 'month': 'Month'},
                color_discrete_sequence=['#00ff41']
            )

            fig.update_layout(
                template='plotly_dark',
                height=350,
                xaxis=dict(title='Month'),
                yaxis=dict(title='Cumulative Cost ($)')
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cumulative spending data available")

    def render_cost_breakdown(self):
        """Render cost breakdown by model and user"""
        col1, col2 = st.columns(2)

        with col1:
            self._render_model_spending_breakdown()

        with col2:
            self._render_user_spending_breakdown()

    def _render_model_spending_breakdown(self):
        """Render spending breakdown by model"""
        st.markdown("#### 🤖 Model Spending Breakdown")

        model_data = self._get_model_spending_data()

        if model_data:
            df = pd.DataFrame(model_data)
            df = df.sort_values('spending', ascending=True)

            fig = px.bar(
                df,
                x='spending',
                y='model',
                title='Spending by Model',
                labels={'spending': 'Cost ($)', 'model': 'Model'},
                orientation='h',
                color_discrete_sequence=['#00ff41']
            )

            fig.update_layout(
                template='plotly_dark',
                height=400,
                xaxis=dict(title='Cost ($)'),
                yaxis=dict(title='Model')
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model spending data available")

    def _render_user_spending_breakdown(self):
        """Render spending breakdown by user"""
        st.markdown("#### 👥 Top Users by Spending")

        user_data = self._get_top_users_spending()

        if user_data:
            df = pd.DataFrame(user_data.head(10))

            # Create treemap for visual representation
            fig = px.treemap(
                df,
                path=['user_id'],
                values='spending',
                title='Top 10 Users by Spending',
                color='spending',
                color_continuous_scale='Viridis'
            )

            fig.update_layout(
                template='plotly_dark',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show data table
            st.dataframe(
                df[['user_id', 'spending', 'requests', 'avg_cost']],
                column_config={
                    "user_id": "User ID",
                    "spending": st.column_config.NumberColumn(
                        "Total Spend",
                        format="$%.2f"
                    ),
                    "requests": "Requests",
                    "avg_cost": st.column_config.NumberColumn(
                        "Avg Cost/Request",
                        format="$%.4f"
                    )
                },
                hide_index=True
            )
        else:
            st.info("No user spending data available")

    def render_budget_management(self):
        """Render budget management and projections"""
        st.markdown("### 🎯 Budget Management")

        budget_data = self._get_budget_data()

        col1, col2 = st.columns(2)

        with col1:
            self._render_budget_usage_chart(budget_data)

        with col2:
            self._render_cost_projection_chart()

        # Budget controls
        st.markdown("#### 📊 Budget Controls")

        col3, col4, col5 = st.columns(3)

        with col3:
            monthly_budget = st.number_input(
                "Monthly Budget ($)",
                value=budget_data['monthly_budget'],
                min_value=0,
                step=100
            )

        with col4:
            alert_threshold = st.slider(
                "Alert Threshold (%)",
                value=budget_data['alert_threshold'],
                min_value=50,
                max_value=100,
                step=5
            )

        with col5:
            auto_block = st.checkbox(
                "Auto-block on exceed",
                value=budget_data.get('auto_block', False)
            )

        if st.button("Update Budget Settings"):
            self._update_budget_settings(monthly_budget, alert_threshold, auto_block)
            st.success("Budget settings updated!")

    def _render_budget_usage_chart(self, budget_data):
        """Render budget usage gauge"""
        st.markdown("#### 📊 Budget Usage")

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = budget_data['usage_percentage'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Monthly Budget Usage (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 90], 'color': "orange"},
                    {'range': [90, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': budget_data['alert_threshold']
                }
            }
        ))

        fig.update_layout(
            height=300,
            template='plotly_dark'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Budget: ${budget_data['monthly_budget']:,.2f} | " +
                  f"Used: ${budget_data['used_amount']:,.2f} | " +
                  f"Remaining: ${budget_data['remaining']:,.2f}")

    def _render_cost_projection_chart(self):
        """Render cost projection chart"""
        st.markdown("#### 🔮 Cost Projection")

        projection_data = self._get_cost_projections()

        if projection_data:
            df = pd.DataFrame(projection_data)
            df['date'] = pd.to_datetime(df['date'])

            fig = go.Figure()

            # Add historical data
            fig.add_trace(go.Scatter(
                x=df[df['type'] == 'historical']['date'],
                y=df[df['type'] == 'historical']['cost'],
                name='Historical',
                line=dict(color='#00ff41', width=2)
            ))

            # Add projection
            fig.add_trace(go.Scatter(
                x=df[df['type'] == 'projection']['date'],
                y=df[df['type'] == 'projection']['cost'],
                name='Projected',
                line=dict(color='#ffaa00', width=2, dash='dash')
            ))

            # Add budget line
            fig.add_hline(
                y=projection_data[-1]['budget_limit'],
                line_dash="dot",
                line_color="red",
                annotation_text="Budget Limit"
            )

            fig.update_layout(
                title='30-Day Cost Projection',
                xaxis=dict(title='Date'),
                yaxis=dict(title='Cost ($)'),
                template='plotly_dark',
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No projection data available")

    def render_export_section(self):
        """Render export functionality"""
        st.markdown("### 📤 Export Reports")

        export_format = st.selectbox(
            "Select export format",
            ["CSV", "JSON", "Excel"]
        )

        date_range = st.date_input(
            "Select date range",
            value=(
                datetime.now() - timedelta(days=30),
                datetime.now()
            ),
            max_value=datetime.now()
        )

        export_type = st.selectbox(
            "Export data type",
            ["Spending Data", "User Analytics", "Model Performance", "Full Report"]
        )

        if st.button("Generate Export"):
            data = self._generate_export_data(export_type, date_range)
            file_data = self._format_export_data(data, export_format)

            if file_data:
                filename = f"helix_export_{export_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.{export_format.lower()}"

                st.download_button(
                    label=f"Download {export_format} file",
                    data=file_data,
                    file_name=filename,
                    mime=self._get_mime_type(export_format)
                )
                st.success(f"Export ready! Click to download {filename}")
            else:
                st.error("Failed to generate export data")

    def _get_cost_overview(self) -> Dict[str, Any]:
        """Get cost overview metrics"""
        cost_savings = self.dashboard.get_cost_savings()
        daily_spend = self.dashboard.get_redis_sorted_set("helix:spend:total")

        mtd_spend = sum(daily_spend.values()) if daily_spend else 0
        cache_savings = cost_savings['total_savings']
        cache_savings_rate = cost_savings['cache_savings_rate']

        return {
            'mtd_spend': mtd_spend,
            'mtd_growth': 15.2,  # Mock data
            'cache_savings': cache_savings,
            'cache_savings_rate': cache_savings_rate,
            'daily_average': mtd_spend / 30,
            'daily_average_change': 2.50,
            'budget_remaining': 874.50,
            'budget_usage': 12.6
        }

    def _get_daily_spending_data(self) -> List[Dict]:
        """Get daily spending data for the last 30 days"""
        daily_spend = self.dashboard.get_redis_sorted_set("helix:spend:total")

        if daily_spend:
            data = []
            for date_str, amount in daily_spend.items():
                data.append({
                    'date': date_str,
                    'amount': float(amount)
                })
            return data[-30:]  # Last 30 days
        return []

    def _get_monthly_spending_data(self) -> List[Dict]:
        """Get monthly spending data"""
        # Mock data - in production, aggregate from daily data
        months = []
        for i in range(12):
            month_date = datetime.now() - timedelta(days=30 * i)
            months.append({
                'month': month_date.strftime('%Y-%m'),
                'amount': np.random.uniform(50, 300)
            })
        return months

    def _get_model_spending_data(self) -> List[Dict]:
        """Get spending data broken down by model"""
        # Mock data - in production, come from Redis analytics
        models = ['gpt-4', 'gpt-3.5-turbo', 'claude-3', 'llama-3-70b']
        data = []
        for model in models:
            data.append({
                'model': model,
                'spending': np.random.uniform(20, 150)
            })
        return data

    def _get_top_users_spending(self) -> pd.DataFrame:
        """Get top users by spending"""
        if not self.dashboard.redis_client:
            return pd.DataFrame()

        try:
            user_spend_keys = self.dashboard.redis_client.keys("helix:spend:user:*")
            user_data = []

            for key in user_spend_keys[:50]:  # Limit to top 50 for performance
                user_id = key.decode().replace("helix:spend:user:", "")
                daily_spend = self.dashboard.get_redis_sorted_set(key.decode())

                if daily_spend:
                    total_spend = sum(daily_spend.values())
                    total_requests = len(daily_spend)

                    user_data.append({
                        'user_id': user_id,
                        'spending': total_spend,
                        'requests': total_requests,
                        'avg_cost': total_spend / max(total_requests, 1)
                    })

            df = pd.DataFrame(user_data)
            return df.sort_values('spending', ascending=False) if not df.empty else df

        except Exception as e:
            st.error(f"Error fetching user spending data: {e}")
            return pd.DataFrame()

    def _get_budget_data(self) -> Dict[str, Any]:
        """Get budget data"""
        return {
            'monthly_budget': 1000.0,
            'used_amount': 125.50,
            'remaining': 874.50,
            'usage_percentage': 12.6,
            'alert_threshold': 80,
            'auto_block': False
        }

    def _get_cost_projections(self) -> List[Dict]:
        """Get cost projection data"""
        projections = []
        today = datetime.now()

        # Historical data (last 15 days)
        for i in range(15, 0, -1):
            date = today - timedelta(days=i)
            projections.append({
                'date': date,
                'cost': np.random.uniform(15, 45),
                'type': 'historical',
                'budget_limit': 1000 / 30  # Daily budget limit
            })

        # Projected data (next 15 days)
        for i in range(1, 16):
            date = today + timedelta(days=i)
            projections.append({
                'date': date,
                'cost': np.random.uniform(20, 50),
                'type': 'projection',
                'budget_limit': 1000 / 30
            })

        return projections

    def _update_budget_settings(self, monthly_budget: float, alert_threshold: int, auto_block: bool):
        """Update budget settings"""
        # In production, save to Redis or database
        pass

    def _generate_export_data(self, export_type: str, date_range) -> Optional[Dict]:
        """Generate data for export"""
        try:
            if export_type == "Spending Data":
                return {
                    'daily_spending': self._get_daily_spending_data(),
                    'model_breakdown': self._get_model_spending_data()
                }
            elif export_type == "User Analytics":
                return {
                    'top_users': self._get_top_users_spending().to_dict('records')
                }
            # Add other export types
            return {"message": "Export data generated successfully"}
        except Exception as e:
            st.error(f"Error generating export: {e}")
            return None

    def _format_export_data(self, data: Dict, format_type: str) -> Optional[str]:
        """Format data for export"""
        try:
            if format_type == "CSV":
                output = io.StringIO()
                pd.DataFrame(data['daily_spending']).to_csv(output, index=False)
                return output.getvalue()
            elif format_type == "JSON":
                return json.dumps(data, indent=2, default=str)
            elif format_type == "Excel":
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    pd.DataFrame(data['daily_spending']).to_excel(writer, sheet_name='Daily Spending', index=False)
                return output.getvalue()
        except Exception as e:
            st.error(f"Error formatting export: {e}")
            return None

    def _get_mime_type(self, format_type: str) -> str:
        """Get MIME type for export format"""
        mime_types = {
            "CSV": "text/csv",
            "JSON": "application/json",
            "Excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
        return mime_types.get(format_type, "text/plain")

    def render(self):
        """Main render method for cost analysis page"""
        self.render_cost_overview()
        st.markdown("---")
        self.render_spending_trends()
        st.markdown("---")
        self.render_cost_breakdown()
        st.markdown("---")
        self.render_budget_management()
        st.markdown("---")
        self.render_export_section()