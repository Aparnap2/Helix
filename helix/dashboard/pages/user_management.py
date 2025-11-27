# Helix Dashboard - User Management Page
# Top users analytics and team usage patterns

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import json
import hashlib

class UserManagementPage:
    """User management page for analytics and team usage patterns"""

    def __init__(self, dashboard):
        self.dashboard = dashboard

    def render_user_overview(self):
        """Render user overview metrics"""
        st.markdown("### 👥 User Overview")

        user_metrics = self._get_user_metrics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="👤 Total Users",
                value=f"{user_metrics['total_users']:,}",
                delta=f"+{user_metrics['new_users_today']} today"
            )

        with col2:
            st.metric(
                label="🔥 Active Users",
                value=f"{user_metrics['active_users']:,}",
                delta=f"{user_metrics['active_rate']:.1f}% of total"
            )

        with col3:
            st.metric(
                label="💰 Avg. Spend/User",
                value=f"${user_metrics['avg_spend_per_user']:.2f}",
                delta=f"${user_metrics['avg_spend_change']:+.2f} from yesterday"
            )

        with col4:
            st.metric(
                label="📊 Avg. Requests/User",
                value=f"{user_metrics['avg_requests_per_user']:.0f}",
                delta=f"+{user_metrics['request_growth']:.1f}% this week"
            )

    def render_top_users(self):
        """Render top users by spending and usage"""
        col1, col2 = st.columns(2)

        with col1:
            self._render_spending_leaderboard()

        with col2:
            self._render_usage_leaderboard()

    def _render_spending_leaderboard(self):
        """Render spending leaderboard"""
        st.markdown("#### 💰 Top Users by Spending")

        spending_data = self._get_top_users_spending()

        if not spending_data.empty:
            # Create horizontal bar chart
            top_20 = spending_data.head(20)

            fig = px.bar(
                top_20,
                x='total_spend',
                y='user_id',
                title='Top 20 Users by Total Spending',
                labels={'total_spend': 'Total Spend ($)', 'user_id': 'User ID'},
                orientation='h',
                color='total_spend',
                color_continuous_scale='Viridis'
            )

            fig.update_layout(
                template='plotly_dark',
                height=500,
                xaxis=dict(title='Total Spend ($)'),
                yaxis=dict(title='User ID')
            )

            # Reverse y-axis to show highest spender at top
            fig.update_yaxes(autorange="reversed")

            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            st.markdown("**Detailed Spending Data**")
            st.dataframe(
                spending_data,
                column_config={
                    "user_id": "User ID",
                    "total_spend": st.column_config.NumberColumn(
                        "Total Spend",
                        format="$%.2f",
                        help="Total amount spent by user"
                    ),
                    "total_requests": st.column_config.NumberColumn(
                        "Total Requests",
                        format=",.0f"
                    ),
                    "avg_cost_per_request": st.column_config.NumberColumn(
                        "Avg Cost/Request",
                        format="$%.4f"
                    ),
                    "days_active": st.column_config.NumberColumn(
                        "Days Active",
                        format=",.0f"
                    ),
                    "avg_daily_spend": st.column_config.NumberColumn(
                        "Avg Daily Spend",
                        format="$%.2f"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No user spending data available")

    def _render_usage_leaderboard(self):
        """Render usage leaderboard"""
        st.markdown("#### 📊 Top Users by Usage")

        usage_data = self._get_top_users_usage()

        if not usage_data.empty:
            # Create bubble chart
            fig = px.scatter(
                usage_data,
                x='total_requests',
                y='avg_latency',
                size='total_tokens',
                hover_name='user_id',
                title='User Usage Patterns',
                labels={
                    'total_requests': 'Total Requests',
                    'avg_latency': 'Avg Latency (ms)',
                    'total_tokens': 'Total Tokens',
                    'user_id': 'User ID'
                },
                color='success_rate',
                color_continuous_scale='RdYlGn'
            )

            fig.update_layout(
                template='plotly_dark',
                height=500,
                xaxis=dict(title='Total Requests'),
                yaxis=dict(title='Average Latency (ms)')
            )

            st.plotly_chart(fig, use_container_width=True)

            # Usage metrics table
            st.markdown("**Usage Metrics**")
            st.dataframe(
                usage_data,
                column_config={
                    "user_id": "User ID",
                    "total_requests": st.column_config.NumberColumn(
                        "Total Requests",
                        format=",.0f"
                    ),
                    "total_tokens": st.column_config.NumberColumn(
                        "Total Tokens",
                        format=",.0f"
                    ),
                    "avg_latency": st.column_config.NumberColumn(
                        "Avg Latency (ms)",
                        format="%.1f"
                    ),
                    "success_rate": st.column_config.NumberColumn(
                        "Success Rate (%)",
                        format="%.1f"
                    ),
                    "cache_hit_rate": st.column_config.NumberColumn(
                        "Cache Hit Rate (%)",
                        format="%.1f"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No user usage data available")

    def render_user_analytics(self):
        """Render detailed user analytics"""
        st.markdown("### 📈 User Analytics")

        col1, col2 = st.columns(2)

        with col1:
            self._render_user_growth_chart()

        with col2:
            self._render_user_activity_patterns()

    def _render_user_growth_chart(self):
        """Render user growth over time"""
        st.markdown("#### 📊 User Growth Trend")

        growth_data = self._get_user_growth_data()

        if growth_data:
            df = pd.DataFrame(growth_data)
            df['date'] = pd.to_datetime(df['date'])

            fig = go.Figure()

            # Add cumulative users
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['cumulative_users'],
                name='Total Users',
                fill='tonexty',
                line=dict(color='#00ff41', width=2),
                fillcolor='rgba(0, 255, 65, 0.1)'
            ))

            # Add daily new users
            fig.add_trace(go.Bar(
                x=df['date'],
                y=df['new_users'],
                name='New Users',
                marker_color='#ffaa00',
                opacity=0.7
            ))

            fig.update_layout(
                title='User Growth (Last 30 Days)',
                xaxis=dict(title='Date'),
                yaxis=dict(title='Number of Users'),
                template='plotly_dark',
                height=350,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user growth data available")

    def _render_user_activity_patterns(self):
        """Render user activity patterns"""
        st.markdown("#### ⏰ User Activity Patterns")

        activity_data = self._get_user_activity_patterns()

        if activity_data:
            df = pd.DataFrame(activity_data)

            # Create heatmap of activity by hour and day
            pivot_df = df.pivot_table(
                values='activity_count',
                index='hour',
                columns='day',
                aggfunc='mean'
            )

            fig = px.imshow(
                pivot_df,
                title='User Activity Heatmap (Last 7 Days)',
                labels=dict(x="Day", y="Hour", color="Activity Count"),
                color_continuous_scale='Viridis',
                x=range(7),
                y=range(24)
            )

            fig.update_layout(
                template='plotly_dark',
                height=350,
                xaxis=dict(title='Day of Week', ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], tickvals=list(range(7))),
                yaxis=dict(title='Hour of Day')
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No activity pattern data available")

    def render_team_analysis(self):
        """Render team usage analysis"""
        st.markdown("### 👨‍👩‍👧‍👦 Team Usage Analysis")

        team_data = self._get_team_data()

        if team_data:
            # Team selector
            team_names = [team['team_name'] for team in team_data]
            selected_team = st.selectbox("Select Team", team_names)

            # Display selected team data
            selected_team_data = next(team for team in team_data if team['team_name'] == selected_team)

            col1, col2 = st.columns(2)

            with col1:
                self._render_team_metrics(selected_team_data)

            with col2:
                self._render_team_member_breakdown(selected_team_data)
        else:
            st.info("No team data available")

    def _render_team_metrics(self, team_data):
        """Render team metrics"""
        st.markdown(f"#### 🏢 {team_data['team_name']} Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Members",
                value=team_data['member_count']
            )

        with col2:
            st.metric(
                label="Total Spend",
                value=f"${team_data['total_spend']:,.2f}"
            )

        with col3:
            st.metric(
                label="Total Requests",
                value=f"{team_data['total_requests']:,}"
            )

        with col4:
            st.metric(
                label="Avg per Member",
                value=f"${team_data['avg_per_member']:.2f}"
            )

    def _render_team_member_breakdown(self, team_data):
        """Render team member breakdown"""
        st.markdown("#### 👤 Team Members")

        if team_data['members']:
            df = pd.DataFrame(team_data['members'])

            fig = px.bar(
                df,
                x='spend',
                y='user_id',
                title='Team Member Spending',
                labels={'spend': 'Spend ($)', 'user_id': 'User ID'},
                orientation='h',
                color_discrete_sequence=['#00ff41']
            )

            fig.update_layout(
                template='plotly_dark',
                height=300,
                xaxis=dict(title='Spend ($)'),
                yaxis=dict(title='User ID')
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No team member data available")

    def render_user_settings(self):
        """Render user management settings"""
        st.markdown("### ⚙️ User Management Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏷️ User Groups")

            # User group management
            group_col1, group_col2 = st.columns(2)

            with group_col1:
                new_group_name = st.text_input("New Group Name")
            with group_col2:
                if st.button("Create Group") and new_group_name:
                    self._create_user_group(new_group_name)
                    st.success(f"Group '{new_group_name}' created!")

            # Display existing groups
            groups = self._get_user_groups()
            if groups:
                st.markdown("**Existing Groups**")
                for group in groups:
                    st.markdown(f"- {group} ({len(self._get_group_members(group))} members)")

        with col2:
            st.markdown("#### 🚨 User Alerts")

            alert_threshold = st.number_input(
                "Alert Threshold ($/day)",
                value=100.0,
                min_value=10.0,
                max_value=1000.0,
                step=10.0
            )

            auto_block_threshold = st.number_input(
                "Auto-block Threshold ($/day)",
                value=500.0,
                min_value=50.0,
                max_value=5000.0,
                step=50.0
            )

            if st.button("Update Thresholds"):
                self._update_user_thresholds(alert_threshold, auto_block_threshold)
                st.success("User thresholds updated!")

        # User search and management
        st.markdown("#### 🔍 User Search & Management")

        search_col1, search_col2, search_col3 = st.columns(3)

        with search_col1:
            search_user = st.text_input("Search User ID")

        with search_col2:
            action = st.selectbox("Action", ["View Details", "Modify Limits", "Send Alert", "Block"])

        with search_col3:
            if st.button("Execute Action") and search_user:
                self._execute_user_action(search_user, action)

    def _get_user_metrics(self) -> Dict[str, Any]:
        """Get comprehensive user metrics"""
        return {
            'total_users': np.random.randint(500, 2000),
            'new_users_today': np.random.randint(5, 25),
            'active_users': np.random.randint(300, 1500),
            'active_rate': np.random.uniform(60, 80),
            'avg_spend_per_user': np.random.uniform(5, 50),
            'avg_spend_change': np.random.uniform(-5, 10),
            'avg_requests_per_user': np.random.uniform(10, 100),
            'request_growth': np.random.uniform(-10, 25)
        }

    def _get_top_users_spending(self) -> pd.DataFrame:
        """Get top users by spending"""
        if not self.dashboard.redis_client:
            return pd.DataFrame()

        try:
            user_spend_keys = self.dashboard.redis_client.keys("helix:spend:user:*")
            user_data = []

            for key in user_spend_keys[:100]:  # Limit to top 100 for performance
                user_id = key.decode().replace("helix:spend:user:", "")
                daily_spend = self.dashboard.get_redis_sorted_set(key.decode())

                if daily_spend:
                    total_spend = sum(daily_spend.values())
                    total_requests = len(daily_spend)
                    days_active = len(daily_spend)

                    user_data.append({
                        'user_id': user_id,
                        'total_spend': total_spend,
                        'total_requests': total_requests,
                        'avg_cost_per_request': total_spend / max(total_requests, 1),
                        'days_active': days_active,
                        'avg_daily_spend': total_spend / max(days_active, 1)
                    })

            df = pd.DataFrame(user_data)
            return df.sort_values('total_spend', ascending=False) if not df.empty else df

        except Exception as e:
            st.error(f"Error fetching user spending data: {e}")
            return pd.DataFrame()

    def _get_top_users_usage(self) -> pd.DataFrame:
        """Get top users by usage metrics"""
        spending_data = self._get_top_users_spending()

        if not spending_data.empty:
            # Simulate additional usage metrics
            usage_data = []
            for _, row in spending_data.iterrows():
                usage_data.append({
                    'user_id': row['user_id'],
                    'total_requests': row['total_requests'],
                    'total_tokens': row['total_requests'] * np.random.uniform(100, 1000),
                    'avg_latency': np.random.uniform(50, 500),
                    'success_rate': np.random.uniform(95, 100),
                    'cache_hit_rate': np.random.uniform(20, 60)
                })
            return pd.DataFrame(usage_data)
        return pd.DataFrame()

    def _get_user_growth_data(self) -> List[Dict]:
        """Get user growth data for the last 30 days"""
        growth_data = []
        cumulative = 0

        for i in range(30, 0, -1):
            date = datetime.now() - timedelta(days=i)
            new_users = np.random.randint(5, 30)
            cumulative += new_users

            growth_data.append({
                'date': date,
                'new_users': new_users,
                'cumulative_users': cumulative
            })
        return growth_data

    def _get_user_activity_patterns(self) -> List[Dict]:
        """Get user activity patterns by hour and day"""
        activity_data = []

        for hour in range(24):
            for day in range(7):
                activity_data.append({
                    'hour': hour,
                    'day': day,
                    'activity_count': np.random.randint(0, 50) * (1 - abs(12 - hour) / 12)  # Peak at noon
                })
        return activity_data

    def _get_team_data(self) -> List[Dict]:
        """Get team usage data"""
        # Mock team data
        teams = []
        team_names = ['Engineering', 'Product', 'Sales', 'Marketing', 'Support']

        for team_name in team_names:
            members = []
            member_count = np.random.randint(5, 20)

            for i in range(member_count):
                members.append({
                    'user_id': f'{team_name.lower()}_user_{i:02d}',
                    'spend': np.random.uniform(5, 100)
                })

            total_spend = sum(m['spend'] for m in members)
            total_requests = np.random.randint(100, 1000) * member_count

            teams.append({
                'team_name': team_name,
                'member_count': member_count,
                'total_spend': total_spend,
                'total_requests': total_requests,
                'avg_per_member': total_spend / member_count,
                'members': members
            })

        return teams

    def _get_user_groups(self) -> List[str]:
        """Get user groups"""
        return ['Premium Users', 'Enterprise', 'Development', 'Testing', 'Trial Users']

    def _get_group_members(self, group_name: str) -> List[str]:
        """Get members of a user group"""
        # Mock data
        member_count = np.random.randint(5, 50)
        return [f'{group_name.lower().replace(" ", "_")}_user_{i:03d}' for i in range(member_count)]

    def _create_user_group(self, group_name: str):
        """Create a new user group"""
        # In production, save to database
        pass

    def _update_user_thresholds(self, alert_threshold: float, auto_block_threshold: float):
        """Update user alert thresholds"""
        # In production, save to configuration
        pass

    def _execute_user_action(self, user_id: str, action: str):
        """Execute action on a user"""
        # In production, implement actual actions
        st.info(f"Executed '{action}' on user '{user_id}'")

    def render(self):
        """Main render method for user management page"""
        self.render_user_overview()
        st.markdown("---")
        self.render_top_users()
        st.markdown("---")
        self.render_user_analytics()
        st.markdown("---")
        self.render_team_analysis()
        st.markdown("---")
        self.render_user_settings()