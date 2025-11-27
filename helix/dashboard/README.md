# Helix AI Gateway Dashboard

🔮 **Comprehensive real-time monitoring and analytics for Helix AI Gateway**

## Overview

The Helix Dashboard provides enterprise-grade monitoring and management capabilities for the AI Gateway, featuring:

- **Real-time Metrics**: Live monitoring of costs, cache performance, and usage patterns
- **Multi-page Interface**: Organized sections for different aspects of system monitoring
- **Interactive Visualizations**: Rich charts and graphs powered by Plotly
- **Export Functionality**: Download reports in multiple formats
- **Alert System**: Configurable alerts for critical events
- **Responsive Design**: Works on desktop and mobile devices

## Features

### 📊 Main Dashboard Pages

1. **Overview** (`🏠 Overview`)
   - Real-time key metrics (cost savings, cache hit rate, latency, requests)
   - Performance trend charts
   - Active alerts and notifications
   - System health indicators

2. **Cost Analysis** (`💰 Cost Analysis`)
   - Spending trends and projections
   - Budget management and alerts
   - Model-wise cost breakdown
   - User spending analytics
   - Export financial reports

3. **Cache Performance** (`🚀 Cache Performance`)
   - Cache hit rate analytics (exact + semantic)
   - Performance optimization recommendations
   - Similarity score distributions
   - Cache type breakdowns
   - Cache management controls

4. **Security** (`🔒 Security`)
   - PII incident monitoring
   - Entity type breakdowns
   - Compliance score tracking
   - Regulatory status monitoring
   - Security settings and alerts

5. **User Management** (`👥 User Management`)
   - Top users by spending and usage
   - Team usage analytics
   - User growth trends
   - Activity pattern analysis
   - User group management

6. **System Health** (`⚙️ System Health`)
   - Resource monitoring (CPU, memory, disk)
   - Redis performance metrics
   - Service health status
   - Log monitoring and error tracking
   - System alerts and notifications

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Redis server (for data storage and caching)
- Node.js (for additional tools, optional)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/helix.git
   cd helix/helix/dashboard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Redis:**
   ```bash
   # Using Docker
   docker run -d --name helix_redis -p 6379:6379 redis/redis-stack:latest

   # Or install locally
   # Follow Redis installation guide for your OS
   ```

4. **Configure environment:**
   ```bash
   export REDIS_URL="redis://localhost:6379"
   export REFRESH_INTERVAL=5
   export THEME=dark
   ```

5. **Run the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

6. **Open your browser:**
   Navigate to `http://localhost:8501`

### Docker Deployment

Using Docker Compose (recommended):

```bash
# From the helix root directory
docker-compose -f docker-compose.dashboard.yml up -d
```

This will start:
- Redis server on port 6379
- Dashboard on port 8501
- Persistent volumes for data storage

## 📋 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `REFRESH_INTERVAL` | `5` | Auto-refresh interval in seconds |
| `THEME` | `dark` | Dashboard theme (`dark`/`light`) |
| `ENABLE_EXPORTS` | `true` | Enable data export functionality |
| `ENABLE_ALERTS` | `true` | Enable alert system |
| `MAX_DATA_POINTS` | `1000` | Maximum data points for charts |
| `TIMEZONE_OFFSET` | `0` | Timezone offset from UTC |

### Redis Configuration

The dashboard expects Redis to be configured with the following data structures:

```
# Exact cache (hash)
HSET helix:exact:<sha256(prompt+model)>
     response "..."
     model "gpt-4o"
     cost_usd "0.00032"
     created_at 1732642151

# Semantic vector index (RedisSearch)
FT.CREATE idx:semantic
  ON HASH
  PREFIX 1 "helix:vector:"
  SCHEMA
    prompt TEXT
    model TEXT
    response_json TEXT
    vector VECTOR HNSW 12 DIM 384 DISTANCE_METRIC COSINE

# Request metrics
SET helix:requests:total 12345
SET helix:requests:cache_hits 6789

# Cost tracking (sorted sets)
ZINCRBY helix:spend:user:<user_id> <amount> <date>
ZINCRBY helix:spend:total <amount> <date>

# PII incidents (list)
RPUSH helix:pii:incidents "{\"user_id\":\"123\",\"entity_type\":\"CREDIT_CARD\",\"timestamp\":\"2024-01-01T00:00:00Z\"}"
```

## 🔧 Customization

### Adding New Pages

1. Create a new page file in `pages/` directory:
   ```python
   # pages/custom_page.py
   import streamlit as st

   class CustomPage:
       def __init__(self, dashboard):
           self.dashboard = dashboard

       def render(self):
           st.markdown("### Custom Page")
           st.write("Your custom content here")

   def __init__(self, dashboard):
       self.dashboard = dashboard

   def render(self):
       st.markdown("### Custom Page")
       st.write("Your custom content here")
   ```

2. Add the page to the navigation in `dashboard.py`:
   ```python
   page_options = {
       # ... existing pages ...
       "🔧 Custom Page": "custom_page",
   }

   # Import and initialize the page
   from pages.custom_page import CustomPage
   pages['custom_page'] = CustomPage(self)
   ```

### Modifying Charts

All charts use Plotly Express and Graph Objects. You can customize:

- Colors and themes
- Chart types
- Data ranges
- Labels and annotations

Example customization:
```python
import plotly.express as px

# Custom color scheme
fig = px.line(
    df,
    x='timestamp',
    y='value',
    color_discrete_sequence=['#00ff41', '#ffaa00']
)

# Update layout
fig.update_layout(
    template='plotly_dark',
    height=400,
    showlegend=True
)
```

## 📊 Data Sources

The dashboard integrates with:

1. **Redis**: Real-time metrics and caching data
2. **LiteLLM Hook System**: Request processing and analytics
3. **System Metrics**: CPU, memory, disk usage via `psutil`
4. **PII Detection**: Presidio integration for security monitoring

## 🚨 Alerts and Notifications

### Alert Types

- **Critical**: System failures, high latency, budget exceeded
- **Warning**: Performance degradation, low cache hit rates
- **Info**: Status updates, user actions

### Alert Configuration

Alerts can be configured through the dashboard or via environment variables:

```python
# In the Security page
alert_threshold = st.number_input("Alert Threshold (incidents/hour)", value=10)
notification_methods = st.multiselect("Notification Methods", ["Email", "Slack", "Webhook"])
```

## 📤 Export Functionality

The dashboard supports exporting data in multiple formats:

- **CSV**: Spreadsheet-compatible format
- **JSON**: Machine-readable format
- **Excel**: Rich spreadsheet with multiple sheets

### Export Types

- Spending data and cost analysis
- User analytics and usage patterns
- Cache performance metrics
- Security and PII incident reports
- System health logs

## 🔒 Security Considerations

- **Authentication**: Configure as needed for your environment
- **Data Privacy**: All PII is automatically redacted before display
- **Access Control**: Implement role-based access if required
- **Audit Logging**: All actions and access attempts are logged

## 🐛 Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check Redis status
   redis-cli ping

   # Verify Redis URL
   echo $REDIS_URL
   ```

2. **Page Import Errors**
   - Ensure running from the correct directory
   - Check Python path includes the dashboard directory
   - Verify all dependencies are installed

3. **Charts Not Loading**
   - Check internet connection for external assets
   - Verify Plotly version compatibility
   - Clear browser cache

4. **Slow Performance**
   - Increase Redis connection pool size
   - Optimize data queries
   - Reduce refresh interval

### Logs

Check the logs for detailed error information:

```bash
# Streamlit logs
tail -f logs/streamlit.log

# Application logs
tail -f logs/app.log

# Redis logs
docker logs helix_redis
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Lint code
flake8 dashboard/
black dashboard/
```

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](../../LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Link to full documentation]
- **Issues**: [GitHub Issues](https://github.com/your-org/helix/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/helix/discussions)
- **Email**: support@yourorg.com

## 🔮 Roadmap

- [ ] Real-time WebSocket connections for live updates
- [ ] Mobile app companion
- [ ] Advanced ML-powered anomaly detection
- [ ] Integration with popular monitoring systems (Prometheus, Grafana)
- [ ] Multi-tenant support
- [ ] Advanced user roles and permissions

---

**Built with ❤️ for the Helix AI Gateway Community**