# NVII Dashboard System

## Overview

Professional, weekly-updatable dashboard for comprehensive NVII (REX NVDA Growth & Income ETF) analysis and tracking.

## Features

### 📊 **Interactive Dashboard**
- Real-time performance metrics
- NVII vs NVDA comparison charts
- Weekly dividend tracking
- Risk analysis and alerts
- Professional HTML interface

### 📈 **Performance Tracking**
- Total return vs capital return analysis
- Dividend contribution visualization
- Risk-adjusted returns (Sharpe ratio)
- Maximum drawdown analysis
- Weekly performance heatmaps

### 💰 **Dividend Analysis**
- Weekly dividend timeline
- Projected annual yield
- Dividend consistency scoring
- Trend analysis (increasing/stable/decreasing)

### ⚠️ **Alert System**
- Performance alerts (vs NVDA)
- Volatility warnings
- Dividend yield notifications
- Risk metric changes

## Quick Start

### Single Command Update
```bash
# Update everything weekly
python3 scripts/weekly_update.py
```

### Manual Dashboard Generation
```bash
# Generate dashboard only
python3 scripts/dashboard_generator.py
```

### View Dashboard
```bash
# Open in browser
open dashboard/index.html
# or
firefox dashboard/index.html
```

## File Structure

```
NVII/
├── dashboard/
│   ├── index.html                 # Main dashboard
│   └── historical_snapshots.jsonl # Historical data
├── scripts/
│   ├── dashboard_generator.py     # Dashboard generator
│   ├── weekly_update.py          # Weekly update script
│   └── total_returns_comparison.py
├── docs/                          # Analysis reports & CSVs
└── logs/                          # Update logs
```

## Dashboard Components

### 1. **Header Section**
- Current NVII and NVDA prices
- Last update timestamp
- Key performance metrics

### 2. **Alerts & Insights**
- Automated alerts for significant changes
- Performance notifications
- Risk warnings

### 3. **Performance Analysis**
- NVII vs NVDA cumulative returns chart
- Weekly performance heatmap
- Total return vs capital return comparison

### 4. **Dividend Analysis**
- Weekly dividend payment timeline
- Projected annual yield calculations
- Dividend consistency metrics

### 5. **Risk Analysis**
- Volatility comparison charts
- Maximum drawdown analysis
- Risk-adjusted return metrics

### 6. **Market Data Table**
- Detailed side-by-side comparison
- Daily/weekly/monthly changes
- Volume and technical indicators

## Update Schedule

### Weekly Updates (Recommended)
```bash
# Every Monday morning
python3 scripts/weekly_update.py
```

### What Gets Updated:
- Current market prices
- Dividend payments
- Performance metrics
- Risk calculations
- Charts and visualizations
- Alert notifications

## Data Sources

- **Primary**: Yahoo Finance (via yfinance)
- **Backup**: Manual data entry support
- **Frequency**: Real-time during market hours
- **Historical**: 6-month rolling window

## Customization

### Modify Alert Thresholds
Edit `scripts/dashboard_generator.py`:
```python
self.config = {
    'dividend_target': 0.30,    # 30% target yield
    'risk_free_rate': 0.045,    # 4.5% risk-free rate
    'target_leverage': 1.25,    # 1.25x leverage target
}
```

### Add Custom Metrics
1. Modify `analyze_performance()` method
2. Add new chart in `generate_charts()`
3. Update HTML template in `_generate_*_section()`

## Troubleshooting

### Common Issues

**Dashboard not updating:**
```bash
# Check logs
cat logs/weekly_update.log

# Manual regeneration
python3 scripts/dashboard_generator.py
```

**Missing data:**
```bash
# Verify internet connection
# Check if markets are open
# Run total returns comparison first
python3 scripts/total_returns_comparison.py
```

**Charts not displaying:**
- Ensure matplotlib is installed
- Check browser console for errors
- Verify image data in HTML source

### Manual Data Entry

If automated data fetch fails, you can manually update:
1. Edit CSV files in `docs/` directory
2. Run dashboard generator
3. Review alerts for data quality issues

## Advanced Usage

### Historical Analysis
```bash
# View historical snapshots
cat dashboard/historical_snapshots.jsonl | jq '.'
```

### Custom Time Periods
Modify date ranges in `dashboard_generator.py`:
```python
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
```

### Export Data
All data is available in CSV format in `docs/` directory:
- `nvii_nvda_summary_stats.csv`
- `nvii_nvda_daily_data.csv`
- `nvii_detailed_data.csv`
- `nvda_detailed_data.csv`

## Integration

### API Access
Dashboard data can be accessed programmatically:
```python
import json
with open('dashboard/historical_snapshots.jsonl') as f:
    data = [json.loads(line) for line in f]
```

### External Tools
- Import CSVs into Excel/Google Sheets
- Use data in R/Python analysis
- Connect to business intelligence tools

## Support

### Documentation
- See `CLAUDE.md` for project overview
- Check `docs/` for analysis reports
- Review logs for debugging

### Updates
- Dashboard system is self-updating
- Weekly cadence recommended
- Manual updates supported

---

**Last Updated:** 2025-08-17  
**Version:** 1.0  
**Maintainer:** NVII Analysis System