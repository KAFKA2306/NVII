# GitHub Automation Setup Guide

## Overview

This guide sets up automated weekly updates for the NVII dashboard using GitHub Actions, with automatic deployment to GitHub Pages.

## Features

### 🤖 **Automated Weekly Updates**
- Runs every Monday at 9:00 AM UTC
- Fetches latest NVII and NVDA data
- Updates all dashboard components
- Commits and pushes changes automatically
- Deploys to GitHub Pages

### 🔧 **Manual Updates**
- On-demand dashboard updates
- Multiple update types (full, dashboard-only, data-only)
- Custom commit messages
- Immediate deployment option

### 📊 **GitHub Pages Dashboard**
- Professional web interface
- Public access at `https://username.github.io/NVII/`
- Automatic SSL certificate
- Mobile-responsive design

## Setup Instructions

### 1. Create GitHub Repository

```bash
# Initialize repository
git init
git add .
git commit -m "🚀 Initial NVII Dashboard Setup"

# Add GitHub remote (replace with your username)
git remote add origin https://github.com/YOUR_USERNAME/NVII.git
git branch -M main
git push -u origin main
```

### 2. Enable GitHub Pages

1. Go to repository **Settings** → **Pages**
2. Select **Source**: "GitHub Actions"
3. The dashboard will be available at: `https://YOUR_USERNAME.github.io/NVII/`

### 3. Configure Repository Permissions

**Required Permissions:**
1. **Settings** → **Actions** → **General**
   - ✅ Allow all actions and reusable workflows
   - ✅ Read and write permissions for GITHUB_TOKEN
   - ✅ Allow GitHub Actions to create and approve pull requests

2. **Settings** → **Environments**
   - Create environment: `github-pages`
   - No protection rules needed for public repos

### 4. Test the Setup

**Manual Test:**
1. Go to **Actions** tab
2. Select "Manual NVII Dashboard Update"
3. Click "Run workflow"
4. Choose update type: "full"
5. Monitor the execution

**Verify Results:**
- Check dashboard at your GitHub Pages URL
- Verify files in `dashboard/` and `docs/` directories
- Review commit history for automated updates

## Workflow Details

### Weekly Automated Update (`weekly-update.yml`)

**Schedule:** Every Monday at 9:00 AM UTC

**Process:**
1. **Market Check**: Verifies it's a weekday
2. **Data Fetch**: Gets latest NVII/NVDA prices
3. **Analysis**: Runs comprehensive dashboard update
4. **Validation**: Ensures all files generated correctly
5. **Commit**: Pushes changes with detailed message
6. **Deploy**: Updates GitHub Pages automatically
7. **Notify**: Creates issue if workflow fails

**Triggers:**
- `schedule`: Weekly on Mondays
- `workflow_dispatch`: Manual trigger with options
- `push`: On changes to scripts or workflows

### Manual Update (`manual-update.yml`)

**Options:**
- **Full Update**: Complete dashboard regeneration
- **Dashboard Only**: HTML dashboard refresh
- **Data Only**: CSV and analysis updates

**Features:**
- Custom commit messages
- Optional GitHub Pages deployment
- Detailed execution summary
- Artifact upload for debugging

## File Structure

```
NVII/
├── .github/
│   ├── workflows/
│   │   ├── weekly-update.yml      # Automated weekly updates
│   │   └── manual-update.yml      # Manual update options
│   └── FUNDING.yml               # Sponsorship info (optional)
├── dashboard/
│   └── index.html               # Main dashboard (deployed to Pages)
├── docs/
│   ├── *.csv                    # Data files
│   ├── *.md                     # Analysis reports
│   └── *.png                    # Charts and visualizations
├── scripts/
│   ├── dashboard_generator.py   # Core dashboard engine
│   ├── weekly_update.py         # Update orchestrator
│   └── total_returns_comparison.py
└── logs/
    └── weekly_update.log        # Execution logs
```

## Monitoring and Maintenance

### Success Indicators
- ✅ Weekly commits appear automatically
- ✅ Dashboard URL shows latest data
- ✅ No failure issues created
- ✅ CSV files contain current week's data

### Troubleshooting

**Common Issues:**

1. **Workflow Fails**
   - Check Actions tab for error logs
   - Verify internet connectivity to Yahoo Finance
   - Ensure required files exist

2. **GitHub Pages Not Updating**
   - Check Pages deployment in Actions
   - Verify Pages source is set to "GitHub Actions"
   - Allow 1-2 minutes for CDN refresh

3. **Missing Data**
   - Verify market hours (weekdays only)
   - Check if NVII/NVDA tickers are correct
   - Review API rate limits

**Manual Recovery:**
```bash
# Clone and run locally
git clone https://github.com/YOUR_USERNAME/NVII.git
cd NVII
pip install -r requirements.txt
python3 scripts/weekly_update.py

# Push manually if needed
git add -A
git commit -m "Manual update recovery"
git push
```

### Workflow Notifications

**Automatic Issue Creation:**
- Failed workflows create GitHub issues
- Include error details and recovery steps
- Tagged with `bug`, `automation`, `dashboard`

**Success Indicators:**
- Commit messages include update details
- Workflow artifacts available for 30 days
- Release tags created for weekly milestones

## Customization

### Modify Update Schedule

Edit `.github/workflows/weekly-update.yml`:
```yaml
schedule:
  # Run every day at 2:00 PM UTC
  - cron: '0 14 * * *'
  
  # Run twice per week (Monday and Thursday)
  - cron: '0 9 * * MON,THU'
```

### Add Environment Variables

For API keys or custom settings:
1. **Settings** → **Secrets and variables** → **Actions**
2. Add repository secrets
3. Reference in workflow: `${{ secrets.SECRET_NAME }}`

### Custom Deployment

Replace GitHub Pages with custom hosting:
```yaml
- name: Deploy to Custom Server
  run: |
    scp -r dashboard/* user@server:/var/www/nvii/
```

## Security Considerations

- ✅ No sensitive data in repository
- ✅ API keys stored as GitHub secrets
- ✅ Limited workflow permissions
- ✅ Public dashboard (no authentication needed)

## Cost Considerations

- ✅ GitHub Actions: 2,000 minutes/month free
- ✅ GitHub Pages: Unlimited public sites
- ✅ Storage: 1GB free repository storage
- ✅ Weekly updates: ~5 minutes/week = ~20 minutes/month

**Estimated Usage:** Well within free tier limits.

---

## Quick Start Checklist

- [ ] Create GitHub repository
- [ ] Push NVII code to repository
- [ ] Enable GitHub Pages (Actions source)
- [ ] Set workflow permissions (read/write)
- [ ] Test manual workflow execution
- [ ] Verify dashboard deployment
- [ ] Monitor first automated run (next Monday)

**Dashboard URL:** `https://YOUR_USERNAME.github.io/NVII/`

Once set up, your NVII dashboard will automatically update every Monday with the latest market data, analysis, and visualizations!