# GitHub Actions Troubleshooting Guide

## Quick Fix for Current Issue

The GitHub Actions workflow failed due to a dependency issue. Here's the immediate fix:

### ✅ **Fixed Issue: `investiny>=0.8.0` not found**

**Problem**: The package `investiny>=0.8.0` doesn't exist (latest is 0.7.2)
**Solution**: Replaced with `yfinance>=0.2.0` which is more reliable and actively maintained

**Updated requirements.txt:**
```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
pandas-datareader>=0.10.0
yfinance>=0.2.0
```

## Common GitHub Actions Issues & Solutions

### 1. **Dependency Installation Failures**

**Symptoms:**
- `ERROR: Could not find a version that satisfies the requirement`
- `ERROR: No matching distribution found`

**Solutions:**
```bash
# Check available versions locally
pip index versions package_name

# Use compatible versions
pip install "package_name>=min_version,<max_version"

# Pin exact versions if needed
pip install package_name==1.2.3
```

### 2. **File Path Issues (Linux vs macOS)**

**Problem**: `stat -f%z` command fails on Linux (GitHub Actions uses Ubuntu)
**Solution**: Use portable commands
```bash
# Instead of: stat -f%z "$file" (macOS only)
# Use: stat -c%s "$file" 2>/dev/null || wc -c < "$file"
```

### 3. **Market Hours Check**

**Issue**: Workflow runs on weekends unnecessarily
**Solution**: Enhanced market check
```yaml
current_day=$(date +%u)  # 1=Monday, 7=Sunday
if [[ $current_day -le 5 ]] || [[ "$force_update" == "true" ]]; then
  echo "market_open=true" >> $GITHUB_OUTPUT
fi
```

### 4. **GitHub Pages Deployment Fails**

**Common Causes:**
- Pages not enabled in repository settings
- Wrong source configuration
- Missing permissions

**Fix Steps:**
1. **Repository Settings** → **Pages**
2. Source: "GitHub Actions" (not "Deploy from branch")
3. **Settings** → **Actions** → **General**
   - ✅ Read and write permissions
   - ✅ Allow GitHub Actions to create PRs

### 5. **Python Import Errors**

**Issue**: `ModuleNotFoundError: No module named 'scripts'`
**Solution**: Fix import paths in scripts
```python
# Instead of: from scripts.module import Class
# Use: from .module import Class
# Or: import sys; sys.path.append('.')
```

### 6. **Permissions Denied**

**Issue**: Git push fails with permission errors
**Solution**: Check workflow permissions
```yaml
permissions:
  contents: write    # Required for git push
  pages: write      # Required for Pages deployment
  id-token: write   # Required for OIDC
```

## Manual Recovery Steps

### If Workflow Completely Fails:

1. **Run Locally First:**
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/NVII.git
cd NVII

# Install dependencies
pip install -r requirements.txt

# Test dashboard generation
python3 scripts/weekly_update.py

# Verify all files created
ls -la dashboard/ docs/
```

2. **Fix and Push:**
```bash
# Add fixes
git add .
git commit -m "Fix GitHub Actions workflow"
git push
```

3. **Test Workflow:**
- Go to **Actions** tab
- Select "Manual NVII Dashboard Update"
- Click "Run workflow" with default settings
- Monitor execution logs

### Emergency Dashboard Update:

```bash
# Quick manual update
python3 scripts/dashboard_generator.py

# Check generated files
ls -la dashboard/index.html

# If successful, commit
git add dashboard/
git commit -m "Emergency dashboard update"
git push
```

## Debugging GitHub Actions

### View Detailed Logs:
1. Repository **Actions** tab
2. Click on failed workflow run
3. Expand each step to see detailed output
4. Look for red ❌ marks indicating failures

### Download Artifacts:
- Failed runs often upload artifacts with logs
- Download for local debugging
- Check `logs/weekly_update.log` for Python errors

### Test Locally Before Push:
```bash
# Simulate GitHub Actions environment
export GITHUB_WORKSPACE=$(pwd)
export GITHUB_ACTOR="test-user"
export PYTHONUNBUFFERED=1

# Run update script
python3 scripts/weekly_update.py

# Check exit code
echo "Exit code: $?"
```

## Monitoring & Maintenance

### Success Indicators:
- ✅ Green checkmarks in Actions tab
- ✅ Weekly commits appear automatically  
- ✅ Dashboard URL shows latest data
- ✅ No failure issues created

### Set Up Notifications:
1. **Repository Settings** → **Notifications**
2. Enable email notifications for:
   - Actions failures
   - Push events
   - Issues created

### Regular Maintenance:
- Review dependency updates monthly
- Check GitHub Actions usage quota
- Monitor dashboard performance
- Update Python version annually

## Contact & Support

### GitHub Actions Documentation:
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Troubleshooting](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows)

### Repository Issues:
- Create GitHub issue for persistent problems
- Include workflow logs and error messages
- Tag with `bug`, `automation`, `github-actions`

---

## Quick Reference Commands

```bash
# Test requirements.txt locally
pip install -r requirements.txt

# Run dashboard update
python3 scripts/weekly_update.py

# Check git status
git status

# Push fixes
git add . && git commit -m "Fix workflow" && git push

# View GitHub Pages
open https://YOUR_USERNAME.github.io/NVII/
```

The workflow should now run successfully with the fixed dependencies!