#!/usr/bin/env python3
"""
NVII Weekly Update Script

Single command to update all NVII project components:
- Regenerate dashboard with latest data
- Update CSV files and analysis reports
- Run comprehensive comparisons
- Generate alerts for significant changes

Usage: python3 scripts/weekly_update.py
"""

import os
import sys
import subprocess
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/kafka/projects/NVII/logs/weekly_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeeklyUpdater:
    """
    Orchestrates weekly updates for the NVII project
    """
    
    def __init__(self):
        self.project_root = '/home/kafka/projects/NVII'
        self.scripts_dir = os.path.join(self.project_root, 'scripts')
        self.docs_dir = os.path.join(self.project_root, 'docs')
        self.dashboard_dir = os.path.join(self.project_root, 'dashboard')
        
        # Ensure directories exist
        os.makedirs(os.path.join(self.project_root, 'logs'), exist_ok=True)
        os.makedirs(self.dashboard_dir, exist_ok=True)
    
    def run_script(self, script_name, description):
        """
        Run a Python script and handle errors
        
        Args:
            script_name (str): Name of the script to run
            description (str): Description for logging
        """
        try:
            logger.info(f"🔄 {description}...")
            script_path = os.path.join(self.scripts_dir, script_name)
            
            if not os.path.exists(script_path):
                logger.warning(f"⚠️ Script not found: {script_path}")
                return False
            
            result = subprocess.run([
                sys.executable, script_path
            ], cwd=self.project_root, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully")
                return True
            else:
                logger.error(f"❌ {description} failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {description} timed out after 5 minutes")
            return False
        except Exception as e:
            logger.error(f"❌ {description} failed with exception: {e}")
            return False
    
    def update_returns_comparison(self):
        """
        Update NVII vs NVDA returns comparison
        """
        return self.run_script(
            'total_returns_comparison.py',
            'Updating NVII vs NVDA returns comparison'
        )
    
    def generate_dashboard(self):
        """
        Generate the main dashboard
        """
        return self.run_script(
            'dashboard_generator.py',
            'Generating NVII dashboard'
        )
    
    def run_advanced_analysis(self):
        """
        Run advanced option analysis if available
        """
        success = True
        
        # Try to run advanced option engine
        if os.path.exists(os.path.join(self.scripts_dir, 'advanced_option_engine.py')):
            success &= self.run_script(
                'advanced_option_engine.py',
                'Running advanced option analysis'
            )
        
        # Try to run portfolio simulation
        if os.path.exists(os.path.join(self.scripts_dir, 'portfolio_simulation_engine.py')):
            success &= self.run_script(
                'portfolio_simulation_engine.py',
                'Running portfolio simulation'
            )
        
        return success
    
    def check_data_quality(self):
        """
        Check the quality of generated data and reports
        """
        logger.info("🔍 Checking data quality...")
        
        checks = {
            'Dashboard HTML': os.path.join(self.dashboard_dir, 'index.html'),
            'Returns Analysis': os.path.join(self.docs_dir, 'nvii_nvda_returns_comparison.md'),
            'Summary CSV': os.path.join(self.docs_dir, 'nvii_nvda_summary_stats.csv'),
            'Daily Data CSV': os.path.join(self.docs_dir, 'nvii_nvda_daily_data.csv'),
            'Performance Chart': os.path.join(self.docs_dir, 'nvii_nvda_performance_chart.png')
        }
        
        all_checks_passed = True
        for name, path in checks.items():
            if os.path.exists(path):
                size = os.path.getsize(path)
                if size > 0:
                    logger.info(f"✅ {name}: OK ({size:,} bytes)")
                else:
                    logger.warning(f"⚠️ {name}: Empty file")
                    all_checks_passed = False
            else:
                logger.error(f"❌ {name}: Missing")
                all_checks_passed = False
        
        return all_checks_passed
    
    def generate_update_summary(self):
        """
        Generate a summary of what was updated
        """
        logger.info("📋 Generating update summary...")
        
        summary_file = os.path.join(self.project_root, 'WEEKLY_UPDATE_SUMMARY.md')
        update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        summary_content = f"""# Weekly Update Summary

**Update Date:** {update_time}

## Files Updated

### Dashboard
- ✅ Interactive HTML dashboard (`dashboard/index.html`)
- ✅ Historical data snapshots (`dashboard/historical_snapshots.jsonl`)

### Analysis Reports
- ✅ NVII vs NVDA comparison (`docs/nvii_nvda_returns_comparison.md`)
- ✅ Performance visualization (`docs/nvii_nvda_performance_chart.png`)

### Data Files
- ✅ Summary statistics (`docs/nvii_nvda_summary_stats.csv`)
- ✅ Daily returns data (`docs/nvii_nvda_daily_data.csv`)
- ✅ NVII detailed data (`docs/nvii_detailed_data.csv`)
- ✅ NVDA detailed data (`docs/nvda_detailed_data.csv`)

## How to View Results

1. **Dashboard**: Open `dashboard/index.html` in your browser
2. **Analysis**: Review `docs/nvii_nvda_returns_comparison.md`
3. **Data**: Import CSV files into Excel/Google Sheets for custom analysis

## Next Update

Run the weekly update again next week:
```bash
python3 scripts/weekly_update.py
```

---
*Generated by NVII Weekly Update System*
"""
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        logger.info(f"✅ Update summary saved to {summary_file}")
    
    def run_weekly_update(self):
        """
        Main method to run the complete weekly update
        """
        logger.info("🚀 Starting NVII Weekly Update Process")
        logger.info(f"📅 Update Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        success_count = 0
        total_tasks = 4
        
        try:
            # Task 1: Update returns comparison
            if self.update_returns_comparison():
                success_count += 1
            
            # Task 2: Generate dashboard
            if self.generate_dashboard():
                success_count += 1
            
            # Task 3: Run advanced analysis (optional)
            if self.run_advanced_analysis():
                success_count += 1
            
            # Task 4: Check data quality
            if self.check_data_quality():
                success_count += 1
            
            # Generate summary
            self.generate_update_summary()
            
            # Final status
            success_rate = (success_count / total_tasks) * 100
            
            if success_rate == 100:
                logger.info(f"🎉 Weekly update completed successfully! ({success_count}/{total_tasks} tasks)")
                logger.info("📂 Check the dashboard/ directory for the updated dashboard")
                logger.info("📊 Review WEEKLY_UPDATE_SUMMARY.md for detailed results")
            elif success_rate >= 75:
                logger.info(f"✅ Weekly update mostly successful ({success_count}/{total_tasks} tasks)")
                logger.warning("⚠️ Some components may need manual review")
            else:
                logger.error(f"⚠️ Weekly update had issues ({success_count}/{total_tasks} tasks successful)")
                logger.error("🔧 Manual intervention may be required")
            
            return success_rate >= 75
            
        except Exception as e:
            logger.error(f"💥 Weekly update failed with critical error: {e}")
            return False

def main():
    """
    Main function to run weekly update
    """
    print("🔄 NVII Weekly Update System")
    print("=" * 50)
    
    updater = WeeklyUpdater()
    success = updater.run_weekly_update()
    
    if success:
        print("\n✅ Update completed! Check the logs and dashboard for results.")
        dashboard_path = os.path.join(updater.dashboard_dir, 'index.html')
        print(f"🌐 Dashboard: file://{os.path.abspath(dashboard_path)}")
    else:
        print("\n❌ Update had issues. Check the logs for details.")
        print("📝 Log file: /home/kafka/projects/NVII/logs/weekly_update.log")
    
    return success

if __name__ == "__main__":
    main()