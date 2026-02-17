#!/usr/bin/env python3
"""
Advanced CSV Data Analysis Tool for Experiment Results
Provides statistical analysis, visualizations, and hypothesis testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import os
import glob

class ExperimentCSVAnalyzer:
    def __init__(self, csv_file: str = None, log_directory: str = "/home/ascc304/carla_garage/experiment_logs/P002/EV"):
        """Initialize analyzer with CSV file"""
        self.log_directory = log_directory
        
        if csv_file is None:
            # Find the most recent CSV file in the experiment logs directory
            csv_pattern = os.path.join(log_directory, "experiment_analysis_*.csv")
            csv_files = glob.glob(csv_pattern)
            if csv_files:
                csv_file = max(csv_files, key=os.path.getctime)
                print(f"📊 Using most recent CSV: {os.path.basename(csv_file)}")
                print(f"   From directory: {log_directory}")
            else:
                raise FileNotFoundError(f"No experiment analysis CSV files found in {log_directory}!")
        
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.setup_analysis()
        
    def setup_analysis(self):
        """Setup data for analysis"""
        print(f"📈 Loaded data: {len(self.df)} trials")
        print(f"   Conditions: {self.df['condition'].unique()}")
        print(f"   Participants: {self.df['participant'].unique()}")
        
        # Clean condition names for better display
        self.df['condition_clean'] = self.df['condition'].str.replace('_P001', '').str.replace('_', ' ').str.title()
        
        # Convert string numbers to numeric
        numeric_columns = ['minimum_distance', 'safety_margin', 'safety_score', 'first_alert_distance', 
                          'peak_risk_reached', 'emergency_events_count', 'total_alerts', 'false_positive_risk']
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print("✅ Data prepared for analysis")

    # ------------------------------------------------------------------ #
    # EV-specific metrics (excluding maneuver delay)
    # ------------------------------------------------------------------ #
    def ev_metrics(self, g_threshold: float = 10.0, lateral_tol: float = 1.2):
        """
        Compute EV-only metrics:
          - unsafe_lane_change (bool) [if rear_gaps available]
          - min_rear_gap_at_lc (meters) [if rear_gaps available]
          - rear_vehicle_count_at_lc [if rear_gaps available]
          - ev_min_distance, ev_peak_risk, ev_alert_count, ev_first_alert_time (from runtime logs)
        Assumes the CSV contains per-trial columns:
          lane_change_time, target_lane ('left'/'right'), rear_gaps (list/str) [optional],
          ev_min_distance, ev_peak_risk, ev_alert_count, ev_first_alert_time (optional),
          and condition contains 'EV' to filter EV trials.
        """
        df = self.df.copy()
        ev_mask = df['condition'].str.contains('EV', case=False, na=False)
        ev_df = df[ev_mask].copy()

        # Parse rear_gaps if stored as stringified list
        def _parse_gaps(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return [float(v) for v in x]
            if isinstance(x, str):
                try:
                    # Allow comma-separated or repr of list
                    x_clean = x.strip().strip('[]')
                    if not x_clean:
                        return []
                    return [float(v) for v in x_clean.split(',')]
                except Exception:
                    return []
            return []

        if 'rear_gaps' in ev_df.columns:
            ev_df['rear_gaps'] = ev_df['rear_gaps'].apply(_parse_gaps)
        else:
            ev_df['rear_gaps'] = [[] for _ in range(len(ev_df))]

        min_gaps = []
        rv_counts = []
        unsafe_flags = []
        for gaps in ev_df['rear_gaps']:
            if gaps:
                mg = min(gaps)
                min_gaps.append(mg)
                rv_counts.append(len(gaps))
                unsafe_flags.append(mg < g_threshold)
            else:
                min_gaps.append(np.nan)
                rv_counts.append(0)
                unsafe_flags.append(False)

        ev_df['min_rear_gap_at_lc'] = min_gaps
        ev_df['rear_vehicle_count_at_lc'] = rv_counts
        ev_df['unsafe_lane_change'] = unsafe_flags

        # EV runtime metrics if present
        if 'ev_min_distance' in ev_df.columns:
            ev_df['ev_min_distance'] = pd.to_numeric(ev_df['ev_min_distance'], errors='coerce')
        if 'ev_peak_risk' in ev_df.columns:
            ev_df['ev_peak_risk'] = pd.to_numeric(ev_df['ev_peak_risk'], errors='coerce')
        if 'ev_alert_count' in ev_df.columns:
            ev_df['ev_alert_count'] = pd.to_numeric(ev_df['ev_alert_count'], errors='coerce')

        # Aggregate summaries
        summary = {
            'trials': len(ev_df),
            'unsafe_rate_pct': 100.0 * ev_df['unsafe_lane_change'].mean() if len(ev_df) else np.nan,
            'min_rear_gap_mean': np.nanmean(ev_df['min_rear_gap_at_lc']) if len(ev_df) else np.nan,
            'min_rear_gap_median': np.nanmedian(ev_df['min_rear_gap_at_lc']) if len(ev_df) else np.nan,
            'rear_vehicle_count_mean': np.nanmean(ev_df['rear_vehicle_count_at_lc']) if len(ev_df) else np.nan,
            'ev_min_distance_mean': np.nanmean(ev_df['ev_min_distance']) if 'ev_min_distance' in ev_df.columns else np.nan,
            'ev_peak_risk_mean': np.nanmean(ev_df['ev_peak_risk']) if 'ev_peak_risk' in ev_df.columns else np.nan,
            'ev_alert_count_mean': np.nanmean(ev_df['ev_alert_count']) if 'ev_alert_count' in ev_df.columns else np.nan,
        }

        print("\n=== EV Scenario Metrics (position-only rear-gap check) ===")
        print(f"Trials: {summary['trials']}")
        print(f"Unsafe lane-change rate: {summary['unsafe_rate_pct']:.1f}% (gap<thr={g_threshold} m)")
        print(f"Min rear gap at LC: mean {summary['min_rear_gap_mean']:.2f} m, median {summary['min_rear_gap_median']:.2f} m")
        print(f"Rear vehicle count at LC: mean {summary['rear_vehicle_count_mean']:.2f}")
        if 'ev_min_distance' in ev_df.columns:
            print(f"EV min distance: mean {summary['ev_min_distance_mean']:.2f} m")
        if 'ev_peak_risk' in ev_df.columns:
            print(f"EV peak risk: mean {summary['ev_peak_risk_mean']:.2f}")
        if 'ev_alert_count' in ev_df.columns:
            print(f"EV alert count: mean {summary['ev_alert_count_mean']:.2f}")

        return ev_df, summary
    
    def descriptive_statistics(self):
        """Generate comprehensive descriptive statistics"""
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS BY CONDITION")
        print("="*60)
        
        key_metrics = ['minimum_distance', 'safety_margin', 'safety_score', 
                      'first_alert_distance', 'total_alerts', 'peak_risk_reached']
        
        for metric in key_metrics:
            if metric in self.df.columns and self.df[metric].notna().any():
                print(f"\n--- {metric.replace('_', ' ').title()} ---")
                stats_table = self.df.groupby('condition_clean')[metric].agg([
                    'count', 'mean', 'std', 'min', 'max', 'median'
                ]).round(2)
                print(stats_table)
        
        # Safety outcome distribution
        print(f"\n--- Safety Outcome Distribution ---")
        outcome_table = pd.crosstab(self.df['condition_clean'], self.df['safety_outcome'], 
                                   margins=True, normalize='index') * 100
        print(outcome_table.round(1))
        
        # Alert timing distribution
        print(f"\n--- Alert Timing Distribution ---")
        timing_table = pd.crosstab(self.df['condition_clean'], self.df['alert_timing'], 
                                  margins=True, normalize='index') * 100
        print(timing_table.round(1))
    
    def statistical_tests(self):
        """Perform statistical hypothesis testing"""
        print("\n" + "="*60)
        print("STATISTICAL HYPOTHESIS TESTING")
        print("="*60)
        
        # Key metrics for testing
        test_metrics = ['minimum_distance', 'safety_margin', 'safety_score', 'total_alerts']
        
        conditions = self.df['condition_clean'].unique()
        
        for metric in test_metrics:
            if metric in self.df.columns and self.df[metric].notna().any():
                print(f"\n--- {metric.replace('_', ' ').title()} Analysis ---")
                
                # Group data by condition
                groups = [self.df[self.df['condition_clean'] == cond][metric].dropna() 
                         for cond in conditions]
                
                # Filter out empty groups
                groups = [g for g in groups if len(g) > 0]
                condition_names = [cond for i, cond in enumerate(conditions) 
                                 if len(self.df[self.df['condition_clean'] == cond][metric].dropna()) > 0]
                
                if len(groups) >= 2:
                    # ANOVA test (if more than 2 groups)
                    if len(groups) > 2:
                        f_stat, p_value = stats.f_oneway(*groups)
                        print(f"ANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.3f}")
                        
                        if p_value < 0.05:
                            print("✅ Significant difference between conditions (p < 0.05)")
                        else:
                            print("❌ No significant difference between conditions (p ≥ 0.05)")
                    
                    # Pairwise t-tests
                    print("\nPairwise Comparisons:")
                    for i in range(len(groups)):
                        for j in range(i+1, len(groups)):
                            t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
                            effect_size = self._cohens_d(groups[i], groups[j])
                            
                            print(f"{condition_names[i]} vs {condition_names[j]}:")
                            print(f"  t-statistic: {t_stat:.3f}, p-value: {p_val:.3f}")
                            print(f"  Effect size (Cohen's d): {effect_size:.3f}")
                            
                            if p_val < 0.05:
                                significance = "significant"
                                if abs(effect_size) > 0.8:
                                    effect_desc = "large effect"
                                elif abs(effect_size) > 0.5:
                                    effect_desc = "medium effect"
                                else:
                                    effect_desc = "small effect"
                                print(f"  ✅ {significance} difference with {effect_desc}")
                            else:
                                print(f"  ❌ No significant difference")
                            print()
    
    def _cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        
        return (group1.mean() - group2.mean()) / pooled_std
    
    def create_visualizations(self, save_plots: bool = True):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig_counter = 1
        
        # 1. Safety Margin Comparison
        plt.figure(figsize=(12, 8))
        
        # Box plot
        plt.subplot(2, 2, 1)
        self.df.boxplot(column='safety_margin', by='condition_clean', ax=plt.gca())
        plt.title('Safety Margin by Condition')
        plt.ylabel('Safety Margin (m)')
        plt.xlabel('Condition')
        plt.xticks(rotation=45)
        
        # Violin plot
        plt.subplot(2, 2, 2)
        sns.violinplot(data=self.df, x='condition_clean', y='safety_margin')
        plt.title('Safety Margin Distribution')
        plt.xticks(rotation=45)
        
        # Safety Score comparison
        plt.subplot(2, 2, 3)
        self.df.boxplot(column='safety_score', by='condition_clean', ax=plt.gca())
        plt.title('Safety Score by Condition')
        plt.ylabel('Safety Score')
        plt.xlabel('Condition')
        plt.xticks(rotation=45)
        
        # Total Alerts comparison
        plt.subplot(2, 2, 4)
        sns.barplot(data=self.df, x='condition_clean', y='total_alerts', ci=95)
        plt.title('Average Total Alerts by Condition')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        if save_plots:
            plot_file = os.path.join(self.log_directory, f'experiment_analysis_plots_{fig_counter}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Saved plot {fig_counter}: {os.path.basename(plot_file)} to {self.log_directory}")
        plt.show()
        fig_counter += 1
        
        # 2. Safety Outcome Analysis
        plt.figure(figsize=(12, 6))
        
        # Safety outcome distribution
        plt.subplot(1, 2, 1)
        outcome_counts = pd.crosstab(self.df['condition_clean'], self.df['safety_outcome'])
        outcome_counts.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Safety Outcome Distribution by Condition')
        plt.ylabel('Number of Trials')
        plt.xlabel('Condition')
        plt.xticks(rotation=45)
        plt.legend(title='Safety Outcome')
        
        # Alert timing distribution
        plt.subplot(1, 2, 2)
        timing_counts = pd.crosstab(self.df['condition_clean'], self.df['alert_timing'])
        timing_counts.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Alert Timing Distribution by Condition')
        plt.ylabel('Number of Trials')
        plt.xlabel('Condition')
        plt.xticks(rotation=45)
        plt.legend(title='Alert Timing')
        
        plt.tight_layout()
        if save_plots:
            plot_file = os.path.join(self.log_directory, f'experiment_analysis_plots_{fig_counter}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Saved plot {fig_counter}: {os.path.basename(plot_file)} to {self.log_directory}")
        plt.show()
        fig_counter += 1
        
        # 3. Risk Analysis
        plt.figure(figsize=(12, 6))
        
        # Peak risk comparison
        plt.subplot(1, 2, 1)
        sns.boxplot(data=self.df, x='condition_clean', y='peak_risk_reached')
        plt.title('Peak Risk Reached by Condition')
        plt.xticks(rotation=45)
        
        # Emergency events comparison
        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.df, x='condition_clean', y='emergency_events_count')
        plt.title('Emergency Events Count by Condition')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        if save_plots:
            plot_file = os.path.join(self.log_directory, f'experiment_analysis_plots_{fig_counter}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Saved plot {fig_counter}: {os.path.basename(plot_file)} to {self.log_directory}")
        plt.show()
        
        print("✅ All visualizations generated")
    
    def generate_summary_insights(self):
        """Generate key insights and recommendations"""
        print("\n" + "="*60)
        print("KEY INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Best performing condition
        safety_means = self.df.groupby('condition_clean')['safety_margin'].mean()
        best_condition = safety_means.idxmax()
        best_margin = safety_means.max()
        
        collision_rates = self.df.groupby('condition_clean')['safety_outcome'].apply(
            lambda x: (x == 'collision').mean() * 100
        )
        safest_condition = collision_rates.idxmin()
        lowest_collision_rate = collision_rates.min()
        
        print(f"🏆 BEST SAFETY MARGIN: {best_condition} ({best_margin:.2f}m)")
        print(f"🛡️ LOWEST COLLISION RATE: {safest_condition} ({lowest_collision_rate:.1f}%)")
        
        # Alert effectiveness
        alert_means = self.df.groupby('condition_clean')['total_alerts'].mean()
        most_alerts = alert_means.idxmax()
        fewest_alerts = alert_means.idxmin()
        
        print(f"🚨 MOST ALERTS: {most_alerts} ({alert_means.max():.1f} avg)")
        print(f"🔇 FEWEST ALERTS: {fewest_alerts} ({alert_means.min():.1f} avg)")
        
        # Statistical significance summary
        print(f"\n📊 STATISTICAL SIGNIFICANCE:")
        print(f"   Use the statistical tests above to confirm significance")
        print(f"   Look for p-values < 0.05 and effect sizes > 0.5")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Focus on {best_condition} for optimal safety margins")
        print(f"   2. {safest_condition} shows best collision prevention")
        print(f"   3. Balance alert frequency vs. effectiveness")
        print(f"   4. Consider driver state adaptation strategies")
    
    def export_summary_stats(self, filename: str = None):
        """Export summary statistics to CSV in the experiment logs directory"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_summary_stats_{timestamp}.csv"
        
        # Save in the experiment logs directory
        filepath = os.path.join(self.log_directory, filename)
        
        # Create summary statistics
        key_metrics = ['minimum_distance', 'safety_margin', 'safety_score', 'total_alerts']
        
        summary_stats = []
        for metric in key_metrics:
            if metric in self.df.columns:
                stats_df = self.df.groupby('condition_clean')[metric].agg([
                    'count', 'mean', 'std', 'min', 'max', 'median'
                ]).round(3)
                stats_df['metric'] = metric
                stats_df = stats_df.reset_index()
                summary_stats.append(stats_df)
        
        if summary_stats:
            final_stats = pd.concat(summary_stats, ignore_index=True)
            final_stats.to_csv(filepath, index=False)
            print(f"✅ Summary statistics exported to: {os.path.basename(filepath)}")
            print(f"   Saved in: {self.log_directory}")
            return filepath
        else:
            print("❌ No valid metrics found for export")
            return None

def main():
    """Main analysis execution"""
    print("📈 Advanced CSV Data Analysis Tool")
    print("="*50)
    
    try:
        # Initialize analyzer
        analyzer = ExperimentCSVAnalyzer()
        
        # Run all analyses
        analyzer.descriptive_statistics()
        analyzer.statistical_tests()
        analyzer.create_visualizations()
        analyzer.generate_summary_insights()
        analyzer.export_summary_stats()
        
        print("\n🎯 Complete analysis finished!")
        print("Check the generated plots and summary files for detailed insights.")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 