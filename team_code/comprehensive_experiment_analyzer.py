#!/usr/bin/env python3
"""
Comprehensive Experiment Analysis Tool
Analyzes experiment data across conditions, scenarios, and participants
Combines JSON log processing with advanced statistical analysis and visualization
"""

import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')

class ComprehensiveExperimentAnalyzer:
    def __init__(self, log_directory: str = "/home/ascc304/carla_garage/experiment_logs/All"):
        self.log_directory = log_directory
        self.raw_data = []
        self.df = None
        self.summary_stats = {}
        
        print(f"🔬 Comprehensive Experiment Analyzer")
        print(f"📁 Target directory: {log_directory}")
        
    def load_all_experiment_data(self) -> int:
        """Load all JSON experiment files from the directory"""
        json_files = glob.glob(os.path.join(self.log_directory, "**/*.json"), recursive=True)
        
        print(f"🔍 Found {len(json_files)} JSON files")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    log_data = json.load(f)
                    
                # Extract metrics from each log
                metrics = self._extract_comprehensive_metrics(log_data, file_path)
                if metrics:
                    self.raw_data.append(metrics)
                    
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                
        print(f"✅ Successfully loaded {len(self.raw_data)} experiment trials")
        
        # Convert to DataFrame for analysis
        if self.raw_data:
            self.df = pd.DataFrame(self.raw_data)
            self._prepare_dataframe()
            
        return len(self.raw_data)
    
    def _extract_comprehensive_metrics(self, log_data: Dict, file_path: str) -> Dict:
        """Extract comprehensive metrics from a single log file"""
        try:
            # Parse filename for metadata
            filename = os.path.basename(file_path)
            path_parts = file_path.split(os.sep)
            
            # Try to extract participant from path (e.g., .../P001/... or .../P002/...)
            participant = None
            for part in path_parts:
                if part.startswith('P') and len(part) == 4 and part[1:].isdigit():
                    participant = part
                    break
            
            # Parse filename: ttc_SC-EV_fusion_P001_T1_timestamp.json or similar
            parts = filename.replace('.json', '').split('_')
            
            if len(parts) >= 5:
                # Extract scenario from parts (SC-EV, SC-PED, etc.)
                scenario = None
                condition = None
                trial = None
                session = None
                
                for i, part in enumerate(parts):
                    if part.startswith('SC-'):
                        scenario = part
                        # Condition is everything between scenario and participant
                        condition_parts = []
                        for j in range(i+1, len(parts)):
                            if parts[j].startswith('P') and len(parts[j]) == 4:
                                participant = parts[j]
                                if j+1 < len(parts):
                                    trial = parts[j+1]
                                if j+2 < len(parts):
                                    session = parts[j+2]
                                break
                            else:
                                condition_parts.append(parts[j])
                        condition = '_'.join(condition_parts) if condition_parts else 'unknown'
                        break
                
                # Fallback parsing if above doesn't work
                if not scenario:
                    scenario = parts[1] if len(parts) > 1 else 'unknown'
                if not condition:
                    condition = parts[2] if len(parts) > 2 else 'unknown'
                if not participant:
                    participant = parts[-3] if len(parts) >= 3 else 'unknown'
                if not trial:
                    trial = parts[-2] if len(parts) >= 2 else 'unknown'
                if not session:
                    session = parts[-1] if len(parts) >= 1 else 'unknown'
            else:
                return None
            
            # Extract data from log structure
            metadata = log_data.get('metadata', {})
            experiment_metrics = log_data.get('experiment_metrics', {})
            enhanced_safety = log_data.get('enhanced_safety_analysis', {})
            alert_effectiveness = log_data.get('alert_effectiveness_analysis', {})
            
            # Safety outcomes
            safety_outcome = enhanced_safety.get('safety_outcome', {})
            minimum_distance = safety_outcome.get('minimum_distance')
            safety_margin = safety_outcome.get('safety_margin')
            safety_score = safety_outcome.get('safety_score', 0.0)
            
            # Alert metrics
            timing_analysis = alert_effectiveness.get('timing_classification', {})
            effectiveness_analysis = alert_effectiveness.get('effectiveness_measurement', {})
            precision_analysis = alert_effectiveness.get('precision_analysis', {})
            
            # Emergency events
            emergency_events = enhanced_safety.get('emergency_events', {})
            peak_risk = emergency_events.get('peak_risk_reached', 0.0)
            emergency_count = emergency_events.get('emergency_events_count', 0)
            
            # Driver behavior metrics
            driver_behavior = enhanced_safety.get('driver_behavior_analysis', {})
            reaction_metrics = driver_behavior.get('reaction_metrics', {})
            
            return {
                # Identifiers
                'scenario': scenario,
                'condition': condition,
                'participant': participant,
                'trial': trial,
                'session': session,
                'filename': filename,
                'file_path': file_path,
                
                # Experiment metadata
                'duration': metadata.get('duration_seconds', 0),
                'total_frames': metadata.get('total_frames', 0),
                
                # Primary safety outcomes
                'safety_outcome': experiment_metrics.get('safety_outcome', 'unknown'),
                'minimum_distance': minimum_distance,
                'safety_margin': safety_margin,
                'safety_score': safety_score,
                
                # Alert effectiveness metrics
                'alert_timing': timing_analysis.get('timing_classification', 'unknown'),
                'timing_quality': timing_analysis.get('timing_quality', 'unknown'),
                'first_alert_distance': timing_analysis.get('first_alert_distance'),
                'alert_effectiveness': effectiveness_analysis.get('effectiveness_classification', 'unknown'),
                'alert_precision': precision_analysis.get('precision_classification', 'unknown'),
                'false_positive_risk': precision_analysis.get('false_positive_risk', 0.0),
                
                # Risk and emergency metrics
                'peak_risk_reached': peak_risk,
                'emergency_events_count': emergency_count,
                'total_alerts': experiment_metrics.get('total_alerts', 0),
                
                # Driver behavior metrics
                'reaction_time': reaction_metrics.get('average_reaction_time'),
                'max_deceleration': reaction_metrics.get('max_deceleration'),
                'driver_state_at_alert': timing_analysis.get('driver_state_at_alert', 'unknown'),
                
                # Performance metrics
                'completion_rate': metadata.get('completion_rate', 100.0),
                'scenario_success': metadata.get('scenario_success', True)
            }
            
        except Exception as e:
            print(f"⚠️ Error extracting metrics from {file_path}: {e}")
            return None
    
    def _prepare_dataframe(self):
        """Prepare DataFrame for analysis"""
        print("📊 Preparing data for analysis...")
        
        # Clean scenario names
        self.df['scenario_clean'] = self.df['scenario'].str.replace('SC-', '').str.upper()
        
        # Clean condition names
        self.df['condition_clean'] = self.df['condition'].str.replace('_', ' ').str.title()
        
        # Convert numeric columns
        numeric_columns = ['minimum_distance', 'safety_margin', 'safety_score', 'first_alert_distance',
                          'peak_risk_reached', 'emergency_events_count', 'total_alerts', 'false_positive_risk',
                          'reaction_time', 'max_deceleration', 'duration', 'completion_rate']
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Create binary safety outcome
        self.df['collision'] = (self.df['safety_outcome'] == 'collision').astype(int)
        
        print(f"✅ Data prepared:")
        print(f"   Total trials: {len(self.df)}")
        print(f"   Scenarios: {sorted(self.df['scenario_clean'].unique())}")
        print(f"   Conditions: {sorted(self.df['condition_clean'].unique())}")
        print(f"   Participants: {sorted(self.df['participant'].unique())}")
    
    def cross_condition_analysis(self):
        """Analyze differences across experimental conditions"""
        print("\n" + "="*70)
        print("CROSS-CONDITION ANALYSIS")
        print("="*70)
        
        key_metrics = ['safety_margin', 'safety_score', 'total_alerts', 'minimum_distance']
        
        for metric in key_metrics:
            if metric in self.df.columns and self.df[metric].notna().any():
                print(f"\n--- {metric.replace('_', ' ').title()} by Condition ---")
                
                # Descriptive statistics
                stats_table = self.df.groupby('condition_clean')[metric].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(3)
                print(stats_table)
                
                # Statistical testing
                conditions = self.df['condition_clean'].unique()
                if len(conditions) > 1:
                    groups = [self.df[self.df['condition_clean'] == cond][metric].dropna() 
                             for cond in conditions]
                    groups = [g for g in groups if len(g) > 0]
                    
                    if len(groups) >= 2:
                        if len(groups) > 2:
                            f_stat, p_value = stats.f_oneway(*groups)
                            print(f"ANOVA: F={f_stat:.3f}, p={p_value:.3f}")
                        else:
                            t_stat, p_value = stats.ttest_ind(groups[0], groups[1])
                            print(f"T-test: t={t_stat:.3f}, p={p_value:.3f}")
        
        # Collision rate analysis
        print(f"\n--- Collision Rates by Condition ---")
        collision_rates = self.df.groupby('condition_clean')['collision'].agg(['count', 'sum', 'mean']).round(3)
        collision_rates['collision_rate_percent'] = collision_rates['mean'] * 100
        print(collision_rates)
    
    def cross_scenario_analysis(self):
        """Analyze differences between EV and PED scenarios"""
        print("\n" + "="*70)
        print("CROSS-SCENARIO ANALYSIS (EV vs PED)")
        print("="*70)
        
        scenarios = self.df['scenario_clean'].unique()
        print(f"Scenarios found: {scenarios}")
        
        key_metrics = ['safety_margin', 'safety_score', 'total_alerts', 'minimum_distance']
        
        for metric in key_metrics:
            if metric in self.df.columns and self.df[metric].notna().any():
                print(f"\n--- {metric.replace('_', ' ').title()} by Scenario ---")
                
                # Descriptive statistics
                stats_table = self.df.groupby('scenario_clean')[metric].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(3)
                print(stats_table)
                
                # Statistical testing between scenarios
                if len(scenarios) == 2:
                    group1 = self.df[self.df['scenario_clean'] == scenarios[0]][metric].dropna()
                    group2 = self.df[self.df['scenario_clean'] == scenarios[1]][metric].dropna()
                    
                    if len(group1) > 0 and len(group2) > 0:
                        t_stat, p_value = stats.ttest_ind(group1, group2)
                        effect_size = self._cohens_d(group1, group2)
                        print(f"T-test: t={t_stat:.3f}, p={p_value:.3f}, Cohen's d={effect_size:.3f}")
                        
                        if p_value < 0.05:
                            print(f"✅ Significant difference between {scenarios[0]} and {scenarios[1]}")
                        else:
                            print(f"❌ No significant difference between scenarios")
        
        # Scenario-specific collision rates
        print(f"\n--- Collision Rates by Scenario ---")
        scenario_collision = self.df.groupby('scenario_clean')['collision'].agg(['count', 'sum', 'mean']).round(3)
        scenario_collision['collision_rate_percent'] = scenario_collision['mean'] * 100
        print(scenario_collision)
    
    def cross_participant_analysis(self):
        """Analyze differences across participants"""
        print("\n" + "="*70)
        print("CROSS-PARTICIPANT ANALYSIS")
        print("="*70)
        
        participants = sorted(self.df['participant'].unique())
        print(f"Participants: {participants}")
        
        key_metrics = ['safety_margin', 'safety_score', 'total_alerts', 'reaction_time']
        
        for metric in key_metrics:
            if metric in self.df.columns and self.df[metric].notna().any():
                print(f"\n--- {metric.replace('_', ' ').title()} by Participant ---")
                
                # Descriptive statistics
                stats_table = self.df.groupby('participant')[metric].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(3)
                print(stats_table)
                
                # ANOVA across participants
                groups = [self.df[self.df['participant'] == p][metric].dropna() for p in participants]
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) >= 2:
                    if len(groups) > 2:
                        f_stat, p_value = stats.f_oneway(*groups)
                        print(f"ANOVA: F={f_stat:.3f}, p={p_value:.3f}")
                    else:
                        t_stat, p_value = stats.ttest_ind(groups[0], groups[1])
                        print(f"T-test: t={t_stat:.3f}, p={p_value:.3f}")
        
        # Participant performance summary
        print(f"\n--- Participant Performance Summary ---")
        participant_summary = self.df.groupby('participant').agg({
            'collision': ['count', 'sum', 'mean'],
            'safety_margin': 'mean',
            'safety_score': 'mean',
            'total_alerts': 'mean'
        }).round(3)
        print(participant_summary)
    
    def multi_factor_analysis(self):
        """Analyze interactions between condition, scenario, and participant"""
        print("\n" + "="*70)
        print("MULTI-FACTOR INTERACTION ANALYSIS")
        print("="*70)
        
        # Three-way breakdown
        if len(self.df) > 0:
            print(f"\n--- Trial Distribution ---")
            distribution = self.df.groupby(['scenario_clean', 'condition_clean', 'participant']).size().unstack(fill_value=0)
            print(distribution)
            
            # Safety outcomes by factors
            print(f"\n--- Safety Outcomes by All Factors ---")
            safety_breakdown = pd.crosstab([self.df['scenario_clean'], self.df['condition_clean']], 
                                         [self.df['participant'], self.df['safety_outcome']], 
                                         margins=True)
            print(safety_breakdown)
            
            # Mean safety metrics by factors
            print(f"\n--- Mean Safety Margin by Factors ---")
            safety_means = self.df.groupby(['scenario_clean', 'condition_clean', 'participant'])['safety_margin'].mean().unstack(fill_value=np.nan)
            print(safety_means.round(2))
    
    def _cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        
        return (group1.mean() - group2.mean()) / pooled_std
    
    def create_comprehensive_visualizations(self, save_plots: bool = True):
        """Create comprehensive visualizations for all analyses"""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*70)
        
        plt.style.use('default')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Multi-dimensional overview
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Experiment Analysis Overview', fontsize=16)
        
        # Safety margin by condition
        self.df.boxplot(column='safety_margin', by='condition_clean', ax=axes[0,0])
        axes[0,0].set_title('Safety Margin by Condition')
        axes[0,0].set_xlabel('Condition')
        axes[0,0].set_ylabel('Safety Margin (m)')
        
        # Safety margin by scenario
        self.df.boxplot(column='safety_margin', by='scenario_clean', ax=axes[0,1])
        axes[0,1].set_title('Safety Margin by Scenario')
        axes[0,1].set_xlabel('Scenario')
        
        # Safety margin by participant
        self.df.boxplot(column='safety_margin', by='participant', ax=axes[0,2])
        axes[0,2].set_title('Safety Margin by Participant')
        axes[0,2].set_xlabel('Participant')
        
        # Total alerts by condition
        if 'total_alerts' in self.df.columns:
            sns.barplot(data=self.df, x='condition_clean', y='total_alerts', ax=axes[1,0])
            axes[1,0].set_title('Average Total Alerts by Condition')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Collision rates by scenario
        collision_by_scenario = self.df.groupby('scenario_clean')['collision'].mean() * 100
        collision_by_scenario.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Collision Rate by Scenario (%)')
        axes[1,1].set_ylabel('Collision Rate (%)')
        
        # Safety score distribution
        if 'safety_score' in self.df.columns:
            self.df['safety_score'].hist(bins=20, ax=axes[1,2])
            axes[1,2].set_title('Safety Score Distribution')
            axes[1,2].set_xlabel('Safety Score')
            axes[1,2].set_ylabel('Frequency')
        
        plt.tight_layout()
        if save_plots:
            plot_file = os.path.join(self.log_directory, f'comprehensive_analysis_overview_{timestamp}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Saved overview plot: {os.path.basename(plot_file)}")
        plt.show()
        
        # 2. Scenario comparison detailed plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('EV vs PED Scenario Detailed Comparison', fontsize=16)
        
        # Safety metrics comparison
        if len(self.df['scenario_clean'].unique()) >= 2:
            sns.violinplot(data=self.df, x='scenario_clean', y='safety_margin', ax=axes[0,0])
            axes[0,0].set_title('Safety Margin Distribution')
            
            sns.barplot(data=self.df, x='scenario_clean', y='total_alerts', ax=axes[0,1])
            axes[0,1].set_title('Average Total Alerts')
            
            # Collision comparison
            collision_data = self.df.groupby('scenario_clean')['collision'].agg(['count', 'sum']).reset_index()
            collision_data['collision_rate'] = collision_data['sum'] / collision_data['count'] * 100
            sns.barplot(data=collision_data, x='scenario_clean', y='collision_rate', ax=axes[1,0])
            axes[1,0].set_title('Collision Rate by Scenario (%)')
            
            # Heatmap of conditions vs scenarios
            heatmap_data = pd.crosstab(self.df['condition_clean'], self.df['scenario_clean'], 
                                     values=self.df['safety_margin'], aggfunc='mean')
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', ax=axes[1,1])
            axes[1,1].set_title('Mean Safety Margin: Condition vs Scenario')
        
        plt.tight_layout()
        if save_plots:
            plot_file = os.path.join(self.log_directory, f'scenario_comparison_{timestamp}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Saved scenario comparison: {os.path.basename(plot_file)}")
        plt.show()
        
        # 3. Participant performance matrix
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Participant Performance Analysis', fontsize=16)
        
        # Participant performance metrics
        participant_metrics = self.df.groupby('participant').agg({
            'safety_margin': 'mean',
            'safety_score': 'mean',
            'total_alerts': 'mean',
            'collision': 'mean'
        }).round(3)
        
        participant_metrics[['safety_margin', 'safety_score']].plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Safety Metrics by Participant')
        axes[0,0].legend(['Safety Margin', 'Safety Score'])
        
        participant_metrics['total_alerts'].plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Average Total Alerts by Participant')
        
        participant_metrics['collision'].plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Collision Rate by Participant')
        axes[1,0].set_ylabel('Collision Rate')
        
        # Participant consistency (standard deviation)
        participant_std = self.df.groupby('participant')['safety_margin'].std()
        participant_std.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Safety Margin Consistency (Lower = More Consistent)')
        axes[1,1].set_ylabel('Standard Deviation')
        
        plt.tight_layout()
        if save_plots:
            plot_file = os.path.join(self.log_directory, f'participant_analysis_{timestamp}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Saved participant analysis: {os.path.basename(plot_file)}")
        plt.show()
        
        print("✅ All visualizations generated")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
{'='*80}
COMPREHENSIVE EXPERIMENT ANALYSIS REPORT
Generated: {timestamp}
Data Directory: {self.log_directory}
{'='*80}

=== DATASET OVERVIEW ===
Total Trials: {len(self.df)}
Scenarios: {', '.join(sorted(self.df['scenario_clean'].unique()))}
Conditions: {', '.join(sorted(self.df['condition_clean'].unique()))}
Participants: {', '.join(sorted(self.df['participant'].unique()))}

=== OVERALL PERFORMANCE SUMMARY ===
"""
        
        # Overall statistics
        overall_stats = {
            'Total Collisions': self.df['collision'].sum(),
            'Overall Collision Rate': f"{self.df['collision'].mean() * 100:.1f}%",
            'Average Safety Margin': f"{self.df['safety_margin'].mean():.2f}m",
            'Average Safety Score': f"{self.df['safety_score'].mean():.2f}",
            'Average Total Alerts': f"{self.df['total_alerts'].mean():.1f}"
        }
        
        for metric, value in overall_stats.items():
            report += f"{metric}: {value}\n"
        
        # Best performing combinations
        report += f"\n=== TOP PERFORMING COMBINATIONS ===\n"
        
        # Best condition
        condition_performance = self.df.groupby('condition_clean').agg({
            'safety_margin': 'mean',
            'collision': 'mean'
        }).round(3)
        best_condition = condition_performance['safety_margin'].idxmax()
        safest_condition = condition_performance['collision'].idxmin()
        
        report += f"Best Safety Margin: {best_condition} ({condition_performance.loc[best_condition, 'safety_margin']:.2f}m)\n"
        report += f"Lowest Collision Rate: {safest_condition} ({condition_performance.loc[safest_condition, 'collision']*100:.1f}%)\n"
        
        # Scenario comparison
        scenario_performance = self.df.groupby('scenario_clean').agg({
            'safety_margin': 'mean',
            'collision': 'mean'
        }).round(3)
        
        report += f"\n=== SCENARIO COMPARISON ===\n"
        for scenario in scenario_performance.index:
            safety_margin = scenario_performance.loc[scenario, 'safety_margin']
            collision_rate = scenario_performance.loc[scenario, 'collision'] * 100
            report += f"{scenario}: Safety Margin = {safety_margin:.2f}m, Collision Rate = {collision_rate:.1f}%\n"
        
        # Participant ranking
        participant_performance = self.df.groupby('participant').agg({
            'safety_margin': 'mean',
            'collision': 'mean'
        }).round(3)
        participant_ranking = participant_performance.sort_values('safety_margin', ascending=False)
        
        report += f"\n=== PARTICIPANT RANKING (by Safety Margin) ===\n"
        for i, (participant, row) in enumerate(participant_ranking.iterrows()):
            report += f"{i+1}. {participant}: {row['safety_margin']:.2f}m (Collision Rate: {row['collision']*100:.1f}%)\n"
        
        report += f"\n{'='*80}\n"
        
        return report
    
    def save_comprehensive_results(self):
        """Save all analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive report
        report = self.generate_comprehensive_report()
        report_file = os.path.join(self.log_directory, f"comprehensive_analysis_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save processed data
        if self.df is not None:
            csv_file = os.path.join(self.log_directory, f"comprehensive_analysis_data_{timestamp}.csv")
            self.df.to_csv(csv_file, index=False)
            
            # Save summary statistics
            summary_file = os.path.join(self.log_directory, f"comprehensive_summary_stats_{timestamp}.csv")
            
            summary_stats = []
            for grouping in ['condition_clean', 'scenario_clean', 'participant']:
                if grouping in self.df.columns:
                    group_stats = self.df.groupby(grouping).agg({
                        'safety_margin': ['count', 'mean', 'std'],
                        'safety_score': 'mean',
                        'total_alerts': 'mean',
                        'collision': ['sum', 'mean']
                    }).round(3)
                    group_stats.columns = ['_'.join(col).strip() for col in group_stats.columns]
                    group_stats['grouping_type'] = grouping
                    group_stats = group_stats.reset_index()
                    summary_stats.append(group_stats)
            
            if summary_stats:
                pd.concat(summary_stats, ignore_index=True).to_csv(summary_file, index=False)
        
        print(f"✅ Comprehensive analysis saved:")
        print(f"   Report: {os.path.basename(report_file)}")
        print(f"   Data: {os.path.basename(csv_file)}")
        print(f"   Summary: {os.path.basename(summary_file)}")
        
        return report_file, csv_file, summary_file

def main():
    """Main execution function"""
    print("🔬 Comprehensive Experiment Analysis Tool")
    print("="*60)
    
    try:
        # Initialize analyzer
        analyzer = ComprehensiveExperimentAnalyzer()
        
        # Load all data
        count = analyzer.load_all_experiment_data()
        
        if count == 0:
            print("❌ No experiment data found!")
            return
        
        # Perform all analyses
        analyzer.cross_condition_analysis()
        analyzer.cross_scenario_analysis()
        analyzer.cross_participant_analysis()
        analyzer.multi_factor_analysis()
        
        # Generate visualizations
        analyzer.create_comprehensive_visualizations()
        
        # Generate and display report
        report = analyzer.generate_comprehensive_report()
        print(report)
        
        # Save all results
        analyzer.save_comprehensive_results()
        
        print("🎯 Comprehensive analysis complete!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 