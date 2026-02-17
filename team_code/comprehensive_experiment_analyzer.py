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
    def __init__(self, log_directory: str = "/home/ascc304/carla_garage/experiment_logs/ALL"):
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
        
        # Debug: Show data availability for key metrics
        print(f"\n📊 Data availability:")
        key_metrics = ['safety_margin', 'minimum_distance', 'safety_score', 'collision']
        for metric in key_metrics:
            if metric in self.df.columns:
                non_null_count = self.df[metric].notna().sum()
                print(f"   {metric}: {non_null_count}/{len(self.df)} records ({non_null_count/len(self.df)*100:.1f}%)")
                if non_null_count > 0:
                    print(f"      Range: {self.df[metric].min():.2f} - {self.df[metric].max():.2f}")
        
        # Show sample of data
        print(f"\n📋 Sample data (first 3 rows):")
        display_cols = ['condition_clean', 'scenario_clean', 'participant', 'safety_margin', 'minimum_distance', 'collision']
        available_cols = [col for col in display_cols if col in self.df.columns]
        print(self.df[available_cols].head(3).to_string())
    
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
        """Create IEEE 2-column format visualizations focusing on key insights"""
        print("\n" + "="*70)
        print("GENERATING IEEE FORMAT VISUALIZATIONS")
        print("="*70)
        
        # Set IEEE paper style
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'serif',
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 12
        })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []
        
        # Figure 1: Cross-Condition Performance Comparison (IEEE 2-column width)
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))
        
        # Check available columns and use alternative metrics if safety_margin is missing
        print(f"📊 Available columns: {list(self.df.columns)}")
        print(f"📊 Safety margin data availability: {self.df['safety_margin'].notna().sum()} / {len(self.df)} records")
        
        # Safety Margin by Condition (or use minimum_distance as fallback)
        condition_order = ['Internal Only', 'Fusion', 'External Only']
        available_conditions = [cond for cond in condition_order if cond in self.df['condition_clean'].unique()]
        
        # Choose the best available safety metric
        safety_metric = 'safety_margin'
        safety_label = 'Safety Margin (m)'
        
        if self.df['safety_margin'].notna().sum() == 0 or 'safety_margin' not in self.df.columns:
            if 'minimum_distance' in self.df.columns and self.df['minimum_distance'].notna().sum() > 0:
                safety_metric = 'minimum_distance'
                safety_label = 'Minimum Distance (m)'
                print(f"📊 Using minimum_distance as fallback for safety metric")
            elif 'safety_score' in self.df.columns and self.df['safety_score'].notna().sum() > 0:
                safety_metric = 'safety_score'
                safety_label = 'Safety Score'
                print(f"📊 Using safety_score as fallback for safety metric")
        
        safety_data = []
        condition_labels = []
        
        for cond in available_conditions:
            cond_data = self.df[self.df['condition_clean'] == cond][safety_metric].dropna()
            if len(cond_data) > 0:
                safety_data.append(cond_data.values)
                condition_labels.append(cond)
        
        if len(safety_data) > 0:
            colors = ['#2E8B57', '#4682B4', '#CD5C5C'][:len(condition_labels)]
            bp1 = ax1.boxplot(safety_data, labels=[c.replace(' ', '\n') for c in condition_labels], 
                             patch_artist=True, widths=0.6)
            
            for patch, color in zip(bp1['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            
            ax1.set_title(f'(a) {safety_label.split("(")[0].strip()}\nby Condition', fontweight='bold')
            ax1.set_ylabel(safety_label)
            ax1.set_xlabel('Condition')
        else:
            ax1.text(0.5, 0.5, 'No safety data\navailable', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('(a) Safety Metrics by Condition', fontweight='bold')
            ax1.set_xlabel('Condition')
            ax1.set_ylabel('Safety Metric')
        
        # Collision Rate by Condition
        collision_rates = self.df.groupby('condition_clean')['collision'].mean() * 100
        collision_rates = collision_rates.reindex([c for c in condition_order if c in collision_rates.index])
        
        bars = ax2.bar(range(len(collision_rates)), collision_rates.values, 
                      color=colors[:len(collision_rates)], alpha=0.8, width=0.6)
        ax2.set_title('(b) Collision Rate by Condition', fontweight='bold')
        ax2.set_ylabel('Collision Rate (%)')
        ax2.set_xlabel('Condition')
        ax2.set_xticks(range(len(collision_rates)))
        ax2.set_xticklabels([c.replace(' ', '\n') for c in collision_rates.index], rotation=0)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=6)
        
        plt.tight_layout()
        fig1_file = os.path.join(self.log_directory, f'ieee_condition_comparison_{timestamp}.png')
        plt.savefig(fig1_file, dpi=300, bbox_inches='tight')
        saved_files.append(fig1_file)
        plt.close()
        
        # Figure 2: Scenario Analysis (IEEE 2-column width) 
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(7.5, 3.5))
        
        # Safety Margin by Scenario (using same metric as condition chart)
        scenarios = sorted(self.df['scenario_clean'].unique())
        if len(scenarios) >= 1:
            scenario_data = []
            scenario_labels = []
            
            for scen in scenarios:
                scen_data = self.df[self.df['scenario_clean'] == scen][safety_metric].dropna()
                if len(scen_data) > 0:
                    scenario_data.append(scen_data.values)
                    scenario_labels.append(scen)
            
            if len(scenario_data) > 0:
                scenario_colors = ['#FF6B6B', '#4ECDC4'][:len(scenario_labels)]
                
                bp2 = ax3.boxplot(scenario_data, labels=scenario_labels, patch_artist=True, widths=0.6)
                
                for patch, color in zip(bp2['boxes'], scenario_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
                
                ax3.set_title(f'(c) {safety_label.split("(")[0].strip()}\nby Scenario', fontweight='bold')
                ax3.set_ylabel(safety_label)
                ax3.set_xlabel('Scenario')
            else:
                ax3.text(0.5, 0.5, 'No safety data\navailable', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=12)
                ax3.set_title('(c) Safety Metrics by Scenario', fontweight='bold')
                ax3.set_xlabel('Scenario')
                ax3.set_ylabel('Safety Metric')
            
            # Collision Rate by Scenario
            scenario_collision = self.df.groupby('scenario_clean')['collision'].mean() * 100
            bars2 = ax4.bar(range(len(scenario_collision)), scenario_collision.values,
                           color=scenario_colors[:len(scenario_collision)], alpha=0.8, width=0.6)
            ax4.set_title('(d) Collision Rate by Scenario', fontweight='bold')
            ax4.set_ylabel('Collision Rate (%)')
            ax4.set_xlabel('Scenario')
            ax4.set_xticks(range(len(scenario_collision)))
            ax4.set_xticklabels(scenario_collision.index)
            
            # Add value labels
            for bar in bars2:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=6)
        
        plt.tight_layout()
        fig2_file = os.path.join(self.log_directory, f'ieee_scenario_comparison_{timestamp}.png')
        plt.savefig(fig2_file, dpi=300, bbox_inches='tight')
        saved_files.append(fig2_file)
        plt.close()
        
        print(f"📊 Saved IEEE format visualizations:")
        for file in saved_files:
            print(f"   - {os.path.basename(file)}")
        
        return saved_files
    
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
        
        # Critical interaction analysis
        collision_matrix = self.df.pivot_table(
            values='collision', 
            index='scenario_clean', 
            columns='condition_clean', 
            aggfunc='mean'
        ) * 100
        
        max_collision_idx = collision_matrix.stack().idxmax()
        max_collision_rate = collision_matrix.stack().max()
        
        report += f"\n=== CRITICAL SAFETY INSIGHT ===\n"
        report += f"🚨 HIGHEST RISK COMBINATION: {max_collision_idx[0]} + {max_collision_idx[1]} ({max_collision_rate:.1f}% collision rate)\n"
        
        # Analysis of why External Only + PED is problematic
        if max_collision_idx[0] == 'PED' and max_collision_idx[1] == 'External Only':
            ped_ext_data = self.df[(self.df['scenario_clean'] == 'PED') & 
                                   (self.df['condition_clean'] == 'External Only')]
            avg_alert_dist = ped_ext_data['first_alert_distance'].mean()
            
            report += f"\n📊 WHY EXTERNAL ONLY + PED IS PROBLEMATIC:\n"
            report += f"- Pedestrians often detected late by ego sensors (avg alert distance: {avg_alert_dist:.1f}m)\n"
            report += f"- When driver is inattentive, minimal reaction time available\n"
            report += f"- Demonstrates necessity of driver state monitoring for safety\n"
            report += f"- Sensor-only approach insufficient for sudden pedestrian appearances\n"
        
        # Fusion benefits
        report += f"\n=== DRIVER STATE INTEGRATION BENEFITS ===\n"
        for scenario in ['EV', 'PED']:
            scenario_data = self.df[self.df['scenario_clean'] == scenario]
            ext_only = scenario_data[scenario_data['condition_clean'] == 'External Only']['collision'].mean() * 100
            fusion = scenario_data[scenario_data['condition_clean'] == 'Fusion']['collision'].mean() * 100
            internal_only = scenario_data[scenario_data['condition_clean'] == 'Internal Only']['collision'].mean() * 100
            
            fusion_improvement = ext_only - fusion
            internal_improvement = ext_only - internal_only
            
            report += f"{scenario} Scenario Improvements:\n"
            report += f"  External Only → Fusion: {ext_only:.1f}% → {fusion:.1f}% ({fusion_improvement:+.1f}% reduction)\n"
            report += f"  External Only → Internal Only: {ext_only:.1f}% → {internal_only:.1f}% ({internal_improvement:+.1f}% reduction)\n"
        
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
        
        # Key research implications
        report += f"\n=== KEY RESEARCH IMPLICATIONS ===\n"
        report += f"1. Sensor-only safety systems have critical limitations, especially for pedestrian scenarios\n"
        report += f"2. Late detection of pedestrians creates dangerous situations when drivers are inattentive\n"
        report += f"3. Driver state integration (fusion approach) significantly improves safety outcomes\n"
        report += f"4. Internal risk assessment complements external sensor data for comprehensive safety\n"
        report += f"5. Multi-modal risk assessment is essential for robust autonomous safety systems\n"
        
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

    def scenario_condition_interaction_analysis(self):
        """Analyze critical interactions between scenarios and conditions"""
        print("\n" + "="*70)
        print("SCENARIO × CONDITION INTERACTION ANALYSIS")
        print("="*70)
        
        # Create interaction breakdown
        interaction_stats = self.df.groupby(['scenario_clean', 'condition_clean']).agg({
            'collision': ['count', 'sum', 'mean'],
            'safety_margin': 'mean',
            'total_alerts': 'mean',
            'first_alert_distance': 'mean'
        }).round(3)
        
        print(f"\n--- Critical Interaction: Collision Rates by Scenario × Condition ---")
        collision_matrix = self.df.pivot_table(
            values='collision', 
            index='scenario_clean', 
            columns='condition_clean', 
            aggfunc=['count', 'sum', 'mean'],
            fill_value=0
        )
        
        # Display collision rates as percentages
        collision_rates = collision_matrix['mean'] * 100
        print("Collision Rate (%) by Scenario × Condition:")
        print(collision_rates.round(1))
        
        # Identify the most dangerous combination
        max_collision_idx = collision_rates.stack().idxmax()
        max_collision_rate = collision_rates.stack().max()
        
        print(f"\n🚨 HIGHEST RISK COMBINATION: {max_collision_idx[0]} + {max_collision_idx[1]}")
        print(f"   Collision Rate: {max_collision_rate:.1f}%")
        
        # Analyze why External Only + PED is problematic
        if max_collision_idx[0] == 'PED' and max_collision_idx[1] == 'External Only':
            print(f"\n📊 ANALYSIS: External Only + PED High Risk Factors:")
            
            # Get External Only PED data
            ext_ped_data = self.df[(self.df['scenario_clean'] == 'PED') & 
                                   (self.df['condition_clean'] == 'External Only')]
            
            # Compare with other PED conditions
            ped_data = self.df[self.df['scenario_clean'] == 'PED']
            ped_comparison = ped_data.groupby('condition_clean').agg({
                'collision': 'mean',
                'safety_margin': 'mean',
                'first_alert_distance': 'mean',
                'total_alerts': 'mean'
            }).round(3)
            
            print("   PED Scenario by Condition:")
            for condition in ped_comparison.index:
                collision_rate = ped_comparison.loc[condition, 'collision'] * 100
                safety_margin = ped_comparison.loc[condition, 'safety_margin']
                alert_distance = ped_comparison.loc[condition, 'first_alert_distance']
                
                if condition == 'External Only':
                    print(f"   🔴 {condition}: {collision_rate:.1f}% collision, {safety_margin:.1f}m safety margin, {alert_distance:.1f}m alert distance")
                else:
                    print(f"   ✅ {condition}: {collision_rate:.1f}% collision, {safety_margin:.1f}m safety margin, {alert_distance:.1f}m alert distance")
        
        # Show the benefit of fusion
        print(f"\n💡 FUSION BENEFIT ANALYSIS:")
        fusion_benefit = {}
        
        for scenario in ['EV', 'PED']:
            scenario_data = self.df[self.df['scenario_clean'] == scenario]
            ext_only = scenario_data[scenario_data['condition_clean'] == 'External Only']['collision'].mean() * 100
            fusion = scenario_data[scenario_data['condition_clean'] == 'Fusion']['collision'].mean() * 100
            internal_only = scenario_data[scenario_data['condition_clean'] == 'Internal Only']['collision'].mean() * 100
            
            fusion_improvement = ext_only - fusion
            internal_improvement = ext_only - internal_only
            
            print(f"   {scenario} Scenario:")
            print(f"     External Only → Fusion: {ext_only:.1f}% → {fusion:.1f}% ({fusion_improvement:+.1f}%)")
            print(f"     External Only → Internal Only: {ext_only:.1f}% → {internal_only:.1f}% ({internal_improvement:+.1f}%)")
            
            fusion_benefit[scenario] = {
                'fusion_improvement': fusion_improvement,
                'internal_improvement': internal_improvement
            }
        
        return collision_rates, fusion_benefit
    
    def sensor_limitation_analysis(self):
        """Analyze sensor limitations and late detection scenarios"""
        print("\n" + "="*70)
        print("SENSOR LIMITATION & LATE DETECTION ANALYSIS")
        print("="*70)
        
        # Analyze alert distances by scenario and condition
        print(f"\n--- First Alert Distance Analysis ---")
        alert_analysis = self.df.groupby(['scenario_clean', 'condition_clean'])['first_alert_distance'].agg([
            'count', 'mean', 'std', 'min'
        ]).round(2)
        print(alert_analysis)
        
        # Identify late detection cases (short alert distances)
        late_detection_threshold = 10.0  # meters
        late_detections = self.df[self.df['first_alert_distance'] < late_detection_threshold]
        
        print(f"\n--- Late Detection Cases (Alert Distance < {late_detection_threshold}m) ---")
        late_detection_stats = late_detections.groupby(['scenario_clean', 'condition_clean']).agg({
            'collision': ['count', 'sum', 'mean'],
            'first_alert_distance': 'mean'
        }).round(3)
        print(late_detection_stats)
        
        # Calculate correlation between alert distance and collision
        correlation = self.df['first_alert_distance'].corr(1 - self.df['collision'])  # Negative collision for positive correlation
        print(f"\n📈 Correlation between Alert Distance and Safety: {correlation:.3f}")
        
        # Analyze PED External Only specifically
        ped_ext_data = self.df[(self.df['scenario_clean'] == 'PED') & 
                               (self.df['condition_clean'] == 'External Only')]
        
        if len(ped_ext_data) > 0:
            print(f"\n🚶 PED + External Only Detailed Analysis:")
            print(f"   Average Alert Distance: {ped_ext_data['first_alert_distance'].mean():.1f}m")
            print(f"   Late Detection Rate: {(ped_ext_data['first_alert_distance'] < late_detection_threshold).mean()*100:.1f}%")
            print(f"   Collision Rate: {ped_ext_data['collision'].mean()*100:.1f}%")
            
            # Show why this is problematic
            late_ped_collisions = ped_ext_data[ped_ext_data['first_alert_distance'] < late_detection_threshold]['collision'].mean() * 100
            early_ped_collisions = ped_ext_data[ped_ext_data['first_alert_distance'] >= late_detection_threshold]['collision'].mean() * 100
            
            print(f"   Late Detection Collision Rate: {late_ped_collisions:.1f}%")
            print(f"   Early Detection Collision Rate: {early_ped_collisions:.1f}%")
    
    def driver_attentiveness_necessity_analysis(self):
        """Analyze why driver state monitoring is necessary"""
        print("\n" + "="*70)
        print("DRIVER ATTENTIVENESS NECESSITY ANALYSIS")
        print("="*70)
        
        # Compare conditions that include driver state vs those that don't
        external_only_perf = self.df[self.df['condition_clean'] == 'External Only'].agg({
            'collision': 'mean',
            'safety_margin': 'mean',
            'safety_score': 'mean'
        })
        
        fusion_perf = self.df[self.df['condition_clean'] == 'Fusion'].agg({
            'collision': 'mean',
            'safety_margin': 'mean',
            'safety_score': 'mean'
        })
        
        internal_only_perf = self.df[self.df['condition_clean'] == 'Internal Only'].agg({
            'collision': 'mean',
            'safety_margin': 'mean',
            'safety_score': 'mean'
        })
        
        print(f"--- Driver State Integration Impact ---")
        print(f"External Only (No Driver State):")
        print(f"  Collision Rate: {external_only_perf['collision']*100:.1f}%")
        print(f"  Safety Margin: {external_only_perf['safety_margin']:.1f}m")
        print(f"  Safety Score: {external_only_perf['safety_score']:.2f}")
        
        print(f"\nFusion (Sensor + Driver State):")
        print(f"  Collision Rate: {fusion_perf['collision']*100:.1f}%")
        print(f"  Safety Margin: {fusion_perf['safety_margin']:.1f}m")
        print(f"  Safety Score: {fusion_perf['safety_score']:.2f}")
        
        print(f"\nInternal Only (Driver State Focus):")
        print(f"  Collision Rate: {internal_only_perf['collision']*100:.1f}%")
        print(f"  Safety Margin: {internal_only_perf['safety_margin']:.1f}m")
        print(f"  Safety Score: {internal_only_perf['safety_score']:.2f}")
        
        # Calculate improvements
        fusion_improvement = (external_only_perf['collision'] - fusion_perf['collision']) * 100
        internal_improvement = (external_only_perf['collision'] - internal_only_perf['collision']) * 100
        
        print(f"\n💡 DRIVER STATE BENEFITS:")
        print(f"  Fusion vs External Only: {fusion_improvement:.1f}% collision reduction")
        print(f"  Internal Only vs External Only: {internal_improvement:.1f}% collision reduction")
        
        # Show statistical significance
        from scipy import stats
        ext_collisions = self.df[self.df['condition_clean'] == 'External Only']['collision']
        fusion_collisions = self.df[self.df['condition_clean'] == 'Fusion']['collision']
        
        t_stat, p_value = stats.ttest_ind(ext_collisions, fusion_collisions)
        print(f"\n📊 Statistical Significance (External Only vs Fusion):")
        print(f"   t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
        if p_value < 0.05:
            print(f"   ✅ Statistically significant improvement (p < 0.05)")
        else:
            print(f"   ❌ Not statistically significant (p ≥ 0.05)")

    def create_interaction_visualizations(self, save_plots: bool = True):
        """Create IEEE format interaction visualizations focusing on key insights"""
        print("\n" + "="*70)
        print("GENERATING IEEE INTERACTION VISUALIZATIONS")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine the best available safety metric (same logic as comprehensive visualizations)
        safety_metric = 'safety_margin'
        safety_label = 'Safety Margin (m)'
        
        if self.df['safety_margin'].notna().sum() == 0 or 'safety_margin' not in self.df.columns:
            if 'minimum_distance' in self.df.columns and self.df['minimum_distance'].notna().sum() > 0:
                safety_metric = 'minimum_distance'
                safety_label = 'Minimum Distance (m)'
            elif 'safety_score' in self.df.columns and self.df['safety_score'].notna().sum() > 0:
                safety_metric = 'safety_score'
                safety_label = 'Safety Score'
        
        # Figure 3: Critical Interaction Analysis (IEEE 2-column width)
        fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(7.5, 3.5))
        
        # Collision Rate Heatmap - Most important interaction insight
        collision_matrix = self.df.pivot_table(
            values='collision', 
            index='scenario_clean', 
            columns='condition_clean', 
            aggfunc='mean'
        ) * 100
        
        # Reorder columns to match our hierarchy
        condition_order = ['Internal Only', 'Fusion', 'External Only']
        collision_matrix = collision_matrix.reindex(columns=[c for c in condition_order if c in collision_matrix.columns])
        
        sns.heatmap(collision_matrix, annot=True, fmt='.1f', cmap='Reds', 
                   ax=ax5, cbar_kws={'label': 'Collision Rate (%)'})
        ax5.set_title('(e) Collision Rate by\nScenario × Condition', fontweight='bold')
        ax5.set_ylabel('Scenario')
        ax5.set_xlabel('Condition')
        ax5.tick_params(axis='x', rotation=45)
        
        # Driver State Integration Benefits - Key research finding
        fusion_data = []
        for scenario in sorted(self.df['scenario_clean'].unique()):
            scenario_data = self.df[self.df['scenario_clean'] == scenario]
            scenario_results = {}
            
            for condition in condition_order:
                if condition in scenario_data['condition_clean'].unique():
                    collision_rate = scenario_data[scenario_data['condition_clean'] == condition]['collision'].mean() * 100
                    scenario_results[condition] = collision_rate
            
            scenario_results['Scenario'] = scenario
            fusion_data.append(scenario_results)
        
        fusion_df = pd.DataFrame(fusion_data).set_index('Scenario')
        
        # Plot grouped bar chart
        x = np.arange(len(fusion_df.index))
        width = 0.25
        colors = ['#2E8B57', '#4682B4', '#CD5C5C']
        
        for i, (condition, color) in enumerate(zip(condition_order, colors)):
            if condition in fusion_df.columns:
                values = fusion_df[condition].values
                bars = ax6.bar(x + i*width, values, width, label=condition, 
                              color=color, alpha=0.8)
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    if not np.isnan(val):
                        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                                f'{val:.1f}%', ha='center', va='bottom', fontsize=5)
        
        ax6.set_title('(f) Driver State Integration\nBenefits', fontweight='bold')
        ax6.set_ylabel('Collision Rate (%)')
        ax6.set_xlabel('Scenario')
        ax6.set_xticks(x + width)
        ax6.set_xticklabels(fusion_df.index)
        ax6.legend(title='Condition', loc='upper left', fontsize=7, frameon=True, 
                  fancybox=True, shadow=False, ncol=1, bbox_to_anchor=(0.02, 0.98))
        
        plt.tight_layout()
        fig3_file = os.path.join(self.log_directory, f'ieee_interaction_analysis_{timestamp}.png')
        plt.savefig(fig3_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved IEEE interaction visualization: {os.path.basename(fig3_file)}")
        
        # Figure 4: Participant Performance Comparison (IEEE 2-column width)
        fig4, (ax7, ax8) = plt.subplots(1, 2, figsize=(7.5, 3.5))
        
        # Participant Safety Performance
        participants = sorted(self.df['participant'].unique())
        if len(participants) > 1:
            # Safety metric by participant
            participant_safety_data = []
            participant_labels = []
            
            for participant in participants:
                p_data = self.df[self.df['participant'] == participant][safety_metric].dropna()
                if len(p_data) > 0:
                    participant_safety_data.append(p_data.values)
                    participant_labels.append(participant)
            
            if len(participant_safety_data) > 0:
                participant_colors = plt.cm.Set3(np.linspace(0, 1, len(participant_labels)))
                
                bp3 = ax7.boxplot(participant_safety_data, labels=participant_labels, 
                                 patch_artist=True, widths=0.6)
                
                for patch, color in zip(bp3['boxes'], participant_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
                
                ax7.set_title(f'(g) {safety_label.split("(")[0].strip()}\nby Participant', fontweight='bold')
                ax7.set_ylabel(safety_label)
                ax7.set_xlabel('Participant')
                ax7.tick_params(axis='x', rotation=45)
            else:
                ax7.text(0.5, 0.5, 'No safety data\navailable', ha='center', va='center', 
                        transform=ax7.transAxes, fontsize=12)
                ax7.set_title('(g) Safety by Participant', fontweight='bold')
            
            # Collision Rate by Participant
            participant_collision = self.df.groupby('participant')['collision'].agg(['count', 'sum']).reset_index()
            participant_collision = participant_collision[participant_collision['count'] > 0]  # Only participants with data
            participant_collision['collision_rate'] = (participant_collision['sum'] / participant_collision['count']) * 100
            
            if len(participant_collision) > 0:
                bars3 = ax8.bar(range(len(participant_collision)), participant_collision['collision_rate'].values,
                               color=participant_colors[:len(participant_collision)], alpha=0.8, width=0.6)
                ax8.set_title('(h) Collision Rate by Participant', fontweight='bold')
                ax8.set_ylabel('Collision Rate (%)')
                ax8.set_xlabel('Participant')
                ax8.set_xticks(range(len(participant_collision)))
                ax8.set_xticklabels(participant_collision['participant'].values, rotation=45)
                
                # Add value labels on bars
                for bar, rate in zip(bars3, participant_collision['collision_rate'].values):
                    ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                            f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=5)
            else:
                ax8.text(0.5, 0.5, 'No collision data\navailable', ha='center', va='center', 
                        transform=ax8.transAxes, fontsize=12)
                ax8.set_title('(h) Collision Rate by Participant', fontweight='bold')
        else:
            # Single participant case
            ax7.text(0.5, 0.5, 'Only one participant\nin dataset', ha='center', va='center', 
                    transform=ax7.transAxes, fontsize=12)
            ax7.set_title('(g) Safety by Participant', fontweight='bold')
            
            ax8.text(0.5, 0.5, 'Only one participant\nin dataset', ha='center', va='center', 
                    transform=ax8.transAxes, fontsize=12)
            ax8.set_title('(h) Collision Rate by Participant', fontweight='bold')
        
        plt.tight_layout()
        fig4_file = os.path.join(self.log_directory, f'ieee_participant_comparison_{timestamp}.png')
        plt.savefig(fig4_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved IEEE participant comparison: {os.path.basename(fig4_file)}")
        
        return [fig3_file, fig4_file]

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
        analyzer.scenario_condition_interaction_analysis()
        analyzer.sensor_limitation_analysis()
        analyzer.driver_attentiveness_necessity_analysis()
        
        # Generate visualizations
        basic_viz_files = analyzer.create_comprehensive_visualizations()
        interaction_viz_files = analyzer.create_interaction_visualizations()
        
        print(f"\n📊 All IEEE format visualizations generated:")
        all_viz_files = basic_viz_files + (interaction_viz_files if isinstance(interaction_viz_files, list) else [interaction_viz_files])
        for viz_file in all_viz_files:
            print(f"   - {os.path.basename(viz_file)}")
        
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