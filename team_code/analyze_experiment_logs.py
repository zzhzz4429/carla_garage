#!/usr/bin/env python3
"""
Efficient Experiment Log Analysis Tool
Analyzes all experiment logs to compare conditions and extract insights
"""

import json
import os
import glob
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

class ExperimentLogAnalyzer:
    def __init__(self, log_directory: str = "/home/ascc304/carla_garage/experiment_logs/P002/PED"):
        self.log_directory = log_directory
        self.data = []
        self.summary_stats = {}
        
    def load_all_logs(self) -> int:
        """Load all JSON log files from the experiment directory"""
        json_files = glob.glob(os.path.join(self.log_directory, "*.json"))
        
        print(f"🔍 Found {len(json_files)} log files in {self.log_directory}")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    log_data = json.load(f)
                    
                # Extract key metrics from the log
                metrics = self._extract_metrics(log_data, file_path)
                if metrics:
                    self.data.append(metrics)
                    
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                
        print(f"✅ Successfully loaded {len(self.data)} experiment logs")
        return len(self.data)
    
    def _extract_metrics(self, log_data: Dict, file_path: str) -> Dict:
        """Extract key metrics from a single log file"""
        try:
            # Parse filename for metadata - handle new format with scenario
            filename = os.path.basename(file_path)
            parts = filename.replace('.json', '').split('_')
            
            # New format: ttc_SC-EV_fusion_P001_T1_timestamp.json
            # Old format: ttc_fusion_P001_T1_timestamp.json
            if len(parts) >= 5 and parts[1].startswith('SC-'):
                # New format with scenario
                scenario = parts[1]
                condition = '_'.join(parts[2:-3])  # Handle multi-word conditions
                participant = parts[-3]
                trial = parts[-2]
                session = parts[-1]
            elif len(parts) >= 4:
                # Old format without scenario
                scenario = 'SC-EV'  # Default scenario
                condition = '_'.join(parts[1:-3])  # Handle multi-word conditions
                participant = parts[-3]
                trial = parts[-2]
                session = parts[-1]
            else:
                return None
            
            # Extract metadata
            metadata = log_data.get('metadata', {})
            experiment_metrics = log_data.get('experiment_metrics', {})
            enhanced_safety = log_data.get('enhanced_safety_analysis', {})
            alert_effectiveness = log_data.get('alert_effectiveness_analysis', {})
            
            # Get scenario from metadata if available, otherwise use parsed value
            scenario = metadata.get('scenario_name', scenario)
            
            # Core safety metrics
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
            
            return {
                # Experiment metadata
                'scenario': scenario,
                'condition': condition,
                'participant': participant,
                'trial': trial,
                'session': session,
                'filename': filename,
                'duration': metadata.get('duration_seconds', 0),
                
                # Safety outcomes
                'safety_outcome': experiment_metrics.get('safety_outcome', 'unknown'),
                'minimum_distance': minimum_distance,
                'safety_margin': safety_margin,
                'safety_score': safety_score,
                
                # Alert effectiveness
                'alert_timing': timing_analysis.get('timing_classification', 'unknown'),
                'timing_quality': timing_analysis.get('timing_quality', 'unknown'),
                'first_alert_distance': timing_analysis.get('first_alert_distance'),
                'alert_effectiveness': effectiveness_analysis.get('effectiveness_classification', 'unknown'),
                'alert_precision': precision_analysis.get('precision_classification', 'unknown'),
                'false_positive_risk': precision_analysis.get('false_positive_risk', 0.0),
                
                # Risk metrics
                'peak_risk_reached': peak_risk,
                'emergency_events_count': emergency_count,
                'total_alerts': experiment_metrics.get('total_alerts', 0),
                
                # Driver context
                'driver_state_at_alert': timing_analysis.get('driver_state_at_alert', 'unknown')
            }
            
        except Exception as e:
            print(f"⚠️ Error extracting metrics from {file_path}: {e}")
            return None
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics by condition"""
        if not self.data:
            return {}
        
        df = pd.DataFrame(self.data)
        
        # Group by condition
        summary = {}
        
        for condition in df['condition'].unique():
            condition_data = df[df['condition'] == condition]
            
            # Safety metrics
            safety_stats = {
                'total_trials': len(condition_data),
                'safety_outcomes': condition_data['safety_outcome'].value_counts().to_dict(),
                'avg_minimum_distance': condition_data['minimum_distance'].mean() if condition_data['minimum_distance'].notna().any() else None,
                'avg_safety_margin': condition_data['safety_margin'].mean() if condition_data['safety_margin'].notna().any() else None,
                'avg_safety_score': condition_data['safety_score'].mean(),
                'collision_rate': (condition_data['safety_outcome'] == 'collision').sum() / len(condition_data) * 100
            }
            
            # Alert effectiveness
            alert_stats = {
                'alert_timing_distribution': condition_data['alert_timing'].value_counts().to_dict(),
                'alert_effectiveness_distribution': condition_data['alert_effectiveness'].value_counts().to_dict(),
                'avg_first_alert_distance': condition_data['first_alert_distance'].mean() if condition_data['first_alert_distance'].notna().any() else None,
                'avg_total_alerts': condition_data['total_alerts'].mean(),
                'false_positive_rate': condition_data['false_positive_risk'].mean()
            }
            
            # Risk metrics
            risk_stats = {
                'avg_peak_risk': condition_data['peak_risk_reached'].mean(),
                'avg_emergency_events': condition_data['emergency_events_count'].mean()
            }
            
            summary[condition] = {
                'safety_metrics': safety_stats,
                'alert_metrics': alert_stats,
                'risk_metrics': risk_stats
            }
        
        self.summary_stats = summary
        return summary
    
    def compare_conditions(self) -> Dict:
        """Compare key metrics across conditions"""
        if not self.summary_stats:
            self.generate_summary_statistics()
        
        # Key comparison metrics
        comparison = {
            'safety_margin_ranking': {},
            'collision_rate_ranking': {},
            'alert_timing_quality': {},
            'alert_effectiveness': {}
        }
        
        # Safety margin comparison
        for condition, stats in self.summary_stats.items():
            avg_margin = stats['safety_metrics'].get('avg_safety_margin')
            if avg_margin is not None:
                comparison['safety_margin_ranking'][condition] = avg_margin
        
        # Collision rate comparison
        for condition, stats in self.summary_stats.items():
            collision_rate = stats['safety_metrics'].get('collision_rate', 0)
            comparison['collision_rate_ranking'][condition] = collision_rate
        
        # Alert timing quality
        for condition, stats in self.summary_stats.items():
            timing_dist = stats['alert_metrics'].get('alert_timing_distribution', {})
            appropriate_rate = timing_dist.get('appropriate', 0) / sum(timing_dist.values()) * 100 if timing_dist else 0
            comparison['alert_timing_quality'][condition] = appropriate_rate
        
        # Alert effectiveness
        for condition, stats in self.summary_stats.items():
            effect_dist = stats['alert_metrics'].get('alert_effectiveness_distribution', {})
            effective_rate = (effect_dist.get('effective', 0) + effect_dist.get('highly_effective', 0)) / sum(effect_dist.values()) * 100 if effect_dist else 0
            comparison['alert_effectiveness'][condition] = effective_rate
        
        return comparison
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report"""
        if not self.data:
            return "No data loaded. Please run load_all_logs() first."
        
        summary = self.generate_summary_statistics()
        comparison = self.compare_conditions()
        
        report = f"""
{'='*60}
EXPERIMENT LOG ANALYSIS REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Log Directory: {self.log_directory}
{'='*60}

=== DATASET OVERVIEW ===
Total Trials Analyzed: {len(self.data)}
Conditions Found: {', '.join(summary.keys())}
Scenarios Found: {', '.join(set([d['scenario'] for d in self.data]))}
Participants: {', '.join(set([d['participant'] for d in self.data]))}

"""
        
        # Condition-by-condition breakdown
        for condition, stats in summary.items():
            safety = stats['safety_metrics']
            alerts = stats['alert_metrics']
            risks = stats['risk_metrics']
            
            # Format values safely
            avg_min_dist = f"{safety['avg_minimum_distance']:.2f}m" if safety['avg_minimum_distance'] else "N/A"
            avg_safety_margin = f"{safety['avg_safety_margin']:.2f}m" if safety['avg_safety_margin'] else "N/A"
            avg_alert_dist = f"{alerts['avg_first_alert_distance']:.1f}m" if alerts['avg_first_alert_distance'] else "N/A"
            
            report += f"""
=== {condition.upper().replace('_', ' ')} CONDITION ===
Trials: {safety['total_trials']}

Safety Performance:
  Average Minimum Distance: {avg_min_dist}
  Average Safety Margin: {avg_safety_margin}
  Average Safety Score: {safety['avg_safety_score']:.2f}
  Collision Rate: {safety['collision_rate']:.1f}%
  Safety Outcomes: {safety['safety_outcomes']}

Alert Performance:
  Average First Alert Distance: {avg_alert_dist}
  Average Total Alerts: {alerts['avg_total_alerts']:.1f}
  False Positive Rate: {alerts['false_positive_rate']:.1f}
  Alert Timing: {alerts['alert_timing_distribution']}
  Alert Effectiveness: {alerts['alert_effectiveness_distribution']}

Risk Metrics:
  Average Peak Risk: {risks['avg_peak_risk']:.3f}
  Average Emergency Events: {risks['avg_emergency_events']:.1f}

"""
        
        # Cross-condition comparison
        report += f"""
{'='*60}
CROSS-CONDITION COMPARISON
{'='*60}

Safety Margin Ranking (Higher = Better):
"""
        
        # Sort conditions by safety margin
        safety_ranking = sorted(comparison['safety_margin_ranking'].items(), 
                              key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
        for i, (condition, margin) in enumerate(safety_ranking):
            if margin is not None:
                report += f"  {i+1}. {condition.replace('_', ' ').title()}: {margin:.2f}m\n"
        
        report += f"""
Collision Rate Ranking (Lower = Better):
"""
        collision_ranking = sorted(comparison['collision_rate_ranking'].items(), key=lambda x: x[1])
        for i, (condition, rate) in enumerate(collision_ranking):
            report += f"  {i+1}. {condition.replace('_', ' ').title()}: {rate:.1f}%\n"
        
        report += f"""
Alert Timing Quality (% Appropriate):
"""
        timing_ranking = sorted(comparison['alert_timing_quality'].items(), key=lambda x: x[1], reverse=True)
        for condition, rate in timing_ranking:
            report += f"  {condition.replace('_', ' ').title()}: {rate:.1f}%\n"
        
        report += f"""
Alert Effectiveness Rate:
"""
        effectiveness_ranking = sorted(comparison['alert_effectiveness'].items(), key=lambda x: x[1], reverse=True)
        for condition, rate in effectiveness_ranking:
            report += f"  {condition.replace('_', ' ').title()}: {rate:.1f}%\n"
        
        report += "\n" + "="*60 + "\n"
        
        return report
    
    def save_analysis_results(self, output_file: str = None):
        """Save analysis results to files in the experiment logs directory"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"experiment_analysis_{timestamp}"
        
        # Save files in the same directory as the experiment logs
        report_file = os.path.join(self.log_directory, f"{output_file}_report.txt")
        
        # Save comprehensive report
        report = self.generate_report()
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save raw data as CSV for further analysis
        csv_file = None
        if self.data:
            df = pd.DataFrame(self.data)
            csv_file = os.path.join(self.log_directory, f"{output_file}_data.csv")
            df.to_csv(csv_file, index=False)
            
            print(f"✅ Analysis saved to {self.log_directory}:")
            print(f"   Report: {os.path.basename(report_file)}")
            print(f"   Data: {os.path.basename(csv_file)}")
        else:
            print(f"✅ Report saved to {self.log_directory}:")
            print(f"   Report: {os.path.basename(report_file)}")
        
        return report_file, csv_file

def main():
    """Main analysis execution"""
    print("🔬 Experiment Log Analysis Tool")
    print("="*50)
    
    # Initialize analyzer
    analyzer = ExperimentLogAnalyzer()
    
    # Load all available logs
    count = analyzer.load_all_logs()
    
    if count == 0:
        print("❌ No experiment logs found!")
        return
    
    # Generate and display report
    report = analyzer.generate_report()
    print(report)
    
    # Save results
    analyzer.save_analysis_results()
    
    print("🎯 Analysis complete! Check the generated files for detailed results.")

if __name__ == "__main__":
    main() 