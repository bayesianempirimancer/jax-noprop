#!/usr/bin/env python3
"""
Generate comprehensive report from experiment results.
"""

import sys
import os
import argparse
try:
    from results_tracker import ResultsTracker
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.flow_models.results_tracker import ResultsTracker


def main():
    """Generate comprehensive report."""
    parser = argparse.ArgumentParser(description='Generate comprehensive experiment report')
    parser.add_argument('--results_file', type=str, default='experiment_results.md',
                       help='Markdown file with experiment results')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for detailed report (auto-generated if not provided)')
    parser.add_argument('--json_file', type=str, default='experiment_data.json',
                       help='JSON file to save experiment data')
    
    args = parser.parse_args()
    
    print("📊 Generating comprehensive experiment report...")
    print(f"Results file: {args.results_file}")
    
    # Check if results file exists
    if not os.path.exists(args.results_file):
        print(f"❌ Results file not found: {args.results_file}")
        print("Run some experiments first using run_experiments.py")
        return
    
    # Load results tracker
    tracker = ResultsTracker(args.results_file)
    
    # Generate detailed report
    report_file = tracker.generate_detailed_report(args.output_file)
    print(f"📄 Detailed report saved to: {report_file}")
    
    # Save JSON data
    tracker.save_json(args.json_file)
    
    # Print summary
    summary = tracker.get_experiment_summary()
    print(f"\n📈 Experiment Summary:")
    print(f"  Total experiments: {summary['total_experiments']}")
    print(f"  Completed experiments: {summary['completed_experiments']}")
    
    if summary['completed_experiments'] > 0:
        print(f"\n🔍 Best performing experiments:")
        
        # Find best experiments
        completed = [exp for exp in summary['experiments'] if exp['status'] == 'completed']
        
        if completed:
            # Best validation loss
            best_val = min(completed, key=lambda x: x['results'].get('final_val_loss', float('inf')))
            print(f"  Best validation loss: {best_val['name']}")
            print(f"    Val loss: {best_val['results'].get('final_val_loss', 'N/A')}")
            print(f"    Parameters: recon_weight={best_val['parameters'].get('recon_weight', 'N/A')}, decoder_type={best_val['parameters'].get('decoder_type', 'N/A')}")
            
            # Best training accuracy
            best_acc = max(completed, key=lambda x: x['results'].get('train_accuracy', 0))
            print(f"  Best training accuracy: {best_acc['name']}")
            print(f"    Train acc: {best_acc['results'].get('train_accuracy', 'N/A')}")
            print(f"    Parameters: recon_weight={best_acc['parameters'].get('recon_weight', 'N/A')}, decoder_type={best_acc['parameters'].get('decoder_type', 'N/A')}")
    
    print(f"\n✅ Report generation completed!")
    print(f"📄 Detailed report: {report_file}")
    print(f"📊 JSON data: {args.json_file}")


if __name__ == "__main__":
    main()
