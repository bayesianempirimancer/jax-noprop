"""
Results tracking system for experiment management and markdown reporting.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np


class ResultsTracker:
    """Track and manage experiment results with markdown reporting."""
    
    def __init__(self, results_file: str = "experiment_results.md"):
        """
        Initialize results tracker.
        
        Args:
            results_file: Path to markdown file for storing results
        """
        self.results_file = results_file
        self.experiments = []
        self.start_time = datetime.now()
        
        # Initialize markdown file if it doesn't exist
        if not os.path.exists(self.results_file):
            self._initialize_markdown_file()
    
    def _initialize_markdown_file(self):
        """Initialize the markdown file with header."""
        with open(self.results_file, 'w') as f:
            f.write("# Diffusion Model Experiment Results\n\n")
            f.write(f"**Started:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Experiment Summary\n\n")
            f.write("| Experiment | Parameters | Train Loss | Val Loss | Train Acc | Val Acc | Time | Status |\n")
            f.write("|------------|------------|------------|----------|-----------|---------|------|--------|\n")
    
    def add_experiment(
        self,
        experiment_name: str,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        status: str = "completed"
    ):
        """
        Add an experiment result.
        
        Args:
            experiment_name: Name of the experiment
            parameters: Dictionary of parameters used
            results: Dictionary of results
            status: Status of the experiment
        """
        experiment = {
            'name': experiment_name,
            'parameters': parameters,
            'results': results,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        self.experiments.append(experiment)
        self._update_markdown_file(experiment)
    
    def _update_markdown_file(self, experiment: Dict[str, Any]):
        """Update the markdown file with new experiment."""
        with open(self.results_file, 'a') as f:
            # Add experiment to summary table
            params_str = self._format_parameters(experiment['parameters'])
            results = experiment['results']
            
            train_loss = results.get('final_train_loss', 'N/A')
            val_loss = results.get('final_val_loss', 'N/A')
            train_acc = results.get('train_accuracy', 'N/A')
            val_acc = results.get('val_accuracy', 'N/A')
            training_time = results.get('training_time', 'N/A')
            
            if isinstance(training_time, (int, float)):
                training_time = f"{training_time:.1f}s"
            
            f.write(f"| {experiment['name']} | {params_str} | {train_loss} | {val_loss} | {train_acc} | {val_acc} | {training_time} | {experiment['status']} |\n")
    
    def _format_parameters(self, params: Dict[str, Any]) -> str:
        """Format parameters for display in table."""
        key_params = ['recon_weight', 'decoder_type', 'recon_loss_type', 'reg_weight']
        formatted = []
        for key in key_params:
            if key in params:
                formatted.append(f"{key}={params[key]}")
        return ", ".join(formatted)
    
    def generate_detailed_report(self, output_file: Optional[str] = None):
        """Generate a detailed markdown report."""
        if output_file is None:
            output_file = f"detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(output_file, 'w') as f:
            f.write("# Detailed Experiment Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Experiments:** {len(self.experiments)}\n\n")
            
            # Summary statistics
            self._write_summary_statistics(f)
            
            # Individual experiment details
            f.write("## Individual Experiments\n\n")
            for i, exp in enumerate(self.experiments):
                f.write(f"### Experiment {i+1}: {exp['name']}\n\n")
                f.write(f"**Status:** {exp['status']}\n")
                f.write(f"**Timestamp:** {exp['timestamp']}\n\n")
                
                # Parameters
                f.write("**Parameters:**\n")
                for key, value in exp['parameters'].items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")
                
                # Results
                f.write("**Results:**\n")
                for key, value in exp['results'].items():
                    if isinstance(value, (int, float)):
                        f.write(f"- {key}: {value:.6f}\n")
                    else:
                        f.write(f"- {key}: {value}\n")
                f.write("\n")
                
                f.write("---\n\n")
            
            # Analysis section
            self._write_analysis_section(f)
        
        print(f"Detailed report saved to: {output_file}")
        return output_file
    
    def _write_summary_statistics(self, f):
        """Write summary statistics to file."""
        f.write("## Summary Statistics\n\n")
        
        if not self.experiments:
            f.write("No experiments completed yet.\n\n")
            return
        
        # Filter completed experiments
        completed = [exp for exp in self.experiments if exp['status'] == 'completed']
        
        if not completed:
            f.write("No completed experiments.\n\n")
            return
        
        # Extract metrics
        train_losses = [exp['results'].get('final_train_loss') for exp in completed if 'final_train_loss' in exp['results']]
        val_losses = [exp['results'].get('final_val_loss') for exp in completed if 'final_val_loss' in exp['results']]
        train_accs = [exp['results'].get('train_accuracy') for exp in completed if 'train_accuracy' in exp['results']]
        val_accs = [exp['results'].get('val_accuracy') for exp in completed if 'val_accuracy' in exp['results']]
        
        f.write("### Performance Metrics\n\n")
        
        if train_losses:
            f.write(f"- **Train Loss:** {np.mean(train_losses):.6f} ± {np.std(train_losses):.6f}\n")
            f.write(f"  - Min: {np.min(train_losses):.6f}\n")
            f.write(f"  - Max: {np.max(train_losses):.6f}\n\n")
        
        if val_losses:
            f.write(f"- **Validation Loss:** {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}\n")
            f.write(f"  - Min: {np.min(val_losses):.6f}\n")
            f.write(f"  - Max: {np.max(val_losses):.6f}\n\n")
        
        if train_accs:
            f.write(f"- **Train Accuracy:** {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}\n")
            f.write(f"  - Min: {np.min(train_accs):.4f}\n")
            f.write(f"  - Max: {np.max(train_accs):.4f}\n\n")
        
        if val_accs:
            f.write(f"- **Validation Accuracy:** {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}\n")
            f.write(f"  - Min: {np.min(val_accs):.4f}\n")
            f.write(f"  - Max: {np.max(val_accs):.4f}\n\n")
    
    def _write_analysis_section(self, f):
        """Write analysis section to file."""
        f.write("## Analysis\n\n")
        
        if len(self.experiments) < 2:
            f.write("Need at least 2 experiments for meaningful analysis.\n\n")
            return
        
        # Group by parameters
        param_groups = {}
        for exp in self.experiments:
            if exp['status'] != 'completed':
                continue
            
            # Create parameter signature
            params = exp['parameters']
            key_params = ['recon_weight', 'decoder_type', 'recon_loss_type']
            sig = tuple(params.get(k, 'default') for k in key_params)
            
            if sig not in param_groups:
                param_groups[sig] = []
            param_groups[sig].append(exp)
        
        f.write("### Parameter Effects\n\n")
        
        for sig, exps in param_groups.items():
            if len(exps) < 2:
                continue
            
            f.write(f"**Parameter combination:** {dict(zip(['recon_weight', 'decoder_type', 'recon_loss_type'], sig))}\n")
            
            # Calculate statistics for this group
            train_losses = [exp['results'].get('final_train_loss', 0) for exp in exps]
            val_losses = [exp['results'].get('final_val_loss', 0) for exp in exps]
            
            f.write(f"- Experiments: {len(exps)}\n")
            f.write(f"- Avg Train Loss: {np.mean(train_losses):.6f}\n")
            f.write(f"- Avg Val Loss: {np.mean(val_losses):.6f}\n\n")
        
        f.write("### Recommendations\n\n")
        
        # Find best performing experiments
        completed = [exp for exp in self.experiments if exp['status'] == 'completed']
        if completed:
            # Sort by validation loss (lower is better)
            best_val = min(completed, key=lambda x: x['results'].get('final_val_loss', float('inf')))
            f.write(f"**Best validation performance:** {best_val['name']}\n")
            f.write(f"- Parameters: {self._format_parameters(best_val['parameters'])}\n")
            f.write(f"- Val Loss: {best_val['results'].get('final_val_loss', 'N/A')}\n")
            f.write(f"- Val Accuracy: {best_val['results'].get('val_accuracy', 'N/A')}\n\n")
            
            # Sort by training accuracy (higher is better)
            best_train_acc = max(completed, key=lambda x: x['results'].get('train_accuracy', 0))
            f.write(f"**Best training accuracy:** {best_train_acc['name']}\n")
            f.write(f"- Parameters: {self._format_parameters(best_train_acc['parameters'])}\n")
            f.write(f"- Train Accuracy: {best_train_acc['results'].get('train_accuracy', 'N/A')}\n")
            f.write(f"- Train Loss: {best_train_acc['results'].get('final_train_loss', 'N/A')}\n\n")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        return {
            'total_experiments': len(self.experiments),
            'completed_experiments': len([exp for exp in self.experiments if exp['status'] == 'completed']),
            'experiments': self.experiments
        }
    
    def save_json(self, filename: str = "experiment_data.json"):
        """Save experiment data as JSON."""
        with open(filename, 'w') as f:
            json.dump(self.get_experiment_summary(), f, indent=2)
        print(f"Experiment data saved to: {filename}")


def generate_experiment_name(parameters: Dict[str, Any]) -> str:
    """Generate a descriptive experiment name from parameters."""
    name_parts = []
    
    # Add key parameters to name
    if 'recon_weight' in parameters:
        name_parts.append(f"recon{parameters['recon_weight']}")
    
    if 'decoder_type' in parameters:
        name_parts.append(f"dec{parameters['decoder_type']}")
    
    if 'recon_loss_type' in parameters:
        name_parts.append(f"loss{parameters['recon_loss_type']}")
    
    if 'reg_weight' in parameters and parameters['reg_weight'] > 0:
        name_parts.append(f"reg{parameters['reg_weight']}")
    
    return "_".join(name_parts) if name_parts else f"exp_{int(time.time())}"
