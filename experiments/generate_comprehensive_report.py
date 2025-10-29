#!/usr/bin/env python3
"""
Generate a comprehensive report from the experiment results.
"""

import os
import re
from datetime import datetime

def parse_experiment_results():
    """Parse the experiment results from the markdown file."""
    results_file = "experiment_results.md"
    
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found!")
        return []
    
    experiments = []
    
    with open(results_file, 'r') as f:
        lines = f.readlines()
    
    # Find the table section
    in_table = False
    for line in lines:
        line = line.strip()
        
        # Skip header lines
        if line.startswith('|') and 'Experiment' in line:
            in_table = True
            continue
        elif line.startswith('|') and '---' in line:
            continue
        elif line.startswith('|') and in_table:
            # Parse table row
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 8:
                experiment = {
                    'name': parts[0],
                    'parameters': parts[1],
                    'train_loss': float(parts[2]) if parts[2] != 'N/A' else None,
                    'val_loss': float(parts[3]) if parts[3] != 'N/A' else None,
                    'train_acc': float(parts[4]) if parts[4] != 'N/A' else None,
                    'val_acc': float(parts[5]) if parts[5] != 'N/A' else None,
                    'time': parts[6],
                    'status': parts[7]
                }
                experiments.append(experiment)
        elif not line.startswith('|'):
            in_table = False
    
    return experiments

def generate_report():
    """Generate comprehensive report."""
    experiments = parse_experiment_results()
    
    if not experiments:
        print("No experiments found!")
        return
    
    report_file = f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, 'w') as f:
        f.write("# Comprehensive Diffusion Model Experiment Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Experiments:** {len(experiments)}\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        
        train_losses = [exp['train_loss'] for exp in experiments if exp['train_loss'] is not None]
        val_losses = [exp['val_loss'] for exp in experiments if exp['val_loss'] is not None]
        train_accs = [exp['train_acc'] for exp in experiments if exp['train_acc'] is not None]
        val_accs = [exp['val_acc'] for exp in experiments if exp['val_acc'] is not None]
        
        if train_losses:
            f.write(f"- **Train Loss:** {min(train_losses):.6f} - {max(train_losses):.6f} (avg: {sum(train_losses)/len(train_losses):.6f})\n")
        if val_losses:
            f.write(f"- **Val Loss:** {min(val_losses):.6f} - {max(val_losses):.6f} (avg: {sum(val_losses)/len(val_losses):.6f})\n")
        if train_accs:
            f.write(f"- **Train Accuracy:** {min(train_accs):.4f} - {max(train_accs):.4f} (avg: {sum(train_accs)/len(train_accs):.4f})\n")
        if val_accs:
            f.write(f"- **Val Accuracy:** {min(val_accs):.4f} - {max(val_accs):.4f} (avg: {sum(val_accs)/len(val_accs):.4f})\n")
        
        f.write("\n")
        
        # Individual experiments
        f.write("## Individual Experiments\n\n")
        for i, exp in enumerate(experiments):
            f.write(f"### Experiment {i+1}: {exp['name']}\n\n")
            f.write(f"**Parameters:** {exp['parameters']}\n")
            f.write(f"**Status:** {exp['status']}\n\n")
            
            f.write("**Results:**\n")
            if exp['train_loss'] is not None:
                f.write(f"- Train Loss: {exp['train_loss']:.6f}\n")
            if exp['val_loss'] is not None:
                f.write(f"- Val Loss: {exp['val_loss']:.6f}\n")
            if exp['train_acc'] is not None:
                f.write(f"- Train Accuracy: {exp['train_acc']:.4f}\n")
            if exp['val_acc'] is not None:
                f.write(f"- Val Accuracy: {exp['val_acc']:.4f}\n")
            f.write(f"- Training Time: {exp['time']}\n\n")
            
            f.write("---\n\n")
        
        # Analysis
        f.write("## Analysis\n\n")
        
        # Find best performing experiments
        if val_losses:
            best_val_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
            best_val_exp = experiments[best_val_idx]
            f.write(f"**Best Validation Performance:** {best_val_exp['name']}\n")
            f.write(f"- Val Loss: {best_val_exp['val_loss']:.6f}\n")
            f.write(f"- Parameters: {best_val_exp['parameters']}\n\n")
        
        if train_accs:
            best_train_acc_idx = max(range(len(train_accs)), key=lambda i: train_accs[i])
            best_train_acc_exp = experiments[best_train_acc_idx]
            f.write(f"**Best Training Accuracy:** {best_train_acc_exp['name']}\n")
            f.write(f"- Train Accuracy: {best_train_acc_exp['train_acc']:.4f}\n")
            f.write(f"- Parameters: {best_train_acc_exp['parameters']}\n\n")
        
        # Parameter analysis
        f.write("### Parameter Effects\n\n")
        
        # Group by reconstruction weight
        recon_weights = {}
        for exp in experiments:
            # Extract recon_weight from parameters string
            params = exp['parameters']
            if 'recon_weight=' in params:
                match = re.search(r'recon_weight=([0-9.]+)', params)
                if match:
                    recon_weight = float(match.group(1))
                    if recon_weight not in recon_weights:
                        recon_weights[recon_weight] = []
                    recon_weights[recon_weight].append(exp)
        
        if recon_weights:
            f.write("**Reconstruction Weight Effects:**\n")
            for recon_weight in sorted(recon_weights.keys()):
                exps = recon_weights[recon_weight]
                avg_val_loss = sum(exp['val_loss'] for exp in exps if exp['val_loss'] is not None) / len([exp for exp in exps if exp['val_loss'] is not None])
                avg_train_acc = sum(exp['train_acc'] for exp in exps if exp['train_acc'] is not None) / len([exp for exp in exps if exp['train_acc'] is not None])
                f.write(f"- recon_weight={recon_weight}: avg val_loss={avg_val_loss:.6f}, avg train_acc={avg_train_acc:.4f}\n")
            f.write("\n")
        
        # Group by decoder type
        decoder_types = {}
        for exp in experiments:
            params = exp['parameters']
            if 'decoder_type=' in params:
                match = re.search(r'decoder_type=([a-z]+)', params)
                if match:
                    decoder_type = match.group(1)
                    if decoder_type not in decoder_types:
                        decoder_types[decoder_type] = []
                    decoder_types[decoder_type].append(exp)
        
        if decoder_types:
            f.write("**Decoder Type Effects:**\n")
            for decoder_type in sorted(decoder_types.keys()):
                exps = decoder_types[decoder_type]
                avg_val_loss = sum(exp['val_loss'] for exp in exps if exp['val_loss'] is not None) / len([exp for exp in exps if exp['val_loss'] is not None])
                avg_train_acc = sum(exp['train_acc'] for exp in exps if exp['train_acc'] is not None) / len([exp for exp in exps if exp['train_acc'] is not None])
                f.write(f"- decoder_type={decoder_type}: avg val_loss={avg_val_loss:.6f}, avg train_acc={avg_train_acc:.4f}\n")
            f.write("\n")
        
        # Recommendations
        f.write("### Recommendations\n\n")
        
        if val_losses and train_accs:
            # Find best overall performance (low val loss + high train acc)
            best_overall_idx = min(range(len(experiments)), key=lambda i: 
                (experiments[i]['val_loss'] or float('inf')) - (experiments[i]['train_acc'] or 0))
            best_overall = experiments[best_overall_idx]
            
            f.write(f"**Recommended Configuration:** {best_overall['name']}\n")
            f.write(f"- Parameters: {best_overall['parameters']}\n")
            f.write(f"- Performance: val_loss={best_overall['val_loss']:.6f}, train_acc={best_overall['train_acc']:.4f}\n\n")
        
        f.write("### Key Findings\n\n")
        f.write("1. **Reconstruction Weight Impact:** Compare performance across different recon_weight values\n")
        f.write("2. **Decoder Type Impact:** Compare performance across different decoder_type values\n")
        f.write("3. **Overfitting Analysis:** Check for gaps between train and validation performance\n")
        f.write("4. **Convergence:** Look at training time and final loss values\n\n")
        
        f.write("### Next Steps\n\n")
        f.write("1. Run more experiments with intermediate parameter values\n")
        f.write("2. Test different learning rates and batch sizes\n")
        f.write("3. Experiment with different network architectures\n")
        f.write("4. Analyze training curves and trajectory plots\n")
    
    print(f"Comprehensive report saved to: {report_file}")
    return report_file

if __name__ == "__main__":
    generate_report()
