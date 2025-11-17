"""
Plotting utility to visualize different flow schedules.

This module provides functions to plot alpha and sigma values
for all available flow schedule types.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
from pathlib import Path

from src.embeddings.flow_schedules import (
    LinearFlowSchedule,
    CosineFlowSchedule,
    SigmoidFlowSchedule,
    ExponentialFlowSchedule,
    CauchyFlowSchedule,
    LaplaceFlowSchedule,
    PolynomialFlowSchedule,
)


def plot_flow_schedules(
    schedule_classes: Optional[List] = None,
    t_range: tuple = (0.0, 1.0),
    num_points: int = 1000,
    default_params: Optional[dict] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (14, 6),
    ndims: int = 2
):
    """
    Plot alpha, sigma, and SNR values for different flow schedules.
    
    Creates a three-panel figure showing:
    - Alpha values (α(t))
    - Sigma values (σ(t))
    - Signal-to-noise ratio (α²(t) / σ²(t))
    
    Args:
        schedule_classes: List of schedule classes to plot. If None, plots all available types.
        t_range: Tuple of (t_min, t_max) for time range
        num_points: Number of time points to evaluate
        default_params: Dictionary of default parameters for schedules
        output_path: Path to save the figure. If None, displays the figure.
        figsize: Figure size (width, height) - ignored, uses (21, 6) for three panels
        ndims: Number of dimensions for the schedule (default: 2)
    """
    if schedule_classes is None:
        schedule_classes = [
            LinearFlowSchedule,
            CosineFlowSchedule,
            SigmoidFlowSchedule,
            ExponentialFlowSchedule,
            CauchyFlowSchedule,
            LaplaceFlowSchedule,
            PolynomialFlowSchedule,
        ]
    
    if default_params is None:
        default_params = {
            "alpha_min": 0.0,
            "alpha_max": 1.0,
            "sigma_min": 0.0,
            "sigma_max": 1.0,
            "k": 10.0,  # for SigmoidFlowSchedule
            "beta": 2.0,  # for ExponentialFlowSchedule
            "loc": 0.5,  # for CauchyFlowSchedule and LaplaceFlowSchedule
            "log_scale": -1.0,  # for CauchyFlowSchedule and LaplaceFlowSchedule
            "log_power": 0.69,  # for PolynomialFlowSchedule (power = exp(0.693147) ≈ 2.0)
        }
    
    # Create time points
    t = np.linspace(t_range[0], t_range[1], num_points)
    t_jax = jnp.array(t)
    
    # Initialize JAX random key
    key = jr.PRNGKey(42)
    
    # Create figure with three panels
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    
    # Colors for different schedules
    colors = plt.cm.tab10(np.linspace(0, 1, len(schedule_classes)))
    
    # Store results
    results = {}
    
    for i, schedule_class in enumerate(schedule_classes):
        try:
            # Get schedule name
            schedule_name = schedule_class.__name__.replace('FlowSchedule', '').lower()
            
            # Create schedule with default parameters
            schedule_kwargs = {
                "ndims": ndims,
                "learnable": True,
                "alpha_min": default_params.get("alpha_min", 0.0),
                "alpha_max": default_params.get("alpha_max", 1.0),
                "sigma_min": default_params.get("sigma_min", 0.0),
                "sigma_max": default_params.get("sigma_max", 1.0),
            }
            
            # Add schedule-specific parameters
            if schedule_name == "sigmoid":
                schedule_kwargs["k"] = default_params.get("k", 10.0)
            elif schedule_name == "exponential":
                schedule_kwargs["beta"] = default_params.get("beta", 2.0)
            elif schedule_name == "cauchy" or schedule_name == "laplace":
                schedule_kwargs["loc"] = default_params.get("loc", 0.5)
                schedule_kwargs["log_scale"] = default_params.get("log_scale", -1.0)
            elif schedule_name == "polynomial":
                schedule_kwargs["log_power"] = default_params.get("log_power", 0.0)
            
            schedule = schedule_class(**schedule_kwargs)
            
            # Initialize and evaluate alpha
            variables_alpha = schedule.init(key, t_jax, method=schedule.alpha)
            alpha_vals = schedule.apply(variables_alpha, t_jax, method=schedule.alpha)
            alpha_vals = np.array(alpha_vals)
            
            # Initialize and evaluate sigma
            variables_sigma = schedule.init(key, t_jax, method=schedule.sigma)
            sigma_vals = schedule.apply(variables_sigma, t_jax, method=schedule.sigma)
            sigma_vals = np.array(sigma_vals)
            
            # Compute SNR: alpha^2 / sigma^2
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            snr_vals = (alpha_vals ** 2) / (sigma_vals ** 2 + epsilon)
            
            # Plot alpha
            ax1.plot(t, alpha_vals, label=schedule_name, color=colors[i], linewidth=2, alpha=0.8)
            
            # Plot sigma
            ax2.plot(t, sigma_vals, label=schedule_name, color=colors[i], linewidth=2, alpha=0.8)
            
            # Plot SNR
            ax3.plot(t, snr_vals, label=schedule_name, color=colors[i], linewidth=2, alpha=0.8)
            
            results[schedule_name] = {
                'alpha': alpha_vals,
                'sigma': sigma_vals,
                'snr': snr_vals
            }
            
        except Exception as e:
            print(f"Warning: Could not plot {schedule_class.__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Format alpha panel
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel(r'$\alpha(t)$', fontsize=14)
    ax1.set_title('Flow Schedules: Alpha', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    ax1.set_xlim(t_range)
    ax1.set_ylim([0, 1])
    
    # Format sigma panel
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel(r'$\sigma(t)$', fontsize=14)
    ax2.set_title('Flow Schedules: Sigma', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    ax2.set_xlim(t_range)
    ax2.set_ylim([0, 1])
    
    # Format SNR panel
    ax3.set_xlabel('Time t', fontsize=12)
    ax3.set_ylabel(r'SNR = $\alpha^2(t) / \sigma^2(t)$', fontsize=14)
    ax3.set_title('Flow Schedules: Signal-to-Noise Ratio', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=10)
    ax3.set_xlim(t_range)
    ax3.set_yscale('log')  # Use log scale for SNR as it can span many orders of magnitude
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved flow schedule plot to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return results


if __name__ == "__main__":
    # Example usage
    plot_flow_schedules(
        output_path="artifacts/flow_schedules_comparison.png"
    )

