"""
Plotting utility to visualize different noise schedules.

This module provides functions to plot alpha_bar and gamma_prime values
for all available noise schedule types.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
from pathlib import Path

from src.embeddings.noise_schedules import create_noise_schedule, NoiseSchedule


class ScheduleWrapper(nn.Module):
    """Wrapper module to enable standalone schedule evaluation."""
    schedule: NoiseSchedule
    
    @nn.compact
    def __call__(self, t: jnp.ndarray):
        """Call the schedule's get_alpha_bar method."""
        return self.schedule.get_alpha_bar(t)
    
    @nn.compact
    def get_alpha_bar_gamma_prime(self, t: jnp.ndarray):
        """Call the schedule's get_alpha_bar_gamma_prime method."""
        return self.schedule.get_alpha_bar_gamma_prime(t)


def plot_noise_schedules(
    schedule_types: Optional[List[str]] = None,
    t_range: tuple = (0.0, 1.0),
    num_points: int = 1000,
    default_params: Optional[dict] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (14, 6)
):
    """
    Plot alpha_bar and gamma values for different noise schedules.
    
    Args:
        schedule_types: List of schedule types to plot. If None, plots all available types.
        t_range: Tuple of (t_min, t_max) for time range
        num_points: Number of time points to evaluate
        default_params: Dictionary of default parameters for schedules
        output_path: Path to save the figure. If None, displays the figure.
        figsize: Figure size (width, height)
    """
    if schedule_types is None:
        schedule_types = [
            "linear",
            "cosine",
            "sigmoid",
            "exponential",
            "cauchy",
            "laplace",
            "quadratic",
            "polynomial",
        ]
    
    if default_params is None:
        default_params = {
            "alpha_bar_min": 0.01,
            "alpha_bar_max": 0.99,
            "s": 0.008,
            "k": 10.0,
            "t_mid": 0.5,
            "beta": 2.0,
            "loc": 0.5,
            "scale": 0.1,
            "power": 2.0,
        }
    
    # Create time points
    t = np.linspace(t_range[0], t_range[1], num_points)
    t_jax = jnp.array(t)
    
    # Initialize JAX random key
    key = jax.random.PRNGKey(42)
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Colors for different schedules
    colors = plt.cm.tab10(np.linspace(0, 1, len(schedule_types)))
    
    # Store results
    results = {}
    
    for i, schedule_type in enumerate(schedule_types):
        try:
            # Create schedule with default parameters
            # Extract relevant parameters for this schedule type
            schedule_kwargs = {}
            # All schedules now use alpha_bar_min and alpha_bar_max
            if schedule_type in ["linear", "cosine", "sigmoid", "exponential", "cauchy", "laplace", "quadratic", "polynomial"]:
                schedule_kwargs["alpha_bar_min"] = default_params.get("alpha_bar_min", 0.01)
                schedule_kwargs["alpha_bar_max"] = default_params.get("alpha_bar_max", 0.99)
            
            schedule = create_noise_schedule(schedule_type, learnable=False, **schedule_kwargs)
            
            # Wrap schedule in a module to enable initialization
            wrapper = ScheduleWrapper(schedule=schedule)
            
            # Initialize parameters using the get_alpha_bar_gamma_prime method
            variables = wrapper.init(key, t_jax, method=wrapper.get_alpha_bar_gamma_prime)
            
            # Get alpha_bar and gamma_prime values
            alpha_bar_gamma_prime = wrapper.apply(variables, t_jax, method=wrapper.get_alpha_bar_gamma_prime)
            if alpha_bar_gamma_prime is None:
                raise ValueError(f"get_alpha_bar_gamma_prime returned None for {schedule_type}")
            alpha_bar = np.array(alpha_bar_gamma_prime[0])
            gamma_prime = np.array(alpha_bar_gamma_prime[1])
            
            # Plot alpha_bar
            ax1.plot(t, alpha_bar, label=schedule_type, color=colors[i], linewidth=2, alpha=0.8)
            
            # Plot gamma_prime
            ax2.plot(t, gamma_prime, label=schedule_type, color=colors[i], linewidth=2, alpha=0.8)
            
            results[schedule_type] = {
                'alpha_bar': alpha_bar,
                'gamma_prime': gamma_prime
            }
            
        except Exception as e:
            print(f"Warning: Could not plot {schedule_type}: {e}")
            continue
    
    # Format alpha_bar panel
    ax1.set_xlabel('Time t', fontsize=12)
    ax1.set_ylabel(r'$\bar{\alpha}(t)$', fontsize=14)
    ax1.set_title('Noise Schedules: Alpha Bar', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    ax1.set_xlim(t_range)
    ax1.set_ylim([0, 1])
    
    # Format gamma_prime panel
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel(r'$\gamma\'(t)$', fontsize=14)
    ax2.set_title('Noise Schedules: Gamma Prime', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    ax2.set_xlim(t_range)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved noise schedule plot to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return results


if __name__ == "__main__":
    # Example usage
    plot_noise_schedules(
        output_path="artifacts/noise_schedules_comparison.png"
    )

