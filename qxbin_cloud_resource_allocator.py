"""
QxBin Cloud Use-Case #2
Probabilistic Resource Allocator / Load Balancer — Cloud / Server Tier

Ensemble of many Binary Probability Matrices that evolve in parallel.
Each "cubit chain" represents a candidate allocation strategy.
Feedback loop steers the whole ensemble toward a target utilization
or fairness objective. Classical hardware only. Scales with cores.

Author: Rupesh Malpani | pikk.company | QxBin Framework
"""

import numpy as np
from typing import Optional, Dict

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _evolve_batch_python(states, biases, ns, ms):
    """Pure-Python fallback when Numba is unavailable."""
    n_cubits = states.shape[0]
    for i in range(n_cubits):
        b = biases[i]
        nn = ns[i]
        mm = ms[i]
        frac = b ** nn
        tail = (1.0 - b) ** mm
        blended = (states[i] * frac + (1.0 - states[i]) * tail) * 0.5
        total = blended.sum()
        if total > 1e-12:
            states[i] = blended / total
        else:
            states[i] = np.ones_like(blended) / blended.size
    return states


if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _evolve_batch(states, biases, ns, ms):
        n_cubits = states.shape[0]
        for i in prange(n_cubits):
            b = biases[i]
            nn = ns[i]
            mm = ms[i]
            frac = b ** nn
            tail = (1.0 - b) ** mm
            blended = (states[i] * frac + (1.0 - states[i]) * tail) * 0.5
            total = blended.sum()
            if total > 1e-12:
                states[i] = blended / total
            else:
                states[i] = np.ones_like(blended) / blended.size
        return states
else:
    _evolve_batch = _evolve_batch_python


class QxBinCloudResourceAllocator:
    """
    Cloud-tier QxBin: many parallel cubit chains that collectively
    search for good resource allocations under uncertainty.
    """

    def __init__(self, num_workers: int = 32, grid_size: int = 6,
                 num_resources: int = 4):
        self.num_workers = num_workers          # parallel cubit chains
        self.grid_size = grid_size
        self.num_resources = num_resources      # e.g. servers, GPUs, queues
        self.states = np.random.rand(num_workers, grid_size, grid_size).astype(np.float64)
        for i in range(num_workers):
            s = self.states[i].sum()
            if s > 0:
                self.states[i] /= s

    def evolve(self, target_load: Optional[np.ndarray] = None):
        """
        One parallel evolution step.
        target_load: optional preferred utilization vector (length num_resources)
        """
        if target_load is None:
            biases = np.random.uniform(0.55, 0.85, self.num_workers)
        else:
            # Bias each chain toward the current target preference
            mean_pref = float(np.clip(np.mean(target_load), 0.1, 0.9))
            biases = np.clip(
                mean_pref + np.random.normal(0, 0.08, self.num_workers),
                0.15, 0.92
            )
        ns = np.random.randint(1, 5, self.num_workers)
        ms = np.random.randint(1, 4, self.num_workers)
        self.states = _evolve_batch(self.states, biases, ns, ms)
        return self.states.mean(axis=0)

    def allocate(self, current_loads: np.ndarray,
                 target_mean: float = 0.65,
                 max_steps: int = 60) -> Dict:
        """
        Feedback optimization: steer ensemble until average probability
        mass aligns with desired load balance, then collapse to a concrete
        resource assignment vector.
        """
        assert len(current_loads) == self.num_resources
        history = []
        for step in range(max_steps):
            # Soft target derived from inverse of current load (prefer under-used)
            inv = 1.0 / (current_loads + 0.15)
            inv /= inv.sum()
            agg = self.evolve(target_load=inv)
            current = float(agg.mean())
            history.append(current)
            if abs(current - target_mean) < 0.012:
                break

        # Project aggregate grid onto resource dimension
        resource_probs = agg.mean(axis=0)
        resource_scores = np.interp(
            np.linspace(0, 1, self.num_resources),
            np.linspace(0, 1, self.grid_size),
            resource_probs
        )
        resource_scores = resource_scores / resource_scores.sum()

        # Soft assignment + hard argmax for the primary choice
        primary = int(np.argmax(resource_scores))
        return {
            "resource_probabilities": resource_scores,
            "primary_resource": primary,
            "ensemble_mean": float(agg.mean()),
            "steps": len(history),
            "history": history,
            "aggregate_grid": agg,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("QxBin Cloud — Probabilistic Resource Allocator")
    print("Tier: Cloud / Server / Batch")
    print("=" * 60)

    allocator = QxBinCloudResourceAllocator(
        num_workers=40, grid_size=6, num_resources=4
    )
    # Simulated current load on 4 resources (0 = idle, 1 = saturated)
    current = np.array([0.82, 0.45, 0.91, 0.33])
    print(f"Current loads: {current}")

    result = allocator.allocate(current, target_mean=0.62, max_steps=50)

    names = ["Server-A", "Server-B", "Server-C", "Server-D"]
    print(f"\nConverged in {result['steps']} steps")
    print(f"Ensemble mean probability: {result['ensemble_mean']:.4f}")
    print("\nSoft allocation probabilities:")
    for name, p in zip(names, result["resource_probabilities"]):
        print(f"  {name}: {p:.3f}")
    print(f"\n>>> PRIMARY ALLOCATION → {names[result['primary_resource']]}")
    print("\nCloud ensemble ready for continuous rebalancing loop.")
