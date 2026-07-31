"""
QxBin Edge Use-Case #1
Adaptive Decision under Uncertainty — Edge / IoT / Micro-Robot Tier

Personal cubit that continuously updates a Binary Probability Matrix from
noisy sensor readings and collapses only when a decision is forced.
Room-temperature. Zero special hardware. Runs on a Raspberry Pi or phone.

Author: Rupesh Malpani | pikk.company | QxBin Framework
"""

import numpy as np
from typing import Tuple, List


class QxBinEdgeAdaptiveDecision:
    """
    Edge-tier QxBin: turns noisy multi-sensor input into a steerable
    probability grid, then measures only when action is required.
    """

    def __init__(self, grid_size: int = 5, num_actions: int = 4):
        self.grid_size = grid_size
        self.num_actions = num_actions  # e.g. N, E, S, W or 4 discrete choices
        # Start near-uniform (maximum uncertainty)
        self.state = np.ones((grid_size, grid_size), dtype=np.float64)
        self._normalize()
        self.history: List[np.ndarray] = []

    def _normalize(self):
        s = self.state.sum()
        if s > 1e-12:
            self.state /= s

    def ingest_sensors(self, readings: np.ndarray, confidence: float = 0.7,
                       n: int = 2, m: int = 1):
        """
        readings: array of length num_actions, values in [0,1]
                  (higher = more favorable for that action)
        confidence: how strongly we trust this sensor snapshot (0-1)
        n, m: directed exponent coordinates (QxBin core)
        """
        assert len(readings) == self.num_actions
        # Map action scores onto the probability grid via powered fractions
        bias = float(np.clip(np.mean(readings), 0.05, 0.95))
        frac = bias ** n
        tail = (1.0 - bias) ** m

        # Build a soft action preference vector then outer-product into grid
        action_vec = np.interp(
            np.linspace(0, 1, self.grid_size),
            np.linspace(0, 1, self.num_actions),
            readings
        )
        action_vec = action_vec * frac + (1.0 - action_vec) * tail
        new_matrix = np.outer(action_vec, action_vec)

        # Confidence-weighted blend (superposition update)
        alpha = float(np.clip(confidence, 0.05, 0.95))
        self.state = (1.0 - alpha) * self.state + alpha * new_matrix
        self._normalize()
        self.history.append(self.state.copy())
        return self.state

    def decide(self, temperature: float = 0.15) -> Tuple[int, np.ndarray]:
        """
        Collapse the probability matrix into a discrete action.
        temperature softens the collapse (higher = more exploratory).
        Returns (chosen_action_index, collapsed_grid)
        """
        # Project grid onto action axis by averaging columns
        action_probs = self.state.mean(axis=0)
        # Softmax with temperature for controlled exploration
        logits = np.log(action_probs + 1e-12) / max(temperature, 1e-3)
        exp_l = np.exp(logits - logits.max())
        soft = exp_l / exp_l.sum()

        # Map continuous soft vector back to discrete action slots
        action_scores = np.interp(
            np.linspace(0, 1, self.num_actions),
            np.linspace(0, 1, self.grid_size),
            soft
        )
        action_scores /= action_scores.sum()
        chosen = int(np.random.choice(self.num_actions, p=action_scores))

        # Full collapse for visualization / logging
        flat = self.state.flatten()
        idx = np.random.choice(len(flat), p=flat)
        collapsed = np.zeros_like(flat)
        collapsed[idx] = 1.0
        return chosen, collapsed.reshape(self.state.shape)

    def entropy(self) -> float:
        """Current uncertainty of the personal cubit (nats)."""
        p = self.state.flatten()
        p = p[p > 1e-12]
        return float(-np.sum(p * np.log(p)))


if __name__ == "__main__":
    print("=" * 60)
    print("QxBin Edge — Adaptive Decision under Uncertainty")
    print("Tier: Edge / IoT / Micro-robot")
    print("=" * 60)

    qx = QxBinEdgeAdaptiveDecision(grid_size=5, num_actions=4)
    actions = ["North", "East", "South", "West"]

    # Simulate a few noisy sensor updates (e.g. obstacle distances inverted)
    sensor_streams = [
        np.array([0.9, 0.4, 0.2, 0.6]),   # strong preference North
        np.array([0.7, 0.8, 0.3, 0.5]),   # now East looks good too
        np.array([0.3, 0.9, 0.4, 0.2]),   # East dominant
    ]

    for i, readings in enumerate(sensor_streams):
        qx.ingest_sensors(readings, confidence=0.65 + i * 0.1, n=3, m=1)
        print(f"\nStep {i+1} | Entropy = {qx.entropy():.3f}")
        print("Probability grid (rounded):")
        print(np.round(qx.state, 3))

    choice, collapsed = qx.decide(temperature=0.12)
    print(f"\n>>> DECISION COLLAPSED → {actions[choice]} (index {choice})")
    print("Collapsed matrix:")
    print(collapsed)
    print("\nEdge cubit ready for continuous sensor loop.")
