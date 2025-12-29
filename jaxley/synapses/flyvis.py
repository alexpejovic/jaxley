# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional

from jax.nn import relu

from jaxley.synapses.synapse import Synapse


class FlyvisBasic(Synapse):
    """
    Compute synaptic current for tanh synapse (no state).
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_sign": 1.0,
            f"{prefix}_count": 1.0,
            f"{prefix}_strength": 1.0,
        }
        self.synapse_states = {}

    def update_states(
        self,
        states: Dict,
        delta_t: float,
        pre_voltage: float,
        post_voltage: float,
        params: Dict,
    ) -> Dict:
        """Return updated synapse state and current."""
        return {}

    def compute_current(
        self, states: Dict, pre_voltage: float, post_voltage: float, params: Dict
    ) -> float:
        """Return updated synapse state and current."""
        prefix = self._name
        sign = params[f"{prefix}_sign"]
        count = params[f"{prefix}_count"]
        strength = params[f"{prefix}_strength"]
        weight = sign * count * strength
        current = weight * relu(pre_voltage)
        return current
