# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Dict, Optional

from jax import Array
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
        # self.node_params = {"capacitance": 10.0}

    def update_states(
        self,
        synapse_states: dict[str, Array],
        synapse_params: dict[str, Array],
        pre_voltage: Array,
        post_voltage: Array,
        pre_states: dict[str, Array],
        post_states: dict[str, Array],
        pre_params: dict[str, Array],
        post_params: dict[str, Array],
        delta_t: float,
    ) -> Dict:
        """Return updated synapse state and current."""
        return {}

    def compute_current(
        self,
        synapse_states: dict[str, Array],
        synapse_params: dict[str, Array],
        pre_voltage: Array,
        post_voltage: Array,
        pre_states: dict[str, Array],
        post_states: dict[str, Array],
        pre_params: dict[str, Array],
        post_params: dict[str, Array],
        delta_t: float,
    ) -> float:
        """Return updated synapse state and current."""
        prefix = self._name
        sign = synapse_params[f"{prefix}_sign"]
        count = synapse_params[f"{prefix}_count"]
        strength = synapse_params[f"{prefix}_strength"]
        weight = sign * count * strength
        current = -weight * relu(pre_voltage)
        return current
