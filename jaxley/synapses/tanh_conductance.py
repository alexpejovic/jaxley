# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>


import jax.numpy as jnp

from jaxley.synapses.synapse import Synapse


class TanhConductanceSynapse(Synapse):
    """Compute synaptic current for a simple conductance synapse (no state).

    This is just the TanhRateSynapse with current affected by the postsynaptic
    conductance.
    """

    def __init__(self, name: str | None = None):
        super().__init__(name)
        prefix = self._name
        self.synapse_params = {
            f"{prefix}_gS": 1e-4,
            f"{prefix}_e_syn": 0.0,  # mV
            f"{prefix}_x_offset": -70.0,
            f"{prefix}_slope": 1.0,
        }
        self.synapse_states = {}

    def update_states(
        self,
        states: dict,
        delta_t: float,
        pre_voltage: float,
        post_voltage: float,
        params: dict,
    ) -> dict:
        """Return updated synapse state and current."""
        return {}

    def compute_current(
        self, states: dict, pre_voltage: float, post_voltage: float, params: dict
    ) -> float:
        """Return updated synapse state and current."""
        prefix = self._name
        tanh_pre_voltage = jnp.tanh(
            (pre_voltage - params[f"{prefix}_x_offset"]) * params[f"{prefix}_slope"]
        )
        current = (
            tanh_pre_voltage
            * params[f"{prefix}_gS"]
            * (post_voltage - params[f"{prefix}_e_syn"])
        )
        return current
