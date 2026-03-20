# This file is part of Jaxley, a differentiable neuroscience simulator. Jaxley is
# licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from typing import Optional

import jax.numpy as jnp
from jax import Array

from jaxley.channels import Channel


class RateFlyvisExp(Channel):
    """Rate-based, unit-less, neuron model."""

    def __init__(self, name: Optional[str] = None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__("RateFlyvis")
        self.channel_params = {f"{self.name}_rest": 0.0}
        self.channel_states = {}
        self.current_name = f"{self.name}_rate"

    def update_states(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        """Voltages get pulled towards zero."""
        tau = params["capacitance"]
        rest = params[f"{self.name}_rest"]
        v_new = (voltage * jnp.exp(-delta_t / tau)) + (
            (rest) * (1 - jnp.exp(-delta_t / tau))
        )
        return {"v": v_new}

    def compute_current(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        return 0

    def init_state(
        self,
        states: dict[str, Array],
        params: dict[str, Array],
        voltage: Array,
        delta_t: float,
    ):
        return {}
