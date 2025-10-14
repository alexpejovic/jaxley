from typing import NamedTuple

import numpy as np
from jax import Array

from jaxley.channels import Channel
from jaxley.utils.cell_utils import (
    params_to_pstate,
    query_channel_states_and_params,
)


class ChannelQueryItem(NamedTuple):
    channel: Channel
    params: dict
    states: dict
    indicies: np.ndarray


class IntegrationData(NamedTuple):
    channel_query_list: list[ChannelQueryItem]


# Arguments are the same as the ones passed into integrate and init_fn!!!
def query_integration_data(
    module, # TODO: Figure out how to add Module type here without circular importing
    params: list[dict[str, Array]] = [],
    all_states: dict | None = None,
    param_state: list[dict] | None = None,
    delta_t: float = 0.025,
) -> IntegrationData:
    """Queries data needed to quickly integrate the Module."""
    # ------- SECTION COPIED FROM integrate.init_fn() ---------------
    pstate = params_to_pstate(params, module.indices_set_by_trainables)
    if param_state is not None:
        pstate += param_state

    all_params = module.get_all_parameters(pstate)
    all_states = (
        module.get_all_states(pstate, all_params, delta_t)
        if all_states is None
        else all_states
    )
    # ------- SECTION COPIED FROM integrate.init_fn() ---------------

    channel_query_item_list = _query_channel_data(
        module, all_params, all_states, param_state
    )
    module.integration_data = IntegrationData(channel_query_item_list)


def _query_channel_data(
    module,
    params: list[dict[str, Array]] = [],
    all_states: dict | None = None,
    param_state: list[dict] | None = None,
    delta_t: float = 0.025,
) -> tuple[list, list]:
    channel_nodes = module.nodes
    channels = module.channels + module.pumps

    # Update states of the channels.
    indices = channel_nodes.index.to_numpy()
    channel_query_item_list = []

    for channel in channels:
        channel_param_names = list(channel.channel_params)
        channel_param_names += [
            "radius",
            "length",
            "axial_resistivity",
            "capacitance",
        ]
        channel_state_names = list(channel.channel_states)
        channel_state_names += module.membrane_current_names
        channel_indices = indices[channel_nodes[channel._name].astype(bool)]

        channel_params = query_channel_states_and_params(
            params, channel_param_names, channel_indices
        )
        channel_states = query_channel_states_and_params(
            all_states, channel_state_names, channel_indices
        )

        channel_query_item_list.append(
            ChannelQueryItem(channel, channel_params, channel_states, channel_indices)
        )

    return channel_query_item_list
