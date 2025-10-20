from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from jaxley.modules import Module, Network


class SynapseConstants(NamedTuple):
    pre_syn_inds: list
    post_syn_ids: list
    synapse_names: list


class IntegrateHandler:
    def __init__(self, module: Module):
        self.channel_constants = self._build_channel_constants(module)
        self.channel_states = self._build_channel_states(module)
        self.synapse_constants: SynapseConstants = self._build_synapse_constants(module)
        self.synapse_states = self._build_synapse_states(module)

    def _build_synapse_states(self, module: Network):
        pass

    def _build_synapse_constants(self, module: Network) -> SynapseConstants:
        grouped_syns = module.edges.groupby("type", sort=False, group_keys=False)
        pre_syn_inds = grouped_syns["pre_index"].apply(list)
        post_syn_inds = grouped_syns["post_index"].apply(list)
        synapse_names = list(grouped_syns.indices.keys())
        return SynapseConstants(pre_syn_inds, post_syn_inds, synapse_names)

    def _build_channel_states(self, module: Module):
        pass

    def _build_channel_constants(self, module: Module):
        pass
