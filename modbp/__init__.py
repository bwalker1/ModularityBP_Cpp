from __future__ import absolute_import

from .ModularityBP import ModularityBP
from .ModularityBP import calc_modularity
from .ModularityBP import _get_avg_entropy

from .GenerateGraphs import RandomERGraph,RandomSBMGraph,MultilayerSBM,MultilayerGraph,MergedMultilayerGraph,generate_planted_partitions_dynamic_sbm,generate_planted_partitions_sbm,convertMultilayertoMergedMultilayer

__version__ = "unknown" #default
from ._version import __version__