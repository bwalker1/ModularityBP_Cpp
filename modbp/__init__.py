from __future__ import absolute_import

from .ModularityBP import ModularityBP
from .ModularityBP import calc_modularity
from .GenerateGraphs import RandomERGraph,RandomSBMGraph,MultilayerSBM,MultilayerGraph,generate_planted_partitions_dynamic_sbm,generate_planted_partitions_sbm

__version__ = "unknown" #default
from ._version import __version__