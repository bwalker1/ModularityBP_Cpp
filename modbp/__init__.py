from __future__ import absolute_import

from .ModularityBP import ModularityBP
from .ModularityBP import calc_modularity
from .InferenceBP import InferenceBP
from .GenerateGraphs import RandomERGraph,RandomSBMGraph,MultilayerSBM,MultilayerGraph,generate_planted_partitions_dynamic_sbm,generate_planted_partitions_sbm
from .bp import BP_Modularity #we want to eventually hid this from main import and restrict interactions to interface
from .bp import BP_Inference