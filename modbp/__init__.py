from __future__ import absolute_import

from .ModularityBP import ModularityBP
from .InferenceBP import InferenceBP
from .GenerateGraphs import RandomERGraph,RandomSBMGraph,MultilayerSBM,MultilayerGraph
from .bp import BP_Modularity #we want to eventually hid this from main import and restrict interactions to interface
from .bp import BP_Inference