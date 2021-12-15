import os
from enum import Enum

class Dir(Enum):
    ROOT = os.path.expandvars('${SIMULATION_DIR}/bachelor_thesis/simulation_ws/')
    PLOT = ROOT + 'plots/'
    PARAMS = ROOT + 'parameters/'
    CHECKPOINT = ROOT + 'models/'
    RESULTS = ROOT + 'results/'
    TRACK = ROOT + 'track/'
