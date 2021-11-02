from enum import Enum

class Dir(Enum):
    ROOT = '/home/angj/priv/bachelor_thesis/simulation_ws/'
    PLOT = ROOT + 'plots/'
    PARAMS = ROOT + 'parameters/'
    CHECKPOINT = ROOT + 'models/'
    RESULTS = ROOT + 'results/'
