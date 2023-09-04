"""
This files provide utilities to help with data generation. 
"""

import numpy as np
import pandapower as pp
import torch

def perturb_topology():
    """
    Steps:
        1. load topology
        2. randomly remove lines (<- control: e.g. how many?)
        3. check connectivity
        4. if yes, return; else revert step 2 and retry. 
    """
    ...
    
    raise NotImplementedError