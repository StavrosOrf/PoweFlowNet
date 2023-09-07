"""
This files provide utilities to help with data generation. 
"""
import warnings
import copy

import networkx as nx
import numpy as np
import pandapower as pp
import torch

def perturb_topology(net, num_lines_to_remove=0, num_lines_to_add=0):
    """
    Steps:
        1. load topology
        2. randomly remove lines (<- control: e.g. how many?)
        3. check connectivity
        4. if yes, return; else revert step 2 and retry. 
    """
    if num_lines_to_remove == 0 and num_lines_to_add == 0:
        return net
    
    max_attempts = 20
    # 1. load topology
    lines_indices = np.array(net.line.index)
    lines_from_bus = net.line['from_bus'].values # from 0, shape (num_lines,)
    lines_to_bus = net.line['to_bus'].values # shape (num_lines,)
    line_numbers = np.arange(start=0, stop=len(lines_from_bus))
    bus_numbers = net.bus.index.values # shape (num_buses,)
    
    rng = np.random.default_rng()
    # 2. remove lines
    num_disconnected_bus = 1
    num_attempts = 0
    while num_disconnected_bus > 0:
        net_perturbed = copy.deepcopy(net)
        if num_attempts == max_attempts:
            warnings.warn("Could not find a connected graph after {} attempts. Return original graph.".format(max_attempts))
            return net
        to_be_removed = rng.choice(line_numbers, size=num_lines_to_remove, replace=False)
        pp.drop_lines(net_perturbed, lines_indices[to_be_removed])
        num_disconnected_bus = len(pp.topology.unsupplied_buses(net_perturbed))
        num_attempts += 1
    
    # 3. add lines
    for _ in range(num_lines_to_add):
        from_bus, to_bus = rng.choice(bus_numbers, size=2, replace=False)
        copied_line = net.line.iloc[rng.choice(line_numbers, size=1, replace=False)]
        pp.create_line_from_parameters(
            net_perturbed, 
            from_bus, 
            to_bus, 
            copied_line['length_km'].item(), 
            copied_line['r_ohm_per_km'].item(), 
            copied_line['x_ohm_per_km'].item(), 
            copied_line['c_nf_per_km'].item(), 
            copied_line['max_i_ka'].item()
        )
        
    return net_perturbed