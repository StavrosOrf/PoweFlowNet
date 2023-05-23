#write an example code for a power flow using pypower
import pypower.api as pp
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

# load the case
# case = pp.case4gs()
# case = pp.case300()
case = pp.case9()

# print(case)
# change the values of generation for case
print(case['gen'])
pp.runpf(case)

# case = pp.case9()
case['gen'][0,1] = 50
case['gen'][1,1] = 50
# case['gen'][2,1] = 0.5
# case['gen'][3,1] = 0.5
print(case['gen'])
pp.runpf(case)

#Generate adjacency matrix of case buses' connections
# print(case['branch'])
print("\n\n\n\n======================================")
print(case)
#get the number of buses
n = case['bus'].shape[0]
print(n)

bus_names = case['bus'][:,0].tolist()


A = np.zeros((n,n))
for edge in case['branch']:

    edge_1 = bus_names.index(edge[0])
    edge_2 = bus_names.index(edge[1])

    A[edge_1,edge_2] = 1
    A[edge_2,edge_1] = 1



# for edge in case['branch']:
#     A[int(edge[0]-1),int(edge[1]-1)] = 1


print(A)

from pygsp import graphs, plotting

#Turn adjacency matrix into pygsp graph and visualize it
G = graphs.Graph(A)

G.set_coordinates()

#plot the graph such as a power network that has buses and branches

G.plot()
plt.show()






