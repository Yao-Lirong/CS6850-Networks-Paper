from graph1 import *
import numpy as np
import matplotlib.pyplot as plt


NUM_GRAPHS = 10
NUM_UPDATES = 1000
REPORT_FREQUENCY = 10

attractors = []

for i in range(NUM_GRAPHS):
	print("Working on Graph %d" %(i))
	g = Graph(10, 3)
	this_attractors = []

	for j in range(NUM_UPDATES):
		if j % REPORT_FREQUENCY == 0:
			this_attractors.append(g.num_attractors(sample_size=200))
			# this_edge_num.append(g.num_edges)
		g.update_graph()

	attractors.append(this_attractors)

attractors = np.array(attractors)
mean_attractors = np.mean(attractors, axis=0)

plt.plot([i * REPORT_FREQUENCY for i in range(len(mean_attractors))], mean_attractors)
plt.xlabel('Update Steps', fontsize=15)
plt.ylabel('Approximate Number of Attractors', fontsize=15)
plt.show()
# plt.savefig("result.png")
