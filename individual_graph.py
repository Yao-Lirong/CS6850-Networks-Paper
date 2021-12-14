from graph1 import *
import numpy as np
import matplotlib.pyplot as plt


NUM_GRAPHS = 10
NUM_UPDATES = 500
REPORT_FREQUENCY = 10

attractors = []
edge_num = []

for i in range(NUM_GRAPHS):
	print("Working on Graph %d" %(i))
	g = Graph(10, 3)
	this_attractors = []
	this_edge_num = []
	g.show_graph(str(i) + "_1.png")
	early_stop_point = NUM_UPDATES

	for j in range(NUM_UPDATES):
		if j % REPORT_FREQUENCY == 0:
			this_attractors.append(g.num_attractors(sample_size=100))
			this_edge_num.append(g.num_edges)
		if not g.update_graph():
			early_stop_point = j
			break

	attractors.append(this_attractors)
	print(this_edge_num)
	print("Stopped at %d updates with %d edges" %(early_stop_point, g.num_edges))
	g.show_graph(str(i) + "_2.png")

	plt.plot([i * REPORT_FREQUENCY for i in range(len(this_attractors))], this_attractors)
	plt.xlabel('Update Steps', fontsize=15)
	plt.ylabel('Approximate Number of Attractors', fontsize=15)
	
	# plt.annotate("#edges = " + str(g.num_edges), xy = (0.8 * NUM_UPDATES, 8))
	plt.savefig(str(i) + "_3.png")
	plt.clf()
	# print(this_edge_num)

