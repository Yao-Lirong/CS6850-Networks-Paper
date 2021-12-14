"""

Graph with only normal edges
color == True when this node is Black; color == False when this node is White
edge has value True when it is a normal edge; has value False when it has an inverse effect

"""

import random

from numpy import split, vectorize
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import reduce
from statistics import mean
import numpy as np

SAMPLE_TIMES = 10
SAMPLE_SIZE = 20
DEBUG = False
MAX_TRY = 100
MAX_ADD_FAILING = 50
MAX_DEL_FAILING = 50

add_failing = 0
del_failing = 0

def rand_bool():
	return True if random.randint(0,1) else False

probability_transformer = "exp"
def transform_prob(lst, tran = ""):
	if tran == "":
		return lst
	elif tran == "square":
		return np.square(lst).tolist()
	elif tran == "four":
		return np.square(np.square(lst)).tolist()
	elif tran == "exp":
		return np.exp(lst).tolist()

class Node:
	def __init__(self, id, init_color = None):
		if init_color == None: init_color = rand_bool()
		self.id = id
		self.out_edges = []
		self.in_edges = []
		self.current_color = init_color # is black if True; white if False
		self.next_color = False
	
	def update(self):
		self.current_color = self.next_color
		

def str_list_to_int(s):
	s = reduce(lambda a, x : a + str(x), s, "")
	s = int(s, 2)
	return s


class Graph:
	def __init__(self, num_nodes, k = 3):
		assert k < num_nodes
		self.k = k
		self.num_nodes = num_nodes
		self.nodes = [Node(i) for i in range(num_nodes)]
		self.indexes = [i for i in range(num_nodes)]
		self.num_edges = 0
		self.kn = self.k * self.num_nodes

		node_pairs = [(i, j) 
					  for i in range(self.num_nodes) 
					  for j in range(i, self.num_nodes)] # j in range(i,n) 就是带自环的
		edges = random.sample(node_pairs, self.k * self.num_nodes)
		for (i, j) in edges:
			self.add_edge(i,j)
			self.add_edge(j,i)
		self.G = nx.DiGraph()
		self.G.add_nodes_from(self.indexes)
		self.pos = nx.spring_layout(self.G)

	def add_edge(self, frm, to, normal = None):
		"""add an edge from node id [frm] to node id [to]
		   [normal] is True if this is a normal edge; False if it's an inverse edge
		   Keep the original edge if there is already one present
		"""
		if normal == None: normal = True
		u = self.nodes[frm]
		v = self.nodes[to]
		if list(filter(lambda edge: edge[0] == frm, u.out_edges)) == []:
			u.out_edges.append((to, normal))
			v.in_edges.append((frm, normal))
			self.num_edges += 1

	def del_edge(self, frm, to):
		u = self.nodes[frm]
		v = self.nodes[to]
		u_e = list(filter(lambda edge: edge[0] == to, u.out_edges))[0]
		u.out_edges.remove(u_e)
		v_e = list(filter(lambda edge: edge[0] == frm, v.in_edges))[0]
		v.in_edges.remove(v_e)
		self.num_edges -= 1

	def all_edges(self):
		edges = []
		for node in self.nodes:
			for edge in node.out_edges:
				edges.append((node.id, edge[0]))
		return edges

	def get_black_num(self, node):
		"""Returns black input received by Node object [node]
		"""
		black = 0
		for (frm, normal) in node.in_edges:
			black += 1 if ((self.nodes[frm].current_color and normal) or \
							((not self.nodes[frm].current_color) and (not normal))) \
						else 0
		return black

	def update_node_values(self):
		for node in self.nodes:
			black = self.get_black_num(node)
			node.next_color = node.current_color if 2 * black == len(node.in_edges) else \
				   			  True if 2 * black > len(node.in_edges) else False
		for node in self.nodes:
			node.update()
	
	# Returns False if we have consecutively failed adding MAX_ADD_FAILING times
	# and therefore wants to terminate any further update on this graph
	def update_graph_add(self):

		vs = []
		global add_failing
		num_try = 0

		while vs == []:
			# node with higher out degree is more likely to be chosen as the starting node
			u_weights = list(map(lambda x: len(x.out_edges), self.nodes))
			u_weights = transform_prob(u_weights, probability_transformer)
			u = random.choices(self.indexes, u_weights)[0]
			node_u = self.nodes[u]

			# node with higher in degree and is not already connected from u 
			# is more likely to be chosen as the end node.
			# We don't want to add self-loops in this update stage. 
			u_out = list(map(lambda x : x[0], node_u.out_edges))
			u_out.append(u)
			vs = [index for index in self.indexes if index not in u_out]

			num_try += 1
			if num_try > MAX_TRY:
				add_failing += 1
				if DEBUG: print("Addition Failed")
				return add_failing <= MAX_ADD_FAILING
		add_failing = 0

		v_weights = list(map(lambda x: len(x.in_edges), [self.nodes[v] for v in vs]))
		v_weights = transform_prob(v_weights, probability_transformer)
		v = random.choices(vs, v_weights)[0]

		self.add_edge(u, v)

		if DEBUG: print("Added a normal edge (%d, %d)" % (u, v))
		return True

	def update_graph_delete(self):
		# Try to get start and end points at most MAX_TRY times 
		# If tried more than that many times, delete fails
		num_try = 0

		# nodes with lower in degree is more likely to be chosen as the end node
		# but we don't delete edges to nodes that only have 1 or fewer in-edge
		v_weights = list(map(lambda x: len(x.in_edges), self.nodes))
		sorted_v_weights = sorted(v_weights)
		v_inverse_map = {i:j for i,j in zip(sorted_v_weights, reversed(sorted_v_weights))}
		v_weights = list(map(lambda x: v_inverse_map[x], v_weights))
		v_weights = transform_prob(v_weights, probability_transformer)
		
		v = random.choices(self.indexes, v_weights)[0]
		global del_failing
		while len(self.nodes[v].in_edges) <= 1:
			v = random.choices(self.indexes, v_weights)[0]
			num_try += 1
			if num_try > MAX_TRY:
				del_failing += 1
				if DEBUG: print("Deletion Failed")
				return del_failing <= MAX_DEL_FAILING

		# Start nodes are chosen from nodes connected to v
		# Nodes with lower out_degree are more likely to be chosen
		# but we don't choose the nodes that only have one out edge (which connects to v)
		v_in = list(map(lambda x : x[0], self.nodes[v].in_edges))
		u_weights = list(map(lambda x: len(x.out_edges), [self.nodes[u] for u in v_in]))
		sorted_u_weights = sorted(u_weights)
		u_inverse_map = {i:j for i,j in zip(sorted_u_weights, reversed(sorted_u_weights))}
		u_weights = list(map(lambda x: u_inverse_map[x], u_weights))
		u_weights = transform_prob(u_weights, probability_transformer)
		
		u = random.choices(v_in, u_weights)[0]
		while len(self.nodes[u].out_edges) <= 1:
			u = random.choices(v_in, u_weights)[0]
			num_try += 1
			if num_try > MAX_TRY:
				del_failing += 1
				if DEBUG: print("Deletion Failed")
				return del_failing <= MAX_DEL_FAILING
		del_failing = 0

		self.del_edge(u, v)
		if DEBUG: print("Removed edge (%d, %d)" % (u, v))

		return True


	def update_graph(self):
		add = random.choices([True, False], [.7, .3])[0] \
			  if self.num_edges <= self.kn \
			  else random.choices([True, False], [.3, .7])[0]
		if DEBUG:
			print("edges: %d, k*n: %d " %(self.num_edges, self.kn) + ("add" if add else "delete"))
		result = self.update_graph_add() if add else self.update_graph_delete()
		self.update_node_values()
		return result

	def node_colors(self):
		"""Returns a list of 0 or 1s representing White or Black at corresponding position
		"""
		colors = list(map(lambda x : 1 if x.current_color else 0, self.nodes))
		return colors

	def set_node_colors(self, colors):
		""" colors is an array of 0 or 1s. Set the corresponding node color to Black if 1, to White if 0
		"""
		for i in range(len(colors)):
			self.nodes[i].current_color = True if colors[i] == 1 else False

	def num_attractors_sample_once(self, sample_size = SAMPLE_SIZE):
		visited = set()
		attractors = 0
		for i in range(sample_size):
			# print("----- #%d -----" %(i) )

			state = [random.randint(0,1) for _ in range(self.num_nodes)]
			cur_bin_state = str_list_to_int(state)
			while cur_bin_state in visited:
				state = [random.randint(0,1) for _ in range(self.num_nodes)]
				cur_bin_state = str_list_to_int(state)
			
			prev_bin_state = 0

			while cur_bin_state not in visited:
				# print("  visiting " + bin(cur_bin_state))
				visited.add(cur_bin_state)
				self.set_node_colors(state)
				self.update_node_values()
				prev_bin_state = cur_bin_state
				state = self.node_colors()
				cur_bin_state = str_list_to_int(state)
			
			# Add 1 when states remain the same after one update 
			# or even the previous statement is not true but we exited the loop:
			# that means we found a loop pattern, we count the whole loop as one attractor. 
			if (cur_bin_state == prev_bin_state) or (attractors == 0):
				attractors += 1
			
		
		# for vis in visited: print(bin(vis), sep=" ")
		if DEBUG: print("After visiting %d states, we found %d attractors." %(len(visited), attractors))
		approx_attractors = attractors * pow(2,self.num_nodes) / len(visited)
		# return approx_attractors
		return attractors

	def num_attractors(self, sample_times = SAMPLE_TIMES, sample_size = SAMPLE_SIZE):
		res = [self.num_attractors_sample_once(sample_size) for _ in range(sample_times)]
		return mean(res)
	
	def show_graph(self, savepath = None):
		self.G.remove_edges_from(list(self.G.edges()))
		white_nodes = []
		black_nodes = []
		red_inverse_edges = []
		black_normal_edges = []

		for node in self.nodes:
			if node.current_color: black_nodes.append(node.id)
			else: white_nodes.append(node.id)
			for edge in node.out_edges:
				self.G.add_edge(node.id, edge[0])
				if edge[1]: black_normal_edges.append((node.id, edge[0]))
				else: red_inverse_edges.append((node.id, edge[0]))
		
		node_colors = [0 if node in black_nodes else 1
						for node in self.G.nodes()]

		
		nx.draw_networkx_nodes(self.G, self.pos, cmap=mpl.cm.cool, 
							node_color = node_colors, node_size = 500)
		nx.draw_networkx_labels(self.G, self.pos)
		nx.draw_networkx_edges(self.G, self.pos, edgelist=red_inverse_edges, edge_color='red', arrows=True)
		nx.draw_networkx_edges(self.G, self.pos, edgelist=black_normal_edges, arrows=True)
		
		if savepath != None: plt.savefig(savepath)
		else: plt.show()
		plt.clf()
