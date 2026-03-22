import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def draw_coupling_map(coupling_map, pos):
    plt.figure(figsize=(6,6))


    for i, node in enumerate(coupling_map.nodes):
        nx.draw_networkx_nodes(coupling_map, pos = pos, nodelist=[node], node_color=[coupling_map.nodes[node]['color']], node_size=10)
        # nx.draw_networkx_nodes(coupling_map, pos = pos, nodelist=[node], node_color='lightblue', node_size=10)

    # Draw edges and labels
    nx.draw_networkx_edges(coupling_map, pos = pos, edge_color='gray')
    # nx.draw_networkx_labels(coupling_map, pos, font_size=6, font_weight='bold', font_color='black')
    plt.show()
    return

def create_coupling_graph(D, L, nc):
    pos_graph = nx.DiGraph()
    computation_area = []
    node_index = 0
    pos = {}
    for i in range(D + 1):
        for j in range(D):
            for k in range(L):
                pos_graph.add_node(node_index)
                pos[node_index] = (i * (L + 1) - 1, j * (L + 1) + k)
                if k:
                    # for r in range(1, k + 1):
                    #     pos_graph.add_edge(node_index, node_index - r)
                    #     pos_graph.add_edge(node_index - r, node_index)
                    pos_graph.add_edge(node_index, node_index - 1)
                    pos_graph.add_edge(node_index - 1, node_index)  

                if k % int(np.ceil(L / (nc + 1))) == int(np.ceil(L / (nc + 1))) - 1:
                    computation_area.append(node_index)
                node_index += 1

    for i in range(D + 1):
        for j in range(D):
            for k in range(L):
                pos_graph.add_node(node_index)
                pos[node_index] = (j * (L + 1) + k, i * (L + 1) - 1)
                if k:
                    # for r in range(1, k + 1):
                    #     pos_graph.add_edge(node_index, node_index - r)
                    #     pos_graph.add_edge(node_index - r, node_index)
                    pos_graph.add_edge(node_index, node_index - 1)
                    pos_graph.add_edge(node_index - 1, node_index)                    
                # if k == L - 1:
                #     pos_graph.add_edge(node_index - L + 1, node_index)
                if i < D and k == 0:
                    # ru
                    pos_graph.add_edge(node_index, j * L * D + i * L)
                    pos_graph.add_edge(j * L * D + i * L, node_index)
                if i > 0 and k == 0:
                    pos_graph.add_edge(node_index, j * L * D + i * L - 1)
                    pos_graph.add_edge(j * L * D + i * L - 1, node_index)
                if i < D and k == L - 1:
                    pos_graph.add_edge(node_index, (j + 1) * L * D + i * L)
                    pos_graph.add_edge((j + 1) * L * D + i * L, node_index)
                if i > 0 and k == L - 1:
                    pos_graph.add_edge(node_index, (j + 1) * L * D + i * L - 1)
                    pos_graph.add_edge((j + 1) * L * D + i * L - 1, node_index)
                if k % int(np.ceil(L / (nc + 1))) == int(np.ceil(L / (nc + 1))) - 1:
                    computation_area.append(node_index)
                node_index += 1
                
    for node in pos_graph.nodes():
        pos_graph.nodes[node]['color'] = 'lightblue'
    return pos_graph, pos, computation_area

def create_linear_graph(D, n):
    pos_graph = nx.DiGraph()
    pos = {}
    computation_area = []
    for j in range(D):
        for i in range(n):
            computation_area.append(i + j * n)
            pos_graph.add_node(i + j * n)
            if i + j * n:
                pos_graph.add_edge(i + j * n, i + j * n - 1)
                pos_graph.add_edge(i + j * n - 1, i + j * n)
            pos[i + j * n] = (0, i + j * n + j)
    # pos_graph.add_edge(0, n - 1)
    # pos_graph.add_edge(n - 1, 0)
    for node in pos_graph.nodes():
        pos_graph.nodes[node]['color'] = 'lightblue'

    print(pos_graph.number_of_nodes())
    return pos_graph, pos, computation_area
# create_coupling_graph(3, 5)

def create_qec_coupling_graph(D, nm):
    pos_graph = nx.DiGraph()
    pos_normal_graph = nx.DiGraph()
    computation_area = []
    magic_state_area = []
    node_index = 0
    pos = {}

    nc = 3

    L1 = 4
    L2 = 2
    for i in range(D + 1):
        for j in range(D):
             
            for k in range(L1):
                pos_graph.add_node(node_index)
                pos[node_index] = (i * (L2 + 1) - 1, j * (L1 + 1) + k)
                if k:
                    # for r in range(1, k + 1):
                    #     pos_graph.add_edge(node_index, node_index - r)
                    #     pos_graph.add_edge(node_index - r, node_index)
                    pos_graph.add_edge(node_index, node_index - 1)
                    pos_graph.add_edge(node_index - 1, node_index)  

                if i != 0 and i != D and k == L1 // 2:
                        pos_normal_graph.add_node(node_index)
                        if j:
                            pos_normal_graph.add_edge(node_index, node_index - L1)
                            pos_normal_graph.add_edge(node_index - L1, node_index)
                        if i > 1:
                            pos_normal_graph.add_edge(node_index, node_index - L1 * D)
                            pos_normal_graph.add_edge(node_index - L1 * D, node_index)

                if k % int(np.ceil(L1 / (nc + 1))) == int(np.ceil(L1 / (nc + 1))) - 1:
                    computation_area.append(node_index)
                    
                if (i == 0 or i == D) and k == L1 // 2:
                    if j % int(np.ceil(D / (nm + 1))) == int(np.ceil(D / (nm + 1))) - 1:
                        magic_state_area.append(node_index)  

                node_index += 1


    for i in range(D + 1):
        for j in range(D):
            for k in range(L2):
                pos_graph.add_node(node_index)
                pos[node_index] = (j * (L2 + 1) + k, i * (L1 + 1) - 1)
                if k:
                    # for r in range(1, k + 1):
                    #     pos_graph.add_edge(node_index, node_index - r)
                    #     pos_graph.add_edge(node_index - r, node_index)
                    pos_graph.add_edge(node_index, node_index - 1)
                    pos_graph.add_edge(node_index - 1, node_index)                    
                # if k == L - 1:
                #     pos_graph.add_edge(node_index - L + 1, node_index)
                if i < D and k == 0:
                    # ru
                    pos_graph.add_edge(node_index, j * L1 * D + i * L1)
                    pos_graph.add_edge(j * L1 * D + i * L1, node_index)
                if i > 0 and k == 0:
                    pos_graph.add_edge(node_index, j * L1 * D + i * L1 - 1)
                    pos_graph.add_edge(j * L1 * D + i * L1 - 1, node_index)
                if i < D and k == L2 - 1:
                    pos_graph.add_edge(node_index, (j + 1) * L1 * D + i * L1)
                    pos_graph.add_edge((j + 1) * L1 * D + i * L1, node_index)
                if i > 0 and k == L2 - 1:
                    pos_graph.add_edge(node_index, (j + 1) * L1 * D + i * L1 - 1)
                    pos_graph.add_edge((j + 1) * L1 * D + i * L1 - 1, node_index)
                # if k % int(np.ceil(L / (nc + 1))) == int(np.ceil(L / (nc + 1))) - 1:
                #     computation_area.append(node_index)
                node_index += 1
                
    for node in pos_graph.nodes():
        pos_graph.nodes[node]['color'] = 'lightblue'

    for node in pos_normal_graph.nodes():
        pos_normal_graph.nodes[node]['color'] = 'lightblue'
    print(len(computation_area))
    return pos_graph, pos_normal_graph, pos, computation_area, magic_state_area


