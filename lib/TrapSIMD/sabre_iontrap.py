from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from random import sample
from qiskit.transpiler import CouplingMap
# from coupling_graph import *
import  matplotlib.pyplot as plt
from .benchmarks.generate_benchmark import *
from qiskit.converters import circuit_to_dag
import networkx as nx
import numpy as np

def create_coupling_graph(D, L):
    pos_graph = nx.DiGraph()

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

                node_index += 1

    N = node_index
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
                node_index += 1
                
    for node in pos_graph.nodes():
        pos_graph.nodes[node]['color'] = 'lightblue'
    return pos_graph, pos

def draw_coupling_map(coupling_map, pos):
    plt.figure(figsize=(6,6))


    for i, node in enumerate(coupling_map.nodes):
        nx.draw_networkx_nodes(coupling_map, pos = pos, nodelist=[node], node_color=[coupling_map.nodes[node]['color']], node_size=10)
        # nx.draw_networkx_nodes(coupling_map, pos = pos, nodelist=[node], node_color='lightblue', node_size=10)

    # Draw edges and labels
    nx.draw_networkx_edges(coupling_map, pos = pos, edge_color='gray')
    nx.draw_networkx_labels(coupling_map, pos, font_size=6, font_weight='bold', font_color='black')
    plt.show()
    return

def sabre_compiler(D, L, CZ, n, alg, draw_flag):
    pg, pos = create_coupling_graph(D, CZ)
    pg_origin = pg.copy()
    pos_origin = pos.copy()
    if draw_flag:
        draw_coupling_map(pg, pos)
                    
    index = 0
    index_map = {}
    for pn in pg.nodes():
        index_map[index] = pn
        index += 1

    pg = nx.convert_node_labels_to_integers(pg)

    if alg == 'QFT':
        circ_origin = genQiskitQFT(n)
    elif alg == 'QAOA':
        circ_origin = genQiskitQAOA(n)
    elif alg == 'BV':
        circ_origin = genQiskitBV(n)
    elif alg == 'RCA':
        circ_origin = genQiskitRCA(n)
    elif alg == 'VQE':
        circ_origin = genQiskitVQE(n)

    def partition_graph_max_size_k(G, k):
        visited = set()
        partitions = []

        nodes = list(G.nodes())
        i = 0
        while i < len(nodes):
            group = []
            while len(group) < k and i < len(nodes):
                if nodes[i] not in visited:
                    group.append(nodes[i])
                    visited.add(nodes[i])
                i += 1
            if group:
                partitions.append(G.subgraph(group).copy())

        return partitions

    connect_graph = nx.Graph()
    for i in range(n):
        connect_graph.add_node(i)

    for data in circ_origin:
        if data.operation.name == 'cx':
            q0 = data.qubits[0]._index
            q1 = data.qubits[1]._index
            connect_graph.add_edge(q0, q1)

    subgraphs = partition_graph_max_size_k(connect_graph, CZ)
    m = len(subgraphs)

    l_circ = QuantumCircuit(m)

    for i in range(m):
        for j in range(i + 1, m):
            for ni in subgraphs[i].nodes():
                for nj in subgraphs[j].nodes():
                    if connect_graph.has_edge(ni, nj):
                        l_circ.cz(i, j)
                        l_circ.h(i)
                        l_circ.h(j)

    pg_abstract, pos_abstract = create_coupling_graph(D, 1)
    circ_abstract = transpile(
        l_circ,
        basis_gates=["cx", "id", "u2", "u1", "u3", 'swap'],
        coupling_map=CouplingMap([[x[0], x[1]] for x in list(pg_abstract.edges())]),
        optimization_level=2,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=0
    )

    layout = circ_abstract.layout.initial_layout.get_virtual_bits()
    initial_layout = {}
    for q in layout:
        reg_name = getattr(q._register, "name", None) or getattr(q._register, "_name", None)
        if reg_name == 'q':
            print(q._index)
            for i, nq in enumerate(subgraphs[q._index]):            
                initial_layout[circ_origin.qubits[nq]] = index_map[layout[q]] * CZ + i


    new_layout = {}
    circ = transpile(
        circ_origin,
        basis_gates=["cx", "id", "u2", "u1", "u3", 'swap'],
        coupling_map=CouplingMap([[x[0], x[1]] for x in list(pg_origin.edges())]),
        initial_layout=initial_layout,
        layout_method=None,
        optimization_level=2,
        routing_method="sabre",
        seed_transpiler=0
    )

    layout = circ.layout.initial_layout.get_virtual_bits()
    for q in layout:
        reg_name = getattr(q._register, "name", None) or getattr(q._register, "_name", None)
        if reg_name == 'q':
            new_layout[layout[q]] = q._index
            pg_origin.nodes[layout[q]]['color'] = 'blue'
        else:
            new_layout[layout[q]] = q._index + n
    if draw_flag:
        draw_coupling_map(pg_origin, pos)
    print(new_layout)

    def simulate(circ):
        num_shuttles = 0
        num_1dswap_gates = 0
        num_2dswap_gates = 0
        num_1dmove_gates = 0
        num_2dmove_gates = 0
        num_2q_gates = 0
        num_1q_gates = 0

        flag_2dswap = False
        flag_2dshift = False
        dir_2dswap = -1
        dir_2dshift = -1

        sq_time = 5
        tq_time = 25 + 58 * int(np.ceil(L / (CZ + 1)))
        onedswap_time = 200 + 58 * int(np.ceil(L / (CZ + 1)))
        twodswap_time = 400 + 100 + 58 * (int(np.ceil(L / (CZ + 1))) + L % int(np.ceil(L / (CZ + 1))))
        # twodswap_time = 200
        oned_move_time = 58 * int(np.ceil(L / (CZ + 1)))
        twod_move_time = 250 + 58 * (int(np.ceil(L / (CZ + 1))) + L % int(np.ceil(L / (CZ + 1))))

        dag = circuit_to_dag(circ)
        execution_time = 0
        clock_map = {}
        inter_trap_time = 0
        while True:

            front_layer = dag.front_layer()
            for node in front_layer:
                if node.op.num_qubits == 1:
                    if node not in clock_map and not flag_2dswap and not flag_2dshift:
                        num_1q_gates += 1
                        clock_map[node] = sq_time

            if dag.depth() == 0:
                break
            
            for node in front_layer:
                if node.op._name == 'cx':
                    if node not in clock_map and not flag_2dswap and not flag_2dshift:
                        num_2q_gates += 1
                        pos0 = node.qargs[0]._index 
                        pos1 = node.qargs[1]._index
                        if abs(pos1 - pos0) == 1:                  
                            clock_map[node] = tq_time
                        else:
                            flag_2dshift = True
                            clock_map[node] = 2 * twod_move_time
            for node in front_layer:
                if node.op._name == 'swap':
                    q0 = new_layout[node.qargs[0]._index]
                    q1 = new_layout[node.qargs[1]._index]


                    pos0 = node.qargs[0]._index 
                    pos1 = node.qargs[1]._index
                    
                    if q0 >= n and q1 >= n:
                        print("alarm!!!")
                        print(q0, q1, pos0, pos1)
                    if q0 >= n:
                        src = pos1
                        dst = pos0
                        if node not in clock_map and abs(pos[dst][0] - pos[src][0]) + abs(pos[dst][1] - pos[src][1]) == 1  and not flag_2dswap and not flag_2dshift:
                            num_1dmove_gates += 1
                            clock_map[node] = oned_move_time
                        else:
                            if node not in clock_map and (len(clock_map) == 0 or (flag_2dshift and dir_2dshift == (pos[dst][0] - pos[src][0], pos[dst][1] - pos[src][1]))):
                                num_2dmove_gates += 1
                                clock_map[node] = twod_move_time
                                flag_2dshift = True
                                dir_2dshift = (pos[dst][0] - pos[src][0], pos[dst][1] - pos[src][1])

                    elif q1 >= n:
                        # print('ancilla')
                        src = pos0
                        dst = pos1
                        if node not in clock_map and abs(pos[dst][0] - pos[src][0]) + abs(pos[dst][1] - pos[src][1]) == 1  and not flag_2dswap and not flag_2dshift:
                            num_1dmove_gates += 1
                            clock_map[node] = oned_move_time
                        else:
                            if node not in clock_map and (len(clock_map) == 0 or (flag_2dshift and dir_2dshift == (pos[dst][0] - pos[src][0], pos[dst][1] - pos[src][1]))) and not flag_2dswap:
                                num_2dmove_gates += 1
                                clock_map[node] = twod_move_time
                                flag_2dshift = True
                                dir_2dshift = (pos[dst][0] - pos[src][0], pos[dst][1] - pos[src][1])
                                print("2d move")
                    else:
                        if pos0 < D * (D - 1) * L:
                            p0 = pos1
                            p1 = pos0
                        else:
                            p0 = pos0
                            p1 = pos1

                        src = p0
                        dst = p1

                        if node not in clock_map and abs(pos[dst][0] - pos[src][0]) + abs(pos[dst][1] - pos[src][1]) == 1  and not flag_2dswap and not flag_2dshift:
                            num_1dswap_gates += 1
                            clock_map[node] = onedswap_time
                        else:
                            if node not in clock_map and (len(clock_map) == 0 or (flag_2dswap and dir_2dswap == (pos[dst][0] - pos[src][0], pos[dst][1] - pos[src][1]))) and not flag_2dshift:
                                num_2dswap_gates += 1
                                clock_map[node] = twodswap_time
                                flag_2dswap = True
                                dir_2dswap = (pos[dst][0] - pos[src][0], pos[dst][1] - pos[src][1])

            min_time = min(clock_map.values())
            execution_time += min_time
            for node in clock_map.copy():
                clock_map[node] -= min_time
                if min_time == twod_move_time or min_time == twodswap_time:
                    num_shuttles += 1
                if clock_map[node] == 0:
                    del clock_map[node]
                    dag.remove_op_node(node)
                    if node.op._name == 'swap':
                        q0 = new_layout[node.qargs[0]._index]
                        q1 = new_layout[node.qargs[1]._index]

                        pos0 = node.qargs[0]._index 
                        pos1 = node.qargs[1]._index
                        new_layout[pos0] = q1
                        new_layout[pos1] = q0                    
            if flag_2dshift:
                inter_trap_time += 500


            if flag_2dswap:
                inter_trap_time += 500

            if len(clock_map) == 0:
                # if flag_2dshift or flag_2dswap:
                #     num_shuttles += 1
                flag_2dswap = False
                flag_2dshift = False
                dir_2dswap = -1
                dir_2dshift = -1            
                    # new_layout[pos0] = q1
                    # new_layout[pos1] = q0



            if dag.depth() == 0 and len(clock_map) == 0:
                break
        
        num_1dmove_gates = (num_1dmove_gates + num_2q_gates) * int(np.ceil(L / (CZ + 1))) + num_2dmove_gates  * 2 * ((int(np.ceil(L / (CZ + 1))) + L % int(np.ceil(L / (CZ + 1)))) - 1) + num_2dswap_gates * 4 * ((int(np.ceil(L / (CZ + 1))) + L % int(np.ceil(L / (CZ + 1)))) - 1)


        tg_fidelity = 1 - 18.3 * 10 ** (-4)
        sg_fidelity = 1 - 0.25 * 10 ** (-4)
        transport_fidelity = 1 - 2 * 10 ** (-4)
        tg_fidelity = tg_fidelity
        transport_fidelity = transport_fidelity


        fidelity = tg_fidelity ** num_2q_gates * sg_fidelity ** num_1q_gates * transport_fidelity ** (num_2q_gates + num_1dmove_gates + num_2dmove_gates * 2 + num_2dswap_gates * 4)
        tg_overall_fidelity = tg_fidelity ** num_2q_gates
        sg_overall_fidelity = sg_fidelity ** num_1q_gates
        transport_overall_fidelity = transport_fidelity ** (num_2q_gates + num_1dmove_gates + num_2dmove_gates * 2 + num_2dswap_gates * 4)
        results = {
            "n_2q_gate": num_2q_gates,
            "n_1q_gate": num_1q_gates,
            "n_1dswap_gate": num_1dswap_gates,
            "n_2dswap_gate": num_2dswap_gates,
            "n_1dmove_gate": num_1dmove_gates,
            "n_2dmove_gate": num_2dmove_gates,
            "num_of_shuttles": num_shuttles,
            "inter_trap_time": inter_trap_time,
            "execution_time": execution_time,
            "tg_overall_fidelity": tg_overall_fidelity,
            "sg_overall_fidelity": sg_overall_fidelity,
            "transport_overall_fidelity": transport_overall_fidelity,
            "fidelity": fidelity
        }

        return results

    result = simulate(circ)
    execution_time = result['execution_time']
    return result