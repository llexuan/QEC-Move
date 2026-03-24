from .coupling_graph import *
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag
from collections import deque
from networkx.algorithms import community
from .benchmarks.generate_benchmark import *

from random import sample
import numpy as np
import math

# swap or move op
class OpNode:
    def __init__(self, op, p0, p1):
        self.op = op
        self.p0 = p0
        self.p1 = p1


Infinity = float('inf')

def grid_compiler(D, L, CZ, circ_origin, draw_flag):
    n = circ_origin.num_qubits
    pg, pos, computation_area = create_coupling_graph(D, L, CZ)
    node_list = []
    # pg, pos, computation_area = create_linear_graph(D, L)
    # print(len(list(nx.connected_components(nx.Graph(pg)))))
    # print(computation_area)

    circ_sabre = transpile(
        circ_origin,
        basis_gates=["cx", "id", "u2", "u1", "u3", 'swap'],
        coupling_map=CouplingMap([[x[0], x[1]] for x in list(pg.edges())]),
        optimization_level=2,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=0
    )

    target_compute_pos = {}
    qubit_to_pos = {}
    pos_to_qubit = {}
    new_layout = {}
    layout = circ_sabre.layout.initial_layout.get_virtual_bits()
    for q in layout:
        reg_name = getattr(q._register, "name", None) or getattr(q._register, "_name", None)
        if reg_name == 'q':
            new_layout[layout[q]] = q._index
            qubit_to_pos[q._index] = layout[q]
            pos_to_qubit[layout[q]] = q._index
            pg.nodes[layout[q]]['color'] = 'blue'
        else:
            new_layout[layout[q]] = q._index + n
            qubit_to_pos[q._index + n] = layout[q]
            pos_to_qubit[layout[q]] = q._index + n

    for p in computation_area:
        pg.nodes[p]['color'] = 'red'

    # print(qubit_to_pos)
    if draw_flag:
        draw_coupling_map(pg, pos)

    # print(pos)
    circ = transpile(
        circ_origin,
        basis_gates=["cx", "id", "u2", "u1", "u3"], 
        optimization_level = 2
    )

    def distance(pos0, pos1):
        return abs(pos0[0] - pos1[0]) + abs(pos0[1] - pos1[1])

    def coupling_distance(p0, p1):
        path = nx.shortest_path(pg, source=p0, target=p1)
        # for p in path:
        if CZ == L:
            return len(path)
        shortest_path = Infinity
        target_p = -1
        if (qubit_to_pos[p0], qubit_to_pos[p1]) not in target_compute_pos:

            for p in computation_area:
                l0 = len(nx.shortest_path(pg, source=p0, target=p))
                if l0 >= shortest_path:
                    continue
                l1 = len(nx.shortest_path(pg, source=p1, target=p))
                # if l0 == 0:
                #     l0 = -1
                # if l1 == 0:
                #     l1 = -1
                if l0 + l1 < shortest_path:
                    shortest_path = l0 + l1
                    target_p = p
        

            target_compute_pos[(qubit_to_pos[p0], qubit_to_pos[p1])] = target_p
        else:
            shortest_path = len(nx.shortest_path(pg, source=p0, target=target_compute_pos[(qubit_to_pos[p0], qubit_to_pos[p1])])) + len(nx.shortest_path(pg, source=p1, target=target_compute_pos[(qubit_to_pos[p0], qubit_to_pos[p1])]))


        # print("shortest path", shortest_path)
        return shortest_path + 0.5 * len(path)


    def virtual_execute(op, pos_to_qubit_next, qubit_to_pos_next, real_pos_to_qubit):
        pos_to_qubit_result = pos_to_qubit_next.copy()
        qubit_to_pos_result = qubit_to_pos_next.copy()
        
        p0 = op.p0
        p1 = op.p1

        q0 = real_pos_to_qubit[p0]
        q1 = real_pos_to_qubit[p1]

        # if q0 < n:
        #     if real_pos_to_qubit[p0] != pos_to_qubit_next[p0]:
        #         # print("difference alarm!!!")
        #         print(p0, q0, pos_to_qubit_next[p0])

        # if q1 < n:
        #     if real_pos_to_qubit[p1] != pos_to_qubit_next[p1]:
        #         # print("difference alarm!!!")
        #         print(p1, q1, pos_to_qubit_next[p1])

        pos_to_qubit_result[p0] = q1
        pos_to_qubit_result[p1] = q0

        qubit_to_pos_result[q0] = p1
        qubit_to_pos_result[q1] = p0

        return pos_to_qubit_result, qubit_to_pos_result

    def real_execute(op, pos_to_qubit_next, qubit_to_pos_next, clock_map, num_1dmove_gates, num_1dswap_gates, move_target_pos, move_src_pos, swap_nodes, add_into_clock_index, real_pos_to_qubit):
        p0 = op.p0
        p1 = op.p1

        q0 = real_pos_to_qubit[p0]
        q1 = real_pos_to_qubit[p1]

        # if q0 < n:
        #     if real_pos_to_qubit[p0] != pos_to_qubit_next[p0]:
        #         # print("difference alarm!!!")
        #         print(p0, q0, pos_to_qubit_next[p0])

        # if q1 < n:
        #     if real_pos_to_qubit[p1] != pos_to_qubit_next[p1]:
        #         # print("difference alarm!!!")
        #         print(p1, q1, pos_to_qubit_next[p1])


        pos_to_qubit_next[p0] = q1
        pos_to_qubit_next[p1] = q0



        qubit_to_pos_next[q0] = p1
        qubit_to_pos_next[q1] = p0



        if op.op == 'move':
            clock_map[op] = move_1d_time
            num_1dmove_gates += 1
            move_target_pos.append(pos[p1])
            move_src_pos.append(pos[p0])
        elif op.op == 'swap':
            clock_map[op] = swap_1d_time
            num_1dswap_gates += 1
            swap_nodes.append(q0)
            swap_nodes.append(q1)

        add_into_clock_index += 1

        return pos_to_qubit_next, qubit_to_pos_next, clock_map, num_1dmove_gates, num_1dswap_gates, move_target_pos, move_src_pos, swap_nodes, add_into_clock_index


    def create_extended_successor_set(shuttle, dag):   
        E = list()
        for gate in shuttle:
            # print(dag.bfs_successors(gate))
            for gate_successor in dag.bfs_successors(gate):
                # print('successor op', gate_successor)
                # print(gate_successor[0].op._name)
                if len(E) <= 20 and gate_successor[0].op.num_qubits == 2:
                    E.append(gate_successor[0])
                    # E = []
        return E

    def sabre_score(dag, shuttle_node_group, qubit_to_pos, pos_to_qubit):
        score_cur = 0
        for s in shuttle_node_group:
            q0 = s.qargs[0]._index
            q1 = s.qargs[1]._index

            p0 = qubit_to_pos[q0]
            p1 = qubit_to_pos[q1]

            score_cur += coupling_distance(p0, p1)
        
        # E = create_extended_successor_set(shuttle_node_group, dag)
        # score_next  = 0
        # for s in E:
        #     q0 = s.qargs[0]._index
        #     q1 = s.qargs[1]._index

        #     p0 = qubit_to_pos[q0]
        #     p1 = qubit_to_pos[q1]

        #     score_next += coupling_distance(p0, p1)

        score = 0
        if len(shuttle_node_group):
            score += score_cur / len(shuttle_node_group)
        # if len(E):
        #     score += 0.5 * score_next / len(E) 
        return score


        

    num_1dswap_gates = 0
    num_2dswap_gates = 0
    num_1dmove_gates = 0
    num_2dmove_gates = 0
    num_2q_gates = 0
    num_1q_gates = 0
    congest_count = 0
    schedule_count = 0

    sq_time = 5
    tq_time = 25
    swap_1d_time = 200
    swap_2d_time = 500
    move_1d_time = 58
    move_2d_time = 250
    flag_2d_move = False
    flag_2d_swap = False

    dir_2dswap = -1
    dir_2dshift = -1

    num_shuttles = 0

    dag = circuit_to_dag(circ)

    execution_time = 0
    inter_trap_time = 0
    clock_map = {}
    shuttle_group = deque([])
    shuttle_node_group = deque([])
    move_target_pos = []
    move_src_pos = []
    swap_nodes = []
    move_2d_directions = []
    swap_2d_directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    swap_back = {}
    move_back = {}
    # swap_data_flow = {}
    pos_to_qubit_next = pos_to_qubit.copy()
    qubit_to_pos_next = qubit_to_pos.copy()

    # up down left right
    for i in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for j in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            move_2d_directions.append((i, j))


    while dag.depth():
        schedule_count += 1
        operations = []
        front_layer = dag.front_layer()


        if not flag_2d_move and not flag_2d_swap:
            add_into_clock_index = 0
            for node in front_layer:
                if node.op.num_qubits == 1:
                    if node not in clock_map:
                        num_1q_gates += 1
                        clock_map[node] = sq_time
                        add_into_clock_index += 1
                elif node.op._name == 'cx':
                    if node not in shuttle_node_group:
                        shuttle_node_group.append(node)
                    q0 = node.qargs[0]._index
                    q1 = node.qargs[1]._index
                    p0 = qubit_to_pos[q0]
                    p1 = qubit_to_pos[q1]
                    if node not in clock_map:
                        if p1 in pg.neighbors(p0) and (p0 in computation_area or p1 in computation_area):
                        # if abs(p1 - p0) == 1
                        # if p1 in pg.neighbors(p0):
                            if (q0, q1) in shuttle_group:
                                shuttle_group.remove((q0, q1))

                            elif (q1, q0) in shuttle_group:
                                shuttle_group.remove((q1, q0))

                            num_2q_gates += 1
                            clock_map[node] = tq_time
                            add_into_clock_index += 1

                            if q0 in swap_back:
                                del swap_back[q0]
                                
                            
                            if q1 in swap_back:
                                del swap_back[q1]
                            
                            if q0 in move_back:
                                del move_back[q0]
                                
                            
                            if q1 in move_back:
                                del move_back[q1]

                        else:
                            if (q0, q1) not in shuttle_group:
                                shuttle_group.append((q0, q1))

            # schedule shuttle   

            # for s in shuttle_group:
            #     print(s[0], s[1], qubit_to_pos[s[0]], qubit_to_pos[s[1]])
            shuttle_qubits = []
            executed_nodes = []
            
            # print(shuttle_group)
            # for s in shuttle_group:
            #     q0 = s[0]
            #     q1 = s[1]
            #     shuttle_qubits.append(q0)
            #     shuttle_qubits.append(q1)       



            while True:
                operations_candidate_list = []
                for s in shuttle_group:           
                    q0 = s[0]
                    q1 = s[1] 
                    p0 = qubit_to_pos[q0]
                    p1 = qubit_to_pos[q1]

                    pos0 = pos[p0]
                    pos1 = pos[p1]

                    if q0 not in executed_nodes:
                        # find whether there is 1d move
                        move_1d_flag_q0 = False
                        if q0 not in swap_nodes and pos0 not in move_src_pos:
                            for neigh in pg.neighbors(p0):
                                qn = pos_to_qubit[neigh]
                                posn = pos[neigh]
                                if qn >= n or posn in move_src_pos:  
                                    if coupling_distance(neigh, qubit_to_pos_next[q1]) < coupling_distance(p0, qubit_to_pos_next[q1]):
                                        if (qn not in move_back or move_back[qn] != p0) and (q0 not in move_back or move_back[q0] != neigh):
                                            if distance(posn, pos0) == 1:
                                                if posn not in move_target_pos:
                                                    node = OpNode('move', p0, neigh)
                                                    # print("1dmove", p0, neigh, q0, qn)
                                                    # clock_map[node] = move_1d_time
                                                    # num_1dmove_gates += 1
                                                    # move_target_pos.append(posn)
                                                    # move_src_pos.append(pos0)
                                                    # move_1d_flag_q0 = True
                                                    # add_into_clock_index += 1
                                                    # break
                                                    operations_candidate_list.append(node)
                    

                    if q1 not in executed_nodes:
                        move_1d_flag_q1 = False
                        if q1 not in swap_nodes and pos1 not in move_src_pos:
                            for neigh in pg.neighbors(p1):
                                qn = pos_to_qubit[neigh]
                                posn = pos[neigh]
                                if qn >= n or posn in move_src_pos:
                                    if coupling_distance(neigh, qubit_to_pos_next[q0]) < coupling_distance(qubit_to_pos_next[q0], p1):
                                        if (qn not in move_back or move_back[qn] != p1) and (q1 not in move_back or move_back[q1] != neigh):
                                            if distance(posn, pos1) == 1:
                                                if posn not in move_target_pos:
                                                    node = OpNode('move', p1, neigh)
                                                    # print("1dmove", p1, neigh, q1, qn)
                                                    # clock_map[node] = move_1d_time
                                                    # num_1dmove_gates += 1
                                                    # move_target_pos.append(posn)
                                                    # move_src_pos.append(pos1)
                                                    # move_1d_flag_q1 = True
                                                    # add_into_clock_index += 1
                                                    # break
                                                    operations_candidate_list.append(node)

                    # find whether there is 1d swap
                    if q0 not in executed_nodes:
                        swap_1d_flag_q0 = False
                        if pos0 not in move_src_pos and q0 not in swap_nodes:
                            for neigh in pg.neighbors(p0):
                                qn = pos_to_qubit[neigh]
                                posn = pos[neigh]
                                if qn < n and qn not in swap_nodes and posn not in move_src_pos:
                                    if coupling_distance(neigh, qubit_to_pos_next[q1]) < coupling_distance(p0, qubit_to_pos_next[q1]):
                                        if distance(posn, pos0) == 1 and (qn not in swap_back or p0 not in swap_back[qn]) and (q0 not in swap_back or neigh not in swap_back[q0]):
                                            node = OpNode('swap', p0, neigh)
                                            # print(move_src_pos)
                                            # print("1dswap", p0, neigh, q0, qn, pos0, posn)
                                            # clock_map[node] = swap_1d_time
                                            # num_1dswap_gates += 1
                                            # swap_nodes.append(q0)
                                            # swap_nodes.append(qn)
                                            # print(swap_nodes)
                                            # # print("swap nodes append1", q0, qn)
                                            # swap_1d_flag_q0 = True
                                            # add_into_clock_index += 1
                                            # break
                                            operations_candidate_list.append(node)
                                        
                    if q1 not in executed_nodes:
                        swap_1d_flag_q1 = False
                        if pos1 not in move_src_pos and q1 not in swap_nodes:
                            for neigh in pg.neighbors(p1):
                                qn = pos_to_qubit[neigh]
                                posn = pos[neigh]
                                if qn < n and qn not in swap_nodes and posn not in move_src_pos:
                                    if coupling_distance(neigh, qubit_to_pos_next[q0]) < coupling_distance(qubit_to_pos_next[q0], p1):
                                        if distance(posn, pos1) == 1 and (qn not in swap_back or p1 not in swap_back[qn]) and (q1 not in swap_back or neigh not in swap_back[q1]):
                                            node = OpNode('swap', p1, neigh)
                                            # print(move_src_pos)
                                            # print("1dswap", p1, neigh, q1, qn, pos0, posn)
                                            # clock_map[node] = swap_1d_time
                                            # num_1dswap_gates += 1
                                            # swap_nodes.append(q1)
                                            # swap_nodes.append(qn)
                                            # print(swap_nodes)
                                            # # print("swap nodes append2", q1, qn)
                                            # swap_1d_flag_q1 = True
                                            # add_into_clock_index += 1
                                            # break
                                            
                                            operations_candidate_list.append(node)
                
                if len(operations_candidate_list) == 0:
                    break
                
                min_score = sabre_score(dag, shuttle_node_group, qubit_to_pos_next, pos_to_qubit_next)
                # min_score = Infinity
                min_op = -1
                # print("virtual")
                
                # if pos_to_qubit_next != pos_to_qubit:
                #     print("strange1")
                for op in operations_candidate_list:
                    
                    pos_to_qubit_temp, qubit_to_pos_temp = virtual_execute(op, pos_to_qubit_next, qubit_to_pos_next, pos_to_qubit)
                    score = sabre_score(dag, shuttle_node_group, qubit_to_pos_temp, pos_to_qubit_temp)
                    # print("min score", min_score)
                    # print("score", score)
                    if score <= min_score:
                        min_op = op
                        min_score = score
                    elif score == min_score:
                        # print(op, min_op)
                        if op.op == 'move' and (min_op != -1 and min_op.op == 'swap'):
                            min_op = op
                            min_score = score                    
                # if pos_to_qubit_next != pos_to_qubit:
                #     print("strange2")    
                # print("real")
                # break_flag = False
                    
                if min_op != -1:
                    pos_to_qubit_next, qubit_to_pos_next, clock_map, num_1dmove_gates, num_1dswap_gates, move_target_pos, move_src_pos, swap_nodes, add_into_clock_index = real_execute(min_op, pos_to_qubit_next, qubit_to_pos_next, clock_map, num_1dmove_gates, num_1dswap_gates, move_target_pos, move_src_pos, swap_nodes, add_into_clock_index, pos_to_qubit)
                    executed_nodes.append(pos_to_qubit[min_op.p0])
                    executed_nodes.append(pos_to_qubit[min_op.p1])
                    # print("1d", min_op.op, pos_to_qubit[min_op.p0], pos_to_qubit[min_op.p1], min_op.p0, min_op.p1)
                    # if break_flag:
                    #     break
                else:
                    break

        # consider 2d shuttling

        # if add_into_clock_index == 0 and len(clock_map) == 0:
        if True:
            # print("consider 2d shuttling")
            # consider 2d move
            move_2d_dir_groups = {}
            for move_dir_2d in move_2d_directions:
                move_2d_dir_groups[move_dir_2d] = []

                for s in shuttle_group:
                    
                    q0 = s[0]
                    q1 = s[1] 
                    p0 = qubit_to_pos[q0]
                    p1 = qubit_to_pos[q1]

                    pos0 = pos[p0]
                    pos1 = pos[p1]
                    
                    # print(q0, q1, p0, p1)
                    if pos0 not in move_src_pos and q0 not in swap_nodes:
                        for neigh in pg.neighbors(p0):
                            qn = pos_to_qubit[neigh]
                            posn = pos[neigh]
                            if qn >= n:  
                                if coupling_distance(neigh, qubit_to_pos_next[q1]) < coupling_distance(p0, qubit_to_pos_next[q1]):
                                    if (qn not in move_back or move_back[qn] != p0) and (q0 not in move_back or move_back[q0] != neigh):
                                        # print("move", "distance satisfy")
                                        # print((dir0, dir1), move_dir_2d, q1, qn, p1, neigh)
                                        if p0 < (D + 1) * D * L:
                                            dir0 = (posn[0] - pos0[0], 0)
                                            dir1 = (0, posn[1] - pos0[1])
                                        else:
                                            dir0 = (0, posn[1] - pos0[1])
                                            dir1 = (posn[0] - pos0[0], 0)
                                            
                                        if (dir0, dir1) == move_dir_2d:
                                            node = OpNode('move', p0, neigh)
                                            move_2d_dir_groups[move_dir_2d].append(node)
                                            break
                        

                        if p0 < (D + 1) * D * L:
                            # at end of one trap
                            if p0 % D == L - 1:
                                if (p0 % (D * L)) // L != D - 1:
                                    pn = p0 + 1
                                    qn = pos_to_qubit[pn]
                                    posn = pos[pn]
                                    if qn >= n:
                                        if coupling_distance(pn, qubit_to_pos_next[q1]) < coupling_distance(p0, qubit_to_pos_next[q1]):
                                            if (qn not in move_back or move_back[qn] != p0) and (q0 not in move_back or move_back[q0] != pn):
                                                # print("move", "distance satisfy")
                                                if posn[1] - pos0[1] == 2 and ((0, 1), (0, 1)) == move_dir_2d:
                                                    node = OpNode('move', p0, pn)
                                                    move_2d_dir_groups[move_dir_2d].append(node)
                                                # else:
                                                #     print("shift alarm !!!")
                            elif p0 % D == 0:
                                if (p0 % (D * L)) // L != 0:
                                    pn = p0 - 1
                                    qn = pos_to_qubit[pn]
                                    posn = pos[pn]
                                    if qn >= n:
                                        if coupling_distance(pn, qubit_to_pos_next[q1]) < coupling_distance(p0, qubit_to_pos_next[q1]):
                                            if (qn not in move_back or move_back[qn] != p0) and (q0 not in move_back or move_back[q0] != pn):
                                                # print("move", "distance satisfy")
                                                if posn[1] - pos0[1] == -2 and ((0, -1), (0, -1)) == move_dir_2d:
                                                    node = OpNode('move', p0, pn)
                                                    move_2d_dir_groups[move_dir_2d].append(node)
                                                # else:
                                                #     print("shift alarm !!!")
                        else:
                            # at end of one trap
                            p_loc = p0 -(D + 1) * D * L
                            if p_loc % D == L - 1:
                                if (p_loc % (D * L)) // L != D - 1:
                                    pn = p0 + 1
                                    qn = pos_to_qubit[pn]
                                    posn = pos[pn]
                                    if qn >= n:
                                        if coupling_distance(pn, qubit_to_pos_next[q1]) < coupling_distance(p0, qubit_to_pos_next[q1]):
                                            if (qn not in move_back or move_back[qn] != p0) and (q0 not in move_back or move_back[q0] != pn):
                                                # print("move", "distance satisfy")
                                                if posn[0] - pos0[0] == 2 and ((1, 0), (1, 0)) == move_dir_2d:
                                                    node = OpNode('move', p0, pn)
                                                    move_2d_dir_groups[move_dir_2d].append(node)
                                                # else:
                                                #     print("shift alarm !!!")
                            elif p_loc % D == 0:
                                if (p_loc % (D * L)) // L != 0:
                                    pn = p0 - 1
                                    qn = pos_to_qubit[pn]
                                    posn = pos[pn]
                                    if qn >= n:
                                        if coupling_distance(pn, qubit_to_pos_next[q1]) < coupling_distance(p0, qubit_to_pos_next[q1]):
                                            if (qn not in move_back or move_back[qn] != p0) and (q0 not in move_back or move_back[q0] != pn):
                                                # print("move", "distance satisfy")
                                                if posn[0] - pos0[0] == -2 and ((-1, 0), (-1, 0)) == move_dir_2d:
                                                    node = OpNode('move', p0, pn)
                                                    move_2d_dir_groups[move_dir_2d].append(node)
                                                # else:
                                                #     print("shift alarm !!!")    
                                                                
                    if pos1 not in move_src_pos and q1 not in swap_nodes:
                        for neigh in pg.neighbors(p1):
                            qn = pos_to_qubit[neigh]
                            posn = pos[neigh]
                            if qn >= n:  
                                # print(q1, p1, qn, neigh)
                                # print(qubit_to_pos_next[q0], coupling_distance(neigh, qubit_to_pos_next[q0]), coupling_distance(qubit_to_pos_next[q0], p1))
                                if coupling_distance(neigh, qubit_to_pos_next[q0]) < coupling_distance(qubit_to_pos_next[q0], p1):
                                    if (qn not in move_back or move_back[qn] != p1) and (q1 not in move_back or move_back[q1] != neigh):
                                        # print("distance satisfy")
                                        # print("move", (dir0, dir1), move_dir_2d, q1, qn, p1, neigh)
                                        # print(posn, pos1)
                                        if p1 < (D + 1) * D * L:
                                            dir0 = (posn[0] - pos1[0], 0)
                                            dir1 = (0, posn[1] - pos1[1])
                                        else:
                                            dir0 = (0, posn[1] - pos1[1])
                                            dir1 = (posn[0] - pos1[0], 0)
                                        # print((dir0, dir1), move_dir_2d)
                                        if (dir0, dir1) == move_dir_2d:
                                            node = OpNode('move', p1, neigh)
                                            move_2d_dir_groups[move_dir_2d].append(node)
                                            break

                        if p1 < (D + 1) * D * L:
                            # at end of one trap
                            if p1 % D == L - 1:
                                if (p1 % (D * L)) // L != D - 1:
                                    pn = p1 + 1
                                    qn = pos_to_qubit[pn]
                                    posn = pos[pn]
                                    if qn >= n:
                                        if coupling_distance(pn, qubit_to_pos_next[q0]) < coupling_distance(qubit_to_pos_next[q0], p1):
                                            if (qn not in move_back or move_back[qn] != p1) and (q1 not in move_back or move_back[q1] != pn):
                                                # print("move", "distance satisfy")
                                                if posn[1] - pos1[1] == 2 and ((0, 1), (0, 1)) == move_dir_2d:
                                                    node = OpNode('move', p1, pn)
                                                    move_2d_dir_groups[move_dir_2d].append(node)
                                                # else:
                                                #     print("shift alarm !!!")
                            elif p1 % D == 0:
                                if (p1 % (D * L)) // L != 0:
                                    pn = p1 - 1
                                    qn = pos_to_qubit[pn]
                                    posn = pos[pn]
                                    if qn >= n:
                                        if coupling_distance(pn, qubit_to_pos_next[q0]) < coupling_distance(qubit_to_pos_next[q0], p1):
                                            if (qn not in move_back or move_back[qn] != p1) and (q1 not in move_back or move_back[q1] != pn):
                                                # print("move", "distance satisfy")
                                                if posn[1] - pos1[1] == -2 and ((0, -1), (0, -1)) == move_dir_2d:
                                                    node = OpNode('move', p1, pn)
                                                    move_2d_dir_groups[move_dir_2d].append(node)
                                                # else:
                                                #     print("shift alarm !!!")
                        else:
                            # at end of one trap
                            p_loc = p1 -(D + 1) * D * L
                            if p_loc % D == L - 1:
                                if (p_loc % (D * L)) // L != D - 1:
                                    pn = p1 + 1
                                    qn = pos_to_qubit[pn]
                                    posn = pos[pn]
                                    if qn >= n:
                                        if coupling_distance(pn, qubit_to_pos_next[q0]) < coupling_distance(qubit_to_pos_next[q0], p1):
                                            if (qn not in move_back or move_back[qn] != p1) and (q1 not in move_back or move_back[q1] != pn):
                                                # print("move", "distance satisfy")
                                                if posn[0] - pos1[0] == 2 and ((1, 0), (1, 0)) == move_dir_2d:
                                                    node = OpNode('move', p1, pn)
                                                    move_2d_dir_groups[move_dir_2d].append(node)
                                                # else:
                                                #     print("shift alarm !!!")
                            elif p_loc % D == 0:
                                if (p_loc % (D * L)) // L != 0:
                                    pn = p1 - 1
                                    qn = pos_to_qubit[pn]
                                    posn = pos[pn]
                                    if qn >= n:
                                        if coupling_distance(pn, qubit_to_pos_next[q0]) < coupling_distance(qubit_to_pos_next[q0], p1):
                                            if (qn not in move_back or move_back[qn] != p1) and (q1 not in move_back or move_back[q1] != pn):
                                                # print("move", "distance satisfy")
                                                if posn[0] - pos1[0] == -2 and ((-1, 0), (-1, 0)) == move_dir_2d:
                                                    node = OpNode('move', p1, pn)
                                                    move_2d_dir_groups[move_dir_2d].append(node)
                                                # else:
                                                #     print("shift alarm !!!")     

            # max_group_size = 0
            # group_dir = -1
            # for m in move_2d_dir_groups:
            #     if len(move_2d_dir_groups[m]) > max_group_size:
            #         group_dir = m
            #         max_group_size = len(move_2d_dir_groups[m])

            min_score = sabre_score(dag, shuttle_node_group, qubit_to_pos_next, pos_to_qubit_next)
            group_dir = -1
            # print(move_2d_dir_groups)
            for m in move_2d_dir_groups:
                pos_to_qubit_temp, qubit_to_pos_temp  = pos_to_qubit_next.copy(), qubit_to_pos_next.copy()
                for op in move_2d_dir_groups[m]:
                    pos_to_qubit_temp, qubit_to_pos_temp = virtual_execute(op, pos_to_qubit_temp, qubit_to_pos_temp, pos_to_qubit)
                
                score = sabre_score(dag, shuttle_node_group, qubit_to_pos_temp, pos_to_qubit_temp)
                if score <= min_score and len(move_2d_dir_groups[m]):
                    group_dir = m
                    min_score = score

            if add_into_clock_index == 0 and len(clock_map) == 0 or sabre_score(dag, shuttle_node_group, qubit_to_pos_next, pos_to_qubit_next) - min_score >= 5:
                group_dir = group_dir
            else:
                group_dir = -1
            # min_score = sabre_score(dag, shuttle_node_group, qubit_to_pos_next, pos_to_qubit_next)
            # min_op = -1

            # for op in operations_candidate_list:
            #     pos_to_qubit_temp, qubit_to_pos_temp = virtual_execute(op, pos_to_qubit_next, qubit_to_pos_next)
            #     score = sabre_score(dag, shuttle_node_group, qubit_to_pos_temp, pos_to_qubit_temp)
            #     if score < min_score:
            #         min_op = op
            #         min_score = score
            # if min_op != -1:
            #     pos_to_qubit_next, qubit_to_pos_next, clock_map, num_1dmove_gates, num_1dswap_gates, move_target_pos, move_src_pos, swap_nodes, add_into_clock_index = real_execute(min_op, pos_to_qubit_next, qubit_to_pos_next, clock_map, num_1dmove_gates, num_1dswap_gates, move_target_pos, move_src_pos, swap_nodes, add_into_clock_index)
            #     executed_nodes.append(pos_to_qubit_next[op.p0])
            #     executed_nodes.append(pos_to_qubit_next[op.p1])
            #     print("1d", op.op, pos_to_qubit_next[op.p0], pos_to_qubit_next[op.p1], p0, p1)
            # else:
            #     break
            
            if group_dir != -1:
                flag_2d_move = True
                # find 2d move groups
                for node in move_2d_dir_groups[group_dir]:
                    clock_map[node] = move_2d_time

                    p0 = node.p0
                    p1 = node.p1

                    q0 = pos_to_qubit[p0]
                    q1 = pos_to_qubit[p1]

                    pos_to_qubit_next[p0] = q1
                    pos_to_qubit_next[p1] = q0
                    qubit_to_pos_next[q0] = p1
                    qubit_to_pos_next[q1] = p0
                    move_src_pos.append(pos[node.p0])
                    move_target_pos.append(pos[node.p1])
                    num_2dmove_gates += 1
            else:
                # consider 2d swaps
                swap_2d_dir_groups = {}
                min_swap_2d_scores = {}
                for swap_dir_2d in swap_2d_directions:
                    swap_2d_dir_groups[swap_dir_2d] = []
                    for s in shuttle_group:
                        
                        q0 = s[0]
                        q1 = s[1] 
                        p0 = qubit_to_pos[q0]
                        p1 = qubit_to_pos[q1]

                        pos0 = pos[p0]
                        pos1 = pos[p1]
                        
                        if pos0 not in move_src_pos and q0 not in swap_nodes:
                            for neigh in pg.neighbors(p0):
                                qn = pos_to_qubit[neigh]
                                posn = pos[neigh]
                                if posn not in move_src_pos and qn not in swap_nodes:
                                    if qn < n and coupling_distance(neigh, qubit_to_pos_next[q1]) < coupling_distance(p0, qubit_to_pos_next[q1]):
                                        # print("swap1", "distance satisfy")
                                        
                                        if p0 < (D + 1) * D * L:
                                            dir = (posn[0] - pos0[0], posn[1] - pos0[1])
                                        else:
                                            dir = (pos0[0] - posn[0], pos0[1] - posn[1])
                                        
                                        # print("swap1", dir, swap_dir_2d, q0, qn, p0, neigh)
                                        if dir == swap_dir_2d and (qn not in swap_back or p0 not in swap_back[qn]) and (q0 not in swap_back or neigh not in swap_back[q0]):
                                            # node = OpNode('swap', p0, neigh)
                                            if (p0, neigh) not in swap_2d_dir_groups[swap_dir_2d] and (neigh, p0) not in swap_2d_dir_groups[swap_dir_2d]:
                                                # print("add 2d swap1", q0, qn, p0, neigh)
                                                swap_2d_dir_groups[swap_dir_2d].append((p0, neigh))
                                                # print(s, p0, neigh, dir, swap_dir_2d)
                                                break


                        if pos1 not in move_src_pos and q1 not in swap_nodes:
                            for neigh in pg.neighbors(p1):
                                qn = pos_to_qubit[neigh]
                                posn = pos[neigh]
                                
                                if posn not in move_src_pos and qn not in swap_nodes:
                                    
                                    if qn < n and coupling_distance(neigh, qubit_to_pos_next[q0]) < coupling_distance(qubit_to_pos_next[q0], p1):
                                        # print("swap2", "distance satisfy")
                                        
                                        if p1 < (D + 1) * D * L:
                                            dir = (posn[0] - pos1[0], posn[1] - pos1[1])
                                        else:
                                            dir = (pos1[0] - posn[0], pos1[1] - posn[1])
                                        
                                        # print("swap2", dir, swap_dir_2d, q1, qn, p1, neigh)
                                        if dir == swap_dir_2d and (qn not in swap_back or p1 not in swap_back[qn]) and (q1 not in swap_back or neigh not in swap_back[q1]):
                                            # node = OpNode('swap', p1, neigh)
                                            if (p1, neigh) not in swap_2d_dir_groups[swap_dir_2d] and (neigh, p1) not in swap_2d_dir_groups[swap_dir_2d]:
                                                # print("add 2d swap2", q1, qn, p1, neigh)
                                                swap_2d_dir_groups[swap_dir_2d].append((p1, neigh))
                                                # print(s, p1, neigh, dir, swap_dir_2d)
                                                break
                # max_group_size = 0
                # group_dir = -1
                # for m in swap_2d_dir_groups:
                #     if len(swap_2d_dir_groups[m]) > max_group_size:
                #         group_dir = m
                #         max_group_size = len(swap_2d_dir_groups[m])
                # min_score=  sabre_score(dag, shuttle_node_group, qubit_to_pos, pos_to_qubit)
                
                min_score = Infinity
                group_dir = -1

                for m in swap_2d_dir_groups:
                    if len(swap_2d_dir_groups[m]):
                        pos_to_qubit_temp, qubit_to_pos_temp  = pos_to_qubit_next.copy(), qubit_to_pos_next.copy()
                        for op in swap_2d_dir_groups[m]:
                            p0 = op[0]
                            p1 = op[1]
                            node = OpNode('swap', p0, p1)
                            pos_to_qubit_temp, qubit_to_pos_temp = virtual_execute(node, pos_to_qubit_temp, qubit_to_pos_temp, pos_to_qubit)
                        
                        score = sabre_score(dag, shuttle_node_group, qubit_to_pos_temp, pos_to_qubit_temp)
                        if score < min_score:
                            group_dir = m
                            min_score = score

                if add_into_clock_index == 0 and len(clock_map) == 0 or sabre_score(dag, shuttle_node_group, qubit_to_pos_next, pos_to_qubit_next) - min_score >= 5:
                    group_dir = group_dir
                else:
                    group_dir = -1

                if group_dir != -1:
                    flag_2d_swap = True
                    # find 2d swap groups
                    # print(swap_2d_dir_groups[group_dir])
                    for pair in swap_2d_dir_groups[group_dir]:
                        p0 = pair[0]
                        p1 = pair[1]
                        node = OpNode('swap', p0, p1)
                        clock_map[node] = swap_2d_time

                        q0 = pos_to_qubit[p0]
                        q1 = pos_to_qubit[p1]

                        pos_to_qubit_next[p0] = q1
                        pos_to_qubit_next[p1] = q0
                        qubit_to_pos_next[q0] = p1
                        qubit_to_pos_next[q1] = p0
                        num_2dswap_gates += 1
                        swap_nodes.append(pos_to_qubit[node.p0])
                        swap_nodes.append(pos_to_qubit[node.p1])  
                        # print(swap_nodes)  
                        # print("2dswap nodes append", pos_to_qubit[node.p0], pos_to_qubit[node.p1])          
                else:

                    if add_into_clock_index == 0 and len(clock_map) == 0:
                        # print("alarm!!!")
                        congest_count += 1
                        swap_back = {}
                        move_back = {} 
                        target_compute_pos = {}          
        # print("clear clock")
        # print(clock_map)
        # clear clock map
        if len(clock_map):                
            min_time = min(clock_map.values())
            execution_time += min_time
            if min_time == 250 or min_time == 500:
                inter_trap_time += min_time
                num_shuttles += 1
                if flag_2d_swap:
                    flag_2d_swap = False
                
                if flag_2d_move:
                    flag_2d_move = False
            for node in clock_map.copy():
                if min_time >= move_2d_time or clock_map[node] < move_2d_time:
                    clock_map[node] -= min_time

                if clock_map[node] == 0:
                    del clock_map[node]

                    if isinstance(node, OpNode):
                        
                        p0 = node.p0
                        p1 = node.p1

                        q0 = pos_to_qubit[p0]
                        q1 = pos_to_qubit[p1]
                        if min_time == 250 or min_time == 500:
                            node_list.append(('inter-' + node.op, [q0, q1]))
                        node_list.append(('intra-' + node.op, [q0, q1]))
                        
                        # print("remove OpNode", node.op, q0, q1, p0, p1)
                        if node.op == 'move':
                            move_target_pos.remove(pos[p1])
                            move_src_pos.remove(pos[p0])
                            move_back[q0] = p0

                        elif node.op == 'swap':
                            # print(swap_nodes)
                            # print("swap nodes remove", q0, q1)
                            swap_nodes.remove(q0)
                            swap_nodes.remove(q1)
                            if q0 in swap_back:
                                swap_back[q0].append(p0)
                            else:
                                swap_back[q0] = [p0]
                            # swap_back[q1] = p1
                        # print(swap_back)  

                        qubit_to_pos[q0] = p1
                        qubit_to_pos[q1] = p0
                        pos_to_qubit[p0] = q1
                        pos_to_qubit[p1] = q0
                    else:
                        node_list.append((node.op._name, [q._index for q in node.qargs]))
                        if node in shuttle_node_group:
                            shuttle_node_group.remove(node)
                        dag.remove_op_node(node)
                        if node.op.num_qubits == 2 and (node.qargs[0]._index, node.qargs[1]._index) in target_compute_pos:
                            del target_compute_pos[(node.qargs[0]._index, node.qargs[1]._index)]
                        # print("remove")
        # print(clock_map)
        # print(execution_time, dag.depth())   
        if execution_time >= 30000000:
            break

    tg_fidelity = 1 - 18.3 * 10 ** (-4)
    sg_fidelity = 1 - 0.25 * 10 ** (-4)
    transport_fidelity = 1 - 2 * 10 ** (-4)


    fidelity = tg_fidelity ** num_2q_gates * sg_fidelity ** num_1q_gates * transport_fidelity ** (num_1dmove_gates + num_2dmove_gates * 2 + num_1dswap_gates + num_2dswap_gates * 4)
    tg_overall_fidelity = tg_fidelity ** num_2q_gates
    sg_overall_fidelity = sg_fidelity ** num_1q_gates
    transport_overall_fidelity = transport_fidelity ** (num_1dmove_gates + num_2dmove_gates * 2 + num_1dswap_gates + num_2dswap_gates * 4)
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
        "fidelity": fidelity,
        "congest_freq": congest_count / schedule_count
    }
    return results, node_list
