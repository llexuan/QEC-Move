from qiskit import QuantumCircuit
from random import sample
from math import *
import random

def genQiskitBV(program_qubit_num, percentage=0.5):
    all_possible_gates = [(0,i) for i in range(1,program_qubit_num)]
    gates = list(set(sample(all_possible_gates, k=int(len(all_possible_gates)*percentage))))
    gates.sort()
    print("{}/{} gates selected : {} ... {}".format(len(gates), len(all_possible_gates), gates[:10], gates[-10:]))
    
    qiskit_circuit = QuantumCircuit(program_qubit_num)
    for c,t in gates:
        qiskit_circuit.cx(c,t)
    return qiskit_circuit
    

def genQiskitQAOA(program_qubit_num, percentage=0.5):
    all_possible_gates = [(i,j) for i in range(program_qubit_num) for j in range(i+1, program_qubit_num)]
    gates = list(set(sample(all_possible_gates, k=int(len(all_possible_gates)*percentage))))
    gates.sort()
    print("{}/{} gates selected : {} ... {}".format(len(gates), len(all_possible_gates), gates[:10], gates[-10:]))
    
    qiskit_circuit = QuantumCircuit(program_qubit_num)
    for c,t in gates:
        qiskit_circuit.cx(c,t)
        qiskit_circuit.rz(pi/3, t)
        qiskit_circuit.cx(c,t)
    return qiskit_circuit


def qiskit_control_rotation(qiskit_circuit, c, r):
    qiskit_circuit.rz(pi/3, c)
    qiskit_circuit.rz(pi/3, r)
    qiskit_circuit.cx(c,r)
    qiskit_circuit.rz(pi/3, r)
    qiskit_circuit.cx(c,r)

def genQiskitQFT(program_qubit_num):
    qiskit_circuit = QuantumCircuit(program_qubit_num)
    for target in range(program_qubit_num - 1):
        qiskit_circuit.h(target)
        for control in range(target + 1, program_qubit_num):
            qiskit_circuit.rz(pi/3, control)
            qiskit_circuit.rz(pi/3, target)
            qiskit_circuit.h(target)
            qiskit_circuit.cz(control, target)
            qiskit_circuit.h(target)
            qiskit_circuit.rz(pi/3, target)
            qiskit_circuit.h(target)
            qiskit_circuit.cz(control, target)
            qiskit_circuit.h(target)
    qiskit_circuit.h(program_qubit_num - 1)
    return qiskit_circuit


def genQiskitVQE(program_qubit_num):
    qiskit_circuit = QuantumCircuit(program_qubit_num)
    for i in range(program_qubit_num):
        for j in range(i+1, program_qubit_num):
            qiskit_circuit.cx(i,j)
    print('{} gates'.format(program_qubit_num**2/2))
    return qiskit_circuit

def genQiskitRCA(program_qubit_num):
    qiskit_circuit = QuantumCircuit(program_qubit_num)
    #qubit_num = 3 * n + 1
    for i in range(program_qubit_num - 3):
        qiskit_circuit.cx(i + 2,i + 1)
        qiskit_circuit.cx(i + 2,i)
        qiskit_circuit.h(i + 2)
        qiskit_circuit.cx(i + 1, i + 2)
        qiskit_circuit.rz(pi / 3, i + 2)
        qiskit_circuit.cx(i, i + 2)
        qiskit_circuit.rz(pi / 3, i + 2)
        qiskit_circuit.cx(i + 1, i + 2)
        qiskit_circuit.rz(pi / 3, i + 2)
        qiskit_circuit.cx(i, i + 2)
        
        qiskit_circuit.rz(pi / 3, i + 1)
        qiskit_circuit.rz(pi / 3, i + 2)
        qiskit_circuit.h(i + 2)
        
        qiskit_circuit.cx(i, i + 1)
        qiskit_circuit.rz(pi / 3, i)
        qiskit_circuit.rz(pi / 3, i + 1)
        qiskit_circuit.cx(i, i + 1)
    qiskit_circuit.cx(program_qubit_num - 2, program_qubit_num - 1)
    for i in range(program_qubit_num - 4, -1, -1):
        qiskit_circuit.h(i + 2)
        qiskit_circuit.cx(i + 1, i + 2)
        qiskit_circuit.rz(pi / 3, i + 2)
        qiskit_circuit.cx(i, i + 2)
        qiskit_circuit.rz(pi / 3, i + 2)
        qiskit_circuit.cx(i + 1, i + 2)
        qiskit_circuit.rz(pi / 3, i + 2)
        qiskit_circuit.cx(i, i + 2)
        
        qiskit_circuit.rz(pi / 3, i + 1)
        qiskit_circuit.rz(pi / 3, i + 2)
        qiskit_circuit.h(i + 2)
        
        qiskit_circuit.cx(i, i + 1)
        qiskit_circuit.rz(pi / 3, i)
        qiskit_circuit.rz(pi / 3, i + 1)
        qiskit_circuit.cx(i, i + 1)
        qiskit_circuit.cx(i + 2,i)
        qiskit_circuit.cx(i ,i + 1)

    return qiskit_circuit