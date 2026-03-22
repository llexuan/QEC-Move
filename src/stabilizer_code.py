import re
import qiskit
import numpy as np
from qiskit import qasm2
from bposd.css import css_code

def extract(fname):
    '''
        Input parsing
    '''
    stabilizers = []
    logicals = []

    stabNo = 0
    data_idx = 0
    # ctrlNo = 0

    with open(fname) as f:
        css_line = f.readline()
        is_CSS = css_line.endswith('True\n')

        order_line = f.readline()
        is_ordered = order_line.endswith('True\n')

        qreg_line = f.readline()
        dataNum = int(re.match(r'qreg q\[(\d+)\]', qreg_line).group(1))

        for line in f:
            name = re.match(r'stabilizer\s+(\w+).*', line)
            if name:
                if is_ordered:
                    match = re.findall(r'\(([XYZ]), q\[(\d+)\], (\d+)\)', line)
                    S_k = {'name':name.group(1), 'Data':[], 'Ctrl':[]}
                    for Pauli, data_idx, t in match:
                        data_idx = int(data_idx)
                        t = int(t)
                        S_k['Data'].append(data_idx)
                        S_k['Ctrl'].append([data_idx, Pauli, t])
                    stabilizers.append(S_k)
                    stabNo += 1
                else:
                    match = re.findall(r'\(([XYZ]), q\[(\d+)\]\)', line)
                    S_k = {'name':name.group(1), 'Data':[], 'Ancilla':[], 'Ctrl':[]}
                    for Pauli, data_idx in match:
                        data_idx = int(data_idx)
                        S_k['Data'].append(data_idx)
                        S_k['Ctrl'].append([data_idx, Pauli, None])
                    stabilizers.append(S_k)
                    stabNo += 1
            else:
                name = re.match(r'logical\s+(\w+).*', line)
                match = re.findall(r'\(([XYZ]), q\[(\d+)\]\)', line)
                L_k = {'name':name.group(1), 'Type': None, 'Data':[]}
                for Pauli, data_idx in match:
                    data_idx = int(data_idx)
                    L_k['Type'] = Pauli
                    L_k['Data'].append(data_idx)
                logicals.append(L_k)
                    
    return is_CSS, is_ordered, stabilizers, logicals, dataNum


def to_qiskit(fname, is_CSS, is_ordered, stabilizers, dataNum, iter=1):
    stabNum = len(stabilizers)
    qc = qiskit.QuantumCircuit(dataNum + stabNum, stabNum)
    qc.reset(range(dataNum + stabNum))

    for _ in range(iter):
        cx_list = []
        for s_idx, stab in enumerate(stabilizers):
            anc_idx = s_idx + dataNum
            if (not is_CSS) or stab['Ctrl'][0][1] == 'X':
                qc.h(anc_idx)
            
            if is_CSS:
                for data_idx, Pauli, t in stab['Ctrl']:
                    if Pauli == 'X':
                        cx_list.append(("cx", anc_idx, data_idx, t))
                    elif Pauli == 'Z':
                        cx_list.append(("cx", data_idx, anc_idx, t))
            else:
                raise Exception("TODO")

        if is_ordered:
            for gate, q0, q1, _ in sorted(cx_list, key=lambda x: x[3]):
                if gate == "cx":
                    qc.cx(q0, q1)

        for s_idx, stab in enumerate(stabilizers):
            anc_idx = s_idx + dataNum
            if (not is_CSS) or stab['Ctrl'][0][1] == 'X':
                qc.h(anc_idx)
        
        for s_idx in range(stabNum):
            anc_idx = s_idx + dataNum
            qc.measure(anc_idx, s_idx)
        
        for s_idx in range(stabNum):
            anc_idx = s_idx + dataNum
            qc.reset(anc_idx)

    with open(fname, "w") as f:
        f.write(qasm2.dumps(qc))


def surface_code(fname, d=3):
    def coord2index(i, j):        
        return (d * i + j)

    with open(fname, "w") as f:
        stabilizersNum = 0
        f.write('# is_CSS True\n')
        f.write('# is_ordered True\n')
        f.write('qreg q[%d]\n' % (d * d))

        f.write('logical LZ ')
        for idx in range(d):
            if idx != 0:
                f.write(", ")
            f.write(f'(Z, q[{idx}])')
        f.write("\n")

        for i in range(d - 1):
            for j in range(d - 1):
                stabilizersNum += 1
                f.write('stabilizer S%d ' % stabilizersNum)
                
                top = coord2index(i, j)
                left = coord2index(i + 1, j)
                right = coord2index(i, j + 1)
                down = coord2index(i + 1, j + 1)

                if (i + j) % 2:
                    for k, idx in enumerate([top, right, left, down]):
                        f.write('(X, q[{idx}], {k})'.format(idx=idx, k=k))
                        if k == 3:
                            f.write('\n')
                        else:
                            f.write(', ')
                else:
                    for k, idx in enumerate([top, left, right, down]):
                        f.write('(Z, q[{idx}], {k})'.format(idx=idx, k=k))
                        if k == 3:
                            f.write('\n')
                        else:
                            f.write(', ')
        
        for j in range(d // 2):
            stabilizersNum += 1
            f.write('stabilizer S%d ' % stabilizersNum)
            f.write('(X, q[{i0}], 2), (X, q[{i1}], 3)\n'.format(i0 = coord2index(0, j*2), i1 = coord2index(0, j*2+1)))

            stabilizersNum += 1
            f.write('stabilizer S%d ' % stabilizersNum)
            if d % 2 == 1:
                f.write('(X, q[{i0}], 0), (X, q[{i1}], 1)\n'.format(i0 = coord2index(d-1,  j*2+1), i1 = coord2index(d-1, j*2+2)))
            else:
                f.write('(X, q[{i0}], 0), (X, q[{i1}], 1)\n'.format(i0 = coord2index(d-1,  j*2), i1 = coord2index(d-1, j*2+1)))
        
        for i in range((d - 1) // 2):
            stabilizersNum += 1
            f.write('stabilizer S%d ' % stabilizersNum)
            f.write('(Z, q[{i0}], 2), (Z, q[{i1}], 3)\n'.format(i0 = coord2index(i*2+1, 0), i1 = coord2index(i*2+2, 0)))

            stabilizersNum += 1
            f.write('stabilizer S%d ' % stabilizersNum)
            if d % 2 == 1:
                f.write('(Z, q[{i0}], 0), (Z, q[{i1}], 1)\n'.format(i0 = coord2index(i*2, d-1), i1 = coord2index(i*2+1, d-1)))
            else:
                f.write('(Z, q[{i0}], 0), (Z, q[{i1}], 1)\n'.format(i0 = coord2index(i*2+1, d-1), i1 = coord2index(i*2+2, d-1)))


def bb_code(fname, ell=6, m=6):
    # Takes as input a binary square matrix A
    # Returns the rank of A over the binary field F_2
    def rank2(A):
        rows,n = A.shape
        X = np.identity(n,dtype=int)

        for i in range(rows):
            y = np.dot(A[i,:], X) % 2
            not_y = (y + 1) % 2
            good = X[:,np.nonzero(not_y)]
            good = good[:,0,:]
            bad = X[:, np.nonzero(y)]
            bad = bad[:,0,:]
            if bad.shape[1]>0 :
                bad = np.add(bad,  np.roll(bad, 1, axis=1) ) 
                bad = bad % 2
                bad = np.delete(bad, 0, axis=1)
                X = np.concatenate((good, bad), axis=1)
        # now columns of X span the binary null-space of A
        return n - X.shape[1]

    sX= ['idle', 1, 4, 3, 5, 0, 2]
    sZ= [3, 5, 0, 1, 2, 4, 'idle']

    # [[144, 12]]
    a1,a2,a3 = 3,1,2
    b1,b2,b3 = 3,1,2
    b1,b2,b3 = 3,1,2

    # code length
    n = 2*m*ell
    n2 = m*ell

    # cyclic shift matrices 
    I_ell = np.identity(ell,dtype=int)
    I_m = np.identity(m,dtype=int)
    I = np.identity(ell*m,dtype=int)
    x = {}
    y = {}
    for i in range(ell):
        x[i] = np.kron(np.roll(I_ell,i,axis=1),I_m)
    for i in range(m):
        y[i] = np.kron(I_ell,np.roll(I_m,i,axis=1))

    A = (x[a1] + y[a2] + y[a3]) % 2
    B = (y[b1] + x[b2] + x[b3]) % 2

    A1 = x[a1]
    A2 = y[a2]
    A3 = y[a3]
    B1 = y[b1]
    B2 = x[b2]
    B3 = x[b3]

    AT = np.transpose(A)
    BT = np.transpose(B)

    hx = np.hstack((A,B))
    hz = np.hstack((BT,AT))

    qcode = css_code(hx, hz)
    lz = qcode.lz.toarray()
    print(lz.shape)

    with open(fname, "w") as f:
        f.write('# is_CSS True\n')
        f.write('# is_ordered True\n')
        f.write('qreg q[%d]\n' % (n))

        for l_idx in range(lz.shape[0]):
            f.write(f'logical LZ{l_idx} ')
            for idx in range(lz.shape[1]):
                if lz[l_idx][idx] != 0:
                    f.write(", ")
                    f.write(f'(Z, q[{idx}])')
            f.write("\n")

        # Give a name to each qubit
        # Define a linear order on the set of qubits
        lin_order = {}
        data_qubits = []
        Xchecks = []
        Zchecks = []

        cnt = 0
        for i in range(n2):
            node_name = ('Xcheck', i)
            Xchecks.append(node_name)
            lin_order[node_name] = cnt
            cnt += 1

        for i in range(n2):
            node_name = ('data_left', i)
            data_qubits.append(node_name)
            lin_order[node_name] = cnt
            cnt += 1

        for i in range(n2):
            node_name = ('data_right', i)
            data_qubits.append(node_name)
            lin_order[node_name] = cnt
            cnt += 1

        for i in range(n2):
            node_name = ('Zcheck', i)
            Zchecks.append(node_name)
            lin_order[node_name] = cnt
            cnt += 1

        # compute the list of neighbors of each check qubit in the Tanner graph
        nbs = {}
        # iterate over X checks
        for i in range(n2):
            check_name = ('Xcheck', i)
            # left data qubits
            nbs[(check_name,0)] = ('data_left',np.nonzero(A1[i,:])[0][0])
            nbs[(check_name,1)] = ('data_left',np.nonzero(A2[i,:])[0][0])
            nbs[(check_name,2)] = ('data_left',np.nonzero(A3[i,:])[0][0])
            # right data qubits
            nbs[(check_name,3)] = ('data_right',np.nonzero(B1[i,:])[0][0])
            nbs[(check_name,4)] = ('data_right',np.nonzero(B2[i,:])[0][0])
            nbs[(check_name,5)] = ('data_right',np.nonzero(B3[i,:])[0][0])

        # iterate over Z checks
        for i in range(n2):
            check_name = ('Zcheck', i)
            # left data qubits
            nbs[(check_name,0)] = ('data_left',np.nonzero(B1[:,i])[0][0])
            nbs[(check_name,1)] = ('data_left',np.nonzero(B2[:,i])[0][0])
            nbs[(check_name,2)] = ('data_left',np.nonzero(B3[:,i])[0][0])
            # right data qubits
            nbs[(check_name,3)] = ('data_right',np.nonzero(A1[:,i])[0][0])
            nbs[(check_name,4)] = ('data_right',np.nonzero(A2[:,i])[0][0])
            nbs[(check_name,5)] = ('data_right',np.nonzero(A3[:,i])[0][0])
        
        for s_idx, control in enumerate(Xchecks):
            f.write('stabilizer SX%d ' % s_idx)
            for t in range(7):
                direction = sX[t]
                if direction != 'idle':
                    if t != 1:
                        f.write(", ")
                    f.write(f'(X, q[{int(nbs[(control, direction)][1])}], {t})')
            f.write("\n")

        for s_idx, target in enumerate(Zchecks):
            f.write('stabilizer SZ%d ' % s_idx)
            for t in range(7):
                direction = sZ[t]
                if direction != 'idle':
                    if t != 0:
                        f.write(", ")
                    f.write(f'(Z, q[{int(nbs[(target, direction)][1])}], {t})')
            f.write("\n")