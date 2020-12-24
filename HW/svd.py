'''
    SVD Algorithm
'''

import numpy as np
np.set_printoptions(precision=6, suppress=True)


def update(M, u, v, c):
    n = list(v.shape)[0]

    if np.all(np.dot(M, v) == 0):
        print("Shuffled")
        v = np.random.random((n, 1))

    u = np.dot(M, v)
    if np.linalg.norm(u) != 0:
        u = u / np.linalg.norm(u)
    v = np.dot(M.T, u)
    c = np.linalg.norm(v)
    if np.linalg.norm(v) != 0:
        v = v / np.linalg.norm(v)
    return M, u, v, c

def orthogonal_expansion(group, dim):
    rank_A = len(group)
    if rank_A == 0:
        A = np.eye(dim)
        return A
    m = list(group[0].shape)[0]
    A = np.zeros([m, m])
    if rank_A < m:
        for j in range(0, m):
            x = np.zeros([m, 1])
            x[j][0] = 1
            for k in range(0, rank_A):
                A[:, [k]] = group[k]
            A[:, [rank_A]] = x
            if np.linalg.matrix_rank(A) == rank_A + 1:
                y = x.copy()
                for i in range(0, rank_A):
                    x -= 1.0 * np.dot(x.T, group[k]) * group[k]
                x = x / np.linalg.norm(x)
                group.append(x)
                rank_A += 1
            if rank_A == m:
                break

    for i in range(0, m):
        A[:, [i]] = group[i]
    return A


def main():
    M = np.array([[0, 0], [-1, 1]])
    #M = np.eye(4, 4) #+ np.ones([5, 4])
    #M = -np.random.random((4, 4))
    m = list(M.shape)[0]
    n = list(M.shape)[1]
    u = np.ones([m, 1])
    v = np.ones([n, 1])
    #v = np.random.random((n, 1))
    c = 1.0
    k = min(m, n)
    N = M.copy()

    out_dict = dict()
    out_dict['M'] = []
    out_dict['u'] = []
    out_dict['v'] = []
    out_dict['c'] = []


    for i in range(1, k+1):
        while not np.all(abs(M) < 1e-12):
            for j in range(1, 20+1):
                M, u, v, c= update(M, u, v, c)
            #print(M, u, v, c)
            out_dict['u'].append(u)
            out_dict['v'].append(v)
            out_dict['c'].append(c)
            out_dict['M'].append(M)
            M = M - c * np.dot(u, v.T)

            u = np.ones([m, 1])
            v = np.random.random((n, 1))
            c = 1.0

    print("*******************************\nOriginal M:\n{}".format(N))
    rank = len(out_dict['c'])
    print("rank={}\n".format(rank))
    U_true, s, V_true = np.linalg.svd(N, full_matrices=True)
    D_true = np.zeros_like(M)
    D_true[:k, :k] = np.diag(s)
    print("*******************************\nTrue U,D,V:\n{}\n{}\n{}\n".format(U_true, D_true, V_true))


    D = np.zeros_like(M)
    U = orthogonal_expansion(out_dict['u'], m)
    V = orthogonal_expansion(out_dict['v'], n)

    if rank > 0:
        for i in range(0, rank):
            D[i][i] = out_dict['c'][i]
    print("*******************************\nComputed U,D,V:\n{}\n{}\n{}\n".format(U, D, V))
    #print(np.dot(U.T, U), '\n', np.dot(V, V.T))
    K = np.dot(np.dot(U, D), V.T)
    print("Recovery Test UDV*: \n{}".format(K))
    print("L2 Loss = {}".format(np.linalg.norm(K-N)))






if __name__ == '__main__':
    main()