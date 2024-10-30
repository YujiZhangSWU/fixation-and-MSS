import networkx as nx
import numpy as np
import pandas as pd


def calculation_of_ancestral_random_walk():
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            A[i, j] = W[i, j] / np.sum(W[i])
    return A


# calculation of m^ij_k
def marginal_effects_of_k_on_selection(k):
    M_k = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == k:
                M_k[i, j] = A[j, i] * (1 - A[j, i]) / N
            else:
                M_k[i, j] = -A[j, i] * A[j, k] / N
    return M_k


def identity_by_state_probabilities():
    # Coefficient matrix
    C = np.zeros((N * N, N * N))
    b = np.ones((N * N, 1)) * u / 2

    for row in range(N * N):
        i = int(row / N)
        j = row % N
        # phi_ii = 1
        if i == j:
            C[row, row] = 1
            b[row, 0] = 1
        else:
            for k in range(i * N, (i + 1) * N):
                if k % N == j:
                    C[row, k] = 1 - (1 - u) * 0.5 * A[j, k % N]
                else:
                    C[row, k] = -(1 - u) * 0.5 * A[j, k % N]

            for k in range(j, N ** 2 + j, N):
                if int(k / N) == i:
                    C[row, k] = 1 - (1 - u) * 0.5 * A[i, int(k / N)]
                else:
                    C[row, k] = -(1 - u) * 0.5 * A[i, int(k / N)]

    return np.linalg.solve(C, b).reshape(N, N)


graph = "ER"
adjacency_matrix = pd.read_csv(graph + ".csv", header=None)
G = nx.from_pandas_adjacency(adjacency_matrix)
N = len(G.nodes)
W = nx.adjacency_matrix(G).todense()

# mutation rate
u = 0.05
# transition matrix for the ancestral random walk, where A_ij = w_ij / w_i
A = calculation_of_ancestral_random_walk()

# A vector of the death probabilities under neutral drift
D = np.ones(N) / N

# mutation-weighted reproductive values
Pi = np.sum(u * np.linalg.inv(np.identity(N) - (1 - u) * A), axis=0)

# matrix of pairwise identity-by-state probabilities
Phi = identity_by_state_probabilities()

# interaction matrix w_ij / w_i
Omega = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        Omega[i, j] = W[i, j] / np.sum(W[i])

K1 = 0
for i in range(N):
    for j in range(N):
        for k in range(N):
            M_k = marginal_effects_of_k_on_selection(k)
            for l in range(N):
                K1 += 1 / (2 * u) * Pi[i] * M_k[j, i] * Omega[k, l] * (-(Phi[i, k] + Phi[i, l]) + (1 - u) * (Phi[j, k] + Phi[j, l]) + u)
K2 = 0
for i in range(N):
    for j in range(N):
        for k in range(N):
            M_k = marginal_effects_of_k_on_selection(k)
            for l in range(N):
                K2 += 1 / (2 * u) * Pi[i] * M_k[j, i] * Omega[k, l] * (-(Phi[i, k] - Phi[i, l]) + (1 - u) * (Phi[j, k] - Phi[j, l]))

# DG
print(K1, K2)
print("DG")
critical_ratio = (K1 + K2) / (K1 - K2)
print(critical_ratio)

# \Delta b = 1 and c = 1
# critical_ratio_with_transitions = (K1 + K2) / (K1 - K2) - K2 / (K1 - K2)
# print(critical_ratio_with_transitions)


# SG
print("SG")
critical_ratio = (K1 + 2 * K2) / (2 * K1)
print(critical_ratio)

# save data
df = pd.DataFrame({'critical ratio': [critical_ratio]})
df.to_csv(graph + " u = " + str(u) + ".txt", index=False)
