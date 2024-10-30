# calculate three dimensional coalescence times for any connected unweighted graph
# author: Yuji Zhang

import networkx as nx
import numpy as np
import pandas as pd

# adjacency_matrix = pd.read_csv("WS.csv", header=None)
files = ["beetle.txt", "karate.txt", "primate.txt"]
# graphs = ["BA", "ER", "WS"]

graphs = ["RG"]
files = ["women"]

for file in files:
    print(file)
    # adjacency_matrix = pd.read_csv("empirical networks\\" + file, header=None)
    # G = nx.from_pandas_adjacency(adjacency_matrix)
    # G = nx.read_edgelist("empirical networks\\" + file)
    G = nx.davis_southern_women_graph()
    adjacencyMatrix = nx.adjacency_matrix(G).todense()
    N = len(G.nodes)
    # One step transition probability matrix
    P = np.zeros((N, N))

    for row in range(N):
        sumOverRow = np.sum(adjacencyMatrix[row])
        for col in range(N):
            P[row, col] = adjacencyMatrix[row, col] / sumOverRow
    P2 = P @ P
    P3 = P @ P @ P
    Pi = np.zeros(N)
    W = np.sum(adjacencyMatrix)
    for i in range(N):
        Pi[i] = np.sum(adjacencyMatrix[i]) / W

    A = np.zeros((N * N, N * N))
    B = np.ones((N * N, 1)) / 2
    eta_ii = [i * N + i for i in range(N)]
    for row in range(N * N):
        if row in eta_ii:
            A[row, row] = 1
            B[row, 0] = 0
        else:
            i = int(row / N)
            j = row % N
            for k in range(i * N, (i + 1) * N):
                if k % N == j:
                    A[row, k] = 1 - 0.5 * P[j, k % N]
                else:
                    A[row, k] = -0.5 * P[j, k % N]
            for k in range(j * N, (j + 1) * N):
                A[row, k] = -0.5 * P[i, k % N]

    X = np.linalg.solve(A, B)
    Eta_ij = X.reshape(N, N)

    # three dimensional coalescence times
    A = np.zeros((N * N * N, N * N * N))
    np.fill_diagonal(A, 1)

    B = np.ones((N * N * N, 1)) / 3

    for row in range(N * N * N):
        i = int(row / N ** 2)
        j = int((row - i * N ** 2) / N)
        k = row % N
        if i == j and j == k:
            A[row, row] = 1
            B[row, 0] = 0
        elif i == j:
            A[row, row] = 1
            B[row, 0] = Eta_ij[i, k]
        elif i == k:
            A[row, row] = 1
            B[row, 0] = Eta_ij[i, j]
        elif j == k:
            A[row, row] = 1
            B[row, 0] = Eta_ij[i, k]
        else:
            for l in range(j * N + k, N ** 3 + j * N + k, N ** 2):
                if int(l / N ** 2) == i:
                    A[row, l] -= 1 / 3 * P[i, int(l / N ** 2)]
                elif int(l / N ** 2) == j:
                    B[row, 0] += 1 / 3 * P[i, j] * Eta_ij[j, k]
                elif int(l / N ** 2) == k:
                    B[row, 0] += 1 / 3 * P[i, k] * Eta_ij[j, k]
                else:
                    A[row, l] = - 1 / 3 * P[i, int(l / N ** 2)]
            for l in range(i * N ** 2 + k, (i + 1) * N ** 2 + k, N):
                if int(int(l - i * N ** 2) / N) == j:
                    A[row, l] -= 1 / 3 * P[j, int(int(l - i * N ** 2) / N)]
                elif int(int(l - i * N ** 2) / N) == i:
                    B[row, 0] += 1 / 3 * P[j, i] * Eta_ij[i, k]
                elif int(int(l - i * N ** 2) / N) == k:
                    B[row, 0] += 1 / 3 * P[j, k] * Eta_ij[i, k]
                else:
                    A[row, l] = - 1 / 3 * P[j, int(int(l - i * N ** 2) / N)]
            for l in range(i * N ** 2 + j * N, i * N ** 2 + (j + 1) * N):
                if l % N == k:
                    A[row, l] -= 1 / 3 * P[k, l % N]
                elif l % N == j:
                    B[row, 0] += 1 / 3 * P[k, j] * Eta_ij[i, j]
                elif l % N == i:
                    B[row, 0] += 1 / 3 * P[k, i] * Eta_ij[i, j]
                else:
                    A[row, l] = - 1 / 3 * P[k, l % N]

    X = np.linalg.solve(A, B)
    Eta_ijk = X.reshape(N, N, N)

    eta_1 = 0
    for i in range(N):
        for j in range(N):
            eta_1 += Pi[i] * P[i, j] * Eta_ij[i, j]

    eta_2 = 0
    for i in range(N):
        for j in range(N):
            eta_2 += Pi[i] * P2[i, j] * Eta_ij[i, j]

    eta_3 = 0
    for i in range(N):
        for j in range(N):
            eta_3 += Pi[i] * P3[i, j] * Eta_ij[i, j]

    Lambda_2 = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                Lambda_2 += Pi[i] * P2[i, j] * P[j, k] * Eta_ijk[i, j, k]

    u2 = [1, 0]
    u1 = [0, 1]

    Delta_b = 1
    c = 1

    # DG
    ratioDG0 = eta_2 / (eta_3 - eta_1)
    ratioDG1 = ratioDG0 + ((Lambda_2 - eta_2) / (eta_3 - eta_1) * u2[1] - (Lambda_2 - eta_3) / (eta_3 - eta_1) * u1[
        1]) * Delta_b / c

    # PGG
    delta_2 = 0.5
    Xi = (delta_2 - 1) * (Lambda_2 - eta_1) + eta_2 + eta_3 - eta_1
    ratioPGG0 = 2 * eta_2 / Xi
    ratioPGG1 = ratioPGG0 + (
            (delta_2 + 1) * (Lambda_2 - eta_1) / Xi * u2[1] - (2 * Lambda_2 - eta_1 - eta_2 - eta_3) / Xi * u1[
        1]) * Delta_b / c

    # SG
    Delta_b = 0.6
    ratioSG0 = (Lambda_2 - eta_1 - 2 * eta_2) / (2 * (Lambda_2 - eta_2 - eta_3))
    ratioSG1 = ratioSG0 + ((Lambda_2 - eta_1) / (eta_2 + eta_3 - Lambda_2) * u2[1] - (
            2 * Lambda_2 - eta_1 - eta_2 - eta_3) / (eta_2 + eta_3 - Lambda_2) * u1[1]) * Delta_b / c

    # DS
    u = [0.6, 0.4]
    ratioDS = (u[1] * eta_1 + 2 * eta_2 - u[1] * Lambda_2) / (
            -2 * u[0] * eta_1 + 2 * u[1] * eta_2 + 2 * eta_3 - 2 * u[1] * Lambda_2)

    print(ratioDG0, ratioDG1)
    print(ratioPGG0, ratioPGG1)
    print(ratioSG0, ratioSG1)

    df = pd.DataFrame({'critical ratio without game transition DG': [ratioDG0], 'critical ratio DG': [ratioDG1],
                       'critical ratio without game transition PGG': [ratioPGG0], 'critical ratio PGG': [ratioPGG1]})
    # df = pd.DataFrame({'critical ratio': [ratioDS]})
    #
    df.to_csv("empirical networks\\threshold\\" + file, sep='\t', index=False)
    # df.to_csv(graph + " DS 0.6.txt", index=False)

    # u = [0.4, 0.6]
    # ratioDS = (u[1] * eta_1 + 2 * eta_2 - u[1] * Lambda_2) / (
    #         -2 * u[0] * eta_1 + 2 * u[1] * eta_2 + 2 * eta_3 - 2 * u[1] * Lambda_2)
    # print(ratioDS)
    # df = pd.DataFrame({'critical ratio': [ratioDS]})
    # df.to_csv(graph + " DS 0.4.txt", index=False)
