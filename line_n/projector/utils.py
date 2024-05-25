import numpy as np


def connect_points(adj_matrix):
    connections = []
    start_point = get_new_start_point(adj_matrix)
    while start_point is not None:
        end_point = get_end_point(adj_matrix, start_point)

        adj_matrix[start_point, end_point] -= 1
        adj_matrix[end_point, start_point] -= 1
        connections.append((start_point, end_point))

        trace_ends = adj_matrix[end_point].sum() <= 0

        start_point = end_point if not trace_ends else get_new_start_point(adj_matrix)
    return connections


def get_new_start_point(adj_matrix):
    # First try to get the node with least (odd) connections
    odd_connection_idx = np.nonzero(adj_matrix.sum(axis=1) % 2 == 1)[0]
    if len(odd_connection_idx) > 0:
        return odd_connection_idx[adj_matrix[odd_connection_idx].sum(axis=1).argmin()]

    # If there are no nodes with connections left, end process
    if adj_matrix.max() == 0:
        return None

    # If no node has odd connections, find the node with most connections.
    # Select the least connected node that is neighbour of the most connected one
    max_conn = adj_matrix.sum(axis=1).argmax()
    useful_neighbors_idx = np.nonzero(adj_matrix[max_conn] > 0)[0]
    max_conn_useful_neighbors = adj_matrix[useful_neighbors_idx]
    return useful_neighbors_idx[max_conn_useful_neighbors.sum(axis=1).argmin()]


def get_end_point(adj_matrix, start_point):
    # Try to get the even neighbor with most connections
    neighbors = np.nonzero(adj_matrix[start_point])[0]
    even_neighbors = neighbors[adj_matrix[neighbors].sum(axis=1) % 2 == 0]

    if len(even_neighbors) > 0:
        return even_neighbors[adj_matrix[even_neighbors].sum(axis=1).argmax()]

    # if no neighbor has even connections, return the most connected neighbor
    return neighbors[adj_matrix[neighbors].sum(axis=1).argmax()]
