"""
Spherical Network Family (SNF) model from Lunagomez et al (2020) and helper funcitons.

"""

import numpy as np
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt

def hamming(G1, G2):
    return np.sum(G1 != G2)

def n_edges(N):
    return N * (N - 1) // 2

def edge_index(N):
    return list(combinations(range(N), 2))

def sim_SNF(T, Gm, gamma, p, probs):
    """ MCMC simulation of the SNF model with parameters:
        T     : number of time steps
        Gm    : Central network
        gamma : decay parameter
        p     : probability of connection
        probs : list of probabilities

        Simulating for phi(x) = x.
        All networks represented by their vectorisation.
        Take final state as simulated value.

        Typical values: p = [0.05, 0.1, 0.2] probs = [1, 1, 1]
    """
    e = len(Gm)

    #set up
    G_chain = np.zeros((T+1, e))
    G_curr = np.random.binomial(1, np.mean(Gm), size=e)
    G_chain[0, :] = G_curr

    for i in range(T):
       # proposal
        j = np.random.multinomial(1, probs)
        if j[0]==1:
            G_prop = np.abs(G_curr - np.random.binomial(1, p[0], size=e))
        elif j[1] == 1:
            G_prop = np.abs(G_curr - np.random.binomial(1, p[1], size=e))
        else:
            G_prop = np.abs(G_curr - np.random.binomial(1, p[2], size=e))

        a = min(0,-gamma * hamming(G_prop, Gm) + gamma * hamming(G_curr, Gm))
        u = np.random.uniform()
        if np.log(u) < a:
            G_curr = G_prop

        G_chain[i+1, :] = G_curr

    return G_chain


def random_ER_graph(N, p, rng):
    "Graph with N nodes, probability of edge p, stored as edge vector"
    return (rng.random(n_edges(N)) < p).astype(int)

def generate_population(n, T, Gm, gamma, p, probs):
    """
    Generate a population of n observed networks from the SNF model
    centred at Gm with decay parameter gamma.

    Each network is one independent MCMC chain of length T,
    with the final state taken as the simulated draw.
    Shape (n, e).
    """
    e = len(Gm)
    population = np.zeros((n, e))

    for k in range(n):
        chain = sim_SNF(T, Gm, gamma, p, probs)
        population[k, :] = chain[-1, :]

    return population

def generate_population_CER(n, Gm, gamma):
    """
    Exact draw from the CER stationary distribution pi(G) propto exp(-gamma * hamming(G, Gm)).
    Each edge independently flips from Gm with prob alpha = 1/(1+exp(gamma)).
    """
    alpha = 1.0 / (1.0 + np.exp(gamma))
    e = len(Gm)
    flips = np.random.binomial(1, alpha, size=(n, e))
    population = np.abs(Gm[None, :] - flips)
    return population

def edge_vector_to_graph(edge_vector, N):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for (i, j), edge in zip(edge_index(N), edge_vector):
        if edge:
            G.add_edge(i, j)

    return G

def draw_graph(edge_vec, N, ax=None, pos=None, title=None):
    G = edge_vector_to_graph(edge_vec, N)
    if pos is None:
        pos = nx.spring_layout(G, seed=0)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    nx.draw(G, pos, ax=ax, with_labels=True, node_color="lightblue",
            edge_color="gray", node_size=200)
    if title:
        ax.set_title(title)
    return pos

def draw_graph_diff(edge_vec1, edge_vec2, N, ax=None, pos=None, title=None,
                     color_common="gray", color_only1="tab:blue", color_only2="tab:red",
                     label1="only G1", label2="only G2", label_common="common",
                     width=2, legend=True):
    """
    Draw the difference between two graphs on shared node positions.

    edge_vec1, edge_vec2 : edge vectors for the two graphs to compare
    color_common : color for edges present in both graphs
    color_only1  : color for edges only in graph 1
    color_only2  : color for edges only in graph 2
    """
    G1 = edge_vector_to_graph(edge_vec1, N)
    G2 = edge_vector_to_graph(edge_vec2, N)

    edges1 = set(frozenset(e) for e in G1.edges())
    edges2 = set(frozenset(e) for e in G2.edges())

    common = edges1 & edges2
    only1 = edges1 - edges2
    only2 = edges2 - edges1

    def to_edgelist(fs_set):
        return [tuple(e) for e in fs_set]

    if pos is None:
        G_union = nx.Graph()
        G_union.add_nodes_from(range(N))
        G_union.add_edges_from(edges1 | edges2)
        pos = nx.spring_layout(G_union, seed=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    G_nodes = nx.Graph()
    G_nodes.add_nodes_from(range(N))
    nx.draw_networkx_nodes(G_nodes, pos, ax=ax, node_color="lightblue", node_size=200)
    nx.draw_networkx_labels(G_nodes, pos, ax=ax)

    if common:
        nx.draw_networkx_edges(G_nodes, pos, ax=ax, edgelist=to_edgelist(common),
                                edge_color=color_common, width=width, label=label_common)
    if only1:
        nx.draw_networkx_edges(G_nodes, pos, ax=ax, edgelist=to_edgelist(only1),
                                edge_color=color_only1, width=width, label=label1)
    if only2:
        nx.draw_networkx_edges(G_nodes, pos, ax=ax, edgelist=to_edgelist(only2),
                                edge_color=color_only2, width=width, label=label2)

    if title:
        ax.set_title(title)
    if legend:
        ax.legend()
    ax.set_axis_off()

    return pos

def average_network(population, N):
    """Compute the average network from a population of edge vectors."""
    avg_edges = np.mean(population, axis=0)
    avg_graph = (avg_edges > 0.5).astype(int)  # threshold at 0.5
    return avg_graph