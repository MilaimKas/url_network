
#%%

import pandas as pd
import numpy as np
from random import randint, seed
from datetime import timedelta
import networkx as nx
from app_helpers import helpers
from scipy.stats import dirichlet, poisson
from networkx.algorithms.community import greedy_modularity_communities



def generate_dummy_df(n_sessions=100, min_pages_per_session=2, max_pages_per_session=6, url_sample=None, seed_value=42, choice_prob=None):
    """
    Generate a DataFrame that simulates user sessions, where each row corresponds
    to a session with:
        - a list of URLs visited in order,
        - and time spent per URL (chrono).
    
    Returns:
        pd.DataFrame with columns:
            - session_id
            - url_path: list of full URLs
            - chrono: list of times spent per URL (float)
    """
    seed(seed_value)
    np.random.seed(seed_value)

    # define start and end dates for sessions
    start_date = pd.to_datetime("2023-01-01")
    end_date = pd.to_datetime("2023-03-01")

    # Fake hierarchical URL fragments to construct full URLs
    if url_sample is None:
        url_sample = helpers.url_sample
    
    # Use Dirichlet distribution to generate probabilities of choosing each URL
    if choice_prob is None:
        choice_prob = dirichlet.rvs(poisson.rvs(200, size=len(url_sample)))[0]
    elif isinstance(choice_prob, (list, np.ndarray)):
        if len(choice_prob) != len(url_sample):
            raise ValueError("Length of choice_prob must match length of url_sample")
        if not np.isclose(sum(choice_prob), 1):
            raise ValueError("choice_prob must sum to 1")
    else:
        raise ValueError("choice_prob must be None or a list/array of probabilities")

    sessions = []
    for i in range(n_sessions):
        session_id = f"session_{i:04d}"
        session_length = randint(min_pages_per_session, max_pages_per_session)

        url_path = []
        url_node = []  # To store the last part of the URL as node
        for _ in range(session_length):
            url = np.random.choice(url_sample, p=choice_prob)    
            url_path.append(url)
            url_node.append(url.split("/")[-1])  # Extract the last part of the URL as node

        # Simulate time spent (e.g., exponential time-on-page)
        chrono = np.round(np.random.exponential(scale=20, size=session_length), 2)
        
        # construct random date column
        random_days = randint(0, (end_date - start_date).days)
        random_date = start_date + timedelta(days=random_days)

        sessions.append({
            "session_id": session_id,
            "url_path": url_path,
            "chrono": list(chrono),
            "url_node": url_node,
            "device": "desktop" if randint(0, 1) == 0 else "mobile",
            "date": random_date
        })
    
    return pd.DataFrame(sessions)


def info_edge(G, att_weight, shift=0):
    """
    returns a df with edges and shifted weigths
    """

    df = pd.DataFrame({"edge":list(nx.edges(G).keys())})
        
    for att in att_weight: 
        tmp = list(nx.get_edge_attributes(G, att).values())
        tmp = abs(np.asarray(tmp)-shift)
        df = df.assign(**{att: tmp})

    return df


def info_node(G, att_weight):
    """
    returns a df with nodes and attr name
    """

    df = pd.DataFrame({"node":list(nx.nodes(G).keys())})
    
    for att in att_weight: 
        df = df.assign(**{att: list(nx.get_node_attributes(G, att).values())})
    
    return df


def get_all_paths(A, B, G, C=None, cutoff=5):
    """
    return and iterator object with all possible paths from A to B containing C
    cutoff = max path length
    """
    if C is None:
        path = nx.all_simple_paths(G, A, B, cutoff=cutoff)
        return [p for p in path]
    else:
        path = nx.all_simple_paths(G, A, B, cutoff=cutoff)
        return [p for p in path if C in p]


def get_shortest_k_path(A, B, att, G, C=None, shift=None, k=10):
    print(f"[get_shortest_k_path] source: {A}, target: {B}, k: {k}, shift: {shift}, C: {C}")
    
    if att is None:
        att = "weights"
    try:
        edge_weights = nx.get_edge_attributes(G, att)
        print(f"[get_shortest_k_path] Edge weights extracted, total: {len(edge_weights)}")
    except Exception as err:
        raise ValueError(f"Attribute {att} not found in network: {err}")

    if shift is None:
        shift = max(edge_weights.values()) + 1
        print(f"[get_shortest_k_path] shift computed as: {shift}")

    if not G.has_node(A) or not G.has_node(B):
        raise ValueError(f"Missing node: {A} or {B}")
    
    if G.has_edge(A, B):
        print("[get_shortest_k_path] Direct edge exists")
    
    try:
        paths = nx.shortest_simple_paths(G, source=A, target=B, weight=att)
        print("[get_shortest_k_path] shortest_simple_paths iterator created")
    except Exception as e:
        raise ValueError(f"Error finding path: {e}")

    results = []
    for p in paths:
        print(f"[get_shortest_k_path] path {len(results)}: {p}")
        # if intermediates specified, filter paths
        if C and C is not None and C not in p:
            continue
        score_abs = float(sum([G[u][v][att] for u, v in zip(p, p[1:])]))
        score_norm = float(score_abs / (len(p) - 1) if len(p) > 1 else score_abs)
        results.append((p, score_abs, score_norm))
        # stop if we have enough paths
        if len(results) >= k:
            break

    print(f"[get_shortest_k_path] total paths collected: {len(results)}")
    return pd.DataFrame(results, columns=["path", "score_abs", "score_norm"])


def get_flow(G, B, att, shift=0):
    """
    return session flow through B
    """

    list_in = [G[u][v][att] for u,v in G.in_edges(B)]
    list_out = [G[u][v][att] for u,v in G.out_edges(B)]

    flow_in = shift*len(list_in) - sum(list_in)
    flow_out = shift*len(list_out) - sum(list_out)

    return flow_in, flow_out


def get_flow_nodes(G, att, shift=0):
    """
    returns flow of all nodes
    """
    
    flow_in = []
    flow_out = []
    for n in G.nodes():
        i, o = get_flow(G, n, att, shift=shift)
        flow_in.append(i)
        flow_out.append(o)

    df_tmp = pd.DataFrame({"nodes":G.nodes(), "flow in":flow_in, "flow out":flow_out})
    df_tmp["flow"] = df_tmp["flow in"] - df_tmp["flow out"] 

    df_tmp = df_tmp.sort_values(by="flow", key=abs, ascending=False).reset_index()

    return df_tmp 


def compute_metrics(G):
    """
    Compute various network metrics on the given directed graph G.
    Returns a dictionary of computed metrics.
    """        
    
    # Filter for navigation edges
    nav_edges = [(u, v) for u, v, a in G.edges(data=True) if a.get("group") == "network"]
    nav_G = G.edge_subgraph(nav_edges).copy()

    # 1. Entropy over edge weights
    weights = np.array([G[u][v].get("weight", 0) for u, v in nav_edges])
    weight_probs = weights / weights.sum()
    entropy = -np.sum(weight_probs * np.log2(weight_probs + 1e-12))

    # 2. Modularity (undirected view)
    try:
        communities = list(greedy_modularity_communities(nav_G.to_undirected()))
        modularity = nx.algorithms.community.modularity(nav_G.to_undirected(), communities)
    except:
        modularity = 0

    # 3. Largest weakly connected component
    components = list(nx.weakly_connected_components(nav_G))
    largest_component = max(len(c) for c in components) if components else 0

    # 4. Degree statistics
    degrees = [d for _, d in nav_G.degree()]
    degree_mean = np.mean(degrees)
    degree_std = np.std(degrees)

    # 5. Node flow std
    flow_df = get_flow_nodes(G=nav_G,  att="weight")
    flow_std = flow_df["flow"].std()

    # 6. Pagerank entropy (centralization measure)
    pr = nx.pagerank(nav_G, alpha=0.85)
    pr_values = np.array(list(pr.values()))
    pr_probs = pr_values / pr_values.sum()
    pr_entropy = -np.sum(pr_probs * np.log2(pr_probs + 1e-12))

    # 7. Counts
    node_count = nav_G.number_of_nodes()
    edge_count = nav_G.number_of_edges()

    return {
        "entropy": entropy, 
        "modularity": modularity, 
        "largest_component": largest_component,
        "degree_mean": degree_mean, 
        "degree_std": degree_std, 
        "flow_std": flow_std, 
        "pagerank_entropy": pr_entropy,
        "nodes": node_count,
        "edges": edge_count
    }
