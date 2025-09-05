"""
Class for the Parship domain network

TODO: vectorize create_network function using
nx.from_pandas_edgelist(df, source = 'node1', target = 'node2', edge_attr = 'edge_weight', create_using = nx.DiGraph())
"""

import networkx as nx

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from collections import defaultdict

from networkx.readwrite import json_graph

from app_helpers.network_functions import *
from app_helpers.helpers import safe_float


class WebNetwork:
    """
    Class to create and manage a domain network based on URLs.
    The domain network is a directed graph where nodes are URLs and edges represent the paths between them.
    The class provides methods to create the network from a DataFrame, update the network with new nodes,
    calculate node positions, and analyze the network.
    """

    def __init__(self, df=None, domain_dir="domain", domain_file_name="domain", main_url_list=None, pos_kwargs={}):
        """
        Initialize domain graph object with empty weight.
        If df is provided, create a domain network based on the urls in df['url_node'].
        If no df is provided, read the domain network from file.
        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame containing the network data, must contain 'url_node' column.
        domain_dir : str, optional
            Directory where the domain network file is stored or will be saved.
        domain_file_name : str, optional
            Name of the domain network file (without extension).
        main_url_list : list, optional.
            List of main URLs to be highlighted in the network.
        """

        self.df = df
        self.domain_dir = domain_dir
        self.domain_file_name = domain_file_name
        self.main_url_list = main_url_list
        self.domain = nx.DiGraph()
        self.current_navigation_network = nx.DiGraph()

        # set max weight to deal with inverse weight
        self.max_path_weight = None

        # check if needed columns are in df 
        if df is not None:
            if "url_path" not in df.columns:
                raise ValueError("DataFrame must contain 'url_path' as column")
            if "chrono" not in df.columns:
                raise ValueError("DataFrame must contain 'chrono' as column")
            if "url_node" not in df.columns:
                print("WARNING: DataFrame does not contain 'url_node' column, using 'url_path' to create it")
                df["url_node"] = df["url_path"].apply(lambda row: [xx.split("/")[-1] for xx in row])
        
        try:
            self.domain = nx.read_gml(f"{domain_dir}/{domain_file_name}.gml")
            print(f"Domain network loaded from {domain_dir}/{domain_file_name}.gml")
        except FileNotFoundError:
            if df is None:
                raise ValueError("No domain network found and no DataFrame provided to create one.")


        # df provided to update domain
        # ----------------------------------------------------------------------

        if df is not None:
            if not "url_path" in list(df.keys()):
                raise ValueError("DataFrame must contain 'url_path' as column")

            # define max path weight to deal with inverse weight in networkx
            self.df = df     
            self.max_path_weight = len(df)

            # add new nodes to domain
            self.update_domain(df)

            # create a navigation graph
            self.create_navigation_graph(df)
        
        # get node position using domain network
        print("Calculating node positions for domain network...")
        self.pos = self.update_pos(**pos_kwargs)


    # functions to update or change domain
    ####################################################################################################

    def update_pos(self, method="energy", scale=10, **kwargs):
        """
        using Fruchterman-Reingold force-directed algorithm to calculate nodes position using only the domain edges
        """
        pos = nx.spring_layout(self.domain, method=method, scale=scale, **kwargs)
        self.pos = pos.copy()
        return pos

    def update_domain(self, df):
        """
        Build an undirected domain graph based on structure of URLs.
        Example: 'home/information' implies edge between 'home' and 'information'.
        """
        if self.domain is None:
            self.domain = nx.DiGraph()
            
        for path_list in df['url_path']:
            for url in path_list:
                tokens = [t for t in url.strip("/").split("/") if t]
                for i in range(len(tokens) - 1):
                    u, v = tokens[i], tokens[i+1]
                    self.domain.add_edge(u, v)
        
        # add time attributes to domain nodes
        self.reset_nodes_att(G=self.domain)
        self.save_domain()

    def save_domain(self, path=None):
        """
        Save the current domain graph to a GML file.
        If path is not provided, use the default domain directory and file name.
        """
        if path is None:
            path = f"{self.domain_dir}/{self.domain_file_name}.gml"
        nx.write_gml(self.domain, path, stringizer=None)
        print(f"Domain network saved to {path}")
    
    def domain_from_url_list(self, url_list):
        """
        Create a domain network from a list of URLs.
        Each URL is split into its components, and edges are created between consecutive components.
        """
        domain = nx.DiGraph()
        for url in url_list:
            tokens = [t for t in url.strip("/").split("/") if t]
            for i in range(len(tokens) - 1):
                u, v = tokens[i], tokens[i+1]
                domain.add_edge(u, v)
        self.domain = domain
        # add time attributes to domain nodes
        self.reset_nodes_att(G=self.domain)
        self.save_domain()
        return domain


    # Creation WEB_network
    ####################################################################################################

    def create_navigation_graph(self, df=None):
        """
        create a new network object new_G based on list of urls
                full_path: ["home", "forum", "tour", "tour-bla-bla-bla"] 
                create 3 edges:
                    home -> forum
                    forum -> tour
                    tour -> tour-bla-bla-bla
        
        The edges A->B attribute "weight" is incremented for each session with entry_page = ep
        that goes through A and B. 

        update domain if new nodes are found
        """

        print("Creating navigation graph from DataFrame...")

        if df is None:
            df = self.df.copy()
        
        if "url_node" not in df.columns:
            print("WARNING: DataFrame does not contain 'url_node' column, using 'url_path' to create it")
            # take last non-empty part of url_path as node
            df["url_node"] = df["url_path"].apply(lambda row: [xx.split("/")[-1] for xx in row])       
        
        for _, row in df.iterrows():
            path = row['url_node']
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                if u != v:
                    if self.current_navigation_network.has_edge(u, v):
                        self.current_navigation_network[u][v]['weight'] += 1
                    else:
                        self.current_navigation_network.add_edge(u, v, weight=1)

    
    def add_time_weight(self, df=None, G=None):
        """
        Return graph object new_G with time attributes for the nodes updated from G 
        """

        if G is None:
            G = self.current_navigation_network

        if df is None:
            df = self.df.copy()

        if "url_node" not in df.columns:
            print("WARNING: DataFrame does not contain 'url_node' column, using 'url_path' to create it")
            df["url_node"] = df["url_path"].apply(lambda row: [xx.split("/")[-1] for xx in row])  

        # Accumulate time per node
        tot_time_dict = defaultdict(list)
        # relative time wrt to all session time
        rel_time_dict = defaultdict(list)

        for _, row in df.iterrows():
            node = row["url_node"]
            times = row["chrono"]

            for page, t in zip(node, times):
                tot_time_dict[page].append(t)
                rel_time_dict[page].append(t / sum(times) if sum(times) > 0 else 0)

        # Set node attributes
        for node in G.nodes:
            times = tot_time_dict.get(node, [])
            rel_times = rel_time_dict.get(node, [])
            if times:
                G.nodes[node]["avg_abs_time"] = float(np.mean(times))
                G.nodes[node]["total_time"] = float(np.sum(times))
                G.nodes[node]["std_abs_time"] = float(np.std(times))
                # relative time
                G.nodes[node]["avg_rel_time"] = float(np.mean(rel_times))
                G.nodes[node]["std_rel_time"] = float(np.std(rel_times))

            else:
                G.nodes[node]["avg_abs_time"] = 0.0
                G.nodes[node]["avg_rel_time"] = 0.0
                G.nodes[node]["total_time"] = 0.0
                G.nodes[node]["std_abs_time"] = 0.0
                G.nodes[node]["std_rel_time"] = 0.0
        
    def remove_edges(self, G=None, thres=0):
        """
        remove edges with weight =< thresh in G
        Note: unused edges (weiight = 0) arise from coupling between domain and network G
        """
        
        if G is None:
            G = self.current_navigation_network

        edges_to_remove = [(u,v) for u,v in G.edges() if G[u][v]["weight"] >= self.max_path_weight-thres]
        G.remove_edges_from(edges_to_remove)


    def reset_nodes_att(self, G=None):
        """
        reset node attributes to 0 for avg_rel_time, avg_abs_time, total_time and std_time
        """

        if G is None:
            G = self.current_navigation_network

        nx.set_node_attributes(G, values = 0, name = "avg_rel_time") 
        nx.set_node_attributes(G, values = 0, name = "avg_abs_time")
        nx.set_node_attributes(G, values = 0, name = "total_time")
        nx.set_node_attributes(G, values = 0, name = "std_abs_time")
        nx.set_node_attributes(G, values = 0, name = "std_rel_time")

        
    def reset_edge_att(self, G=None, ini_weight=None):
        """
        reset edge attributes to 0 for weight and group
        """

        if G is None:
            G = self.current_navigation_network
        if ini_weight is None:
            ini_weight = 0

        nx.set_edge_attributes(G, values = ini_weight, name = "weight")
        nx.set_edge_attributes(G, values = None, name = "group")


    # Analysis of network
    ####################################################################################################

    def info_edge(self, G=None, shift=None):
        """
        returns a df with edges as index and weigths
        """

        if shift is None:
            shift = 0
        if G is None:
            G=self.current_navigation_network

        edges = nx.edges(G).keys()
        weight_list = list(nx.get_edge_attributes(G,"weight").values())
        weight_list_shifted = abs(np.asarray(weight_list)-shift)
        # check if edges is part of the domain
        group_list = ["domain" if self.domain.has_edge(e[0], e[1]) else "network" for e in edges]
                
        df = pd.DataFrame(index=pd.MultiIndex.from_tuples(list(edges)), data={"weight":weight_list_shifted, "group":group_list})

        # sort by weight
        df = df.sort_values("weight", ascending=False)

        return df


    def info_node(self, sort_by="total_time"):
        """
        returns a df with nodes as index and attributes
        """

        G = self.current_navigation_network

        node = list(nx.nodes(G))
        col1 = nx.get_node_attributes(G, "total_time").values()
        col2 = nx.get_node_attributes(G, "avg_abs_time").values()
        col3 = nx.get_node_attributes(G, "avg_rel_time").values()
        col4 = nx.get_node_attributes(G, "std_abs_time").values()
        col5 = nx.get_node_attributes(G, "std_rel_time").values()

        df_tmp = pd.DataFrame({"nodes":node, "total_time":col1, "avg_abs_time":col2,"avg_rel_time":col3, "std_abs_time":col4, "std_rel_time":col5})
        df_tmp = df_tmp.sort_values(sort_by, ascending=False)
        df_tmp.set_index("nodes", inplace=True)
        df_tmp = df_tmp.round({"total_time":2, "avg_abs_time":2,"avg_rel_time":2, "std_abs_time":2, "std_rel_time":2})

        return df_tmp


    def get_shortest_k_path(self, entry_page, exit_page, G=None, C=None, shift=None, k=10):
        """
        return the k shortest path form entry_page to exit_page passing by page C
        using entry_page_weight attribute as weight
        """

        if G is None:
            G=self.current_navigation_network.copy()
        
        # remove unused edges
        self.remove_edges(G, thres=0)

        return get_shortest_k_path(entry_page, exit_page, "weight", G, C, shift=shift, k=k)


    def get_all_paths(self, entry_page, exit_page, G=None, C=None, cutoff=10):
        """
        return and iterator object with all possible paths from entry_page to exit_page
        going through page C
        cutoff = max path length
        """

        if G is None:
            G=self.current_navigation_network

        G_new = G.copy()

        # remove unused edges
        self.remove_edges(G_new, thres=0)

        return get_all_paths(entry_page, exit_page, G_new, C, cutoff=cutoff)

    def get_flow(self, B, G=None, shift=None):
        """
        number of session that enter and exit at node B
        """

        if G is None:
            G = self.current_navigation_network
        
        if shift is None:
            shift = self.max_path_weight

        return get_flow(G, B, "weight", shift=shift)

    def get_flow_nodes(self, G=None, shift=None):
        """
        returns flow of all nodes (as index) sorted by absolute value of flow difference
        """
        if G is None:
            G = self.current_navigation_network
        
        flow_in = []
        flow_out = []
        for n in G.nodes():
            i, o = self.get_flow(n, G=G, shift=shift)
            flow_in.append(i)
            flow_out.append(o)
        
        df_tmp = pd.DataFrame({"nodes":G.nodes(), "flow in":flow_in, "flow out": flow_out})
        df_tmp.set_index("nodes", inplace=True)
        df_tmp["flow"] = df_tmp["flow in"] - df_tmp["flow out"] 
        df_tmp = df_tmp.sort_values(by="flow", key=abs, ascending=False)

        return  df_tmp


    # Plotting: helpers for plotly and dashboards
    ####################################################################################################

    def width_from_weight(self, min_w, max_w, shift=None, thresh=None):
        """
        return a dataframe with edges, shifted weight and associated width
        max_w and min_w are the max and min width, respectively
        """

        df_tmp = self.info_edge(shift=shift)

        # keep only network edges within threshold
        if thresh is not None:
            
            mask = ((thresh[0] >= df_tmp["weight"]) | (df_tmp["weight"] > thresh[1])) & (df_tmp["group"]=="network")
            df_tmp = df_tmp[~mask]

            # replace weight_domain not between threshold by 0
            df_tmp.loc[((df_tmp["weight"] > thresh[1]) | (df_tmp["weight"] <= thresh[0])) & (df_tmp["group"]=="domain"), 'weight'] = 0

        # linear relation between weight and width and final width between min_w and max_w
        if df_tmp["weight"].max() == 0:
            print("WARNING: all edges have 0 weight. Define widths as min_w")
            df_tmp["widths"] = min_w * np.ones(len(df_tmp))
        else:
            # calculate linear constant with fallback if all weight are the same
            if df_tmp["weight"].max() == df_tmp["weight"].min():
                deno = 1
            else:
                deno = df_tmp["weight"].max() - df_tmp["weight"].min()
            k = (max_w - min_w) / deno
            df_tmp["widths"] = min_w + k * df_tmp["weight"]
            # normalize widths so that min width is min_w and max width is max_w
            df_tmp["widths"] = (df_tmp["widths"] - df_tmp["widths"].min()) / (deno) * (max_w - min_w) + min_w
        
        return df_tmp
    
    def size_from_time(self, min_size, max_size, time_att="total_time"):
        """
        return a dataframe with nodes, time attributes and size associated to attribute time
        max_size and min_size are the max and min node size, respectively
        """

        df_tmp = self.info_node()

        # define linear constant
        time_max = df_tmp[time_att].max()
        if time_max == 0:
            print("WARNING: all nodes have 0 time spent")
            df_tmp["node_size"] = np.zeros(len(df_tmp))
        else:
            k = (max_size - min_size)/time_max
            # calculate size
            df_tmp["node_size"] = min_size + k*df_tmp[time_att]
        
        return df_tmp
    
    def plot(self, time_att="total_time", thresh=5, node_size=[1, 10], edge_size=[0.1, 1], label_list=["home", "login"], only_domain=False):
        """
        Simple plot function for testing
        """
        
        nodes_df = self.size_from_time(node_size[0], node_size[1], time_att)
        edges_df = self.width_from_weight(edge_size[1], edge_size[0])
        edges_df = edges_df[edges_df["weight"] > thresh]

        # draw nodes
        nx.draw_networkx_nodes(self.domain, pos=self.pos,
                            nodelist=nodes_df.index,
                            node_size=nodes_df["node_size"],
                            node_color='dodgerblue',
                            alpha=0.8)

        # draw edges for domain
        nx.draw_networkx_edges(self.domain, pos=self.pos,
                            edgelist = self.domain.edges,
                            width=0.5, arrows=False,
                            edge_color="dodgerblue",
                            alpha=0.7)

        # draw edges for network
        if only_domain is False:
            nx.draw_networkx_edges(self.current_navigation_network, pos=self.pos,
                                edgelist =edges_df.index,
                                width=edges_df["widths"],
                                edge_color="red",
                                alpha=1,
                                arrowsize=0.5)

        # draw labels
        nodelabel = dict([(n, n) if (n in label_list) else (n, "") for n in nodes_df.index])
        nx.draw_networkx_labels(self.domain, pos=self.pos,
                                labels=nodelabel,
                                font_color='black',
                                font_size=12,
                                font_weight=1000
                                )
        #plt.box(False)
        plt.show()

    def export_gml(self, path="GML/network.gml"):
        nx.write_gml(self.current_navigation_network, path, stringizer=None)

    def to_cytoscape_json(self, node_size_range=(5, 20), edge_width_range=(0.1, 5), pos_kwargs={}):
        """
        Export full graph (domain + navigation edges) for Cytoscape.
        """

        domain_graph = self.domain
        navigation_graph = self.current_navigation_network
        pos = self.update_pos(**pos_kwargs)

        # Compute node sizes and edge widths for navigation edges
        node_sizes_df = self.size_from_time(node_size_range[0], node_size_range[1])
        edge_widths_df = self.width_from_weight(edge_width_range[0], edge_width_range[1])
        node_flow_df = self.get_flow_nodes()

        # Create a unified graph
        G = nx.DiGraph()
        G.add_nodes_from(navigation_graph.nodes(data=True))
        G.add_edges_from(navigation_graph.edges(data=True))

        # Add all nodes from domain graph not present in navigation
        for node in domain_graph.nodes:
            if node not in G:
                G.add_node(node, size=node_size_range[0])

        # Add all domain edges not present in navigation
        for u, v in domain_graph.edges:
            if not G.has_edge(u, v):
                G.add_edge(u, v, group="domain", weight=1e-6, widths=edge_width_range[0])

        # Export to cytoscape
        data = json_graph.cytoscape_data(G)

        # add positions, flow, time att and sizes
        for node in data["elements"]["nodes"]:
            node_id = node["data"]["id"]
            if node_id in pos:
                x, y = pos[node_id]
                node["position"] = {"x": float(x), "y": float(y)}
            else:
                raise ValueError(f"Node {node_id} not found in position dictionary.")
            node["data"]["size"] = safe_float(node_sizes_df.get("node_size", node_size_range[0]).get(node_id), node_size_range[0])
            # Add tooltip info
            avg = safe_float(node_sizes_df.get("avg_abs_time").get(node_id, 0), 0)
            total = safe_float(node_sizes_df.get("total_time").get(node_id, 0), 0)
            avgrel = safe_float(node_sizes_df.get("avg_rel_time", 0).get(node_id, 0), 0)
            flow = safe_float(node_flow_df.get("flow", 0).get(node_id, 0), 0)
            node["data"]["AvgAbs"] = avg
            node["data"]["Total"] = total
            node["data"]["AvgRel"] = avgrel
            node["data"]["flow"] = flow
            # for hover tooltip
            node["data"]["title"] = (
                        f"<strong>{node_id}</strong><br>"
                        "<br>"
                        f"Time spent in seconds:<br>"
                        f"AvgAbs: {avg:.3f}<br>"
                        f"Total: {total:.3f}<br>"
                        f"AvgRel: {avgrel:.3f}<br>"
                        # new line
                        "<br>"
                        f"Flow: {flow}</br>"
                        )


        # add edge and arrow widths and groups
        for edge in data["elements"]["edges"]:
            source = edge["data"]["source"]
            target = edge["data"]["target"]
            group = edge_widths_df.get("group", "domain").get((source, target), "domain")
            edge["data"]["group"] = group
            width = edge_widths_df.get("widths", edge_width_range[0]).get((source, target), edge_width_range[0])
            edge["data"]["width"] = safe_float(width, default=edge_width_range[0])
            weight = safe_float(edge_widths_df.get("weight").get((source, target), 0), 0)
            edge["data"]["title"] = (
                                    f"<strong>{source} → {target}</strong><br>"
                                    f"Weight: {weight}"
                                )

        return data






