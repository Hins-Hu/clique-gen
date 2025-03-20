import itertools
from shapely.geometry import Point, MultiPoint
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobipy import Model, GRB, quicksum
from scipy.spatial import Delaunay
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Point
from copy import deepcopy

def generate_delaunary_graph(num_nodes, width, one_way_prob = 0.2, edge_ratio = 0.8):

    """
    Generate a random directed graph with a road-like structure using Delaunay triangulation.
    Parameters
    ----------
    num_nodes : int
        Number of intersections (nodes).
    width : float
        Width of the unit square.
    one_way_prob : float
        Probability that a road is one-way.
    edge_ratio : float
        Probability of keeping additional edges from Delaunay triangulation.
    Returns
    -------
    H : networkx.DiGraph
        Directed graph representing the road-like structure
    """


    # Generate random node positions in a unit square
    points = np.random.rand(num_nodes, 2) * width

    # Compute Delaunay triangulation (planar by definition)
    tri = Delaunay(points)
    H = nx.DiGraph()  # Directed graph

    # Add nodes with positions
    for i, (x, y) in enumerate(points):
        H.add_node(i, pos=(x, y))

    # Add edges from Delaunay triangulation (planar)
    for simplex in tri.simplices:
        for i in range(3):
            u, v = int(simplex[i]), int(simplex[(i + 1) % 3])  # Convert to Python int
            dist = np.linalg.norm(points[u] - points[v])

            # Randomly assign one-way or two-way road
            if np.random.rand() < one_way_prob:  # One-way road
                H.add_edge(u, v, weight=dist)
            else:  # Two-way road
                H.add_edge(u, v, weight=dist)
                H.add_edge(v, u, weight=dist)  # Add reverse edge for two-way road

    # Sparsify, but keep some extra edges to make it denser
    edges = list(H.edges())
    np.random.shuffle(edges)
    for u, v in edges:
        if np.random.rand() > edge_ratio:  # Randomly remove edges with some probability
            weight = H[u][v]['weight']  # Save the weight before removing the edge
            H.remove_edge(u, v)
            if not nx.is_strongly_connected(H):  # Ensure the graph remains weakly connected
                H.add_edge(u, v, weight=weight)  # Restore the edge with its weight

    return H

def compute_shortest_path_distances(H, power = 2):

    """
    Compute shortest path distances raised to the power.
    Parameters:
    -----------
    H : networkx.DiGraph
        Directed graph.
    power : int
        Power to raise the shortest path distances.

    Returns:
    --------
    cost_dict : dict
        Dictionary of shortest path distances raised to the power
    """


    if H is not None:

        shortest_distances = dict(nx.all_pairs_dijkstra_path_length(H, weight='weight'))
        cost_dict = {key: {inner_key: value**power for inner_key, value in inner_dict.items()}
                     for key, inner_dict in shortest_distances.items()}

    return cost_dict

def generate_od_demand_mixed(H, cluster_centers, cluster_radius, is_cluster=True, is_uniform=True, cluster_factor=1):

    """
    Generates origin-destination demand within multiple overlapping clusters and assigns random demand outside the clusters.

    Parameters:
    - H: NetworkX graph with 'pos' node attributes
    - cluster_centers: List of cluster centers in [(x1, y1), (x2, y2), ...]
    - cluster_radius: List of radii for each cluster
    - outside_demand: Boolean indicating whether to generate uniform demand outside clusters

    Returns:
    - demand: Dictionary containing intra-cluster and external demand.
    - cluster_map: Dictionary mapping each node to its assigned clusters.
    """

    nodes = list(H.nodes())
    pos = {n: np.array(H.nodes[n]['pos']) for n in nodes}
    demand = defaultdict(lambda: defaultdict(int))

    if is_cluster:
        cluster_map = defaultdict(list)

        # Assign nodes to multiple clusters
        for n in nodes:
            for c, r in zip(cluster_centers, cluster_radius):
                if np.linalg.norm(pos[n] - np.array(c)) <= r:
                    cluster_map[n].append(tuple(c))

        # Create intra-cluster demand for each cluster separately
        clusters = {tuple(center): [] for center in cluster_centers}
        for n, centers in cluster_map.items():
            for center in centers:
                clusters[center].append(n)

        for cluster_nodes in clusters.values():
            for i in cluster_nodes:
                for j in cluster_nodes:
                    if i != j:
                        demand[i][j] += cluster_factor * np.random.rand()  # Accumulate demand for nodes in multiple clusters

    # Assign uniform demand outside clusters
    if is_uniform:
        for i in nodes:
            for j in nodes:
                if i != j:
                    demand[i][j] += np.random.rand()  # Lower magnitude than intra-cluster demand

    return demand, cluster_map

def visualize_demand_pattern(H, demand, filename="output/demand_pattern.png"):
    """
    Visualize the demand pattern on a graph.

    Parameters:
    - H: NetworkX graph with 'pos' node attributes
    - demand: Dictionary containing demand between nodes
    - filename: Output filename for the plot
    """
    pos = nx.get_node_attributes(H, 'pos')
    nodes = list(H.nodes())

    total_demand = {n: 0 for n in nodes}
    for i in demand:
        for j in demand[i]:
            total_demand[i] += demand[i][j]  # Outgoing demand
            total_demand[j] += demand[i][j]  # Incoming demand

    max_demand = max(total_demand.values(), default=1)
    node_colors = [total_demand[n] / max_demand for n in nodes]

    x_vals = [pos[n][0] for n in nodes]
    y_vals = [pos[n][1] for n in nodes]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(x_vals, y_vals, c=node_colors, cmap='viridis', s=50, alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Demand Level')
    ax.set_title("Demand Heatmap")

    plt.savefig(filename)
    plt.close()


# precompute the dictionary of the origins and destinations
def pre_computations(no_of_trips, data):
    origins = dict()
    dest = dict()
    # benefit = dict()
    for trip in range(no_of_trips):
        o = (data.loc[trip, 'origin'])
        d = (data.loc[trip, 'dest'])
        origins[trip] = Point(o)
        dest[trip] = Point(d)
    return origins, dest


def visualize_optimal_zones(H, zones, filename="output/zones_plot.png"):

    """
    Visualize the selected zones on a graph.

    Parameters:
    - H: NetworkX graph with 'pos' node attributes
    - zones: List of selected zones, where each zone is a list of node indices
    - filename: Output filename for the plot
    """


    # Graph node positions
    pos = nx.get_node_attributes(H, 'pos')

    # Generate a dynamic list of colors
    cmap = plt.get_cmap("tab10")
    zone_colors = [cmap(i) for i in range(len(zones))]

    # Identify all nodes that are not part of any zone using H.nodes()
    all_zone_nodes = set(node for zone in zones for node in zone)
    non_zone_nodes = set(H.nodes()) - all_zone_nodes

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the nodes not part of any zone in gray (from H.nodes())
    non_zone_x_vals = [pos[node][0] for node in non_zone_nodes]
    non_zone_y_vals = [pos[node][1] for node in non_zone_nodes]
    ax.scatter(non_zone_x_vals, non_zone_y_vals, color='gray', s=50, label="Non-Zone Nodes", edgecolor='black', alpha=0.5)

    # Plot each zone with a different color
    for i, zone in enumerate(zones):
        zone_nodes = [node for node in zone if node in pos]
        if zone_nodes:  # Only plot if there are nodes in the current zone
            x_vals_zone = [pos[node][0] for node in zone_nodes]
            y_vals_zone = [pos[node][1] for node in zone_nodes]
            ax.scatter(x_vals_zone, y_vals_zone, color=zone_colors[i], s=100, label=f"Zone {i+1}", edgecolor='black', alpha=0.7)

    # Title and legend
    ax.set_title("Selected Zones Visualization")
    ax.legend()

    # Save the plot instead of showing it
    plt.savefig(filename)
    plt.close()


# Check whether two nodes are close enough
def two_nodes_close(pair, max_diameter, distances):
    return (distances[pair[0]][pair[1]] <= max_diameter) and (distances[pair[1]][pair[0]] <= max_diameter)


# Check whether a node is close to a clique (i.e., a set of nodes close enough)
def is_node_close_to_clique(node, clique, max_diameter, distances):
    for n in clique:
        if not two_nodes_close((node, n), max_diameter, distances):
            return False
    return True


#*[Hins] I just realized that this does not exclude all cliques that are inside a convex hull of another clique
#*[Hins] because the order of visiting clique is arbitrary

#*[Hins] Another observation, when distances matrix is not symmetric (e.g. one-way roads exist),
#*[Hins] a node inside a convex hull of a clique geographically may not be added to that clique
def convex_hull_extend_on_map(clique, nodes, pos, pairwise_map, visited_cliques):
    
    # a tag
    is_extended = False
    
    # pos is calculated before hand
    points = []
    for node in clique:
        points.append(pos[node])

    # create the convex hull
    convex_hull = MultiPoint(points).convex_hull
    # create the set that will have all the points that have to be grouped together
    extend_set = deepcopy(clique)
    # checks for all the possible trips that could be encapsulated
    for node in set(nodes) - clique:
        
        # Filter out trips that are not shaerable
        is_node_extendable = True
        for item in clique:
            if not pairwise_map[tuple(sorted((node, item)))]:
                is_node_extendable = False
                break
        if not is_node_extendable:
            continue

        # this means that the trip is encapsualted by the hull
        if convex_hull.contains(pos[node]):
            extend_set.add(node)
    if extend_set != clique:
        is_extended = True

    return is_extended, extend_set
    

def clique_generator_on_map(H, max_diameter, distances, connectivity_threshold):

    # Initialization
    nodes = list(H.nodes())
    shared_map = defaultdict(list)
    shared_map[1] = [{n} for n in nodes]
    visited_cliques = defaultdict(int)
    card = 2
    max_card = 2
    
    # Pre-computation for efficiency
    pos = {n: Point(H.nodes[n]['pos']) for n in nodes}
    pairwise_map = defaultdict(int)
    for pair in itertools.combinations(nodes, 2):
        pair = tuple(sorted(pair))
        pairwise_map[pair] = two_nodes_close(pair, max_diameter, distances)
    
    while True:
        # Termination condition
        if card > max_card + 1:
            break

        prev_list = shared_map[card - 1]
        for clique in prev_list:
            for node in set(nodes) - clique:
                
                # Check if the new clique is visited already
                clique_key = tuple(sorted(clique | {node}))
                if visited_cliques[clique_key] == 1:
                    continue

                # Check if the new clique is valid when a new node is added
                if is_node_close_to_clique(node, clique, max_diameter, distances):
                        
                    is_extended, extended_clique = convex_hull_extend_on_map(clique | {node}, nodes, pos, pairwise_map, visited_cliques)
                    if is_extended:
                        extended_key = tuple(sorted(extended_clique))
                        if not visited_cliques[extended_key]:
                            new_card = len(extended_clique)
                            shared_map[new_card].append(extended_clique)
                            visited_cliques[extended_key] = 1
                            max_card = max(max_card, new_card)
                    else:
                        shared_map[card].append(clique | {node})
                        clique_key = tuple(sorted(clique | {node}))
                        visited_cliques[clique_key] = 1
                        max_card = max(max_card, card)
                
                else:
                    # We also mark the invalid cliques as visited to avoid rechecking
                    clique_key = tuple(sorted(clique | {node}))
                    visited_cliques[clique_key] = 1
                    
        print(f"Cardinality {card} has {len(shared_map[card])} cliques")
        print(f"Cardinality {card} complete")
        
        # Increase the cardinality
        card += 1
        
    # Extract the final list of cliques
    clique_list = []
    for cliques in shared_map.values():
        for clique in cliques:
            clique_list.append(tuple(sorted(clique)))
    
    #!Debugging
    seen = set()
    count = 0
    for clique in clique_list:
        key = tuple(sorted(clique))
        if key in seen:
            count += 1
        else:
            seen.add(key)
    print(f"{count} duplicates found")


    return clique_list, max_card

def solve_ILP(H, clique_list, demand, num_zones):
    
    # Initialization
    V = list(H.nodes())
        
    # Pre-computation to extract the useful pairs of nodes
    valid_pairs = set()
    pair_2_clique = defaultdict(list)
    for clique in clique_list:
        for pair in itertools.permutations(clique, 2):
            valid_pairs.add(pair)
            pair_2_clique[pair].append(clique)
    
    
    # Machine-specific Gurobi license should be set up in advance
    params = {
    "WLSACCESSID": '30df8a74-359b-436f-ba5e-7160dd5221c3',
    "WLSSECRET": 'afb8bd5f-4cc0-44c2-b3e3-d85cd0f1c5b8',
    "LICENSEID": 2434804,
    }
    env = gp.Env(params=params)
    
    # Add variables
    model = gp.Model(env=env)
    x = {}
    for clique in clique_list:
        x[clique] = model.addVar(vtype = GRB.BINARY, name=f"x_{clique}")
    #TODO DEBUG: we observe duplicate cliques here, which is not supposed to happen
    #TODO It does not affect the final result. Reserve 4 later debugging
    # x = model.addVars(clique_list, vtype=GRB.BINARY, name=f"x")
    w = {}
    for i, j in valid_pairs:
        w[i, j] = model.addVar(vtype=GRB.BINARY, name=f"w_{i}_{j}")
    
    model.setObjective(quicksum(demand[i][j] * w[i, j] for i, j in valid_pairs), GRB.MAXIMIZE)
    
    # Constraints: linking x and w
    for i, j in valid_pairs:
        model.addConstr(
            w[i, j] <= quicksum(x[clique] for clique in pair_2_clique[(i, j)]),
            name=f"link_{i}_{j}"
        )

    # Constraint: Max number of cliques
    model.addConstr(
        gp.quicksum(x[clique] for clique in clique_list) <= num_zones
    )
    
    # Optimization
    model.optimize()    
    if model.status == (GRB.OPTIMAL or GRB.SUBOPTIMAL):
        print("The max coverage ILP was solved to optimality.")
        selected_zones = [clique for clique in clique_list if x[clique].X > 0.5]
        print("Selected candidate zones:", selected_zones)

    else:
        print("Fail to solve the max coverage ILP.")
        
    return selected_zones
    
    

def solve_ILP_non_overlap(clique_list, demand, num_zones):
    
    benefit = {}
    for clique in clique_list:
        benefit[clique] = sum([demand[pair[0]][pair[1]] for pair in itertools.permutations(clique, 2)])

    # Machine-specific Gurobi license should be set up in advance
    env = gp.Env()
    
    # Create a model
    model = gp.Model(env=env)
    
    # create decision variables - we make one to indicate if the clique is chosen or not
    y = {}
    
    # stores whether or not it is a 1 or 0 basically (selected or not)
    for clique in clique_list:
        y[clique] = model.addVar(vtype = GRB.BINARY, obj = benefit[clique], name=f"y_{clique}")
    
    # objective function
    model.setObjective(gp.quicksum(benefit[clique] * y[clique] for clique in clique_list), GRB.MAXIMIZE)
    
    # constraints
    # ensuring that the cliques selected do not overlap
    nodes = set()
    for clique in clique_list:
        for n in clique:
            nodes.add(n)
    # create the constraint that one node can only be selected at one time (non-overlapping)
    for n in nodes:
    # we only need to add the constraint if it is acc in the clique
        model.addConstr(
            gp.quicksum(y[clique] for clique in clique_list if n in clique) <= 1
        )

    # adding the constraint that we need to select less than m
    model.addConstr(
        gp.quicksum(y[clique] for clique in clique_list) <= num_zones
    )

    model.optimize()
    if model.status == GRB.OPTIMAL:
        selected_zones = [clique for clique in clique_list if y[clique].X > 0.5]
        print("Selected candidate zones:", selected_zones)
        
    return selected_zones


def visualize_graph(G, node_size=300, font_size=12, edge_labels=False, layout='spring', figsize=(8, 6), filename='output/base_graph.png'):
    """
    Visualizes a NetworkX graph and saves it to a file.

    Parameters:
        G (networkx.Graph): The input graph.
        node_size (int): Size of the nodes.
        font_size (int): Font size for node labels.
        edge_labels (bool): Whether to display edge weights/labels.
        layout (str): Layout algorithm ('spring', 'circular', 'kamada_kawai', 'shell').
        figsize (tuple): Size of the figure.
        filename (str): Path to save the figure.
    """
    layout_functions = {
        'spring': nx.spring_layout,
        'circular': nx.circular_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'shell': nx.shell_layout
    }
    
    if layout not in layout_functions:
        raise ValueError(f"Unsupported layout '{layout}'. Choose from {list(layout_functions.keys())}.")

    pos = layout_functions[layout](G)
    
    plt.figure(figsize=figsize)
    nx.draw(G, pos, with_labels=True, node_size=node_size, font_size=font_size, edge_color='gray', node_color='skyblue')
    
    if edge_labels:
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=font_size)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.close()


def baseline(H, demand, max_diameter, distances, num_zones):
    
    # Initialization
    nodes = set(H.nodes())
    selected_zones = []
    
    # Pre-computation for efficiency
    pairwise_map = defaultdict(int)
    for pair in itertools.combinations(nodes, 2):
        pair = tuple(sorted(pair))
        pairwise_map[pair] = two_nodes_close(pair, max_diameter, distances)
    

    for i in range(num_zones):

        if len(nodes) < 2:
            raise ValueError(f"Not enough nodes to form {num_zones} zones.")

        # Select pair with highest demand        
        pair = max(
            ((demand[o][d] + demand[d][o], o, d) for o, d in itertools.combinations(nodes, 2) if pairwise_map[tuple(sorted((o, d)))]),
            default=None
        )
        if pair is None:
            raise ValueError("No valid pairs found within the distance threshold.")
        
        _, o, d = pair
        zone = {o, d}
        nodes -= {o, d}

        # Dynamically add a node to the current zone that leads to the largest increment in demand served
        while True:
            best_candidate = None
            best_demand = -1

            for n in nodes:
                if is_node_close_to_clique(n, zone, max_diameter, distances):
                    # Compute total demand between n and nodes already in the zone
                    demand_to_zone = sum(demand[n][z] + demand[z][n] for z in zone)
                    if demand_to_zone > best_demand:
                        best_demand = demand_to_zone
                        best_candidate = n

            if best_candidate is None:
                print(f"No more candidate nodes to be added. Complete the expansion for zone {i}")
                break  # no eligible node to add

            zone.add(best_candidate)
            nodes.remove(best_candidate)

        selected_zones.append(zone)
        
    return selected_zones


def calculate_total_demand_served(selected_zones, demand):

    total_demand = 0
    for zone in selected_zones:
        for i in zone:
            for j in zone:
                if i != j:
                    total_demand += demand[i][j]
                    total_demand += demand[j][i]
    return total_demand