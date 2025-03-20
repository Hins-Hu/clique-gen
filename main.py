from utils import generate_delaunary_graph, compute_shortest_path_distances, generate_od_demand_mixed, clique_generator_on_map
from utils import visualize_demand_pattern, visualize_optimal_zones, solve_ILP, visualize_graph, baseline, calculate_total_demand_served
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os
import cProfile
import pstats


def main(): 

  # Parameters
  NUM_ZONES = 4
  MAX_DIAMETER = 2
  ALGO = "clique_generation" # "clique_generation" or "baseline"
  NUM_NODE = 200
  SEED = 42
  np.random.seed(SEED)
  ONE_WAY_PROB = 0.0
  EDGE_RATIO = 0.8
  CENTERS = [(1.8, 6.3), (2.8, 2), (9.5, 3.8)]
  RADIUS = [1, 1, 1]

  # Create an output folder
  if not os.path.exists("output"):
    os.makedirs("output")

  # Generate the graph
  G = generate_delaunary_graph(NUM_NODE, 10, one_way_prob=ONE_WAY_PROB, edge_ratio=EDGE_RATIO)
  visualize_graph(G)

  # Compute shortest path distances
  # Use the original distance but not the square one
  distances = compute_shortest_path_distances(G, power = 1)

  # Generate demand with a mixed distribution (clusters + uniform)
  demand, _ = generate_od_demand_mixed(G, cluster_centers=CENTERS, cluster_radius=RADIUS, cluster_factor=10)
  total_demand = sum(sum(inner.values()) for inner in demand.values())
  print("Total demand:", total_demand)
  visualize_demand_pattern(G, demand, filename="output/demand_pattern.png")

  # Run the CliqueGen + ZoningILP algorithm  
  if ALGO == "clique_generation":
    
    # Phase 1: CliqueGen
    lst, cardinality = clique_generator_on_map(G, MAX_DIAMETER, distances, None)
    
    print(len(lst))
    print("Highest Cardinality", cardinality)
    
    # Phase 2: Solving ZoningILP
    selected_zones = solve_ILP(G, lst, demand, NUM_ZONES)
    print("Selected zones:", selected_zones)
    print("Total demand served:", calculate_total_demand_served(selected_zones, demand))
    visualize_optimal_zones(G, selected_zones, filename="output/optimal_zones.png")
  
  # Run the baseline algorithm
  elif ALGO == "baseline":
    selected_zones = baseline(G, demand, MAX_DIAMETER, distances, NUM_ZONES)
    print("Selected zones:", selected_zones)
    print("Total demand served:", calculate_total_demand_served(selected_zones, demand))
    visualize_optimal_zones(G, selected_zones, filename="output/zones_by_heuristic.png")
    
  else:
    raise ValueError("Invalid algorithm selected. Choose 'clique_generation' or 'baseline'.")
  

if __name__ == "__main__":

  # Uncomment the following lines for profiling
  
  # profiler = cProfile.Profile()
  # profiler.enable()
  try:
    main()
  except MemoryError:
    print("Program terminated due to OOM error.")
  # finally:
  #   profiler.disable()
  #   stats = pstats.Stats(profiler)
  #   stats.strip_dirs()
  #   stats.sort_stats("cumulative")

  #   # Save profiling results to a file
  #   with open("profile.out", "w") as f:
  #       stats.stream = f  # Redirect output to the file
  #       stats.print_stats()

  #   # Print profiling results to stdout (captured in nohup log)
  #   print("Profiling results:")
  #   stats.stream = None  # Reset output to stdout
  #   stats.print_stats(20)  # Print the top 20 functions
