import networkx as nx
import random
import os
import pandas as pd
from GraphGenusGenerator import GraphGenusGenerator
from HC_Algorithm import hill_climb
from RHC_Algorithm import random_hill_climb

seed=random.randint(0, 2**32 - 1)
max_iter = 1000
number_of_nodes = 10
complete_graph = False
problem = GraphGenusGenerator().generate(seed=seed, number_of_nodes=number_of_nodes, max_connections_per_node=10, complete_graph=complete_graph)

# Get the adjacency list (which is a graph object)
adj_list = problem.adj_list
graph = nx.Graph(adj_list)

def is_power_of_two(n):
    return (n != 0) and (n & (n - 1)) == 0

def get_best_result(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Sort the DataFrame by fitness and time
    df_sorted = df.sort_values(by=['Fitness', 'Time'])

    # Get the result with the lowest time while still having the lowest fitness
    best_result = df_sorted.iloc[0]

    return best_result

# Run the hill climb function and collect iteration data
# final_fitness, final_state, iteration_data = hill_climb(problem=problem, max_iter=max_iter, seed=seed, record_file="hill_climb_iterations.csv")
final_fitness, final_state, iteration_data = random_hill_climb(problem=problem, max_iters=max_iter, restarts=10, random_state=seed, record_file="random_hill_climb_iterations.csv")


# Convert the iteration data to a DataFrame
df_run_stats = pd.DataFrame(iteration_data)

# Filter the data to include only the iterations with iter = to a power of 2 and the last one
filtered_data = df_run_stats[df_run_stats['Iteration'].apply(is_power_of_two) | (df_run_stats['Iteration'] == df_run_stats['Iteration'].max())]

def get_best_result(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Find the minimum fitness value
    min_fitness = df['Fitness'].min()

    # Filter the DataFrame to include only the rows with the minimum fitness
    min_fitness_df = df[df['Fitness'] == min_fitness]

    # Sort the filtered DataFrame by fitness evaluation time
    min_fitness_df_sorted = min_fitness_df.sort_values(by='FitnessEvalTime')

    # Get the first row (lowest fitness evaluation time among the lowest fitness values)
    best_result = min_fitness_df_sorted.iloc[0]

    return best_result

# Example usage:
file_path = "hill_climb_iterations.csv"  # Change the file path accordingly
best_result = get_best_result(file_path)

# Convert the best result to a DataFrame
df_best_result = pd.DataFrame([best_result])

def save_best_result(best_result, number_of_nodes, complete_graph, seed, file_path="best_results.csv"):
    data = {
        'Number of Nodes': [number_of_nodes],
        'Is Complete': [complete_graph],
        'Final Fitness': [best_result['Fitness']],
        'Time': [best_result['Time']],
        'Seed': [seed],
        'Final State': [best_result['State']]
    }
    df = pd.DataFrame(data)
    # Check if the file exists
    if os.path.exists(file_path):
        # If the file exists, append the new data
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        # If the file doesn't exist, create a new one
        df.to_csv(file_path, index=False)

# Example usage:
file_path = "best_results.csv"  # Change the file path accordingly
save_best_result(best_result, number_of_nodes, complete_graph, seed, file_path)