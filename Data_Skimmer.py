import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

def average_arrays(array_list):
    """Averages a list of arrays."""
    max_len = max(len(arr) for arr in array_list)
    padded_arrays = [np.pad(arr, (0, max_len - len(arr)), 'edge') for arr in array_list]
    return np.mean(padded_arrays, axis=0)

def get_fitness_over_time(df, target_length=13):
    """Ensures that the fitness values have a consistent length."""
    fitness_values = df['Fitness'].tolist()
    if len(fitness_values) >= target_length:
        return fitness_values[:target_length]
    else:
        return fitness_values + [fitness_values[-1]] * (target_length - len(fitness_values))

def process_file(file_path):
    df = pd.read_csv(file_path)
    average_plato_score = df['Plato'].mean()
    average_best_fitness = df['Final Fitness'].mean()
    best_fitness = df['Final Fitness'].min()
    
    fitness_over_time_list = df['Fitness List'].apply(ast.literal_eval).apply(lambda x: get_fitness_over_time(pd.DataFrame({'Fitness': x}))).tolist()
    if fitness_over_time_list:
        average_fitness_over_time = average_arrays(fitness_over_time_list)
    else:
        average_fitness_over_time = None

    return average_plato_score, average_best_fitness, best_fitness, average_fitness_over_time

def main():
    # Define the directory containing the CSV files
    data_dir = '/Users/noahsmith/Documents/AROTTGGP (Summer Research Project 2024)/AROTTGGP/Data'
    
    # Define the rows and columns
    graph_types = ['complete', 'complete_bipartite']
    algorithms = ['Random_Hill_Climb', 'Simulated_Annealing', 'Genetic']
    nodes = range(5, 16)
    columns = [0, 1, 2, 4, 8, 16, 32, 64, 128, 258, 512, 1024, 'Plateau']
    
    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=columns)
    
    # Iterate through the combinations of algorithms, graph types, and nodes
    for graph_type in graph_types:
        for algorithm_name in algorithms:
            for node in nodes:
                file_name = f"{algorithm_name}_{graph_type}_{node}_Best_Results_Summary.csv"
                file_path = os.path.join(data_dir, file_name)
                
                if os.path.exists(file_path):
                    average_plato_score, average_best_fitness, best_fitness, average_fitness_over_time = process_file(file_path)
                    row_name = f"{algorithm_name} ({graph_type}) {node}"
                    row_data = average_fitness_over_time.tolist() if average_fitness_over_time is not None else [None] * (len(columns) - 1)
                    row_data.append(average_plato_score)
                    results_df.loc[row_name] = row_data

    # Save the results to a CSV file
    results_df.to_csv('aggregated_results.csv', index=True)
    
    # Plotting (optional)
    plt.figure(figsize=(10, 6))
    for row in results_df.index:
        plt.plot(results_df.columns[:-1], results_df.loc[row][:-1], marker='o', label=row)
    plt.xticks(ticks=range(len(columns)-1), labels=columns[:-1])
    plt.xlabel('Iterations')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness Across Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
