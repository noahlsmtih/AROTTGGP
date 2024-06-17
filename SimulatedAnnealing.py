import time
import numpy as np
from mlrose_hiive.algorithms.decay import GeomDecay
from mlrose_hiive.algorithms.decay import ArithDecay
from mlrose_hiive.algorithms.decay import ExpDecay

def simulated_annealing(problem, schedule=ExpDecay(), max_attempts=10,
                        max_iters=np.inf, init_state=None, curve=False,
                        fevals=False, random_state=None,
                        state_fitness_callback=None, callback_user_info=None):
    """Use simulated annealing to find the optimum for a given
    optimization problem.
    Parameters
    ----------
    ... (same as before)
    """

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
            or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
        and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    # Initialize problem, time and attempts counter
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    if state_fitness_callback is not None:
        # initial call with base data
        state_fitness_callback(iteration=0,
                               state=problem.get_state(),
                               fitness=problem.get_adjusted_fitness(),
                               fitness_evaluations=problem.fitness_evaluations,
                               user_data=callback_user_info)

    iteration_data = []
    attempts = 0
    iters = 0
    continue_iterating = True
    start_time = time.time()

    while (attempts < max_attempts) and (iters < max_iters):
        temp = schedule.evaluate(iters)
        iters += 1
        problem.current_iteration += 1

        if temp == 0:
            break

        else:
            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor(problem.get_state())
            next_fitness = problem.eval_fitness(next_state)

            # Calculate delta E and change prob
            current_fitness = problem.get_fitness()
            delta_e = next_fitness - current_fitness
            prob = np.exp(delta_e / temp)
            # print(f'{iters} : {current_fitness}')
            # If best neighbor is an improvement or random value is less
            # than prob, move to that state and reset attempts counter
            if (delta_e > 0) or (np.random.uniform() < prob):
                problem.set_state(next_state)
                attempts = 0
            else:
                attempts += 1

        fitness_eval_time = time.time() - start_time

        if curve:
            iteration_data.append({
                'Iteration': iters,
                'Fitness': problem.get_adjusted_fitness(),
                'FitnessEvalTime': fitness_eval_time,
                'Time': time.time() - start_time,
                'State': problem.get_state()
            })

        # invoke callback
        if state_fitness_callback is not None:
            max_attempts_reached = (attempts == max_attempts) or (iters == max_iters) or problem.can_stop()
            continue_iterating = state_fitness_callback(iteration=iters,
                                                        attempt=attempts + 1,
                                                        done=max_attempts_reached,
                                                        state=problem.get_state(),
                                                        fitness=problem.get_adjusted_fitness(),
                                                        fitness_evaluations=problem.fitness_evaluations,
                                                        curve=np.asarray(iteration_data) if curve else None,
                                                        user_data=callback_user_info)

        # break out if requested
        if not continue_iterating:
            break

    best_fitness = problem.get_maximize() * problem.get_fitness()
    best_state = problem.get_state()

    return best_state, best_fitness, iteration_data if curve else None
