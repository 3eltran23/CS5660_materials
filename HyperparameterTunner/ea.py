
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools


def count_diagonal_attacks(permutation):
    attack_count = 0
    for i in range(len(permutation)):
        for j in range(i + 1, len(permutation)):
            if abs(i - j) == abs(permutation[i] - permutation[j]):
                attack_count += 1
    return attack_count


def fitnessFunction(permutation):
    n = len(permutation)
    total_pairs = n * (n - 1) // 2
    return total_pairs - count_diagonal_attacks(permutation)

## Parent Selection
def select_best_two(
    population,
    tournament_size=5,
    num_parents=2,
):
    tournament = random.sample(
        population,
        k=tournament_size,
    )
    tournament.sort(
        key=lambda individual: individual.fitness,
        reverse=True,
    )
    return tournament[:num_parents]


# Survival selection
def replace_worst(population, offspring):

    population_size = len(population)
    candidates = list(population) + list(offspring)
    return tools.selBest(candidates, k=population_size)


# Recombination - cut and crossfill crossover
def complete_child(child: list[int], donor: list[int]) -> None:
    """Append missing genes from the donor, preserving their order."""
    for gene in donor:
        if gene not in child:
            child.append(gene)


def recombination(
    parent_1: list[int],
    parent_2: list[int],
    cut_point: int | None = None,
) -> tuple[list[int], list[int]]:
    n = len(parent_1)

    # Step 1: Select a crossover point i in {1, ..., n - 1}.
    if cut_point is None:
        cut_point = random.randrange(1, n)
    elif not 1 <= cut_point < n:
        raise ValueError(f"cut_point must be between 1 and {n - 1}.")

    # Steps 2-3: Copy the first segment of each parent.
    child_1 = parent_1[:cut_point]
    child_2 = parent_2[:cut_point]

    # Step 4: Complete each child using the other parent.
    complete_child(child_1, parent_2)
    complete_child(child_2, parent_1)

    return child_1, child_2


def cut_and_crossfill(individual_1, individual_2):
    child_1, child_2 = recombination(list(individual_1), list(individual_2))
    individual_1[:] = child_1
    individual_2[:] = child_2
    return individual_1, individual_2


##### Mutation - Swap mutation
def select_random_positions(n: int, k: int = 2) -> list[int]:
    return random.sample(range(n), k=k)


def swap_mutation(chromosome: list[int]) -> list[int]:
    mutated = chromosome.copy()
    i, j = select_random_positions(len(mutated))
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def swap(individual):
    """Adapt the copy-returning function to DEAP's in-place tuple contract."""
    individual[:] = swap_mutation(list(individual))
    return (individual,)


################


def build_toolbox(n_queens, tournament_size=5):
    """Configure DEAP for one N-Queens problem size."""
    toolbox = base.Toolbox()

    if not hasattr(creator, "FitnessMaxNQueens"):
        creator.create("FitnessMaxNQueens", base.Fitness, weights=(1.0,))

    if not hasattr(creator, "NQueensIndividual"):
        creator.create(
        "NQueensIndividual",
        list,
        fitness=creator.FitnessMaxNQueens,
        )

    toolbox.register("permutation", random.sample, range(n_queens), n_queens)
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.NQueensIndividual,
        toolbox.permutation,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda individual: (fitnessFunction(individual),))
    toolbox.register(
        "select",
        select_best_two,
        tournament_size=tournament_size,
        num_parents=2,
    )
    toolbox.register("mate", cut_and_crossfill)
    toolbox.register("mutate", swap)
    toolbox.register("survive", replace_worst)
    return toolbox

def run_n_queens_ea(
    n_queens=8,
    population_size=100,
    max_fitness_evaluations=10_000,
    crossover_probability=1.0,
    mutation_probability=0.8,
    tournament_size=5,
    seed=42,
    verbose=False,
):
    """Run the evolutionary algorithm for the N-Queens problem.
    Args:
        n_queens: The number of queens and the size of the chessboard.
        population_size: The number of individuals in the population.
        max_fitness_evaluations: The maximum number of fitness evaluations allowed.
        crossover_probability: The probability of performing crossover on parents.
        mutation_probability: The probability of mutating an offspring.
        tournament_size: The number of individuals in each tournament selection.
        seed: Random seed for reproducibility.
        verbose: If True, print progress information.
    """
    random.seed(seed)
    toolbox = build_toolbox(n_queens, tournament_size)
    population = toolbox.population(n=population_size)
    optimum = n_queens * (n_queens - 1) // 2

    # evaluates every individual
    for individual, fitness in zip(
        population,
        map(toolbox.evaluate, population),
    ):
        individual.fitness.values = fitness

    fitness_evaluations = len(population)
    generation = 0
    best = tools.selBest(population, k=1)[0]
    history = [
        {
            "generation": generation,
            "evaluations": fitness_evaluations,
            "best_fitness": best.fitness.values[0],
            "mean_fitness": np.mean([ind.fitness.values[0] for ind in population]),
        }
    ]
    while (
        best.fitness.values[0] < optimum
        and fitness_evaluations < max_fitness_evaluations
    ):
        generation += 1
        parents = toolbox.select(population)
        offspring = list(map(toolbox.clone, parents))

        if random.random() < crossover_probability:
            toolbox.mate(offspring[0], offspring[1])
            del offspring[0].fitness.values
            del offspring[1].fitness.values

        for child in offspring:
            if random.random() < mutation_probability:
                toolbox.mutate(child)
                if child.fitness.valid:
                    del child.fitness.values

        invalid_offspring = [child for child in offspring if not child.fitness.valid]
        remaining_budget = max_fitness_evaluations - fitness_evaluations
        invalid_offspring = invalid_offspring[:remaining_budget]

        for child, fitness in zip(
            invalid_offspring,
            map(toolbox.evaluate, invalid_offspring),
        ):
            child.fitness.values = fitness
        fitness_evaluations += len(invalid_offspring)

        evaluated_offspring = [child for child in offspring if child.fitness.valid]
        population[:] = toolbox.survive(population, evaluated_offspring)
        best = tools.selBest(population, k=1)[0]

        history.append(
            {
                "generation": generation,
                "evaluations": fitness_evaluations,
                "best_fitness": best.fitness.values[0],
                "mean_fitness": np.mean([ind.fitness.values[0] for ind in population]),
            }
        )

    return {
        "best_permutation": list(best),
        "best_fitness": int(best.fitness.values[0]),
        "evaluations": fitness_evaluations,
        "success": int(best.fitness.values[0] == optimum),
        "history": pd.DataFrame(history),
    }