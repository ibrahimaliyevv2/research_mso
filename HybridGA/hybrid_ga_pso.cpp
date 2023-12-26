// hybrid_ga_pso.cpp
#include "kernel.h"
#include <algorithm> // Include the algorithm header for random_shuffle

// Function to perform tournament selection for GA
Individual tournamentSelection(const std::vector<Individual>& population) {
    int tournamentSize = 5; // Adjust as needed
    std::vector<Individual> tournament;

    // Randomly select individuals for the tournament
    for (int i = 0; i < tournamentSize; i++) {
        int index = getRandom(0, population.size() - 1);
        tournament.push_back(population[index]);
    }

    // Select the best individual from the tournament
    return *std::max_element(tournament.begin(), tournament.end(), [](const Individual& a, const Individual& b) {
        return a.fitness < b.fitness;
    });
}

// Function to perform one-point crossover for GA
Individual onePointCrossover(const Individual& parent1, const Individual& parent2) {
    int crossoverPoint = getRandom(1, NUM_OF_DIMENSIONS - 2); // Avoid endpoints
    Individual child;
    child.genome.reserve(NUM_OF_DIMENSIONS);

    for (int i = 0; i < crossoverPoint; i++) {
        child.genome.push_back(parent1.genome[i]);
    }

    for (int i = crossoverPoint; i < NUM_OF_DIMENSIONS; i++) {
        child.genome.push_back(parent2.genome[i]);
    }

    return child;
}

// Function to perform mutation for GA
void mutation(Individual& individual) {
    for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
        if (getRandomClamped() < GA_MUTATION_RATE) {
            individual.genome[i] = getRandom(START_RANGE_MIN, START_RANGE_MAX);
        }
    }
}

// Function to perform fitness evaluation for GA
void evaluateFitness(std::vector<Individual>& population) {
    for (Individual& individual : population) {
        individual.fitness = host_fitness_function(&individual.genome[0]);
    }
}

void geneticAlgorithm(std::vector<Individual>& gaPopulation) {
    // Selection, Crossover, and Mutation for GA
    std::vector<Individual> newPopulation;

    while (newPopulation.size() < GA_POPULATION_SIZE) {
        Individual parent1 = tournamentSelection(gaPopulation);
        Individual parent2 = tournamentSelection(gaPopulation);

        Individual child = onePointCrossover(parent1, parent2);
        mutation(child);
        newPopulation.push_back(child);
    }

    // Update the GA population with the new population
    gaPopulation = newPopulation;

    // Fitness Evaluation for GA
    evaluateFitness(gaPopulation);
}

// Function to update the PSO particles based on the GA results
void updateParticlesWithGA(std::vector<Individual>& gaPopulation, float* positions, float* velocities, float* pBests, float* gBest) {
    // Update particles based on GA results
    for (int i = 0; i < NUM_OF_PARTICLES * NUM_OF_DIMENSIONS; i++) {
        // Apply GA-based updates to particle positions and velocities
        // You may use information from the gaPopulation here
        // For simplicity, we'll just copy GA's best solution into PSO's gBest
        gBest[i % NUM_OF_DIMENSIONS] = gaPopulation[0].genome[i % NUM_OF_DIMENSIONS];
    }
}

void hybrid_ga_pso(float* positions, float* velocities, float* pBests, float* gBest, std::vector<Individual>& gaPopulation, int* neighborIndices) {
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // PSO Update
        cpu_pso(positions, velocities, pBests, gBest);

        // Genetic Algorithm Update
        geneticAlgorithm(gaPopulation);

        // Update PSO particles based on the results of the genetic algorithm
        updateParticlesWithGA(gaPopulation, positions, velocities, pBests, gBest);
    }
}

