#ifndef KERNEL_H
#define KERNEL_H

#include <cmath>
#include <ctime>
#include <iostream>
#include <string>
#include <vector> // Include the vector header for the GA population

const int SELECTED_OBJ_FUNC = 3;
const int NUM_OF_PARTICLES = 512;
const int NUM_OF_DIMENSIONS = 3;
const int MAX_ITER = NUM_OF_DIMENSIONS * 10000 - NUM_OF_PARTICLES;
const float START_RANGE_MIN = -5.12f;
const float START_RANGE_MAX = 5.12f;
const float OMEGA = 0.5;
const float c1 = 1.5;
const float c2 = 1.5;
const float phi = 3.1415;
const int NUM_OF_NEIGHBORS = 5;

// Define a structure for individuals in the GA population
struct Individual {
    std::vector<float> genome;
    float fitness;
};

// Define GA parameters
const int GA_POPULATION_SIZE = 100; // Adjust population size, mutation rate, and crossover rate as needed
const float GA_MUTATION_RATE = 0.01;
const float GA_CROSSOVER_RATE = 0.7;

float getRandom(float low, float high);
float getRandomClamped();
float host_fitness_function(float x[]);
void precomputeNeighborIndices(float *positions, int *neighborIndices);
void cpu_pso(float *positions, float *velocities, float *pBests, float *gBest);
void initializeGAPopulation(std::vector<Individual>& gaPopulation, int populationSize);
void updateGAPopulation(std::vector<Individual>& gaPopulation, float *positions, float *pBests, float *gBest);
void updateBestSolution(float *gBest, const std::vector<Individual>& gaPopulation);
extern "C" void cuda_pso(float *positions, float *velocities, float *pBests, float *gBest, int *neighborIndices);

#endif  // KERNEL_H
