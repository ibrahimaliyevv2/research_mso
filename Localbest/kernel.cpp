#include "kernel.h"
#include <cfloat>  // For FLT_MAX

float host_fitness_function(float x[]) {
    float res = 0;
    float somme = 0;
    float produit = 0;
    float y1 = 0;
    float yn = 0;

    switch (SELECTED_OBJ_FUNC) {
        case 0: {
            y1 = 1 + (x[0] - 1) / 4;
            yn = 1 + (x[NUM_OF_DIMENSIONS - 1] - 1) / 4;

            res += pow(sin(phi * y1), 2);

            for (int i = 0; i < NUM_OF_DIMENSIONS - 1; i++) {
                float y = 1 + (x[i] - 1) / 4;
                float yp = 1 + (x[i + 1] - 1) / 4;
                res += pow(y - 1, 2) * (1 + 10 * pow(sin(phi * yp), 2)) + pow(yn - 1, 2);
            }
            break;
        }
        case 1: {
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2) - 10 * cos(2 * phi * zi) + 10;
            }
            res -= 330;
            break;
        }
        case 2: {
            for (int i = 0; i < NUM_OF_DIMENSIONS - 1; i++) {
                float zi = x[i] - 0 + 1;
                float zip1 = x[i + 1] - 0 + 1;
                res += 100 * (pow(pow(zi, 2) - zip1, 2)) + pow(zi - 1, 2);
            }
            res += 390;
            break;
        }
        case 3: {
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                somme += pow(zi, 2) / 4000;
                produit *= cos(zi / sqrt(i + 1));
            }
            res = somme - produit + 1 - 200;
            break;
        }
        case 4: {
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2);
            }
            res -= 450;
            break;
        }
    }

    return res;
}

float getRandom(float low, float high) {
    return low + (static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) + 1.0f)) * (high - low);
}

float getRandomClamped() {
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void precomputeNeighborIndices(float *positions, int *neighborIndices) {
    for (int i = 0; i < NUM_OF_PARTICLES; i++) {
        float distances[NUM_OF_PARTICLES];
        for (int j = 0; j < NUM_OF_PARTICLES; j++) {
            if (i != j) {
                float dx = positions[i * NUM_OF_DIMENSIONS] - positions[j * NUM_OF_DIMENSIONS];
                float dy = positions[i * NUM_OF_DIMENSIONS + 1] - positions[j * NUM_OF_DIMENSIONS + 1];
                float distance = sqrtf(dx * dx + dy * dy);
                distances[j] = distance;
            } else {
                distances[j] = FLT_MAX; // Exclude itself from neighbors
            }
        }

        for (int k = 0; k < NUM_OF_NEIGHBORS; k++) {
            int minIndex = -1;
            float minDistance = FLT_MAX;
            for (int j = 0; j < NUM_OF_PARTICLES; j++) {
                if (distances[j] < minDistance) {
                    minDistance = distances[j];
                    minIndex = j;
                }
            }
            distances[minIndex] = FLT_MAX; // Mark this particle as considered
            neighborIndices[i * NUM_OF_NEIGHBORS + k] = minIndex;
        }
    }
}


void cpu_pso(float *positions, float *velocities, float *pBests, float *gBest) {
    float temp[NUM_OF_DIMENSIONS];

    srand(static_cast<unsigned>(time(nullptr)));

    for (int i = 0; i < NUM_OF_PARTICLES * NUM_OF_DIMENSIONS; i++) {
        positions[i] = getRandom(START_RANGE_MIN, START_RANGE_MAX);
        pBests[i] = positions[i];
        velocities[i] = 0;
    }

    for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
        gBest[k] = pBests[k];

    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < NUM_OF_PARTICLES * NUM_OF_DIMENSIONS; i++) {
            float rp = getRandomClamped();
            float rg = getRandomClamped();

            velocities[i] = OMEGA * velocities[i] + c1 * rp * (pBests[i] - positions[i]) + c2 * rg * (gBest[i % NUM_OF_DIMENSIONS] - positions[i]);
            positions[i] += velocities[i];
        }

        for (int i = 0; i < NUM_OF_PARTICLES * NUM_OF_DIMENSIONS; i += NUM_OF_DIMENSIONS) {
            for (int k = 0; k < NUM_OF_DIMENSIONS; k++) {
                temp[k] = pBests[i + k];
            }

            if (host_fitness_function(temp) < host_fitness_function(gBest)) {
                for (int k = 0; k < NUM_OF_DIMENSIONS; k++) {
                    gBest[k] = temp[k];
                }
            }
        }
    }
}
