#include <cuda_runtime.h>
#include <cuda.h>
#include <cfloat>  // For FLT_MAX

#include "kernel.h"

__device__ float tempParticle1[NUM_OF_DIMENSIONS];
__device__ float tempParticle2[NUM_OF_DIMENSIONS];

/* Objective function
0: Levy 3-dimensional
1: Shifted Rastigrin's Function
2: Shifted Rosenbrock's Function
3: Shifted Griewank's Function
4: Shifted Sphere's Function
*/
/**
 * Runs on the GPU, called from the GPU.
*/
__device__ float fitness_function(int selectedObjFunc, float x[]) {
    float res = 0;
    float somme = 0;
    float produit = 0;
    float y1 = 0;
    float yn = 0;

    switch (SELECTED_OBJ_FUNC)  {
        case 0: 
            y1 = 1 + (x[0] - 1)/4;
            yn = 1 + (x[NUM_OF_DIMENSIONS-1] - 1)/4;

            res += pow(sin(phi*y1), 2);

            for (int i = 0; i < NUM_OF_DIMENSIONS-1; i++) {
                float y = 1 + (x[i] - 1)/4;
                float yp = 1 + (x[i+1] - 1)/4;
                res += pow(y - 1, 2)*(1 + 10*pow(sin(phi*yp), 2)) + pow(yn - 1, 2);
            }
            break;
        case 1: 
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2) - 10*cos(2*phi*zi) + 10;
            }
            res -= 330;
            break;
        
        case 2:
            for (int i = 0; i < NUM_OF_DIMENSIONS-1; i++) {
                float zi = x[i] - 0 + 1;
                float zip1 = x[i+1] - 0 + 1;
                res += 100 * ( pow(pow(zi, 2) - zip1, 2)) + pow(zi - 1, 2);
            }
            res += 390;
            break;
        case 3:
            for (int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                somme += pow(zi, 2)/4000;
                produit *= cos(zi/pow(i+1, 0.5));
            }
            res = somme - produit + 1 - 200; 
            break;
        case 4:
            for(int i = 0; i < NUM_OF_DIMENSIONS; i++) {
                float zi = x[i] - 0;
                res += pow(zi, 2);
            }
            res -= 450;
            break;
    }

    return res;
}

/**
 * 
 * Runs on the GPU, called from the CPU or the GPU
*/
__global__ void kernelUpdateParticle(float *positions, float *velocities,
                                     float *pBests, float *gBest, int *neighborIndices, int selectedObjFunc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS)
        return;

    int particleIndex = i / NUM_OF_DIMENSIONS;
    int numNeighbors = NUM_OF_NEIGHBORS; // Change this to the desired neighborhood size

    // Initialize local best index to the current particle
    int localBestIndex = particleIndex;

    // Find the best particle within the local neighborhood
    for (int j = 0; j < numNeighbors; j++) {
        int neighborIndex = neighborIndices[particleIndex * numNeighbors + j];
        if (neighborIndex < NUM_OF_PARTICLES) {
            int neighborPBestIndex = neighborIndex * NUM_OF_DIMENSIONS;
            int localBestPBestIndex = localBestIndex * NUM_OF_DIMENSIONS;
            
            if (fitness_function(selectedObjFunc, &positions[neighborPBestIndex]) < fitness_function(selectedObjFunc, &positions[localBestPBestIndex])) {
                localBestIndex = neighborIndex;
            }
        }
    }

    // Update velocity and position based on local best
    // You can use rp and rg as before
    float rp = getRandomClamped();
    float rg = getRandomClamped();

    for (int j = 0; j < NUM_OF_DIMENSIONS; j++) {
        velocities[i] = OMEGA * velocities[i] +
            c1 * rp * (positions[pBests[i]] - positions[i]) +
            c2 * rg * (positions[pBests[localBestIndex * NUM_OF_DIMENSIONS + j]] - positions[i]);
        positions[i] += velocities[i];
    }
}

/**
 * Runs on the GPU, called from the CPU or the GPU
*/
__global__ void kernelUpdatePBest(float *positions, float *pBests, float* gBest)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= NUM_OF_PARTICLES * NUM_OF_DIMENSIONS || i % NUM_OF_DIMENSIONS != 0)
        return;

    for (int j = 0; j < NUM_OF_DIMENSIONS; j++)
    {
        tempParticle1[j] = positions[i + j];
        tempParticle2[j] = pBests[i + j];
    }

    if (fitness_function(tempParticle1) < fitness_function(tempParticle2))
    {
        for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
            pBests[i + k] = positions[i + k];
    }
}

__device__ void precomputeNeighborIndices(float *positions, int *neighborIndices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUM_OF_PARTICLES)
        return;

    float distances[NUM_OF_PARTICLES];
    for (int j = 0; j < NUM_OF_PARTICLES; j++) {
        if (i != j) {
            float dx = positions[i * NUM_OF_DIMENSIONS] - positions[j * NUM_OF_DIMENSIONS];
            float dy = positions[i * NUM_OF_DIMENSIONS + 1] - positions[j * NUM_OF_DIMENSIONS + 1];
            float distance = sqrtf(dx * dx + dy * dy);
            distances[j] = distance;
        } else {
            distances[j] = FLT_MAX;
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
        distances[minIndex] = FLT_MAX;
        neighborIndices[i * NUM_OF_NEIGHBORS + k] = minIndex;
    }
}

extern "C" void cuda_pso(float *positions, float *velocities, float *pBests, float *gBest, int *neighborIndices)
{
    int size = NUM_OF_PARTICLES * NUM_OF_DIMENSIONS;

    float *devPos;
    float *devVel;
    float *devPBest;
    float *devGBest;
    int *devNeighborIndices;

    cudaMalloc((void**)&devPos, sizeof(float) * size);
    cudaMalloc((void**)&devVel, sizeof(float) * size);
    cudaMalloc((void**)&devPBest, sizeof(float) * size);
    cudaMalloc((void**)&devGBest, sizeof(float) * NUM_OF_DIMENSIONS);
    cudaMalloc((void**)&devNeighborIndices, sizeof(int) * NUM_OF_PARTICLES * NUM_OF_NEIGHBORS);

    int threadsNum = 32;
    int blocksNum = ceil(size / threadsNum);

    cudaMemcpy(devPos, positions, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devVel, velocities, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devPBest, pBests, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devGBest, gBest, sizeof(float) * NUM_OF_DIMENSIONS, cudaMemcpyHostToDevice);
    cudaMemcpy(devNeighborIndices, neighborIndices, sizeof(int) * NUM_OF_PARTICLES * NUM_OF_NEIGHBORS, cudaMemcpyHostToDevice);

    for (int iter = 0; iter < MAX_ITER; iter++)
    {
        float rp = getRandomClamped();
        float rg = getRandomClamped();

        kernelUpdateParticle<<<blocksNum, threadsNum>>>(devPos, devVel, devPBest, devGBest, devNeighborIndices);
        kernelUpdatePBest<<<blocksNum, threadsNum>>>(devPos, devPBest, devGBest);

        cudaMemcpy(pBests, devPBest, sizeof(float) * size, cudaMemcpyDeviceToHost);

        for (int i = 0; i < size; i += NUM_OF_DIMENSIONS)
        {
            for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
            {
                tempParticle1[k] = pBests[i + k];
                tempParticle2[k] = gBest[k];
            }

            if (fitness_function(tempParticle1) < fitness_function(tempParticle2))
            {
                for (int k = 0; k < NUM_OF_DIMENSIONS; k++)
                    gBest[k] = tempParticle1[k];
            }
        }

        cudaMemcpy(devGBest, gBest, sizeof(float) * NUM_OF_DIMENSIONS, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(positions, devPos, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, devVel, sizeof(float) * size, cudaMemcpyDeviceToHost);

    cudaFree(devPos);
    cudaFree(devVel);
    cudaFree(devPBest);
    cudaFree(devGBest);
    cudaFree(devNeighborIndices);
}
