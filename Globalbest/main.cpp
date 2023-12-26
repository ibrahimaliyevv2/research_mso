#include "kernel.h"

int main(int argc, char **argv) {
    if (argc == 3) {
        int dim = std::stoi(argv[1]);
        int pop = std::stoi(argv[2]);
    }

    float positions[NUM_OF_PARTICLES * NUM_OF_DIMENSIONS];
    float velocities[NUM_OF_PARTICLES * NUM_OF_DIMENSIONS];
    float pBests[NUM_OF_PARTICLES * NUM_OF_DIMENSIONS];
    float gBest[NUM_OF_DIMENSIONS];

    printf("Type \t Time \t  \t Minimum\n");

    clock_t begin = clock();
    cpu_pso(positions, velocities, pBests, gBest);
    clock_t end = clock();

    printf("CPU \t ");
    printf("%10.3lf \t", (double)(end - begin) / CLOCKS_PER_SEC);

    printf(" %f\n", host_fitness_function(gBest));

    return 0;
}
