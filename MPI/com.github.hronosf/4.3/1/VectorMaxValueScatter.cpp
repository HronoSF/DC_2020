#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <random>

// Usual search function
int find_max(int a[], int len) {
    int i;
    int max = a[0];

    for (i = 1; i < len; ++i) {
        if (a[i] > max) {
            max = a[i];
        }
    }

    return max;
}

// generate vector:
void generate_data(int a[], int len) {
    int i;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(1, 10);

    for (i = 0; i < len; ++i) {
        a[i] = uni(rng);
    }
}

main(int argc, char *argv[]) {
    int *local_a;
    int array_len = atoi(argv[1]);    // vector size
    int *a = new int[array_len];  // The vector to search max
    int quotient;
    int max, local_max;
    int num_proc, my_rank, i;

    double start, stop; // to measure time

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        generate_data(a, array_len);

        start = MPI_Wtime();
    }

    quotient = array_len / num_proc;
    local_a = new int[quotient];

    MPI_Scatter(a, quotient, MPI_INT, local_a, quotient, MPI_INT, 0, MPI_COMM_WORLD);

    local_max = find_max(local_a, quotient);

    MPI_Reduce(&local_max, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        stop = MPI_Wtime();
        printf("The parallel max search executed for %f second\n", stop - start);
    }

    MPI_Finalize();
    return 0;
}