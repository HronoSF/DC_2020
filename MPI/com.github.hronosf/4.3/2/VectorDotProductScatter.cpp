#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <random>

// Usual product function
double dotProduct(double a[], double b[], int len) {
    int i;
    double sum = 0;

    for (i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

// generate vector:
void generate_data(double a[], int len) {
    int i;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(1, 10);

    for (i = 0; i < len; ++i) {
        a[i] = uni(rng);
    }
}

main(int argc, char *argv[]) {
    double *local_a, *local_b;
    int array_len = atoi(argv[1]);    // vector size
    double *a = new double[array_len];  // The vector to search max
    double *b = new double[array_len];  // The vector to search max
    int quotient;
    double dot, local_dot;
    int num_proc, my_rank, i;

    double start, stop; // to measure time


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        generate_data(a, array_len);
        generate_data(b, array_len);

        start = MPI_Wtime();
    }

    quotient = array_len / num_proc;
    local_a = new double[quotient];
    local_b = new double[quotient];

    MPI_Scatter(a, quotient, MPI_DOUBLE, local_a, quotient, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, quotient, MPI_DOUBLE, local_b, quotient, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    local_dot += dotProduct(local_a, local_b, quotient);

    free(local_a);
    free(local_b);

    MPI_Reduce(&local_dot, &dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        stop = MPI_Wtime();
        printf("The parallel dot product executed for %f second\n", stop - start);
    }

    MPI_Finalize();
    return 0;
}