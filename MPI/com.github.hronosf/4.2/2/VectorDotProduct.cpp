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

int main(int argc, char *argv[]) {
    int my_rank;
    int rank;    // Loop variable for the processes
    int num_proc;     // Total number of processes
    int array_len = atoi(argv[2]);    // vector size
    // Length of the init vector
    int isSequential; // To execute sequential
    int quotient;  // sub-vector size: array_len/num_proc
    int rem;  // How many larger sub-vector: array_len % num_proc

    int sub_start;  // Start of one of the sub-vector
    int sub_len;  // Length of sub-vector
    double *a = new double[array_len];  // The vector to search max
    double *b = new double[array_len];  // The vector to search max
    double my_sum; // Max for sub-vector
    double global_sum; // Maximum for the main vector
    double local_sum; // Local sum from one process
    MPI_Status status;  // status for receive

    double start, stop; // to measure time

    int tag_size = 0;
    int tag_a = 1;
    int tag_b = 2;
    int tag_prod = 3;

    // MPI initialization:
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    // Main process logic:
    if (my_rank == 0) {
        isSequential = atoi(argv[1]); // if == 1 => consistent execution will be executed

        // to broke up vector for multiple processes:
        quotient = array_len / num_proc;
        rem = array_len % num_proc;

        printf("\nProcess number %d\n", num_proc);
        printf("The P part %d\n", quotient);
        printf("Number of processes that need an additional element %d\n\n", rem);

        generate_data(a, array_len);
        generate_data(b, array_len);

        if (isSequential) {
            start = MPI_Wtime();
            dotProduct(a, b, array_len);
            stop = MPI_Wtime();
            printf("The sequential dot product executed for %f second\n", stop - start);
        }

        start = MPI_Wtime();

        for (rank = 0; rank < rem; ++rank) {
            sub_len = quotient + 1;
            // rank * quotient - the number of element in the part before this part
            // rank - how many part of size 1 is before

            sub_start = rank * quotient + rank;
            MPI_Send(&sub_len, 1, MPI_INT, rank, tag_size, MPI_COMM_WORLD);
            MPI_Send(&(a[sub_start]), sub_len, MPI_DOUBLE, rank, tag_a, MPI_COMM_WORLD);
            MPI_Send(&(b[sub_start]), sub_len, MPI_DOUBLE, rank, tag_b, MPI_COMM_WORLD);
        }

        for (rank = rem + 1; rank < num_proc; ++rank) {
            sub_len = quotient;
            // rank * quotient - the number of element in the part this part
            // rem - how many part of size 1 is before

            sub_start = rank * quotient + rem;
            MPI_Send(&sub_len, 1, MPI_INT, rank, tag_size, MPI_COMM_WORLD);
            MPI_Send(&(a[sub_start]), sub_len, MPI_DOUBLE, rank, tag_a, MPI_COMM_WORLD);
            MPI_Send(&(b[sub_start]), sub_len, MPI_DOUBLE, rank, tag_b, MPI_COMM_WORLD);
        }

        sub_len = rem == 0 ? quotient : quotient + 1;
        global_sum = dotProduct(a, b, sub_len);

        // get local sums from processes:
        for (rank = 1; rank < num_proc; ++rank) {
            MPI_Recv(&local_sum, 1, MPI_DOUBLE, MPI_ANY_SOURCE, tag_prod, MPI_COMM_WORLD, &status);
            global_sum += local_sum;
        }

        stop = MPI_Wtime();
        printf("The parallel dot product executed for %f second\n", stop - start);
    } else {
        // Receive sub-vector length:
        MPI_Recv(&sub_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        // Receive sub-vectors:
        MPI_Recv(a, sub_len, MPI_DOUBLE, 0, tag_a, MPI_COMM_WORLD, &status);
        MPI_Recv(b, sub_len, MPI_DOUBLE, 0, tag_b, MPI_COMM_WORLD, &status);

        my_sum = dotProduct(a, b, sub_len);

        // Send back local sum:
        MPI_Send(&my_sum, 1, MPI_INT, 0, tag_prod, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}