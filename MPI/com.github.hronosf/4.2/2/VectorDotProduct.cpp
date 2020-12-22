#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_LEN 100000

// Usual product function
int dotProduct(int a[], int b[], int len) {
    int i;
    int sum = 0;

    for (i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

// generate vector:
void generate_data(int a[], int len) {
    int i;

    // to get random numbers:
    struct timeval time;
    gettimeofday(&time, (struct timezone *) 0);
    srand((int) time.tv_sec);

    for (i = 0; i < len; ++i) {
        a[i] = i * rand();
    }
}

int main(int argc, char *argv[]) {
    int my_rank;
    int rank;    // Loop variable for the processes
    int num_proc;     // Total number of processes
    int array_len;  // Length of the init vector
    int isSequential; // To execute sequential
    int quotient;  // sub-vector size: array_len/num_proc
    int rem;  // How many larger sub-vector: array_len % num_proc

    int sub_start;  // Start of one of the sub-vector
    int sub_len;  // Length of sub-vector
    int a[MAX_LEN];  // The vector to search max
    int b[MAX_LEN];  // The vector to search max
    int my_sum; // Max for sub-vector
    int global_sum; // Maximum for the main vector
    int local_sum; // Local sum from one process
    MPI_Status status;  // status for receive

    int tag_size = 0;
    int tag_a = 1;
    int tag_b = 2;
    int tag_prod = 3;

    // MPI initialization:
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Main process logic:
    if (my_rank == 0) {
        isSequential = atoi(argv[1]); // if == 1 => consistent execution will be executed
        array_len = atoi(argv[2]);    // vector size
        MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

        // to broke up vector for multiple processes:
        quotient = array_len / num_proc;
        rem = array_len % num_proc;

        printf("\nProcess number %d\n", num_proc);
        printf("The P part %d\n", quotient);
        printf("Number of processes that need an additional element %d\n\n", rem);

        generate_data(a, array_len);
        generate_data(b, array_len);

        if (isSequential) {
            printf("The sequential dot product gives  %d\n", dotProduct(a, b, array_len));
        }

        for (rank = 0; rank < rem; ++rank) {
            sub_len = quotient + 1;
            // rank * quotient - the number of element in the part before this part
            // rank - how many part of size 1 is before

            sub_start = rank * quotient + rank;

            printf("Sended sub len %d\n", sub_len);

            MPI_Send(&sub_len, 1, MPI_INT, rank, tag_size, MPI_COMM_WORLD);
            MPI_Send(&(a[sub_start]), sub_len, MPI_INT, rank, tag_a, MPI_COMM_WORLD);
            MPI_Send(&(b[sub_start]), sub_len, MPI_INT, rank, tag_b, MPI_COMM_WORLD);
        }

        for (rank = rem; rank < num_proc; ++rank) {
            sub_len = quotient;
            // rank * quotient - the number of element in the part this part
            // rem - how many part of size 1 is before

            sub_start = rank * quotient + rem;

            printf("Sended sub len %d\n", sub_len);

            MPI_Send(&sub_len, 1, MPI_INT, rank, tag_size, MPI_COMM_WORLD);
            MPI_Send(&(a[sub_start]), sub_len, MPI_INT, rank, tag_a, MPI_COMM_WORLD);
            MPI_Send(&(b[sub_start]), sub_len, MPI_INT, rank, tag_b, MPI_COMM_WORLD);
        }

        sub_len = rem == 0 ? quotient : quotient + 1;
        global_sum = dotProduct(a, b, sub_len);

        // get local sums from processes:
        for (rank = 1; rank < num_proc; ++rank) {
            MPI_Recv(&local_sum, 1, MPI_INT, MPI_ANY_SOURCE, tag_prod, MPI_COMM_WORLD, &status);
            global_sum += local_sum;
        }

        printf("The parallel dot product gives %d\n", global_sum);
    } else {
        // Receive sub-vector length:
        MPI_Recv(&sub_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        // Receive sub-vectors:
        MPI_Recv(a, sub_len, MPI_INT, 0, tag_a, MPI_COMM_WORLD, &status);
        MPI_Recv(b, sub_len, MPI_INT, 0, tag_b, MPI_COMM_WORLD, &status);

        my_sum = dotProduct(a, b, sub_len);

        // Send back local sum:
        MPI_Send(&my_sum, 1, MPI_INT, 0, tag_prod, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}