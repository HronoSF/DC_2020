#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_LEN 100000

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
    int search_array[MAX_LEN];  // The vector to search max
    int my_max; // Max for sub-vector
    int global_max; // Maximum for the main vector
    int local_max; // Local max from one process
    MPI_Status status;  // status for receive

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

        generate_data(search_array, array_len);

        if (isSequential) {
            printf("The sequential search gives %d\n", find_max(search_array, array_len));
        }

        for (rank = 1; rank < rem; ++rank) {
            sub_len = quotient + 1;
            // rank * quotient - the number of element in the part before this part
            // rank - how many part of size 1 is before

            sub_start = rank * quotient + rank;
            MPI_Send(&sub_len, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
            MPI_Send(&(search_array[sub_start]), sub_len, MPI_INT, rank, 0, MPI_COMM_WORLD);
        }

        for (rank = rem; rank < num_proc; ++rank) {
            sub_len = quotient;
            // rank * quotient - the number of element in the part this part
            // rem - how many part of size 1 is before

            sub_start = rank * quotient + rem;
            MPI_Send(&sub_len, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
            MPI_Send(&(search_array[sub_start]), sub_len, MPI_INT, rank, 0, MPI_COMM_WORLD);
        }

        // find local max:
        sub_len = rem == 0 ? quotient : quotient + 1;
        global_max = find_max(search_array, sub_len);

        // get max from processes:
        for (rank = 1; rank < num_proc; ++rank) {
            MPI_Recv(&local_max, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            if (local_max > global_max) {
                global_max = local_max;
            }
        }

        printf("The parallel search gives %d\n", global_max);
    } else {
        // Receive sub-vector length:
        MPI_Recv(&sub_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        // Receive sub-vector:
        MPI_Recv(search_array, sub_len, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        my_max = find_max(search_array, sub_len);

        // Send back local max:
        MPI_Send(&my_max, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}