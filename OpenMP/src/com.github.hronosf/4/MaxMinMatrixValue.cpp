#include<time.h>
#include <omp.h>
#include <random>
#include <bits/stdc++.h>
#include "../OpenMPTask.hpp"

class MaxMinMatrixValue : public OpenMPTask {
public:
    MaxMinMatrixValue() {
        std::cout << "\nMaxMinMatrixValue data:\n";
    }

protected:
    void execute_task() override {
        operations = {"critical", "reduction", "consistent"};

        // low doun element count not to wait for too long:
        elementCount = {4, 100, 1000, 10000};

        // to get random numbers:
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(30000, 100000);

        for (int &nT : threadCount) {
            std::vector<double> reduceExecutionTime;
            std::vector<double> atomicExecutionTime;
            std::vector<double> consistentExecutionTime;
            std::map<std::string, std::vector<double>> tempStatistic;

            for (int i : elementCount) {
                // dump data:
                std::vector<std::vector<double>> matrix;
                for (int j = 0; j < i; j++) {
                    std::vector<double> row;

                    for (int k = 0; k < i; k++) {
                        row.push_back(dist(mt));
                    }

                    matrix.push_back(row);
                }

                double tempResult[i];

#pragma omp parallel for default(none) shared(tempResult, matrix, i)
                for (int j = 0; j < i; j++) {
                    tempResult[j] = matrix[j][0];
                }

                // evaluate via reduction:
                reduceExecutionTime.push_back(execute(tempResult, matrix, i, nT, &max_min_reduction));

                // evaluate via critical:
                atomicExecutionTime.push_back(execute(tempResult, matrix, i, nT, &max_min_critical));

                // evaluate consistently:
                consistentExecutionTime.push_back(execute(tempResult, matrix, i, nT, &max_min_consistently));
            }

            tempStatistic.insert({"critical", atomicExecutionTime});
            tempStatistic.insert({"reduction", reduceExecutionTime});
            tempStatistic.insert({"consistent", consistentExecutionTime});

            statistic.insert({nT, tempStatistic});
        }
    };

private:
    static double execute(double *tempResult, const std::vector<std::vector<double>> &matrix,
                          int rowCount, int threadCount,
                          const std::function<void(double *tempResult,
                                                   const std::vector<std::vector<double>> &matrix,
                                                   int rowCount)> &func) {
        omp_set_num_threads(threadCount);
        clock_t tStart, tEnd;

        tStart = clock();

        func(tempResult, matrix, rowCount);

        tEnd = clock();

        return ((double) (tEnd - tStart) / CLOCKS_PER_SEC);
    }

    static void max_min_reduction(double *tempResult, const std::vector<std::vector<double>> &matrix, int rowCount) {
        int i;
#pragma omp parallel for default(none) shared(tempResult, rowCount, matrix) private(i)
        for (i = 0; i < rowCount; i++) {
#pragma omp parallel for default(none) shared(matrix, rowCount) shared(i) reduction(min: tempResult[i])
            for (int j = 0; j < rowCount; j++) {
                if (matrix[i][j] < tempResult[i]) {
                    tempResult[i] = matrix[i][j];
                }
            }
        }

        double max_of_min = tempResult[0];

#pragma omp parallel for default(none) shared(tempResult, rowCount) reduction(max: max_of_min)  private(i)
        for (i = 1; i < rowCount; i++) {
            if (tempResult[i] > max_of_min) {
                max_of_min = tempResult[i];
            }
        }
    }

    static void max_min_critical(double *tempResult, const std::vector<std::vector<double>> &matrix, int rowCount) {
        int i;

#pragma omp parallel for default(none) shared(tempResult, matrix, rowCount) private(i)
        for (i = 0; i < rowCount; i++) {
            for (int j = 0; j < rowCount; j++) {
                if (matrix[i][j] < tempResult[i])
#pragma omp critical
                {
                    if (matrix[i][j] < tempResult[i]) {
                        tempResult[i] = matrix[i][j];
                    }
                }
            }
        }

        double max_of_min = tempResult[0];

#pragma omp parallel for default(none) shared(tempResult, max_of_min, rowCount) private(i)
        for (i = 1; i < rowCount; i++) {
            if (tempResult[i] > max_of_min) {
#pragma omp critical
                {
                    if (tempResult[i] > max_of_min) {
                        max_of_min = tempResult[i];
                    }
                }
            }
        }

    }

    static void max_min_consistently(double *tempResult, const std::vector<std::vector<double>> &matrix, int rowCount) {
        int i;

        for (i = 0; i < rowCount; i++) {
            for (int j = 0; j < rowCount; j++) {
                if (matrix[i][j] < tempResult[i]) {
                    tempResult[i] = matrix[i][j];
                }
            }
        }

        double max_of_min = tempResult[0];

        for (i = 0; i < rowCount; i++) {
            if (tempResult[i] > max_of_min) {
                {
                    max_of_min = tempResult[i];
                }
            }
        }
    }
};