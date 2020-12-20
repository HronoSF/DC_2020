#include<time.h>
#include <omp.h>
#include <random>
#include <bits/stdc++.h>
#include "../OpenMPTask.hpp"

class VectorDotProductTask : public OpenMPTask {
public:
    VectorDotProductTask() {
        std::cout << "\nVectorDotProductTask data:";
    }

protected:
    void execute_task() override {
        operations = {"atomic", "reduction", "consistent"};

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
                std::vector<double> aN, bN;
                for (int j = 0; j < i; j++) {
                    aN.push_back(dist(mt));
                    bN.push_back(j + dist(mt));
                }

                // evaluate via reduction:
                reduceExecutionTime.push_back(execute(aN, bN, i, nT, &dot_product_reduction));

                // evaluate via atomic:
                atomicExecutionTime.push_back(execute(aN, bN, i, nT, &dot_product_atomic));

                // evaluate consistently:
                consistentExecutionTime.push_back(execute(aN, bN, i, nT, &dot_product_consistently));
            }

            tempStatistic.insert({"atomic", atomicExecutionTime});
            tempStatistic.insert({"reduction", reduceExecutionTime});
            tempStatistic.insert({"consistent", consistentExecutionTime});

            statistic.insert({nT, tempStatistic});
        }
    };

private:

    static double
    execute(const std::vector<double> &a, const std::vector<double> &b, int elementCount, int threadCount,
            const std::function<void(
                    const std::vector<double> &a, const std::vector<double> &b, int &elementCount)> &func) {
        omp_set_num_threads(threadCount);
        clock_t tStart, tEnd;

        tStart = clock();

        func(a, b, elementCount);

        tEnd = clock();

        return ((double) (tEnd - tStart) / CLOCKS_PER_SEC);
    }

    static void
    dot_product_reduction(const std::vector<double> &a, const std::vector<double> &b, int &elementCount) {
        double sum = 0;
        int i;

#pragma omp parallel for default(none) shared(a, b, elementCount) reduction(+:sum)
        for (i = 0; i < elementCount; i++) {
            sum += a[i] * b[i];
        }
    }

    static void dot_product_atomic(const std::vector<double> &a, const std::vector<double> &b, int elementCount) {
        double sum = 0;
        int i;

#pragma omp parallel for default(none) shared(a, b, elementCount, sum)
        for (i = 0; i < elementCount; i++) {
#pragma omp atomic
            sum += a[i] * b[i];
        }
    }

    static void dot_product_consistently(const std::vector<double> &a, const std::vector<double> &b, int elementCount) {
        double sum = 0;

        for (int i = 0; i < elementCount; i++) {
            sum += a[i] * b[i];
        }
    }
};