#include<time.h>
#include <omp.h>
#include <random>
#include <bits/stdc++.h>
#include "../OpenMPTask.hpp"

class ReductionTask : public OpenMPTask {

public:
    ReductionTask() {
        std::cout << "\nReductionTask data:\n";
    }

protected:
    void execute_task() override {
        operations = {"atomic", "reduction", "consistent", "lock", "critical"};
        elementCount = {10, 1000, 100000, 1000000};

        // to get random numbers:
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(30000, 100000);

        for (int &nT : threadCount) {
            std::vector<double> reduceExecutionTime;
            std::vector<double> atomicExecutionTime;
            std::vector<double> lockExecutionTime;
            std::vector<double> criticalExecutionTime;
            std::vector<double> defaultExecutionTime;
            std::vector<double> consistentExecutionTime;

            std::map<std::string, std::vector<double>> tempStatistic;

            for (int i : elementCount) {
                // dump data:
                std::vector<double> aN;
                for (int j = 0; j < i; j++) {
                    aN.push_back(dist(mt));
                }

                // evaluate via reduction:
                reduceExecutionTime.push_back(execute(aN, i, nT, &openmp_reduction));

                // evaluate via critical:
                criticalExecutionTime.push_back(execute(aN, i, nT, &reduction_critical));

                // evaluate via lock:
                lockExecutionTime.push_back(execute(aN, i, nT, &reduction_lock));

                // evaluate via atomic:
                atomicExecutionTime.push_back(execute(aN, i, nT, &reduction_atomic));

                // evaluate consistently:
                consistentExecutionTime.push_back(execute(aN, i, nT, &consistently));
            }

            tempStatistic.insert({"atomic", atomicExecutionTime});
            tempStatistic.insert({"reduction", reduceExecutionTime});
            tempStatistic.insert({"consistent", consistentExecutionTime});
            tempStatistic.insert({"lock", lockExecutionTime});
            tempStatistic.insert({"critical", criticalExecutionTime});

            statistic.insert({nT, tempStatistic});
        }
    };


private:

    static double execute(const std::vector<double> &a, int elementCount, int threadCount,
                          const std::function<void(const std::vector<double> &a, int elementCount)> &func) {
        omp_set_num_threads(threadCount);
        clock_t tStart, tEnd;

        tStart = clock();

        func(a, elementCount);

        tEnd = clock();

        return ((double) (tEnd - tStart) / CLOCKS_PER_SEC);
    }

    static void reduction_atomic(const std::vector<double> &a, int elementCount) {
        double sum = 0;
        int i;
#pragma omp parallel for default(none) shared(sum, elementCount, a)
        for (i = 0; i < elementCount; i++) {
#pragma omp atomic
            sum += a[i];
        }
    }

    static void reduction_critical(const std::vector<double> &a, int elementCount) {
        double sum = 0;
        int i;

#pragma omp  parallel for default(none) shared(sum, elementCount, a)
        for (i = 0; i < elementCount; i++) {
#pragma omp critical
            sum += a[i];
        }
    }

    static void reduction_lock(const std::vector<double> &a, int elementCount) {
        omp_lock_t lock;
        omp_init_lock(&lock);

        double sum = 0;
        int i;

#pragma omp  parallel for default(none) shared(sum, elementCount, a, lock)
        for (i = 0; i < elementCount; i++) {
            omp_set_lock(&lock);
            sum += a[i];
            omp_unset_lock(&lock);
        }

        omp_destroy_lock(&lock);
    }

    static void openmp_reduction(const std::vector<double> &a, int elementCount) {
        double sum = 0;
        int i;

#pragma omp parallel for default(none) shared(a, elementCount) private(i) reduction(+: sum)
        for (i = 0; i < elementCount; i++) {
            sum += a[i];
        }
    }

    static void consistently(const std::vector<double> &a, int elementCount) {
        double sum = 0;
        int i;

        for (i = 0; i < elementCount; i++) {
            sum += a[i];
        }
    }
};
