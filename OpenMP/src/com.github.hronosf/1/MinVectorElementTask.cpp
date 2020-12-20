#include<time.h>
#include <omp.h>
#include <random>
#include <bits/stdc++.h>
#include "../OpenMPTask.hpp"

class MinVectorElementTask : public OpenMPTask {
public:
    MinVectorElementTask() {
        std::cout << "\nMinVectorElementTask data:";
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
                std::vector<double> aN;
                for (int j = 0; j < i; j++) {
                    aN.push_back(dist(mt));
                }

                // evaluate via reduction:
                reduceExecutionTime.push_back(execute(aN, i, nT, &min_elem_reduction));

                // evaluate via atomic:
                atomicExecutionTime.push_back(execute(aN, i, nT, &min_elem_critical));

                // evaluate consistently:
                consistentExecutionTime.push_back(execute(aN, i, nT, &min_elem_consistently));
            }

            tempStatistic.insert({"atomic", atomicExecutionTime});
            tempStatistic.insert({"reduction", reduceExecutionTime});
            tempStatistic.insert({"consistent", consistentExecutionTime});

            statistic.insert({nT, tempStatistic});
        }
    }


private:
    static double execute(const std::vector<double> &a, int elementCount, int threadCount,
                          const std::function<void(const std::vector<double> &a, int &elementCount)> &func) {
        omp_set_num_threads(threadCount);
        clock_t tStart, tEnd;

        tStart = clock();

        func(a, elementCount);

        tEnd = clock();

        return ((double) (tEnd - tStart) / CLOCKS_PER_SEC);
    }

    static void min_elem_reduction(const std::vector<double> &a, int &elementCount) {
        int minValue = a[0];
        int i;

#pragma omp parallel for default(none) shared(a, elementCount) reduction(min : minValue)
        for (i = 1; i < elementCount; i++) {
            if (a[i] < minValue) {
                minValue = a[i];
            }
        }
    }

    static void min_elem_critical(const std::vector<double> &a, int &elementCount) {
        int minValue = a[0];
        int i;

#pragma omp parallel for default(none) shared(a, elementCount, minValue)
        for (i = 1; i < elementCount; i++) {
            if (a[i] < minValue) {
#pragma omp critical
                if (a[i] < minValue) {
                    minValue = a[i];
                }
            }
        }
    }

    static void min_elem_consistently(const std::vector<double> &a, int &elementCount) {
        int minValue = a[0];
        int i;

        for (i = 1; i < elementCount; i++) {
            if (a[i] < minValue) {
                minValue = a[i];
            }
        }
    }
};

