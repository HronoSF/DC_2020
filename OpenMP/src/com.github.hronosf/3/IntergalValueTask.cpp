#include<time.h>
#include <omp.h>
#include <random>
#include <bits/stdc++.h>
#include "../OpenMPTask.hpp"

class IntergalValueTask : public OpenMPTask {
public:
    IntergalValueTask() {
        std::cout << "\nIntergalValueTask data:\n";
    }

protected:
    void execute_task() override {
        operations = {"atomic", "reduction", "consistent", "critical"};

        int a = -10;
        int b = 0;

        for (int &nT : threadCount) {
            std::vector<double> reduceExecutionTime;
            std::vector<double> atomicExecutionTime;
            std::vector<double> consistentExecutionTime;
            std::vector<double> criticalExecutionTime;
            std::map<std::string, std::vector<double>> tempStatistic;

            for (int i : elementCount) {
                double h = (b - a) / (double) i;

                // evaluate via reduction:
                reduceExecutionTime.push_back(execute(h, i, nT, &integral_value_reduction));

                // evaluate via atomic:
                atomicExecutionTime.push_back(execute(h, i, nT, &integral_value_atomic));

                // evaluate critical:
                criticalExecutionTime.push_back(execute(h, i, nT, &integral_value_critical));

                // evaluate consistently:
                consistentExecutionTime.push_back(execute(h, i, nT, &integral_value_consistent));
            }

            tempStatistic.insert({"atomic", atomicExecutionTime});
            tempStatistic.insert({"reduction", reduceExecutionTime});
            tempStatistic.insert({"consistent", consistentExecutionTime});
            tempStatistic.insert({"critical", criticalExecutionTime});

            statistic.insert({nT, tempStatistic});
        }
    };

private:

    static double f(double x) {
        return cos(x);
    }

    static double execute(double h, int elementCount, int threadCount,
                          const std::function<void(int elementCount, double h)> &func) {
        omp_set_num_threads(threadCount);
        clock_t tStart, tEnd;

        tStart = clock();

        func(elementCount, h);

        tEnd = clock();

        return ((double) (tEnd - tStart) / CLOCKS_PER_SEC);
    }

    static void integral_value_reduction(int elementCount, double h) {
        double sum = 0;
        int i;
        double x;

#pragma omp parallel for default(none) shared(h, elementCount) private(x) reduction(+:sum)
        for (i = 0; i < elementCount; i++) {
            x = h * i;
            sum += f(x);
        }

        sum *= h;
    }


    static void integral_value_atomic(int elementCount, double h) {
        double sum = 0;
        int i;
        double x;

#pragma omp parallel for default(none) private (x) shared(elementCount, sum, h)
        for (i = 0; i < elementCount; i++) {
            x = h * i;
#pragma omp atomic
            sum += f(x);
        }

        sum *= h;
    }

    static void integral_value_critical(int elementCount, double h) {
        double sum = 0;
        int i;
        double x;

#pragma omp parallel for default(none) shared(elementCount, h, sum) private(x)
        for (i = 0; i < elementCount; i++) {
            x = h * i;
#pragma omp critical
            sum += f(x);
        }

        sum *= h;
    }

    static void integral_value_consistent(int elementCount, double h) {
        double sum = 0;
        int i;
        double x;

        for (i = 0; i < elementCount; i++) {
            x = h * i;
            sum += f(x);
        }

        sum *= h;
    }
};