#include<time.h>
#include <omp.h>
#include <random>
#include <bits/stdc++.h>

void
dot_product_reduction(const std::vector<double> &a, const std::vector<double> &b, int &elementCount) {
    double sum = 0;
    int i;

#pragma omp parallel for default(none) shared(a, b, elementCount) reduction(+:sum)
    for (i = 0; i < elementCount; i++) {
        sum += a[i] * b[i];
    }
}

void dot_product_atomic(const std::vector<double> &a, const std::vector<double> &b, int elementCount) {
    double sum = 0;
    int i;

#pragma omp parallel for default(none) shared(a, b, elementCount, sum)
    for (i = 0; i < elementCount; i++) {
#pragma omp atomic
        sum += a[i] * b[i];
    }
}

void dot_product_consistently(const std::vector<double> &a, const std::vector<double> &b, int elementCount) {
    double sum = 0;

    for (int i = 0; i < elementCount; i++) {
        sum += a[i] * b[i];
    }
}

double
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

int main() {
    // to get random numbers:
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(30000, 100000);

    // test data:
    std::vector<int> threadCount = {1, 4, 8, 16};
    std::vector<int> elementCount = {100000, 1000000, 10000000, 100000000};

    std::map<int, std::map<std::string, std::vector<double>>> statistic;

    for (int &nT : threadCount) {
        std::vector<double> reduceExecutionTime;
        std::vector<double> atomicExecutionTime;
        std::vector<double> consistentExecutionTime;
        std::map<std::string, std::vector<double>> tempStatistic;

        for (int i = 0; i < elementCount.size(); i++) {
            // dump data:
            std::vector<double> aN, bN;
            for (int j = 0; j < elementCount[i]; j++) {
                aN.push_back(dist(mt));
                bN.push_back(j + dist(mt));
            }

            // evaluate via reduction:
            reduceExecutionTime.push_back(execute(aN, bN, elementCount[i], nT, &dot_product_reduction));

            // evaluate via atomic:
            atomicExecutionTime.push_back(execute(aN, bN, elementCount[i], nT, &dot_product_atomic));

            // evaluate consistently:
            consistentExecutionTime.push_back(execute(aN, bN, elementCount[i], nT, &dot_product_consistently));
        }

        tempStatistic.insert({"Atomic", atomicExecutionTime});
        tempStatistic.insert({"Reduction", reduceExecutionTime});
        tempStatistic.insert({"Consistent", consistentExecutionTime});

        statistic.insert({nT, tempStatistic});
    }


    std::vector<std::string> operations = {"Atomic", "Reduction", "Consistent"};
    for (auto const &operationName: operations) {
        std::cout << "\n" << operationName << ":\n";

        std::cout << "{";
        for (auto const &dataMap : statistic) {
            auto const &mapNameToExecTimeVector = dataMap.second;

            const std::vector<double> &data = mapNameToExecTimeVector.at(operationName);

            std::cout << dataMap.first << ":[";
            for (int i = 0; i < data.size(); ++i) {
                std::cout << data[i];
                if (i != data.size() - 1) {
                    std::cout << ",";
                }
            }

            std::cout << "]";

            if (dataMap.first != 16) {
                std::cout << ",";
            }
        }
        std::cout << "}";
    }
}