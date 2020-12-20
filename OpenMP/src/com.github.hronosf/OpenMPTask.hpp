#pragma once

#include <random>
#include <bits/stdc++.h>

class OpenMPTask {
public:
    void run() {
        execute_task();
        printDataInPythonFormat();
        std::cout << '\n';
    };

    virtual ~OpenMPTask() = default;

protected:
    std::vector<std::string> operations;
    std::map<int, std::map<std::string, std::vector<double>>> statistic;

    // test data:
    std::vector<int> threadCount = {2, 4, 8, 16};
    std::vector<int> elementCount = {100000, 1000000, 10000000, 100000000};

    // method to run task:
    virtual void execute_task() {};

    void printDataInPythonFormat() {
        for (auto const &operationName: operations) {
            std::cout << "\n" << operationName << "_data =";

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
    };
};