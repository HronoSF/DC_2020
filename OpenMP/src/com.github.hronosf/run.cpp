#include "./1/MinVectorElementTask.cpp"
#include "./2/VectorDotProductTask.cpp"
#include "./3/IntergalValueTask.cpp"
#include "./4/MaxMinMatrixValue.cpp"
#include "./6/ReductionTask.cpp"

int main() {
    (new MinVectorElementTask())->run();
    (new VectorDotProductTask())->run();
    (new IntergalValueTask())->run();
    (new MaxMinMatrixValue())->run();
    (new ReductionTask())->run();
};