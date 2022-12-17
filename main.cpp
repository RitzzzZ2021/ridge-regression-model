#include "RidgeRegression.h"

int main() {
    string filename = "dataset\\abalone";

    // 定义岭回归的惩罚项系数
    double lambda = 1.0;
    // 定义学习率
    double learning_rate = 0.1;
    // 定义最大迭代次数
    int maxIter = 50;

    RidgeRegression model(filename, maxIter);
    model.gradient(lambda, learning_rate);
    model.print_parameter();
}