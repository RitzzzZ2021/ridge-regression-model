#include "RidgeRegression.h"

int main() {
    string filename = "dataset\\abalone";

    // 定义岭回归的惩罚项系数
    double lambda = 1;
    // 定义学习率
    double learning_rate = 0.1;
    // 定义最大迭代次数
    int maxIter = 100;
    int epsilon = 0.0001;

    RidgeRegression model(filename, maxIter, epsilon);

    cout << "---Gradient Descent---" << endl;
    model.gradient(lambda, learning_rate);
    model.print_parameter();
    cout << endl << "---Conjucate Descent---" << endl;
    model.conjucate(0, learning_rate);
    model.print_parameter();
/*
    model.newton(lambda, 0.1, 0.5);
    model.print_parameter();*/

}