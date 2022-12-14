#ifndef _RIDGEREGRESSION_H
#define _RIDGEREGRESSION_H

#include <vector>
typedef vector<double> vec;
typedef vector<vec> mat;

class RidgeRegression {
private:
    mat data;
    int n; // times of recursion 
    static double cost();
    static double p();
    vec theta;
public:
    RidgeRegression(double x[], double y[], int n);
    void train(double alpha, int iteration);
    double predict(double x);
};


#endif