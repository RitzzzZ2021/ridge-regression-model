#ifndef _RIDGEREGRESSION_H
#define _RIDGEREGRESSION_H

#include <Eigen/Core>
#include <Eigen/QR>
#include <vector>
using namespace std;
using namespace Eigen;

typedef vector<double> vec;
typedef vector<vec> mat;

class RidgeRegression {
private:
    MatrixXd data;
    VectorXd target;
    VectorXd theta; // parameters
    int iteration; // times of recursion 
    static double cost();
    static double p();
    
public:
    RidgeRegression(int iteration){
        this->iteration = iteration;
    }
    MatrixXd return_data() { return this->data;}
    void train(double alpha, int iteration);
    double predict(double x);
};


#endif