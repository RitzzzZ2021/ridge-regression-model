#ifndef _RIDGEREGRESSION_H
#define _RIDGEREGRESSION_H

#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;
using namespace Eigen;

typedef vector<double> vec;
typedef vector<vec> mat;

class RidgeRegression {
private:
    MatrixXd data;
    VectorXd label;
    VectorXd theta; // parameters
    int epsilon;
    int iteration; // training iterations
    vec loss_array;
    void read_data(string filename);
    double objective(VectorXd x);
    double step_size(double a, VectorXd x, VectorXd d, VectorXd g);
    
public:
    RidgeRegression(string filename, int iteration, int epsilon);

    MatrixXd return_data() { return this->data; }
    vec loss_at_each_iteration() { return loss_array; }

    void gradient(double lambda, double learning_rate);
    void conjugate(double lambda, double learning_rate);
    void quasi_newton(double lambda, double learning_rate);

    // for debug
    void print_data();
    void print_label();
    void print_parameter();
};


#endif