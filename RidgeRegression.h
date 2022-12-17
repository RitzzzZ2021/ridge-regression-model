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
    int iteration; // training iterations
    void read_data(string filename);
    
public:
    RidgeRegression(string filename, int iteration);

    MatrixXd return_data() { return this->data;}

    void gradient(double lambda, double learning_rate);

    // for debug
    void print_data();
    void print_label();
    void print_parameter();
};


#endif