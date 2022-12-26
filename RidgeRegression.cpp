#include "RidgeRegression.h"

/* Read in training data. The number of rows represents the number of training vectors.
The number of columns indicates the dimension of data. */
void RidgeRegression::read_data(string filename)
{
    ifstream inFile(filename);
    string lineStr;
    mat datain;
    int row = 0, col;
    while(getline(inFile, lineStr)) // read data row by row
	{
        col = 0;
        string line(lineStr);
        string str, str_strip;
        vec a_row;
        a_row.clear();
        while(!line.empty())
		{   
            // elements are separated by space
            size_t pos0 = line.find(' '); 
            
            //cout << "row = " << row << ", col = " << col << endl;
            if(pos0 == string::npos) { // end
                str = line;
                size_t pos = str.find(':');
                if(pos == string::npos){
                    str_strip = str;
                } else {
                    str_strip = str.substr(pos+1);
                }
                double x=stod(str_strip);
                a_row.push_back(x);
                col++;
                break;
            } else {
                str = line.substr(0, pos0);
                size_t pos = str.find(':');
                if(pos == string::npos){
                    str_strip = str;
                } else {
                    str_strip = str.substr(pos+1);
                }
                double x=stod(str_strip);
                a_row.push_back(x);
                col++;
                line = line.substr(pos0+1);
            }
			
        }
        datain.push_back(a_row);
        row++;
    }
    
    MatrixXd X(row, col-1);
    VectorXd Y(row);
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++) {
            if(j == 0) {
                Y(i) = datain[i][j];
            } else {
                X(i, j-1) = datain[i][j];
            }
        }
    }
    data = X;
    label = Y;
    cout << "data: " << data.rows() << " rows, " << data.cols() << " columns" << endl;
}

/* Construct */
RidgeRegression::RidgeRegression(string filename, int iteration, int epsilons) {
    read_data(filename);
    this->iteration = iteration;    
    this->epsilon = epsilon;
}

double RidgeRegression::objective(VectorXd x) {
    return (data * x - label).norm();
}

/* Apply Armijo-Goldstein rule to calculate step size. */
double RidgeRegression::step_size(double a, VectorXd x, VectorXd d, VectorXd g)
{
    double gamma = 0.4;
    while(objective(x + a*d) > (objective(x) + gamma*a*g.transpose()*d)) {
        a *= 0.5;
        x = x + a*d;
    }
    return a;
}

/* Gradient Descent */
void RidgeRegression::gradient(double lambda, double learning_rate)
{
    theta = VectorXd::Zero(data.cols());
    double loss = 0;
    loss_array.clear();
    double a = learning_rate;
    // update the parameter iteratively
    for (int iter = 0; iter < iteration; ++iter)
    {
        // calculate the cost
        loss = (0.5 * (data * theta - label).squaredNorm() + lambda * theta.squaredNorm()) / data.rows();
        loss_array.push_back(loss);
        //std::cout << "Loss at iteration " << iter << ": " << loss << std::endl;

        // calculate the gradient
        VectorXd gradient = (data.transpose() * (data * theta - label) + 2* lambda * theta) / data.rows();
        //std::cout << "Gradient at iteration " << iter << ": " << std::endl;
        
        a = step_size(a, theta, -gradient, gradient);
        theta = theta - a * gradient;
    }
}


void RidgeRegression::conjugate(double lambda, double learning_rate)
{
    theta = VectorXd::Zero(data.cols());
    double loss = 0;
    loss_array.clear();
    MatrixXd Q = data.transpose() * data;
    VectorXd g1;
    VectorXd theta1;
    VectorXd d1;
    double a = learning_rate;
    
    // calculate the gradient
    VectorXd g = (data.transpose() * (data * theta - label) + 2 * lambda * theta) / data.rows();
    // search direction
    VectorXd d = -g;

    for (int iter = 0; iter < iteration; ++iter)
    {
        a = step_size(a, theta, d, g);
        theta1 = theta + a*d;
        loss = (0.5 * (data * theta - label).squaredNorm() + lambda * theta.squaredNorm()) / data.rows();
        loss_array.push_back(loss);
        //std::cout << "Loss at iteration " << iter << ": " << loss << std::endl;
        g1 = (data.transpose() * (data * theta1 - label) + 2*lambda * theta1) / data.rows();
        double beta = ((g1.transpose()*g1)/(g.transpose()*g))(0, 0); // Fletcher-Reeves Formula
        d1 = -g1 + beta*d;
        //std::cout << "Gradient at iteration " << iter << ": " << g << std::endl;
        g = g1;
        d = d1;
        theta = theta1;
    }
}

void RidgeRegression::quasi_newton(double lambda, double learning_rate)
{
    int n = data.rows(), m = data.cols();
    double a = learning_rate;
    double loss = 0;
    loss_array.clear();
    theta = VectorXd::Zero(m);
    VectorXd g = (data.transpose() * (data * theta - label) + lambda * theta) / data.rows();
    VectorXd theta1, g1, d;
    MatrixXd H = MatrixXd::Identity(m, m);
    for(int i = 0; i < iteration; i++){
        if(g.norm() == 0) {
            cout << i << " iterations" << endl;
            break;
        }
        //std::cout << "Gradient at iteration " << i << ": " << g << std::endl;
        d = -H*g;
        a = step_size(a, theta, d, g);
        theta1 = theta + a * d;
        loss = (0.5 * (data * theta - label).squaredNorm() + lambda * theta.squaredNorm()) / data.rows();
        loss_array.push_back(loss);
        //std::cout << "Loss at iteration " << i << ": " << loss << std::endl;
        g1 = (data.transpose() * (data * theta1 - label) + lambda * theta1) / data.rows();
        VectorXd s = theta1 - theta;
        VectorXd y = g1 - g;
        H = H + (s*s.transpose())/(y.transpose()*s) - (H*y*y.transpose()*H.transpose())/(y.transpose()*H*y);
        theta = theta1;
        g = g1;
    }
}

void RidgeRegression::print_data()
{
    for(int i = 0; i < data.rows(); i++) {
        for(int j = 0; j < data.cols(); j++) {
            cout << data(i, j);
            if(j < data.cols() - 1) cout << " ";
            else cout << endl;
        }
    }
}

void RidgeRegression::print_label() 
{
    for(int i = 0; i < label.size(); i++) {
        cout << label(i) << " ";
    }
    cout << endl;
}

void RidgeRegression::print_parameter()
{
    cout << "Parameters: (";
    for(int i = 0; i < theta.size(); i++) {
        cout << theta(i);
        if (i == theta.size() -1) cout << ")" << endl;
        else cout << ", ";
    }
}

