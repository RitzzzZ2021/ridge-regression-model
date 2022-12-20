#include "RidgeRegression.h"

void RidgeRegression::read_data(string filename)
{
    ifstream inFile(filename);
    string lineStr;

    mat datain;
    int row = 0, col;
    while(getline(inFile, lineStr))
	{
        col = 0;
        string line(lineStr);
        string str, str_strip;
        vec a_row;
        a_row.clear();
        while(!line.empty())
		{   
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
    cout << "data: " << data.rows() << " " << data.cols() << endl;
    cout << "label: " << label.rows() << " " << label.cols() << endl;
}

RidgeRegression::RidgeRegression(string filename, int iteration, int epsilons) {
    read_data(filename);
    this->iteration = iteration;    
    this->epsilon = epsilon;
}

double RidgeRegression::objective(VectorXd x) {
    return (data * x - label).norm();
}

double RidgeRegression::step_size(double a, VectorXd x, VectorXd d, VectorXd g)
{
    double gamma = 0.4;
    while(objective(x + a*d) > (objective(x) + gamma*a*g.transpose()*d)) {
        a *= 0.5;
        x = x + a*d;
    }
    return a;
}


void RidgeRegression::gradient(double lambda, double learning_rate)
{
    theta = VectorXd::Zero(data.cols());
    double loss = 0;
    loss_array.clear();
    double a = learning_rate;
    // 迭代更新模型参数
    for (int iter = 0; iter < iteration; ++iter)
    {
        // 计算损失函数
        loss = (0.5 * (data * theta - label).squaredNorm() + lambda * theta.squaredNorm()) / data.rows();
        loss_array.push_back(loss);
        //std::cout << "Loss at iteration " << iter << ": " << loss << std::endl;

        // 计算梯度
        VectorXd gradient = (data.transpose() * (data * theta - label) + 2* lambda * theta) / data.rows();
        /*if(gradient.norm() < epsilon && iter > 20) {
            cout << iter << " iterations" << endl;
            break;
        }*/
        //std::cout << "Gradient at iteration " << iter << ": " << std::endl;
        
        // 更新模型参数
        a = step_size(a, theta, -gradient, gradient);
        theta = theta - a * gradient;
    }
}

void RidgeRegression::conjucate(double lambda, double learning_rate)
{
    theta = VectorXd::Zero(data.cols());
    double loss = 0;
    loss_array.clear();
    MatrixXd Q = data.transpose() * data;
    VectorXd g1;
    VectorXd theta1;
    VectorXd d1;
    double a = learning_rate;
    
    VectorXd g = (data.transpose() * (data * theta - label) + 2 * lambda * theta) / data.rows();
    VectorXd d = -g;
    /*
    double a = -((g.transpose() * d)/(d.transpose() * Q * d))(0, 0); // length of step
    //cout << a1.rows() << " " << a1.cols() << " " << g1.rows() << " " << g1.cols() << endl;

    theta = theta - a*g;
    VectorXd old_g = g;
    */
    // 迭代更新模型参数
    for (int iter = 0; iter < iteration; ++iter)
    {
        //a = ((g.transpose() * d)/(d.transpose() * Q * d))(0, 0); // step length
        a = step_size(a, theta, d, g);
        // 计算损失函数
        theta1 = theta + a*d;
        loss = (0.5 * (data * theta - label).squaredNorm() + lambda * theta.squaredNorm()) / data.rows();
        loss_array.push_back(loss);
        //std::cout << "Loss at iteration " << iter << ": " << loss << std::endl;
        g1 = (data.transpose() * (data * theta1 - label) + 2*lambda * theta1) / data.rows();
        /*if(g.norm() < epsilon && iter > 20) {
            theta = theta1;
            cout << iter << " iterations" << endl;
            break;
        }*/

        double beta = ((g1.transpose()*g1)/(g.transpose()*g))(0, 0); // Fletcher-Reeves Formula
        d1 = -g1 + beta*d;
        //std::cout << "Gradient at iteration " << iter << ": " << g << std::endl;
        /*if((theta1 - theta).norm() < epsilon) {
            theta = theta1;
            cout << iter << " iterations" << endl;
            break;
        }*/
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
        /*if(g.norm() < epsilon && i > 20) {
            cout << i << " iterations" << endl;
            break;
        }*/
        //std::cout << "Gradient at iteration " << i << ": " << g << std::endl;
        d = -H*g;
        a = step_size(a, theta, d, g);
        theta1 = theta + a * d;
        loss = (0.5 * (data * theta - label).squaredNorm() + lambda * theta.squaredNorm()) / data.rows();
        loss_array.push_back(loss);
        //cout << "data * theta - label" << (data * theta - label) << endl;
        //if(loss<epsilon) {cout << "=0" <<endl ; break;}
        //std::cout << "Loss at iteration " << i << ": " << loss << std::endl;
        g1 = (data.transpose() * (data * theta1 - label) + lambda * theta1) / data.rows();
        VectorXd s = theta1 - theta;
        VectorXd y = g1 - g;
        H = H + (s*s.transpose())/(s.transpose()*y) - (H*y*y.transpose()*H.transpose())/(y.transpose()*H*y);
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

