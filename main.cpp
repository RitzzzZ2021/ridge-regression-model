#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "RidgeRegression.h"

MatrixXd read_data(string filename)
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
    cout << "row = " << row << ", col = " << col << endl;
    MatrixXd model(row, col);
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++) {
            model(i, j) = datain[i][j];
            //cout << model(i, j) << " ";
        }
        //cout << endl;
    }
    return model;
}


using namespace std;
using namespace Eigen;

// 定义损失函数
double loss(const MatrixXd &X, const VectorXd &y, const VectorXd &w, double lambda) {
  int n = X.rows();
  int p = X.cols();

  // 计算残差
  VectorXd r = X * w - y;

  // 计算损失函数
  return 0.5 * r.squaredNorm() / n + 0.5 * lambda * w.squaredNorm();
}

// 定义梯度函数
VectorXd gradient(const MatrixXd &X, const VectorXd &y, const VectorXd &w, double lambda) {
  int n = X.rows();
  int p = X.cols();

  // 计算残差
  VectorXd r = X * w - y;

  // 计算梯度
  return X.transpose() * r / n + lambda * w;
}

int main() {
    read_data("dataset\\abalone");
    /*
    // 定义训练数据
    int n = 10, p = 5;
    MatrixXd X(n, p);
    VectorXd y(n);

    // 随机生成训练数据
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
        X(i, j) = j + i * 0.1;
        }
        y(i) = 3 * X(i, 0) + 2 * X(i, 1) + X(i, 2) + 0.5 * X(i, 3) + 0.1 * X(i, 4);
    }

    // 定义岭回归的惩罚项系数
    double lambda = 1.0;

    // 定义初始参数
    VectorXd w = VectorXd::Zero(p);

    // 定义学习率
    double alpha = 0.01;

    // 定义最大迭代次数
    int maxIter = 10000;

   */
}