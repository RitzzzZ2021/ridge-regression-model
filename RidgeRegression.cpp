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
    this->data = X;
    this->label = Y;
    cout << "data: " << data.rows() << " " << data.cols() << endl;
    cout << "label: " << label.rows() << " " << label.cols() << endl;   
    //print_data();
    //print_label(); 
}

RidgeRegression::RidgeRegression(string filename, int iteration) {
    read_data(filename);
    this->theta = VectorXd::Zero(data.cols()); // initialize
    this->iteration = iteration;    
}

void RidgeRegression::gradient(double lambda, double learning_rate)
{
    // 迭代更新模型参数
    for (int iter = 0; iter < iteration; ++iter)
    {
        // 计算损失函数
        double loss = 0.5 * (data * theta - label).squaredNorm() / data.rows() + 0.5 * lambda * theta.squaredNorm();
        std::cout << "Loss at iteration " << iter << ": " << loss << std::endl;

        // 计算梯度
        VectorXd gradient = (data.transpose() * (data * theta - label)) / data.rows() + lambda * theta;
        std::cout << "Gradient at iteration " << iter << ": " << std::endl;
        for(int i = 0; i < gradient.size(); i++) {
            cout << gradient(i) << " ";
        }
        cout << endl;

        // 更新模型参数
        theta = theta - learning_rate * gradient;
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

/*
//牛顿法
MatrixXd newton(MatrixXd feature, MatrixXd label, int iterMax, double sigma, double delta)
{
	double epsilon = 0.1;
	int n = feature.cols();
	MatrixXd w = MatrixXd::Zero(n, 1);
	MatrixXd g;
	MatrixXd G;
	MatrixXd d;
	double m;
	int it = 0;
	while (it <= iterMax)
	{
		g = first_derivative(feature, label, w);
		if (epsilon >= g.norm())
		{
			break;			
		}
		G = second_derivative(feature);
		d = -G.inverse() * g;
		m = get_min_m(feature, label, sigma, delta, d, w, g);
		w = w + pow(sigma, m) * d;
		it++;
	}
	return w;
}

//获取最小m
int get_min_m(MatrixXd feature, MatrixXd label, double sigma, double delta, MatrixXd w, MatrixXd d, MatrixXd g)
{
	int m = 0;
	MatrixXd w_new;
	MatrixXd left;
	MatrixXd right;
	while (true)
	{
		w_new = w + pow(sigma, m) * d;
		left = get_error(feature, label, w_new);
		right = get_error(feature, label, w) + delta * pow(sigma,m) * g.transpose() * d;
		if (left(0,0) <= right(0,0))
		{
			break;
		}
		else
		{
			m += 1;
		}
	}
	return m;
}

//计算误差
MatrixXd get_error(MatrixXd feature, MatrixXd label, MatrixXd w)
{
	return (label - feature * w).transpose() * (label - feature * w) / 2;
}

//一阶导
MatrixXd first_derivative(MatrixXd feature, MatrixXd label, MatrixXd w)
{
	int m = feature.rows();
	int n = feature.cols();
	MatrixXd g = MatrixXd::Zero(n, 1);
	MatrixXd err;
	for (int i = 0; i < m; i++)
	{
		err = label.block(i,0,1,1) - feature.row(i) * w;
		for (int j = 0; j < n; j++)
		{
			g.row(j) -= err * feature(i, j);
		}
	}
	return g;
}

//二阶导
MatrixXd second_derivative(MatrixXd feature)
{
	int m = feature.rows();
	int n = feature.cols();
	MatrixXd G = MatrixXd::Zero(n, n);
	MatrixXd x_left;
	MatrixXd x_right;
	for (int i = 0; i < m; i++)
	{
		x_left = feature.row(i).transpose();
		x_right= feature.row(i);
		G += x_left * x_right;
	}
	return G;
}
*/