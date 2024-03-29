#include "RidgeRegression.h"

void output_loss(string filename, vec a)
{
    if(a.empty()) {
        cout << "Empty array" << endl;
        return;
    }
    ofstream outfile;
    outfile.open(filename);
    for(int i = 0; i < a.size(); i++) {
        outfile << a[i];
        if(i < a.size() - 1) outfile << " ";
    }
    outfile.close();
}

void train(RidgeRegression model, double lambda, double learning_rate)
{
    cout << "---Gradient Descent---" << endl;
    model.gradient(lambda, learning_rate);
    model.print_parameter();
    output_loss("gradient_loss", model.loss_at_each_iteration());

    cout << endl << "---Conjucate Descent---" << endl;
    model.conjugate(lambda, learning_rate);
    model.print_parameter();
    output_loss("conjucate_loss", model.loss_at_each_iteration());

    cout << endl << "---Quasi-Newton---" << endl;
    model.quasi_newton(lambda, learning_rate);
    model.print_parameter();
    output_loss("quasiNewton_loss", model.loss_at_each_iteration());
}

int main() {
    string filename = "dataset\\abalone";

    // 定义岭回归的惩罚项系数
    double lambda = 0.1;

    // 定义学习率
    double learning_rate = 0.1;
    int epsilon = 0.000001;
    int max_iter = 100;

    //RidgeRegression model1("dataset\\abalone_scale", max_iter, epsilon);
    //RidgeRegression model2("dataset\\bodyfat_scale", max_iter, epsilon);
    RidgeRegression model3("dataset\\housing_scale", max_iter, epsilon);
    //train(model1, lambda, learning_rate);
    //train(model2, lambda, learning_rate);
    train(model3, lambda, learning_rate);
}