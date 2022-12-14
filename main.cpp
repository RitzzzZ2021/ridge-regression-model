#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "RidgeRegression.h"

using namespace std;

void read_data(string filename, mat &data)
{
    ifstream inFile(filename);
    string lineStr;
    vector<vector<string>> strArray;
    while(getline(inFile, lineStr))
	{
        stringstream ss(lineStr);
        string str, str_strip;
        size_t pos = str.find(':');
        while(getline(ss,str,' '))
		{
            if(pos == string::npos){
                str_strip = str;
            } else {
                str_strip = str.substr(pos+1);
            }
			double x=stod(str_strip);
        }
     }
}

int main()
{

    read_data("dataset/abalone");
}