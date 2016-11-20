#include <iostream>
#include<fstream>
#include <string> 
#include <vector>
#include <math.h>
#include <time.h>
using namespace std;
vector<vector<double>> dataMat;
vector<int> dataLabel;

vector<vector<double>> trainingSet;
vector<int> trainingLabel;
vector<vector<double>> testingSet;
vector<int> testingLabel;

void loadDataSet()
{
	ifstream in;
	string filename = "testSet.txt";
	string line_temp;
	int line = 0;
	in.open(filename, ios::in);//ios::in 表示以只读的方式读取文件
	while (getline(in, line_temp))
	{
		line++;
	}
	in.close();
	in.open(filename, ios::in);
	for (int i = 0; i < line; i++)
	{
		vector<double>temp_mat_line;
		temp_mat_line.push_back(1.0);
		for (int j = 0; j < 2; j++)
		{
			double temp_mat;
			in >> temp_mat;
			temp_mat_line.push_back(temp_mat);
			in.get();
		}
		dataMat.push_back(temp_mat_line);
		int temp_mat;
		in >> temp_mat;
		dataLabel.push_back(temp_mat);
		in.get();
	}
	in.close();
	
}
double sigmoid(double inX)
{
	return 1.0 / (1 + exp(-inX));
}

vector<double> gradAscent(vector<vector<double>>dataMatin, vector<int>classLabels)
{
	int m = dataMatin.size();
	int n = dataMatin[0].size();
	double alpha = 0.001;
	int maxCycles = 500;
	vector<double>weights(n, 1);
	vector<double>h(m,0);
	for (int i = 0; i < maxCycles; i++)
	{
		vector<double>error;
		for(int j =0;j<m;j++)
		{
			double mat_X_wei=0;
			for(int k=0;k<n;k++)
			{
				mat_X_wei+=weights[k]*dataMat[j][k];
			}
			h[j]=sigmoid(mat_X_wei);
		}
		for(int j=0;j<m;j++)
		{
			error.push_back(classLabels[j]-h[j]);
		}
		for(int j=0;j<n;j++)
		{
			double al_Mat_err=0;
			for(int k=0;k<m;k++)
			{
				al_Mat_err+=dataMat[k][j]*error[k];
			}
			al_Mat_err*=alpha;
			weights[j] = weights[j]+ al_Mat_err;
		}
	}
	return weights;
}
//梯度下降法
vector<double> stoGrandAscent0(vector<vector<double>>dataMatrix,vector<int>classLabels)
{
	int m = dataMatrix.size();
	int n = dataMatrix[0].size();
	double alpha = 0.01;
	vector<double>weights(n, 1);	
	for (int j = 0; j<m; j++)
	{
		double mat_X_wei = 0;
		for (int k = 0; k<n; k++)
		{
			mat_X_wei += weights[k] * dataMat[j][k];
		}
		double error=classLabels[j]-sigmoid(mat_X_wei);
		for (int k = 0; k<n; k++)
		{
			weights[k] = weights[k] + alpha* dataMat[j][k] * error;
		}
	}
	return weights;
}
//随机梯度下降法
vector<double> stoGrandAscent1(vector<vector<double>>dataMatrix, vector<int>classLabels,int numIter = 150)
{
	int m = dataMatrix.size();
	int n = dataMatrix[0].size();
	srand((unsigned)time(NULL));
	vector<double>weights(n, 1);
	for (int i = 0; i < numIter; i++)
	{
		vector<int> vec_rand;
		for (int x = 0; x < m; x++)
			vec_rand.push_back(x);
		for (int j = 0; j < m; j++)
		{
			double mat_X_wei = 0;
			double alpha = 0.01 + 4 / (1.0 + j + i);
			/******************************/
			int rand_index = rand() % vec_rand.size();
			vector<int>::iterator it = vec_rand.begin() + rand_index;
			int now_index = *it;
			vec_rand.erase(it);
			/*****************************/
			for (int k = 0; k < n; k++)
			{
				mat_X_wei += weights[k] * dataMatrix[now_index][k];
			}
			double error = classLabels[now_index] - sigmoid(mat_X_wei);
			for (int k = 0; k < n; k++)
			{
				weights[k] = weights[k] + alpha* dataMatrix[now_index][k] * error;
			}
		}
	}
	return weights;
}

bool classifyVector(vector<double>inX, vector<double>weights)
{
	double sum = 0.0;
	for (int i = 0; i < inX.size(); i++)
	{
		sum += inX[i] * weights[i];
	}
	if (sigmoid(sum)>0.5)
		return 1;
	else
		return 0;
}

double colicTest()
{
	ifstream in1;
	string filename1 = "horseColicTraining.txt";
	string line_temp;
	int line = 0;
	in1.open(filename1, ios::in);//ios::in 表示以只读的方式读取文件
	while (getline(in1, line_temp))
	{
		line++;
	}
	in1.close();
	in1.open(filename1, ios::in);
	for (int i = 0; i < line; i++)
	{
		vector<double>temp_mat_line;
		for (int j = 0; j < 21; j++)
		{
			double temp_mat;
			in1 >> temp_mat;
			temp_mat_line.push_back(temp_mat);
			in1.get();
		}
		trainingSet.push_back(temp_mat_line);
		double temp_mat;
		in1 >> temp_mat;
		trainingLabel.push_back(int(temp_mat));
		in1.get();

	}

	in1.close();
	vector<double>trainWeights = stoGrandAscent1(trainingSet, trainingLabel, 100);
	int errorCount = 0;
	double numTestVec = 0.0;

	ifstream in2;
	string filename2 = "horseColicTest.txt";
	line = 0;
	in2.open(filename2, ios::in);//ios::in 表示以只读的方式读取文件
	while (getline(in2, line_temp))
	{
		numTestVec += 1;
		line++;
	}
	in2.close();
	in2.open(filename2, ios::in);
	for (int i = 0; i < line; i++)
	{
		vector<double>temp_mat_line;
		for (int j = 0; j < 21; j++)
		{
			double temp_mat;
			in2 >> temp_mat;
			temp_mat_line.push_back(temp_mat);
			in2.get();
		}
		testingSet.push_back(temp_mat_line);
		double temp_mat;
		in2 >> temp_mat;
		testingLabel.push_back(int(temp_mat));
		if (int(classifyVector(temp_mat_line, trainWeights)) != int(temp_mat))
			errorCount++;
		in2.get();
	}
	in2.close();
	return double(errorCount) / numTestVec;
}

void multiTest()
{
	int numTests = 10;
	double errorSum = 0.0;
	for (int i = 0; i < numTests; i++)
		errorSum += colicTest();
	cout << "everage error rate is： " << errorSum / numTests << endl;
}

int main()
{
//	loadDataSet();
	//gradAscent(dataMat, dataLabel);
	//stoGrandAscent1(dataMat, dataLabel);
	multiTest();
	cin.get();
	return 0;
}