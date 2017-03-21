#include "mathOperations.h"

/*
	Computes and returns L1 norm of a vector
	X : n x 1 array
*/
double l1Norm(double * X, int n){
	double Xnorm = 0;
	int i;
	for(i = 0; i < n; i++){
		Xnorm += fabs(X[i]);
	}
	return Xnorm;
}

/*
	Computes and returns L2 norm of a vector
	X : n x 1 array
*/
double l2Norm(double * X, int n){
	double Xnorm = 0;
	int i;
	for(i = 0; i < n; i++){
		Xnorm += X[i]*X[i];
	}
	return sqrt(Xnorm);
}

/*
	Computes the product M x v and stores values in u
	M : cols x rows array
	v : rows x 1 array
	u : cols x 1 array 
*/
double matVecProd(double * M, double * v, double * u, int cols, int rows){
	int i,j;	
	for(i = 0; i < rows; i++){
		u[i] = 0;
		for(j = 0; j < cols; j++){
			u[i] += M[i * rows + j] * v[j] ;
		}
	}
}

/*
	Computes difference of vectors and stores in third vector
	a : size n double array
	b : size n double array
	c : size n double array
*/
void vecDiff(double * a, double * b, double * c, int n){
	int i;
	for(i = 0; i < n; i++){
		c[i] = a[i] - b[i];
	}
}

/*
	Computes cross product of two vectors
	v : sizeOfV double array
	w : sizeOfW double array
	M : size sizeOfV x sizeOfW double array
*/
void cross(double * v, double *w, double *M, int sizeOfV, int sizeOfW){
	int i,j;
	for(i = 0; i < sizeOfV; i++){
		for(j = 0; j < sizeOfW; j++){
			M[i*sizeOfW+j] = v[i]*w[j];
		}
	}
}

/*
	Computes cross product of two vectors
	v : sizeOfV double array
	w : sizeOfW double array
	M : size sizeOfV x sizeOfW double array
*/
double dot(double * v, int n){
	int i;
	double sum;
	sum = 0;
	for(i = 0; i < n; i++){
		sum += v[i]*v[i];
	}
	return sum;
}

/*
	Computes trace of
	A : square matrix 
	size : size of matrix A
*/
double trace(double *A, int size){
	int i;
	double sum;
	sum = 0;
	
	for(i = 0; i <= size*size; i=i+size+1){
		sum += A[i];
	}
	
	return sum;
}

