#include "mathOperations.h"

/*
	NOTE: use assert to check if input is compatible
	e.g. in case of index compatibility in matrix*matrix multiplication
*/

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

/* This is replaced by mvm. Currently this exists so that your code does not break down.
	Computes the product M x v and stores values in u
	M : cols x rows array
	v : rows x 1 array
	u : cols x 1 array
*/
void matVecProd(double * M, double * v, double * u, int cols, int rows){
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
double dot(double * v, double * w, int n){
	int i;
	double sum;
	sum = 0.;
	for(i = 0; i < n; i++){
		sum += v[i]*w[i];
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

/*
	n,m: A is n*m and B is m*n and C is  n*n
	TODO: check implementation



void gramMatrix(double * A, double * AtA, double rows, double cols){
	mmm(A, true, A, AtA, rows, cols, cols);
}
*/

/*
	C=A*B multiplication
	transposeA indicates whether to transpose A
	A is matrix of size Arows*Acols
	B is matrix of size Acols*Bcols
	C is matrix of size Arows*Bcols
*/
void mmm(double *A, bool transposeA, double *B, double *C, int Arows, int Acols, int Bcols){
    int i, j, k;

	if (transposeA){
        for(i = 0; i < Acols; i++){
			for(j = 0; j < Bcols; j++){
				for(k = 0; k < Arows; k++){
					C[i*Bcols+j] += A[k*Acols+i]*B[k*Bcols+j];
				}
			}
		}
	} else {
		for(i = 0; i < Arows; i++){
			for(j = 0; j < Bcols; j++){
				for(k = 0; k < Acols; k++){
					C[i*Bcols+j] += A[i*Acols+k]*B[k*Bcols+j];
				}
			}
		}
	}

}


/*
	a*M*v multiplication
	a is a constant
	M is a rows*cols matrix
	v is vector
	trans indicates whether to use transpose of M
*/
void amvm(double a, double * M, bool trans, double * v, double * w, int rows, int cols){
  int i ,j;
  if (trans){
    for(i = 0; i < cols; i++){
      w[i] = 0;
      for(j = 0; j < rows; j++){
        w[i] += M[j * cols + i] * v[j] ;
      }
      w[i] *=a;
    }
  } else {
    for(i = 0; i < rows; i+=1){
      w[i] = 0;
      for(j = 0; j < cols; j++){
        w[i] += M[i * cols + j] * v[j] ;
      }
      w[i] *=a;
    }
  }
}


/*
	same as amvm but without multiplication by constant.
	M*v multiplication
	M is a rows*cols matrix
	v is vector
	trans indicates whether to use transpose of M
*/
void mvm(double * M, bool trans, double * v, double * w, int rows, int cols){
	int i ,j;
  if (trans){
    for(i = 0; i < cols; i++){
      w[i] = 0;
      for(j = 0; j < rows; j++){
        w[i] += M[j * cols + i] * v[j] ;
      }
      w[i] *=a;
    }
  } else {
    for(i = 0; i < rows; i+=1){
      w[i] = 0;
      for(j = 0; j < cols; j++){
        w[i] += M[i * cols + j] * v[j] ;
      }
      w[i] *=a;
    }
  }
}

/*
	Y=a*X+Y addition
	a is a scalar
	X is a vector of length size
	Y is a vector of length size
*/
void daxpy(double a, double * X, double * Y, double * Y, int size){
	int i;
	for(i = 0; i < size; i++){
		Y[i] = a*X[i] + Y[i];
	}
}