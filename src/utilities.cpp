#include "utilities.h"
/*
	Computes cost to be used for LARS in step 4 of Algorithm 1.
	D : rows x cols matrix
	x : rows x 1 vector	
	a : rows x 1 vector	
	cols : Number of features per observation
	rows : Number of observations
	lambda : regularization parameter
*/
double larsCost(double * D, double * x, double * a, int cols, int rows, double lambda){
	
	double * diff = (double *)mkl_calloc(rows * sizeof(double), sizeof(double));
	double * prod = (double *)mkl_calloc(rows * sizeof(double), sizeof(double));
	
	matVecProd(D, a, prod, cols, rows);
	vecDiff(x, prod, diff, rows);
	
	double cost;
	cost = 0.5 * l2Norm(diff) + lambda * l1Norm(x);

	mkl_free(diff);
	mkl_free(prod);

	return cost;
}

/*	
	Cost to be optimized while updating dictionary. Step 7 in Algorithm 1.
	D : rows x cols matrix
	A : cols x cols matrix
	B : rows x rows matrix
*/
// double updateCost(D, A, B, rows, cols){
// 	return (Trace(D^TDA) - Trace(D^TB));
// 	
// 	
// }

/*	
	Cost to be optimized while updating dictionary. Step 7 in Algorithm 1.
	D : rows x cols matrix
	A : cols x cols matrix
	B : rows x rows matrix
*/
// void updateDict(D, A, B, int rows, int cols){
// 	
// 	
// }
