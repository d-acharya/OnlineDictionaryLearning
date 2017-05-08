#include <vector>
#include <algorithm>
/*
*
* n :	length of y
* X : 	n * m matrix	
* beta : vector of length m
*/

/*
* Things to consider for optimization:
* transpose or double loop
*/

void lars(double * y, size_t n, size_t m, double * X, double * beta, double lambda){
	double * mu = (double*) calloc(m,sizeof(double));
	size_t k  = 0;
	std::vector<size_t> A;
	for(k=0; k<n; k++){
		double * c = (double*) calloc(m,sizeof(double));
		size_t max_j = 0;
		double C = -INF;
		for(size_t i = 0; i < m; i++){
			for(size_t j = 0; j < n; j++){
				c[i] += X[m * j + i] * (y[j] - mu[j]);
			}
			
			if(c[i]>C){
				max_j = i;
				C = c[i];
			}
		}
		A.push_back(max_j);

		double * X_A = (double*) calloc(n*A.size(),sizeof(double));
		
		for(size_t i = 0; i < A.size(); i++){
			for(size_t j = 0; j < n; j++){
				X_A[m*j+i] = X[m*j+A[i]];
			}
		}


		double * G_A = (double*) calloc(A.size()*A.size(),sizeof(double));
		// replace by using transpose of X_A
		for(size_t i = 0; i < A.size(); i++){
			for(size_t j = 0; j < A.size(); j++){
				for(size_t l = 0; l < n; l++){
					G_A[i*A.size()+j] = X_A[m*l + A[i]] * X_A[m*l + A[j]];
				}
			}
		}
		double * G_A_INV = (double*) calloc(A.size()*A.size(),sizeof(double));
		// to be implemented
		matrixInverse(G_A, G_A_INV);

		double * y_k = (double*) calloc(n,sizeof(double));
		double * X_A_transpose = (double*) calloc(n*A.size(), sizeof(double));
		double * X_A_transpose_y = (double*) calloc(A.size(), sizeof(double));
		transpose(X_A, X_A_transpose);
		matVecProd(X_A_transpose, y, X_A_transpose_y);

		matVecProd(G_A_INV, X_A_transpose_y, y_k);
		double * a = (double*) calloc(n, sizeof(double));
		double * y_k_minus_mu_k = (double*) calloc(n, sizeof(double));
		vecDiff(y_k, mu_k, y_k_minus_mu_k);
		matVecProd(X_A_transpose, y_k_minus_mu_k);
		
		double gamma = +INF;
		sort(A.begin(), A.end());
		
		for (int i = 0, j = 0; i < n; i++) {
			if (i == A[j]) {
				j++
				continue;
			}
			
			if ((C - c[j])/(C - a[j]) > 0){
				gamma = min(gamma, (C - c[j])/(C - a[j]));
			}
		
			if (min(gamma, (C + c[j])/(C + a[j])) > 0){
				gamma = min(gamma, (C + c[j])/(C + a[j]));
			}
		}

		// computation of betaK
		double * s_A = (double*) calloc(A.size(), sizeof(double));
		for(int i = 0; i < A.size(); i++){
			s_A[i] = (double(0) < c(A[i])) - (c(A[i]) < double(0));
		}

		double * G_INV_s_A = (double*) calloc(A.size(), sizeof(double));
		//double * sG_INV_s = (double*) calloc(A.size(), sizeof(double));;
		matVecProd(G_INV, s_A, G_INV_s_A);
		double sG_INV_s = dot(s_A, G_INV_s_A);
		double coeff = gamma*1./sqrt(sG_INV_s);
		for(int i = 0; i < m; i++ ){
			betaK[i] = coeff*G_INV_s_A[m];
		}


	}


}

