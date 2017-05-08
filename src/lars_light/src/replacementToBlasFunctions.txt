// line 161
// cblas_ddot(N, &X[N*c1], 1, &X[N*c2], 1);
dot(&X[N*c1], &X[N*c2], N);

// line 172
//cblas_dgemv(CblasColMajor
//cols = p
//rows = N
//matVecProd(double * X, double * y, double * Xty, int p, int N)
//cblas_dgemv(CblasColMajor, CblasTrans, N, p, 1.0, X, N, y, 1, 0.0, Xty, 1);
mvm(X, true, y, Xty, N, p);

// line 180
//cblas_dgemm(CblasColMajor
//cols = p
//rows = N
//cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, p, p, N, 1.0, X, N, X, N, 0.0, XtX_buf, p);
mmm(X, transpose, X, XtX_buf, p, N, p);

// line 197
//cblas_daxpy()
//cblas_daxpy(p, wval[i], XtX_col[beta[i].first], 1, a, 1);
daxpy(wval[i], XtX_col[beta[i].first], a, p);

// line 203
//cblas_daxpy(N, wval[i], &X[beta[i].first*N], 1, Xw, 1);
daxpy(wval[i], &X[beta[i].first*N], Xw, N);

// line 205
// now do X'*(X*w)
// cblas_dgemv(CblasColMajor,CblasTrans,N,p,1.0,X,N,Xw,1,0.0,a,1);
mvm(X, true, Xw, a, N, p);

// line 222
//cblas_daxpy(p, -beta[i].second, XtX_col[beta[i].first], 1, tmp_p, 1)
daxpy(-beta[i].second, XtX_facecol[beta[i].first], tmp_p, p);

// line 229
//cblas_daxpy(N, -beta[i].second, &X[N*beta[i].first], 1, Xw, 1);
daxpy(-beta[i].second, &X[N*beta[i].first], Xw, N);

// line 231
// now compute 2*X'*Xw = 2*X'*(y - X*beta)
//cblas_dgemv(CblasColMajor,CblasTrans,N,p,2.0,X,N,Xw,1,0.0,tmp_p,1);
amvm(2.0, X, true, Xw, tmp_p, N, p);

// file dense_cholesky:
// line 92/95
//cblas_ddot( N, a, 1, b, 1);
dot(a, b, N);
