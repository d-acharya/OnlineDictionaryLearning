#include <math.h>

double l1Norm(double *X, int n);
double l2Norm(double *X, int n);
void matVecProd(double *M, double *v, double *u, int cols, int rows);
void vecDiff(double *a, double *b, double *c, int n);
void cross(double *v, double *w, double *M, int sizeOfV, int sizeOfW);
double dot(double *v, int n);
double trace(double *A, int size);
void transpose(double * A, double * AT, size_t n, size_t m);
void mmm(double *A, bool transposeA, double *B, double *C, int Arows, int Acols, int Bcols);
void amvm(double a, double * M, bool trans, double * v, double * w, int rows, int cols);
void mvm(double * M, bool trans, double * v, double * w, int rows, int cols);
void axpy(double a, double * X, double * Y, double * Y, int size);
