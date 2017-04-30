#include <math.h>

double l1Norm(double *X, int n);
double l2Norm(double *X, int n);
void matVecProd(double *M, double *v, double *u, int cols, int rows);
void vecDiff(double *a, double *b, double *c, int n);
void cross(double *v, double *w, double *M, int sizeOfV, int sizeOfW);
double dot(double *v, int n);
double trace(double *A, int size);
void transpose(double * A, double * AT, size_t n, size_t m);