#include <math.h>

#include "util.h"

#ifndef MATH_OPERATION_H
#define MATH_OPERATION_H
// Input:
// Output:
// description
// Used by:
Real l1Norm(Real *X, int n);

// Input:
// Output:
// description
// Used by:
Real l2Norm(Real *X, int n);

// Input:
// Output:
// description
// Used by:
void matVecProd(Real *M, Real *v, Real *u, int cols, int rows);

// Input:
// Output:
// description
// Used by:
void vecDiff(Real *a, Real *b, Real *c, int n);

// Input:
// Output:
// description
// Used by:
void cross(Real *v, Real *w, Real *M, int sizeOfV, int sizeOfW);

// Input:
// Output:
// description
// Used by:
Real dot(const Real *v, int n);

// Input:
// Output:
// description
// Used by:
Real dot(const Real * v, const Real * w, int n);

// Input:
// Output:
// description
// Used by:
Real trace(Real *A, int size);

// Input:
// Output:
// description
// Used by:
void transpose(Real * A, Real * AT, int n, int m);

// Input:
// Output:
// description
// Used by:
void mmm(const Real *A, bool transposeA, const Real *B, Real *C, int Arows, int Acols, int Bcols);

// Input:
// Output:
// description
// Used by:
void amvm(Real a, const Real * M, bool trans, const Real * v, Real * w, int rows, int cols);

// Input:
// Output:
// description
// Used by: lars.h(iterate())
void mvm(const Real * M, bool trans, const Real * v, Real * w, int rows, int cols);


// Input:
// Output:
// description
// Used by:
void axpy(Real a, const Real * X, Real * Y, int size);

#endif
