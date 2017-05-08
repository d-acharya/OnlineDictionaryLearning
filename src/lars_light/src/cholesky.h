//-*-c++-*-
#ifndef CHOLESKY_H
#define CHOLESKY_H

/* Cholesky utility routines
 *
 * Functions that handle updating cholesky factorization of matrix
 * when you add or remove a column.

 * Basically this corresponds to the normal loop that happens in
 * cholesky anyways.

 * Update Method: assume we have L for X and that X' = X with an
 * additional row/column. Given L and the new column/row, compute the
 * last row of L ( column of L' ).  We assume that I can access the
 * row.  I'm supposed to be computing in L and that A has the new
 * column that I need L and A can be the same, which means you're
 * overwriting the lower triangle of A L and A should be row major,
 * because we're dotting rows together.  Then j = index of the new row
 * of L == the new column of A
 *
 * LARS++, Copyright (C) 2007 Varun Ganapathi, David Vickery, James
 * Diebel, Stanford University
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301 USA.
*/

#include <numeric>
#include <cstdio>
#include <cmath>

#include "util.h"

const Real EPSILON = 1e-9;

/////////////
// Methods //
/////////////

// Updates the cholesky (L) after having added data to row (j)
// assume L = nxn matrix
// current L is of nxn size, but the memory is stored in a NxN data structure
inline void update_cholesky(Real* L, int j, const int n, const int N) {
  Real sum = 0.0;
  Real eps_small = EPSILON;
  int i, k;
  for (i = 0; i < j; ++i) {
    sum = L[j * N + i];
    for (k = 0; k < i; ++k) {
      sum -= L[i * N + k] * L[j * N + k];
    }
    L[j * N + i] = sum / L[i * N + i];
  }
  sum = L[j * N + i];
  for (k = 0; k < i; k++) {
    sum -= L[i * N + k] * L[j * N + k];
  }
  if (sum <= 0.0) sum = eps_small;
  L[j * N + j] = sqrt(sum);
}

// Backsolve the cholesky (L) for unknown (x) given a right-hand-side (b)
// and a total number of rows/columns (n).
//
// Solves for x in Lx=b
// x can be b
// => Assume L is nxn, x and b is vector of length n
// assume L = nxn matrix
// current L is of nxn size, but the memory is stored in a NxN data structure
inline void backsolve(const Real *L, Real *x, const Real *b, const int n, const int N) {
  int i, k;
  Real sum;
  for (i = 0; i < n; i++) {
    for (sum = b[i], k = i-1; k >= 0; k--) {
      sum -= L[i * N + k] * x[k];
    }
    x[i] = sum / L[i * N + i];
  }

  for (i = n-1; i>= 0; i--) {
    for (sum = x[i], k = i+1; k < n; k++) {
      sum -= L[k * N + i] * x[k];
    }
    x[i] = sum / L[i * N + i];
  }
}

// Downdates the cholesky (L) by removing row (id), given that there
// are (nrows) total rows.
//
// Given that A = L * L';
// We correct L if row/column id is removed from A.
// We do not resize L in this function, but we set its last row to 0, since
// L is one row/col smaller after this function
// n is the number of rows and cols in the cholesky
// void downdate_cholesky(Real *L, int nrows, int id) {
//   Real a = 0, b = 0, c = 0, s = 0, tau = 0;
//   int lth = nrows - 1;
//   int L_rows = nrows, L_cols = nrows;
//
//   int i, j;
//   for (i = id; i < lth; ++i) {
//     for (j = 0; j < i; ++j) {
//       L[i * L_cols + j] = L[(i+1) * L_cols + j];
//     }
//     a = L[(i+1) * L_cols + i];
//     b = L[(i+1) * L_cols + (i + 1)];
//     if (b == 0) {
//       L[i * L_cols + i] = a;
//       continue;
//     }
//     if (fabs(b) > fabs(a)) {
//       tau = -a / b;
//       s = 1.0 / sqrt(1.0 + tau*tau);
//       c = s*tau;
//     } else {
//       tau = -b / a;
//       c = 1.0 / sqrt(1.0 + tau*tau);
//       s = c * tau;
//     }
//     L[i * L_cols + i] = c * a - s * b;
//     for (j = i+2; j <= lth; ++j) {
//       a = L[j * L_cols + i];
//       b = L[j * L_cols + (i+1)];
//       L[j * L_cols + i] = c*a - s*b;
//       L[J * L_cols + (i+1)] = s*a + c*b;
//     }
//   }
//   for (i = 0; i <= lth; ++i) {
//     L[lth * L_cols + i] = 0.0;
//   }
// }
#endif
