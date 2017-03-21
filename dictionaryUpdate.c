#include "dictionaryUpdate.h"

inline double max(double a, double b) {return a > b? a : b;}

void dictionaryUpdate(double * D, const double *A, const double *B, int m, int k) {

  double *u = (double*) malloc(sizeof(double) * m);
  double threshold = 0.1;
  int run = 1;
  while (run) {
    run = 0;
    for (int j = 0; j < k; j++) {
      // b_j => B[][j];
      for (int t = 0; t < m; t++) {
        double da_j_t = 0.0;
        // D[t][] * A[][j];
        for (int tt = 0; tt < k; tt++) {
          da_j_t += D[t*k + tt] * A[tt*k + j];
        }
        u[t] = (B[t*k + j] - da_j_t) / A[j*k + j] + D[t*k + j];
      }

      double base = 1.0 / max(l2Norm(u, m), 1.0);

      for (int t = 0; t < m; t++) {
        double temp = base * u[t];
        if (fabs(temp - D[t*k + j]) > threshold) run = 1;
        D[t*k + j] = base * u[t];
      }
    }
  }
  free(u);
}

int main() {}
