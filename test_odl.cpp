#include <cstdio>

#include "util.h"
#include "OnlineDictionaryLearning.h"

int main() {

  // Initailize data
  int D = 5, K = 2;
  Real *Xt = (Real*) malloc(D * K * sizeof(Real));
  Real *y = (Real*) malloc(D * sizeof(Real));
  Real *y_r = (Real*) malloc(D * sizeof(Real));
  Idx *beta;
  Real lambda = 0.1;

  prepareData(D, K, 1, true, Xt, y);

  y[0] = 2;
  y[1] = 3;
  y[2] = 1;
  y[3] = 2;
  y[4] = 2;

  DictionaryLearning dl(lambda, D, K);
  dl.iterate(y);

  y[0] = 3;
  y[1] = 4;
  y[2] = 1;
  y[3] = 3;
  y[4] = 3;

  dl.iterate(y);

  dl.recover(y, y_r);
  Real sqr_error = Real(0.0);
  for (int j = 0; j < D; j++)
    sqr_error += (y[j] - y_r[j]) * (y[j] - y_r[j]);
  printf("error = %.3f\n", sqrt(sqr_error));
}
