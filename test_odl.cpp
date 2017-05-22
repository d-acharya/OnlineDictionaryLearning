#include <cstdio>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "timer.h"
#include "timer_id.h"
#include "util.h"
#include "OnlineDictionaryLearning.h"
#include <fstream>
#include <iostream>
#include <string>
using namespace std;
int main() {

  // Initailize data
  int D = 10, K = 15, r = 5000;
  Real *y = (Real*) malloc(D * r * sizeof(Real));
  Real *y_r = (Real*) malloc(D * sizeof(Real));
  Idx *beta;
  Real lambda = 0.5;
  Timer timer(END_ITR);
  DictionaryLearning dl(lambda, D, K, timer);

  fstream myfile;
  cout << "open " << "../y_"+to_string(D)+"_"+to_string(r) << endl;
  myfile.open("../y_"+to_string(D)+"_"+to_string(r), ios_base::in);
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < D; j++) {
      myfile >> y[D*i + j];
    }
  }
  myfile.close();

  Real error(0.0), min_error(1e10), max_error(0.0);
  int count;
  for (int t = 0; t < 1000; t++) {
    for (int i = 0; i < r/2; i++) {
      if (l2Norm(y+D*i, D) == 0) continue;
      dl.iterate(y+D*i);
    }
    error = 0.0, min_error = 1e10, max_error = 0.0;
    count = 0;
    for (int i = r/2; i < r; i++) {
      if (l2Norm(y+D*i, D) == 0) continue;
      dl.recover(y+D*i, y_r);
      vecDiff(y+D*i, y_r, y_r, D);
      Real norm = l2Norm(y_r, D);
      error += norm;
      count++;
      min_error = fmin(min_error, norm);
      max_error = fmax(max_error, norm);
    }
    error = sqrt(error);
    error /= count;
    cout << "iteration " << t << ", test mse = " << error
         << "(max=" << max_error << ", min=" << min_error << ")" << endl;
  }

  cout << " # non-zeros vector = " << count << endl;
  cout << "Learned Dictionary:" << endl;
  for (int i = 0; i < K; i++) {
    cout << "[";
    for (int j = 0; j < D; j++) {
      cout << dl.Dt[D*i + j] << ", ";
    }
    cout << "]," << endl;
  }
  cout << endl;

  cout << "Use true dictionary to decode y..." << endl;
  cout << "open " << "../X_"+to_string(D)+"_"+to_string(K) << endl;
  myfile.open("../X_"+to_string(D)+"_"+to_string(K), ios_base::in);
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < D; j++) {
      myfile >> dl.Dt[D*i + j];
    }
  }
  myfile.close();
  error = 0.0, min_error = 1e10, max_error = 0.0;
  count = 0;
  for (int i = r/2; i < r; i++) {
    if (l2Norm(y+D*i, D) == 0) continue;
    dl.recover(y+D*i, y_r);
    vecDiff(y+D*i, y_r, y_r, D);
    Real norm = l2Norm(y_r, D);
    error += norm;
    count++;
    min_error = fmin(min_error, norm);
    max_error = fmax(max_error, norm);
  }
  error = sqrt(error);
  error /= count;
  cout << "the best MSE possible = " << error;
  cout << "(max=" << max_error << ", min=" << min_error << ")" << endl << endl;

  cout << "Use random dictionary to decode y..." << endl;
  prepare_Xt(D, K, true, dl.Dt);
  error = 0.0, min_error = 1e10, max_error = 0.0;
  count = 0;
  for (int i = r/2; i < r; i++) {
    if (l2Norm(y+D*i, D) == 0) continue;
    dl.recover(y+D*i, y_r);
    vecDiff(y+D*i, y_r, y_r, D);
    Real norm = l2Norm(y_r, D);
    error += norm;
    count++;
    min_error = fmin(min_error, norm);
    max_error = fmax(max_error, norm);
  }
  timer.print(1);
  error = sqrt(error);
  error /= count;
  cout << "the worst MSE possible = " << error;
  cout << "(max=" << max_error << ", min=" << min_error << ")" << endl;
  /*
  cout << "Predicted Beta:" << endl;
  for (int i = 0; i < r; i++) {
    dl.sparse_coding(y+D*i);
    int l = dl.lars_ptr->active_itr;
    beta = dl.lars_ptr->beta;
    for (int j = 0; j < l; j++)
      cout << "(" << i << ", " << beta[j].id << ") " << beta[j].v << endl;
  }
  */

  /*

  prepare_Xt(D, K, true, dl.Dt);

  y[0] = 2;
  y[1] = 3;
  y[2] = 1;
  y[3] = 2;
  y[4] = 2;

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
  */
}
