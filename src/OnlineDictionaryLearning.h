#include "util.h"
#include "lars.h"
#ifndef DICTIONARY_LEARNING_H
#define DICTIONARY_LEARNING_H
struct DictionaryLearning {
  const int m, k;
  Real *Dt; // kxm, transpose of D(mxk)
  Real *A;  // kxk
  Real *B;  // mxk
  Real *tmp; // m-vector
  Lars *lars_ptr;
  DictionaryLearning(Real lambda_in, int m_in, int k_in);
  void iterate(Real *const x);
  void sparse_coding(Real *const x); // for training
  void sparse_coding(Real *const x, Real *const alpha); // for testing, return dense alpha
  void recover(Real *const x, Real*const x_r);
  void update_dict();
};

DictionaryLearning::DictionaryLearning(Real lambda_in, int m_in, int k_in) :
m(m_in), k(k_in) {
  Dt = (Real*) malloc(m * k * sizeof(Real));
  A = (Real*) malloc(k * k * sizeof(Real));
  B = (Real*) malloc(m * k * sizeof(Real));
  tmp = (Real*) calloc(m, sizeof(Real));
  lars_ptr = new Lars(Dt, m, k, lambda_in); //TODO refactor Lars for y
}

void DictionaryLearning::update_dict() {
  Real threshold = 1e-4;
  bool converge = false;
  while (!converge) {
    converge = true;
    for (int j = 0; j < k; j++) {
      // b_j => B[][j];
      for (int t = 0; t < m; t++) {
        double da_j_t = 0.0;
        // D[t][] * A[][j];
        for (int tt = 0; tt < k; tt++) {
          da_j_t += Dt[tt*k + t] * A[tt*k + j];
        }
        tmp[t] = (B[t*k + j] - da_j_t) / A[j*k + j] + Dt[j*k + t];
      }

      double base = 1.0 / fmax(l2Norm(tmp, m), 1.0);

      for (int t = 0; t < m; t++) {
        double temp = base * tmp[t];
        if (fabs(temp - Dt[j*k + t]) > threshold)
          converge = false;
        Dt[j*k + t] = base * tmp[t];
      }
    }
  }
}

void DictionaryLearning::sparse_coding(Real *const x, Real *const alpha) {
  sparse_coding(x);
  // cope the Lars.beta into a dense vector alpha
  Idx *beta = lars_ptr->beta;
  int l = lars_ptr->active_itr;
  memset(alpha, 0, k * sizeof(Real));
  for (int i = 0; i < l; i++)
    alpha[beta[i].id] = beta[i].v;
}

void DictionaryLearning::recover(Real *const x, Real*const x_r) {
    sparse_coding(x);
    Idx *alpha = lars_ptr->beta;
    int l = lars_ptr->active_itr;
    memset(x_r, 0, m * sizeof(Real));
    for (int i = 0; i < l; i++)
      axpy(alpha[i].v, Dt + alpha[i].id * m, x_r, m);
}

void DictionaryLearning::sparse_coding(Real *const x) {
  lars_ptr->set_y(x); //reset temporary data in Lars, and set Lars.y to x
  lars_ptr->solve();
}

void DictionaryLearning::iterate(Real *const x) { // run line 4-7 of algorithm 1
  sparse_coding(x);
  Idx *alpha = lars_ptr->beta;
  int l = lars_ptr->active_itr;
  // A += alpha*alpha.T, B += x*alpha.T
  for (int i = 0; i < l; i++) {
    for (int j = 0; j < l; j++)
      A[alpha[j].id * k + alpha[i].id] += alpha[i].v * alpha[j].v;
    for (int j = 0; j < m; j++)
      B[j * k + alpha[i].id] += x[j] * alpha[i].v;
  }
  update_dict();
}
#endif
