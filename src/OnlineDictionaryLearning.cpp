#include "OnlineDictionaryLearning.h"
#include "mathOperations.h"
#include <cstdlib>

#ifndef DICTIONARY_LEARNING_CPP
#define DICTIONARY_LEARNING_CPP

DictionaryLearning::DictionaryLearning(Real lambda_in, int m_in, int k_in) :
m(m_in), k(k_in), epsilon(1e-4) {
  Dt = (Real*) malloc(k * m * sizeof(Real)); // TODO Initailize Dt with independent cols
  At = (Real*) malloc(k * k * sizeof(Real));
  Bt = (Real*) malloc(k * m * sizeof(Real));
  tmp = (Real*) calloc(m, sizeof(Real));
  lars_ptr = new Lars(Dt, m, k, lambda_in);
}

void DictionaryLearning::update_dict() {
  bool converge = false;
  while (!converge) {
    converge = true;
    for (int j = 0; j < k; j++) {
      amvm(At[j*k + j], Dt, true, At+j*k, tmp, k, m); //tmp = Daj
      vecDiff(Bt+j*m, tmp, tmp, m);
      axpy(-At[j*k + j], tmp, Dt+j*m, m);
      Real norm = l2Norm(Dt+j*m, m);
      if (norm > 1)
        dot(1.0/norm, Dt+j*m, m);
      if (l2Norm(tmp, m) >= epsilon * At[j*k + j])
        converge = false;
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
      At[alpha[j].id * k + alpha[i].id] += alpha[i].v * alpha[j].v;
    for (int j = 0; j < m; j++)
      Bt[alpha[i].id * m + j] += x[j] * alpha[i].v;
  }
  update_dict();
}

#endif
