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
}

#endif
