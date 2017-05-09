#include "util.h"
#include "lars.h"

#ifndef DICTIONARY_LEARNING_H
#define DICTIONARY_LEARNING_H
struct DictionaryLearning {
  const int m, k;
  const int T; // # iteration of dictionary update
  Real *Dt; // kxm, transpose of D(mxk)
  Real *At;  // kxk
  Real *Bt;  // mxk
  Real *tmp; // m-vector
  Lars *lars_ptr;
  const Real epsilon;

  DictionaryLearning(Real lambda_in, int m_in, int k_in);

  void update_dict();
  void sparse_coding(Real *const x); // for training
  void sparse_coding(Real *const x, Real *const alpha); //for testing, return dense_alpha
  void recover(Real *const x, Real*const x_r);
  void iterate(Real *const x);
};

#endif
