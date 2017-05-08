typedef double Real;
struct DictionaryLearning {
  const int m, k;
  const Real *Dt; // transpose of D
  const Real *A;
  const Real *B;
  const Real *tmp;
  const Real *alpha; // to store the sparse coding of each x //TODO make it sparse vector
  DictionaryLearning(Real lambda_in, Real *Dt_in, int m_in, int k_in);
  // run line 4-7 of algorithm 1
  void iterate(Real *x, int T=1000);

  void sparse_coding(Real *x);
  void update_dict();
}
DictionaryLearning::DictionaryLearning(Real lambda_in, Real *D_in, int m_in, int k_in, int T_in) :
Dt(Dt_in), m(m_in), k(k_in) {}

void DictionaryLearning::update_dict() {
  //double *tmp = (double*) malloc(sizeof(double) * m);
  double threshold = 0.1;
  bool converge = true; //TODO make it a function
  while (converge) {
    converge = false
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

      double base = 1.0 / max(l2Norm(tmp, m), 1.0);

      for (int t = 0; t < m; t++) {
        double temp = base * tmp[t];
        if (fabs(temp - Dt[j*k + t]) > threshold) run = 1;
        Dt[j*k + t] = base * tmp[t];
      }
    }
  }
  //free(tmp);
}

void DictionaryLearning::sparse_coding(Real *x, Real *alpha) {
  //TODO use Lars
}

void DictionaryLearning::iterate(Real *x) {
  sparse_coding(Real *x);
  //TODO update A & B, line 5 & 6 of algorithm 1
  update_dict()
}
