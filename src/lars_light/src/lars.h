#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cassert>

#include "util.h"
#include "mathOperations.h"
#include "cholesky.h"

#ifndef LARS_H
#define LARS_H

struct Lars {

  int K, D;
  int active_size, active_itr;

  const Real *Xt; //a KxD matrix, transpose of X;
  const Real *y; //a Dx1 vector
  Idx *beta, *beta_old; // current beta and old beta solution [Not sorted]

  int *active; // active[i] = position beta of active param or -1
  Real *c; //
  Real *w; // sign c[active_set]
  Real *u; // unit direction of each iteration
  Real *a; // store Xt * u
  Real *L; // lower triangular matrix of the gram matrix of X_A (pxp)

  Real lambda, lambda_new, lambda_old;

  Real *tmp; // temporary storage for active correlations

  FILE *fid;


  // allocate all needed memory
  // input fixed numbers
  Lars(int D_in, int K_in, Real lambda_in);

  ~Lars();

  // input y for computation
  void set_y(const Real *y_in);

  void solve();

  bool iterate();

//  void calculateParameters();

  void getParameters(Idx** beta_out) const; // get final beta

  Real compute_lambda(); // compute lambda given beta
};

Lars::Lars(int D_in, int K_in, Real lambda_in):
    D(D_in), K(K_in), lambda(lambda_in) {

  beta = (Idx*) calloc(D, sizeof(Idx));
  beta_old = (Idx*) calloc(D, sizeof(Idx));

  // Initializing
  active_size = fmin(K, D);
  active = (int*) malloc(K * sizeof(int));

  c = (Real*) calloc(K, sizeof(Real));
  w = (Real*) calloc(active_size, sizeof(Real));
  u = (Real*) calloc(D, sizeof(Real));
  a = (Real*) calloc(K, sizeof(Real));
  L = (Real*) calloc(active_size * active_size, sizeof(Real));
  tmp = (Real*) calloc((K>D?K:D), sizeof(Real));

  fid = stderr;
}

void Lars::set_y(const Real *y_in) {
  y = y_in; 

  memset(beta, 0, D*sizeof(Idx));
  memset(beta_old, 0, D*sizeof(Idx));

  active_itr = 0;
  memset(active, -1, K * sizeof(int));
  memset(w, 0, active_size * sizeof(Real));
  memset(u, 0, D * sizeof(Real));
  memset(a, 0, K * sizeof(Real));
  memset(L, 0, active_size * active_size * sizeof(Real));

  mvm(Xt, false, y, c, K, D);

  fid = stderr;
}

Lars::~Lars() {
  print("Lars() DONE\n");
}

bool Lars::iterate() {
  if (active_itr > active_size) return false;

  Real C = 0.0;
  int cur = -1;
  for (int i = 0; i < K; ++i) {
    print("c[%d]=%.3f active[i]=%d\n", i, c[i], active[i]);
    if (active[i] != -1) continue;
    if (fabs(c[i]) > C) {
      cur = i;
      C = fabs(c[i]);
    }
  }
  // All remainging C are 0
  if (cur == -1) return false;

  print("Active set size is now %d\n", active_itr + 1);
  print("Activate %d column with %.3f value\n", cur, C);

  print("active[%d]=%d cur=%d D=%d\n", active_itr, active[cur], cur, D);
  assert(active[cur] == -1 and active_itr < D);

  active[cur] = active_itr;
  beta[active_itr] = Idx(cur, 0);

  // calculate Xt_A * Xcur, Matrix * vector
  // new active row to add to gram matrix of active set
  for (int i = 0; i <= active_itr; ++i) {
    tmp[i] = dot(Xt + cur * D, Xt + beta[i].id * D, D);
  }
  // L[active_itr][] = tmp[];
  for (int i = 0; i <= active_itr; ++i) {
    L[active_itr*active_size + i] = tmp[i];
  }

  print("L before cholesky\n");
  for (int i = 0; i <= active_itr; ++i) {
    for (int j = 0; j <= active_itr; ++j) {
      print("%.3f  ", L[i * active_size + j]);
    }
    print("\n");
  }
  print("\n");

  update_cholesky(L, active_itr, active_itr+1, active_size);

  print("L after cholesky\n");
  for (int i = 0; i <= active_itr; ++i) {
    for (int j = 0; j <= active_itr; ++j) {
      print("%.3f  ", L[i * active_size + j]);
    }
    print("\n");
  }
  print("\n");

  // set w[] = sign(c[])
  for (int i = 0; i <= active_itr; ++i) {
    w[i] = sign(c[beta[i].id]);
  }

  // w = R\(R'\s)
  // w is now storing sum of all rows? in the inverse of G_A
  backsolve(L, w, w, active_itr+1, active_size);

  // AA is is used to finalize w[]
  // AA = 1 / sqrt(sum of all entries in the inverse of G_A);
  Real AA = 0.0;
  for (int i = 0; i <= active_itr; ++i) {
    AA += w[i] * sign(c[beta[i].id]);
  }
  AA = 1.0 / sqrt(AA);
  print("AA: %.3f\n", AA);

  // get the actual w[]
  for (int i = 0; i <= active_itr; ++i) {
    w[i] *= AA;
  }
  print("w solved :");
  for (int i = 0; i < D; ++i) print("%.3f ", w[i]);
  print("\n");

  // get a = X' X_a w
  // Now do X' (X_a w)
  // can store a X' X_a that update only some spaces when adding new col to
  // activation set
  // Will X'(X_a w) be better? // more
  // X' (X_a w) more flops less space?
  // (X' X_a) w less flops more space?
  memset(a, 0, K*sizeof(Real));
  memset(u, 0, D*sizeof(Real));
  // u = X_a * w
  for (int i = 0; i <= active_itr; ++i) {
    axpy(w[i], &Xt[beta[i].id * D], u, D);
  }
  // a = X' * u
  mvm(Xt, false, u, a, K, D);

  print("u : ");
  for (int i = 0; i < D; i++) print("%.3f ", u[i]);
  print("\n");

  print("a : ");
  for (int i = 0; i < K; i++) print("%.3f ", a[i]);
  print("\n");

  Real gamma = C / AA;
  int gamma_id = cur;
  if (active_itr < active_size) {
    print("C=%.3f AA=%.3f\n", C, AA);
    for (int i = 0; i < K; i++) {
      if (active[i] != -1) continue;
      Real t1 = (C - c[i]) / (AA - a[i]);
      Real t2 = (C + c[i]) / (AA + a[i]);
      print("%d : t1 = %.3f, t2 = %.3f\n", i, t1, t2);

      if (t1 > 0 and t1 < gamma) gamma = t1, gamma_id=i;
      if (t2 > 0 and t2 < gamma) gamma = t2, gamma_id=i;
    }
  }
  print("gamma = %.3f from %d col\n", gamma, gamma_id);

  // add gamma * w to beta
  for (int i = 0; i <= active_itr; ++i)
    beta[i].v += gamma * w[i];

  // update correlation with a
  for (int i = 0; i < K; ++i)
    c[i] -= gamma * a[i];

  print("beta: ");
  for (int i = 0; i <= active_itr; ++i) print("%d %.3f ", beta[i].id, beta[i].v);
  print("\n");

  active_itr++;
  return true;
}

void Lars::solve() {
  int itr = 0;
  while (iterate()) {
    // compute lambda_new
    print("=========== The %d Iteration ends ===========\n\n", itr);

    //calculateParameters();
    lambda_new = compute_lambda();

    print("---------- lambda_new : %.3f lambda_old: %.3f lambda: %.3f\n", lambda_new, lambda_old, lambda);
    for (int i = 0; i < active_itr; i++)
      print("%d : %.3f %.3f\n", beta[i].id, beta[i].v, beta_old[i].v);

    if (lambda_new > lambda) {
      lambda_old = lambda_new;
      memcpy(beta_old, beta, active_itr * sizeof(Idx));
    } else {
      Real factor = (lambda_old - lambda) / (lambda_old - lambda_new);
      for (int j = 0; j < active_itr; j++) {
//        beta[j].v = beta_old[j].v * (1.f - factor) + factor * beta[j].v;
        beta[j].v = beta_old[j].v + factor * (beta[j].v - beta_old[j].v);
      }
      break;
    }
    itr++;
  }
  print("LARS DONE\n");
}


//void Lars::calculateParameters() {
//  for (int i = 0; i < active_itr; i++) {
//    print("beta[%d] = %.3f\n", beta[i].id, beta[i].v);
//  }
//
//  mvm(Xt, false, y, tmp, D, K);
//
//  Real *tmp2 = (Real*) calloc(active_itr, sizeof(Real));
//  for (int i = 0; i < active_itr; ++i) {
//    tmp2[i] = tmp[beta[i].id];
//  }
//  backsolve(L, tmp2, tmp2, active_itr, active_size);
//
//  for (int i = 0; i < active_itr; ++i) {
//    beta[i].id = beta[i].id;
//    beta[i].v = tmp2[i];
//  }
//
//  free(tmp2);
//}

void Lars::getParameters(Idx** beta_out) const {
  *beta_out = beta;
}


// computes lambda given beta, lambda = max(abs(2*X'*(X*beta - y)))
inline Real Lars::compute_lambda() {
  // compute (y - X*beta)
  memcpy(tmp, y, D * sizeof(Real));
  for (int i = 0; i < active_itr; i++) {
    for (int j = 0; j < D; j++)
      tmp[j] -= Xt[beta[i].id * D + j] * beta[i].v;
  }
  // compute X'*(y - X*beta)
  Real max_lambda = Real(0.0);
  for (int i = 0; i < K; ++i) {
    max_lambda = fmax(max_lambda, fabs(dot(Xt + i * D, tmp, D)));
  }
  return max_lambda;
}

#endif
