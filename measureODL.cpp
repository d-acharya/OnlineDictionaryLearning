#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

#include <x86intrin.h>

#include "util.h"
#include "lars.h"
#include "mathOperations.h"
#include "rdtsc.h"
#include "timer.h"
#include ""

#define RUNS 10
#define CYCLES_REQUIRED 1e7

void measure(const int D, const int K, Real *Xt, Real *y, Real *beta, Real *beta_h, Real lambda, Timer timer) {
  tsc_counter start, end;
  double cycles = 0.;
  size_t num_runs = RUNS;

  CPUID(); RDTSC(start); CPUID(); RDTSC(end);
  CPUID(); RDTSC(start); CPUID(); RDTSC(end);
  CPUID(); RDTSC(start); CPUID(); RDTSC(end);

  // Warm-up phase: determine number of runs needed
  // to ignore the timing overhead
  DictionaryLearning dict( timer);

  while(1) {
    CPUID(); RDTSC(start);
    for (int i = 0; i < num_runs; ++i) {
      lars.set_y(y);
      lars.solve();
    }
    CPUID(); RDTSC(end);

    cycles = (double) (COUNTER_DIFF(end, start));

    if (cycles >= CYCLES_REQUIRED) break;

    num_runs *= 2;
  }

  CPUID(); RDTSC(start);
  for (int i = 0; i < num_runs; ++i) {
      lars.set_y(y);
      lars.solve();
  }
  CPUID(); RDTSC(end);

  cycles = (double) (COUNTER_DIFF(end, start)) / num_runs;

  printf("cycles used : %.3f\n", cycles);

  #ifdef VERIFY
    lars.getParameters(beta_h);
    Real sqr_err = get_square_error(beta, beta_h, K);
    if (sqr_err > 1e-5) 
      printf("\nVALIDATION FAILED: get error %.3f in lars with lambda %.3f\n\n", sqr_err, lambda);
  #endif
}

void set_value(const int D, const int K, Real *Xt, Real *y,
Real *beta) {
  prepare_Xt(D, K, true, Xt);
  prepare_Beta(K, 1, beta);
  memset(y, 0, sizeof(Real) * D);
  
  for (int i = 0; i < K; i++) {
    axpy(beta[i], &Xt[i * D], y, D);
  }
}

int main() {
  const int Max_D = 1000, Max_K = 1000;
  Real lambda = 0;
  Timer timer(END_ITR);

  Real *Xt = (Real*) malloc(sizeof(Real) * Max_D * Max_K);
  Real *y = (Real*) malloc(sizeof(Real) * Max_D);
  Real *beta = (Real*) malloc(sizeof(Real) * Max_K);
  Real *beta_h = (Real*) malloc(sizeof(Real) * Max_K);

  for (int i = 100; i < Max_D; i += 100) {
    printf("\nD = %d, K = %d\n", i , i);
    timer.reset();
    set_value(i, i, Xt, y, beta);
    measure(i, i, Xt, y, beta, beta_h, lambda, timer);
    timer.print();
  }
}
