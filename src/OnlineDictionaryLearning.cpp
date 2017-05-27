#include "OnlineDictionaryLearning.h"
#include "mathOperations.h"
#include <cstdlib>
#include "immintrin.h"
#include <iostream>
#include "mkl.h"
#include "mkl_blas.h"

#ifndef DICTIONARY_LEARNING_CPP
#define DICTIONARY_LEARNING_CPP

#define is_aligned(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)




DictionaryLearning::DictionaryLearning(Real lambda_in, int m_in, int k_in, Timer &timer) :
m(m_in), k(k_in), epsilon(1e-2), T(1), timer(timer) {

  timer.start(DICT_INIT);
  Dt = (Real*) aligned_alloc(k * m, k * m * sizeof(Real));
  prepare_Xt(m, k, true, Dt);
  At = (Real*) aligned_alloc(k * k, k * k * sizeof(Real));
  Bt = (Real*) aligned_alloc(k * m, k * m * sizeof(Real));
  tmp=(Real *)_mm_malloc(m*sizeof(Real), 4*sizeof(Real));
  std::cout<<"m: "<<m<<"Size of Real "<<sizeof(Real)<<" Size of Array "<<m*sizeof(Real)<<std::endl;
  if(!(((unsigned long)tmp & 31) == 0)){
    std::cout<<"Not aligned"<<std::endl;
  }
  skips=0;
  for(int i = 0; i < m; i++) tmp[i] = 0.;
  
  lars_ptr = new Lars(Dt, m, k, lambda_in, timer);

  timer.end(DICT_INIT);

}
  
// for each j and T, (m*2k+m+2m+(m+m-1+1)+m)
// Total data movement:
// m*k+k+k+m+m+m+1+m+m+m+m+m
void DictionaryLearning::update_dict() {
  timer.start(DICT_UPDATE);
  print("update_dict()\n");
  bool converge = false;
  
  for (int t = 0; t < T; t++) {
    converge = true;

    for (int j = 0; j < k; j++) {
      if (At[j*k + j] < epsilon) continue;//{ skips++; continue;  }
      //noSkips ++;
      // Scalar version of code
      //for(int i = 0; i < m; i++) tmp[i] = 0.;
      //mvm(Dt, true, At+j*k, tmp, k, m); //tmp = Daj
      //vecDiff(Bt+j*m, tmp, tmp, m);
      //print("A%d%d = %.3f\n", j, j, At[j*k + j]);
      //for (int k = 0; k < lars_ptr->active_itr; k++)
      //  print("(%d, %.3f) ", lars_ptr->beta[k].id, lars_ptr->beta[k].v);
      //axpy(1.0/At[j*k + j], tmp, Dt+j*m, m);
      //Real norm = l2Norm(Dt+j*m, m);
      //print("norm(dj) = %.3f\n", norm);
      //dot(1.0/norm, Dt+j*m, m);
      

      for(int i = 0; i < m; i++) tmp[i] = 0.;
      double one = 1.0;
      double zero = 0.0;
      int iInt = 1;      
      
      //timer.start(DICT_MVM);
      //cblas_dgemv(CblasColMajor,CblasNoTrans,m, k, one, Dt, m, &At[j*k], iInt, zero, tmp, iInt);
      //mvm(Dt, true, At+j*k, tmp, k, m); //tmp = Daj
      
/*      
      for (int x = 0; x < m; x++) {
        tmp[x] = 0.0;
        //double a = 0;
        for (int y = 0; y < k; y++) {
          tmp[x] += Dt[y * m + x] * At[j*k+y];
        }
        //tmp[x] = a;
      }
*/      

      
      Real norm = 0;
      
      Real recipA = 1.0/At[j*k + j];
      for (int x = 0; x < m/4; x++){
        double a1 = 0.0;
        double a2 = 0.0;
        double a3 = 0.0;
        double a4 = 0.0;

        int bS=4;
        for (int y = 0; y < k/bS; y++) {
          a1 += Dt[(bS*y+0) * m + 4*x] * At[j*k+(bS*y+0)];
          a2 += Dt[(bS*y+0) * m + 4*x+1] * At[j*k+(bS*y+0)];
          a3 += Dt[(bS*y+0) * m + 4*x+2] * At[j*k+(bS*y+0)];
          a4 += Dt[(bS*y+0) * m + 4*x+3] * At[j*k+(bS*y+0)];

          a1 += Dt[(bS*y+1) * m + 4*x] * At[j*k+(bS*y+1)];
          a2 += Dt[(bS*y+1) * m + 4*x+1] * At[j*k+(bS*y+1)];
          a3 += Dt[(bS*y+1) * m + 4*x+2] * At[j*k+(bS*y+1)];
          a4 += Dt[(bS*y+1) * m + 4*x+3] * At[j*k+(bS*y+1)];

          a1 += Dt[(bS*y+2) * m + 4*x] * At[j*k+(bS*y+2)];
          a2 += Dt[(bS*y+2) * m + 4*x+1] * At[j*k+(bS*y+2)];
          a3 += Dt[(bS*y+2) * m + 4*x+2] * At[j*k+(bS*y+2)];
          a4 += Dt[(bS*y+2) * m + 4*x+3] * At[j*k+(bS*y+2)];

          a1 += Dt[(bS*y+3) * m + 4*x] * At[j*k+(bS*y+3)];
          a2 += Dt[(bS*y+3) * m + 4*x+1] * At[j*k+(bS*y+3)];
          a3 += Dt[(bS*y+3) * m + 4*x+2] * At[j*k+(bS*y+3)];
          a4 += Dt[(bS*y+3) * m + 4*x+3] * At[j*k+(bS*y+3)];

        }
        //Real a2;
        //Real a2 = Dt[j*m+x];
        //a2 = recipA*(Bt[j*m+x]-a)+a2;
        //Dt[j*m+x] = a2;
        //norm += a2*a2;
        //Dt[j*m+2*x] = a1;
        //Dt[j*m+2*x+1] = a2;
        /*tmp[4*x]=Bt[j*m+4*x]-a1;
        tmp[4*x+1]=Bt[j*m+4*x+1]-a2;
        tmp[4*x+2]=Bt[j*m+4*x+2]-a3;
        tmp[4*x+3]=Bt[j*m+4*x+3]-a4;*/
        Dt[j*m+4*x] += recipA*(Bt[j*m+4*x]-a1);
        norm +=Dt[j*m+4*x]*Dt[j*m+4*x];
        Dt[j*m+4*x+1] += recipA*(Bt[j*m+4*x+1]-a2);
        norm +=Dt[j*m+4*x+1]*Dt[j*m+4*x+1];
        Dt[j*m+4*x+2] += recipA*(Bt[j*m+4*x+2]-a3);
        norm +=Dt[j*m+4*x+2]*Dt[j*m+4*x+2];
        Dt[j*m+4*x+3] += recipA*(Bt[j*m+4*x+3]-a4);
        norm +=Dt[j*m+4*x+3]*Dt[j*m+4*x+3];
        //tmp[2*x+2]=a3;
        //tmp[2*x+3]=a4;
      }
      
      norm = sqrt(norm);
      

      //timer.end(DICT_MVM);
      
      //vecDiff(Bt+j*m, tmp, tmp, m);
      /*
      int endM = m-(m%4);
      for(int id = 0; id < endM; id+=4){
        __m256d tmpVec = _mm256_load_pd(&tmp[id]);
        __m256d BtVec = _mm256_load_pd(&Bt[j*m+id]);
        __m256d diff = _mm256_sub_pd(BtVec, tmpVec);
        _mm256_store_pd(&tmp[id], diff);
      }
      for(int id = endM; id < m; id++){
        tmp[id] = Bt[j*m+id]-tmp[id];
      }
      */
      
      //timer.start(AXPY);
      
      //Real recipA = 1.0/At[j*k + j];
      //cblas_daxpy(m, recipA, tmp, 1, &Dt[j*m], 1);
      //axpy(recipA, tmp, Dt+j*m, m);
      
      //timer.end(AXPY);


      //timer.start(DICT_L2);
      //Real norm = cblas_dnrm2(m, Dt+j*m, 1);
      //Real norm = l2Norm(Dt+j*m, m);
      //timer.end(DICT_L2);

      //dot(1.0/norm, Dt+j*m, m);
      //timer.start(DICT_DOT);
      int endM = m-(m%4);     
      Real recip = 1.0/norm;
      __m256d coef = _mm256_set1_pd(recip);
      for(int id = 0; id < endM; id+=4){
        __m256d xVec = _mm256_load_pd(&Dt[j*m+id]);
        __m256d prod = _mm256_mul_pd(xVec, coef);
        _mm256_store_pd(&Dt[j*m+id], prod);
      }
      for(int id = endM; id < m; id++){
        Dt[j*m+id] = recip*Dt[j*m+id];
      }
      //timer.end(DICT_DOT);
      
    }
  }
  timer.end(DICT_UPDATE);  
}


//flop counts: 
void DictionaryLearning::sparse_coding(Real *const x, Real *const alpha) {
  sparse_coding(x);
  // cope the Lars.beta into a dense vector alpha
  timer.start(SPARSE_CODING_ALPHA);
  Idx *beta = lars_ptr->beta;
  int l = lars_ptr->active_itr;
  memset(alpha, 0, k * sizeof(Real));
  for (int i = 0; i < l; i++)
    alpha[beta[i].id] = beta[i].v;
  timer.end(SPARSE_CODING_ALPHA);
}

//flop counts: l*2m
void DictionaryLearning::recover(Real *const x, Real*const x_r) {
    sparse_coding(x);
    timer.start(RECOVER);
    Idx *alpha = lars_ptr->beta;
    int l = lars_ptr->active_itr;
    memset(x_r, 0, m * sizeof(Real));
    for (int i = 0; i < l; i++)
      axpy(alpha[i].v, Dt + alpha[i].id * m, x_r, m);
    timer.end(RECOVER);
}

//flop counts: 
void DictionaryLearning::sparse_coding(Real *const x) {
  print("sparse_coding()\n");
  timer.start(SPARSE_CODING);
  lars_ptr->set_y(x); //reset temporary data in Lars, and set Lars.y to x
  lars_ptr->solve();
  timer.end(SPARSE_CODING);
}

//flop counts: l*(2l+2m)
void DictionaryLearning::iterate(Real *const x) { // run line 4-7 of algorithm 1
  sparse_coding(x);
  timer.start(DICT_ITERATE);

  Idx *alpha = lars_ptr->beta;
  int l = lars_ptr->active_itr;
  // A += alpha*alpha.T, B += x*alpha.T
  for (int i = 0; i < l; i++) {

    for (int j = 0; j<l;j++){
      At[alpha[j].id * k + alpha[i].id] += alpha[i].v * alpha[j].v;
    }


    for (int j = 0; j < m; j++){
      Bt[alpha[i].id * m + j] += x[j] * alpha[i].v;
    }

  }
  timer.end(DICT_ITERATE);
  update_dict();
}

#endif