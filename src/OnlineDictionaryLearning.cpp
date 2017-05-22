#include "OnlineDictionaryLearning.h"
#include "mathOperations.h"
#include <cstdlib>
#include "immintrin.h"
#include <iostream>

#ifndef DICTIONARY_LEARNING_CPP
#define DICTIONARY_LEARNING_CPP

#define is_aligned(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)

DictionaryLearning::DictionaryLearning(Real lambda_in, int m_in, int k_in, Timer &timer) :
m(m_in), k(k_in), epsilon(1e-2), T(5), timer(timer) {
  timer.start(DICT_INIT);
  /*
  Dt = (Real*) calloc(k * m, sizeof(Real));
  prepare_Xt(m, k, true, Dt); // Initailize Dt with random normalized cols
  At = (Real*) malloc(k * k * sizeof(Real));
  Bt = (Real*) malloc(k * m * sizeof(Real));
  tmp = (Real*) calloc(m, sizeof(Real));
  lars_ptr = new Lars(Dt, m, k, lambda_in, timer);
  */

  Dt = (Real*) aligned_alloc(k * m, k * m * sizeof(Real));
  prepare_Xt(m, k, true, Dt); // Initailize Dt with random normalized cols
  At = (Real*) aligned_alloc(k * k, k * k * sizeof(Real));
  Bt = (Real*) aligned_alloc(k * m, k * m * sizeof(Real));
  tmp = (Real*) aligned_alloc(m, m*sizeof(Real));
  memset(tmp, 0, m*sizeof(Real));
  lars_ptr = new Lars(Dt, m, k, lambda_in, timer);
  timer.end(DICT_INIT);

}


//for each j and T, (m*2k+m+2m+(m+m-1+1)+m)
void DictionaryLearning::update_dict() {
  timer.start(DICT_UPDATE);
  print("update_dict()\n");
  bool converge = false;
/*
  if(!is_aligned(Dt,32)){
    std::cout<<"Dt Unaligned"<<std::endl;
  }
  if(!is_aligned(At,32)){
    std::cout<<"At Unaligned"<<std::endl;
  }
  if(!is_aligned(Bt,32)){
    std::cout<<"Bt Unaligned"<<std::endl;
  }
*/


  for (int t = 0; t < T; t++) {
    converge = true;
    for (int j = 0; j < k; j++) {
      if (At[j*k + j] < epsilon) continue; // when At[j*k + j] is zero
      
      memset(tmp, 0, m*sizeof(Real));
      
      
      // Scalar version of code
      /*
      mvm(Dt, true, At+j*k, tmp, k, m); //tmp = Daj      
      vecDiff(Bt+j*m, tmp, tmp, m);


      print("A%d%d = %.3f\n", j, j, At[j*k + j]);
      for (int l = 0; l < lars_ptr->active_itr; l++)
        print("(%d, %.3f) ", lars_ptr->beta[l].id, lars_ptr->beta[l].v);
      

      axpy(1.0/At[j*k + j], tmp, Dt+j*m, m);

      Real norm = l2Norm(Dt+j*m, m);

      print("norm(dj) = %.3f\n", norm);
      dot(1.0/norm, Dt+j*m, m);
      */


      
      //mvm(Dt, true, At+j*k, tmp, k, m); //tmp = Daj            

      for (int id1 = 0; id1 < k; id1++){
        double Ai = At[j*k+id1];
        __m256d vec = _mm256_set1_pd(Ai);
        int endId = m - (m%4);
        /*
  if(!is_aligned(Dt,32)){
    std::cout<<"Dt Unaligned"<<std::endl;
  }
  if(!is_aligned(At,32)){
    std::cout<<"At Unaligned"<<std::endl;
  }
  if(!is_aligned(Bt,32)){
    std::cout<<"Bt Unaligned"<<std::endl;
  }
  if(!is_aligned(tmp,32)){
    std::cout<<"tmp Unaligned"<<std::endl;
  }
  //std::cout<<"asdasd"<<endId<<" "<<m<<std::endl;
  */
        for(int id2 = 0; id2 < endId; id2+=4){
          //
            //if(!is_aligned(&Dt[id1*m+id2],32)){
             // std::cout<<"tmp Unaligned"<<id2<<std::endl;
            //}
            
          __m256d DtRow = _mm256_loadu_pd(&Dt[id1*m+id2]);
          __m256d tmpVec = _mm256_load_pd(&tmp[id2]);
          __m256d res = _mm256_fmadd_pd(vec, DtRow, tmpVec);
          _mm256_store_pd(tmp+id2, res);
          //std::cout<<id2<<std::endl;
        }

        for(int id2 = endId; id2 < m; id2++){
          tmp[id2] += Dt[id1*m+id2]*Ai;
        }

      }
      



      //vecDiff(Bt+j*m, tmp, tmp, m);
      
      int endM = m-(m%4);
      for(int id = 0; id < endM; id+=4){
        __m256d tmpVec = _mm256_loadu_pd(&tmp[id]);
        __m256d BtVec = _mm256_loadu_pd(&Bt[j*m+id]);
        __m256d diff = _mm256_sub_pd(BtVec, tmpVec);
        _mm256_storeu_pd(&tmp[id], diff);
      }
      for(int id = endM; id < m; id++){
        tmp[id] = Bt[j*m+id]-tmp[id];
      }



      print("A%d%d = %.3f\n", j, j, At[j*k + j]);
      for (int l = 0; l < lars_ptr->active_itr; l++)
        print("(%d, %.3f) ", lars_ptr->beta[l].id, lars_ptr->beta[l].v);
      

//      axpy(1.0/At[j*k + j], tmp, Dt+j*m, m);
      
      endM = m-(m%4);
      double recip = 1.0/At[j*k + j];
      __m256d coef = _mm256_set1_pd(recip);
      for(int id = 0; id < endM; id+=4){
        __m256d tmpVec = _mm256_loadu_pd(tmp + id);
        __m256d DtVec = _mm256_loadu_pd(&Dt[j*m+id]);
        __m256d sum = _mm256_fmadd_pd(coef, tmpVec, DtVec);
        _mm256_storeu_pd(&Dt[j*m+id], sum);
      }
      for(int id = endM; id < m; id++){
        Dt[j*m+id] = recip*tmp[id]+Dt[j*m+id];
      }
      


      //Real norm = l2Norm(Dt+j*m, m);
 
      double norm = 0.;
      endM = m-(m%4);
      __m256d accum = _mm256_set1_pd(0.);
      for(int id = 0; id < endM; id+=4){
        __m256d xVec = _mm256_load_pd(&Dt[j*m+id]);
        __m256d accum = _mm256_fmadd_pd(xVec, xVec, accum);
      }
      __m256d hSum = _mm256_hadd_pd(accum,accum);
      norm = ((double*)&hSum)[0] + ((double*)&hSum)[2];

      for(int id = endM; id < m; id++){
        norm += Dt[j*m+id]*Dt[j*m+id];
      }
      norm = sqrt(norm);



      print("norm(dj) = %.3f\n", norm);

 //     dot(1.0/norm, Dt+j*m, m);
 
      //int 
      endM = m-(m%4);     
      recip = 1.0/norm;
      coef = _mm256_set1_pd(recip);
      //double * ptr = &Dt[j*m];
      for(int id = 0; id < endM; id+=4){
        __m256d xVec = _mm256_loadu_pd(&Dt[j*m+id]); // gives seg fault
        __m256d prod = _mm256_mul_pd(xVec, coef);
        _mm256_storeu_pd(&Dt[j*m+id], prod);
      }
      for(int id = endM; id < m; id++){
        Dt[j*m+id] = recip*Dt[j*m+id];
      }
   
      

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
  Idx *alpha = lars_ptr->beta;
  int l = lars_ptr->active_itr;
  // A += alpha*alpha.T, B += x*alpha.T
  timer.start(DICT_ITERATE);
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