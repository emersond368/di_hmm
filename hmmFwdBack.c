#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

double normalizeInPlace(double **, unsigned int,unsigned int);
double normalizeInPlace1D(double *,unsigned int);
double normalizeInPlace3D(double ***,unsigned int,unsigned int);
void multiplyInPlace(double **, double *, double **, unsigned int,unsigned int);
void multiplyInPlace1D(double *, double **, double **, unsigned int,unsigned int);
void multiplyMatrixInPlace(double *, double **, double **, unsigned int,unsigned int);
void multiplyMatrixInPlace1D(double *, double **, double *, unsigned int);
void transposeSquareInPlace(double **, double **, unsigned int);
void outerProductUVInPlace(double **, double **, double *, unsigned int,unsigned int);
void componentVectorMultiplyInPlace(double ***, double **, double **, unsigned int,unsigned int);


double mexFunction(double **out_alpha, double ** out_beta, double ** out_gamma, double * initDist, double ** transmat, double ** softev, int Q, int T, int M)
{
      //double xi_summed = 0;

      double ** transmatT;  //transmatT size = QxQ
      transmatT = malloc(Q*sizeof *transmatT);
      for (int i = 0;i<Q;i++){
            transmatT[i] = malloc(Q*sizeof*transmatT[i]); 
      }

      transposeSquareInPlace(transmatT, transmat, Q);
      
      double * scale;
      scale = malloc(T*sizeof *scale);
     
      double ** alpha; //alpha size = QxT
      alpha = malloc(Q*sizeof*alpha);
      for (int i = 0;i<Q;i++){
            alpha[i] = malloc(T*sizeof*alpha[i]);
      }

      double ** beta; //beta size = QxT
      beta = malloc(Q*sizeof*beta);
      for (int i = 0;i<Q;i++){
            beta[i] = malloc(T*sizeof*beta[i]);
      }

      double ** gamma; //gamma size = QxT
      gamma = malloc(Q*sizeof*gamma);
      for (int i = 0;i<Q;i++){
            gamma[i] = malloc(T*sizeof*gamma[i]);
      }

      /************** Forward ******************/

      int t = 0;
      multiplyInPlace(alpha,initDist,softev,Q,t);
      scale[t] = normalizeInPlace(alpha, Q,t);

      double * m;
      m = malloc(Q*sizeof *m); 
      for (t=1;t<T;++t){
           multiplyMatrixInPlace(m,transmatT,alpha,Q,t-1);      
           multiplyInPlace(alpha,m,softev,Q,t);
           scale[t] = normalizeInPlace(alpha, Q,t);
           //xi_summed = xi_summed + alpha
      }

      double loglik = 0;
      for (t=0;t<T;++t){
           loglik = loglik + log(scale[t]);
      }

      /*************** Backward *****************/

      t = T - 1;
      /* I don't think we need to initialize beta to all zeros. */
      for (int d = 0;d<Q;++d){
          beta[d][t] = 1;
          gamma[d][t] = alpha[d][t];
      }

      double * b;
      b = malloc(Q*sizeof *b);
      
      double *** eta;
      eta = malloc(Q*sizeof *eta); //eta size QxQxT
      for (int i = 0; i< Q;i++){
           eta[i] = malloc(Q*sizeof*eta[i]);
           for (int j = 0; j < Q;j++){
                eta[i][j] = malloc(T*sizeof*eta[i][j]);
           }
       }
      
      double ** squareSpace; // squareSpace size QxQ
      squareSpace = malloc(Q*sizeof *squareSpace);
      for (int i = 0; i< Q;i++){
           squareSpace[i] = malloc(Q*sizeof*squareSpace[i]);
      }

      /*Put the last slice of eta as zeros, to be compatible with Sohrab and Gavin's code.
       There are no values to put there anyways. This means that you can't normalise the
       last matrix in eta, but it shouldn't be used. Note the d<K*K range.
      */

      for (int i = 0;i<Q;i++){
          for (int j = 0;j<Q;j++){
               for (int k = 0;k<Q;k++){
                    eta[i][j][k] = 0;
               }
           }
       }
       
       for (t=(T-2);t>=0;--t){
            /* setting beta */
            multiplyInPlace1D(b,beta,softev,Q,t+1);
            /* Using "m" again instead of defining a new temporary variable.
	       We using a lot of lines to say
	           beta(:,t) = normalize(transmat * b);
	    */
            multiplyMatrixInPlace1D(m, transmat, b, Q);   
            normalizeInPlace1D(m,Q);
	    for (int d=0;d<Q;++d){
                beta[d][t] = m[d];
            }
            /* using "m" again as valueholder */
            /* setting eta, whether we want it or not in the output */
            outerProductUVInPlace(squareSpace,alpha,b,Q,t);
            componentVectorMultiplyInPlace(eta, transmat, squareSpace, Q,t);
            normalizeInPlace3D(eta,Q,t);
            
            /* setting gamma */
            multiplyInPlace1D(m,alpha,beta,Q,t);
            normalizeInPlace1D(m,Q);
            for (int d=0;d<Q;++d){
                gamma[d][t] = m[d];
            }

     }

     /* EXPORT RESULTS */
 
     for (int i = 0;i<Q;i++){
          for (int j = 0;j<T;j++){
               out_alpha[i][j] = alpha[i][j];
               out_beta[i][j] = beta[i][j];
               out_gamma[i][j] = gamma[i][j];
          }
      }

      //out_loglik = &loglik;

      //printf("loglik = %f\n",loglik);

      for (int i=0; i<Q; ++i){
          free(alpha[i]);
          free(beta[i]);
          free(gamma[i]);
      }
     
      free(alpha);
      free(beta);
      free(gamma);
      free(b);
      free(m);

      for (int i=0; i<Q; ++i){
           for (int j=0; j<Q; ++j){
               free(eta[i][j]);
           }
           free(eta[i]);
      }

      free(eta);
      free(scale);
      
      for (int i=0;i<Q;++i){
          free(transmatT[i]);
          free(squareSpace[i]);
      }
      free(transmatT);
      free(squareSpace);
 
      return loglik; 
 
}



/* And returns the normalization constant used.
   I'm assuming that all I'll want to do is to normalize columns
   so I don't need to include a stride variable.
*/


double normalizeInPlace(double ** A, unsigned int N,unsigned int time) {
    unsigned int n;
    double sum = 0;

    for(n=0;n<N;++n) {
	sum += A[n][time];
	if (A[n][time] < 0) {
	    printf("We don't want to normalize if A contains a negative value. This is a logical error.");
	}
    }

    if (sum > 0){
	for(n=0;n<N;++n)
	    A[n][time] /= sum;
    }
    return sum;
}
double normalizeInPlace1D(double * A, unsigned int N) {
    unsigned int n;
    double sum = 0;

    for(n=0;n<N;++n) {
        sum += A[n];
        if (A[n] < 0) {
            printf("We don't want to normalize if A contains a negative value. This is a logical error.");
        }
    }

    if (sum > 0){
        for(n=0;n<N;++n)
            A[n] /= sum;
    }
    return sum;
}
double normalizeInPlace3D(double *** A,unsigned int N,unsigned int time) {
     unsigned int i,j;
     double sum = 0;

     for(i=0;i<N;++i) {
        for(j=0;j<N;++j){
            sum += A[i][j][time];
            if (A[i][j][time] < 0) {
                 printf("We don't want to normalize if A contains a negative value. This is a logical error.");
            }
        }
    }

    if (sum > 0){
        for(i=0;i<N;++i){
           for(j=0;j<N;++j){
                A[i][j][time] /= sum;
           }
        } 
    }
    return sum;

}

void multiplyInPlace(double ** result, double * u, double ** v, unsigned int K,unsigned int time) {
    unsigned int n;

    for(n=0;n<K;++n)
	result[n][time] = u[n] * v[n][time];

    return;
}
void multiplyInPlace1D(double * result, double ** u, double ** v, unsigned int K,unsigned int time) {
    unsigned int n;

    for(n=0;n<K;++n)
	result[n] = u[n][time] * v[n][time];

    return;
}



void multiplyMatrixInPlace(double * result, double ** trans, double ** v, unsigned int K, unsigned int time) {

    unsigned int i,d;

    for(d=0;d<K;++d) {
	result[d] = 0;
	for (i=0;i<K;++i){
	    result[d] += trans[d][i] * v[i][time];
	}
    }
    return;
}

void multiplyMatrixInPlace1D(double * result, double ** trans, double * v, unsigned int K) {

    unsigned int i,d;

    for(d=0;d<K;++d) {
        result[d] = 0;
        for (i=0;i<K;++i){
            result[d] += trans[d][i] * v[i];
        }
    }
    return;
}

void transposeSquareInPlace(double ** out, double ** in, unsigned int K) {

    unsigned int i,j;

    for(i=0;i<K;++i){
	for(j=0;j<K;++j){
	    out[j][i] = in[i][j];
	}
    }
    return;
}


void outerProductUVInPlace(double ** Out, double ** u, double * v, unsigned int K, unsigned int time) {
    unsigned int i,j;

    for(i=0;i<K;++i){
	for(j=0;j<K;++j){
	    Out[i][j] = u[i][time] * v[j];
	}
    }
    return;
}


/* this works for matrices also if you just set the length "L" to be the right value,
   often K*K, instead of just K in the case of vectors
*/

void componentVectorMultiplyInPlace(double *** Out, double ** u, double ** v, unsigned int K,unsigned int time) {
    unsigned int i,j;

    for(i=0;i<K;++i){
        for(j=0;j<K;++j){
	     Out[i][j][time] = u[i][j] * v[i][j];
        }
    }
    return;
}
