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
void duplicate1D(double **,double *,unsigned int,unsigned int);
void duplicate(double **, double **,unsigned int,unsigned int, unsigned int);
void den(double *,double **,unsigned int, unsigned int);
void multiplyInPlaceEta(double ***,double ***,double **,double **,unsigned int,unsigned int, unsigned int);
void divideInPlaceEta(double  **,double ** ,double ** ,unsigned int, unsigned int);
void multiplyInPlaceSquare(double **,double ** ,double ** ,unsigned int);
double normTotal(double **,unsigned int, unsigned int);


double mexFunction(double **out_alpha, double ** out_beta, double ** out_gamma, double ** out_eta, double ** out_xi_summed,double * initDist, double ** transmat, double ** softev, double ** preB2, double ** mixmat,int Q, int T, int M)
{

      //converting 2d input to correct 3d form
      double *** obslik2;  //form QxMxT
      obslik2 = malloc(Q*sizeof *obslik2);
      for (int i=0; i< Q;i++){
         obslik2[i] = malloc(M*sizeof*obslik2[i]);
         for (int j=0; j <M;j++){
              obslik2[i][j] = malloc(T*sizeof*obslik2[i][j]);
         }
      }
      for (int i=0; i< Q;i++){
          for (int j=0;j<M*T;j++){
              obslik2[i][j/T][j%T] = preB2[i][j];
          }
     }



      double ** transmatT;  //transmatT size = QxQ
      transmatT = malloc(Q*sizeof *transmatT);
      for (int i = 0;i<Q;i++){
            transmatT[i] = malloc(Q*sizeof*transmatT[i]); 
      }

      double ** xi_summed;  //xi_summed size = QxQ
      xi_summed = malloc(Q*sizeof *xi_summed);
      for (int i = 0;i<Q;i++){
            xi_summed[i] = malloc(Q*sizeof*xi_summed[i]);
      }
      for (int i = 0;i<Q;i++){
          for (int j = 0;j<Q;j++){
                    xi_summed[i][j] = 0;
           }
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
      }

      double loglik = 0;
      for (t=0;t<T;++t){
           loglik = loglik + log(scale[t]);
      }

      /*************** Backward *****************/

      double * b;
      b = malloc(Q*sizeof *b);
      
      double *** eta;
      eta = malloc(Q*sizeof *eta); //eta size QxMxT
      for (int i = 0; i< Q;i++){
           eta[i] = malloc(M*sizeof*eta[i]);
           for (int j = 0; j < M;j++){
                eta[i][j] = malloc(T*sizeof*eta[i][j]);
           }
       }
      for (int i = 0;i<Q;i++){
          for (int j = 0;j<M;j++){
               for (int k = 0;k<T;k++){
                    eta[i][j][k] = 0;
               }
           }
       }

      double * denom;
      denom = malloc(Q*sizeof *denom);
      
      double ** squareSpace; // squareSpace size QxQ
      squareSpace = malloc(Q*sizeof *squareSpace);
      for (int i = 0; i< Q;i++){
           squareSpace[i] = malloc(Q*sizeof*squareSpace[i]);
      }
      double ** squareSpace2; // squareSpace2 size QxQ
      squareSpace2 = malloc(Q*sizeof *squareSpace2);
      for (int i = 0; i< Q;i++){
           squareSpace2[i] = malloc(Q*sizeof*squareSpace2[i]);
      }


      double ** repmat;  // repmat size QxM
      repmat = malloc(Q*sizeof *repmat);
      for (int i = 0;i<Q;i++){
           repmat[i] = malloc(M*sizeof*repmat[i]);
      }

      double ** repmat2;  // repmat2 size QxM
      repmat2 = malloc(Q*sizeof *repmat2);
      for (int i = 0;i<Q;i++){
           repmat2[i] = malloc(M*sizeof*repmat2[i]);
      }
      double ** repmat3;  // repmat3 size QxM
      repmat3 = malloc(Q*sizeof *repmat3);
      for (int i = 0;i<Q;i++){
           repmat3[i] = malloc(M*sizeof*repmat3[i]);
      }

      t = T - 1;
      /* I don't think we need to initialize beta to all zeros. */

      for (int d = 0;d<Q;++d){
          beta[d][t] = 1;
          gamma[d][t] = alpha[d][t];
      }

      den(denom,softev,Q,t);
      duplicate(repmat,gamma,Q,M,t);
      duplicate1D(repmat2,denom,Q,M);
      divideInPlaceEta(repmat3,repmat,repmat2,Q,M);
      multiplyInPlaceEta(eta,obslik2,mixmat,repmat3,Q,M,t);
      
      

      /*Put the last slice of eta as zeros, to be compatible with Sohrab and Gavin's code.
       There are no values to put there anyways. This means that you can't normalise the
       last matrix in eta, but it shouldn't be used. Note the d<K*K range.
      */

       double summation = 0;
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
            multiplyInPlaceSquare(squareSpace2,transmat,squareSpace,Q);
            summation = normTotal(squareSpace2,Q,Q);
            for (int i=0;i<Q;++i){
                 for (int j=0;j<Q;++j){
                         xi_summed[i][j] = xi_summed[i][j] + squareSpace2[i][j];
                 }
            } 
                
           // normalizeInPlace3D(eta,Q,t);
            
            /* setting gamma */
            multiplyInPlace1D(m,alpha,beta,Q,t);
            normalizeInPlace1D(m,Q);
            for (int d=0;d<Q;++d){
                gamma[d][t] = m[d];
            }
            den(denom,softev,Q,t);
            duplicate(repmat,gamma,Q,M,t);
            duplicate1D(repmat2,denom,Q,M);
            divideInPlaceEta(repmat3,repmat,repmat2,Q,M);
            multiplyInPlaceEta(eta,obslik2,mixmat,repmat3,Q,M,t);

     }

     /* EXPORT RESULTS */
 
     for (int i = 0;i<Q;i++){
          for (int j = 0;j<T;j++){
               out_alpha[i][j] = alpha[i][j];
               out_beta[i][j] = beta[i][j];
               out_gamma[i][j] = gamma[i][j];
          }
      }

      for (int i = 0;i<Q;i++){
           for (int j = 0;j<M*T;j++){
                     out_eta[i][j]= eta[i][j/T][j%T];
            }
       }
       for (int i = 0;i<Q;i++){
            for (int j = 0;j<Q;j++){
                     out_xi_summed[i][j] = xi_summed[i][j];
            }
        }
     
      /*FREE MEMORY */

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
           for (int j=0; j<M; ++j){
               free(eta[i][j]);
               free(obslik2[i][j]);
           }
           free(eta[i]);
           free(obslik2[i]);
      }
      free(eta);
      free(obslik2);

      for (int i=0;i<Q;++i){
            free(xi_summed[i]);
            free(repmat[i]);
            free(repmat2[i]);
            free(repmat3[i]);
            free(squareSpace[i]);
            free(squareSpace2[i]);
            free(transmatT[i]);
      }

      free(squareSpace);
      free(squareSpace2);
      free(xi_summed);
      free(scale);
      free(transmatT);
      free(repmat);
      free(repmat2);
      free(repmat3);
      free(denom);

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
double normTotal(double ** A,unsigned int rows,unsigned int cols){
      unsigned int i,j;
      double sum = 0;
      
      for (i=0;i<rows;++i){
          for (j=0;j<cols;++j){
              sum += A[i][j];
              if (A[i][j] < 0){
                    printf("WARNING found a negative prob");
              }
           }
       }
      
       if (sum > 0){
          for (i=0;i<rows;++i){
               for (j=0;j<cols;++j){
                  A[i][j] /= sum;
               }
           }
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
void multiplyInPlaceSquare(double ** result,double ** u, double ** v,unsigned int K)
{
      unsigned int i,j;
      for (i=0;i<K;++i){
          for (j=0;j<K;++j){
               result[i][j] = u[i][j]*v[i][j];
          }
       }
       return;
}

void divideInPlaceEta(double  ** result, double ** u, double ** v, unsigned int K, unsigned int clusters)
{
     unsigned int i,j;
     for (i=0;i<K;++i){
          for (j=0;j<clusters;++j){
                 result[i][j] = u[i][j]/v[i][j];
          }
      }
      return;
}
void multiplyInPlaceEta(double *** result, double ***u, double ** v, double **w, unsigned int K, unsigned int clusters, unsigned int time)
{
      unsigned int i,j;
      for (i=0;i<K;++i){
          for (j=0;j<clusters;++j){
                 result[i][j][time] = u[i][j][time]*v[i][j]*w[i][j];
          }
      }
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

void den(double * out, double ** u, unsigned K,unsigned int time)
{
     //uses B[:,time] as the denominator
     unsigned int i;
     for (i=0;i<K;++i){
         out[i] = u[i][time];
         if (out[i] == 0){  //output vector is denominator to eqn, must > 0
              out[i] = 1;
         }
     }
     return;
}
void duplicate(double ** out, double ** u, unsigned int K, unsigned int clusters,unsigned int time){
      //duplicates gamma[:,time] into QxM matrix
      unsigned int i,j;
      for (i=0;i<K;++i){
           for (j=0;j<clusters;++j){
                 out[i][j] = u[i][time];
           }
       }
       return;
} 
void duplicate1D(double ** out, double * u, unsigned int K, unsigned int clusters){
      unsigned int i,j;
      //duplicates denom[:] into QxM matrix
      for (i=0;i<K;++i){
           for (j=0;j<clusters;++j){
                 out[i][j] = u[i];
           }
       }
       return;
}

