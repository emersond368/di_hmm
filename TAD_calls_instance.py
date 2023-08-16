from mhmm_em import mhmm_em
from mhmm_logprob import mhmm_logprob
from mixgauss_prob import mixgauss_prob
from viterbi_path import viterbi_path
from hmmFwdBack import hmmFwdBack
from jie_kmeans_init import jie_kmeans_init
import numpy as np

def TAD_calls(testX,prior0,transmat0):

     M = 2
     Q = 3
     mu0, Sigma0,mx = jie_kmeans_init(testX,Q,M)
     Sigma0 = np.reshape(Sigma0,(1,1,Q,M), order = 'F')
     mu0 = np.reshape(mu0,(1,Q,M), order = 'F')
     mixmat0 = (1/float(M))*np.ones((Q,M))

     LL,prior1,transmat1,mu1,Sigma1,mixmat1 = mhmm_em(testX,prior0,transmat0,mu0,Sigma0,mixmat0,max_iter=500,thresh=1e-5)
     loglik = mhmm_logprob(testX, prior1, transmat1, mu1, Sigma1, mixmat1)
     B,B2 = mixgauss_prob(testX,mu1,Sigma1,mixmat1)
     path = viterbi_path(prior1, transmat1, B)
     gamma,alpha,beta,logp = hmmFwdBack(prior1,transmat1,B)

     decodedFromEMmaxMarg = np.argmax(gamma,axis=0)

     if bool(M==1):
          num_p = 1*len(prior1) + (len(transmat1)*len(transmat1[0])) + (len(mu1)*1) + (len(Sigma1)*len(Sigma1[0]))+ (len(mixmat1)*1)
     else:
          num_p = 1*len(prior1) + (len(transmat1)*len(transmat1[0])) +(len(mu1)*len(mu1[0])*len(mu1[0][0])) + (len(Sigma1)*len(Sigma1[0])*len(Sigma1[0][0])*len(Sigma1[0][0][0]))+ (len(mixmat1)*len(mixmat1[0]))  

     aic = np.zeros(1)
     ll = np.zeros(1)
     best_p = np.zeros(len(testX))
     best_d = np.zeros(len(testX))
     best_g1 = np.zeros(len(testX))
     best_g2 = np.zeros(len(testX))
     best_g3 = np.zeros(len(testX))

     aic[0] = -2*loglik + 2*num_p
     ll[0] = loglik
     best_p = path
     best_d = decodedFromEMmaxMarg
     best_g1 = gamma[0,:]
     best_g2 = gamma[1,:]
     best_g3 = gamma[2,:]

     return aic,num_p,ll,best_p,best_d,best_g1,best_g2,best_g3

def main():
     
 #    M= 1
 #    testX = np.loadtxt("/Users/nintendo/programming/Python/Cremins/domaincall_software/testXmhmm.csv",delimiter = ',')
 #    prior0 = np.loadtxt("/Users/nintendo/programming/Python/Cremins/domaincall_software/prior0mhmm.csv",delimiter = ',')
 #    transmat0 = np.loadtxt("/Users/nintendo/programming/Python/Cremins/domaincall_software/transmat0mhmm.csv",delimiter = ',')
 #    mu0 = np.loadtxt("/Users/nintendo/programming/Python/Cremins/domaincall_software/mu0mhmm.csv",delimiter = ',')
 #    Sigma0 = np.loadtxt("/Users/nintendo/programming/Python/Cremins/domaincall_software/Sigma0mhmm.csv",delimiter = ',')
 #    mixmat0 = np.loadtxt("/Users/nintendo/programming/Python/Cremins/domaincall_software/mixmat0mhmm.csv",delimiter = ',')
 #    Sigma0 = np.reshape(Sigma0,(1,1,3,1), order = 'F')
 #    mixmat0 = np.reshape(mixmat0,(3,1),order = 'F')
 #    mu0 = np.reshape(mu0,(1,3,1), order = 'F')
     
 #    aic,num_p,ll,best_p,best_d,best_g1,best_g2,best_g3 = TAD_calls(testX,prior0,transmat0,mu0,Sigma0,mixmat0)

 #    print "aic = ",aic
 #    print "num_p = ",num_p
 #    print "ll = ",ll
 #    print "best_p = ",best_p
 #    print "best_d = ",best_d
 #    print "best_g1 = ",best_g1
 #    print "best_g2 = ",best_g2
 #    print "best_g3 = ",best_g3

     testX = np.loadtxt("testfiles/testXmmhmm.csv",delimiter = ',')
     prior0 = np.loadtxt("testfiles/prior0mmhmm.csv",delimiter = ',')
     transmat0 = np.loadtxt("testfiles/transmat0mmhmm.csv",delimiter = ',')
     mu0 = np.loadtxt("testfiles/mu0mmhmm.csv",delimiter = ',')
     Sigma0 = np.loadtxt("testfiles/Sigma0mmhmm.csv",delimiter = ',')
     mixmat0 = np.loadtxt("testfiles/mixmat0mmhmm.csv",delimiter = ',')
     Sigma0 = np.reshape(Sigma0,(1,1,3,2), order = 'F')
     mu0 = np.reshape(mu0,(1,3,2), order = 'F')

 

     aic,num_p,ll,best_p,best_d,best_g1,best_g2,best_g3 = TAD_calls(testX,prior0,transmat0)
     
     print "aic = ",aic
     print "num_p = ",num_p
     print "ll = ",ll
     print "best_p = ",best_p
     print "best_d = ",best_d
     print "best_g1 = ",best_g1
     print "best_g2 = ",best_g2
     print "best_g3 = ",best_g3

if __name__ == "__main__":
     main()


     
