import numpy as np
from normalize import normalize

def viterbi_path(prior,transmat,obslik):

     # VITERBI Find the most-probable (Viterbi) path through the HMM state trellis.
     # path = viterbi(prior, transmat, obslik)
     #
     # Inputs:
     # prior(i) = Pr(Q(1) = i)
     # transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
     # obslik(i,t) = Pr(y(t) | Q(t)=i)
     #
     # Outputs:
     # path(t) = q(t), where q1 ... qT is the argmax of the above expression.


     # delta(j,t) = prob. of the best sequence of length t-1 and then going to state j, and O(1:t)
     # psi(j,t) = the best predecessor state, given that we ended up in state j at t

     scaled = 1
     T = len(obslik[0])
     Q = len(prior)

     delta = np.zeros((Q,T))
     psi = np.zeros((Q,T))
     path = np.zeros(T)
     scale = np.ones(T)

     t = 0
     delta[:,t] = prior*obslik[:,t]
     if bool(scaled):
          delta[:,t],n = normalize(delta[:,t])
          scale[t] = 1/n

     psi[:,t] = 0 #arbitrary value, no predecessor before t = 0
     for t in range(1,T):
          for j in range(0,Q):
              delta[j,t] = np.max(delta[:,t-1]*transmat[:,j])
              psi[j,t] = np.where(delta[:,t-1]*transmat[:,j] == delta[j,t])[0][0]
              delta[j,t] = delta[j,t]*obslik[j,t]

          if bool(scaled):
              delta[:,t],n = normalize(delta[:,t])
              scale[t] = 1/n

     p = np.max(delta[:,T-1])
     path[T-1] =  np.where(delta[:,T-1] == p)[0][0]  

     for t in range(T-2,-1,-1):
          #print("t = ", t)
          path[int(t)] = psi[int(path[int(t)+1]),int(int(t)+1)]


     # If scaled==0, p = prob_path(best_path)
     # If scaled==1, p = Pr(replace sum with max and proceed as in the scaled forwards algo)
     # Both are different from p(data) as computed using the sum-product (forwards) algorithm 

     return path

def main():

     prior1 = np.loadtxt("testfiles/prior1mmhmm.csv",delimiter = ',')
     transmat1 = np.loadtxt("testfiles/transmat1mmhmm.csv",delimiter = ',')
     B = np.loadtxt("testfiles/B_mmhm_mgp_outer.csv",delimiter = ',')

     path = viterbi_path(prior1, transmat1, B)

     print("path = ",path)

if __name__ == "__main__":
     main()

