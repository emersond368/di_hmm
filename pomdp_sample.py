import numpy as np
from sample_mdp import sample_mdp
from sample_discrete import sample_discrete

def pomdp_sample(initial_prob, transmat, obsmat, act):

     #need to convert transmat to list of arrays

     # SAMPLE_POMDP Generate a random sequence from a Partially Observed Markov Decision Process.
     # [obs, hidden] = sample_pomdp(prior, transmat, obsmat, act)
     #
     # Inputs:
     # prior[i] = Pr(Q[1]=i)
     # transmat[a][i,j] = Pr(Q[t]=j | Q[t-1]=i, A[t]=a) transmat[act][statei][statej],  size AxNxN
     # obsmat[i,k] = Pr(Y(t)=k | Q(t)=i),    size NxK
     # act[a] = A[t], so act[1] is ignored, size A
     #
     # Output:
     # obs and hidden are vectors of length T=len(act)


     length = len(act)
     hidden = sample_mdp(initial_prob, transmat, act)
     obs = np.zeros(length)
     for t in range(0,length):
          obs[t] = sample_discrete(obsmat[hidden[t],:],1,1)

     return (obs,hidden)

def main():

     prior = np.array([1,0])
     a = np.ones(25) - 1
     b = np.ones(25)*2 - 1

     act = np.concatenate([a,b,a,b])

     trans = np.ones((2,2,2))

     trans[0][0] = [0.9,0.1]  #act 1
     trans[0][1] = [0.1,0.9]
     trans[1][0] = [0.1,0.9]  #act 2
     trans[1][1] = [0.9,0.1]
     
     obsmat0 = np.eye(2, dtype = int)

     data1 = pomdp_sample(prior, trans, obsmat0, act)

     print "data1[0] = ",data1[0]
     print "data1[1] = ",data1[1]

if __name__ == "__main__":
     main()

   
