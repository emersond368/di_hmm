import numpy as np
from sample_discrete import sample_discrete

def sample_mdp(prior, trans, act):

     # SAMPLE_MDP Sample a sequence of states from a Markov Decision Process.
     # state = sample_mdp(prior, trans, act)
     #
     # Inputs:
     # prior[i] = Pr(Q[0]=i)
     # trans[a][i,j] = Pr(Q(t)=j | Q(t-1)=i, A(t)=a), trans[act][statex][statey]
     # act(a) = A(t), so act(1) is ignored
     #
     # Output:
     # state is a vector of length T=length(act)

     length = len(act)
     state = np.zeros(length)
     state[0] = sample_discrete(prior,1,1)
     for t in range(1,length):
          state[t] = sample_discrete(trans[int(act[t])][int(state[t-1])],1,1) 

     return state  

def main():
    prior = np.array([1,0])
    a = np.ones(25) - 1
    b = np.ones(25)*2 - 1

    act = np.concatenate([a,b,a,b])

    trans = np.ones((2,2,2))

    trans[0][0] = [0.9,0.1]
    trans[0][1] = [0.1,0.9]
    trans[1][0] = [0.1,0.9]
    trans[1][1] = [0.9,0.1]
    
    test = sample_mdp(prior, trans, act)
    print "test = ", test

if __name__ == "__main__":
     main()
