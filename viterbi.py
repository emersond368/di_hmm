import numpy as np

def viterbi(obs,start_p,trans_p,emit_p):
    #trans_p = transition matrix, N x N (hidden state to hidden state transition, len(states) = N)
    # obs =  sequence of observations, 1xT (output observation over time, obs[i] = 0 .. K-1)
    # start_p = start probability for each hidden state, 1xN
    # emit_P = emission matrix, K x N (probability from hidden state to output observation)
    #          N = len(states), K = len(number of possible observation types)
    
    obslink = np.zeros((len(emit_p),len(obs)),dtype = np.float)
    for t in range(0,len(obs)):
         obslink[:,t] = emit_p[obs[t]]
    
    Vprob = np.zeros((len(obs),len(start_p))) 
    Vprev = np.zeros((len(obs),len(start_p))) 

    for st in range(0,len(start_p)):
        Vprob[0][st] = start_p[st] * obslink[st][0]
        Vprev[0][st] = None    

    Vprob[0] = normalize(Vprob[0])

    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        for st in range(0,len(start_p)):
            max_tr_prob = max(Vprob[t-1][prev_st]*trans_p[prev_st][st] for prev_st in range(0,len(start_p)))
            for prev_st in range(0,len(start_p)):
                if Vprob[t-1][prev_st] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * obslink[st][t]
                    Vprob[t][st] =  max_prob
                    Vprev[t][st] = prev_st
                    break
        Vprob[t] = normalize(Vprob[t])

    print("Vprob =",Vprob)
   
    opt = []
    # The highest probability
    max_prob = max(value for value in Vprob[-1])
    previous = None
    # Get most probable state and its backtrack starting at end
    for st in range(0,len(start_p)):
        if Vprob[-1][st] == max_prob:
            opt.append(st)
            previous = Vprev[-1][st]
            break
    # Follow the backtrack till the first observation
    for t in range(len(Vprev) - 1, 0, -1):
        opt.insert(0, Vprev[t][previous])
        previous = Vprev[t][int(previous)]


    return opt

def normalize(col):
    z = np.sum(col)
    if z == 0:
         z = 1
    return(col/z)

def main():

     observations = [1,2,1]
     start_probability = [0.2,0.3,0.5]
     transition_probability = [[0.5,0.2,0.3],[0.7,0.1,0.2],[0.3,0.4,0.3]]
     emission_probability = [[0.2,0.4,0.4],[0.3,0.6,0.1],[0.7,0.2,0.1]]  

     output = viterbi(observations,start_probability,transition_probability,emission_probability)
     print output


if __name__ == "__main__":
     main()

