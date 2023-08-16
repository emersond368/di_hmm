import numpy as np

def sample_discrete(prob,r,c):
    # SAMPLE_DISCRETE Like the built in 'rand', except we draw from a non-uniform discrete distrib.
    # M = sample_discrete(prob, r, c)
    #
    #  Example: sample_discrete([0.8 0.2], 1, 10) generates a row vector of 10 random integers from {1,2},
    # where the prob. of being 1 is 0.8 and the prob of being 2 is 0.2.

    
    R = np.random.rand(r,c)
    M = np.ones((r,c))
    cumprob = np.ones(len(prob))

    sum = 0
    for i in range(0,len(prob)):
        sum = sum + prob[i]
        cumprob[i] = sum

    if (len(prob) < r*c):
        for i in range(0,len(prob)-1):
             M = M + np.array(R > cumprob[i],dtype = np.int)
    else:
        cumprob2 = cumprob[0:-1]
        for i in range(0,r):
             for j in range(0,c):
                  M[i,j] = np.sum(np.array(R[i,j] > cumprob2,dtype = np.int))+1

    return M-1 #indices start at zero not one

def main():
     M = sample_discrete([0.2,0.3,0.5],2,10)
     print M

if __name__ == "__main__":
     main()


