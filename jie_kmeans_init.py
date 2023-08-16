import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.cluster.vq import kmeans,vq

def jie_kmeans_init(data,Q,M):

     if data.ndim == 1:
          O = 1
          sz = len(data)
     else:
          O = len(data) 
          sz = len(data[0])

     mu0 = np.zeros((O,Q,M))
     sp = np.zeros((Q,M))
     mixmat = np.zeros((Q,M))
     
     Ot1 = np.unique(data[data>0])
     if M <= len(Ot1):
      #    ct = []
      #    iteration = 10 # python default
      #    while (len(ct) != M and iteration > 0):
      #         ct,cls = kmeans2(Ot1,M,iter = iteration)
      #         iteration = iteration - 1
          centroids,_ = kmeans(Ot1,M)
          idx,_ = vq(Ot1,centroids) 
           
          if (len(centroids) == M and len(np.unique(idx)) == M): 
               mu0[0,0,:] = centroids
               sp[0,:] = kmean_var(Ot1,idx,M)
               mixmat[0,:] = mixmatMake(idx,M,sz)
          else:
               limit = len(centroids)
               if len(centroids) > M:
                    centroids = centroids[0:M+1]
               elif len(centroids) < M:
                   while len(centroids) != M: 
                        centroids = np.append(centroids,np.mean(centroids))
               mu0[0,0,:] = centroids
               if len(np.unique(idx)) > M:
                    var = kmean_var(Ot1,idx,M)
               elif len(np.unique(idx)) < M:
                    var = kmean_var(Ot1,idx,len(np.unique(idx)))
                    while len(var) != M:
                         var = np.append(var,np.mean(var))
               sp[0,:] = var
               mixmat[0,:] = mixmatMake(idx,M,sz)
          
     else:
          mu0[0,0,:] = np.linspace(0,np.min(data),num=M)
          sp[0,:] = np.power((np.max(Ot1)/(M-1)/3),2)
          mixmat[0,:] = (1/(M*Q))*np.ones(M)
 
     Ot3 = np.unique(data[data < 0])
     if M <= len(Ot3):
     #     ct = []
     #     iteration = 10 # python default
     #     while (len(ct) != M and iteration > 0):
     #          ct,cls = kmeans2(Ot3,M,iter = iteration)
     #          iteration = iteration - 1
          centroids,_ = kmeans(Ot3,M)
          idx,_ = vq(Ot3,centroids)
          if (len(centroids) == M and len(np.unique(idx)) == M):
               mu0[0,2,:] = centroids
               sp[2,:] = kmean_var(Ot3,idx,M)
               mixmat[2,:] = mixmatMake(idx,M,sz)
          else:
               limit = len(centroids)
               if len(centroids) > M:
                    centroids = centroids[0:M+1]
               elif len(centroids) < M:
                   while len(centroids) != M:
                        centroids = np.append(centroids,np.mean(centroids))
               mu0[0,2,:] = centroids
               if len(np.unique(idx)) > M:
                    var = kmean_var(Ot3,idx,M)
               elif len(np.unique(idx)) < M:
                    var = kmean_var(Ot3,idx,len(np.unique(idx)))
                    while len(var) != M:
                         var = np.append(var,np.mean(var))
               sp[2,:] = var
               mixmat[2,:] = mixmatMake(idx,M,sz)
     else:
          mu0[0,2,:] = np.linspace(0,np.min(data),num=M)
          sp[2,:] = np.power((np.min(Ot3)/(M-1)/3),2)
          mixmat[2,:] = (1/(M*Q))*np.ones(M)

     Ot2 = data[data < (np.max(Ot1)/20)]
     Ot2 = Ot2[Ot2 > (np.min(Ot3)/20)]
     if M <= len(Ot2):
          centroids,_ = kmeans(Ot2,M)
          idx,_ = vq(Ot2,centroids)
          if (len(centroids) == M and len(np.unique(idx)) == M):
               mu0[0,1,:] = centroids
               sp[1,:] = kmean_var(Ot2,idx,M)
               mixmat[1,:] = mixmatMake(idx,M,sz)
          else:
               limit = len(centroids)
               if len(centroids) > M:
                    centroids = centroids[0:M+1]
               elif len(centroids) < M:
                   while len(centroids) != M:
                        centroids = np.append(centroids,np.mean(centroids))
               mu0[0,1,:] = centroids
               if len(np.unique(idx)) > M:
                    var = kmean_var(Ot2,idx,M)
               elif len(np.unique(idx)) < M:
                    var = kmean_var(Ot2,idx,len(np.unique(idx)))
                    while len(var) != M:
                         var = np.append(var,np.mean(var))
               sp[1,:] = var
               mixmat[1,:] = mixmatMake(idx,M,sz)


     else:
          mu0[0,1,:] = np.linspace(0,np.min(data),num=M)
          sp[1,:] = np.power((np.max(Ot1)/20 - np.min(Ot3)/20)/(M-1)/3,2)
          mixmat[1,:] = (1/(M*Q))*np.ones(M)

     print("mixmat sum = ", np.sum(mixmat))
     return mu0,sp,mixmat


def mixmatMake(cls,M,data_size):
  
    section = np.zeros(M)
    for i in range(0,M):
         val = list(cls).count(i)
         section[i] = float(val)/data_size

    return section


def kmean_var(Op,cls,M):
     spv = np.zeros(M)
     for i in range(0,M):
          spv[i] = np.var(Op[cls == i])
          
     return spv
     
def main():

     DI = np.loadtxt("testfiles/DItest.csv",delimiter = ',')

     M = 20 
     Q = 3
     for i in range(1,M+1):
          mu0, sp,mixmat = jie_kmeans_init(DI,Q,i)

          print("mu0 = ",mu0)
          print("sp = ",sp)
          print("mixmat = ", mixmat)

if __name__ == "__main__":
     main()

