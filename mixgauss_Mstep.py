import numpy as np
import numpy.matlib

def mixgauss_Mstep(w, Y, YY, YTY, cov_type = 'full',tied_cov = 0, clamped_cov = [],clamped_mean = [], cov_prior = []):

     # MSTEP_COND_GAUSS Compute MLEs for mixture of Gaussians given expected sufficient statistics
     # function [mu, Sigma] = Mstep_cond_gauss(w, Y, YY, YTY, varargin)
     #
     # We assume P(Y|Q=i) = N(Y; mu_i, Sigma_i)
     # and w(i,t) = p(Q(t)=i|y(t)) = posterior responsibility
     # See www.ai.mit.edu/~murphyk/Papers/learncg.pdf.
     #
     # INPUTS:
     # w(i) = sum_t w(i,t) = responsibilities for each mixture component, imput ndim =2
     #  If there is only one mixture component (i.e., Q does not exist),
     #  then w(i) = N = nsamples,  and 
     #  all references to i can be replaced by 1.
     # YY(:,:,i) = sum_t w(i,t) y(:,t) y(:,t)' = weighted outer product, input ndim = 3
     # Y(:,i) = sum_t w(i,t) y(:,t) = weighted observationsa, input ndim =3
     # YTY(i) = sum_t w(i,t) y(:,t)' y(:,t) = weighted inner product, input ndim = 2
     #   You only need to pass in YTY if Sigma is to be estimated as spherical.
     #
     # Optional parameters may be passed as 'param_name', param_value pairs.
     # Parameter names are shown below; default values in [] - if none, argument is mandatory.
     #
     # 'cov_type' - 'full', 'diag' or 'spherical' ['full']
     # 'tied_cov' - 1 (Sigma) or 0 (Sigma_i) [0]
     # 'clamped_cov' - pass in clamped value, or [] if unclamped [ [] ]
     # 'clamped_mean' - pass in clamped value, or [] if unclamped [ [] ]
     # 'cov_prior' - Lambda_i, added to YY(:,:,i) [0.01*eye(d,d,Q)]
     #
     # If covariance is tied, Sigma has size d*d.
     # But diagonal and spherical covariances are represented in full size.

     Y = np.array(Y,dtype = np.float64)
     YY = np.array(YY,dtype = np.float64)
     YTY = np.array(YTY,dtype = np.float64)
     w = np.array(w,dtype = np.float64)
     vals = Y.shape
     
     Q = 1
     for i in range(1,len(vals)): #condense all dimensions into one after first
          Q = Q*vals[i]  
     Ysz = vals[0]
     
     N = np.sum(w,axis = 0)

     if not cov_prior:  #default empty [] 
          print("mixgauss_Mstep not cov_prior")
          cov_prior = np.zeros((Ysz,Ysz,Q))
          for i in range(0,Q):
               cov_prior[:,:,i] = 0.01*np.eye(Ysz)
      #    cov_prior = np.tile(0.01*np.eye(Ysz),[Q,1,1])
      #    cov_prior = np.matlib.repmat(0.01*np.eye(Ysz),Q,1)
      #    cov_prior = np.transpose(cov_prior, axes=(2, 1, 0))
  
     YY = np.reshape(YY,(Ysz,Ysz,Q),order ='F')
     
     # Set any zero weights to one before dividing
     # This is valid because w(i)=0 => Y(:,i)=0, etc
     w[w==0] = 1
     if bool(clamped_mean):  #clamped_mean is non-empty
          mu = clamped_mean
     else:
          #mu = Y ./ repmat(w(:)', [Ysz 1]);% Y may have a funny size
          mu = np.zeros((Ysz,Q))  #4x6
          for i in range(0,Q): 
              a = i % len(Y[0])
              b = int(i/len(Y[0]))
              if len(Y[0][0]) == 1:
                   mu[:,i] = Y[:,a,b]/w[a,i]
              else:   
                   mu[:,i] = Y[:,a,b]/w[a,b]   # input Y MUST be 3 for ndim, input w MUST be 2 for ndim
 #             print("mu[:,i] = ",mu[:,i])
 #             print("corresponding Y[:,a,b] = ",Y[:,a,b])
 #             print("corresponding w[a,b] = ",w[a,i])

     if bool(clamped_cov): #clamped_cov is non-empty
          Sigma = clamped_cov
          return (mu,Sigma)

     if not bool(tied_cov): #if tied_cov == 0, default option
          Sigma = np.zeros((Ysz,Ysz,Q))
          for i in range(0,Q):
               a = i % len(Y[0])
               b = int(i/len(Y[0]))       
               if cov_type[0] == 's':  #spherical type
                    if len(Y[0][0]) == 1:   #unmixed M=1 case
                         s2 = np.dot((1/float(Ysz)),((YTY[a,b]/w[a,i]) - np.dot(mu[:,i],mu[:,i])))
                    else:
                         s2 = np.dot((1/float(Ysz)),((YTY[a,b]/w[a,b]) - np.dot(mu[:,i],mu[:,i])))
          #          print "s2 = ",s2
                    Sigma[:,:,i] = s2*np.eye(Ysz)
               else:
                    if len(Y[0][0]) == 1: #unmixed M=1 case
                         SS = YY[:,:,i]/w[a,i] - np.outer(mu[:,i],mu[:,i])
                    else:
                         SS = YY[:,:,i]/w[a,b] - np.outer(mu[:,i],mu[:,i])
                    if cov_type[0] == 'd':  #diag type
                         SS = np.eye(len(SS))*np.diag(SS)
                    Sigma[:,:,i] = SS
     else: #tied !=0
          if cov_type[0] == 's':  #spherical typae
               s2 = (1/float(N*Ysz))*(np.sum(YTY) + np.sum(np.diag(np.dot(mu.T,mu))*w)) #actual
       #        w = np.reshape(w,(len(mu[0])),order = 'F') #temp
       #        YTY = np.reshape(YTY,(len(mu[0])), order = 'F') #temp
       #        print(sum(np.diag(np.dot(mu.T,mu))*w))
       #        s2 = (1/float(N[0]*Ysz))*(np.sum(YTY) + np.sum(np.diag(np.dot(mu.T,mu))*w)) #temp
               Sigma = s2*np.eye(Ysz)
          else:
               SS = np.zeros((Ysz,Ysz))
               for i in range(0,Q): #probably could vectorize this...
                    SS = SS + YY[:,:,i]/N - np.outer(mu[:,i],mu[:,i])  #actual
              #      SS = SS + YY[:,:,i]/N[0] - np.outer(mu[:,i],mu[:,i]) #temp for testing
               if cov_type[0] == 'd':
                    Sigma = np.eye(len(SS))*np.diag(SS)
               else:
                    Sigma = SS
     if bool(tied_cov):
          preSigma = Sigma
          Sigma = np.zeros((Ysz,Ysz,Q))
          for i in range(0,Q):
               Sigma[:,:,i] = preSigma
     Sigma = Sigma + cov_prior
   
     return mu,Sigma

def main():
    
     ip = np.array([[446.7469,369.22690],[302.3985,364.6515],[300.4642,220.9969]])
     op = np.zeros((4,4,3,2))
 #    op = np.array([129.3866,-65.1538,3.8718,-17.9649,-65.1538,159.1733,8.5950,30.7550,\
 #         3.8718,8.5950,70.2154,-6.7307,-17.9649,30.7550,-6.7307,87.9716,\
 #         44.8652,1.6636,14.9068,8.4458,1.6636,39.8239,3.4233,4.7827,\
 #         14.9068,3.4233,94.9007,44.1430,8.4458,4.7827,44.1430,122.8087,\
 #         58.3013,20.1143,6.2855,-2.0699,20.1143,46.6704,9.6580,2.4661,\
 #         6.2855,9.6580,113.5524,-44.3034,-2.0699,2.4661,-44.3034,81.9400,\
 #         138.5687,5.5371,-21.3999,27.9621,5.5371,59.7806,-22.2310,-13.7899,\
 #         -21.3999,-22.2310,78.0432,-5.3573,27.9621,-13.7899,-5.3573,92.8344,\
 #         48.3545,-3.7040,5.7409,-18.2977,-3.7040,153.7255,-8.4454,-19.9347,\
 #         5.7409,-8.4454,85.0793,0.9044,-18.2977,-19.9347,0.9044,77.4921,\
 #         118.8099,17.6523,-24.2923,27.8088,17.6523,17.5794,-5.0764,-4.8365,\
 #         -24.2923,-5.0764,35.7944,-6.9292,27.8088,-4.8365,-6.9292,48.8132])

 #    op = np.reshape(op,(4,4,3,2),order ='F')

     op[:,:,0,0] = [[129.3866,58.3013,48.3545,-65.1538],[3.8718,6.2855,5.7409,8.5950],\
                   [44.8652,138.5687,118.8099,1.6636],[14.9068,-21.3999,-24.2923,3.4233]]
     op[:,:,1,0] = [[20.1143,-3.7040,3.8718,6.2855],[9.6580,-8.4454,70.2154,113.5524],\
                   [5.5371,17.6523,14.9068,-21.3999],[-22.2310,-5.0764,94.9007,78.0432]]
     op[:,:,2,0] = [[5.7409,-17.9649,-2.0699,-18.2977],[85.0793,-6.7307,-44.3034,0.9044],\
                    [-24.2923,8.4458,27.9621,27.8088],[35.7944,44.1430,-5.3573,-6.9292]]
     op[:,:,0,1] = [[-65.1538,20.1143,-3.7040,159.1733],[-17.9649,-2.0699,-18.2977,30.7550],\
                    [1.6636,5.5371,17.6523,39.8239],[8.4458,27.9621,27.8088,4.7827]]
     op[:,:,1,1] = [[46.6704,153.7255,8.5950,9.6580],[2.4661,-19.9347,-6.7307,-44.3034],\
                    [59.7806,17.5794,3.4233,-22.2310],[-13.7899,-4.8365,44.1430,-5.3573]]
     op[:,:,2,1] = [[-8.4454,30.7550,2.4661,-19.9347],[0.9044,87.9716,81.9400,77.4921],\
                    [-5.0764,4.7827,-13.7899,-4.8365],[-6.9292,122.8087,92.8344,48.8132]]


    
     m = np.zeros((4,3,2))
     m[:,:,0] = [[63.1986,10.6329,-6.7411],[-106.3036,7.2389,-13.5990],[1.6596,55.9700,-74.4703],\
        [-35.3472,72.7011,48.9295]]
     m[:,:,1] = [[-67.7503,-12.4145,75.3334],[-14.1406,95.2812,10.0597],\
        [19.1364,14.2754,-24.8621],[-55.5247,-14.7454,14.8941]]

     postmix = np.array([[99.8917,86.0249],[76.4194,93.2569],[82.0396,62.3676]])
     testSigma2,testmu2 = mixgauss_Mstep(postmix, m, op, ip)

       #Outputs SHOULD BE:
  #   Sigma2 = np.zeros((4,4,6))
  #   Sigma2[:,:,0] = [[0.9050,1.2569,0.4736,-0.4284],[0.7120,-1.0596,0.0752,-0.2905],\
  #                   [0.4386,1.4049,1.1991,0.0225],[0.3731,-0.5908,-0.2373,-0.0809]]
  #   Sigma2[:,:,1]= [[0.2538,-0.0616,-0.0512,-0.0501],[0.1132,-0.1095,0.8494,1.3958],\
  #                  [-0.0294,0.1616,-0.3314,-0.9768],[-0.4233,-0.1565,0.5451,0.1262]]
  #   Sigma2[:,:,2] = [[0.0732,-0.2326,-0.0998,-0.1740],[1.0234,-0.0995,-0.6905,0.1099],\
  #                   [-0.3707,-0.0475,-0.4731,0.8804],[0.4853,0.6369,0.4761,-0.4302]],\
  #   Sigma2[:,:,3] = [[-1.3676,0.1044,0.1321,1.3420],[-0.3383,-0.0411,-0.1761,0.2514],\
  #                   [0.1945,0.1009,0.1657,0.6065],[-0.4102,0.2189,0.4668,-0.3510]],\
  #   Sigma2[:,:,4] = [[0.4927,1.7844,0.1125,0.0825],[0.1625,-1.2476,-0.2286,-0.3135],\
  #                   [0.6614,0.0321,0.0233,-0.2142],[-0.1689,0.1097,0.4976,-0.0724]]
  #   Sigma2[:,:,5] = [[-1.5844,0.2983,0.5211,-0.6081],[-0.1803,1.3945,1.3781,1.2040],\
  #                   [0.4001,0.1410,-0.3700,0.0177],[-0.3996,1.9306,1.5837,0.7356]]

  #   mu2 = np.array([[0.632671182891071,0.139138752725093,-0.0821688550407364,\
  #          -0.787566158170483,-0.133121517013755,1.20789320095691],\
  #          [-1.06418851616300,0.0947259465528387,-0.165761412781145,\
  #          -0.164377988233639,1.02170670481219,0.161296891334603],\
  #          [0.0166139929543696,0.732405645686828,-0.907736020165871,\
  #          0.222451871493021,0.153076072655214,-0.398638074897864],\
  #          [-0.353855225208901,0.951343507015234,0.596413195578721,\
  #          -0.645449166462269,-0.158115914211174,0.238811498277952]])
            
     print("testSigma2[0][0][1] = ",testSigma2[0][0][1])
     print("testSigma2[2][3][5] = ",testSigma2[2][3][5])
     print("testSigma2[3][1][4] = ",testSigma2[3][1][4])
     print("testmu2 = ",testmu2)

     testSigma2,testmu2 = mixgauss_Mstep(postmix, m, op, ip,cov_type = "diag")
     print("testSigma2[0][0][0] = ",testSigma2[0][0][0])
     print("testSigma2[3][3][3] = ", testSigma2[3][3][3])
     print("testSigma2[3][0][0] = ", testSigma2[3][0][0])
 
     testSigma2,testmu2 = mixgauss_Mstep(postmix, m, op, ip,cov_type = "spherical")
#     print("testSigma2 = ",testSigma2)
     print("testSigma2[0][0][0] = ",testSigma2[0][0][0])
     print("testSigma2[3][3][3] = ", testSigma2[3][3][3])
     print("testSigma2[3][0][0] = ", testSigma2[3][0][0])

 #    testSigma2,testmu2 = mixgauss_Mstep(postmix, m, op, ip,cov_type = "spherical",tied_cov = 1)
 #    print("testSigma2[0][0][0] = ",testSigma2[0][0][0]) #2.61055812965
 #    print("testSigma2[3][3][3] = ", testSigma2[3][3][3]) #2.61055812965
 #    print("testSigma2[3][0][0] = ", testSigma2[3][0][0]) #0

 #    testSigma2,testmu2 = mixgauss_Mstep(postmix, m, op, ip,cov_type = "diagonal",tied_cov = 1)
 #    print("testSigma2[0][0][0] = ",testSigma2[0][0][0]) #-2.01670982025
 #    print("testSigma2[3][3][3] = ", testSigma2[3][3][3]) #-1.39938328844
 #    print("testSigma2[3][0][0] = ", testSigma2[3][0][0]) #0

 #    testSigma2,testmu2 = mixgauss_Mstep(postmix, m, op, ip,cov_type = "full",tied_cov = 1)
 #    print("testSigma2[0][0][0] = ",testSigma2[0][0][0])  #-2.01670982025
 #    print("testSigma2[3][3][3] = ", testSigma2[3][3][3]) #-1.39938328844
 #    print("testSigma2[3][0][0] = ", testSigma2[3][0][0]) #-0.614636028631

if __name__ == "__main__":
     main()


