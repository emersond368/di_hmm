from mhmm_em import mhmm_em
from mhmm_logprob import mhmm_logprob
from mixgauss_prob import mixgauss_prob
from viterbi_path import viterbi_path
from hmmFwdBack import hmmFwdBack
from jie_kmeans_init import jie_kmeans_init
import hmmFwdBackc
import fwdback2c
import numpy as np
import csv
import argparse
#from MatrixMakeBandedHermitian4DI_removelow import MatrixMakeBanded #old version of modv2 - too stringent in removing lowcount values
from MatrixMakeBandedHermitian4DI_nans import MatrixMakeBanded
from DI8 import DI
from convertbed2v2_dynamic import BedConvert

"attempt to remove out empty regions plus counts < 1"


class TAD_calls:
 
     Q = 3  #Hidden State number

     def __init__(self,testX,start,maximum):
          self.testX = testX
          self.maximum = maximum
          self.start = start
          self.aic = np.zeros((maximum-start)+1)
          self.ll = np.zeros((maximum-start)+1)
          self.best_p = np.zeros(((maximum - start)+1 ,len(testX)))
          self.best_d = np.zeros(((maximum - start)+1,len(testX)))
          self.best_g1 = np.zeros(((maximum - start)+1,len(testX)))
          self.best_g2 = np.zeros(((maximum - start)+1,len(testX)))
          self.best_g3 = np.zeros(((maximum-start)+1,len(testX)))
          self.final_g = np.zeros((len(testX),TAD_calls.Q))
          self.comprehensive = np.zeros((len(testX),TAD_calls.Q+2))
          self.final_p = np.array([])
          self.final_d = np.array([])
          self.miniac = 0

     def TAD_calls_Start(self):

          #Initiliaze initial state
          prior0 = np.array([0.33,0.33,0.33]);       # Suppose always starting a TAD

          #Initialize the transition matrix
          #Initalized intuitively with the transition likelihood
          #                       1   2   3
          transmat0 = np.array([[0.33,0.33,0.33],[0.33,0.33,0.33],[0.33,0.33,0.33]]) 
          

          for M in range(self.start,self.maximum+1):  # component mixture range

               #M = 2
               mu0, Sigma0,mx = jie_kmeans_init(self.testX,TAD_calls.Q,M)
               Sigma0 = np.reshape(Sigma0,(1,1,TAD_calls.Q,M), order = 'F')
               mu0 = np.reshape(mu0,(1,TAD_calls.Q,M), order = 'F')
               mixmat0 = (1/float(M))*np.ones((TAD_calls.Q,M))

               LL,prior1,transmat1,mu1,Sigma1,mixmat1 = mhmm_em(self.testX,prior0,transmat0,mu0,Sigma0,mixmat0,max_iter=500,thresh=1e-5)
               loglik = mhmm_logprob(self.testX, prior1, transmat1, mu1, Sigma1, mixmat1)
               B,B2 = mixgauss_prob(self.testX,mu1,Sigma1,mixmat1)
               path = viterbi_path(prior1, transmat1, B)
              # gamma,alpha,beta,logp = hmmFwdBack(prior1,transmat1,B) #python code
               alpha,beta,gamma,logp = hmmFwdBackc.MF(prior1,transmat1,B,1)            

               decodedFromEMmaxMarg = np.argmax(gamma,axis=0)

               mu1_shape_length = len(mu1.shape)
               Sigma1_shape_length = len(Sigma1.shape)
               mixmat1_shape_length = len(mixmat1.shape)

               mu1_size = 1
               for i in range(mu1_shape_length):
                   mu1_size = mu1_size*mu1.shape[i]

               Sigma1_size = 1
               for i in range(Sigma1_shape_length):
                   Sigma1_size = Sigma1_size*Sigma1.shape[i]

               mixmat1_size = 1
               for i in range(mixmat1_shape_length):
                   mixmat1_size = mixmat1_size*mixmat1.shape[i]

               #if bool(M==1):
               #     num_p = 1*len(prior1) + (len(transmat1)*len(transmat1[0])) + (len(mu1)*1) + (len(Sigma1)*len(Sigma1[0]))+ (len(mixmat1)*1)
               #else:
               #     num_p = 1*len(prior1) + (len(transmat1)*len(transmat1[0])) +(len(mu1)*len(mu1[0])*len(mu1[0][0])) + (len(Sigma1)*len(Sigma1[0])*len(Sigma1[0][0])*len(Sigma1[0][0][0]))+ (len(mixmat1)*len(mixmat1[0]))  

               #update
               print("mu1.shape: ",mu1.shape)
               print("Sigma1.shape: ",Sigma1.shape)
               print("mixmat1.shape: ",mixmat1.shape)

               num_p = 1*len(prior1) + (len(transmat1)*len(transmat1[0])) + mu1_size + Sigma1_size + mixmat1_size


               self.aic[M - self.start] = -2*loglik + 2*num_p
               self.ll[M - self.start] = loglik
               self.best_p[M - self.start,:] = path
               self.best_d[M - self.start,:] = decodedFromEMmaxMarg
               self.best_g1[M - self.start,:] = gamma[0,:]
               self.best_g2[M - self.start,:] = gamma[1,:]
               self.best_g3[M - self.start,:] = gamma[2,:]


          #savetxt('gamma_test',gamma)
          order = np.floor(np.log10(np.abs(np.min(self.aic))))
          div = np.power(10,order)

          ind = 0
          paic = np.zeros((self.maximum -self.start)+1)
          paic_score_max = 0 
          for k in range(0,(self.maximum -self.start)+1):
               paic[k] = np.exp((np.min(self.aic)-self.aic[k])/(div*2))
               #if (paic[k] >= paic_score_max):
               if (paic[k] >= 0.90):
                   ind = k 
                   paic_score_max = paic[k]
                   break #stop immediately after paic exceeds 90 (possibility of overfitting
               print("paic[",k,"]"," = ",paic[k])
          print("paic_score_max = ", paic_score_max)
          print("chosen index = ", ind)

          self.final_p = self.best_p[ind,:]
          self.final_d = self.best_d[ind,:]
          self.final_g[:,0] = self.best_g1[ind,:]
          self.final_g[:,1] = self.best_g2[ind,:]
          self.final_g[:,2] = self.best_g3[ind,:]
          self.comprehensive[:,0] = self.final_p + 1 #adjusting hidden states Q to 1-3 for post process
          self.comprehensive[:,1] = self.final_d + 1
          self.comprehensive[:,2] = self.final_g[:,0]
          self.comprehensive[:,3] = self.final_g[:,1]
          self.comprehensive[:,4] = self.final_g[:,2]
          self.final_aic = self.aic[ind]
          self.miniac = np.min(self.aic)


def main():
     
     parser = argparse.ArgumentParser()
     parser.add_argument("count", type=str, help="count file")
     parser.add_argument("bed", type=str, help="bed file")
     parser.add_argument("DIrange", type=int, help="range")
     parser.add_argument("istart", type=int, help="index start")
     parser.add_argument("iend",type=int,help="index end")
     parser.add_argument("startchr",type=int, help="chr start")
     parser.add_argument("endchr",type=int, help="chr end")
     parser.add_argument("binsize",type=int,help="binsize")
     parser.add_argument("window", nargs='?',type=int, default = 1, help="running window size across diagonal")
     parser.add_argument('distance', nargs='?',type = int,default=0, help ='gap from diagonal')
     parser.add_argument('diagonal', nargs='?', default=True, help ='banded matrix = false, normal heatmap = true')
     parser.add_argument('min_cluster', nargs='?',type = int,default=0, help ='minimum cluster number')
     parser.add_argument('cull_size', nargs='?',type = int,default=62, help ='sparse region removal threshold')
     parser.add_argument('region', nargs='?', default="regionX",help ='name of the genome')
     args = parser.parse_args()
 
     if args.diagonal == "False":
          args.diagonal = False

     data = MatrixMakeBanded(args.count,args.istart,args.iend,args.DIrange,args.distance,args.region)

     value = DI(data.Heatmap,args.distance,args.DIrange,args.window,args.diagonal)

     value.DI_metric()
     value.scramble_DI_metric()

     Data2 = BedConvert(args.bed,value.DIset,args.DIrange,args.istart,args.iend,args.startchr,args.endchr,args.binsize,args.window,args.distance,args.cull_size)
     
     pre = np.array(Data2.output3,dtype = float)
     testX = pre[:,3]

     Data3 = TAD_calls(testX,args.min_cluster,10)   
     Data3.TAD_calls_Start()  
     np.savetxt('output/'+ args.region + 'aic.csv', Data3.aic, delimiter=',')
     np.savetxt('output/'+ args.region + 'll.csv', Data3.ll, delimiter=',')
     np.savetxt('output/'+ args.region + 'best_p.csv', Data3.best_p, delimiter=',', fmt='%d')
     np.savetxt('output/'+ args.region + 'best_d.csv', Data3.best_d, delimiter=',',fmt='%d')
     np.savetxt('output/'+ args.region + 'best_g1.csv', Data3.best_g1, delimiter=',')
     np.savetxt('output/'+ args.region + 'best_g2.csv', Data3.best_g2, delimiter=',')
     np.savetxt('output/'+ args.region + 'best_g3.csv', Data3.best_g3, delimiter=',')
     np.savetxt('output/'+ args.region + 'comprehensive.csv', Data3.comprehensive, delimiter=',')

     postp_list = np.zeros((len(Data2.output3),len(Data3.comprehensive[0])+len(Data2.output3[0])))
     postp_list[:,0:len(Data2.output3[0])] = Data2.output3
     postp_list[:,len(Data2.output3[0]):len(Data3.comprehensive[0])+len(Data2.output3[0])] = Data3.comprehensive
     np.savetxt('output/'+ args.region + '_postp_list.csv', postp_list, delimiter='\t',fmt='%.10f')



if __name__ == "__main__":
     main()


     
