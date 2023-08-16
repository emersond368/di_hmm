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
from MatrixMakeBanded4 import MatrixMakeBanded
from DI8 import DI
from convertbed2 import BedConvert

class TAD_calls:
 
     Q = 3  #Hidden State number

     def __init__(self,testX,maximum):
          self.testX = testX
          self.maximum = maximum
          self.aic = np.zeros(maximum)
          self.ll = np.zeros(maximum)
          self.best_p = np.zeros((maximum,len(testX)))
          self.best_d = np.zeros((maximum,len(testX)))
          self.best_g1 = np.zeros((maximum,len(testX)))
          self.best_g2 = np.zeros((maximum,len(testX)))
          self.best_g3 = np.zeros((maximum,len(testX)))
          self.final_g = np.zeros((len(testX),TAD_calls.Q))
          self.comprehensive = np.zeros((len(testX),TAD_calls.Q+2))
          self.final_p = np.array([])
          self.final_d = np.array([])
          self.miniac = 0
          self.TAD_calls_Start()
          

     def TAD_calls_Start(self):

          #Initiliaze initial state
          prior0 = np.array([0.33,0.33,0.33]);       # Suppose always starting a TAD

          #Initialize the transition matrix
          #Initalized intuitively with the transition likelihood
          #                       1   2   3
          transmat0 = np.array([[0.33,0.33,0.33],[0.33,0.33,0.33],[0.33,0.33,0.33]]) 


          for M in range(1,self.maximum+1):  # component mixture range

               #M = 2
               mu0, Sigma0,mx = jie_kmeans_init(self.testX,TAD_calls.Q,M)
               Sigma0 = np.reshape(Sigma0,(1,1,TAD_calls.Q,M), order = 'F')
               mu0 = np.reshape(mu0,(1,TAD_calls.Q,M), order = 'F')
               mixmat0 = (1/float(M))*np.ones((TAD_calls.Q,M))

               LL,prior1,transmat1,mu1,Sigma1,mixmat1 = mhmm_em(self.testX,prior0,transmat0,mu0,Sigma0,mixmat0,max_iter=500,thresh=1e-5)
               loglik = mhmm_logprob(self.testX, prior1, transmat1, mu1, Sigma1, mixmat1)
               B,B2 = mixgauss_prob(self.testX,mu1,Sigma1,mixmat1)
               path = viterbi_path(prior1, transmat1, B)
               alpha,beta,gamma,logp = hmmFwdBackc.MF(prior1,transmat1,B,1)              

               decodedFromEMmaxMarg = np.argmax(gamma,axis=0)

               if bool(M==1):
                    num_p = 1*len(prior1) + (len(transmat1)*len(transmat1[0])) + (len(mu1)*1) + (len(Sigma1)*len(Sigma1[0]))+ (len(mixmat1)*1)
               else:
                    num_p = 1*len(prior1) + (len(transmat1)*len(transmat1[0])) +(len(mu1)*len(mu1[0])*len(mu1[0][0])) + (len(Sigma1)*len(Sigma1[0])*len(Sigma1[0][0])*len(Sigma1[0][0][0]))+ (len(mixmat1)*len(mixmat1[0]))  


               self.aic[M -1] = -2*loglik + 2*num_p
               self.ll[M-1] = loglik
               self.best_p[M-1,:] = path
               self.best_d[M-1,:] = decodedFromEMmaxMarg
               self.best_g1[M-1,:] = gamma[0,:]
               self.best_g2[M-1,:] = gamma[1,:]
               self.best_g3[M-1,:] = gamma[2,:]


          order = np.floor(np.log10(np.abs(np.min(self.aic))))
          div = np.power(10,order)

          ind = 0
          paic = np.zeros(self.maximum)
          for k in range(0,self.maximum):
               paic[k] = np.exp((np.min(self.aic)-self.aic[k])/(div*2))
               if (paic[k] >= 0.9):
                   ind = k
                   print("paic[",k,"]"," = ",paic[k])
                   break
      
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
     parser.add_argument('region', nargs='?', default="regionX",help ='name of the genome')
     args = parser.parse_args()
 
     if args.diagonal == "False":
          args.diagonal = False

     data = MatrixMakeBanded(args.count,args.istart,args.iend,args.DIrange,args.distance,args.region)
     value = DI(data.Heatmap,args.distance,args.DIrange,args.window,args.diagonal)
     value.DI_metric()
     value.scramble_DI_metric()
     print("value.DIset = ",value.DIset)
     print("len(value.DIset) = ",len(value.DIset))
     np.savetxt("DIsetTest",value.DIset,delimiter=',',fmt='%f')

     Data2 = BedConvert(args.bed,value.DIset,args.DIrange,args.istart,args.iend,args.startchr,args.endchr,args.binsize,args.window,args.distance)
     
     print("Data2.output2 = ",Data2.output2)
     pre = np.array(Data2.output2,dtype = float)
     testX = pre[:,3]

     print("testX = ",testX)
     np.savetxt("testX",testX,delimiter=',',fmt='%f')

     Data3 = TAD_calls(testX,20)     
     np.savetxt('output/'+ args.region + 'aic.csv', Data3.aic, delimiter=',')
     np.savetxt('output/'+ args.region + 'll.csv', Data3.ll, delimiter=',')
     np.savetxt('output/'+ args.region + 'best_p.csv', Data3.best_p, delimiter=',', fmt='%d')
     np.savetxt('output/'+ args.region + 'best_d.csv', Data3.best_d, delimiter=',',fmt='%d')
     np.savetxt('output/'+ args.region + 'best_g1.csv', Data3.best_g1, delimiter=',')
     np.savetxt('output/'+ args.region + 'best_g2.csv', Data3.best_g2, delimiter=',')
     np.savetxt('output/'+ args.region + 'best_g3.csv', Data3.best_g3, delimiter=',')
     np.savetxt('output/'+ args.region + 'comprehensive.csv', Data3.comprehensive, delimiter=',')

     postp_list = np.zeros((len(Data2.output2),len(Data3.comprehensive[0])+len(Data2.output2[0])))
     postp_list[:,0:len(Data2.output2[0])] = Data2.output2
     postp_list[:,len(Data2.output2[0]):len(Data3.comprehensive[0])+len(Data2.output2[0])] = Data3.comprehensive
     np.savetxt('output/'+ args.region + '_postp_list.csv', postp_list, delimiter='\t',fmt='%.10f')
     print("printing output/" + args.region + "_postp_list.csv")


if __name__ == "__main__":
     main()


     
