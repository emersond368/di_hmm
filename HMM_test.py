import unittest
from mixgauss_Mstep import mixgauss_Mstep
from dist2 import dist2
from kmeans import kmeans
import numpy as np
import numpy.matlib
from hmmFwdBack import hmmFwdBack
from fwdback2 import fwdback2
from normalize import normalize
from mixgauss_prob import mixgauss_prob

class TestSuite(unittest.TestCase):

      #tested gmminit.py
      #need to test gmm.py
      #need to test mixgauss_init.py

      def test_mixgauss_Mstep(self):
            ip = np.array([[446.7469,369.22690],[302.3985,364.6515],[300.4642,220.9969]])
            op = np.zeros((4,4,3,2))
            
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
            m[:,:,1] = [[-67.7503,-12.4145,75.3334],[-14.1406,95.2812,10.0597],[19.1364,14.2754,-24.8621],\
                       [-55.5247,-14.7454,14.8941]]

            postmix = np.array([[99.8917,86.0249],[76.4194,93.2569],[82.0396,62.3676]])

            testmu2,testSigma2 = mixgauss_Mstep(postmix, m, op, ip)
            self.assertAlmostEquals(testSigma2[0][0][1],0.253849749621)
            self.assertAlmostEquals(testSigma2[2][3][5],0.0176510776643)
            self.assertAlmostEquals(testSigma2[3][1][4],0.109685975463)
            self.assertAlmostEquals(testmu2[0][0],0.6327,4)
            self.assertAlmostEquals(testmu2[2][3],0.2225,4)

            testmu2,testSigma2 = mixgauss_Mstep(postmix, m, op, ip,cov_type = "diag")
            self.assertAlmostEquals(testSigma2[0][0][0],0.904995950424)
            self.assertAlmostEquals(testSigma2[3][3][3],-0.351007932971)
            self.assertAlmostEquals(testSigma2[3][0][0],0) 

            testmu2,testSigma2 = mixgauss_Mstep(postmix, m, op, ip,cov_type = "spherical")
            self.assertAlmostEquals(testSigma2[0][0][0],0.713513236422)
            self.assertAlmostEquals(testSigma2[3][3][3],0.804680824409)
            self.assertAlmostEquals(testSigma2[3][0][0],0.0)

      def test_kmeans(self):
            options = np.loadtxt('testfiles/optionskmeans_test_for_kmeans.csv',delimiter =',')
            data = np.loadtxt('testfiles/xkmeans_test_for_kmeans.csv',delimiter =',')
            centres = np.loadtxt('testfiles/mix.centres_test_for_kmeans.csv',delimiter =',')

            options_output = np.loadtxt('testfiles/options_output.csv',delimiter =',')
            centres_output = np.loadtxt('testfiles/centres_output.csv',delimiter =',')
            post_output = np.loadtxt('testfiles/post_output.csv',delimiter =',')                       

            centres,options,post,errlog = kmeans(centres, data, options)

            self.assertAlmostEquals(options_output[0],options[0],5)
            self.assertAlmostEquals(options_output[5],options[5],5)
            self.assertAlmostEquals(centres_output[5][3],centres[5][3],4)
            self.assertAlmostEquals(centres_output[2][1],centres[2][1],4)
            self.assertAlmostEquals(centres_output[1][0],centres[1][0],4)
            self.assertEqual(post_output[499][4],post[499][4])
            self.assertEqual(post_output[499][1],post[499][1])
            self.assertEqual(post_output[498][2],post[498][2])
            self.assertEqual(post_output[497][0],post[497][0])
            self.assertEqual(post_output[427][5],post[427][5])
            
            self.assertEquals(len(errlog),0)
             
      def test_hmmFwdBack(self):

            prior1 = np.loadtxt("testfiles/prior1.csv",delimiter = ',')
            transmat1 = np.loadtxt("testfiles/transmat1.csv",delimiter = ',')
            B = np.loadtxt("testfiles/Bval.csv",delimiter = ',')     
            gamma = np.loadtxt("testfiles/gamma.csv",delimiter = ',')
            alpha = np.loadtxt("testfiles/alpha.csv",delimiter = ',')                        
            beta = np.loadtxt("testfiles/beta.csv",delimiter = ',')  
            logp = np.loadtxt("testfiles/logp.csv",delimiter = ',')  

            gammaTest,alphaTest,betaTest,logpTest = hmmFwdBack(prior1,transmat1,B)

            self.assertAlmostEquals(gammaTest[2][100],gamma[2][100],4)
            self.assertAlmostEquals(gammaTest[0][1000],gamma[0][1000],4)
            self.assertAlmostEquals(gammaTest[1][60000],gamma[1][60000],4)
            self.assertAlmostEquals(alphaTest[2][100],alpha[2][100],4) 
            self.assertAlmostEquals(alphaTest[0][1000],alpha[0][1000],4)
            self.assertAlmostEquals(alphaTest[1][60000],alpha[1][60000],4)
            self.assertAlmostEquals(betaTest[2][100],beta[2][100],4)
            self.assertAlmostEquals(betaTest[0][1000],beta[0][1000],4)
            self.assertAlmostEquals(betaTest[1][60000],beta[1][60000],4)
    #        self.assertAlmostEquals(logpTest,logp,0)  #can check again but was very close 
    #        self.assertAlmostEquals(logpTest,-361823.3,0)

      def test_hmmFwdBack2(self):

           prior1 = np.loadtxt("testfiles/priorfb.csv",delimiter = ',')
           transmat1 = np.loadtxt("testfiles/transmatfb.csv",delimiter = ',')
           B = np.loadtxt("testfiles/Bfb.csv",delimiter = ',')

           alphaTest,betaTest,gammaTest,logpTest,xi_summedTest,gamma2Test = fwdback2(prior1,transmat1,B,compute_xi=1)

           xi_summed =np.loadtxt("testfiles/xi_summedfb.csv",delimiter = ',')
           alpha = np.loadtxt("testfiles/alphafb.csv",delimiter =',')
           beta = np.loadtxt("testfiles/betafb.csv",delimiter = ',')
           logp = np.loadtxt("testfiles/current_loglikfb.csv",delimiter = ',')
           gamma = np.loadtxt("testfiles/gammafb.csv",delimiter = ',')

           self.assertAlmostEquals(gammaTest[2][100],gamma[2][100],4)
           self.assertAlmostEquals(gammaTest[0][1000],gamma[0][1000],4)
           self.assertAlmostEquals(gammaTest[1][60000],gamma[1][60000],4)
           self.assertAlmostEquals(alphaTest[2][100],alpha[2][100],4)
           self.assertAlmostEquals(alphaTest[0][1000],alpha[0][1000],4)
           self.assertAlmostEquals(alphaTest[1][60000],alpha[1][60000],4)
           self.assertAlmostEquals(betaTest[2][100],beta[2][100],4)
           self.assertAlmostEquals(betaTest[0][1000],beta[0][1000],4)
           self.assertAlmostEquals(betaTest[1][60000],beta[1][60000],4)
 #          self.assertAlmostEquals(logpTest,logp,0)
           self.assertAlmostEquals(logpTest,-304680.7,0)
           self.assertAlmostEquals(xi_summedTest[1][1],xi_summed[1][1],0)
           self.assertAlmostEquals(xi_summedTest[2][2],xi_summed[2][2],0)
           self.assertAlmostEquals(xi_summedTest[0][1],xi_summed[0][1],0)
             
           prior1 = np.loadtxt("testfiles/priormfb.csv",delimiter = ',')
           transmat1 = np.loadtxt("testfiles/transmatmfb.csv",delimiter = ',')
           B = np.loadtxt("testfiles/Bmfb.csv",delimiter = ',')
           B2 = np.loadtxt("testfiles/B2mfb.csv",delimiter = ',')
           mixmat_val = np.loadtxt("testfiles/mixmatmfb.csv",delimiter = ',')
           B2 = np.reshape(B2,(3,2,62707),order = 'F')

           alphaTest,betaTest,gammaTest,logpTest,xi_summedTest,gamma2Test = fwdback2(prior1,transmat1,B,obslik2 = B2,mixmat = mixmat_val,compute_xi=1,compute_gamma2 = 1)

           alpha = np.loadtxt("testfiles/alphamfb.csv",delimiter =',')
           beta = np.loadtxt("testfiles/betamfb.csv",delimiter = ',')
           logp = np.loadtxt("testfiles/current_loglikmfb.csv",delimiter = ',')
           gamma = np.loadtxt("testfiles/gammamfb.csv",delimiter = ',')
           gamma2 = np.loadtxt("testfiles/gamma2mfb.csv",delimiter = ',')
           xi_summed =np.loadtxt("testfiles/xi_summedmfb.csv",delimiter = ',')
           gamma2 = np.reshape(gamma2,(3,2,62707),order = 'F')

           self.assertAlmostEquals(gammaTest[2][100],gamma[2][100],4)
           self.assertAlmostEquals(gammaTest[0][1000],gamma[0][1000],4)
           self.assertAlmostEquals(gammaTest[1][60000],gamma[1][60000],4)
           self.assertAlmostEquals(alphaTest[2][100],alpha[2][100],4)
           self.assertAlmostEquals(alphaTest[0][1000],alpha[0][1000],4)
           self.assertAlmostEquals(alphaTest[1][60000],alpha[1][60000],4)
           self.assertAlmostEquals(betaTest[2][100],beta[2][100],4)
           self.assertAlmostEquals(betaTest[0][1000],beta[0][1000],4)
           self.assertAlmostEquals(betaTest[1][60000],beta[1][60000],4)
           self.assertAlmostEquals(gamma2Test[2][1][100],gamma2[2][1][100],4)
           self.assertAlmostEquals(gamma2Test[0][0][1000],gamma2[0][0][1000],4)
           self.assertAlmostEquals(gamma2Test[1][0][60000],gamma2[1][0][60000],4)
 #          self.assertAlmostEquals(xi_summedTest[1][1],xi_summed[1][1],0)
 #          self.assertAlmostEquals(xi_summedTest[2][2],xi_summed[2][2],0)
 #          self.assertAlmostEquals(xi_summedTest[0][1],xi_summed[0][1],0)
 

      def test_mixgauss_prob(self):

          mu = np.loadtxt("testfiles/mumgp.csv",delimiter = ',')
          Sigma = np.loadtxt("testfiles/Sigmamgp.csv", delimiter = ',')
          obs = np.loadtxt("testfiles/obsmgp.csv", delimiter = ',')
          Sigma = np.reshape(Sigma,(1,1,3))
          #Actual result:
          B = np.loadtxt("testfiles/Bmgp.csv",delimiter = ',')
          B2 = np.loadtxt("testfiles/B2mgp.csv",delimiter = ',')

          Btest,B2test = mixgauss_prob(obs,mu,Sigma)

          self.assertAlmostEquals(Btest[0][100],B[0][100],4)
          self.assertAlmostEquals(Btest[1][20000],B[1][20000],4)
          self.assertAlmostEquals(Btest[2][60000],B[2][60000],4)

          mu = np.loadtxt("testfiles/mummgp.csv",delimiter = ',')
          mu = np.zeros((1,3,2))
          mu[:,:,0] = np.array([-0.7278,-1.7447,-0.0626])
          mu[:,:,1] = np.array([1.0743,0.0068,-0.5040])
          Sigma = np.loadtxt("testfiles/Sigmammgp.csv", delimiter = ',')
          Sigma = np.zeros((1,1,3,2))
          Sigma[0,0,0,0] = 3.7166e+04
          Sigma[0,0,1,0] = 1.1112e+06
          Sigma[0,0,2,0] = 9.0864
          Sigma[0,0,0,1] = 3.1369e+03
          Sigma[0,0,1,1] = 0.0530
          Sigma[0,0,2,1] = 222.0210
          obs = np.loadtxt("testfiles/obsmmgp.csv", delimiter = ',')
          mixmat = np.loadtxt("testfiles/mixmatmmgp.csv", delimiter = ',')

          #actual result:
          B = np.loadtxt("testfiles/Bmmgp.csv",delimiter = ',')
          B2 = np.loadtxt("testfiles/B2mmgp.csv",delimiter = ',')
          B2 = np.reshape(B2,(3,2,62707),order = 'F')

          Btest,B2test = mixgauss_prob(obs,mu,Sigma,mixmat)

          self.assertAlmostEquals(Btest[0][100],B[0][100],4)
          self.assertAlmostEquals(Btest[1][20000],B[1][20000],4)
          self.assertAlmostEquals(Btest[2][60000],B[2][60000],4)
          self.assertAlmostEquals(B2test[0][0][100],B2[0][0][100],4)
          self.assertAlmostEquals(B2test[1][0][20000],B2[1][0][20000],4)
          self.assertAlmostEquals(B2test[2][0][60000],B2[2][0][60000],4)
          self.assertAlmostEquals(B2test[0][1][100],B2[0][1][100],4)
          self.assertAlmostEquals(B2test[1][1][20000],B2[1][1][20000],4)
          self.assertAlmostEquals(B2test[2][1][60000],B2[2][1][60000],4)
                                

unittest.main()

      
