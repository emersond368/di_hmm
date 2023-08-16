import numpy as np
from consist import consist
from kmeans import kmeans
from dist2 import dist2
from ppca import ppca

def gmminit(mix,x,options):

     #GMMINIT Initialises Gaussian mixture model from data
     #
     #	Description
     #	MIX = GMMINIT(MIX, X, OPTIONS) uses a dataset X to initialise the
     #	parameters of a Gaussian mixture model defined by the data structure
     #	MIX.  The k-means algorithm is used to determine the centres. The
     #	priors are computed from the proportion of examples belonging to each
     #	cluster. The covariance matrices are calculated as the sample
     #	covariance of the points associated with (i.e. closest to) the
     #	corresponding centres. For a mixture of PPCA model, the PPCA
     #	decomposition is calculated for the points closest to a given centre.
     #	This initialisation can be used as the starting point for training
     #	the model using the EM algorithm.
     #
     #	See also
     #	GMM
     #

     #	Copyright (c) Ian T Nabney (1996-2001)

     ndata,xdim = x.shape
     
     # Check that inputs are consistent
     errstring = consist(mix,'gmm',x)

     try:
          if errstring != None:
               raise ValueError
     except ValueError:
          print errstring
 
     # Arbitrary width used if variance collapses to zero: make it 'large' so
     # that centre is responsible for a reasonable number of points.

     GMM_WIDTH = 1.0

     # Use kmeans algorithm to set centres
     options[4] = 1

     mix["centres"],options,post,errlog = kmeans(mix["centres"],x,options)
     #Set priors depending on number of points in each cluster
     cluster_sizes = np.sum(post,axis = 0) 
     cluster_sizes[cluster_sizes == 0] = 1 #Make sure that no prior is zero
    # print "cluster_sizes = ",cluster_sizes
     mix["priors"] = cluster_sizes/np.sum(cluster_sizes) #Normalise priors

     if mix["covar_type"] == "spherical":  #tested
          if mix["ncentres"] > 1:
               # Determine widths as distance to nearest centre
               # (or a constant if this is zero)
               cdist = dist2(mix["centres"],mix["centres"])
               cdist = cdist + np.eye(mix["ncentres"])* np.sys.float_info.max
               mix["covars"] = np.min(cdist,axis = 0)
               inter = mix["covars"] <  np.finfo(float).eps
               mix["covars"] = mix["covars"] + GMM_WIDTH*np.array([int(x) for x in inter])
          else:
               #Just use variance of all data points averaged over all dimensions
               mix["covars"] = np.mean(np.diag(np.cov(x.T))) #x.T used cov size = col# X col#
     elif mix["covar_type"] == "diag": #tested
          for j in range(0,mix["ncentres"]):
               #Pick out data points belonging to this centre
               c = x[np.nonzero(post[:,j])[0],:]
               diffs = c - np.outer(np.ones(len(c)),mix["centres"][j,:])
               mix["covars"][j,:] = np.sum(diffs*diffs,axis = 0)/len(c)
               #Replace small entries by GMM_WIDTH value
               mix["covars"][j,:] = mix["covars"][j,:] + GMM_WIDTH*(mix["covars"][j,:] < np.finfo(float).eps)
     elif mix['covar_type'] == "full": #tested
          for j in range(0,mix["ncentres"]):
               # Pick out data points belonging to this centre
               c = x[np.nonzero(post[:,j])[0],:]
               diffs = c - np.outer(np.ones(len(c)),mix["centres"][j,:])
               mix['covars'][:,:,j] = np.dot(diffs.T,diffs)/len(c)
               # Add GMM_WIDTH*Identity to rank-deficient covariance matrices
               if np.linalg.matrix_rank(mix['covars'][:,:,j]) < mix['nin']:
                    mix['covars'][:,:,j] = mix['covars'][:,:,j] + GMM_WIDTH*np.eye(mix['nin'])
     elif mix['covar_type'] == "ppca":
          c = x[np.nonzero(post[:,0])[0],:]
          diffs = c - np.outer(np.ones(len(c)),mix["centres"][0,:])
          tempcovars,tempU,templambda = ppca(np.dot(diffs.T,diffs)/len(c),mix['ppca_dim'])
          try:
              if len(templambda) != mix['ppca_dim']:
                   raise ValueError
              else:
                   mix['covars'][0] = tempcovars
                   mix['U'] = np.zeros((mix["nin"],len(templambda),mix['ncentres']))
                   mix['lambda'] = np.zeros((mix['ncentres'],len(templambda)))
                   mix['U'][:,:,0] = np.array(tempU)
                   mix['lambda'][0,:] = np.array(templambda)
          except ValueError:
                   print "Unable to extract enough components"

          for j in range(1,mix['ncentres']):
               #Pick out data points belonging to this centre
               c = x[np.nonzero(post[:,j])[0],:]
               diffs = c - np.outer(np.ones(len(c)),mix["centres"][j,:])
               tempcovars,tempU,templambda = ppca(np.dot(diffs.T,diffs)/len(c),mix['ppca_dim'])
               mix['U'][:,:,j] = np.array(tempU)   
               mix['lambda'][j,:] = np.array(templambda) 
     else:
          try:
                raise ValueError
          except ValueError:
               print "Unknown covariance type ", mix['covar_type']          
               
   
     return mix

def main():

     mix = {"typei":"gmm"}
     mix["nin"] = 4
     mix["ncentres"] = 6
     mix["covar_type"] = "diag"
     mix["priors"] = np.array([0.1667,0.1667,0.1667,0.1667,0.1667,0.1667])
     mix["centres"] = np.loadtxt("testfiles/mix.centres.csv",delimiter =',')
     mix["covars"] = np.loadtxt("testfiles/mix.covars.csv",delimiter =',')

     #CORRECT output
     mix2 = {"typei":"gmm"}
     mix2["nin"] = 4
     mix2["ncentres"] = 6
     mix2["covar_type"] = "diag" 
     mix2["priors"] = np.array([0.1500,0.1800,0.2240,0.1380,0.1780,0.1300])
     mix2["centres"] = np.loadtxt("testfiles/mix2.centres.csv",delimiter =',')
     mix2["covars"] = np.loadtxt("testfiles/mix2.covars.csv",delimiter =',') 
   
     data = np.loadtxt("testfiles/datagmminit.csv",delimiter = ',')
     options = np.loadtxt("testfiles/optionsgmminit.csv",delimiter = ',')

     mix2test = gmminit(mix,data.T,options)

     print 'mix2test["covars"] = ',mix2test["covars"]
     print 'mix2test["centres"] = ',mix2test["centres"]
     print 'mix2test["priors"] = ',mix2test["priors"] 

     mix3 = {"typei":"gmm"}
     mix3["nin"] = 4
     mix3["ncentres"] = 6
     mix3["covar_type"] = "full"
     mix3["priors"] = np.array([0.1667,0.1667,0.1667,0.1667,0.1667,0.1667])  
     mix3["centres"] = np.loadtxt("testfiles/mix_full.centres.csv",delimiter =',')
     mix3["covars"] = np.zeros((4,4,6))
     for i in range(0,6):
          mix3["covars"][:,:,i] = np.eye(4)

     data2 = np.loadtxt("testfiles/datagmminit_full.csv",delimiter = ',')

     # CORRECT output:
     mix4 = {"typei":"gmm"}
     mix4["nin"] = 4
     mix4["ncentres"] = 6
     mix4["covar_type"] = "full"
     mix4["priors"] = np.array([0.2080,0.2060,0.0980,0.1980,0.1580,0.1320])
     mix4["centres"] = np.loadtxt("testfiles/mix2_full.centres.csv",delimiter =',')
    # mix4["covars"] = np.loadtxt("mix2_full.covars.csv",delimiter =',')
     mix4["covars"] = np.zeros((4,4,6))
     for i in range(0,6):
          mix4["covars"][:,:,0] = np.array([[0.64956,-0.1931,-0.014211,0.0071725],\
              [-0.1931,0.69898,0.067688,0.16436],[-0.014211,0.067688,0.31711,-0.021906],\
              [0.0071725,0.16436,-0.021906,0.47755]])
          mix4["covars"][:,:,1] = np.array([[0.58214,-0.15204,-0.15567,-0.1113],\
              [-0.15204,0.47533,0.083517,0.086198],[-0.15567,0.083517,0.39626,0.095749],\
              [-0.1113,0.086198,0.095749,0.43176]])
          mix4["covars"][:,:,2] = np.array([[0.62724,0.063395,-0.06147,0.061478],\
              [0.063395,0.29681,0.11054,-0.060322],[-0.06147,0.11054,0.46969,0.041476],\
              [0.061478,-0.060322,0.041476,0.44543]])
          mix4["covars"][:,:,3] = np.array([[0.73035,-0.064659,-0.071611,-0.10047],\
              [-0.064659,0.31406,-0.034304,-0.012658],[-0.071611,-0.034304,0.50824,0.17434],\
              [-0.10047,-0.012658,0.17434,0.56001]])
          mix4["covars"][:,:,4] = np.array([[0.61152,-0.35252,-0.20144,-0.10289],\
              [-0.35252,0.71917,0.12995,0.11987],[-0.20144,0.12995,0.52881,0.099271],\
              [-0.10289,0.11987,0.099271,0.67897]])
          mix4["covars"][:,:,5] = np.array([[0.47642,0.017914,0.00432,-0.20817],\
              [0.017914,0.36542,0.13884,-0.012466],[0.00432,0.13884,0.44844,0.08632],\
              [-0.20817,-0.012466,0.08632,0.56058]])

     mix4test = gmminit(mix3,data2.T,options)

     print 'mix4test["covars"] = ',mix4test["covars"]
     print 'mix4test["centres"] = ',mix4test["centres"]
     print 'mix4test["priors"] = ',mix4test["priors"]

     mix5 = {"typei":"gmm"}
     mix5["nin"] = 4
     mix5["ncentres"] = 6
     mix5["covar_type"] = "spherical"
     mix5["priors"] = np.array([0.1667,0.1667,0.1667,0.1667,0.1667,0.1667])
     mix5["centres"] = np.loadtxt("testfiles/mix_sphere.centres.csv",delimiter =',')
     mix5["covars"] = np.loadtxt("testfiles/mix_sphere.covars.csv",delimiter =',')
     mix5["nwts"] = 36

     #Correct output:
     mix6 = {"typei":"gmm"}
     mix6["nin"] = 4
     mix6["ncentres"] = 6
     mix6["covar_type"] = "spherical"
     mix6["priors"] = np.array([0.1760,0.1760,0.1620,0.1060,0.1480,0.2320])
     mix6["centres"] = np.loadtxt("testfiles/mix2_sphere.centres.csv",delimiter =',')
     mix6["covars"] = np.loadtxt("testfiles/mix2_sphere.covars.csv",delimiter =',')
     mix6["nwts"] = 36

     data3 = np.loadtxt("testfiles/datagmminit_sphere.csv",delimiter = ',')

     mix6test = gmminit(mix5,data3.T,options)

     print 'mix6test["covars"] = ',mix6test["covars"]
     print 'mix6test["centres"] = ',mix6test["centres"]
     print 'mix6test["priors"] = ',mix6test["priors"]

     mix7 = {"typei":"gmm"}
     mix7["nin"] = 4
     mix7["ncentres"] = 6
     mix7["covar_type"] = "ppca"
     mix7["priors"] = np.array([0.1667,.1667,0.1667,0.1667,0.1667,0.1667])
     mix7["centres"] = np.loadtxt("testfiles/mix_ppca.centres.csv",delimiter = ',')
     mix7["covars"] = np.loadtxt("testfiles/mix_ppca.covars.csv",delimiter = ',')
     mix7["nwts"] = 36
     mix7["ppca_dim"] = 3

     data4 = np.loadtxt("testfiles/datagmminit_ppca.csv",delimiter = ',')

     mix8test = gmminit(mix7,data4.T,options)

     #Correct output:

     mix8 = {"typei":"gmm"}
     mix8["nin"] = 4
     mix8["ncentres"] = 6
     mix8["covar_type"] = "ppca"
     mix8["priors"] = np.array([0.1360,0.1760,0.1720,0.2340,0.1280,0.1540])
     mix8["centres"] = np.loadtxt("testfiles/mix2_ppca.centres.csv",delimiter = ',')
     mix8["covars"] = np.loadtxt("testfiles/mix2_ppca.covars.csv",delimiter = ',')
     mix8["nwts"] = 36
     mix8["ppca_dim"] = 3
     mix8["lambda"] = np.loadtxt("testfiles/mix2_ppca.lambda.csv",delimiter = ',')
     mix8["U"] = np.zeros((4,3,6))
     mix8["U"][:,:,0] = np.array([[0.5232,-0.8492,-0.0585],[-0.0188,0.0664,-0.5309],\
                        [0.1002,0.0933,-0.8363],[0.8461,0.5156,0.1234]])
     mix8["U"][:,:,1] = np.array([[0.5245,-0.6747,0.2784],[0.2740,0.5056,0.8153],\
                        [0.3852,-0.2822,-0.0270],[0.7081,0.4576,-0.5070]])
     mix8["U"][:,:,2] = np.array([[0.8293,-0.1671,0.4522],[-0.1950,0.0055,-0.2352],\
                        [-0.5178,-0.1221,0.8405],[0.0781,0.9783,0.1835]])
     mix8["U"][:,:,3] = np.array([[-0.7555,-0.5133,0.3247],[0.3342,0.2049,0.8656],\
                        [-0.5636,0.8111,0.0750],[0.0045,0.1913,-0.3738]])
     mix8["U"][:,:,4] = np.array([[0.6502,0.3769,-0.6163],[-0.1419,0.1401,0.2929],\
                        [0.7103,-0.5526,0.4324],[0.2293,0.7300,0.5894]])
     mix8["U"][:,:,5] = np.array([[-0.0372,0.4339,0.7737],[-0.3508,0.6649,-0.5782],\
                        [-0.9268,-0.1889,0.2090],[0.1287,0.5779,0.1527]])

     print 'mix8test["priors"] = ',mix8test["priors"]
     print 'mix8test["covars"] = ',mix8test["covars"]
     print 'mix8test["centres"] = ',mix8test["centres"]
     print 'mix8test["lambda"] = ',mix8test["lambda"]
     print 'mix8test["U"] = ', mix8test["U"]
     print "mix8['U'][2,2,2] = ", mix8['U'][2,2,2]
     print "mix8test['U'][2,2,2] = ", mix8test['U'][2,2,2]
     print "mix8['U'][2,2,2] = ", mix8['U'][0,0,0]
     print "mix8test['U'][2,2,2] = ", mix8test['U'][0,0,0]

if __name__ == "__main__":
     main()     
