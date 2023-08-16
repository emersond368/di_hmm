import numpy as np
from dist2 import dist2

def kmeans(centres, data, options,store = 0):

#     KMEANS	Trains a k means cluster model.
#
#	Description
#	 CENTRES = KMEANS(CENTRES, DATA, OPTIONS) uses the batch K-means
#	algorithm to set the centres of a cluster model. The matrix DATA
#	represents the data which is being clustered, with each row
#	corresponding to a vector. The sum of squares error function is used.
#	The point at which a local minimum is achieved is returned as
#	CENTRES.  The error value at that point is returned in OPTIONS(8).
#       OPTIONS is `1x14 array
#
#	[CENTRES, OPTIONS, POST, ERRLOG] = KMEANS(CENTRES, DATA, OPTIONS)
#	also returns the cluster number (in a one-of-N encoding) for each
#	data point in POST and a log of the error values after each cycle in
#	ERRLOG.    The optional parameters have the following
#	interpretations.
#
#	OPTIONS[0] is set to 1 to display error values; also logs error
#	values in the return argument ERRLOG. If OPTIONS(1) is set to 0, then
#	only warning messages are displayed.  If OPTIONS(1) is -1, then
#	nothing is displayed.
#
#	OPTIONS[1] is a measure of the absolute precision required for the
#	value of CENTRES at the solution.  If the absolute difference between
#	the values of CENTRES between two successive steps is less than
#	OPTIONS(2), then this condition is satisfied.
#
#	OPTIONS[2] is a measure of the precision required of the error
#	function at the solution.  If the absolute difference between the
#	error functions between two successive steps is less than OPTIONS(3),
#	then this condition is satisfied. Both this and the previous
#	condition must be satisfied for termination.
#
#	OPTIONS[13] is the maximum number of iterations; default 100.
#
#	See also
#	GMMINIT, GMMEM
#

#	Copyright (c) Ian T Nabney (1996-2001)
        
     (ndata, data_dim) = data.shape
     (ncentres,dim) = centres.shape
     errlog = np.array([])

     try:
         if int(dim) == int(data_dim):
              print dim, data_dim
         else:
              raise ValueError
     except ValueError:
         print "Data dimension does not match dimension of centres"

     try:
         if ncentres <= ndata:
              print ncentres, ndata
         else:
              raise ValueError
     except ValueError:
         print "More centres than data"

     if bool(options[13]):
         nliters  = options[13]
     else:
         nliters = 99

     if store != 0:
         store = 1
         errlog = np.zeros(nliters)

     # Check if centres and posteriors need to be initialised from data
     #print "options[4] = ",options[4]

     if bool(options[4] == 1):
         #Do the initialisation
         perm = np.random.permutation(ndata)
         perm = perm[0:ncentres]
         perm = np.array([138, 241, 384, 90, 203, 11]) # testing, turnon for HMM_test.py
         #Assign first ncentres (permuted) data points as centres
         centres = data[perm,:]

     
     id = np.eye(ncentres)

     for n in range(0,int(nliters)):
         
         #Save old centres to check for termination
         old_centres = centres
         
         #calculate posteriors based on existing centres
         d2 = dist2(data,centres)
    #     if n == 0:
    #          print "d2 = ",d2
    #          print "old_centres = ",old_centres
 
         #Assign each point to nearest centre
         minvals = np.amin(d2.T,axis = 0)
         indices = np.zeros(len(minvals))
         indices = np.array(indices, dtype = int)
         for i in range(0,len(minvals)):
              indices[i] = np.where(d2[i,:] == minvals[i])[0] # store only col
  
         post = id[indices,:]
         num_points = np.sum(post, axis = 0)

         # Adjust the centres based on new posteriors
         for j in range(0,ncentres):
              if (num_points[j] > 0):
                   centres[j,:] = np.sum(data[np.where(post[:,j] == 1),:][0],axis = 0)/num_points[j] #extra dim added

         # Error value is total squared distance from cluster centres
         e = np.sum(minvals)
         if bool(store):
              errlog[n] = e

         if options[0] > 0:
              print "Cycle ",n," Error ",e
         
         if n>0:
              # Test for termination
              if np.max(np.abs(centres-old_centres)) < options[1] and np.abs(old_e - e) < options[2]:
                   options[7] = e
                   return centres,options,post,errlog
         
         old_e = e

     # If we get here, then we haven't terminated in the given number of 
     # iterations.    
     options[7] = e
     if options[0] >= 0:
        print "Maximum number of iterations has been exceeded"

     return centres,options,post,errlog
         

     
     
def main():

     data = np.loadtxt('testfiles/xkmeans.csv',delimiter =',')
     
     options = np.loadtxt('testfiles/optionskmeans.csv',delimiter =',')
     
     centres = np.loadtxt('testfiles/mix.centres.csv',delimiter = ',')

     centres,options,post,errlog = kmeans(centres, data, options)
     print "entres = ", centres
     print "options = ", options
     print "post = ",post
     print "errlog = ",errlog   

if __name__ == "__main__":
     main()
