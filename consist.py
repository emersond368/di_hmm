import numpy as np

def consist(model, typei, *oinputs):

     #CONSIST Check that arguments are consistent.
     #
     #	Description
     #
     #	ERRSTRING = CONSIST(NET, TYPEI, INPUTS) takes a network data structure
     #	NET together with a string TYPEI containing the correct network type,
     #	a matrix INPUTS of input vectors and checks that the data structure
     #	is consistent with the other arguments.  An empty string is returned
     #	if there is no error, otherwise the string contains the relevant
     #	error message.  If the TYPEI string is empty, then any type of network
     # 	is allowed.
     #
     #	ERRSTRING = CONSIST(NET, TYPEI) takes a network data structure NET
     #	together with a string TYPEI containing the correct  network type, and
     #	checks that the two types match.
     #
     #	ERRSTRING = CONSIST(NET, TYPEI, INPUTS, OUTPUTS) also checks that the
     #	network has the correct number of outputs, and that the number of
     #	patterns in the INPUTS and OUTPUTS is the same.  The fields in NET
     #	that are used are
     #	  typei
     #	  nin   model['nin'] is single integer value
     #	  nout  model['nout'] is single integer value
     #
     #  model['NET'] tells how many cols their should be for inputs and outputs
     #  INPUTS and OUTPUTS (*OINPUTS) array data is tested to see if it fits number
     #  specification in model['NET']    
     #
     #	See also
     #	MLPFWD
     #

     #	Copyright (c) Ian T Nabney (1996-2001)

     #Assume that all is OK as default
     errstring = ''

     # If typei string is not empty
     if len(typei) != 0:
          # First check that model has typei field
          if 'typei' not in model:
               errstring = 'Data structure does not contain typei field'
               return errstring
          # Check that model has the correct type
          s = model['typei']
          if typei != s:
               errstring = 'Model typei ' + str(s) + ' does not match expected typei ' + str(typei)
               return errstring
          
     #If inputs are present, check that they have correct dimension
     if len(oinputs) > 0:
         if 'nin' not in model:
              errstring = 'Data structure does not contain nin field'
              return errstring
         size = oinputs[0].shape
         if len(list(size)) ==1:
             data_nin = 1 #inputs col size, arbitrary - python does not distinguish row/col vector
         else:
             data_nin = size[1] #inputs col size
         if model['nin'] != data_nin: #test against model cols
             errstring = 'Dimension of inputs ' + str(data_nin) + ' does not match number of model inputs ' + str(model['nin'])
             return errstring
      
     #If outputs are present, check that they have correct dimension
     if len(oinputs) > 1:
         if 'nout' not in model:
              errstring = 'Data structure does not conatin nout field'
              return errstring
         # mdata_nout = len(oinputs[1][0]) #outputs col size
         size = oinputs[1].shape
         if len(list(size)) == 1:
             data_nout = 1 #outputs col size,arbitrary - python does not distinguish row/col vector
         else:
             data_nout = size[1] #outputs col size
         if model['nout'] != data_nout:  #how many columns in model structure, single value
              errstring = 'Dimension of outputs ' + str(data_nout) + '  does not match number of model outputs ' + str(model['nout'])
              return errstring

         # Also check that number of data points in inputs and outputs is the same
         num_in = len(oinputs[0]) #inputs row size
         num_out = len(oinputs[1]) #outputs col size
         if num_in != num_out:
              errstring = 'Number of input patterns ' + str(num_in) + ' does not match number of output patterns ' + str(num_out)
              return errstring

def main():

     model = {'nin' : 9}  #input must have 8 rows, 9 cols
     model['nout'] = 9  #output must have 8 rows, 9 cols
     model['typei'] = 'gmm'  

     inputs = np.array([8,9,3,4]) #input array of 4 values
     outputs = np.array([9]) #output array of 1 value

     typei = np.array([9.2])
     errstring = consist(model, 'gmm', inputs, outputs)

     print errstring

     typei = 'test'
     errstring = consist(model, typei, inputs, outputs)

     print errstring

     inputs = np.ones((8,9))
     outputs = np.zeros((8,9))

     errstring = consist(model, 'gmm', inputs, outputs)
    
     print "errstring = ",errstring
     print "errstring == None",errstring == None
     

if __name__ == "__main__":
     main()

