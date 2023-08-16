def mixGaussFit(data, nmix, varargin):

     # Fit a mixture of Gaussians via MLE/MAP (using EM)
     #
     #
     # Inputs
     #
     # data     - data(i, :) is the ith case, i.e. data is of size n-by-d
     # nmix     - the number of mixture components to use
     #
     # This file is from pmtk3.googlecode.com

     initParams = np.array([])
     prior = np.array([])
     mixPrior = np.array([])

     n = len(data)
     d = len(data[0])
     model =  {"typei":"mixGauss"}
     model["nmix"] = nmix
     model["d"] = d
     model = setMixPrior(model,mixPrior)

