def gmm(dim,ncentres,covar_type, ppca_dim=1):

     #GMM	Creates a Gaussian mixture model with specified architecture.
     #
     #   Description
     #	 MIX = GMM(DIM, NCENTRES, COVARTYPE) takes the dimension of the space
     #	DIM, the number of centres in the mixture model and the type of the
     #	mixture model, and returns a data structure MIX. The mixture model
     #	type defines the covariance structure of each component  Gaussian:
     #	  'spherical' = single variance parameter for each component: stored as a vector
     #	  'diag' = diagonal matrix for each component: stored as rows of a matrix
     #	  'full' = full matrix for each component: stored as 3d array
     #	  'ppca' = probabilistic PCA: stored as principal components (in a 3d array
     #	    and associated variances and off-subspace noise
     #	 MIX = GMM(DIM, NCENTRES, COVARTYPE, PPCA_DIM) also sets the
     #	dimension of the PPCA sub-spaces: the default value is one.
     #
     #	The priors are initialised to equal values summing to one, and the
     #	covariances are all the identity matrix (or equivalent).  The centres
     #	are initialised randomly from a zero mean unit variance Gaussian.
     #	This makes use of the MATLAB function RANDN and so the seed for the
     #	random weight initialisation can be set using RANDN('STATE', S) where
     #	S is the state value.
     #
     #	The fields in MIX are
     #	  
     #	  type = 'gmm'
     #	  nin = the dimension of the space
     #	  ncentres = number of mixture components
     #	  covartype = string for type of variance model
     #	  priors = mixing coefficients
     #	  centres = means of Gaussians: stored as rows of a matrix
     #	  covars = covariances of Gaussians
     #	 The additional fields for mixtures of PPCA are
     #	  U = principal component subspaces
     #	  lambda = in-space covariances: stored as rows of a matrix
     #	 The off-subspace noise is stored in COVARS.
     #
     #	See also
     #	GMMPAK, GMMUNPAK, GMMSAMP, GMMINIT, GMMEM, GMMACTIV, GMMPOST, 
     #	GMMPROB
     #

     #	Copyright (c) Ian T Nabney (1996-2001)

     try:
          if ncentres < 1:
              raise ValueError
     except ValueError:
          print "Number of centres must be greater than zero"

     mix["typei"] = "gmm"
     mix["nin"] = dim
     mix["ncentres"] = ncentres

     vartypes = ['spherical','diag','full','ppca']

     try:
          if covar_type not in vartypes:
               raise ValueError
          else:
               mix["covar_type"] = covar_type
     except ValueError:
          print "Undefined covariance type"

     #Make default dimension of PPCA subspaces one
     if covar_type == 'ppca':
          try:
              if ppca_dim > dim:
                   raise ValueError
          except ValueError: 
              print "Dimension of PPCA subspaces must be less than data."

          mix["ppca_dim"] = ppca_dim

     # Initialise priors to be equal and summing to one
     mix["priors"] = np.ones(mix["ncentres"])/mix["ncentres"]

     #Initialise centres
     mix["centres"] = np.random.randn(mix["ncentres"], mix["nin"])
     
     # Initialize all the variances to unity
     
     if mix["covar_type"] == 'spherical':
          mix["covars"] = np.ones(mix["ncentres"])
          mix["nwts"] = mix["ncentres"] + mix["ncentres"]*mix["nin"] + mix["ncentres"]
     elif mix["covar_type"] == 'diag':
          # Store diagonals of covariance matrices as rows in a matrix
          mix["covars"] = np.ones((mix["ncentres"],mix["nin"]))
          mix["nwts"] = mix["ncentres"] + mix["ncentres"]*mix["nin"] + mix["ncnetres"]*mix["nin"]
     elif mix["covar_type"] == 'full':
          # Store covariance matrices in a row vector of matrices
          mix["covars"] = np.zeros((mix["nin"],mix["nin"],mix["ncentres"]))
          for i in range(0,mix["ncentres"]):
               mix["covars"][:,:,i] = np.eye(mix["nin"])
          mix["nwts"] = mix["ncentres"] + mix["ncentres"]*mix["nin"] + mix["ncentres"]*mix["nin"]*mix["nin"]
     elif mix["covar_type"] == 'ppca':
          #This is the off-subspace noise: make it smaller than lambdas
          mix["covars"] = 0.1*np.ones(mix["ncentres"])
          # Also set aside storage for principal components and associated variances
          init_space = np.eye(mix["nin"])
          init_space = init_space[:,0:mix["ppca_dim"]]
          init_space[mix["ppca_dim"]:mix["nin"],:] = np.ones[mix["nin"] - mix["ppca_dim"],mix["ppca_dim"]]
          mix["U"] = np.zeros((mix["nin"],mix["nin"],mix["ncentres"]))
          for i in range(0,mix["ncentres"]):
               mix["U"][:,:,i] = init_space
          mix["lambda"] = np.ones((mix["ncentres"],mix["ppca_dim"])
          #Take account of additional parameters
          mix["nwts"] = mix["ncentres"] + mix["ncentres"]*mix["nin"] + mix["ncentres"]\
                        + mix["ncentres"]*mix["ppca_dim"]+mix["ncentres"]*mix["nin"]*mix["ppca_dim"]
     else:
          try:
               raise ValueError
          except ValueError:
               print "Unknown covariance type ",mix["covar_type"]
     
     return mix 
