import ctypes
import os
import numpy as np
import time

#taken from:
#http://pyopengl.sourceforge.net/pydoc/numpy.ctypeslib.html
#http://stackoverflow.com/questions/22425921/pass-a-2d-numpy-array-to-c-using-ctypes/27737099


_mod = ctypes.cdll.LoadLibrary('./_fwdback2.so') 
array_1d_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')
array_2d_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=2, flags='CONTIGUOUS') #I have trouble with array_2d_double,_doublepp instead
_doublepp = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C') 


_mexFunction = _mod.mexFunction
_mexFunction.restype = ctypes.c_double
_mexFunction.argtypes = [_doublepp, _doublepp,_doublepp,_doublepp,_doublepp,array_1d_double,_doublepp,_doublepp,_doublepp,_doublepp,ctypes.c_int,ctypes.c_int,ctypes.c_int]

#start_time = time.time()
#fwdback2.c seems to not work for M =1, will use hmmFwdBack.c instead for M=1 
def MF(initdist,transmat,softev,M,mixmat = np.array([]),B2 = np.array([])):
     start_time = time.time()
  
     out_alpha = np.zeros_like(softev)
     app = (out_alpha.__array_interface__['data'][0]
      + np.arange(out_alpha.shape[0])*out_alpha.strides[0]).astype(np.uintp)
     out_beta = np.zeros_like(softev) 
     bpp = (out_beta.__array_interface__['data'][0]
      + np.arange(out_beta.shape[0])*out_beta.strides[0]).astype(np.uintp)
     out_gamma = np.zeros_like(softev) 
     gpp = (out_gamma.__array_interface__['data'][0]
      + np.arange(out_gamma.shape[0])*out_gamma.strides[0]).astype(np.uintp)
   
     transmatpp = (transmat.__array_interface__['data'][0]
      + np.arange(transmat.shape[0])*transmat.strides[0]).astype(np.uintp)

     SEpp = (softev.__array_interface__['data'][0]
      + np.arange(softev.shape[0])*softev.strides[0]).astype(np.uintp)

     out_xi_summed = np.zeros((len(transmat),len(transmat)))
     xipp= (out_xi_summed.__array_interface__['data'][0]
      + np.arange(out_xi_summed.shape[0])*out_xi_summed.strides[0]).astype(np.uintp)
     inst = 1
     
     #CONVERTING 3D TO 2D, do not know how to pass in 3D array yet
     if len(B2) == 0: 
         inst = 0 
         B2 = np.zeros((len(transmat),len(softev[0])*M))
         shapeB2 = B2.shape
         B2pp = (B2.__array_interface__['data'][0]
           + np.arange(B2.shape[0])*B2.strides[0]).astype(np.uintp)
         out_eta = np.zeros_like(B2)
         epp = (out_eta.__array_interface__['data'][0]
           + np.arange(out_eta.shape[0])*out_eta.strides[0]).astype(np.uintp)
         mixmat = np.zeros((len(transmat),len(transmat)*M))
         mixmatpp = (mixmat.__array_interface__['data'][0]
           + np.arange(mixmat.shape[0])*mixmat.strides[0]).astype(np.uintp)
         
     else:
         shapeB2 = B2.shape
         B2 = np.reshape(B2,(shapeB2[0],shapeB2[1]*shapeB2[2]))
         B2pp = (B2.__array_interface__['data'][0]
           + np.arange(B2.shape[0])*B2.strides[0]).astype(np.uintp)
         out_eta = np.zeros_like(B2)
         epp = (out_eta.__array_interface__['data'][0]
           + np.arange(out_eta.shape[0])*out_eta.strides[0]).astype(np.uintp)
         mixmatpp = (mixmat.__array_interface__['data'][0]
           + np.arange(mixmat.shape[0])*mixmat.strides[0]).astype(np.uintp)

     logp = _mexFunction(app,bpp,gpp,epp,xipp,initdist,transmatpp,SEpp,B2pp,mixmatpp,len(transmat),len(softev[0]),M)
     out_eta = np.reshape(out_eta,shapeB2)
     
     print("--- %s seconds ---" % (time.time() - start_time))
     if bool(inst):
          return out_alpha,out_beta,out_gamma,logp,out_eta,out_xi_summed
     else:
          return out_alpha,out_beta,out_gamma,logp,out_xi_summed

def main():

     prior1 = np.loadtxt("testfiles/priormfb.csv",delimiter = ',')
     transmat1 = np.loadtxt("testfiles/transmatmfb.csv",delimiter = ',')
     B = np.loadtxt("testfiles/Bmfb.csv",delimiter = ',')
     b2 = np.loadtxt("testfiles/B2mfb.csv",delimiter = ',')
     mixmat_val = np.loadtxt("testfiles/mixmatmfb.csv",delimiter = ',')
     b2 = np.reshape(b2,(3,2,62707),order = 'F')

     oa,ob,og,lp,oe,oxs = MF(prior1,transmat1,B,2,mixmat=mixmat_val,B2=b2); 

     print("oa = ",oa)
     print("ob = ",ob)
     print("og = ",og)
     print("lp = ",lp)
     print("oe = ",oe)
     print("oxs = ",oxs)

     print("oe[:,:, 0] = ",oe[:,:, 0])
     print("oe[:,:, 1] = ",oe[:,:, 1])
     print("oe[:,:, 2] = ",oe[:,:, 2])
     print("oe[:,:, 3] = ",oe[:,:, 3])

     #Not using _fwdback2c.so (fwdback2c.c) for M = 1
     prior1 = np.loadtxt("testfiles/priorfb.csv",delimiter = ',')

     transmat1 = np.loadtxt("testfiles/transmatfb.csv",delimiter = ',')

     B = np.loadtxt("testfiles/Bfb.csv",delimiter = ',')

     oa,ob,og,lp,xs = MF(prior1,transmat1,B,1);

     print("oa = ",oa)
     print("ob = ",ob)
     print("og = ",og)
     print("lp = ",lp)
     print("xs = ",xs)

     #M = 3
     prior1 = np.loadtxt("testfiles/prior_3mfb.csv",delimiter = ',')
     transmat1 = np.loadtxt("testfiles/transmat_3mfb.csv",delimiter = ',')
     B = np.loadtxt("testfiles/B_3mfb.csv",delimiter = ',')
     b2 = np.loadtxt("testfiles/B2_3mfb.csv",delimiter = ',')
     mixmat_val = np.loadtxt("testfiles/mixmat_3mfb.csv",delimiter = ',')
     b2 = np.reshape(b2,(3,3,62707),order = 'F')
      
     oa,ob,og,lp,oe,oxs = MF(prior1,transmat1,B,3,mixmat=mixmat_val,B2=b2);

     print("oa = ",oa)
     print("ob = ",ob)
     print("og = ",og)
     print("lp = ",lp)
     print("oe = ",oe)
     print("oxs = ",oxs)

     print("oe[:,:, 0] = ",oe[:,:, 0])
     print("oe[:,:, 1] = ",oe[:,:, 1])
     print("oe[:,:, 2] = ",oe[:,:, 2])
     print("oe[:,:, 3] = ",oe[:,:, 3])

     #M=3 results SHOULD BE:
     alpha = np.loadtxt("testfiles/alpha_3mfb.csv",delimiter = ',')
     beta = np.loadtxt("testfiles/beta_3mfb.csv",delimiter = ',')
     gamma = np.loadtxt("testfiles/gamma_3mfb.csv",delimiter = ',')
     current_loglik = np.loadtxt("testfiles/current_loglik_3mfb.csv",delimiter = ',')
     xi_summed = np.loadtxt("testfiles/xi_summed_3mfb.csv",delimiter = ',')
     gamma2 = np.loadtxt("testfiles/gamma2_3mfb.csv",delimiter = ',')

if __name__ == "__main__":
     main()

    

