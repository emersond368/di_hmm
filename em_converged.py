import numpy as np

def em_converged(loglik,previous_loglik,threshold=1e-4,check_increased=1):

     # EM_CONVERGED Has EM converged?
     # [converged, decrease] = em_converged(loglik, previous_loglik, threshold)
     #
     # We have converged if the slope of the log-likelihood function falls below 'threshold', 
     # i.e., |f(t) - f(t-1)| / avg < threshold,
     # where avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log lik at iteration t.
     # 'threshold' defaults to 1e-4.
     #
     # This stopping criterion is from Numerical Recipes in C p423
     #
     # If we are doing MAP estimation (using priors), the likelihood can decrase,
     # even though the mode of the posterior is increasing.

     converged = 0
     decrease = 0

     if bool(check_increased):
          if (loglik - previous_loglik < -1e-3): #allow for some imprecision
               print("likelihood decreased from ", previous_loglik, " to ", loglik,"!!!")
               decrease = 1
               converged = 0
               return converged, decrease
   
     print("loglik = " ,loglik)
     print("previous_loglik = ",previous_loglik)
     delta_loglik = np.abs(loglik - previous_loglik)
     avg_loglik = (np.abs(loglik) + np.abs(previous_loglik) + np.finfo(float).eps)/2
     if (not np.isnan(delta_loglik/avg_loglik)): 
          if (delta_loglik/avg_loglik < threshold):
               converged = 1

     return converged,decrease 
