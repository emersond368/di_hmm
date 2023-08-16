# Direction Index
# author  Daniel Emerson demerson368@gmail.com
# version 8, Sept 6, 2016
 


# The user selects placement and size of upstream and downstream total summation interaction area
#  by choosing offset from diagonal (gap size or distance away from diagonal) the range or horizontal
#  spread of interaction as well as  sliding window size along diagonal.  Using this desired size and
#  placement, a DI (directionality index) is calculated for each diagonal bin or window along diagonal
#  and is compared to a randomized matrix (off-diagonals randomized).  DI metric is placed in DI.csv and 
#  corresponding randomized DI is placed in randomDI.csv

# Illustration:
# C = diagonal, + = distance/gap (size 2), A = area of total upstream/downstream interaction (size 6)
# with sliding window size 2 and range/spread set to 3 (window moves one bin diagonal at a time).
# 
#       ----AAA++C++AAA---- 
#       -----AAA++C++AAA---



import re
import random
import copy
import csv
import argparse
import numpy as np
from MatrixMakeBanded4 import MatrixMakeBanded

class DI:

     def __init__(self,filename, distance, DIrange, window,diagonal):
          self.Heatmap = filename
          self.window = int(window) #size across diagonal
          self.DIrange = int(DIrange) #DI range
          self.distance = int(distance) #gap from diagonal
          self.diagonal = bool(diagonal) #banded matrix True/False

     def DI_metric(self): 
         # calculate the directionality Index (equation is found in Dixon et al. Nature 2012)

         self._interaction_sum()
         mean = list(map(lambda A,B: (A+B)/2, self.Aset, self.Bset))
         self.DIset = list(map(lambda E,B,A: 0 if ((A-B == 0) or (mean == 0)) else -1*((B-A)/abs(B - A))*( (pow((A - E),2)/E) + (pow((B - E),2)/E)), mean, self.Bset, self.Aset))

         print("Asummation subset",np.array(self.Aset)[10000:10008])
         print("Bsummation subset",np.array(self.Bset)[10000:10008])

         self.DIset = np.array(self.DIset)
         np.savetxt("Asettest",np.array(self.Aset))
         np.savetxt("Bsettest",np.array(self.Bset))

     def _interaction_sum(self):
         # Calculate the total sum of the DIrange of interactions for upstream (Aset) and downstream (Bset) set a certain gap away from center bin
        
         self.Aset = []  # setting to [] to avoid risk of appending multiple copies if function is called more than once
         self.Bset = []
         print("digaonal = ", self.diagonal)         
         if (self.diagonal): #Heatmap is constructed as a square matrix
              for i in range(0,len(self.Heatmap)):
                    if (i - (int(self.DIrange) + int(self.distance)) < 0) or (i + (int(self.DIrange) + int(self.distance)) > len(self.Heatmap)-1): # avoid out of range
                          continue
                    else:
                         #for j in range(1,int(self.DIrange)+1):
                         #      summation = summation + float(self.Heatmap[i][i+int(self.distance)+j])
                         #      summation2 = summation2 + float(self.Heatmap[i][i-int(self.distance)-j])
                         Asummation = np.sum(self.Heatmap[i,i+1+int(self.distance):i+int(self.DIrange)+int(self.distance)+1])
                         Bsummation = np.sum(self.Heatmap[i,i-int(self.DIrange)-int(self.distance)-1:i-1-int(self.distance)])
                    self.Aset.append(Asummation)
                    self.Bset.append(Bsummation)
         else:  #Heatmap is constructed as a banded matrix (only diagonal band saved)
              center_index = int(self.Heatmap.shape[1]/2)  #middle column
              for i in range(0,len(self.Heatmap)):
                   #if ((int(self.DIrange) + int(self.distance)) > center): # avoid out of range
                   #      continue
                   Asummation = np.sum(self.Heatmap[i,center_index+1+int(self.distance):center_index+int(self.DIrange)+int(self.distance)+1])
                   Bsummation = np.sum(self.Heatmap[i,center_index-int(self.DIrange)-int(self.distance):center_index])
                   self.Aset.append(Asummation)
                   self.Bset.append(Bsummation)
              
         if self.window > 1: #default DI only has sliding window = 1 
              change = self._window_sum(self.Aset,self.Bset)

              self.Aset = change[0]
              self.Bset = change[1]

     def _window_sum(self,A,B):
          tempAset = []
          tempBset = []
          for i in range(0,len(A)):
               if i + int(self.window)-1 > len(A)-1:  #last values do not have window (not enough space for window), sic removed
                    continue
               else:
                    temp = 0
                    temp2 = 0
                    for j in range(0,int(self.window)):
                         temp = temp + A[i+j]
                         temp2 = temp2 + B[i+j]
                    tempAset.append(temp)
                    tempBset.append(temp2)
          return (tempAset, tempBset)

     def scramble_DI_metric(self):
          # calculate randomized Directionality Index

          self._scramble_interaction_sum()
          mean = list(map(lambda A,B: (A+B)/2, self.sAset, self.sBset))
          self.sDIset = list(map(lambda E,B,A: 0 if (A-B == 0) or (mean == 0) else -1*((B-A)/abs(B - A))*( (pow((A - E),2)/E) + (pow((B - E),2)/E)), mean, self.sBset, self.sAset))

     def _scramble_interaction_sum(self):
          # same method as interaction_sum() except sum of randomized off diagonals 

          self.sAset = []  # setting to [] to avoid risk of appending multiple copies if function is called more than once 
          self.sBset = []
          center = int(float((len(self.Heatmap[0])-1)/2))  #middle column

          if (self.diagonal):
               for i in range(0,len(self.Heatmap)):
                     if (i - (int(self.DIrange) + int(self.distance)) < 0) or (i + (int(self.DIrange) + int(self.distance)) > len(self.Heatmap)-1): # avoid out of range
                           continue
                     else:
                           self.scramble = list(self.Heatmap[i])  #pick out row of heatmap
                           extract = self.scramble[i]
                           self.scramble.remove(extract)  # remove diagonal
                           random.shuffle(self.scramble)  # scramble off diagonal values
                           self.scramble.insert(i,extract) #add diagonal back in

                           summation = 0
                           summation2 = 0
                           for j in range(1,int(self.DIrange)+1):
                                summation = summation + float(self.scramble[i+int(self.distance)+j])
                                summation2 = summation2 + float(self.scramble[i-int(self.distance)-j])
                           self.sAset.append(summation)
                           self.sBset.append(summation2)
               change = self.windowing(self.sAset,self.sBset)
               self.sAset = change[0]
               self.sBset = change[1]
          else:
               for i in range(0,len(self.Heatmap)):
                    if ((self.DIrange + self.distance) > center): # avoid out of range
                         continue
                    else:
                         self.scramble = list(self.Heatmap[i])  #pick out row of heatmap
                         extract = self.scramble[center]
                         self.scramble.remove(extract)  # remove diagonal
                         random.shuffle(self.scramble)  # scramble off diagonal values
                         self.scramble.insert(center,extract) #add diagonal back in

                         summation = 0
                         summation2 = 0
                         for j in range(1,int(self.DIrange)+1):
                              summation = summation + float(self.scramble[center+int(self.distance)+j])
                              summation2 = summation2 + float(self.scramble[center-int(self.distance)-j])
                         self.sAset.append(summation)
                         self.sBset.append(summation2)
               change = self.windowing(self.sAset,self.sBset)
               self.sAset = change[0]
               self.sBset = change[1]

def REPL():

      parser = argparse.ArgumentParser()
      parser.add_argument("count", type=str, help="count file")
      parser.add_argument("start", type=int, help="start")
      parser.add_argument("end", type=int, help="end")
      parser.add_argument("range", type=int, help="upstream/downstream range")
      parser.add_argument("number", type=str, help ="copy number")
      parser.add_argument("window", nargs='?',type=int, default = 1, help="running window size across diagonal")
      parser.add_argument('distance', nargs='?',type = int,default=0, help ='gap from diagonal')
      parser.add_argument('diagonal', nargs='?', default=True, help ='banded matrix = false, normal heatmap = true')
      parser.add_argument('region', nargs='?', default="regionX")
      args = parser.parse_args()

      print("REPL args.diagonal =", args.diagonal)

      data = MatrixMakeBanded(args.count,args.start,args.end,args.range,args.distance,args.region)

      if args.diagonal == "False":
           args.diagonal = False


      value = DI(data.Heatmap,args.distance,args.range,args.window,args.diagonal)
      value.DI_metric()
      value.scramble_DI_metric()

      print('printing ', len(value.DIset), ' DI values to DI.csv and ', len(value.sDIset),' DI random values to randomDI.csv')

      f = open("DI"+ args.number.zfill(3)+".csv","w")
      wr = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
      wr.writerow(value.DIset)
      f.close()

      f = open("randomDI" + args.number.zfill(3)+ ".csv","w")
      wr = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
      wr.writerow(value.sDIset)
      f.close()

if __name__ == "__main__":
    REPL()
