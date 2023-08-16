#Module for removing DI's at beginning and end of each chromosome and joining with bed file.  Version update to allow for windowing.
#User can now measure DI over region greater than binsize.  Each window DI is non-overlapping (eliminate confusion running HMM and postprocessing where DI would be part of more than one region)
#Version 8 September 6, 2017

import re
import random
import copy
import csv
import argparse
import numpy as np
from MatrixMakeBanded4 import MatrixMakeBanded
from DI8 import DI

"modified script to include cull_DI method which removes additional deadzones as defined as zeros in center for >= 62 consecutive"

class BedConvert:

     def __init__(self,bed_file,DIset,DIrange,istart,iend,binsize,window,distance):
          self.bed = []
          self.DIset = DIset
          self.DIrange = DIrange
          self.distance = distance
          self.istart = istart
          self.iend = iend
          self.binsize = binsize
          self.window = window
          f = open(bed_file,"r")
          for lines in f:
               lines = lines.strip('\n')
               values = lines.split('\t')
               values = [i.strip(' ') for i in values]
               self.bed.append(values)
          f.close()

          self.threshold = (istart + DIrange + distance -1)*binsize  #more conservative, I assumed every chr starts at same base position as chr1
          #self.threshold = (DIrange + distance)*binsize #assumes successive chr start at 0 base position

          self._new()
          self._cull_DI()

     def _new(self):
          """
              removing DI values at end of chromosome that extend to neighboring chromosome

          """
          temp = []
          self.remove_edge_comprehensive = []
          self.DI_remove_edge = []


          print("DIset > 0: ", self.DIset[self.DIset>0])
          print("length DIset > 0: ", len(self.DIset[self.DIset>0]))


          testing_values = []
          increment = 0
          lookat = 0
          mark = []
          chromosome_instance = self.bed[self.DIrange + self.distance + self.istart - 1][0]
          prev_chrom_instance = chromosome_instance
          for bin in range(self.DIrange + self.distance + self.istart - 1,len(self.bed)-self.DIrange-self.distance -(self.window-1)):
               if increment < len(self.DIset):
                    chromosome_instance = self.bed[bin][0]
                    values = [str(chromosome_instance),int(self.bed[bin][1]),int(self.bed[bin][1]) + (int(self.binsize)*int(self.window)),self.DIset[increment]]
                    testing_values.append(self.DIset[increment]) 
                    if chromosome_instance != prev_chrom_instance: #place demarcation line marks at boundary of chr
                         mark.append(bin)
                         prev_chrom_instance = chromosome_instance
                    increment = increment + 1
                    temp.append(values)
          mark.append(bin) #add final chromosome          

          testing_values = np.array(testing_values)
          print("testing_values > 0: ", testing_values[testing_values > 0])

          windowselect = 0
          print("mark: ",mark)
          for i in range(0,len(temp)):
               # remove index bleed through at beginning and end of each chr (DI at beginning and end influenced by previous/next chr
               if (int(temp[i][1]) >= int(self.threshold)):
                    if (mark[lookat]-i) > (self.DIrange + self.distance + self.window-1):
                         #print("lookat: ",lookat)
                         if (windowselect%int(self.window)==0) or (int(temp[i][1]) == int(self.threshold)): #skip over overlapping window portions
                              self.remove_edge_comprehensive.append(temp[i])
                              self.DI_remove_edge.append(temp[i][3])
                              if int(temp[i][1]) == int(self.threshold):
                                  windowselect = 0 #reset window position if at start of chromosome so it starts at given threshold
                         windowselect = windowselect + 1
                         #if ((mark[lookat]-i) <= 0 and lookat < len(mark)-1):
                    if (mark[lookat]-i) <= 0 and (lookat < len(mark)-1):
                         lookat = lookat + 1 #increment to next chr boundary  

          self.DI_remove_edge = np.array(self.DI_remove_edge)

     def _cull_DI(self):
          """
               removing empty sparse regions ( >= 62 consecutive bins) from DI
               

          """
          self.remove_edge_sparse_comprehensive = [] #sparse DI plus coordinates      
          self.DI_remove_edge_sparse = [] #sparse DI only


          history = np.empty(62)

          for i in range(0,len(self.remove_edge_comprehensive)):
               if i > 13:
                    for k in range(0,len(history)):
                         history[k] = self.remove_edge_comprehensive[i-k][3]
                    if np.mean(history) != 0:
                         self.remove_edge_sparse_comprehensive.append(self.remove_edge_comprehensive[i])
                         self.DI_remove_edge_sparse.append(self.remove_edge_comprehensive[i][3])
               else:
                    self.remove_edge_sparse_comprehensive.append(self.remove_edge_comprehensive[i])
                    self.DI_remove_edge_sparse.append(self.remove_edge_comprehensive[i][3])     

          self.DI_remove_edge_sparse = np.array(self.DI_remove_edge_sparse) 

def REPL():

      parser = argparse.ArgumentParser()
      parser.add_argument("count", type=str, help="count file")
      parser.add_argument("bed", type=str, help="bed file")
      parser.add_argument("DIrange", type=int, help="range")
      parser.add_argument("istart", type=int, help="index start")
      parser.add_argument("iend",type=int,help="index end")
      parser.add_argument("binsize",type=int,help="binsize")
      parser.add_argument("window", nargs='?',type=int, default = 1, help="running window size across diagonal")
      parser.add_argument('distance', nargs='?',type = int,default=0, help ='gap from diagonal')
      parser.add_argument('diagonal', nargs='?', default=True, help ='banded matrix = false, normal heatmap = true')
      parser.add_argument('region', nargs='?', default="regionX")
      args = parser.parse_args()

      if args.diagonal == "False":
            args.diagonal = False

      data = MatrixMakeBanded(args.count,args.istart,args.iend,args.DIrange,args.distance,args.region)
      value = DI(data.Heatmap,args.distance,args.DIrange,args.window,args.diagonal)
      value.DI_metric()
      #value.scramble_DI_metric()

      Data2 = BedConvert(args.bed,value.DIset,args.DIrange,args.istart,args.iend,args.binsize,args.window,args.distance)      

      output3 = cull_DI(Data2.output2)



if __name__ == "__main__":
    REPL()
