#Module for removing DI's at beginning and end of each chromosome and joining with bed file.  Version update to allow for windowing.
#User can now measure DI over region greater than binsize.  Each window DI is non-overlapping (eliminate confusion running HMM and postprocessing where DI would be part of more than one region)
#Version 8 September 6, 2016

import re
import random
import copy
import csv
import argparse
import numpy as np
from MatrixMakeBanded4 import MatrixMakeBanded
from DI8 import DI

"modified script to include cull_DI method which removes additional deadzones as defined as zeros in center for >= 15 consecutive"

class BedConvert:

     def __init__(self,bed,DIset,DIrange,istart,iend,startchr,endchr,binsize,window,distance,cull_value):
          self.output = []
          self.output2 = []
          self.output3 = []
          self.filelist = []
          self.filelist2 = DIset
          self.DIrange = DIrange
          self.distance = distance
          self.istart = istart
          self.iend = iend
          self.startchr = startchr
          self.endchr = endchr
          self.binsize = binsize
          self.window = window
          self.bed = bed
          self.cull_value = cull_value
          f = open(self.bed,"r")
          for lines in f:
               lines = lines.strip('\n')
               values = lines.split('\t')
               values = [i.strip(' ') for i in values]
               self.filelist.append(values)

          self.threshold = (istart + DIrange + distance -1)*binsize  #more conservative, I assumed every chr starts at same base position as chr1
          #self.threshold = (DIrange + distance)*binsize #assumes successive chr start at 0 base position

          self.new()
          self.cull_DI()

     def new(self):
          start = 1
          increment = 0
          lookat = 0
          prev = 1
          mark = []
          state = self.filelist[self.DIrange + self.distance + self.istart - 1][0]
          for i in range(self.DIrange + self.distance + self.istart - 1,len(self.filelist)-self.DIrange-self.distance -(self.window-1)):
               if self.filelist[i][0] != state:
                    start = start + 1
                    state = self.filelist[i][0]
               if (start >= self.startchr and start <= self.endchr):
                    if increment < len(self.filelist2):
                         values = [start,self.filelist[i][1],int(self.filelist[i][1]) + (int(self.binsize)*int(self.window)),self.filelist2[increment]]
                         if start != prev: #place demarcation line marks at boundary of chr
                              mark.append(increment)
                              prev = start
                         self.output.append(values)
                         increment = increment + 1
          windowselect = 0
          for i in range(0,len(self.output)):
               #print "mark[lookat] = ",mark[lookat]
               #print "len(mark) = ",len(mark)
               # remove index bleed through at beginning and end of each chr (DI at beginning and end influenced by previous/next chr)
               if (int(self.output[i][1]) >= int(self.threshold)):
                    if (mark[lookat]-i) > (self.DIrange + self.distance + self.window-1) or (mark[lookat]-i) <= 0:
                         if (windowselect%int(self.window)==0) or (int(self.output[i][1]) == int(self.threshold)): #skip over overlapping window portions
                              self.output2.append(self.output[i])
                              if int(self.output[i][1]) == int(self.threshold):
                                  windowselect = 0 #reset window position if at start of chromosome so it starts at given threshold
                         windowselect = windowselect + 1
                         if ((mark[lookat]-i) <= 0 and lookat < self.endchr-2):
                              lookat = lookat + 1 #increment to next chr boundary  

     def cull_DI(self):

          print("using cull_value: ", self.cull_value)
          history = np.empty(self.cull_value)

          for i in range(0,len(self.output2)):
               if i > 13:
                    for k in range(0,len(history)):
                         history[k] = self.output2[i-k][3]
                    if np.mean(history) != 0:
                         self.output3.append(self.output2[i])
               else:
                    self.output3.append(self.output2[i])
     

def REPL():

      parser = argparse.ArgumentParser()
      parser.add_argument("count", type=str, help="count file")
      parser.add_argument("bed", type=str, help="bed file")
      parser.add_argument("DIrange", type=int, help="range")
      parser.add_argument("istart", type=int, help="index start")
      parser.add_argument("iend",type=int,help="index end")
      parser.add_argument("startchr",type=int, help="chr start")
      parser.add_argument("endchr",type=int, help="chr end")
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
      value.scramble_DI_metric()

      cull_value = 62
      Data2 = BedConvert(args.bed,value.DIset,args.DIrange,args.istart,args.iend,args.startchr,args.endchr,args.binsize,args.window,args.distance,cull_value)
      
      #output3 = cull_DI(Data2.output2)

      #f = open("testbed","w")
      #for vals in output3:
      #     print >>f,"%s %s %s %s" % (vals[0],vals[1],vals[2],vals[3])        
      #f.close()

if __name__ == "__main__":
    REPL()
