# MatrixMakeBanded
# version 4 by Daniel Emerson demerson368@gmail.com
# Sept 6 2016

#simpler design than MatrixMakeBanded3.py - makes a zero numpy array where non-zero counts are added
#on top of instead of finding gaps between counts.  Also, much faster

#Program is designed to give a banded heatmap around diagonal value (organized as diagonal+- range)
#so sequence of 1 2 3 4 5 6 7 8 9 10 with range +- 2 gives:
#
#    1 2 3 4 5
#    2 3 4 5 6
#    3 4 5 6 7
#    4 5 6 7 8
#    5 6 7 8 9
#    6 7 8 9 10

# first and last 2 of sequence get cut off (no room for range)
# windowing handled by convertbed2.py after DI8.py is run

# python MatrixMake4.py pcount startindex endindex range heatmapcopy# name

import argparse
import os
import sys
import re
import random
import copy
import csv
import numpy as np

class MatrixMakeBanded:
      Maximum = 0
      Minimum = 100000
      stop_processing = False

      def __init__(self, filename,start,end,distance,gap,region):
           self.filename = filename  #input counts
           self.start = int(start)
           self.end = int(end)
           self.distance = int(distance)
           self.Heatmap = np.zeros((end-start-(2*(gap+distance)-1),2*(distance+gap)+1))
           self.region = region
           self.gap = int(gap)
           self.readfile()


      def readfile(self):

           f = open(self.filename,"r")
           for lines in f:
                lines = lines.strip('\n')
                values = lines.split('\t')
                values = [float(i.strip(' ')) for i in values]
                self.test(values)
                MatrixMakeBanded.Maximum = int(self.end)
                MatrixMakeBanded.Minimum = int(self.start)
                if MatrixMakeBanded.stop_processing:  #keep from wasting too much time processing lines if exceeds the end
                     break
           f.close()



      def test(self,values):
           #test to see if input line from counts is suitable for storage

           if ((values[0] >= self.start + self.distance + self.gap) and (values[0] <= self.end-self.distance-self.gap)):
                firstedge = self.start + self.distance + self.gap
                finaledge = self.end - self.distance - self.gap
                if ((abs(int(values[0])- int(values[1]))) <= int(self.distance)+int(self.gap)): #distance between itself and neighbor
                     if int(values[1]) >  int(values[0]):
                          column = (int(values[1]) - int(values[0])) + self.distance+self.gap
                          row = values[0] - firstedge
                          self.Heatmap[int(row)][int(column)] = float(values[2])
                     elif int(values[1]) < int(values[0]):
                          column = (self.gap + self.distance) - (int(values[0]) - int(values[1]))
                          row = values[0] - firstedge
                          self.Heatmap[int(row)][int(column)] = float(values[2])
                     else: # int(values[1]) ==  int(values[0]):
                          column = self.distance + self.gap
                          row = values[0] - firstedge
                          self.Heatmap[int(row)][int(column)] = float(values[2])
           else:
                if (int(values[0]) > int(self.end)):
                    MatrixMakeBanded.stop_processing = True

      def save(self,fileName):
          np.savetxt(fileName, self.Heatmap, delimiter='\t')



def REPL():


      parser = argparse.ArgumentParser()
      parser.add_argument("count", type=str, help="count file")
      parser.add_argument("start", type=int, help="start")
      parser.add_argument("end", type=int, help="end")
      parser.add_argument("range", type=int, help="upstream/downstream range")
      parser.add_argument("number", type=str, help ="copy of heatmap")
      parser.add_argument('gap', type=int, nargs='?', default=0)
      parser.add_argument('region', nargs='?', default="regionX")
      args = parser.parse_args()

      data = MatrixMakeBanded(args.count,args.start,args.end,args.range,args.gap,args.region)
      data.save(args.region + args.number.zfill(3) + "_bandheatmap.csv")



if __name__ == "__main__":
    REPL()
