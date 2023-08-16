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
# windowing handled by convertbed2v2_62bin.py after DI8.py is run


import argparse
import os
import sys
import re
import random
import copy
import csv
import numpy as np

class MatrixMakeBanded:
      start = 1 #100000

      def __init__(self, filename,distance,gap,label):
           self.filename = filename  #input counts
           self.distance = int(distance)
           self.label = label
           self.gap = int(gap)
           self.readfile()


      def readfile(self):

           end = 1
           f = open(self.filename,"r")
           #find maximum
           for lines in f:
                lines = lines.strip('\n')
                values = lines.split('\t')
                values = [float(i.strip(' ')) for i in values]
                if int(values[0]) > end:
                    end = int(values[0]) 
           f.close()

           self.end = end         

           #initiate heatmap
           self.Heatmap = np.zeros((self.end-MatrixMakeBanded.start-(2*(self.gap-self.distance)-1),2*(self.distance+self.gap)+1))
  
           #run again to populate matrix
           f = open(self.filename,"r")
           for lines in f:
                lines = lines.strip('\n')
                values = lines.split('\t')
                values = [float(i.strip(' ')) for i in values]
                if np.isnan(values[2]) == False:
                    self._populate_matrix(values)
           f.close()



      def _populate_matrix(self,values):
           #test to see if input line from counts is suitable for storage
           firstedge = self.start + self.distance + self.gap
           finaledge = self.end - self.distance - self.gap
           if ((values[0] < self.start + self.distance + self.gap)): #need to use the first set of out bounds downstream to determine in bounds upstream at start
                if ((abs(int(values[0])- int(values[1]))) <= int(self.distance)+int(self.gap)):
                    row = values[0] - firstedge
                    rowtwo = (int(values[1]) - int(values[0])) + int(row) #go below
                    if int(rowtwo) >= 0:
                        columntwo = (self.gap + self.distance) - (int(rowtwo) - int(row)) #col determined column wise go below
                        self.Heatmap[int(rowtwo)][int(columntwo)] = float(values[2])
           if ((values[0] >= MatrixMakeBanded.start + self.distance + self.gap) and (values[0] <= self.end-self.distance-self.gap)):
                if ((abs(int(values[0])- int(values[1]))) <= int(self.distance)+int(self.gap)): #distance between itself and neighbor
                     if int(values[1]) >  int(values[0]): #values given based on upper triangle find lower triangle
                          column = (int(values[1]) - int(values[0])) + self.distance+self.gap #col determined row wise 
                          row = values[0] - firstedge
                          rowtwo = (int(values[1]) - int(values[0])) + int(row) #go below
                          self.Heatmap[int(row)][int(column)] = float(values[2])
                          if int(rowtwo) <= int(finaledge) - int(firstedge):
                               columntwo = (self.gap + self.distance) - (int(rowtwo) - int(row)) #col determined column wise
                               self.Heatmap[int(rowtwo)][int(columntwo)] = float(values[2])
                     elif int(values[1]) < int(values[0]): #values given based on lower triangle find upper triangle
                          column = (self.gap + self.distance) - (int(values[0]) - int(values[1]))
                          row = values[0] - firstedge
                          rowtwo = int(row) - (int(values[0]) - int(values[1])) #go above
                          self.Heatmap[int(row)][int(column)] = float(values[2])
                          if int(rowtwo) >= 0:
                               columntwo = (row - rowtwo) + self.distance+self.gap #col determined column wise
                               self.Heatmap[int(rowtwo)][int(columntwo)] = float(values[2])
                     else: #right on diagonal do not need to transpose
                          column = self.distance + self.gap
                          row = values[0] - firstedge
                          self.Heatmap[int(row)][int(column)] = float(values[2])

      def save(self,fileName):
          np.savetxt(fileName, self.Heatmap, delimiter='\t')



def REPL():


      parser = argparse.ArgumentParser()
      parser.add_argument("count", type=str, help="count file")
      parser.add_argument("range", type=int, help="upstream/downstream range")
      parser.add_argument("number", type=str, help ="copy of heatmap")
      parser.add_argument('gap', type=int, nargs='?', default=0)
      parser.add_argument('region', nargs='?', default="regionX")
      args = parser.parse_args()

      data = MatrixMakeBanded(args.count,args.range,args.gap,args.region)
      data.save(args.region + args.number.zfill(3) + "_bandheatmap.csv")




if __name__ == "__main__":
    REPL()
