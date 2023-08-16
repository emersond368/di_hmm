# Matrix_make
# version 4 by Daniel Emerson demerson368@gmail.com
# Aug 21 2016

#Program is designed to give a square heat map based on the starting bin and ending
# bin provided by the user ((end - start) + 1)x((end - start) + 1) (stored in file
# output_heatmap.csv).  Additionally, a pcount file (output_pcount.csv) and a bed
# file (output_bed.csv) is created.  The count input file of form "bin# bin# count"
# (%i %i (%i or %f)) and input bed file of form "region # # bin#" (%s %i %i %i) is 
# required.  The user has the option to give title.  If none is given, then default
# is regionX

# proper use of program on command line:
# python MatrixMake.py counts_input bed_input startbin endbin (title*)
# * = optional


import argparse
import os
import sys
import re
import random
import copy
import csv


class MatrixMake:
      Maximum = 0
      Minimum = 100000
      stop_processing = False

      def __init__(self, filename,filename2,start,end,region):
           self.Heatmap = []  # square matrix ((end-start)+1) x ((end - start) + 1)
           self.Elongate = []
           self.pcount = [] #output pcounts
           self.bed = [] #output bed
           self.filename = filename  #input counts
           self.filename2 = filename2 #input bed
           self.start = start
           self.end = end
           self.list_of_bins = []
           self.list_of_regions = []
           self.region = region

           for i in range(int(self.start),int(self.end)+1):
               self.list_of_bins.append(int(i))

           self.readfile()
           self.buffer()
           self.postprocess()
           self.pCountMake()
           self.bedMake()


      def readfile(self):

           f = open(self.filename,"r")
           for lines in f:
                lines = lines.strip('\n')
                values = lines.split('\t')
                values = [float(i.strip(' ')) for i in values]
                self.test(values)
                MatrixMake.Maximum = int(self.end)
                MatrixMake.Minimum = int(self.start)
                if MatrixMake.stop_processing:  #keep from wasting too much time processing lines if exceeds the end
                     break
           f.close()

           MatrixMake.stop_processing = False

           f = open(self.filename2,"r")
           processing = 1
           for lines in f:
               lines = lines.strip('\n')
               values = lines.split('\t')
               values = [i.strip(' ') for i in values]
               self.test2(values,processing)
               if MatrixMake.stop_processing:  #keep from wasting too much time processing lines if exceeds the end
                    break
           f.close()



      def test(self,values):
           #test to see if input line from counts is suitable for storage

           if ((int(values[0]) >= int(self.start)) and (int(values[0]) <= int(self.end)) and ((int(values[1]) >= int(self.start)) and (int(values[1]) <= int(self.end)))):
                self.Elongate.append(values)  #store bins that are found in square matrix region
                print "adding bin = ", int(values[0])
           else:
                if (int(values[0]) > int(self.end)):
                    MatrixMake.stop_processing = True

      def test2(self,values,processing):
           #test to see if input line from bed is suitable for storage:

           if (int(values[3]) in self.list_of_bins):
               self.list_of_regions.append(values)  #attach all rows of .bed input to matching bins
               processing = processing + 1
           if (processing > (int(self.end) - int(self.start) + 2)):  # already found all region matchups no longer need to search file
               MatrixMake.stop_processing
    
      def buffer(self):
      # fill in missing 0s based on missing second column
          i = 0

          if int(self.Elongate[0][1]) != MatrixMake.Minimum: #forward fill at beginning of file
               neighbor = MatrixMake.Minimum
               for j in range(0,int(self.Elongate[0][1])-MatrixMake.Minimum):  #add in 0s matching gap size
                    self.Elongate.insert(0 + i,[self.Elongate[0][0], neighbor, 0])
                    i = i + 1
                    neighbor = neighbor + 1
          i = 1
          while (i < len(self.Elongate)):
               if i != (len(self.Elongate) - 1):
                    if int(self.Elongate[i][0]) == int(self.Elongate[i-1][0]): #make sure comparing same bin
                         if int(self.Elongate[i][1] - self.Elongate[i-1][1]) > 1:  #check if gap in neighbor
                              neighbor = self.Elongate[i-1][1]+1
                              gapSize = int((self.Elongate[i][1] - self.Elongate[i-1][1]))-1
                              for j in range(0,gapSize):  #add in 0s matching gap size
                                   self.Elongate.insert(i,[self.Elongate[i][0], neighbor, 0])
                                   neighbor = neighbor + 1
                                   i = i + 1
                    else: # Test beginning and end of each bin (filler to end/beginning)
                         if int(self.Elongate[i-1][1]) != MatrixMake.Maximum:
                              neighbor = self.Elongate[i-1][1]+1
                              gapSize = int(MatrixMake.Maximum - self.Elongate[i-1][1])
                              for j in range(0,gapSize):  #add in 0s matching gap size
                                   self.Elongate.insert(i,[self.Elongate[i-1][0], neighbor, 0])
                                   neighbor = neighbor + 1
                                   i = i + 1
                         if int(self.Elongate[i][1]) != MatrixMake.Minimum:
                              neighbor = MatrixMake.Minimum
                              gapSize = int(self.Elongate[i][1]-MatrixMake.Minimum)
                              for j in range(0,gapSize):  #add in 0s matching gap size
                                   self.Elongate.insert(i,[self.Elongate[i][0], neighbor, 0])
                                   neighbor = neighbor + 1
                                   i = i + 1
               else: #check last row value (add to maximum at end)
 
                    if int(self.Elongate[i][1]) != MatrixMake.Maximum:
                         if int(self.Elongate[i][1] - self.Elongate[i-1][1]) > 1:  #check if gap in prior neighbor and add before
                              neighbor = self.Elongate[i-1][1]+1
                              gapSize = int((self.Elongate[i][1] - self.Elongate[i-1][1]))-1
                              for j in range(0,gapSize):  #add in 0s matching gap size
                                   self.Elongate.insert(i,[self.Elongate[i][0], neighbor, 0])
                                   neighbor = neighbor + 1
                                   i=i+1
                         neighbor = self.Elongate[i][1]+1  #then append after to go toward maximum
                         gapSize = int(MatrixMake.Maximum - self.Elongate[i][1])
                         for j in range(0,gapSize):  #add in 0s matching gap size
                              self.Elongate.append([self.Elongate[i][0], neighbor, 0])
                              neighbor = neighbor + 1
                              i = i + 1
                    else:  #if maximum present at end but gap is also there
                         if int(self.Elongate[i][0]) == int(self.Elongate[i-1][0]):
                              neighbor = self.Elongate[i-1][1]+1  #just check for gap before last value, no need to append afterward
                              gapSize = int((self.Elongate[i][1] - self.Elongate[i-1][1]))-1
                              for j in range(0,gapSize):  #add in 0s matching gap size
                                   self.Elongate.insert(i,[self.Elongate[i][0], neighbor, 0])
                                   neighbor = neighbor + 1
                                   i = i + 1

               i = i + 1

      def HM(self, matrix):  #make square heat map
          row = []
          compare = []
          output = []
          increment = 1
          for i in range(0,len(matrix)):
               if increment == 1:
                    row.append(matrix[i][2])
                    compare.append(int(matrix[i][0]))
                    increment = increment + 1
               else:
                    if (i < len(matrix)-1):
                         if (compare[len(compare)-1] == int(matrix[i][0])):
                              row.append(matrix[i][2])
                              compare.append(int(matrix[i][0]))
                              increment = increment + 1
                         else:
                              output.append(row)
                              compare.append(int(matrix[i][0]))
                              row = []
                              row.append(matrix[i][2])
                              increment = increment + 1
                    else:
                              row.append(matrix[i][2]) #add final value to heatmap
                              output.append(row)
          return output

      def postprocess(self):  #fill in 0s based on missing rows from first column
 
          i = 0         
          if int(self.Elongate[0][0] != int(self.start)):  #user specified start bin that does not exist
                gapSize = int(self.Elongate[0][0] - int(self.start))
                for j in range(0,gapSize):
                     neighbor = self.Elongate[0][0]+j
                     for k in range(MatrixMake.Minimum,MatrixMake.Maximum+1):
                          valueInsert = [neighbor,k,"         0"]
                          self.Elongate.insert(i,valueInsert)
                          i = i+1

          i = 0
          while (i < len(self.Elongate)-2):
               gapSize = int(self.Elongate[i+1][0] - self.Elongate[i][0])
               if (gapSize > 1):  #if column skips over segment (presumably all 0s)
                    for j in range(1,gapSize):  #fill in for however large gap is
                         neighbor = self.Elongate[i][0]+1
                         for k in range(MatrixMake.Minimum,MatrixMake.Maximum+1):
                              i=i+1
                              valueInsert = [neighbor,k,"         0"]
                              self.Elongate.insert(i,valueInsert)
               i = i + 1
          if int(int(self.end) != self.Elongate[len(self.Elongate) -1][0]): #user specified end bin that does not exist
              add_to_end = self.Elongate[len(self.Elongate) -1][0]
              gapSize = int(int(self.end) - add_to_end)
              for j in range(0,gapSize):
                   neighbor = add_to_end+j+1
                   for k in range(MatrixMake.Minimum,MatrixMake.Maximum+1):
                        self.Elongate.append([neighbor,k,"         0"])
        
          self.Heatmap = self.HM(self.Elongate)

      def save(self,fileName,fileName3,fileName4):
          #make output files

          f = open(fileName,"w") # Heatmap (end - start) +1 x (end - start) + 1
          f.write('\n'.join(','.join('{:10}'.format(item) for item in row) for row in self.Heatmap))
          f.close();

          f = open(fileName3,"w") # pCount
          f.write('\n'.join('\t'.join('{:1}'.format(item) for item in row) for row in self.pcount))
          f.close();

          f = open(fileName4,"w") # Bed
          f.write('\n'.join('\t'.join('{:1}'.format(item) for item in row) for row in self.bed))
          f.close();

 #         f = open("elongatetest","w") # Bed
 #         f.write('\n'.join('\t'.join('{:1}'.format(item) for item in row) for row in self.Elongate))
 #         f.close();

      def pCountMake(self):
          #store pcount data

          digits = len(str(MatrixMake.Maximum))
          temppcount =  []
          b = 0
          a = 1
          while (b < len(self.Elongate)):
               for i in range(b,a+b):
                   bin_value1 = int(self.Elongate[i][0]) - int(MatrixMake.Minimum)
                   bin_value2 = int(self.Elongate[i][1]) - int(MatrixMake.Minimum)
                   self.pcount.append([self.region + "_BIN_" + str(bin_value1).zfill(digits),self.region + "_BIN_" + str(bin_value2).zfill(digits),str(self.Elongate[i][2]).strip(" ")])
               b = b + ((int(self.end) - int(self.start))+1)
               a = a + 1

      def bedMake(self):
         #store bed data

         digits = len(str(MatrixMake.Maximum))
         for i in range(0,len(self.list_of_regions)):
              bin_value = int(self.list_of_regions[i][3]) - int(MatrixMake.Minimum)
              self.bed.append([self.list_of_regions[i][0].strip(' '), self.list_of_regions[i][1].strip(' '),self.list_of_regions[i][2].strip(' '),self.region + "_BIN_" +  str(bin_value).zfill(digits)])

def REPL():


      parser = argparse.ArgumentParser()
      parser.add_argument("count", type=str, help="count file")
      parser.add_argument("bed", type=str, help="str file")
      parser.add_argument("start", type=int, help="start")
      parser.add_argument("end", type=int, help="end")
      parser.add_argument("number", type=str, help ="copy of heatmap")
      parser.add_argument('region', nargs='?', default="regionX")
      args = parser.parse_args()

      data = MatrixMake(args.count,args.bed,args.start,args.end,args.region)
      data.save(args.region + args.number.zfill(3) + "_heatmap.csv",args.region + args.number.zfill(3) + "_pvalues.counts",args.region + args.number.zfill(3) +".bed")



if __name__ == "__main__":
    REPL()
