#! /bin/bash
#

bsub -o DI_analysis.o -e DI_analysis.e python DI_TAD_calls2Hermitianmodv2_firstaic90.py input/240min_10000v3_merge.counts input/240min_10000v3_merge.bed 200 10000 1 0 2 mitosis240min_merge_DI200

