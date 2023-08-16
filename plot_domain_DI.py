import matplotlib as mpl
mpl.use('Agg')

from lib5c.plotters.extendable import ExtendableHeatmap
from lib5c.parsers.bed import load_features
import lib5c.plotters
import scipy.sparse
import numpy as np
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('heatmap_name',type=str,help="name of heatmap")
    parser.add_argument('domains',type=str,help="name of domain file with 3 layers")
    parser.add_argument('DI_track',type=str,help="PCA track")
    parser.add_argument('min_DI',type=float,help="minimum PCA value")
    parser.add_argument('max_DI',type=float,help="maximum PCA value")
    parser.add_argument('chr', type=str, help = "chromosome")
    parser.add_argument('start', type=int, help = "start")
    parser.add_argument('end', type=int, help = "end")
    parser.add_argument('resolution',type=int, help = "resolution")
    parser.add_argument('max_color',default = 100,type=float,help="maximum color scale")

    args = parser.parse_args()

    total_heatmap_file = "input/" + args.heatmap_name

    matrix_pre = scipy.sparse.load_npz(total_heatmap_file)
    matrix_pre_csr = matrix_pre.tocsr()
    size = matrix_pre_csr.shape[0]

    domains = lib5c.parsers.bed.load_features("output/" + args.domains)
    domains_chrom_list = domains[args.chr]
    print(domains_chrom_list)

    DI_pos_signal = []
    DI_neg_signal = []
    DI_start = []

    DI = lib5c.parsers.bed.load_features("output/" + args.DI_track)
    DI_chrom_list = DI[args.chr]
    for i in range(len(DI_chrom_list)):
        DI_pos_signal.append(DI_chrom_list[i]['value'])
        DI_neg_signal.append(DI_chrom_list[i]['value'])
        DI_start.append(DI_chrom_list[i]['start'])


    DI_pos_signal = np.array(DI_pos_signal)
    DI_neg_signal = np.array(DI_neg_signal)
    DI_start = np.array(DI_start)

    DI_pos_signal[DI_pos_signal <= 0] = 0 #np.nan
    DI_neg_signal[DI_neg_signal > 0] = 0 #np.nan

    matrix = matrix_pre_csr[int(args.start/args.resolution):int(args.end/args.resolution),int(args.start/args.resolution):int(args.end/args.resolution)].todense()
    some_square_matrix = np.triu(matrix) + np.triu(matrix).T - np.diag(np.diag(matrix))

    h = ExtendableHeatmap(array=some_square_matrix,grange_x={'chrom': args.chr, 'start': args.start,'end': args.end},colorscale=(0, args.max_color),colormap='Reds')


    h.outline_domains(domains_chrom_list,color='green')

    width = 1
    h.add_ax('DI')
    h['DI'].bar(np.array(DI_start)/args.resolution,DI_pos_signal,width,color='g')
    h['DI'].bar(np.array(DI_start)/args.resolution,DI_neg_signal,width,color='r')
    h['DI'].set_xlim((int(args.start/args.resolution),int(args.end/args.resolution)))
    h['DI'].set_ylim((args.min_DI,args.max_DI))
    h['DI'].set_xticklabels([])
    h['DI'].axis('off')

    h.save("output/" + args.domains[:-4] + '_' + args.chr  + '_' + str(args.start)  + '_' + str(args.end) + '_' +  str(args.max_color) + '.png')


if __name__ == "__main__":
    main()
     
