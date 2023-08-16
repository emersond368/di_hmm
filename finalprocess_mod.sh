#! /bin/bash
# inputfile must be bed file plus corresponding DI and hidden state values

name=mitosis240min_merge_DI200
chrom_sizes=chromosome_sizes_mm10.txt 
resolution=10000
window_size=1

region=$(($resolution*$window_size))

awk -F'\t' '{if (NR > 1) {printf("%s\t%d\t%d\t%f\t%d\t%d\t%f\t%f\t%f\n", $1,$2,$3,$4,$5,$6,$7,$8,$9)}}' output/$name\_postp_list.tsv | perl converter_7col.pl > output/$name\_hmm_7colfile_c

awk -F'\t' '{if (NR > 1) {printf("%s\t%d\t%d\t%f\n", $1,$2,$3,$4)}}' output/$name\_postp_list.tsv > output/$name\_DIscore.bed


input=$chrom_sizes
i=0
declare -a array=()
while IFS= read -r line; do
    array[$i]="$line"    
    let i++          
done < "$input"

i=0
while (( $i < ${#array[@]} ));
do
    chromosome=$(echo ${array[$i]} | cut -d ' ' -f 1)
    echo $chromosome

    awk -v awk_val=$chromosome -F'\t' '$1 == awk_val {print $0}' output/$name\_hmm_7colfile_c > output/$chromosome\hmmfile2

    awk -v awk_val=$chromosome -F'\t' '{if (NR > 1) {if (($1 == awk_val) && ($1 == prevchr)) { if ($2 != prev) {printf("%d\t%d\n", prev, $2)}}};prev = $3;prevchr = $1}' output/$name\_postp_list.tsv > output/gaps$chromosome

    perl hmm_probablity_correcter.pl output/$chromosome\hmmfile2 2 0.99 $region | perl hmm-state_caller.pl $chrom_sizes $chromosome | perl hmm-state_domains.pl  | perl final_check.pl output/gaps$chromosome > output/$name\_finaldomains_$chromosome\.bed

    rm output/$chromosome\hmmfile2
    rm output/gaps*
    let i++
done

rm output/$name\_hmm_7colfile_c
