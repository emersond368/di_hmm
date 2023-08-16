#!/usr/bin/perl
use strict;


#open(MYINPUTFILE, "../gene_deserts_cent_telom.bed"); # open for input
#my(@lines) = <MYINPUTFILE>; # read file into list
#@lines = sort(@lines); # sort the list
#my($line);
#foreach $line (@lines) # loop thru list
# {
# my @inst = split('\t', $line);
# print "$inst[0]\n"; # print in sort order
# }
#close(MYINPUTFILE);

MAIN : { 

    my $mark = 0;
    my $increment = 0;
    my $gaps;

    my $infile  = $ARGV[0];

    open(MYINPUTFILE,$infile);
    my(@inputs) = <MYINPUTFILE>; # read file into list        @inputs = sort(@inputs); # sort the list
    @inputs = sort(@inputs);
    my($input);
    close(MYINPUTFILE);

    while (my $line = <STDIN>) {
        chomp $line;
        my ($chr, $start, $end) = split(/\t/,$line);
        $mark = 0;        

        foreach $input (@inputs) # loop thru list
        {
            my @inst = split('\t', $input);
            #print "$inst[0]\n"; # print in sort order
            if ((($start < $inst[1]) && ($start > $inst[0])) || (($end < $inst[1]) && ($end > $inst[0])) || (($start > $inst[0]) && ($inst[1] > $end)) || (($start < $inst[0]) && ($inst[1] < $end))) {
                 $mark = 1;
            } 
         }
         if ($mark == 0) {
              print $chr. "\t" . $start . "\t" . $end . "\n";

         }
    }   
}






