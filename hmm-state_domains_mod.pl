#!/usr/bin/perl

use strict;

MAIN : {

    my $increment = 0;
    my $prev_state = 0;
    my $prev_start;
    my $prev_end;
    my $not_first = 0;
    while (my $line = <>) {
	chomp $line;
	my ($chr, $start, $end, $state) = split(/\t/,$line);
    #test for noncontiguous region
    if ($increment > 0){
         if (($start - $prev_start) > 40000) {
              print $increment. "\t";
              print $start - $prev_start. "\n";
              $prev_state = 0;  #reset
              $not_first = $not_first + 1;
         }
    }
	if ($prev_state == 0) {
	    unless ($state == 3) {
		next;
	    }
	}
    #print "prev_start";
    #print $prev_start. "\n";
    #print "start";
    #print $start. "\n";
    #print $start 
	if ($state != $prev_state) {
	    if (($state == 3) && ($prev_state == 0) && ($not_first == 0)) {
		#print $chr . "\t" . $start . "\t";
	    }
        if (($state == 3) && ($prev_state == 0) && ($not_first > 0)) {
        #print "\n" . $chr . "\t" . $start . "\t";
        }   
	    if (($state == 3) && ($prev_state == 1)) {
		#print $prev_end . "\n" . $chr . "\t" . $start . "\t";
	    }

	}
	$prev_state = $state;
	$prev_start = $start;
	$prev_end = $end;
    $increment = $increment + 1
    }
    if ($prev_state == 1) {
	#print $prev_end . "\n";
    }
}
