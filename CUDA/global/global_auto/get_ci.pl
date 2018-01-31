#!/usr/bin/perl -w

use strict;

use Statistics::PointEstimation;

sub getCI {
  my $data = shift;
  my $sign = shift;
  my $pe = new Statistics::PointEstimation;
  $pe->add_data(@{$data});
  $pe->set_significance($sign);

  return ($pe->mean(), $pe->delta() || 0);
}


my %runtimes;
while(my $line = <>) {
  if($line =~ /(.+) Time is (.+) s/) {
    push(@{$runtimes{$1}}, $2);
  }
}

foreach my $param (sort keys  %runtimes) {
  my ($mean, $delta) = getCI(\@{$runtimes{$param}}, 99);

  print "$param,$mean,$delta\n";
}


