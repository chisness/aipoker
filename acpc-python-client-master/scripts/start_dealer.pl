#!/usr/bin/perl

# Copyright (C) 2011 by the Computer Poker Research Group, University of Alberta

use Socket;

$hostname = `hostname` or die "could not get hostname";
chomp $hostname;
@hostent = gethostbyname( $hostname );
$#hostent >= 4 or die "could not look up $hostname";
$hostip = inet_ntoa( $hostent[ 4 ] );

$#ARGV >= 3 or die "usage: start_dealer.pl matchName gameDefFile #Hands rngSeed player1name player2name ... [options]";

$numPlayers = -1;
open FILE, '<', $ARGV[ 1 ] or die "couldn't open game definition $ARGV[ 1 ]";
while( $_ = <FILE> ) {

    @_ = split;

    if( uc( $_[ 0 ] ) eq 'NUMPLAYERS' ) {
	$numPlayers = $_[ $#_ ];
    }
}
close FILE;

$numPlayers > 1 or die "couldn't get number of players from $ARGV[ 1 ]";


$#ARGV >= 3 + $numPlayers or die "too few arguments on command line";

pipe STDINREADPIPE, STDINWRITEPIPE or die "couldn't create stdin pipe";
pipe STDOUTREADPIPE, STDOUTWRITEPIPE or die "couldn't create stdout pipe";

$dealerPID = fork();
if( $dealerPID == 0 ) {
    # we're the child

    # replace standard in and standard out with pipe
    close STDINWRITEPIPE;
    close STDOUTREADPIPE;
    open STDIN, '<&STDINREADPIPE' or die "can't dup STDIN";
    open STDOUT, '>&STDOUTWRITEPIPE' or die "can't dup STDOUT";
    open STDERR, ">>$ARGV[ 0 ].err" or die "can't open log file $ARGV[ 0 ].err";

    @args = ( "dealer", $ARGV[ 0 ], $ARGV[ 1 ],
	      $ARGV[ 2 ], $ARGV[ 3 ] );

    # add names to the arguments
    for( $p = 0; $p < $numPlayers; ++$p ) {
	push @args, $ARGV[ 4 + $p ];
    }

    # add any extra arguments (options?) to the arguments
    for( $i = 4 + $numPlayers; $i <= $#ARGV; ++$i ) {
	push @args, $ARGV[ $i ];
    }

    exec { "./acpc_infrastructure/dealer" } @args or die "Couldn't run dealer";
}

close STDINREADPIPE;
close STDOUTWRITEPIPE;

$_ = <STDOUTREADPIPE> or die "couldn't read port description from dealer";
@_ = split;
$#_ + 1 >= $numPlayers or die "couldn't get enough ports from $_";

# print the free ports
$| = 1;
print join(" ", $_) . "\n";

waitpid( $dealerPID, 0 );

$_ or die "couldn't get values from dealer";

print $_;

exit( 0 );
