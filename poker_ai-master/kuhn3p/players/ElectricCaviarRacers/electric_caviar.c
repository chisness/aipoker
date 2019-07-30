/*
Copyright (C) 2011 by the Computer Poker Research Group, University of Alberta
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <getopt.h>
#include "game.h"
#include "rng.h"
#include "net.h"

#define MAX( a, b ) ( ( a > b) ? a : b ) 
#define MIN( a, b ) ( ( a < b) ? a : b ) 

int takeAction(rng_state_t *rng, double actionProb){
    /* choose one of the valid actions at random */
    double p = genrand_real2( rng );
    if( p <= actionProb ) {
        return 1;
    }
    return 0;
}

int main( int argc, char **argv )
{
    //int j_counter = 0;
    //int q_counter = 0; 
    //int k_counter = 0; 
    //int a_counter = 0;
    
    int sock, len, r, a, i, card, player;//
    uint8_t numActions;//
    uint8_t round;//
    int32_t min, max;
    uint16_t port;
    double p;
    Game *game;
    MatchState state;
    Action action;
    FILE *file, *toServer, *fromServer;
    struct timeval tv;
    double probs[ NUM_ACTION_TYPES ];
    double A[5][5];//
    double B[5][5];//
    double C[5][5];//
    double opp_c11, opp_b11, opp_b21, opp_b23, opp_b32;  
    double init_opp_c11, init_opp_b11, init_opp_b21, init_opp_b23, init_opp_b32; 
    int right_opp_earnings = 0;
    int left_opp_earnings = 0; 

    double actionProbs[ NUM_ACTION_TYPES ];
    rng_state_t rng;
    char line[ MAX_LINE_LEN ];
  
    /* we make some assumptions about the actions - check them here */
    assert( NUM_ACTION_TYPES == 3 );

    //start with randomized guess values for our opponents move frequency.
    init_opp_c11 = genrand_real2( &rng ) * 0.5;//opp_c11 will most likely be between 0 and 1/2 considering smart opponent.
    init_opp_b21 = genrand_real2( &rng ) * 0.25;//opp_b21 will most likely be between 0 and 1/4 if opponent is smart.
    init_opp_b11 = genrand_real2( &rng ) * init_opp_b21;//opp_b11 is assumed to be less than opp_b21 as logic indicates.
    init_opp_b23 = genrand_real2( &rng ) * 0.25;//opp_b23: 0 - 1/4 is a good starting point.
    init_opp_b32 = 0.60 + (0.20 * genrand_real2( &rng ));//0.60-0.80 is a good starting point for opp_b32

    opp_c11 = init_opp_c11;
    opp_b21 = init_opp_b21;
    opp_b11 = init_opp_b11;
    opp_b23 = init_opp_b23;
    opp_b32 = init_opp_b32;

    C[1][1] = opp_c11;//Initiate our c11 probability in the same range as opponent.
    B[2][1] = opp_b21;//Initiate our b21 probability in the same range as opponent.
    B[1][1] = opp_b11;//Initiate our b11 probability in the same range as opponent.
    B[2][3] = opp_b23;//Initiate our b23 probability in the same range as opponent.
    B[3][2] = opp_b32;//Initiate our b32 probability in the same range as opponent.

    double c11_Possible = 0.0;
    double c11_Actual = 0.0;
    double b11_Possible = 0.0;
    double b11_Actual = 0.0;
    double b21_Possible = 0.0;
    double b21_Actual = 0.0;
    double b23_Possible = 0.0;
    double b23_Actual = 0.0;
    double b32_Possible = 0.0;
    double b32_Actual = 0.0;
            
    if( argc < 4 ) {
    	fprintf( stderr, "usage: player game server port\n" );
    	exit( EXIT_FAILURE );
    }

    /* Define the probabilities of actions for the player */
    //fold   = 0.06
    //call    = 0.47
    //raise  = 0.47
    probs[ a_fold ] = 0.06;
    probs[ a_call ] = ( 1.0 - probs[ a_fold ] ) * 0.5;
    probs[ a_raise ] = ( 1.0 - probs[ a_fold ] ) * 0.5;

    /* Initialize the player's random number state using time */
    gettimeofday( &tv, NULL );
    init_genrand( &rng, tv.tv_usec );

    /* get the game */
    file = fopen( argv[ 1 ], "r" );
    if( file == NULL ) {

    fprintf( stderr, "ERROR: could not open game %s\n", argv[ 1 ] );
    exit( EXIT_FAILURE );
    }
    game = readGame( file );
    if( game == NULL ) {

    fprintf( stderr, "ERROR: could not read game %s\n", argv[ 1 ] );
    exit( EXIT_FAILURE );
    }
    fclose( file );

    /* connect to the dealer */
    if( sscanf( argv[ 3 ], "%"SCNu16, &port ) < 1 ) {

    fprintf( stderr, "ERROR: invalid port %s\n", argv[ 3 ] );
    exit( EXIT_FAILURE );
    }
    sock = connectTo( argv[ 2 ], port );
    if( sock < 0 ) {

    exit( EXIT_FAILURE );
    }
    toServer = fdopen( sock, "w" );
    fromServer = fdopen( sock, "r" );
    if( toServer == NULL || fromServer == NULL ) {

    fprintf( stderr, "ERROR: could not get socket streams\n" );
    exit( EXIT_FAILURE );
    }

    /* send version string to dealer */
    if( fprintf( toServer, "VERSION:%"PRIu32".%"PRIu32".%"PRIu32"\n",
           VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION ) != 14 ) {

    fprintf( stderr, "ERROR: could not get send version to server\n" );
    exit( EXIT_FAILURE );
    }
    fflush( toServer );

    /* play the game! */
    while( fgets( line, MAX_LINE_LEN, fromServer ) ) {
    
        /* ignore comments */
        if( line[ 0 ] == '#' || line[ 0 ] == ';' ) {
            continue;
        }

        len = readMatchState( line, game, &state );
        if( len < 0 ) {

          fprintf( stderr, "ERROR: could not read state %s", line );
          exit( EXIT_FAILURE );
        }
        
        /* add a colon (guaranteed to fit because we read a new-line in fgets) */
        line[ len ] = ':';
        ++len;
        
        /*Get card*/
        switch(state.state.holeCards[state.viewingPlayer][0]) {

            case 39:
                card = 1;
                break;
            
            case 43:
                card = 2;
                break; 
            
            case 47:
                card = 3;
                break;
                
            case 51:
                card = 4;
                break;
                
            default : 
                card = -1;
        }
        
        /*Get actions*/
        player = state.viewingPlayer;
        round = state.state.round;
        numActions = state.state.numActions[round];
        uint8_t actions[numActions];
        //printf("!player:%d | handid:%d | actions:%d | card:%d!\t",player,state.state.handId, numActions, card);
        if (numActions > 0 && numActions < MAX_NUM_ACTIONS){
            for(i = 0 ; i < numActions; i++){
                actions[i] = state.state.action[round][i].type;
                //printf("%d", actions[i]);
            }
        }
        //printf("\n");
   

        if( stateFinished( &state.state ) ) {
            
	   /*if (card == 1){
		j_counter++;
	    }
            if (card == 2){
		q_counter++;
	    }
            if (card == 3){
		k_counter++;
	    } 
            if (card == 4){
		a_counter++;
	    }   

	    printf("As: %d | Ks: %d | Qs: %d | Js: %d\n", a_counter, k_counter, q_counter, j_counter);  

	    printf("Player 0 has: %d\n", state.state.holeCards[0][0]);
            printf("Player 1 has: %d\n", state.state.holeCards[1][0]);
            printf("Player 2 has: %d\n", state.state.holeCards[2][0]);
	    printf("Round: %d", state.state.handId);

	    int player0_result = (int)valueOfState( game, &state.state, 0 );
            int player1_result = (int)valueOfState( game, &state.state, 1 );
            int player2_result = (int)valueOfState( game, &state.state, 2 );
            
            printf("Result: %d | %d | %d\n", player0_result, player1_result, player2_result); */
            
	    //Update opponents earnings.
            if(player == 0){
                right_opp_earnings = right_opp_earnings + (int)valueOfState( game, &state.state, 1 );
                left_opp_earnings = left_opp_earnings + (int)valueOfState( game, &state.state, 2 );
            }
            if(player == 1){
                right_opp_earnings = right_opp_earnings + (int)valueOfState( game, &state.state, 2 );
                left_opp_earnings = left_opp_earnings + (int)valueOfState( game, &state.state, 0 );
            }
            if(player == 2){
                right_opp_earnings = right_opp_earnings + (int)valueOfState( game, &state.state, 0 );
                left_opp_earnings = left_opp_earnings + (int)valueOfState( game, &state.state, 1 );
            }

            /*Learning*/
            if(player != 2){
                if(actions[0] == a_call && actions[1] == a_call){
                    if(state.state.holeCards[2][0] == 39)
                        c11_Possible++;
                        if(actions[2] == a_raise)
                            c11_Actual++;

                        opp_c11 = ((init_opp_c11 * 10.0) + c11_Actual) / (10.0 + c11_Possible);
                }
            }
            //opp_b11
            if(player != 1){
                if(actions[0] == a_call){
                    if(state.state.holeCards[1][0] == 39)
                        b11_Possible++;
                    if(actions[1] == a_raise)
                        b11_Actual++;

                    opp_b11 = ((init_opp_b11 * 10.0) + b11_Actual) / (10.0 + b11_Possible);
                }
            }
            //opp_b21
            if(player != 1){
                    if(actions[0] == a_call){
                        if(state.state.holeCards[1][0] == 43)
                            b21_Possible++;
                        if(actions[1] == a_raise)
                            b21_Actual++;

                        opp_b21 = ((init_opp_b21 * 10.0) + b21_Actual) / (10.0 + b21_Possible);
                }
            }
            //opp_b23
            if(player != 1){
                if(numActions > 3 && actions[3] == a_fold){
                    if(state.state.holeCards[1][0] == 43)
                        b23_Possible++;
                    if(actions[4] == a_call)
                        b23_Actual++;

                    opp_b23 = ((init_opp_b23 * 10.0) + b23_Actual) / (10.0 + b23_Possible);
                }
            }
            //opp_b32
            if(player != 1){
                if(actions[0] == a_raise){
                    if(state.state.holeCards[1][0] == 47)
                        b32_Possible++;
                    if(actions[1] == a_call)
                        b32_Actual++;

                    opp_b32 = ((init_opp_b32 * 10.0) + b32_Actual) / (10.0 + b32_Possible);
                }
            }

	    
            
            continue;
        }

        if( currentPlayer( game, &state.state ) != state.viewingPlayer ) {
            /* we're not acting */

            continue;
        }

        /* build the set of valid actions */
        p = 0;
        for( a = 0; a < NUM_ACTION_TYPES; ++a ) {

          actionProbs[ a ] = 0.0;
        }

        /* consider fold */
        action.type = a_fold;
        action.size = 0;
        if( isValidAction( game, &state.state, 0, &action ) ) {

          actionProbs[ a_fold ] = probs[ a_fold ];
          p += probs[ a_fold ];
        }

        /* consider call */
        action.type = a_call;
        action.size = 0;
        actionProbs[ a_call ] = probs[ a_call ];
        p += probs[ a_call ];

        /* consider raise */
        if( raiseIsValid( game, &state.state, &min, &max ) ) {

          actionProbs[ a_raise ] = probs[ a_raise ];
          p += probs[ a_raise ];
        }
        /* normalise the probabilities  */
        assert( p > 0.0 );
        for( a = 0; a < NUM_ACTION_TYPES; ++a ) {

          actionProbs[ a ] /= p;
        }

        
        /* choose one of the valid actions at random */
        /*p = genrand_real2( &rng );
        for( a = 0; a < NUM_ACTION_TYPES - 1; ++a ) {

          if( p <= actionProbs[ a ] ) {

            break;
          }
          p -= actionProbs[ a ];
        }
        action.type = (enum ActionType)a;*/
        
        
        
        if( a == a_raise ) {

            action.size = min + genrand_int32( &rng ) % ( max - min + 1 );
        }
        
	/*Equilibria*/
        //EQ0 - Table 2
        if(card == 1 && ((numActions > 0  && actions[numActions-1] == a_raise)|| (numActions > 1  &&actions[numActions-2] == a_raise))){
            action.type = a_fold;
        }
        if(card == 4 && ((numActions > 0  && actions[numActions-1] == a_raise)|| (numActions > 1  &&actions[numActions-2] == a_raise))){
            action.type = a_call;
        }
        if(card ==2 && numActions > 1 &&  actions[numActions-2] == a_raise && actions[numActions-1] == a_call)
             action.type = a_fold;

        //Table 3
        //EQ1
        if(card == 1){
            //P1
            if(numActions == 0){
                action.type = a_call;
            }
            //P2
            if(numActions == 1 && actions[0] == a_call){
                if(opp_c11 <= 0.01){
                    //B[1][1] = genrand_real2( &rng ) * B[2][1];//b11<=b21
                    //Let's play smart
                    if(left_opp_earnings >= right_opp_earnings){
                        B[1][1] = 0.25;
                    }
                    else{
                        B[1][1] = 0.0;
                    }		    
           
                    if(takeAction(&rng, B[1][1]))
                        action.type = a_raise;
                    else
                        action.type = a_call;
                }
                else{
                        //B[1][1] = genrand_real2( &rng ) * 0.25;//b11 <= 1/4
                    if(left_opp_earnings >= right_opp_earnings){
                        B[1][1] = 0.25;
                    }
                    else{
                        B[1][1] = 0.0;
                    }	
                    if(takeAction(&rng, B[1][1]))
                        action.type = a_raise;
                    else
                        action.type = a_call;
                }
            }
            //P3
            if(numActions == 2 && actions[0] == a_call && actions[1] == a_call){
                
                double cthresh = MIN(0.5,((2-opp_b11)/(3+2*opp_b11+2*opp_b21)));
                C[1][1] = genrand_real2( &rng ) * cthresh;
                if(takeAction(&rng, C[1][1]))
                    action.type = a_raise;
                else
                    action.type = a_call;
            }
        }
        
        //EQ2
        if(card == 2 && actionProbs[ a_fold ] == 0){
            //P1
            if(numActions == 0){
                action.type = a_call;//a21=0
            }
            //P2
            if(numActions == 1 && actions[0] == a_call){
                
                if(opp_c11 < 0.01){
                    //B[2][1] = genrand_real2( &rng ) * 0.25;//b12<=1/4

                    if(left_opp_earnings >= right_opp_earnings){
                        B[2][1] = 0.25;
                    }
                    else{
                        B[2][1] = 0.0;
                    }	

                    if(takeAction(&rng, B[2][1]))
                        action.type = a_raise;
                    else
                        action.type = a_call;
                }
                else if(opp_c11 > 0 && opp_c11 < 0.5){
                    //B[2][1] = B[1][1];//b21=b11
                    if(left_opp_earnings >= right_opp_earnings){
                        B[2][1] = 0.25;
                    }
                    else{
                        B[2][1] = 0.0;
                    }	
                
                    if(takeAction(&rng, B[2][1]))
                        action.type = a_raise;
                    else
                        action.type = a_call;
                }
                else if(opp_c11 > (0.5-0.01) && opp_c11 < (0.5+0.01)){
                    double bthresh = MIN(B[1][1], (0.5 - 2*B[1][1]));
                    B[2][1] = genrand_real2( &rng ) * bthresh;//b21 <= min(b11,1/2-2b11)
                    if(takeAction(&rng, B[2][1]))
                        action.type = a_raise;
                    else
                        action.type = a_call;
                }
            }
            //P3
            if(numActions == 2 && actions[0] == a_call && actions[1] == a_call){
                
                C[2][1] = 0.5 - C[1][1];//1/2 - c11
                if(takeAction(&rng, C[2][1]))
                    action.type = a_raise;
                else
                    action.type = a_call;
            }
        }
        
        //EQ3
        if(card == 2 && numActions > 1 && actions[numActions-1] == a_raise){
            //P2
            if(numActions == 1)
                action.type = a_fold;//a22=0
            //P3
            if(numActions == 2 && actions[0] == a_call)
                action.type = a_fold;//b22=0
            //P1
            if(numActions == 3 && actions[0] == a_call && actions[1] == a_call)
                action.type = a_fold;//c22=0
        }

        //EQ4
        if(card == 2 && numActions > 1 && actions[numActions-1] == a_fold){
            //P1
            if(numActions == 3 && actions[1] == a_raise){
                action.type = a_fold;//a23=0
            }
            //P2
            if(numActions == 4 && actions[2] == a_raise){
                double bthresh = MAX(0, ((B[1][1] -B[2][1])/(2-2*B[2][1])));
                B[2][3] = genrand_real2( &rng ) * bthresh;//b23 <= max(0,(b11-b21)/2(1-b21))
                if(takeAction(&rng, B[2][3]))
                        action.type = a_call;
                    else
                        action.type = a_fold;
            }
            //P3
            if(numActions == 2){
                action.type = a_fold;//c23=0
            }
        }
        
        //EQ5
        if(card == 3 && actionProbs[ a_fold ] == 0){
            action.type = a_call;
        }
        
        //EQ6
        if(card == 3 && numActions > 1 && actions[numActions-1] == a_raise){
            //P1
            if(numActions == 3)
                action.type = a_fold;
            //P2
            if(numActions == 1) {
                double bthresh = 0.5 + 0.75*(B[1][1] + B[2][1]) + (MAX(B[1][1], B[2][1])/4);
                B[3][2] = genrand_real2( &rng ) * bthresh;
                if(takeAction(&rng, B[3][2]))
                    action.type = a_call;
                else
                    action.type = a_fold;
            }
            //P3
            if(numActions == 2)
                action.type = a_fold;
        }

        //EQ7
        if(card == 3 && numActions > 1 && actions[numActions-1] == a_fold){
            //P1
            if(numActions == 3)
                A[3][3] = 0.5;
            if(takeAction(&rng, A[3][3]))
                action.type = a_call;
            else
            action.type = a_fold;	
            //P2
            if(numActions == 4) {
                double bthresh = 0.5 + 0.5*(B[1][1] + B[2][1]) + (MAX(B[1][1], B[2][1])/2) - B[2][3]*(1-B[2][1]);
                B[3][3] = genrand_real2( &rng ) * bthresh;
                if(takeAction(&rng, B[3][3]))
                    action.type = a_call;
                else
                    action.type = a_fold;
            }
            //P3
            if(numActions == 2) {
                double cthresh_min = 0.5 - opp_b32;
                double cthresh_max = 0.5 - opp_b32 + 0.75*(opp_b11 + opp_b21) + (MAX(opp_b11, opp_b21)/4);
                C[3][3] = cthresh_min + genrand_real2( &rng )*(cthresh_max - cthresh_min);
                if(takeAction(&rng, C[3][3]))
                    action.type = a_call;
                else
                    action.type = a_fold;
            }
        }
        //EQ8
        if(card == 3 && numActions > 1 && actions[numActions-1] == a_call && actions[numActions-2] == a_raise){
            //P1
            if(numActions == 3){
                action.type = a_fold;//a34=0
            }
            //P2
            if(numActions == 4){
                action.type = a_fold;//b34=0
            }
            //P3
            if(numActions == 2){
                C[3][4] = genrand_real2( &rng );
                if(takeAction(&rng, C[3][4]))
                    action.type = a_call;
                else
                    action.type = a_fold;
            }
        }
        
        //EQ9
        if(card == 4 && actionProbs[ a_fold ] == 0){
            //P1
            if(numActions == 0){
                action.type = a_call;//a41=0
            }
            //P2
            if(numActions == 1&& actions[0] == a_call){
                B[4][1]=2*B[1][1] + 2*B[2][1];
                if(takeAction(&rng, B[4][1]))
                    action.type = a_raise;
                else
                    action.type = a_call;
            }
            //P3
            if(numActions == 2 && actions[0] == a_call && actions[1] == a_call){
                action.type = a_raise;//c41
            }
        }
        
        
        /* do the action! */
        assert( isValidAction( game, &state.state, 0, &action ) );
        r = printAction( game, &action, MAX_LINE_LEN - len - 2,
                 &line[ len ] );
        if( r < 0 ) {

          fprintf( stderr, "ERROR: line too long after printing action\n" );
          exit( EXIT_FAILURE );
        }
        len += r;
        line[ len ] = '\r';
        ++len;
        line[ len ] = '\n';
        ++len;

        if( fwrite( line, 1, len, toServer ) != len ) {

          fprintf( stderr, "ERROR: could not get send response to server\n" );
          exit( EXIT_FAILURE );
        }
        fflush( toServer );
    }

  return EXIT_SUCCESS;
}