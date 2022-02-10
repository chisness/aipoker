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

int calculateState(const Game *game, const MatchState state);
void setFixedValues();
void setFamily1();
void setFamily2();
void setFamily3();
void setFamily(int familyNum);
double calculateNewAvg(double newValue, double oldAvg, int oldCount);
double defineProb(double action1, double action2);
int isFixedParameter(int currentCard, int stateNumber);
void setFixedTable();

double fixedTable [5][12];
double table[5][12];

int main( int argc, char **argv )
{
  int sock, len, r, a;
  int32_t min, max;
  uint16_t port;
  Game *game;
  MatchState state;
  Action action;
  FILE *file, *toServer, *fromServer;
  struct timeval tv;
  rng_state_t rng;
  char line[ MAX_LINE_LEN ];

  /* we make some assumptions about the actions - check them here */
  assert( NUM_ACTION_TYPES == 3 );

  if( argc < 4 ) {

    fprintf( stderr, "usage: player game server port\n" );
    exit( EXIT_FAILURE );
  }


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

  int inTrainingMode = 1;
  setFixedValues();
  setFamily(1);

  const double MIN = -99999;
  int gamesPlayed = 0;
  double total = 0;

  int states [2];
  int actions [2];
  int statesUsed = 0;
  double utilities[5][12][2];  //number of cards + 1; number of states; number of actions
  for(int i = 0; i < 5; i++)
  {
    for(int j = 0; j < 12; j++)
    {
      for(int k = 0; k < 2; k++)
      {
        utilities[i][j][k] = MIN;
      }
    }
  }

  int counts[5][12][2];
  for(int i = 0; i < 5; i++)
  {
    for(int j = 0; j < 12; j++)
    {
      for(int k = 0; k < 2; k++)
      {
        counts[i][j][k] = 0;
      }
    }
  }



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

    //exit training phase
    if(gamesPlayed >= 2000)
    {
      inTrainingMode = 0;
    }

    int currentIndex = state.viewingPlayer;
    int currentCard = rankOfCard(state.state.holeCards[currentIndex][0]) - 8; //values 1 -4
    int numberOfActions = state.state.numActions[0];

    if( stateFinished( &state.state ) ) {

      double stateValue = valueOfState(game, &state.state, currentIndex );

      total += stateValue;
      gamesPlayed++;

      //for each state in states, update utilities & count
      for(int i = 0; i < statesUsed; i++)
      {
        int stateNum = states[i];
        int action = actions[i];
        double oldAvg = utilities[currentCard][stateNum][action]; //old utility avg for given card, state,and action

        int oldCount = counts[currentCard][stateNum][action];

        double newAvg = calculateNewAvg(stateValue, oldAvg, oldCount);
        utilities[currentCard][stateNum][action] = newAvg; //set new avg
        counts[currentCard][stateNum][action] += 1; //increment count;

      }
      statesUsed = 0;  //reset for next game
      continue;
    }

    if( currentPlayer( game, &state.state ) != state.viewingPlayer ) {
      /* we're not acting */
      continue;
    }

    int stateNumber = calculateState(game, state);

    /* add a colon (guaranteed to fit because we read a new-line in fgets) */
    line[ len ] = ':';
    ++len;

    action.size = 0;

    //check utilities table, pick best option if possible
    double action1 = utilities[currentCard][stateNumber][0];
    double action2 = utilities[currentCard][stateNumber][1];
    double prob = 0;

    //found actions that we can compare. Only change parameters are not fixed.
    if(action1 != MIN && action2 != MIN && isFixedParameter(currentCard, stateNumber) == 0 && inTrainingMode == 0)
    {
      double utilProb = defineProb(action1, action2);
      double tableProb = table[currentCard][stateNumber];
      //table[currentCard][stateNumber] = (utilProb + tableProb)/2;
      //Take avg of two probabilities. Because of bad performance, we are not using the result from our learning approach
    }

    prob = table[currentCard][stateNumber];


    int max;
    int min;

    //pick action based on random number
    double randomNumber = genrand_real1( &rng );
    if(raiseIsValid(game, &state.state, &max, &min ) != 0)
    {
      if(randomNumber < prob)
      {
        action.type = a_raise;
        actions[statesUsed] = 0;
      }
      else
      {
        action.type = a_call;
        actions[statesUsed] = 1;
      }
    }
    else
    {
      if(randomNumber < prob)
      {
        action.type = a_call;
        actions[statesUsed] = 0;
      }
      else
      {
        action.type = a_fold;
        actions[statesUsed] = 1;
      }
    }

    //store states and action for player
    states[statesUsed] = stateNumber;

    statesUsed++;


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


  int calculateState(const Game *game, const MatchState state)
  {

    int currentIndex = state.viewingPlayer;
    int currentCard = rankOfCard(state.state.holeCards[currentIndex][0]) - 8; //values 1 -4
    int numberOfActions = state.state.numActions[0];


    switch(currentIndex)
    {
      case 0:
      if(numberOfActions == 0)
      {
        return 0;
      }
      else
      {
        //no need to check 1st action
        //check 2nd action
        if(state.state.action[0][1].type == a_call)
        {
          return 3;
        }
        else
        {
          //check 3rd action
          if(state.state.action[0][2].type == a_fold)
          {
            return 6;
          }
          else
          {
            return 9;
          }
        }
      }
      break;
      case 1:
      if(numberOfActions == 1)
      {
        //check 1st action
        if(state.state.action[0][0].type == a_call)
        {
          return 1;
        }
        else
        {
          return 4;
        }
      }
      else //number of actions = 4
      {
        //check 4th action
        if(state.state.action[0][3].type == a_fold)
        {
          return 7;
        }
        else  //call
        {
          return 10;
        }
      }
      break;
      case 2:
      //check 1st action
      if(state.state.action[0][0].type == a_call)
      {
        //check 2nd action
        if(state.state.action[0][1].type == a_call)
        {
          return 2;
        }
        else
        {
          return 5;
        }
      }
      else  //1st action = raise
      {
        //check 2nd action
        if(state.state.action[0][1].type == a_fold)
        {
          return 8;
        }
        else
        {
          return 11;
        }
      }
      break;
    }

    return -1;

  }

  void setFixedValues()
  {
    //total 18 values
    //player 1   - 9
    table[1][0] = 0;
    table[2][0] = 0;
    table[2][3] = 0;
    table[2][6] = 0;
    table[3][0] = 0;
    table[3][3] = 0;
    table[3][6] = 0.5;
    table[3][9] = 0;
    table[4][0] = 0;

    //player 2 - 3
    table[2][4] = 0;
    table[3][1] = 0;
    table[3][10] = 0;

    //player 3  - 6
    table[2][5] = 0;
    table[2][8] = 0;
    table[3][2] = 0;
    table[3][5] = 0;
    table[4][2] = 1;
    table[3][11] = 0.5;

    //set 21 necessary strategy parameters
    for(int i = 3; i < 12; i++)
    {
      table[1][i] = 0;
    }
    for(int i = 9; i < 12; i++)
    {
      table[2][i] = 0;
    }
    for(int i = 3; i < 12; i++)
    {
      table[4][i] = 1;
    }
  }

  void setFamily1()
  {
    table[1][2] = 0; //c11 = 0

    //4 independent variables
    table[2][1] = 0.25; //# b21 <= 0.25
    table[1][1] = 0.25; // # b11 <= b21;
    table[3][4] = (2 + 3*table[1][1] + 4*table[2][1])/4;  //# b32 <= (2 + 3*b11 + 4*b21 / 4
      table[3][8] = 0; //  # c33 = 0

      // dependent variables
      table[2][7] = 0; //  # b23
      table[3][7] = (1 + table[1][1] + 2*table[2][1])/2; //  # b33 = (1  + b11 + 2*b21)/2
      table[4][1] = 2 * table[1][1]+ 2 * table[2][1];; //  # b41 = (2*b11 + 2*b21)
      table[2][2] = 0.5 ; // # c21 = 0.5
    }

    void setFamily2()
    {
      table[1][2] = 0.25;  // c11 = 0.25

      //3 independent variables
      table[1][1] = 0;  // b11 <= 1/4;
      table[3][4] = (2 + 7 * table[1][1]) / 4;  // b32 <= (2+ 7*b11)/4
      table[3][8] = 0;  //c33 = 0

      //dependent variables

      table[2][1] = table[1][1];  // b21 = b11
      table[2][7] = 0;  // b23
      table[3][7] = (1 + 3 * table[1][1]) / 2 ; // b33 = (1 + 3 * b11)/2
      table[4][1] = 4 * table[1][1];  // b41 = (4*b11)
      table[2][2] = 0.5 - table[1][2];  // c21 = 0.5 - c11
    }

    void setFamily3()
    {

      table[1][2] = 0.5;  // c11 = 0.5

      // 5 independent variables

      table[1][1] = 1.0/6;  // b11 <= 1/4;
      table[2][1] = table[1][1] ; // b21 <= b11
      table[2][7] = (table[1][1] - table[2][1]) / 2 * (1- table[2][1]);  // b23 <= (b11 - b21)/ 2*(1-b21)

      table[3][4] = (2 + 4 * table[1][1] + 3 * table[2][1]) / 4 ; // b32
      table[3][8] = 0.5 - table[3][4] + (4*table[1][1] + 3*table[2][1])/4; // c33 = 0.5 - b32 + (4b11 + 3b21)/4

      // dependent variables

      table[3][7] = (1 + table[1][1] + 2*table[2][1]) / 2;  // b33 = (1 + b11 + 2b21)/2
      table[4][1] = 2 * table[1][1] + 2 * table[2][1];  // b41 = (2*b11 + 2*b21)
      table[2][2] = 0;  // c21 = 0

    }

    void setFamily(int familyNum)
    {
      switch(familyNum)
      {
        case 1:
        setFamily1();
        break;
        case 2:
        setFamily2();
        break;
        case 3:
        setFamily3();
        break;
        default:
        setFamily1();
      }
    }

    double calculateNewAvg(double newValue, double oldAvg, int oldCount)
    {
      double total = (oldAvg * oldCount) + newValue;
      return total / (oldCount + 1);
    }

    double defineProb(double action1, double action2)
    {
      if(action1 > action2)
      {
        return 1;
      }
      else
      {
        return 0;
      }


      if(action1 <= 0 && action2 > 0)
      {
        return 0;
      }
      else if(action2 <= 0 && action1 > 0)
      {
        return 1;
      }
      else if(action1 < 0 && action2 < 0)
      {
        action1 *= -1;
        action2 *= -1;

        return (action2)/(action1 + action2); //pick smaller

      }
      else
      {
        return (action1)/(action1 + action2);
      }
    }


    void setFixedTable(int currentCard, int stateNumber)
    {

      for(int i = 0; i < 5; i ++)
      {
        for(int j = 0; j < 12; j++)
        {
          fixedTable[i][j] = -1;
        }
      }

      //total 18 values
      //player 1   - 9
      fixedTable[1][0] = 0;
      fixedTable[2][0] = 0;
      fixedTable[2][3] = 0;
      fixedTable[2][6] = 0;
      fixedTable[3][0] = 0;
      fixedTable[3][3] = 0;
      fixedTable[3][6] = 0.5;
      fixedTable[3][9] = 0;
      fixedTable[4][0] = 0;

      //player 2 - 3
      fixedTable[2][4] = 0;
      fixedTable[3][1] = 0;
      fixedTable[3][10] = 0;

      //player 3  - 6
      fixedTable[2][5] = 0;
      fixedTable[2][8] = 0;
      fixedTable[3][2] = 0;
      fixedTable[3][5] = 0;
      fixedTable[4][2] = 1;
      fixedTable[3][11] = 0.5;

      //set 21 necessary strategy parameters
      for(int i = 3; i < 12; i++)
      {
        fixedTable[1][i] = 0;
      }
      for(int i = 9; i < 12; i++)
      {
        fixedTable[2][i] = 0;
      }
      for(int i = 3; i < 12; i++)
      {
        fixedTable[4][i] = 1;
      }

    }

    //check if current state is fixed
    int isFixedParameter(int currentCard, int stateNumber)
    {
      int notFixed = 0;
      int fixed = 1;

      if(fixedTable[currentCard][stateNumber] == -1)
      {
        return notFixed;
      }
      else
      {
        return fixed;
      }
    }
