package ca.ualberta.cs.poker.free.ringseat;


/**
 * WARNING: players are both zero indexed and one indexed!
 * Should be zero indexed, with -1 indicating no player.
 */
public class RingSeat{
  int[][] players;
  int numPlayers;
  int numGames;
  /** True if the game already has the player */
  boolean[][] gameHasPlayer;
  boolean verbose = false;
  boolean talkingThrough = false;


  int[][] choices;
  int numChoices;
  int[] unforcedChoices;
  int numUnforcedChoices;

  /** seatPairHasPlayerPair[i][j[k][l] is true if seat i has player
   * k at the same time as seat k has player l (and i less than j)
   */
  boolean[][][][] seatPairHasPlayerPair;
  public RingSeat(int numPlayers){
    this.numPlayers = numPlayers;
    this.numGames = numPlayers * (numPlayers-1);
    players = new int[numGames][numPlayers];
    gameHasPlayer = new boolean[numGames][numPlayers];
    seatPairHasPlayerPair = new
    boolean[numPlayers][numPlayers][numPlayers][numPlayers];

    choices = new int[numPlayers * numPlayers * (numPlayers-1)][3];
    unforcedChoices = new int[choices.length];
    numChoices = 0;
    /// Initialize seats so no one is placed.
    for(int game=0;game<numGames;game++){
      for(int seat=0;seat<numPlayers;seat++){
        players[game][seat]=-1;
      }
    }
    numUnforcedChoices = 0;
  }

  /**
   * If we just placed a person at a row and a table,
   * test if this violates anything
   */
  public boolean satisfied(int game, int seat, int player){
    /*if (player<0){
      throw new RuntimeException("Error in satistifed: game="+game
      +" seat=" + seat+ " player="+player);
    }*/
    if (gameHasPlayer[game][player]){
      return false;
    }
    for(int i=0;i<seat;i++){
      int otherPlayer = players[game][i];
      if (otherPlayer!=-1){
        if (seatPairHasPlayerPair[i][seat][otherPlayer][player]){
	  return false;
	}
      }
    }
    
    for(int i=seat+1;i<numPlayers;i++){
      int otherPlayer = players[game][i];
      if (otherPlayer!=-1){
        if (seatPairHasPlayerPair[seat][i][player][otherPlayer]){
	  return false;
	}
      }
    }
    return true;

  }
  /**
   * Put a player in a seat and set the relevant variables.
   * SHOULD NOT be called unless player CAN sit in the seat.
   */
  public void seatPlayer(int game, int seat, int player){
    players[game][seat]=player;
    gameHasPlayer[game][player]=true;
    for(int i=0;i<seat;i++){
      int otherPlayer = players[game][i];
      if (otherPlayer!=-1){
        seatPairHasPlayerPair[i][seat][otherPlayer][player]=true;
      }
    }
    
    for(int i=seat+1;i<numPlayers;i++){
      int otherPlayer = players[game][i];
      if (otherPlayer!=-1){
        seatPairHasPlayerPair[seat][i][player][otherPlayer]=true;
      }
    }
  } 
  
  public void unseatPlayer(int game, int seat, int player){
    //System.out.println("Unseating "+game+"," + seat + "," + player);
    players[game][seat]=-1;
    gameHasPlayer[game][player]=false;
    for(int i=0;i<seat;i++){
      int otherPlayer = players[game][i];
      if (otherPlayer!=-1){
        seatPairHasPlayerPair[i][seat][otherPlayer][player]=false;
      }
    }
    
    for(int i=seat+1;i<numPlayers;i++){
      int otherPlayer = players[game][i];
      if (otherPlayer!=-1){
        seatPairHasPlayerPair[seat][i][player][otherPlayer]=false;
      }
    }
  } 

  public int getDegree(int game, int seat){
    int validPlayers=0;
    for(int player=0;player<numPlayers;player++){
      if (satisfied(game,seat,player)){
        validPlayers++;
      }
    }
    return validPlayers;
  }


  public int getNextValidPlayer(int game, int seat, int player){
    for(player++;player<numPlayers;player++){
      if (satisfied(game,seat,player)){
        return player;
      }
    }
    return -1;
  }
  
  public int getFirstValidPlayer(int game, int seat){
    for(int player=0;player<numPlayers;player++){
      if (satisfied(game,seat,player)){
        return player;
      }
    }
    throw new RuntimeException("No valid player here");
  }

  public void makeChoice(int game, int seat, int player, boolean forced){
    seatPlayer(game,seat,player);
    choices[numChoices][0]=game;
    choices[numChoices][1]=seat;
    choices[numChoices][2]=player;
    if (!forced){
      unforcedChoices[numUnforcedChoices]=numChoices;
      numUnforcedChoices++;
    }
    numChoices++;
  }

  /**
   * If possible, finds a forced choice.
   * If impossible, makes the unforced choice of lowest degree.
   * If there is an impossible choice, returns false.
   */
  public boolean makeChoice(){
    // Basically, infinity
    int minDegree = numPlayers+1;
    int bestSeat = 0;
    int bestGame = 0;
    for(int seat=0;seat<numPlayers;seat++){
      for(int game=0;game<numGames;game++){
        if (players[game][seat]==-1){
          int currentDegree=getDegree(game,seat);
  	  if (minDegree>currentDegree){
	    if (currentDegree==0){
	      return false;
	    }
	    minDegree = currentDegree;
	    bestGame = game;
	    bestSeat = seat;
	  }
	}
      }
    }
    int bestPlayer = getFirstValidPlayer(bestGame,bestSeat);
    boolean forced = (minDegree==1);
    makeChoice(bestGame, bestSeat, bestPlayer, forced);
    return true;
  }

  /**
   * Change the last unforced choice and any preceeding choices.
   */
  public boolean changeUnforcedChoice(){
    if (numUnforcedChoices==0){
      return false;
    }
    int lastUnforcedChoice = unforcedChoices[numUnforcedChoices-1];
    int lastGame = choices[lastUnforcedChoice][0];
    int lastSeat = choices[lastUnforcedChoice][1];
    int lastPlayer = choices[lastUnforcedChoice][2];
    //System.out.println("Last unforced choice:"+lastGame+","
    //+lastSeat+","+lastPlayer);
    for(int i=numChoices-1;i>=lastUnforcedChoice;i--){
      unseatPlayer(choices[i][0],choices[i][1],choices[i][2]);
    }
    numChoices = lastUnforcedChoice;
    numUnforcedChoices--;
    int nextPlayer = getNextValidPlayer(lastGame,lastSeat,lastPlayer);
    if (nextPlayer==-1){
      // This choice was really forced.
      return changeUnforcedChoice();
    } else {
      makeChoice(lastGame,lastSeat,nextPlayer,false);
    }
    return true;
  }

  public boolean findSolution(){
    while(true){
      if (talkingThrough){
        System.out.println(this);
      }
      if (makeChoice()){
        //System.out.println("Choice made");
        if (numChoices==choices.length){
	  //System.out.println("Found solution!");
          return true;
        }
      } else {
        //System.out.println("No choice found");
        if (!changeUnforcedChoice()){
	  //System.out.println("Failed to find anything");
	  return false;
	}
      }
    }
    //return true;
  }

  public String toString(){
    String result = "";
    for(int game=0;game<numGames;game++){
      for(int seat=0;seat<numPlayers;seat++){
        result+=players[game][seat]+" ";
      }
      result+="\n";
    }
    if (verbose){
    for(int seat=0;seat<numPlayers;seat++){
      for(int seatB=0;seatB<numPlayers;seatB++){
        for(int player=0;player<numPlayers;player++){
	  for(int playerB=0;playerB<numPlayers;playerB++){
	    if (seatPairHasPlayerPair[seat][seatB][player][playerB]){
	      result+=("("+player+","+playerB+") in ("+
	      seat+","+seatB+")\n");
	    }
	  }
	}
      }
    }

    for(int c=0;c<numChoices;c++){
      result+=(""+c+
      " g:"+ choices[c][0]+
      " s:"+ choices[c][1]+
      " p:"+ choices[c][2]+"\n");
    }

    result+="Unforced:";
    for(int uc=0;uc<numUnforcedChoices;uc++){
      result+=" "+unforcedChoices[uc];
    }
    result+="\n";
    }
    return result;
  }

  public void breakSymmetry(){
    int game = 0;
    for(int i=0;i<numPlayers;i++){
      for(int j=0;j<numPlayers;j++){
        if (i==j){
	  continue;
	}
	makeChoice(game,0,i,true);
	makeChoice(game,1,j,true);
	game++;
      }
    }
    for(int i=2;i<numPlayers;i++){
      makeChoice(0,i,i,true);
    }
  }

  public static void main(String[] args){
    int numSeats = 3;
    if (args.length>0){
      numSeats=Integer.parseInt(args[0]);
      if (numSeats<0){
        System.err.println("Cannot have a negative number of seats");
	System.exit(-1);
      }
    }
    RingSeat seats = new RingSeat(numSeats);
    seats.breakSymmetry();
    System.out.println("Symmetry broken");
    System.out.println(seats);
    if (seats.findSolution()){
      System.out.println(seats);
    } else {
      System.out.println("Failed to find a solution");
    }
  }

}

