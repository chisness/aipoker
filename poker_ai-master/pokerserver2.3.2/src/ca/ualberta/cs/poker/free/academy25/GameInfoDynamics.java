/*
 * GameInfoDynamics.java
 *
 * This class is designed to handle the state changes as they are observed
 * by a bot playing Poker Academy.
 *
 * Created on April 23, 2006, 6:32 PM
 */

package ca.ualberta.cs.poker.free.academy25;

import com.biotools.meerkat.*;

/**
 *
 * @author Martin Zinkevich
 */
public class GameInfoDynamics {
  /**
   * The button is the player who gives the small blind and acts first on the pre-flop.
   */
  public int button;
  
  /**
   * The gameID is the hand number
   */
  public long gameID;
  
  
  
  public int stage;
  /**
   * NOTE: currentPlayerSeat MUST be changed manually:
   * 1. Before the big blind.
   * 3. after a bet/raise/big blind, then a gameStateChanged event.
   * 4. Before a win event.
   */
  public int currentPlayerSeat;
  
  /**
   * NOTE: the board MUST be changed manually
   * Cards are dealt immediately before a stage event
   */
  Hand board;
  
  /**
   * The hole cards for each player
   */
  Hand[] hole;
  
  public int roundBets;
  /** Game is over right before the gameOverEvent is called  */
  public boolean gameOver;
  /** folding (or losing) makes a player inactive */
  public boolean[] active; 
 /** lastAction[i] last fold(0), call(1), or raise(2) made by the player in seat i, or -1 otherwise. */
  public int[] lastAction; 
  /** The pot: incremented before gameStateChanged, set to zero for a new game */
  public double[] inPot;
  /** The bankroll: decremented before gameStateChanged, incremented before winEvent */
  public double[] bankroll;
  /** The bankroll at the start of the current hand */
  public double[] bankrollAtStart;
  public boolean[] hasActed;
  /** The number of players that have not folded, checked, called, bet, or raised this round. */
  public int numUnacted;
  /** Two at the beginning of the stage, (before first bet or check) Zero at end of stage (after last call), one elsewhere */
  public int numToAct;
  /** One after one person wins, two after two people win */
  public int numWinners;
  public static final double smallBet = 10.0; 

  
  public GameInfoDynamics(){
    double million = 1000000;
    bankroll = new double[2];
    bankroll[0]=million;
    bankroll[1]=million;
    bankrollAtStart=new double[2];
    bankrollAtStart[0]=million;
    bankrollAtStart[1]=million;
    hasActed = new boolean[2];
    doNewGame(0,0);
  }
  
  public void doNewGame(long gameID, int button){
      bankrollAtStart[0]=bankroll[0];
      bankrollAtStart[1]=bankroll[1];
      this.gameID = gameID;
	this.button = button;
        gameOver = false;
	inPot = new double[2];
	inPot[0]=0;
	inPot[1]=0;
        hole = new Hand[2];
	lastAction = new int[2];
	lastAction[0] = -1;
	lastAction[1] = -1;
        active = new boolean[2];
        active[0] = true;
        active[1] = true;
	board = new Hand();
	hole = new Hand[2];
	currentPlayerSeat = button;
        hasActed[0] = false;
        hasActed[1] = false;
        numWinners = 0;
        numToAct = 2;
        numUnacted = 2;
        currentPlayerSeat = button;
        roundBets = 0;
        stage = 0;
  }
  
  public int getOtherSeat(int seat){
      return 1-seat;
  }
  
  public void addToPot(int seat, double amount){
	inPot[seat]+=amount;
	bankroll[seat]-=amount;
  }
  
  public double getAmountToCall(int seat){
	return Math.max(inPot[getOtherSeat(seat)]-inPot[seat],0);
  }
  
  
  public double getAmountToCall(){
	return getAmountToCall(currentPlayerSeat);
  }
  public double getAmountToBet(){
	return ((stage==Holdem.TURN) || (stage==Holdem.RIVER)) ? (smallBet * 2.0) : smallBet;
  }
  
  
  /**
   * Called in-between the action event and the state change
   * The round bets is incremented during the small blind.
   * Does not change the current seat
   */
  public void doPostSmallBlind(){
      roundBets++;
	addToPot(button,smallBet * 0.5);	
  }
  

  /**
   * Called in-between the action event and the state change
   * Does not change the current seat
   */
  public void doPostBigBlind(){
    addToPot(getOtherSeat(button),smallBet);
  }
  
  /**
   * Called in-between the action event and the state change
   * Does not change the current seat
   */
  public void doPostCheckOrCall(){
    lastAction[currentPlayerSeat]=Holdem.CALL;
    addToPot(currentPlayerSeat,getAmountToCall(currentPlayerSeat));
    numToAct--;
    hasActed[currentPlayerSeat]=true;
    if (numUnacted>0){
        numUnacted--;
    }
    
  }
  
  /**
   * Called in-between the action event and the state change
   * Does not change the current seat
   */
  public void doPostBetOrRaise(){
    lastAction[currentPlayerSeat]=Holdem.RAISE;
    roundBets++;
    addToPot(currentPlayerSeat,getAmountToCall(currentPlayerSeat)+getAmountToBet());
    hasActed[currentPlayerSeat]=true;
    if (numToAct>1){
        numToAct--;
    }
    if (numUnacted>0){
        numUnacted--;
    }
  }

  /**
   * Called in-between the action event and the state change
   * Does not change the current seat
   */
  public void doPostFold(){
    lastAction[currentPlayerSeat]=Holdem.FOLD;
    hasActed[currentPlayerSeat]=true;
    numToAct=0;
      if (numUnacted>0){
          numUnacted--;
      }
      active[currentPlayerSeat]=false;
  }
  
  /**
   * This function is for winning the whole pot
   */
  public void doPreWinEvent(int seat){
      bankroll[seat]+=(inPot[0]+inPot[1]);
      currentPlayerSeat = seat;
      numWinners=1;
  }
  
  /**
   * This function is for winning the whole pot
   */
  public void doPreTieEvent(int seat){
      bankroll[seat]+=(inPot[0]+inPot[1])/2.0;
      currentPlayerSeat = seat;
      numWinners++;
  }
  
  
  
  /**
   * Call this before gameOverEvent()
   */
  public void doPreGameOver(){
      gameOver = true;
  }
  
  /**
   * On a new stage, the non-button player begins on all but the first round.
   */
  public void doPreStageEvent(int stage){
      hasActed[0]=false;
      hasActed[1]=false;
      numToAct = 2;
      numUnacted = 2;
      this.stage = stage;
      if (stage==0){
          currentPlayerSeat = button;
      } else {
          currentPlayerSeat = getOtherSeat(button);
      }
      roundBets = 0;
  }
  
  
  /**
   * On a new stage, the non-button player begins on all but the first round.
   */
  public void setBoard(String cards){
      
      board=PokerAcademyClient.getHand(cards);
  }
  /**
   * Flipping the currentPlayerSeat.
   */
  public void changeCurrentSeat(){
      currentPlayerSeat = getOtherSeat(currentPlayerSeat);
  }
  
}
