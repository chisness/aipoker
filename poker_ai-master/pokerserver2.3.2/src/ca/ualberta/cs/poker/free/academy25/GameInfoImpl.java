/*
 * GameInfoImpl.java
 * Designed to work with the Meerkat 2.5 API interface.
 *
 * NOTES:
 * Stage events FOLLOW cards being dealt.
 * Action events PRECEDE actions being taken.
 * 
 * http://www.poker-academy.com/downloads/Meerkat-API-2.5.zip 
 *
 * Created on April 19, 2006, 10:18 AM
 */


package ca.ualberta.cs.poker.free.academy25;

//import ca.ualberta.cs.poker.free.client.ClientPokerDynamics;
import com.biotools.meerkat.*;
import java.util.*;

/**
 *
 * @author Martin Zinkevich
 */
public class GameInfoImpl implements GameInfo{
    private GameInfoDynamics dynamics;
    public PlayerInfoDynamics[] players;
    
    public GameInfoImpl(GameInfoDynamics dynamics){
        this.dynamics = dynamics;
        players = new PlayerInfoDynamics[2];
        players[0] = new PlayerInfoDynamics(dynamics,this,0);
        players[1] = new PlayerInfoDynamics(dynamics,this,1);
    }
    
    

    
    /**
     * Since names are simply a string with the player index,
     * it is all good.
     */
    public static int nameToSeat(String name){
        return Integer.parseInt(name);
    }
  

  
    /**
     * Gets the net gain 
     * Returns zero if the game is over.
     */
    public double getNetGain(int seat){
        if (isGameOver()){
            return 0;
        } 
        return dynamics.bankroll[seat]-dynamics.bankrollAtStart[seat];
    }
    

    /**
     * You can raise if there are less than 4 bets on this round and the number of active players
     * is 2 (everybody).
     */
  public boolean canRaise(int seat){
    return (dynamics.roundBets<4)&&(getNumActivePlayers()==2);
  }

  /**
   * The amount to call is the different between the stake and the pot of seat: however,
   * if someone has folded (is inactive), it is zero.
   */
  public double getAmountToCall(int seat){
    return (getNumActivePlayers()==2) ? dynamics.getAmountToCall(seat) : 0.0;
  }

  /**
   * There are no antes in the variant we are playing, so this returns zero.
   */
  public double getAnte(){
    return 0;
  }
  
  /**
   * Returns the current bankroll of a player.
   * A player's bankroll is decremented with each blind, call, or raise it makes, and is incremented
   * right before its own winEvent()
   */
  public double getBankRoll(int seat){
      return dynamics.bankroll[seat];
  }

  /**
   * This function is primarily for no-limit games, and therefore there is no formal specification here
   * and its use is not recommended.
   */
  public double getBankRollAtRisk(int seat){
      if (dynamics.active[dynamics.getOtherSeat(seat)]){
          return Math.min(getAmountToCall(seat)+getBankRoll(dynamics.getOtherSeat(seat)),getBankRoll(seat));
      } else {
          return 0;
      }
  }


  /**
   * The amount to call, except in terms of the current bet size.
   */
  public double getBetsToCall(int seat){
    return getAmountToCall(seat)/getCurrentBetSize();
  }

  /**
   * The seat that has or will submit the big blind.
   * This is non-button seat.
   */
  public int getBigBlindSeat(){
     return dynamics.getOtherSeat(dynamics.button);
  }

  /**
   * The size of the big blind, equal to the size of the small bet.
   */
  public double getBigBlindSize(){
    return GameInfoDynamics.smallBet;
  }

  /**
   * Returns the visible board. Updated before each stageEvent.
   */
  public Hand getBoard(){
    return new Hand(dynamics.board);
  }
  
  /**
   * Gets the seat with the button.
   * This seat gives the small blind, bets first on the pre-flop,
   * and bets second in later rounds.
   */
  public int getButtonSeat(){
     return dynamics.button;
  }

  
  /**
   * If the current round is the pre-flop or the flop, this is the small bet.
   * If the current round is turn or the river, this is the river.
   */
  public double getCurrentBetSize(){
    return (isTurn()||isRiver()) ? getBigBlindSize() * 2.0 : getBigBlindSize();
  }

  /**
   * Returns the player who is about to act (during getAction) or who has just acted (during actionEvent or gameStateChanged)
   */
  public int getCurrentPlayerSeat(){
    return dynamics.currentPlayerSeat;
  }

  /**
   * Returns the total pot size (infinite bankroll)
   */
  public double getEligiblePot(int seat){
    return getTotalPotSize();
  }

  /**
   * Returns a long between 0 and 999
   */
  public long getGameID(){
    return dynamics.gameID;
  }

  /**
   * There is no log directory in this implementation.
   */
  public String getLogDirectory(){
      throw new RuntimeException("Not implemented.");
  }

  /**
   * Same as the total pot size.
   */
  public double getMainPotSize(){
    return getTotalPotSize();
  }

  /**
   * Same as the current bet size.
   */
  public double getMinRaise(){
    return getCurrentBetSize();
  }

  public int getNumActivePlayers(){
      int result = (dynamics.active[0]) ? 1 : 0;
      if (dynamics.active[1]){
          result++;
      }
    return result;
  }

  public int getNumActivePlayersNotAllIn(){
    return getNumActivePlayers();
  }

  public int getNumberOfAllInPlayers(){
    return 0;
  }

  public int getNumPlayers(){
    return 2;
  }

  public int getNumRaises(){
    return dynamics.roundBets;
  }

  public int getNumSeats(){
    return 2;
  }

  public int getNumSidePots(){
      return 0;
  }

  public int getNumToAct(){
      return dynamics.numToAct;
  }

  public int getNumWinners(){
    return dynamics.numWinners;
  }

  /**
   * Returns the player info for a player 
   * sitting in the seat.
   */
  public PlayerInfo getPlayer(int seat){
      return players[seat];
  }

  /**
   * Returns the player info for a player 
   * with a particular name (names are "0" and "1").
   */
  public PlayerInfo getPlayer(String name){
      return players[nameToSeat(name)];
  }

  
  /**
   * Returns the name for a player 
   * in a seat (names are "0" and "1").
   */
  public String getPlayerName(int seat){
      return ""+seat;
  }

  public int getPlayerSeat(String name){
      return nameToSeat(name);
  }

  /**
   * A PlayerInfo is in the list iff they put amountIn or more into the pot themselves.
   */
  public List getPlayersInPot(double amountIn){
      List<PlayerInfoDynamics> result = new Vector<PlayerInfoDynamics>();
      if (dynamics.inPot[0]>=amountIn){
        result.add(players[0]);
      }
      if (dynamics.inPot[1]>=amountIn){
          result.add(players[1]);
      }
      return result;
  }

  /**
   * No rake for the competition
   */
  public double getRake(){
    return 0;
  }

  /**
   * @note this method appeats to sometimes 
   * have a side effect (of increasing the number of side pots) in poker academy
   * This side effect is ignored here (it always returns zero).
   */
  public double getSidePotSize(int i){
      return 0;
  }

  /**
   * Gets the small blind seat (the button).
   * 
   * In reverse blinds, the button gets the smallBlind.
   */
  public int getSmallBlindSeat(){
    return dynamics.button;
  }

  /**
   * Returns the small blind size.
   * getSmallBlindSize() * 4 == getBigBlindSize() * 2 == getSmallBetSize() * 2 == getBigBetSize()
   *
   */
  public double getSmallBlindSize(){
    return getBigBlindSize()/2.0;
  }

  /**
   * Returns the stage: Holdem.PREFLOP, Holdem.FLOP, Holdem.TURN, or Holdem.RIVER (0-3)
   */
  public int getStage(){
      return dynamics.stage;
  }

  /**
   * The largest amount any individual put in the pot this hand.
   */
  public double getStakes(){
    return Math.max(dynamics.inPot[0],dynamics.inPot[1]);
  }

  /**
   * The total pot size. All other pot functions call this one.
   */
  public double getTotalPotSize(){
    return dynamics.inPot[0]+dynamics.inPot[1];
  }

  /**
   * How many players have not folded, checked, called, bet, or raised this stage?
   */
  public int getUnacted(){
      return dynamics.numUnacted;
  }
  
  /**
   * Both players are always in the game
   */
  public boolean inGame(int seat){
    return true;
  }

  /**
   * All players are active in heads-up
   */
  public boolean isActive(int seat){
    return true;
  }

  /**
   * Has the player bet more than his own blind in this stage?
   */
  public boolean isCommitted(int seat){
      if (getStage()==Holdem.PREFLOP){
      double blind = (seat==getSmallBlindSeat()) ? getSmallBlindSize() : getBigBlindSize();
      return (dynamics.inPot[seat]>blind);
      }
      return (players[seat].getAmountInPotThisRound()>0);
  }

  /**
   * The tournament is fixed limit
   */
  public boolean isFixedLimit(){
    return true;
  }

  /**
   * Flop indicates the postflop
   */
  public boolean isFlop(){
    return getStage()==Holdem.FLOP;
  }

  public boolean isGameOver(){
    return dynamics.gameOver;
  }

  /**
   * The tournament is fixed limit, not "no limit"
   */
  public boolean isNoLimit(){
    return false;
  }

  /**
   * Returns true if the flop has been dealt but not the turn card (after stageEvent(1), before stageEvent(2))
   */
  public boolean isPostFlop(){
    return isFlop();
  }
  
  /**
   * The tournament is fixed limit, not "pot limit"
   */
  public boolean isPotLimit(){
    return false;
  }

  /**
   * Returns true if the river card has been dealt (after stageEvent(1), before stageEvent(2))
   */
  public boolean isPreFlop(){
    return getStage()==Holdem.PREFLOP;
  }

  /**
   * The tournament is reverse blinds, where the button gives a small blind.
   */
  public boolean isReverseBlinds(){
    return true;
  }

  /**
   * Returns true if the river card has been dealt (after stageEvent(3))
   */
  public boolean isRiver(){
    return getStage()==Holdem.RIVER;
  }

  /**
   * The tournament is not simulation.
   */
  public boolean isSimulation(){
    return false;
  }

  public boolean isTurn(){
    return getStage()==Holdem.TURN;
  }

  /**
   * The tournament is not "zip mode".
   */
  public boolean isZipMode(){
    return false;
  }

  /**
   * The next active player.
   */
  public int nextActivePlayer(int seat){
    int otherSeatIndex = dynamics.getOtherSeat(seat);
    if (dynamics.active[otherSeatIndex]){
        return dynamics.getOtherSeat(seat);
    }
    return seat;
  }

  /**
   * Return the other seat.
   */
  public int nextPlayer(int seat){
      return dynamics.getOtherSeat(seat);
  }

  /**
   * Since there are only two seats, the next seat from 1 is 0.
   */
  public int nextSeat(int seat){
    return dynamics.getOtherSeat(seat);
  }

  /**
   * The previous player is the other player.
   */
  public int previousPlayer(int seat){
    return dynamics.getOtherSeat(seat);
  }
  
  /**
   * Convert old cards to new cards.
   * @todo: eliminate
   */
  public static Hand convertToHand(ca.ualberta.cs.poker.free.dynamics.Card[] cards){
      Hand result = new Hand();
      for(int i=0;i<cards.length;i++){
          com.biotools.meerkat.Card c = new com.biotools.meerkat.Card(cards[i].rank.index,cards[i].suit.index);
          result.addCard(c);
      }
      return result;
  }
}

