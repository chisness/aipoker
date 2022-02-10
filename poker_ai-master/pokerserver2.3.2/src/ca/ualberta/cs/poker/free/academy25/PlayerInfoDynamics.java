/*
 * PlayerInfoDynamics.java
 *
 * Created on April 19, 2006, 5:37 PM
 */

package ca.ualberta.cs.poker.free.academy25;
import com.biotools.meerkat.*;

/**
 *
 * @author Martin Zinkevich
 */
public class PlayerInfoDynamics implements PlayerInfo{
    /**
     * The seat of this player
     */
    int playerIndex;
    
    /**
     * The underlying game dynamics
     */
    GameInfoDynamics dynamics;
    
    /**
     * The associated GameInfo
     */
    GameInfoImpl parent;
    
    /** 
     * Creates a new instance of PlayerInfoDynamics
     * @param dynamics the underlying dynamics object
     * @param parent the associated GameInfo object
     * @param playerIndex the seat of this player
     */
    public PlayerInfoDynamics(GameInfoDynamics dynamics, GameInfoImpl parent, int playerIndex) {
        this.dynamics = dynamics;
        this.parent = parent;
        this.playerIndex = playerIndex;
    }
    
    /**
     * Gets the amount to call
     */
    public double getAmountCallable(){
        return getAmountToCall();
    }
    
    /**
     * Gets the amount in pot. 
     * Nonzero even when the game is over.
     */
    public double getAmountInPot(){
        return dynamics.inPot[playerIndex];
    }
    
    /**
     * This gets the amount in pot for the current player
     * during the current round of betting.
     */
    public double getAmountInPotThisRound(){
        if (parent.getStage()==Holdem.PREFLOP){
            return getAmountInPot();
        }
        int bets = parent.getNumRaises();
        double maxThisRound = parent.getCurrentBetSize() * bets;
        return maxThisRound - getAmountToCall();
    }
    
    /**
     * Gets the current bet size (getCurrentBetSize()).
     */
    public double getAmountRaiseable(){
        return parent.getCurrentBetSize();
    }
    
    /**
     * The amount to call (difference between the stakes and this player's pot).
     */
    public double getAmountToCall(){
        return dynamics.getAmountToCall(playerIndex);
    }
    
    
    /**
     * The bankroll of this player.
     */
    public double getBankRoll(){
        return parent.getBankRoll(playerIndex);
    }
    
    /**
     * The bankroll at risk.
     * This function was designed for no-limit games, so its results in this context
     * are a bit funny.
     */
    public double getBankRollAtRisk(){
        return parent.getBankRollAtRisk(playerIndex);
    }
    
    /**
     * The bankroll at the start of the current hand.
     */
    public double getBankRollAtStartOfHand(){
        return dynamics.bankrollAtStart[playerIndex];
    }
    
    /**
     * Returns the bankroll of this player in small bets.
     * Returns getBankRoll()/getGameInfo().getBigBlindSize()
     */
    public double getBankRollInSmallBets(){
        return getBankRoll()/parent.getBigBlindSize();
    }
    
    /**
     * Returns the associated GameInfo object
     */
    public GameInfo getGameInfo(){
        return parent;
    }
    
    /**
     * Returns the last action this player played in this hand,
     * or -1 if no such action exists.
     */
    public int getLastAction(){
        return dynamics.lastAction[playerIndex];
    }
    
    /**
     * Returns "0" or "1"
     */
    public String getName(){
        return "" + playerIndex;
    }
    
    
    
    /**
     * Gets the net gain for the hand.
     * Returns zero if the game is over.
     */
    public double getNetGain(){
        if (parent.isGameOver()){
            return 0;
        } 
        return dynamics.bankroll[playerIndex]-dynamics.bankrollAtStart[playerIndex];
    }
    
    /**
     * Returns the current bet size.
     */
    public double getRaiseAmount(double amountToRaise){
        return getAmountRaiseable();
    }
    
    public Hand getRevealedHand(){
        if (dynamics.hole[playerIndex]==null){
            return null;
        }
        return new Hand(dynamics.hole[playerIndex]);
    }
    
    public int getSeat(){
        return playerIndex;
    }
    
    public boolean hasActedThisRound(){
        return dynamics.hasActed[playerIndex];
    }
    
    /**
     * Always true (infinite bankrolls).
     */
    public boolean hasEnoughToRaise(){
        return true;
    }
    /**
     * Players are always in the game.
     */
    public boolean inGame(){
        return true;
    }
    
    /**
     * A player is active until they fold or lose a showdown.
     */
    public boolean isActive(){
        return dynamics.active[playerIndex];
    }
    
    /**
     * Players are never all in
     */
    public boolean isAllIn(){
        return false;
    }
    
    /**
     * Returns true if this player is the button
     */
    public boolean isButton(){
        return playerIndex==dynamics.button;
    }
    
    /**
     * Test for a voluntary commitment by the player.
     */
    public boolean isCommitted(){
        return parent.isCommitted(playerIndex);
    }
    
    /**
     * A player has folded if his last action was fold.
     */
    public boolean isFolded(){
        return (getLastAction()==Holdem.FOLD);
    }
    
    /**
     * Players never sit out.
     */
    public boolean isSittingOut(){
        return false;
    }
    
    /**
     * Returns the name, "0" or "1"
     */
    public String toString(){
        return getName();
    }
}
