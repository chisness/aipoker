/*
 * ClientPokerDynamics.java
 *
 * Created on April 19, 2006, 10:27 AM
 */

package ca.ualberta.cs.poker.free.client;

import ca.ualberta.cs.poker.free.dynamics.Card;
import ca.ualberta.cs.poker.free.dynamics.PokerDynamics;

/**
 *
 * Maintains a version of the server's state on the client side.
 * Keeps track of bankroll, seat, and the state of the match in a usable form.
 * @author Martin Zinkevich
 */
public class ClientPokerDynamics extends PokerDynamics{
    /**
     * The seat taken by this player.
     */
    public int seatTaken;
    /**
     * The bankroll at the beginning of the hand when (handOver==false): when (handOver==true), it becomes 
     * the bankroll at the end of the hand.
     */
    public double bankroll;
    
    /** Creates a new instance of ClientPokerDynamics */
    public ClientPokerDynamics() {
        super(null);
    }
    
    /**
     * Returns true if it is our turn to act.
     */
    public boolean isOurTurn(){
        return (!handOver)&&(seatTaken==seatToAct);
    }
    
    /**
     * Initialize this from a match state message.
     */
    public void setFromMatchStateMessage(String message){
        int messageTypeColon = message.indexOf(':');
        int seatColon = message.indexOf(':',messageTypeColon+1);
        int handNumberColon = message.indexOf(':',seatColon+1);
        int bettingSequenceColon = message.indexOf(':',handNumberColon+1);
        seatTaken = Integer.parseInt(message.substring(messageTypeColon+1,seatColon));
        handNumber = Integer.parseInt(message.substring(seatColon+1,handNumberColon));
        setCards(message.substring(bettingSequenceColon+1));
        setBettingString(message.substring(handNumberColon+1,bettingSequenceColon));
        if (handOver){
            updateBankroll();
        }
    }
    
    /**
     * Initialize the betting string.
     */
    public void setBettingString(String bettingString){
        initializeBets();
        for(int i=0;i<bettingString.length();i++){
            if (bettingString.charAt(i)!='/'){
                handleAction(bettingString.charAt(i));
            }
        }
    }
    
    /**
     * Get a card from the card sequence.
     */
    public Card getCard(String cardSequence, int currentIndex){
        return new Card(cardSequence.substring(currentIndex,currentIndex+2));
    }
    
    /**
     * Initialize all the cards from the card sequence.
     * Unknown cards are null pointers.
     */
    public void setCards(String cardSequence){
        hole = new Card[2][2];
        board = new Card[5];
        int currentIndex = 0;
        if (cardSequence.charAt(currentIndex)!='|'){
            hole[0][0]=getCard(cardSequence,currentIndex);
            currentIndex += 2;
            hole[0][1]=getCard(cardSequence,currentIndex);
            currentIndex+=2;
        }
        currentIndex++;
        if (currentIndex>=cardSequence.length()){
            return;
        }
        if (cardSequence.charAt(currentIndex)!='/'){
            hole[1][0]=getCard(cardSequence,currentIndex);
            currentIndex += 2;
            hole[1][1]=getCard(cardSequence,currentIndex);
            currentIndex+=2;
        }
        currentIndex++;
        if (currentIndex>=cardSequence.length()){
            return;
        }
        for(int i=0;i<5;i++){
          if (currentIndex>=cardSequence.length()){
              return;
          }
          if (cardSequence.charAt(currentIndex)=='/'){
              currentIndex++;
          }
          board[i]=getCard(cardSequence,currentIndex);
          currentIndex += 2;
        }
    }
    
    /**
     * Returns the betting sequence since the last cards observed.
     */
    public String getRoundBettingSequence(){
        int finalSlash = bettingSequence.lastIndexOf('/');
        if (finalSlash==-1){
            return bettingSequence;
        }
        return bettingSequence.substring(finalSlash+1);
    }
    
    /**
     * Update the bankroll: note that this is ONLY changed at the end of a hand.
     */
    public void updateBankroll(){
        bankroll += amountWon[seatTaken];
    }
}
