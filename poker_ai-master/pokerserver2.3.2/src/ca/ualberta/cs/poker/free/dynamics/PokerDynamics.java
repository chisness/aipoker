/*
 * PokerDynamics.java
 *
 * This class provides the mechanics of reverse-blinds 10-20 limit texas hold'em.
 * 
 * One unusual aspect of this implementation is that it is assumed that both
 * players have sufficient bankrolls (i.e. 24,000) such that they will never not
 * have sufficient funds to call or raise.
 *
 *
 *
 * Created on April 18, 2006, 1:45 PM
 */

package ca.ualberta.cs.poker.free.dynamics;

import java.security.SecureRandom;
import java.io.BufferedReader;
import java.io.IOException;

//import ca.ualberta.cs.poker.free.academy25.GameInfoDynamics;
//import ca.ualberta.cs.poker.free.academy25.PlayerInfoDynamics;
//import ca.ualberta.cs.poker.free.client.*;


/**
 *
 * @author Martin Zinkevich
 */
public class PokerDynamics {
    /**
     * The randomness for the game. null if on the client side.
     */
    SecureRandom random;
    
    /**
     * inPot[i] is the contribution to the pot of the 
     * player in seat i.
     */
    public double[] inPot;
    public double[] amountWon;
    public int roundBets;
    public String bettingSequence;
    public int seatToAct;
    /** Round index incremented when the cards for that round are dealt. 
     * Preflop is 0, Showdown is 4. */
    public int roundIndex;
    /** The next action will be the first action on the round. */
    public boolean firstActionOnRound;
    /** The hand is over */
    public boolean handOver;
    public int winnerIndex;
    
    /** Cards in the hole */
    public Card[][] hole;
    
    /** Full board (may not have been revealed) */
    public Card[] board;
    
    public int handNumber;
    /** Creates a new instance of PokerServer */
    public PokerDynamics(SecureRandom random) {
      handNumber = 0;
      this.random = random;
    }
    
    public PokerDynamics() {
    	handNumber = 0;
    }
    
    public void startHand(){
        initializeBets();
        dealCards();
    }
    
    /*
     * This will load the cards from a file instead of securerandom
     */
    public void startHand( BufferedReader cardFile) {
    	
    	initializeBets();
    	
    	String cards = "";
    	hole = new Card[2][2];
        board = new Card[5];
    	
        try {
    		cards = cardFile.readLine();
    	} catch (IOException e) {
    		System.err.println( "Error reading from specified card file");
    	}
    	
    	
    	if ( cards.length() != 18 ) {
    		System.err.println( "***** Wrong line length in card file");
    	}
    
    	hole[0][0]= new Card( cards.substring(0,2)); 
        hole[0][1]=new Card( cards.substring(2,4));
        hole[1][0]=new Card( cards.substring(4,6));
        hole[1][1]=new Card( cards.substring(6,8));
        
        board[0]=new Card( cards.substring(8,10));
        board[1]=new Card( cards.substring(10,12));
        board[2]=new Card( cards.substring(12,14));
        board[3]=new Card( cards.substring(14,16));
        board[4]=new Card( cards.substring(16,18));
    	
       // System.out.println( "HAND = " + cards );
    	
    	
    }
    
    /** Sets all cards from the SecureRandom device */
    public void dealCards(){
        Card[] dealt = Card.dealNewArray(random,9);
        hole = new Card[2][2];
        board = new Card[5];
        hole[0][0]=dealt[0];
        hole[0][1]=dealt[1];
        hole[1][0]=dealt[2];
        hole[1][1]=dealt[3];
        
        board[0]=dealt[4];
        board[1]=dealt[5];
        board[2]=dealt[6];
        board[3]=dealt[7];
        board[4]=dealt[8];
    }

    
    public int getOtherSeat(int seat){
        return 1-seat;
    }
    
    public void addToPot(double amount, int seat){
      inPot[seat]+=amount;
    }
    
    
    public void initializeBets(){
      bettingSequence = "";
      handOver = false;
      amountWon = null;
      roundIndex = 0;
      firstActionOnRound = true;
      inPot = new double[2];
      inPot[0] = inPot[1] = 0;
      addToPot(10.0,0);
      addToPot(5.0,1);
      roundBets = 1;
      seatToAct = 1;
    }
    
    public void incrementRound(){
        roundIndex++;
        if (roundIndex<4){
            bettingSequence += '/';
            firstActionOnRound = true;
            roundBets=0;
            seatToAct=0;
        } else {
            winnerIndex = getWinner();
            endHand();
        }
    }
    
    
    
    public String getMatchState(int seat){
    		return "MATCHSTATE:" + seat + ":" + handNumber + ":" + bettingSequence + ":" + getCardState(seat);   	
    }
    
    
    
    
    /** The first player's hole cards are visible to the first player
      * always and to everyone at the showdown. */
    public boolean isFirstSeatVisible(int seat){
        return (seat==0)||(roundIndex==4);
    }
    
    public boolean isSecondSeatVisible(int seat){
        return (seat==1)||(roundIndex==4);
    }
    
    /**
     * If seat==2, we want all the card info for the logs, this assumes
     * we will only use seat==2 when the server wants logging info, otherwise
     * 0 or 1
     * 
     * @param seat
     * @return
     */
    public String getCardState(int seat){
        String result = "";
        if (isFirstSeatVisible(seat) || seat == 2 ){
            result = result + hole[0][0]+hole[0][1];
        }
        result = result + "|";
        if (isSecondSeatVisible(seat) || seat == 2){
            result = result + hole[1][0]+hole[1][1];
        }
        if (roundIndex>0){
            result = result + "/" + board[0]+board[1]+board[2];
        }
        if (roundIndex>1){
            result = result + "/" + board[3];
        }
        if (roundIndex>2){
            result = result + "/" + board[4];
        }
        
        return result;
    }
    
    /**
     * Updates the state when a call is made.
     */
    public void handleCall(){
        bettingSequence = bettingSequence + 'c';
      int otherSeat = getOtherSeat(seatToAct);
      addToPot(inPot[otherSeat]-inPot[seatToAct], seatToAct);
      if (firstActionOnRound){
          seatToAct = otherSeat;
          firstActionOnRound = false;
      } else {
          incrementRound();
      }
    }
    
    /**
     * Updates the state when a (legal) raise is made.
     */
    public void handleRaise(){
        bettingSequence=bettingSequence + 'r';
        firstActionOnRound = false;
        int otherSeat = getOtherSeat(seatToAct);
        double betAmount = (roundIndex>=2) ? 20.0 : 10.0;
        addToPot(inPot[otherSeat]+betAmount-inPot[seatToAct],seatToAct);
        roundBets++;
        seatToAct = otherSeat;
    }
    
    /**
     * Updates the state when a (legal) fold is made.
     */
    public void handleFold(){
       bettingSequence=bettingSequence + 'f';
        firstActionOnRound = false;
       winnerIndex = getOtherSeat(seatToAct);
       endHand();
    }

    public void handleAction(char action){
        switch(action){
            case 'c':
                handleCall();
                break;
            case 'r':
                if (roundBets<4){
                    handleRaise();
                    break;
                }
                // Fall through if illegal to raise
            default:
            case 'f':
                handleFold();
        }
    }
    
    /**
     * Returns: -1 on a tie, 0 if first seat has a better hand, 1 if second seat has a better hand.
     */
    public int getWinner(){
        return HandAnalysis.determineWinner(hole,board);
    }
    
    /**
     * After winnerIndex is set, we can end the hand.
     */
    public void endHand(){
        amountWon=new double[2];
        if (winnerIndex!=-1){
            amountWon[winnerIndex]=inPot[getOtherSeat(winnerIndex)];
            amountWon[getOtherSeat(winnerIndex)]=-inPot[getOtherSeat(winnerIndex)];
        }
        handOver = true;
    }
    
    public void setHandNumber(int handNumber){
        this.handNumber = handNumber;
    }
}
