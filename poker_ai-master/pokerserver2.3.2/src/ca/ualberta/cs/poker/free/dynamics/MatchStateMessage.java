/*
 * GameStateMessage.java
 *
 *
 * Created on April 20, 2006, 3:33 PM
 */

package ca.ualberta.cs.poker.free.dynamics;

/**
 *
 * @author Martin Zinkevich
 */
public class MatchStateMessage {
    
    /**
     * The seat taken by the player who receives the message.
     */
    public int seatTaken;
    
    /**
     * The hand number, from 0-999.
     */
    public int handNumber;
    
    /**
     * Contains the hole cards, indexed by seat.
     * This player's cards are in hole[seatTaken]
     */
    public String []hole;
    
    /**
     * Contains the flop cards.
     */
    public String flop;
    
    /**
     * Contains the turn card.
     */
    public String turn;
    /**
     * Contains the river card.
     */
    public String river;
    /**
     * Contains all of the cards on the board.
     */
    public String board;
    
    public String bettingSequence;
    
    public MatchStateMessage(String message){
        int messageTypeColon = message.indexOf(':');
        int seatColon = message.indexOf(':',messageTypeColon+1);
        int handNumberColon = message.indexOf(':',seatColon+1);
        int bettingSequenceColon = message.indexOf(':',handNumberColon+1);
        seatTaken = Integer.parseInt(message.substring(messageTypeColon+1,seatColon));
        handNumber = Integer.parseInt(message.substring(seatColon+1,handNumberColon));
        bettingSequence = message.substring(handNumberColon+1,bettingSequenceColon);
        setCards(message.substring(bettingSequenceColon+1));
    }
    
    /**
     * Tests if this is the end of a stage.
     * Note: this returns false at the showdown.
     */
    public boolean endOfStage(){
        if (bettingSequence.length()==0){
            return false;
        }
        char lastChar = bettingSequence.charAt(bettingSequence.length()-1);
        return lastChar == '/';
    }
    

    
    public int getLastAction(){
        if (bettingSequence.length()==0){
            return -1;
        }
        char lastChar = bettingSequence.charAt((endOfStage()) ? (bettingSequence.length()-2) : (bettingSequence.length()-1));
        switch(lastChar){
            case 'f':
                return 0;
            case 'c':
                return 1;
            case 'r':
                return 2;
            default:
                throw new RuntimeException("Unexpected character in bettingSequence");
        }
    }

    public void setCards(String cardSequence){
        hole = new String[2];
        
        int currentIndex = 0;
        if (cardSequence.charAt(currentIndex)!='|'){
            hole[0]=cardSequence.substring(currentIndex,currentIndex+4);
            currentIndex += 4;
        }
        currentIndex++;
        if (currentIndex>=cardSequence.length()){
            board = "";
            return;
        }
        if (cardSequence.charAt(currentIndex)!='/'){
            hole[1]=cardSequence.substring(currentIndex,currentIndex+4);
            currentIndex += 4;
        }
        currentIndex++;
        if (currentIndex>=cardSequence.length()){
            board="";
            return;
        }
        flop = cardSequence.substring(currentIndex,currentIndex+6);
        currentIndex+=7;
        if (currentIndex>=cardSequence.length()){
            board = flop;
            return;
        }
        turn = cardSequence.substring(currentIndex,currentIndex+2);
        currentIndex+=3;
        if (currentIndex>=cardSequence.length()){
            board = flop + turn;
            return;
        }
        river = cardSequence.substring(currentIndex);
        board = flop + turn + river;
    }
}
