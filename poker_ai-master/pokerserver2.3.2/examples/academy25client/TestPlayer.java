/*
 * TestPlayer.java
 *
 * Created on April 21, 2006, 11:54 AM
 */

import com.biotools.meerkat.*;
import com.biotools.meerkat.util.*;

import java.io.*;
import java.net.*;
import ca.ualberta.cs.poker.free.academy25.PokerAcademyClient;

/**
 *
 * @author Martin Zinkevich
 */
public class TestPlayer implements Player{
    /** Creates a new instance of TestPlayer */
    GameInfo gameinfo;
    int seat;
    Card c1;
    Card c2;
    PrintStream out;

    /**
     * THIS IS THE FUNCTION WHERE THE BOT CHOOSES AN ACTION
     */
    public Action getAction(){
        if (gameinfo.canRaise(gameinfo.getCurrentPlayerSeat())){
            return Action.raiseAction(gameinfo);
        }
        return Action.callAction(gameinfo);
    }

    /**
     * Initialize this player.
     */
    public TestPlayer() {
        try{
        out = new PrintStream(new FileOutputStream("C:\\TestPlayer.log"));
        } catch (FileNotFoundException fnf){
            out = System.out;
        }
    }
        
    /**
     * Called when hole cards are dealt to this player.
     */
    public void holeCards(Card c1, Card c2, int seat){
        this.c1 = c1;
        this.c2 = c2;
        this.seat = seat;
        out.println("Card 1:"+c1);
        out.println("Card 2:"+c2);
        out.println("Seat:"+seat);
    }
    
    public void init(Preferences prefs){
        // For now, init is ***NOT CALLED***
    }

    /**
     * Called after an event is taken but before
     * it has an effect.
     */
    public void actionEvent(int pos, Action act){
    }
    
    /**
     * Called before the blinds or hole cards.
     */
    public void gameStartEvent(GameInfo info){
	gameinfo = info;
    }

    /**
     * Called when the game is over. The bankrolls have
     * been adjusted and getNetGain() is zero.
     */
    public void gameOverEvent(){
      out.println(gameinfo.getBoard());
    }

    /**
     * Called when the state of the game changes after
     * an action (blind, check, call, bet, fold).
     */
    public void gameStateChanged(){
    }
    
    /**
     * Called after someone has revealed their cards at
     * the showdown. There is no mucking in our variant
     * Hold'em.
     */
    public void showdownEvent(int pos, Card c1, Card c2){
    }

    /**
     * Called at the beginning of the game and after
     * cards have been dealt. Changes the values of 
     * currentBetSize() and related methods.
     */
    public void stageEvent(int stage){
    }
    
    /**
     * Called when a player in position pos has won. The bankroll is incremented
     * and the pot is nonzero.
     */
    public void winEvent(int pos, double amount, java.lang.String handName){
    }
    
    /**
     * The hole cards are being dealt.
     */
    public void dealHoleCardsEvent(){
    }
    
    /**
     * A function for startme.bat to call
     */
    public static void main(String[] args) throws Exception{
        TestPlayer tp = new TestPlayer();
        PokerAcademyClient pac = new PokerAcademyClient(tp);
        System.out.println("Attempting to connect to "+args[0]+" on port "+args[1]+"...");

        pac.connect(InetAddress.getByName(args[0]),Integer.parseInt(args[1]));
        System.out.println("Successful connection!");
        pac.run();

    }
}
