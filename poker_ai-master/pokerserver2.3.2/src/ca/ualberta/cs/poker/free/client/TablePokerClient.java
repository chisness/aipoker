/*
 * TablePokerClient.java
 *
 * Created on April 19, 2006, 1:37 PM
 */

package ca.ualberta.cs.poker.free.client;
import java.util.*;
import java.security.*;

/**
 * Stores a map from abstract states (betting histories + abstract cards) to probability triples (fold,call,raise).
 *
 * Maps the cards to a bucket.
 * 
 * @author Martin Zinkevich
 */
public class TablePokerClient extends AdvancedPokerClient{
    Hashtable<String,Vector<Double> > table;
    SecureRandom random;
    public String getBucket(){
        return "0/0/0/0";
    }
    
    /** 
     * Creates a new instance of TablePokerClient 
     */
    public TablePokerClient(){
        table = new Hashtable<String,Vector<Double> >();
        random = new SecureRandom();
    }
    
    /**
     * Take an action according to the table.
     * Maps the cards to a bucket.
     */
    public void takeAction(){
        try{
        String stateString = ""+state.seatTaken+":"+state.bettingSequence+":"+getBucket();
        Vector<Double> distribution = table.get(stateString);
        if (distribution==null){
            sendCall();
            return;
        }
        double dart = random.nextDouble();
        if (distribution.get(0)>dart){
            sendFold();
        } else if (distribution.get(0)+distribution.get(1)>dart){
            sendCall();
        } else {
            sendRaise();
        }
        } catch (Exception e){
            System.out.println(e);
        }
    }
}
