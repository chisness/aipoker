/*
 * TimeoutException.java
 *
 * Created on April 18, 2006, 4:40 PM
 */

package ca.ualberta.cs.poker.free.server;

/**
 * This exception is a little more general than its name implies: any socket or
 * i/o exception during transfer results in a timeout exception.
 *
 * @author Martin Zinkevich
 */
public class TimeoutException extends Exception {
    /**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	/**
     * The index of the player that times out.
     */
    public int playerIndex;
    /**
     * A serious timeout exception is caused by a message that fails to send.
     * Because a failure to send might not be representable by any standard signal,
     * the player forfeits the remainder of his blinds.
     */
    public boolean serious;
    
    /**
     * A short description of the error
     */
    public String description;
    
    /** 
     * Creates a new instance of TimeoutException 
     */
    public TimeoutException(int playerIndex, boolean serious, String description) {
        super("Player timed out:"+playerIndex+((serious)? ":send": ":receive") + ":"+description);
        this.playerIndex = playerIndex;
        this.serious = serious;
        this.description = description;
    }
    public TimeoutException(int playerIndex, boolean serious) {
    	this(playerIndex,serious,"");
    }
    
}
