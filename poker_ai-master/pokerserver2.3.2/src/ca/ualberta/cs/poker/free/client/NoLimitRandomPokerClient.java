/*
 * RandomPokerClient.java
 *
 * Created on April 19, 2006, 2:04 PM
 */

package ca.ualberta.cs.poker.free.client;
import java.io.*;
import java.net.*;
import java.security.*;

/**
 * Plays actions uniformly at random. Useful for debugging purposes.
 * 
 * @author Martin Zinkevich
 */
public class NoLimitRandomPokerClient extends PokerClient {
    SecureRandom random;
    
    /**
     * Chooses an action uniformly at random using an internal secure random number generator.
     */
    public void handleStateChange() throws IOException, SocketException{
        double dart = random.nextDouble();
        if (dart<0.05){
            sendFold();
        } else if (dart<0.55){
            sendCall();
        } else {
        	// A lot of these raises WILL be illegal, but the server should intercept
        	// them.
        	int finalPot = random.nextInt(1001);
            sendRaise(finalPot);
        }
    }
    
    /** 
     * Creates a new instance of RandomPokerClient 
     */
    public NoLimitRandomPokerClient(){
      super();
      random = new SecureRandom();  
    }
    
    /**
     * @param args the command line parameters (IP and port)
     */
    public static void main(String[] args) throws Exception{
        NoLimitRandomPokerClient rpc = new NoLimitRandomPokerClient();
        System.out.println("Attempting to connect to "+args[0]+" on port "+args[1]+"...");

        rpc.connect(InetAddress.getByName(args[0]),Integer.parseInt(args[1]));
        System.out.println("Successful connection!");
        rpc.run();
    }
    
}
