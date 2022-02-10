/*
 * PlayerAgent.java
 * The player's "agent" on the server side. 
 * Sends messages to the player and
 * receives messages from the player.
 * However, this class also maintains
 * the time remaining to respond.
 * Created on April 18, 2006, 1:29 PM
 */

package ca.ualberta.cs.poker.free.server;
import java.net.*;
import java.io.*;
import java.util.*;


/**
 *
 * @author Martin Zinkevich
 */
public class PlayerAgent {
    /**
     * The current message terminator is CR LF (13 10).
     */
    public static final String messageTerminator = "" + ((char)13)+((char)10);
    
    /**
     * The time remaining to use (in milliseconds)
     */
    long timeRemaining;
    
    /**
     * The total (match) elapsed sending time.
     */
    public long elapsedSendingTime;
    
    /**
     * The index of the player represented by this agent.
     */
    int playerIndex;
    
    /**
     * The client socket (for sending and receiving messages)
     */
    Socket socket;
    
    /**
     * The stream for sending messages to a client.
     */
    OutputStream os;
    
    /**
     * The stream for receiving messages from a client.
     */
    InputStream is;
    
    /**
     * The bankroll of the client.
     */
    double bankroll;
    
    /**
     * The protocol for poker communication. Currently "VERSION:1.0.0".
     */
    String protocol;
    
    /**
     * The partial response received from the client.
     */
    String response;
    
    /** 
     * Creates a new instance of PlayerAgent 
     */
    public PlayerAgent(Socket socket, int playerIndex) throws SocketException, IOException{
        this.socket = socket;
        this.playerIndex = playerIndex;
        bankroll = 0;
        protocol = null;
        socket.setTcpNoDelay(true);
        os = socket.getOutputStream();
        is = socket.getInputStream();
        response = "";
    }
    
    /**
     * Reset the time at the beginning of the hand.
     */
    public void setTimeRemaining(long timeRemaining){
        this.timeRemaining = timeRemaining;
    }
    
    /**
     * Send a message to the client. Appends a message terminator.
     */
    public void sendMessage(String message) throws TimeoutException{
        //System.out.println("MessageTerminator:"+((int)(messageTerminator.charAt(0))));
        //System.out.println("MessageTerminator:"+((int)(messageTerminator.charAt(1))));
        try{
            //System.out.println("Message Length:"+message.length());
            
            message = message + messageTerminator;
            //System.out.println("Message Length:"+message.length());
            
            byte[] messageData = message.getBytes();
            long initialTime = new Date().getTime();
            long timeUsed,currentTime;
            socket.setSoTimeout(120000);
            os.write(messageData);
            currentTime = new Date().getTime();
            timeUsed = currentTime - initialTime;
            if (timeUsed>=100000){
                throw new TimeoutException(playerIndex,true);
            }
            elapsedSendingTime += timeUsed;
        } catch(SocketException to){
            throw new TimeoutException(playerIndex,true);
        } catch(IOException io){
            throw new TimeoutException(playerIndex,true);
        }
    }
    
    /**
     * Receives a message from the client. Removes the message terminator.
     */
    public String receiveMessage() throws TimeoutException{
        try{
        long initialTime = new Date().getTime();
        long timeUsed=0;
        long currentTime=initialTime;
        do{
            socket.setSoTimeout((int)(timeRemaining-timeUsed));
            
            response = response + (char)(is.read());
            currentTime = new Date().getTime();
            timeUsed = currentTime - initialTime;
            if (timeUsed>=timeRemaining){
                throw new TimeoutException(playerIndex,false);
            }
        } while(!isComplete(response));
        timeRemaining-=timeUsed;
        String result = response.substring(0,response.length()-messageTerminator.length());
        response = "";
        return result;
        } catch (SocketException to){
            throw new TimeoutException(playerIndex,false);
        } catch (IOException io){
            throw new TimeoutException(playerIndex,false);
        }
    }
    
    /**
     * A message is complete if it ends with the message terminator.
     */
    public boolean isComplete(String result){
        return result.endsWith(messageTerminator);
    }
    
    /**
     * Close the connection to the client.
     */
    public void close() throws IOException{
        os.close();
        is.close();
        socket.close();
    }
    
    /**
     * Increment the bankroll by d.
     */
    public void incrementBankroll(double d){
        bankroll+=d;
    }
}
