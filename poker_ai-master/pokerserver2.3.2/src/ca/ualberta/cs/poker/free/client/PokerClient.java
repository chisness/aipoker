/*
 * PokerClient.java
 *
 * Created on April 18, 2006, 10:26 PM
 */

package ca.ualberta.cs.poker.free.client;
import ca.ualberta.cs.poker.free.server.PlayerAgent;
import java.net.*;
import java.io.*;

/**
 * Root for all example Java implementations of the client interface.
 * Basic functionality: sends a version statement, receives state messages and
 * remembers them, can send replies (actions), and exits on an ENDGAME message
 * from the server.
 *
 * The function to overload is handleStateChange().
 * sendRaise(), sendCall(), and sendFold() can be used to send actions to the server.
 *
 * currentGameStateString has the most recent state information.
 * @author Martin Zinkevich
 */
public class PokerClient implements Runnable{
    /**
     * Socket connecting to the server.
     */
    Socket socket;
    /**
     * Stream from the server.
     */
    InputStream is;
    /**
     * Stream to the server.
     */
    OutputStream os;
    /**
     * Has an ENDGAME signal been received?
     */
    boolean matchOver;
    
    /**
     * Whether the client is verbose (prints messages sent and received to stdout).
     */
    boolean verbose;
    
    /**
     * Sets the verbose flag (if true, prints messages sent and received to stdout).
     */
    public void setVerbose(boolean verbose){
        this.verbose = verbose;
    }
    
    /**
     * Returns the IP address and port number of the client.
     */
    public String getClientID(){
        return ""+socket.getLocalAddress()+":"+socket.getLocalPort();
    }
    
    /**
     * Print a message to stdout if verbose==true.
     */
    public void showVerbose(String message){
        if (verbose){
            System.out.println(message);
        }
    }
    
    /**
     * This is the current game state.
     * It is not changed during a call to handleStateChange()
     */
    public String currentGameStateString;

    /**
     * Override to handle a state change.
     * Observe that a state change does NOT imply
     * it is your turn.
     */
    public void handleStateChange() throws IOException, SocketException{
        sendCall();
    }
    
    /** Creates a new instance of PokerClient. Need to connect(), then run() to start process. */
    public PokerClient(){
        verbose = false;
    }

    /**
     * Connects to the server at the given IP address and port number.
     */
    public void connect(InetAddress iaddr, int port) throws IOException, SocketException{
        socket = new Socket(iaddr,port);
        is = socket.getInputStream();
        os = socket.getOutputStream();
        matchOver = false;
        sendMessage("VERSION:1.0.0");
    }
    
    
    /**
     * Send an action (action should be r, c, or f).
     * Usually called during handleStateChange. 
     * Action will be in response to
     * the state in currentGameStateString.
     */
    public void sendAction(char action) throws IOException, SocketException{
        sendAction(""+action);
    	//sendMessage(currentGameStateString+":"+action);
    }
    

    /**
     * Send an action string (action should be r??, c, or f, where ?? is the final amount in the pot from
     * a player in chips).
     * Usually called during handleStateChange. 
     * Action will be in response to
     * the state in currentGameStateString.
     */
    public void sendAction(String action) throws IOException, SocketException{
        sendMessage(currentGameStateString+":"+action);
    }
    
    /**
     * send a raise action.
     */
    public void sendRaise() throws IOException, SocketException{
        sendAction('r');
    }
    

    /**
     * send a raise action. The final in pot is the total YOU want to have
     * put in the pot after the raise (ie including previous amounts from
     * raises, calls, and blinds.
     */
    public void sendRaise(int finalInPot) throws IOException, SocketException{
        sendAction("r"+finalInPot);
    }
    
    /**
     * send a call action.
     */
    public void sendCall() throws IOException, SocketException{
        sendAction('c');
    }
    
    /**
     * send a fold action.
     */
    public void sendFold() throws IOException, SocketException{
        sendAction('f');
    }
    
    /**
     * Start the client. Should call connect() before running.
     */
    public void run() {
        try{
            while(true){
                String message = receiveMessage();
                if (message.startsWith("MATCHSTATE:")){
                    currentGameStateString = message;
                    handleStateChange();
                } else if (message.equals("ENDGAME")){
                    break;
                }
            }
            close();
        } catch (Exception e){
            e.printStackTrace();
            System.err.println(e);
            System.exit(0);
        }
    }
    
    /**
     * Close the connection. Called in response to an ENDGAME
     * message from the server.
     */
    public synchronized void close() throws IOException{
            matchOver=true;
            try{
            Thread.sleep(1000);
            } catch (InterruptedException ie){
            }
            os.close();
            is.close();
            socket.close();
    }
    
    /**
     * Receive a message from the server. Removes a message terminator (period).
     */
    public String receiveMessage() throws SocketException, IOException{
        String response = "";
        do{
            char c = (char)(is.read());
            //System.out.println("READ:"+(int)c);
            response = response + c;
            
        } while(!isComplete(response));
        response = response.substring(0,response.length()-PlayerAgent.messageTerminator.length());
        showVerbose("CLIENT RECEIVES:"+response);
        return response;
    }
    
    /**
     * Test if the message is complete (contains a terminal character)
     */
    public boolean isComplete(String result){
        return result.endsWith(PlayerAgent.messageTerminator);
    }
    
    /**
     * Send a message to the server. Appends a message terminator (period).
     */
    public synchronized void sendMessage(String message) throws SocketException,IOException{
        showVerbose("CLIENT SENDS:"+message);
        message = message + PlayerAgent.messageTerminator;
        byte[] messageData = message.getBytes();
        if (!matchOver){
            os.write(messageData);
        }
    }
}
