/*
 * PokerServer.java
 *
 * This class implements the socket aspects of the server, relying on
 * the superclass PokerDynamics to handle the rules of the game.
 * Created on April 18, 2006, 10:59 AM
 */

package ca.ualberta.cs.poker.free.server;
import java.security.SecureRandom;
import java.net.*;
import java.io.*;
import java.util.Date;


import ca.ualberta.cs.poker.free.server.PlayerAgent;
import ca.ualberta.cs.poker.free.dynamics.PokerDynamics;
import java.text.*;

/**
 *
 * @author Martin Zinkevich
 */
public class PokerServer extends PokerDynamics implements Runnable{
    /**
     * The socket clients connect to.
     */
    public ServerSocket socket;
    /**
     * The server-side representatives of the players.
     */
    public PlayerAgent[] players;
    
    /**
     * The IP address of the first player
     */
    InetAddress firstPlayerAddress;
    
    /**
     * The IP address of the second player
     */
    InetAddress secondPlayerAddress;
    
    /**
     * Are the seats flipped?
     * True when the first player is the button.
     */
    boolean seatsFlipped;
    
    /**
     * The amount of milliseconds per hand that a player can
     * use to respond to messages.
     */
    int timePerHand;
    
    /**
     * If true, messages sent and received are sent 
     * to standard output.
     */
    boolean verbose;
    
    /**
     * BufferedWriter for the log file entries
     */
    BufferedWriter logFile;
    
    /**
     * This will be the file to read the cards from
     */
    BufferedReader cardFile;
    
    /*
     * This will write bankroll of player 0 to file to determine the winner in tournaments
     */
    String resultFileString;
    
    boolean readCardsFromFile;
    
    
    /**
     * If verbose==true, prints a message to standard output.
     */
    public void verboseMessage(String str){
        if (verbose){
            System.out.println(str);
        }
    }
    
    /**
     * If setVerbose(true), messages sent/received are
     * printed to standard output.
     */
    public void setVerbose(boolean verbose){
        this.verbose = verbose;
    }
    
    /**
     * Initialize a PokerServer.
     * The InetAddress are the client's address we expect to contact us.
     * timePerHand is the amount of time allocated per hand, in milliseconds.
     */
    public PokerServer(SecureRandom random, InetAddress firstPlayerAddress, InetAddress secondPlayerAddress, int timePerHand){
        super(random);
        players = new PlayerAgent[2];
        seatsFlipped = false;
        this.firstPlayerAddress = firstPlayerAddress;
        this.secondPlayerAddress = secondPlayerAddress;
        this.timePerHand = timePerHand;
        this.verbose = false;
        
        // initialise this log file just in case the old constructor is used        
		initialiseLogFile( "logoutput.txt" );
        this.resultFileString="localresult.txt";
        readCardsFromFile = false;
    }
    
    /**
     * Initialize a PokerServer with logging and card reading capabilities.
     * The InetAddress are the client's address we expect to contact us.
     * timePerHand is the amount of time allocated per hand, in milliseconds.
     * @throws UnknownHostException 
     */
    public PokerServer(String firstPlayerAddress, 
    		String secondPlayerAddress, int timePerHand,
    		FileReader cardFileReader, BufferedWriter logFileWriter, String resultFile) throws UnknownHostException{
        super();
        players = new PlayerAgent[2];
        seatsFlipped = false;
        this.firstPlayerAddress = InetAddress.getByName( firstPlayerAddress );
        this.secondPlayerAddress = InetAddress.getByName( secondPlayerAddress );
        this.timePerHand = timePerHand;
        this.verbose = false;
        
        logFile = logFileWriter;
        cardFile = new BufferedReader( cardFileReader );
        resultFileString = resultFile;
        readCardsFromFile = true;
        //initialiseLogFile( "logoutput.txt" );
    }
    
    
    /**
     * Run the server.
     * Note that a small delay is advisable before attempting to connect to the server.
     */
    public void run(){
        try{
        socket = new ServerSocket(0);
        }catch (IOException io){
            System.err.println("2 CRITICAL ERROR: CANNOT START SERVER");
            return;
        }
        int numAcceptedPlayers = 0;
        do{
          try{
          Socket childSocket = socket.accept();
          InetAddress applicantAddress = childSocket.getInetAddress();
          if ((players[0]==null)&& firstPlayerAddress.equals(applicantAddress)){
              try{
              players[0]=new PlayerAgent(childSocket,0);
              } catch (SocketException so){
                  System.err.println("The first player's connection appears broken.");
              } catch (IOException io){
                  System.err.println("The first player's connection appears broken.");                  
              }
              numAcceptedPlayers++;
              verboseMessage("SERVER ACCEPTED PLAYER 0 FROM "+childSocket.getInetAddress());
          } else if ((players[1]==null)&& secondPlayerAddress.equals(applicantAddress)){
              try{
              players[1]=new PlayerAgent(childSocket,1);
              } catch (SocketException so){
                  System.err.println("The second player's connection appears broken.");
              } catch (IOException io){
                  logFile.write( "The first player's connection appears broken." );
            	  System.err.println("The first player's connection appears broken.");                  
              }
              numAcceptedPlayers++;
              verboseMessage("SERVER ACCEPTED PLAYER 1 FROM "+childSocket.getInetAddress());
          } else {
              try{
              childSocket.close();
              } catch (IOException io){
            	  logFile.write("Minor error: unaccepted child failed to close." );
            	  System.err.println("Minor error: unaccepted child failed to close.");
              }
          }
          } catch (IOException io){
            System.err.println("1 CRITICAL ERROR: CANNOT START SERVER");
            return;              
          }
          
        } while(numAcceptedPlayers!=2);
        players[0].setTimeRemaining(1000);
        players[1].setTimeRemaining(1000);
        try{
        if (!players[0].receiveMessage().equals("VERSION:1.0.0")){
            System.err.println("The first player does not acknowledge the protocol.");
            logFile.write( "The first player does not acknowledge the protocol.");
        }
        } catch(TimeoutException to){
            System.err.println("The " + ((to.playerIndex==0) ? "first" : "second") + " player does not acknowledge the protocol.");
   
        } catch (IOException e) {
        	System.err.println( "Exception writing error message to log file");
        }
        try{
        if (!players[1].receiveMessage().equals("VERSION:1.0.0")){
            System.err.println("The second player does not acknowledge the protocol.");
            logFile.write( "The second player does not acknowledge the protocol.");
        }
        } catch(TimeoutException to){
            System.err.println("The " + ((to.playerIndex==0) ? "first" : "second") + " player does not acknowledge the protocol.");
        } catch (IOException e) {
        	System.err.println( "Exception writing error message to log file");
		}
        
        /////////////////////////////////  1000 is proper, 
        
        int numHands = 1000;
        
        try {
			BufferedReader in = new BufferedReader(new FileReader( "config"));
			String str;
        
			str = in.readLine();
			in.close();
			
			// first, increment number of wins
			numHands = Integer.parseInt(str);
				
		} catch (FileNotFoundException e3) {
			System.err.println( "No config file found: Using default 1000 hands");
		} catch (IOException e) {
			System.err.println( "No integer in config file: Using default 1000 hands");
		}
	
		System.out.println( "Using " + numHands + " hands per match");
        boolean[] timeoutOnSend = new boolean[2];
        timeoutOnSend[0]=timeoutOnSend[1]=false;
        for(int i=0;i<numHands;i++){
            try{
              setHandNumber(i);
              playHand();
              writeLog(); // write the log information       
            } catch (TimeoutException to){
                System.out.println("Player "+to.playerIndex+ " timed out on a send on hand "+i+".");
                int numHandsRemaining = 999-i;
                forfeit(to.playerIndex,numHandsRemaining);
                timeoutOnSend[i]=true;
                System.out.println("Exited forfeit");
                break;
            }
            seatsFlipped = !seatsFlipped;
            
        }
        
        
        
        System.out.println("About to create result file");
        System.out.println("resultFileString is "+resultFileString);
        // this has to be done now, lest the log file be created before we actually finish
        try {
		FileWriter fw = new FileWriter(resultFileString);
            System.out.println("New file writer created");
			BufferedWriter resultFileBW = new BufferedWriter( fw);
                System.out.println("result file created");
        
        	resultFileBW.write( "" + players[0].bankroll );
        	resultFileBW.write( "\nServer: IPs " + firstPlayerAddress.getHostAddress() + " " + secondPlayerAddress.getHostAddress());
        	resultFileBW.write("\nServer: wrote results to " + resultFileString );
        	resultFileBW.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			System.err.println( "Error writing result file");
			e1.printStackTrace();
		}
        
        System.out.println("About to send endgame");
        try{
        players[0].sendMessage("ENDGAME");
        players[1].sendMessage("ENDGAME");
        players[0].close();
        players[1].close();
        socket.close();
        } catch (TimeoutException e){
            System.out.println("Timeout exception at endgame.");
            try {
				logFile.write( "***************************     Timeout exception at endgame.");
			} catch (IOException e1) {
				System.err.println( "Error writing endgame timeout exception");
			}
        } catch (IOException io){
            System.out.println("IOException at endgame.");
        }
        for(int i=0;i<2;i++){
            System.out.println("Elapsed sending time for player " + i + ":"+players[i].elapsedSendingTime);
            System.out.println("Final bankroll["+i+"]="+players[i].bankroll);
            
        }
        closeLogFile();
        
        
    }
    
    /**
     * If a player forfeits, he loses all of his blinds for the remainder of the game.
     */
    public void forfeit(int playerIndex, int numHandsRemaining){
       
    	System.out.println("Number of hands remaining:"+numHandsRemaining);
        // The player was the button on the last hand played.
        boolean isButton = (playerToSeat(playerIndex)==1);
        int smallBlindHands = isButton ? (numHandsRemaining/2) : ((numHandsRemaining+1)/2);
        int bigBlindHands = numHandsRemaining - smallBlindHands;
        double penalty = 5.0 * smallBlindHands + 10.0 * bigBlindHands;
        System.out.println("Penalty:"+penalty);
        players[playerIndex].incrementBankroll(-penalty);
        players[1-playerIndex].incrementBankroll(penalty);
        System.out.println("Attempting to write to logfile");
        try {
			logFile.write( "Player forfeits");
			logFile.write( "Penalty:"+penalty );
		} catch (IOException e) {
			System.out.println( "Error writing forfeit data to logfile");
		}

        System.out.println("Wrote to logfile");
    }
    
    /**
     * Sends the match state as it appears to all players.
     */
    public void broadcastMatchState() throws TimeoutException{
        for(int i=0;i<2;i++){
            verboseMessage("SERVER SENDS:"+getMatchState(i)+ " TO PLAYER "+seatToPlayer(i)+".");
              players[seatToPlayer(i)].sendMessage(getMatchState(i));
        }
    }
    
    
    /**
     * Play one hand. One thousand hands make a match.
     */
    public void playHand() throws TimeoutException{
        players[0].setTimeRemaining(timePerHand);
        players[1].setTimeRemaining(timePerHand);
        // if we are reading cards from file, otherwise generate
        if ( readCardsFromFile ) startHand( cardFile );
        else startHand();
        try{
        do{
            broadcastMatchState();
            String response = "";
            do{
              response = players[seatToPlayer(seatToAct)].receiveMessage();
              verboseMessage("MESSAGE RECEIVED BY SERVER FROM "+seatToPlayer(seatToAct)+":"+response);
            } while(!isAppropriate(response));
            verboseMessage("MESSAGE ACKNOWLEDGED BY SERVER FROM "+seatToPlayer(seatToAct)+":"+response);
            handleAction(getActionFromResponse(response));
        } while(!handOver);
        } catch(TimeoutException to){
            System.out.println(to);
            /**
             * A serious violation (more than 100 seconds to RECEIVE a message at the TCP/IP level)
             * results in a forfeit of the remainder of the match.
             */
            if (to.serious){
                winnerIndex = getOtherSeat(playerToSeat(to.playerIndex));
                handOver = true;
                adjustBankrolls();
                throw to;
            } else {
                handleAction('f');
            }
        }
        broadcastMatchState();
        adjustBankrolls();
        /*System.out.println(getMatchState(0));
        System.out.println(getMatchState(1));
        System.out.println("bankroll[0]="+players[0].bankroll);
        System.out.println("bankroll[1]="+players[1].bankroll);*/
        
    }
    
    /**
     * Tests if a response is actually a response to the CURRENT action.
     */
    public boolean isAppropriate(String response){
        if (response.length()<2){
            return false;
        }
        String responseFront = response.substring(0,response.length()-2);
        //System.out.println("FRONT OF RESPONSE:"+responseFront);
        //System.out.println("MATCH STATE:"+getMatchState(seatToAct));
        return (getMatchState(seatToAct).equals(responseFront));
        
    }
    
    /**
     * Gets the last character of a response, which should be 'c', 'r', or 'f'
     */
    public char getActionFromResponse(String response){
        return response.charAt(response.length()-1);
    }
    
    /**
     * Which player is in the seat?
     */
    public int seatToPlayer(int seat){
        return seatsFlipped ? (1-seat) : seat;
    }
    
    /**
     * Which seat is the player in?
     */
    public int playerToSeat(int player){
        return seatsFlipped ? (1-player) : player;
    }
    
    /**
     * Increment the bankroll of the player in seat seat an amount amount.
     */
    public void incrementSeatBankroll(double amount, int seat){
        players[seatToPlayer(seat)].incrementBankroll(amount);
    }
    
    
    /**
     * Adjust the bankrolls according to current hands winnings.
     */
    public void adjustBankrolls(){
        incrementSeatBankroll(amountWon[0],0);
        incrementSeatBankroll(amountWon[1],1);
    }
    
    /**
     * Run the server listening for connections from the localhost 
     * to test the code.
     */
    public static void main(String[] args) throws Exception{
        SecureRandom random = new SecureRandom();
        //boolean verbose = false;
        
        InetAddress iaddr = InetAddress.getByName("127.0.0.1");
        System.out.println(iaddr);
        
        PokerServer server = new PokerServer(random,iaddr,iaddr,7000);
        Thread serverThread = new Thread(server);
        serverThread.start();
        while(server.socket==null){
            Thread.sleep(1000);
            System.out.println("one second");
        }
        System.out.println("Server listening on port"+server.socket.getLocalPort()+"...");
    }
    /**
     * Create file to write the log into 
     * Initialise/close is so that one file descriptor can be used instead
     * of re-creating it each round
     */
    public void initialiseLogFile(String filename) {
    	
    	Format formatter;
    	String header = "";
    	String directory = "data/logs/";
    	
    	filename = directory + filename;
    	
    	new File(directory).mkdirs(); // create directories
    	
    	
        // Get today's date
        Date date = new Date();

        // MAY.01.2006.22.14.02
        formatter = new SimpleDateFormat("MMM.dd.yyyy-HH:mm:ss");
        header = formatter.format(date) + "\n";
        
        
        
    	try {
    		logFile = new BufferedWriter(new FileWriter(filename,true));
    		logFile.write( header );  // TODO write log header
    		
    	} catch (IOException e) {
    		System.err.println( "IOException: Failed creating the log to " + filename);
    	}
    	
    	
    	
    }
    
    /*
     * Clean up after writing the entire log file
     */
    public void closeLogFile() {
    	try {
    		logFile.close();
    	} catch ( IOException e) {
    		System.err.println( "IOException: Failed closing the log to file");
    	}
    }
    
    /**
     * Get the bankroll information for the log
     */
    public String getBankrollString() {
   	
    	return "" + players[0].bankroll + ":" + players[1].bankroll;
    }
    
    /**
     *  Write the necessary logging information to a file
     */
    public void writeLog() {
    	 // TODO write logs to file 
    	String log = "";
    	String filename = "logoutput.txt";
    	
    	log += getMatchState(2);
    	log += ":" + getBankrollString() + "\n";
    	
    	try {
	        logFile.write(log);
	    } catch (IOException e) {
	    	System.err.println( "IOException: Failed writing the log to file " + filename);
	    }
    	
    	
    }
}
