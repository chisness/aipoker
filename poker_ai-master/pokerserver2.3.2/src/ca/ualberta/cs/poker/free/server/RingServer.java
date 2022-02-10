package ca.ualberta.cs.poker.free.server;

import java.io.BufferedWriter;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.util.Vector;

import ca.ualberta.cs.poker.free.dynamics.LimitType;
import ca.ualberta.cs.poker.free.dynamics.MatchType;
import ca.ualberta.cs.poker.free.dynamics.RingDynamics;
import ca.ualberta.cs.poker.free.tournament.MachineInterface;
import ca.ualberta.cs.poker.free.tournament.RemoteMachine;

/**
 * 
 * @author maz
 * TODO handle an interruption.
 */
public class RingServer extends RingDynamics implements Runnable {
	public ServerSocket socket;
	Vector<MachineInterface> playerAddress;
	public RingAgent[] players;
	
	BufferedReader cardSource;
	
	BufferedWriter logFile;
	
	String resultFilename;
	
	/**
     * If true, messages sent and received are sent 
     * to standard output.
     */
    boolean verbose;
    boolean displayPort = false;
    
    /**
     * If verbose==true, prints a message to standard output.
     */
    public void verboseMessage(String str){
        if (verbose){
            System.out.println(str);
        }
    }
    
    /**
     * Print an error message.
     * @param str the error message
     */
    public void errorMessage(String str){
    	System.err.println(str);
    }
    
    
	public RingServer(int numPlayers, MatchType info, Vector<MachineInterface> playerAddress, 
			BufferedReader cardReader, String resultFilename, BufferedWriter logFile, 
			String[] botNames) {
		super(numPlayers, info, botNames);
		this.playerAddress = playerAddress;
		players = new RingAgent[numPlayers];
		this.cardSource = cardReader;
		this.logFile = logFile;
		this.resultFilename = resultFilename;
		this.verbose = false;
	}

	public boolean open(){
		try {
			socket = new ServerSocket(0);
			if (verbose||displayPort){
				System.err.println("Server listening on port "+socket.getLocalPort());
			}
			// JH - Change - Fifteen minutes to login.
			socket.setSoTimeout(15 *60*1000);
		} catch (IOException io) {
			System.err.println("CRITICAL ERROR: CANNOT START SERVER");
			return false;
		}
		return true;
	}
	
	public void close(){
		try{
			if (socket!=null){
			  socket.close();
			}
		} catch (IOException io){
			errorMessage("Cannot close server socket");
		}
	}
	
	public boolean loginPlayers() throws InterruptedException{
		
		int numAcceptedPlayers = 0;
		do {
			Socket childSocket = null;
			boolean childAccepted = false;
			InetAddress applicantAddress = null;
			try {
				childSocket = socket.accept();
				applicantAddress = childSocket.getInetAddress();
			} catch (SocketTimeoutException to){
				errorMessage("Time expired to login");
				logoutPlayers();
				return false;
			} catch (InterruptedIOException ie){
				System.err.println("Login interrupted");
				throw new InterruptedException();
			} catch (IOException io) {
				errorMessage("Cannot start server");
				logoutPlayers();
				return false;
			}

			for (int i = 0; i < players.length; i++) {
				if ((players[i] == null)
						&& playerAddress.get(i).isThisMachine(applicantAddress)) {
					try {
						players[i] = new RingAgent(childSocket, i);
					} catch (SocketException so) {
						errorMessage("The " + i
								+ "th player's connection appears broken.");
					} catch (IOException io) {
						errorMessage("The " + i
								+ "th player's connection appears broken.");
					}
					numAcceptedPlayers++;
					verboseMessage("SERVER ACCEPTED PLAYER " + i + " FROM "
							+ childSocket.getInetAddress());
					childAccepted = true;
					break;
				}
			}
			try {
				
				if (!childAccepted){
					errorMessage("Unaccepted child from "+applicantAddress);
				  childSocket.close();
				}
			} catch (IOException io) {
				errorMessage("Minor error: unaccepted child failed to close.");
			}

		} while (numAcceptedPlayers != players.length);
		
		for(RingAgent player:players){
			try{
			player.setTimeRemaining(1000);
			player.protocol = player.receiveMessage();
	        if (!player.protocol.equals("VERSION:1.0.0")){
	            errorMessage("The first player does not acknowledge the protocol.");
	        } 
		} catch(TimeoutException to){
	            errorMessage("The " + to.playerIndex+"th player does not acknowledge the protocol.");
	        }
		}
		return true;
	}
	
	/**
	 * Log out the players. Gracefully handles null players.
	 */
	public void logoutPlayers(){
		if (players!=null){
		for(RingAgent player:players){
			if (player!=null){
				if (player.inGoodStanding){
				try{
				  player.sendMessage("ENDGAME");

				} catch (TimeoutException to){
					// Do nothing, the game is already over.
				}
			}
			try{
			player.close();
			} catch (IOException io){
				errorMessage("Error: could not close player TCP/IP connection");
			}
			}
		}
		}
	}
	
	/**
	 * Should be called AFTER the server thread is interrupted
	 * (part of terminating a match)
	 * Unneccessary in normal usage
	 */
	public void cleanup(){
		logoutPlayers();
		closePipes();
		close();
	}
	/**
	 * Close all the pipes for cards, logs, et cetera.
	 * NOTE: what happens if pipes are already closed?
	 */
	public void closePipes(){
		
		try{
			if (logFile!=null){
				logFile.close();
			}
		} catch (IOException io){
			errorMessage("Could not close log file");
		}
		
		try{
			if (cardSource!=null){
			  cardSource.close();
			}
		} catch (IOException  io){
			errorMessage("Could not close card source");
		}
	}
	
	public void createResultFile(){
		try{
		FileWriter writer = new FileWriter(resultFilename);
		BufferedWriter bufferedWriter = new BufferedWriter(writer);
		String stackString = ""+stack[0];
		String botNameString = botNames[0];
		for(int i=1;i<players.length;i++){
			stackString += ("|"+stack[i]);
			botNameString += ("|"+botNames[i]);
		}
		bufferedWriter.write(stackString+"\n");
		bufferedWriter.write(botNameString+"\n");
		bufferedWriter.write("Number_of_hands:"+info.numHands+"\n");
		bufferedWriter.write("LimitType:"+info.limitGame+"\n");
		bufferedWriter.write("StackBounds:"+info.stackBoundGame+"\n");
		bufferedWriter.write("Timeout per hand(ms):"+info.timePerHand+"\n");
		bufferedWriter.close();
		} catch (IOException io){
			errorMessage("Error writing result file");
		}
		
	}
	
	public void run() {
		boolean cleanExit = false;
		try{
		if (open()){
			if (loginPlayers()){
				try{
					logFile.write(getHeader());
				} catch(IOException io){
					System.err.println("Error writing to log file");
					io.printStackTrace(System.err);
					return;
				}
				if (info.chessClock){
					for(RingAgent a:players){
						a.setTimeRemaining(info.timePerHand*info.numHands);
					}
				}
				for(int i=0;i<info.numHands;i++){
					playHand();
				}
			}
		}
		  cleanExit = true;
		} catch (InterruptedException ie){
		} finally {
			logoutPlayers();
			close();
			closePipes();
			if (cleanExit){
				createResultFile();
			}
		}

	}
	

	/**
	 * Tests if an action request is for the current state of the game
	 * @param mess an action request
	 * @return true if the beginning of the action request is identical
	 * to the current state of the game
	 */
	public boolean isForCurrentState(String mess){
		if (mess==null){
			return false;
		}
		int lastColon = mess.lastIndexOf(':');
		if (lastColon==-1){
			return false;
		}
		// The message up to the last colon.
		String stateFor = mess.substring(0,lastColon);
		
		String currentState = getMatchState(seatToAct);
		return currentState.equals(stateFor);
	}
	
	/**
	 * Get the last action.
	 * @param mess
	 * @return
	 */
	public String getAction(String mess){
		return mess.substring(mess.lastIndexOf(':')+1);
	}
	
	/**
	 * Play a hand. Some rules for when certain actions are forced (calling when
	 * all in) are enforced inside this function, instead of RingDynamics.
	 *
	 */
	public void playHand() throws InterruptedException{
		nextHand(cardSource);
		sendState();
		for(int i=0;i<players.length;i++){
			if (players[i].inGoodStanding){
				if (!info.chessClock){
				  players[i].setTimeRemaining(info.timePerHand);
				}
			}
		}
		while(!isGameOver()){
			int currentPlayer = seatToPlayer(seatToAct);
			if (isAllIn(seatToAct)){
				// Cannot fold if all in
				// A rule to prevent "chip dumping"
				handleCall();
			} else if (getNumActivePlayersNotAllIn()==1){
				// Have to see it to the end if all in
				// To prevent "chip dumping"
				handleCall();
			} else if (players[currentPlayer].inGoodStanding){
			
				try{
					String mess = null;
					do{
						mess = players[currentPlayer].receiveMessage();
						verboseMessage("Received from "+currentPlayer+":"+mess);
					} while (!isForCurrentState(mess));
					String action = getAction(mess);
					handleAction(action);
				} catch (TimeoutException to){
					
					errorMessage("Player "+currentPlayer+" timed out on hand "+handNumber);
					if (info.chessClock){
						players[currentPlayer].inGoodStanding=false;
					}
					handleFold();
				}
			} else {
				handleFold();
			}
			sendState();
		}
		try{
			//logFile.write(toString());
			logFile.write(getGlobalState()+"\n");
			logFile.flush();
		} catch (IOException io){
			errorMessage("Error writing to log file:"+io);
		}
	}
	
	/**
	 * Send the state to all players in good standing.
	 *
	 */
	public void sendState(){
		verboseMessage("State:\n"+super.toString());
		for(int i=0;i<players.length;i++){
			if (players[i].inGoodStanding){
				try{
					String matchState = getMatchState(playerToSeat(i));
					verboseMessage("Sent to "+i+":"+matchState);
					//verboseMessage("Player "+i+" is port "+players[i].socket.getPort());
					players[i].sendMessage(matchState);
				} catch (TimeoutException to){
					players[i].inGoodStanding = false;
					errorMessage("Player "+i+" timed out on hand "+handNumber);
				}
			}
		}
	}
	public static void showUsage(){
		System.err.println("Usage:java ca.ualberta.cs.poker.free.server.RingServer [-t <MATCHTYPE>| -p1 <IP> <NAME> |-p2 <IP> <NAME> |-c <CARDFILE>|-l <LOGFILE>|-r <RESULTFILE>]*");
		System.exit(0);
	}
	public static String getArg(int index, String[] args){
		if (args.length<=index){
			showUsage();
		}
		return args[index];
	}
	// XXX NOTE XXX
	// This is *hardcoded* for 2 players, so it can't do ring.  Careful of this!
	// -JH
	public static void main(String[] args) throws IOException{
		String matchType = "HEADSUPLIMIT2007";
		String[] ips = new String[2];
		String[] names = new String[2];
		String cardFilename = "";
		String resultFilename = "result.res";
		String logFilename = "";
		for(int i=0;i<ips.length;i++){
			ips[i]="127.0.0.1";
			names[i]="Bot"+i;
		}
		
		for(int i=0;i<args.length;i++){
			if (args[i].equals("-t")){
				i++;
				matchType = getArg(i,args);
			} else if (args[i].startsWith("-p")){
				int playerIndex = Integer.parseInt(args[i].substring(2));
				i++;
				ips[playerIndex]=getArg(i,args);
				i++;
				names[playerIndex]=getArg(i,args);
			} else if (args[i].equals("-c")){
				i++;
				cardFilename = getArg(i,args);
			} else if (args[i].equals("-r")){
				i++;
				resultFilename = getArg(i,args);
			} else if (args[i].equals("-l")){
				i++;
				logFilename = getArg(i,args);
			} else {
				showUsage();
			}
		}
		BufferedWriter log = (logFilename.equals("")) ? new BufferedWriter(new OutputStreamWriter(System.out)) : new BufferedWriter(new FileWriter("logFilename"));
		if (cardFilename.equals("")){
			System.err.println("ERROR: For now, must provide filename for cards");
			showUsage();
		}
		BufferedReader cards = new BufferedReader(new FileReader(cardFilename));
		MatchType type = new MatchType(LimitType.LIMIT,false,0,3000);
		if (matchType.equals("HUMAN")){
			type = new MatchType(LimitType.LIMIT, false, 0, 500);
			type.chessClock = true;
			type.timePerHand = 60000;
		} else if (matchType.equals("HEADSUPLIMIT2007")){
			type = new MatchType(LimitType.LIMIT,false,0,3000);
		} else if (matchType.equals("HEADSUPLIMIT2006")){
			type = new MatchType(LimitType.LIMIT,false,0,1000);
			type.chessClock = false;
			type.timePerHand = 60000;
		} else if (matchType.equals("HEADSUPNOLIMIT2007")){
			type = new MatchType(LimitType.DOYLE,false,0,1000);
		} else if (matchType.equals("HEADSUPLIMIT2009")){
			type = new MatchType(LimitType.LIMIT,true,0,3000);
			type.chessClock = false;
			type.timePerHand = 60000;
		} else if (matchType.equals("HEADSUPNOLIMIT2009")){
			type = new MatchType(LimitType.DOYLE,true,0,3000);
		} 
		
		Vector<MachineInterface> machines = new Vector<MachineInterface>();
		for(int i=0;i<ips.length;i++){
			machines.add(new RemoteMachine(InetAddress.getByName(ips[i]), "/",
					"/", true, false, false));
		}
		RingServer server = new RingServer(2,type,machines,cards,resultFilename,log,names);
		server.displayPort = true;
		server.run();
	}
}
