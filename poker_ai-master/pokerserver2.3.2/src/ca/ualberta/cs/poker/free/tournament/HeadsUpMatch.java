package ca.ualberta.cs.poker.free.tournament;

import java.util.Vector;
import java.io.*;
import java.net.InetAddress;
import java.security.SecureRandom;

import ca.ualberta.cs.poker.free.dynamics.MatchType;
import ca.ualberta.cs.poker.free.server.GenerateCards;
import ca.ualberta.cs.poker.free.server.RingServer;

/*
 * This class contains the necessary information for a match between two competitors to
 * be run. Calling start() will run the entire match.
 * 
 * Note that this class has more generalization than necessary, as
 * hopefully it will soon be refactored so that there is a BasicMatch
 * class which would be large, and several small
 * subclasses: a HeadsUpLimitMatch class, a HeadsUpNoLimitMatch class,
 * and a LimitRingMatch class.
 * @author Christian Smith
 * @author Martin Zinkevich
 * TODO Make this a subclass of RingLimitMatch
 */
public class HeadsUpMatch implements MatchInterface, Runnable{
	MatchType info;
	
	/**
	 * The thread for the server
	 */
	Thread serverThread;
	
	/**
	 * The thread for loading bots and starting the server.
	 */
	Thread matchThread;
	
	RingServer server;
	
	
        /**
	 * The bots in the match (exactly 2).
	 */
	private Vector<BotInterface> bots;

	/**
	 * The machines used in the match.
	 */
	private Vector<MachineInterface> machines;


        /**
	 * The server address.
	 */
	private InetAddress serverAddress;

	/**
	 * The port of the server.
	 */
	private int port;

	/**
	 * The name of the file with the cards.
	 */
	private String cardFile;

	/** 
	 * The name of the match.
	 */
	private String name;

	/**
	 * The name of the result file.
	 * The file is not created until the match is over.
	 */
	public String resultFile;

	/**
	 * The name of the log file.
	 */
	public String logFile;

	/**
	 * The name of the log file for the server.
	 */
	private String serverLogFile;


	/**
	 * The number of cards used in one hand (9 for heads-up).
	 */
	private int numCards;

	/**
	 * If verbose is true, the toString() method gives a detailed
	 * description of the match. Otherwise, it just returns the name.
	 */ 
    static boolean verbose = true;
	
        /**
	 * Creates a new match with the specified bots,
	 * card file, server address, and name.
	 * The timePerHand is 7 seconds and the number of hands is 1000.
	 */
	/*public HeadsUpMatch(Vector<BotInterface> bots, String cardFile, 
	InetAddress serverAddress, String name,LimitType limitGame){
	  this(bots,cardFile,serverAddress,name,7000,(LimitType.LIMIT)==limitGame ? 1000 : 100,limitGame);
	}*/

	/**
	 * Creates a new match with the specified bots,
	 * card file, server address, name, time
	 * per hand, and number of hands.
	 * @param cardFile the filename for the cards. Conventionally,
	 * should end with .crd.
	 */
	public HeadsUpMatch(Vector<BotInterface> bots, String cardFile, 
	InetAddress serverAddress, String name,MatchType info){
	  this.bots = bots;
	  this.cardFile = "data/cards/"+cardFile;
	  this.serverAddress = serverAddress;
	  this.name = name;
	  this.logFile = "data/results/"+name+".log";
	  this.resultFile = "data/results/"+name+".res";
	  this.serverLogFile = "data/serverlog/"+name+".srv";
	  this.numCards = 9;
	  this.info = info;
	}

	public int getHand(){
		if (server==null){
			return 0;
		}
		return server.handNumber;
	}
    /**
	 * Get the bots for this match
	 */
	public Vector<BotInterface> getBots(){
	  return bots;
	}

        /**
	 * Get the machines for this match
	 */
	public Vector<MachineInterface> getMachines(){
	  return machines;
	}

        /**
	 * Use the machines assigned by the MachineRoom.
	 */
	public void useMachines(Vector<MachineInterface> machines){
	  this.machines = machines;
	}

    /**
	 * Start the match.
	 * TODO: Be more cautious with closing what we open.
	 */
	public void startMatch(){
   	  // the server will write out these 3 files
	  try{
	    System.out.println("Match "+name+": creating logs");
	    System.out.println("Card file:"+cardFile);
	    BufferedReader cardReader = new BufferedReader(new FileReader(cardFile));
            BufferedWriter logWriter = new 
	    BufferedWriter(new FileWriter(logFile));
	    //BufferedWriter log = new BufferedWriter(new
	    //FileWriter(serverLogFile));

	    String[] names = new String[2];
	    names[0]=bots.get(0).getName();
	    names[1]=bots.get(1).getName();
	    
	    server = new RingServer(2, // numPlayers
					info,
	    			machines, // playerAddress
					cardReader, // cardReader
					resultFile, // resultFilename
					logWriter, // logFile
					names); // names
	    matchThread = new Thread(this);
	    matchThread.start();
	    /*
			 * PokerServer server = new PokerServer(
			 * machines.get(0).getIP().getHostAddress(),
			 * machines.get(1).getIP().getHostAddress(), timePerHand,
			 * cardReader, logWriter, resultFile);
			 */
	  } catch(IOException e){
		    System.err.println("Error starting server");
		    e.printStackTrace();
		  }
	
	  }

	/**
	 * TODO: Make cleaner, in particular:
	 * Close all the ports on the server thread after interrupting.
	 * These ports would not be closed if there is still a reference to
	 * them, which I think there still is.
	 */
	public void terminate(){
		if (matchThread!=null){
			matchThread.interrupt();
		} 
		if (serverThread!=null){
			serverThread.interrupt();
		}
	}
	
	public void run() {
		try {
			serverThread = new Thread(server);
			serverThread.start();

			// launch server, wait for clients
			while (server.socket == null) {
				Thread.sleep(1000);
				System.out.println("Server sleeping one second.");
			}
			port = server.socket.getLocalPort();

			server.errorMessage("Server reports port: " + port);

			// if port is null, something went horribly wrong
			if (port == 0) {
				server.errorMessage("Error getting port from server");
				return;
			}
			System.out.println("Starting bots...");

			machines.get(0).start(bots.get(0), serverAddress, port);
			if (machines.get(0).getIP().getHostAddress().equals(
					machines.get(1).getIP().getHostAddress())) {
				while (server.players[0] == null) {
					Thread.sleep(1000);
					System.out.println("Waiting for first player to log on.");
					if (isComplete()) {
						// If the match ends, we terminate.
						return;
					}
				}
			}
			machines.get(1).start(bots.get(1), serverAddress, port);
		} catch (InterruptedException e) {

		}

	}

        /**
		 * Test if this match is the same as another.
		 */
	public boolean equals( Object obj ) {
		if( obj instanceof HeadsUpMatch ) {
			return name.equals(((HeadsUpMatch)obj).name);
		}
		return false;
	}



        /**
	 * Test if the match is complete.
	 * This is done by testing for the existence of a results
	 * file.
	 */
	public boolean isComplete(){
	  return new File(resultFile).exists();
	}


        /**
		 * Get the utilities from the match for each player.
		 */
	public Vector<Integer> getUtilities() {
		Vector<Integer> result = new Vector<Integer>();
		try {
			BufferedReader in = new BufferedReader(new FileReader(resultFile));
			String str = in.readLine();
			in.close();
			int lastVert = -1;
			while (true) {
				int nextVert = str.indexOf("|", lastVert + 1);
				if (nextVert == -1) {
					result.add(new Integer(str.substring(lastVert + 1)));
					break;
				}
				result.add(new Integer(str.substring(lastVert + 1, nextVert)));
				lastVert = nextVert;
			}
		} catch (IOException e) {
			System.err.println("Error reading double from result file");
			e.printStackTrace();
			for (int i = 0; i < bots.size(); i++) {
				result.add(0);
			}
		}
		return result;
	}

	public String toString(){
	  if (verbose){
	    String result = 
	      "server:"+serverAddress+
	      " port:"+ port+
	      " card file: "+cardFile+
	      " name:" +name+
              " resultFile:" +resultFile+
              " logFile:" + logFile+
              " serverLogFile:" + serverLogFile+
              " timePerHand:" + info.timePerHand+
              " numHands:" + info.numHands+
              " numCards:" + numCards+
              " limit:"+info.limitGame +"\n";
            for(BotInterface b:bots){
              result += (b.toString()+"\n");
	    }
	    if (machines==null){
	      result += "no machines yet\n";
	    } else {
	      for(MachineInterface m:machines){
	        if (m==null){
		  result+="null\n";
                } else {
		  result+=(m.toString()+"\n");
		}
	      }
	    }
	    return result;
	  }
	  return name;
	}

    /**
	 * Generate the card file this match would require.
	 */
	public void generateCardFile(SecureRandom random){
	  GenerateCards.generateOneFile(cardFile,random,info.numHands,numCards);
	}
	
	public boolean confirmCardFile(){
		return GenerateCards.confirmOneFile(cardFile,info.numHands,numCards);
	}
		
	public String getName(){
		return name;
	}
}
