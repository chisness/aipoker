package ca.ualberta.cs.poker.free.tournament;

import java.util.Hashtable;
import java.util.StringTokenizer;
import java.util.Vector;
import java.io.*;
import java.net.InetAddress;
import java.security.SecureRandom;

import ca.ualberta.cs.poker.free.dynamics.MatchType;
import ca.ualberta.cs.poker.free.server.GenerateCards;
import ca.ualberta.cs.poker.free.server.RingServer;

/*
 * TODO GO OVER THIS AGAIN!!!
 * This class contains the necessary information for a match between two competitors to
 * be run. Calling start() will run the entire match.
 * 
 * Note that this class has more generalization than necessary, as
 * hopefully it will soon be refactored so that there is a BasicMatch
 * class which would be large, and several small
 * subclasses: a RingLimitMatch class, a HeadsUpNoLimitMatch class,
 * and a LimitRingMatch class.
 * @author Christian Smith
 * @author Martin Zinkevich
 * 
 */
public class RingLimitMatch implements MatchInterface, Runnable{
	MatchType info;
	Thread matchThread;
	Thread serverThread;
	RingServer server;
	
    /**
	 * The bots in the match (exactly 2).
	 */
	private Vector<BotInterface> bots;

	BufferedWriter log;
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
	 * If verbose is true, various different events result in
	 * messages to standard error
	 */ 
    static boolean verbose = true;
	
        /**
	 * Creates a new match with the specified bots,
	 * card file, server address, and name.
	 * The timePerHand is 7 seconds and the number of hands is 1000.
	 */
    /*
	public RingLimitMatch(Vector<BotInterface> bots, String cardFile, 
	InetAddress serverAddress, String name){
	  this(bots,cardFile,serverAddress,name,new GameInfo(LimitType.LIMIT,false,0));
	}*/

	/**
	 * Creates a new match with the specified bots,
	 * card file, server address, name, time
	 * per hand, and number of hands.
	 * @param cardFile the filename for the cards. Conventionally,
	 * should end with .crd.
	 */
	public RingLimitMatch(Vector<BotInterface> bots, String cardFile, 
	InetAddress serverAddress, String name, MatchType info){
	  this.bots = bots;
	  this.cardFile = "data/cards/"+cardFile;
	  this.serverAddress = serverAddress;
	  this.name = name;
	  this.logFile = "data/results/"+name+".log";
	  this.resultFile = "data/results/"+name+".res";
	  this.serverLogFile = "data/serverlog/"+name+".srv";
	  this.info = info;
	}
	
	public int getNumCards(){
		return 5+(bots.size()*2);
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

	public void terminate(){
		if (matchThread!=null){
			matchThread.interrupt();
		} 
		if (serverThread!=null){
			serverThread.interrupt();
		}
	}
        /**
	 * Use the machines assigned by the MachineRoom.
	 */
	public void useMachines(Vector<MachineInterface> machines){
	  this.machines = machines;
	}

    /**
	 * Start the match.
	 */
	public void startMatch(){
   	  // the server will write out these 3 files
	  try{
	    System.out.println("Match "+name+": creating logs");
	    FileReader cardReader = new FileReader(cardFile);
            BufferedWriter logWriter = new 
	    BufferedWriter(new FileWriter(logFile));
	    log = new BufferedWriter(new
	    FileWriter(serverLogFile));
        
	    String[] names = new String[bots.size()];
	    for(int botIndex=0;botIndex<bots.size();botIndex++){
	    	names[botIndex]=bots.get(botIndex).getName();
	    }

        server = new RingServer(bots.size(),info,machines,new BufferedReader(cardReader),resultFile,logWriter,names);
        matchThread = new Thread(this);
        matchThread.start();        
	  } catch(Exception e){
	    System.err.println("Error starting server");
	    e.printStackTrace();
	  }
	}

        /**
	 * Test if this match is the same as another.
	 */
	public boolean equals( Object obj ) {
		if( obj instanceof RingLimitMatch ) {
			return name.equals(((RingLimitMatch)obj).name);
		}
		return false;
	}
	
	public int getHand(){
		if (server==null){
			return 0;
		}
		return server.handNumber;
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
	 * TODO: FIX THIS FUNCTION
	 */
	public Vector<Integer> getUtilities(){
	  Vector<Integer> result = new Vector<Integer>();
	  for(int i=0;i<bots.size();i++){
//		  result.add(0);
	  }
	  try{
	    BufferedReader in = new BufferedReader(new
	    FileReader(resultFile));
	    String str = in.readLine();
	    in.close();
	    StringTokenizer st = new StringTokenizer(str,"|");
	    while(st.hasMoreTokens()){
	    	result.add(Integer.parseInt(st.nextToken()));
	    }
	  } catch (IOException e){
	    System.err.println("Error reading double from result file");
	    e.printStackTrace();
      }
	  if (result.size()!=bots.size()){
		  System.err.println("Wrong number of utilities in result file");
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
              " numCards:" + getNumCards()+"\n";
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
	  GenerateCards.generateOneFile(cardFile,random,info.numHands,getNumCards());
	}
	
	public boolean confirmCardFile(){
		return GenerateCards.confirmOneFile(cardFile, info.numHands, getNumCards());
	}
	
	public String getName(){
		return name;
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

            log.write("Server reports port: " + port);

	    // if port is null, something went horribly wrong
	    if (port == 0) {
	      log.write("Error getting port from server" );
	      return;
	    }
            System.out.println("Starting bots...");
        Hashtable<InetAddress,Integer> priorBots = new Hashtable<InetAddress,Integer>();
        
	    for(int i=0;i<machines.size();i++){
	    	MachineInterface current = machines.get(i);
	    	if (priorBots.containsKey(current.getIP())){
	    		int previousBot = priorBots.get(current.getIP());
	    		try{
	    		while(server.players[previousBot]==null){
	    			Thread.sleep(1000);
	    		}
	    		} catch (InterruptedException ie){
	    			// TODO What should be done here?
	    		}
	    	}
	      current.start(bots.get(i),serverAddress,port);
	    }
		} catch (InterruptedException e) {
			System.err.println("Interruption of game");
			e.printStackTrace(System.err);
		} catch (IOException io){
			System.err.println("I/O error");
			io.printStackTrace(System.err);
		}
	}
}
