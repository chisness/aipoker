package ca.ualberta.cs.poker.free.tournament;

import java.io.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.LinkedList;
import java.util.Vector;
import java.security.SecureRandom;

/**
 * The forge is the central piece of the software.
 * The forge maintains links to a Node that provides it
 * with matches to put in the queue, and a MachineRoom
 * that contains the machines required to run those matches.
 * In addition, the forge takes care of keeping track of 
 * which matches are queued, and which matches are running.
 * main() is the main function for the software.
 * The code overall works as follows:
 * Profile is used to construct a MachineRoom and a Node from
 * a profile file.
 * 
 * Forge is then constructed with these objects.
 * runTournament() is called, and goes into the following loop until
 * the tournament is complete.
 * It loads matches from the node into the queue.
 * It tests to see if any matches have been completed. If so, the
 * resources used are returned to the MachineRoom.
 * It tests to see if any matches from the queue can be run with the
 * machines currently in the MachineRoom. If so, the resources are taken
 * from the MachineRoom and the match is removed from the queue and
 * placed in the set of running matches.
 *
 * The forge checks for the existance of result files to determine if the
 * match has completed or not. A shortcoming of this idea is a client crash
 * will only write a result file if the server does, and a server crash will
 * cause the forge to never detect the completion, resulting in the
 * resources for that match to be lost forever. In the long term, this
 * problem needs to be fixed.
 * 
 * @author Christian Smith
 * @author Martin Zinkevich
 *
 */
public class Forge {

        /**
	 * Matches that have been started, but have
	 * not had the relevant machines returned to the 
	 * repository.
	 */
	private LinkedList<MatchInterface> runningMatches;

	/**
	 * Matches that have been entered into the forge,
	 * but not started.
	 */
	private LinkedList<MatchInterface> queuedMatches;

	/**
	 * A MachineRoom with the machines needed.
	 */
    MachineRoom machines;

	/**
	 * The root node for the competition.
	 */
	Node node;

	String statusFileLocation = "status.txt";
	
	/**
	 * Initializes the forge with the specified node
	 * and machines and no queued or running matches.
	 */
	public Forge(MachineRoom machines, Node node) {
	  runningMatches = new LinkedList<MatchInterface>();
	  queuedMatches = new LinkedList<MatchInterface>();
	  this.node = node;
	  this.machines=machines;
	}

	/**
	 * Tests to see if a match has been queued or is
	 * currently running.
	 */
	public boolean runningOrQueued(MatchInterface match){
	  return (runningMatches.contains(match)||queuedMatches.contains(match));
	}  

	/*
	 * Add a match to the list
	 * Makes certain the match is not superfluous (not
	 * already queued or running).
	 */
	public void add(MatchInterface m) {
		
		// Check on disk  ie. data/results/seseries3match2.res
		if (!m.isComplete()&&!runningOrQueued(m)){
	          queuedMatches.addLast(m);
		}	
	}

    /**
     * Remove a match. Stop the match, clean the machines.
     */
    public void remove(MatchInterface m){
    	System.err.println("Forge.remove()");
    	if (queuedMatches.contains(m)){
    		System.err.println("Forge.remove().A");
    		queuedMatches.remove(m);
    	} else if (runningMatches.contains(m)){
    		System.err.println("Forge.remove().B");
    		m.terminate();
    		System.err.println("Forge.remove().B2");
    		machines.returnMachines(m);
    		System.err.println("Forge.remove().B3");
    		runningMatches.remove(m);
    	}
    	System.err.println("Forge.remove().C");
    	showStatus();
    }
    
	/**
	 * All matches that have finished have their assigned machines 
	 * get returned to the machine room and the matches themselves
	 * are removed from the runningMatches.
	 */
	public void completeFinishedMatches() {
		LinkedList<MatchInterface> finishedMatches = new
		LinkedList<MatchInterface>();
		for (MatchInterface m : runningMatches) {
		  if (m.isComplete()){
		    finishedMatches.add(m);
		  }
		}

		for (MatchInterface m : finishedMatches) {
			machines.returnMachines(m);
			runningMatches.remove(m);
		}
	}

        /**
	 * Find a queued match that can be started, remove
	 * it from the queue, start it, and add it to the
	 * running matches.
	 *
	 * If there are no queued matches that can be started,
	 * do nothing.
	 */
        public void startQueuedMatch(){
        	MatchInterface m = machines.chooseMatchToStart(queuedMatches);
        	if (m!=null){
        	queuedMatches.remove(m);
    	  machines.assignMachines(m);
    	  System.out.println("Starting match...");
    	  System.out.println(m);
    	  m.startMatch();
    	  runningMatches.add(m);
    	  System.out.println("Match started");
        	}
        	
	    /*
	    System.out.print("Failed to start match...");
	    if (queuedMatches.size()==0){
	      System.out.println("No queued matches");
	    } else if (machines.getNumMachines()==0){
	      System.out.println("No machines");
	    } else {
	      System.out.println("Machines and queued matches incompatible");
	    }
	    System.out.println("Queued matches:"+queuedMatches.size());
	    System.out.println("Number of machines:"+machines.getNumMachines());
	    System.out.println("Running matches:"+runningMatches.size());
	    */
	    showStatus();
	}
       
    public void showStatus(){
	    try{
	    	FileOutputStream fos = new FileOutputStream( machines.statusFileLocation );
	    	PrintStream ps = new PrintStream(fos);
	    	showStatus(ps);
	    	ps.close();
	    } catch (IOException io){
	    }
    }

	public static void showDate(PrintStream ps){
		DateFormat df = DateFormat.getDateInstance();
    	DateFormat tf = DateFormat.getTimeInstance();
    	Date d = new Date();
    	ps.println(df.format(d)+" "+tf.format(d));
	}
    
    public void showStatus(PrintStream out){
    	showDate(out);
    	out.println("Queued Matches");
    	for(MatchInterface match:queuedMatches){
    		out.println(match);
    	}
    	out.println("Running Matches");
    	for(MatchInterface match:runningMatches){
    		out.println(match);
    		out.println("Hand Number:"+match.getHand());
    	}
    	machines.showStatus(out);
    	out.println("Available Machines:"+machines.getNumMachines());
    }

        /**
	 * Run the tournament.
	 * While the tournament is not over:
         * Load matches from the node into the queue.
         * Test to see if any matches have been completed. If so, the
         * resources used are returned to the MachineRoom.
         * Test to see if any matches from the queue can be run with the
         * machines currently in the MachineRoom. If so, the resources are taken
         * from the MachineRoom and the match is removed from the queue and
         * placed in the set of running matches.
	 */
     public void runTournament(){
	  while(!node.isComplete()){
	    System.out.println("loading new matches into the queue");
	    showDate(System.err);
	    
	    node.load(this);
	    System.err.println("clearing out finished matches");
	    showDate(System.err);
	    
	    completeFinishedMatches();
	    System.err.println("finding something to do");
	    showDate(System.err);
	    
	    startQueuedMatch();
	    System.err.println("Finished trying to queue");
	    
	    showDate(System.err);
	    try{
	    Thread.sleep(5000);
	    }catch (InterruptedException i){
	    	return;
	    }
	  }
	  System.out.println("Important matches complete: Shutting down matches");
	  while(runningMatches.size()>0){
	    System.out.println("clearing out finished matches");
	    completeFinishedMatches();
	    try{
	    Thread.sleep(5000);
	    }catch (InterruptedException i){
	    	return;
	    }
          }

	  System.out.println("Tournament complete");
	  node.showStatistics();
	  System.exit(0);
	}

	

	/**
	 * Shows queued and running matches.
	 */
	public String toString() {
		String ret = "Forge: ";
		
		ret += "running = " + runningMatches.size() + " : ";
		for (int i = 0; i < runningMatches.size(); i++) {
			ret += ((runningMatches.get(i)).toString()+"\n");
		}
		
		ret += "queued = " + queuedMatches.size();
		for (int i = 0; i < runningMatches.size(); i++) {
			ret += ((queuedMatches.get(i)).toString()+"\n");
		}
		
		return ret;
	}

    /**
	 * Runs a competition from a profile file
	 */
	public static void main(String[] args) throws IOException{
	  if (args.length==1){
	    Profile profile = new Profile(args[0]);
	    System.out.println(profile.node.toString());
	  } else if ((args.length==2)&&(args[1].equals("generateCards"))){
	    Profile profile = new Profile(args[0]);
	    profile.node.generateCardFiles(new SecureRandom());
	  } else if ((args.length==2)&&(args[1].equals("confirmCards"))){
		    Profile profile = new Profile(args[0]);
		    if (profile.node.confirmCardFiles()){
		    	System.err.println("Card files present and accounted for.");
		    } else {
		    	System.err.println("Some card files missing.");
		    }
		  } 
	  else if ((args.length==2)&&(args[1].equals("runTournament"))){
	    Profile profile = new Profile(args[0]);
	    Forge f = profile.getForge();
	    f.runTournament();
	  } 
	  else {
	    System.out.println("Usage:");
	    System.out.println("java <profilefile>");
	    System.out.println("java <profilefile> generateCards");
	    System.out.println("java <profilefile> runTournament");
	  }
	} 

}
