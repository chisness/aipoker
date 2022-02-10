package ca.ualberta.cs.poker.free.alien;

import ca.ualberta.cs.poker.free.tournament.*;

import java.util.*;

public class MachineLibrary extends MachineRoom {
	/**
	 * Records which team checked out a machine.
	 * When machines are returned, they are removed from the books that are 
	 * checked out.
	 */
    public Hashtable<String, Vector<MachineInterface> > checkedOut;
    
    LinkedList<String> teamQueue;
    
    /**
     * The total number of machines allowed for a particular team.
     */
    public Hashtable<String, Integer> budget;
    
    public MachineLibrary(MachineRoom room){
    	super(room);
    	init();
    }
    
    public MachineLibrary(){
    	super();
    	init();
    }
    
    public void init(){
    	checkedOut = new Hashtable<String, Vector<MachineInterface> >();
    	budget = new Hashtable<String, Integer>();
    	teamQueue = new LinkedList<String>();
    }
    /**
     * Informs the library of a new team with a specified budget.
     * @param teamName The name of the team
     * @param budget the budget of the team
     */
    public synchronized void addTeam(String teamName, int budget){
    	this.budget.put(teamName,budget);
    	this.checkedOut.put(teamName,new Vector<MachineInterface>());
    	this.teamQueue.add(teamName);
    }
    
    /**
     * Tests if a particular team has the resources to check out a particular
     * number of machines.
     * @param teamName the team who wishes to check out the machines.
     * @param numMachines the number of machines required to be run.
     * @return true if the team can check out the machines.
     */
    public synchronized boolean canCheckOut(String teamName, int numMachines){
    	if (teamName == null){
    		return true;
    	}
    	// If the team has no recorded budget, it cannot check out any machines.
    	if (!teamHasBudget(teamName)){
    		return false;
    	}
    	int teamBudget = budget.get(teamName);
    	int teamSpent = checkedOut.get(teamName).size();
    	return (teamSpent+numMachines<=teamBudget);
    }
    
    public String getTeamName(MatchInterface match){
    	Vector<BotInterface> bots = match.getBots();
    	for(BotInterface bot:bots){
    		if (bot instanceof AlienBot){
    			AlienBot ab = (AlienBot)bot;
    			return ab.creator.account.team;
    		}
    	}
    	return null;
    }
    
    public Vector<BotInterface> getLocalBotsNeeded(MatchInterface match){
    	Vector<BotInterface> bots = match.getBots();
    	Vector<BotInterface> result = new Vector<BotInterface>();
    	for(BotInterface bot:bots){
    		if (!(bot instanceof AlienBot)){
    			result.add(bot);
    		}
    	}
    	return result;
    }
    
    public boolean canPlay(MatchInterface match){
    	String teamName = getTeamName(match);
    	if (teamName==null){
    		return super.canPlay(match);
    	}
    	if (canCheckOut(teamName,getLocalBotsNeeded(match).size())){
    		return super.canPlay(match);
        }
    	/*System.err.println("Team "+teamName+ " at limit");
    	for(MachineInterface machine:checkedOut.get(teamName)){
    		System.err.println(machine);
    	}*/
    	
    	return false;
    }
    
    /**
     * Removes the necessary machines from the machine
     * room and has the match useMachines.
     * Also, checks out the machines on the relevant team account.
     */
    public void assignMachines(MatchInterface m){
      super.assignMachines(m);
      String teamName = getTeamName(m);
  	  if (teamName!=null){
        Vector<MachineInterface> usedMachines = m.getMachines();
        Vector<MachineInterface> localMachines = new Vector<MachineInterface>();
        for(MachineInterface machine:usedMachines){
    	  if (!(machine instanceof AlienMachine)){
    		  localMachines.add(machine);
    	  }
        }
        checkedOut.get(teamName).addAll(localMachines);
        teamQueue.remove(teamName);
        teamQueue.addLast(teamName);
  	  }
    }

	@Override
	public void returnMachines(MatchInterface match) {
		String teamName = getTeamName(match);
		if (teamName!=null){
			Vector<MachineInterface> teamCheckedOut = checkedOut.get(teamName);
			if (teamCheckedOut!=null){
				teamCheckedOut.removeAll(match.getMachines());
			}
		}
		super.returnMachines(match);
	}

	public synchronized boolean teamHasBudget(String teamName) {
		return (budget.get(teamName)!=null) &&(checkedOut.get(teamName)!=null);    	
	}   
    
	
	public MatchInterface chooseMatchToStart(LinkedList<MatchInterface> matches){
		Vector<MatchInterface> playable = new Vector<MatchInterface>();  
		for(MatchInterface m:matches){
		    	if (canPlay(m)){
		    		playable.add(m);
		    	}
		  }
		  if (playable.isEmpty()){
			  return null;
		  }
		  Hashtable<String,MatchInterface> teamsToPlay = new Hashtable<String,MatchInterface>();
		  for(MatchInterface m:playable){
			  String team = getTeamName(m);
			  if (!teamsToPlay.containsKey(team)){
				  teamsToPlay.put(team,m);
			  }
		  }
		  for(String team:teamQueue){
			  if (teamsToPlay.containsKey(team)){
				  return teamsToPlay.get(team);
			  }
		  }
		  return null;
	  }
}
