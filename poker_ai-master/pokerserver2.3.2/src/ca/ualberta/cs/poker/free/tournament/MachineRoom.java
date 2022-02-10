package ca.ualberta.cs.poker.free.tournament;

import ca.ualberta.cs.poker.free.graph.*;

import java.io.PrintStream;
import java.util.LinkedList;
import java.util.Vector;

/**
 * A class representing all the machines that are currently
 * unused. 
 */
public class MachineRoom{
  private Vector<MachineInterface> machines;
  public String statusFileLocation = "status.txt";
  
  public MachineRoom(MachineRoom other){
	  machines = other.machines;
  }
  public MachineRoom(){
    machines = new Vector<MachineInterface>();
  }

  public void add(MachineInterface machine){
   machines.add(machine);
  }
  
  public void remove(MachineInterface machine){
	  machines.remove(machine);
  }

  public MatchInterface chooseMatchToStart(LinkedList<MatchInterface> matches){
	  for(MatchInterface m:matches){
	    	if (canPlay(m)){
	    		return m;
	    	}
	  }
	  return null;
  }
  /**
   * Finds machines for a match. If none can be found,
   * returns null.
   */ 
  private Vector<MachineInterface> getMachines(MatchInterface m){
    Vector<BotInterface> bots = m.getBots();
    /*for(BotInterface b:bots){
      System.out.println("MachineRoom.getMachines():Bot:"+b);
    }*/
    BipartiteGraph<BotInterface,MachineInterface> graph = new
    BipartiteGraph<BotInterface,MachineInterface>(
      new TestBotAndMachine(),bots,machines);
    return graph.getMatching();
  }

  public int getNumMachines(){
    return machines.size();
  }

  class TestBotAndMachine implements
  TestConnection<BotInterface,MachineInterface>{
    public boolean canConnect(BotInterface bot, MachineInterface machine){
      return bot.machineSupports(machine);
    }
  }

  public boolean canPlay(MatchInterface m){
    Vector<MachineInterface> result = getMachines(m);
    for(MachineInterface machine:result){
      if (machine==null){
        return false;
      }
    }
    return true;
  }

  /**
   * Removes the necessary machines from the machine
   * room and has the match useMachines.
   */
  public void assignMachines(MatchInterface m){
    Vector<MachineInterface> results = getMachines(m);
    machines.removeAll(results);
    m.useMachines(results);
  }

  public void returnMachines(MatchInterface match){
    for(MachineInterface machine:match.getMachines()){
      System.out.println("Cleaning machine:"+machine);
      machine.clean();
    }
    Vector<MachineInterface> newMachines = match.getMachines();
    machines.addAll(newMachines);
  }
  
  public String toString(){
	  String result = "";
	  for(MachineInterface machine:machines){
		  result += (machine+"\n");
	  }
	  return result;
  }

  public void showStatus(PrintStream ps){
  	  ps.println("Available Machines:"+getNumMachines());
	  for(MachineInterface machine:machines){
		  ps.println(machine);
	  }
  }
}

