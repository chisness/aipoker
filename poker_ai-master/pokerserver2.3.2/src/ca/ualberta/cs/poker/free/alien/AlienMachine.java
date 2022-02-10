package ca.ualberta.cs.poker.free.alien;
import ca.ualberta.cs.poker.free.tournament.*;
import java.net.InetAddress;
import java.io.*;

/**
 * A pseudo-machine that links to the client
 * who will start a match.
 */
public class AlienMachine implements MachineInterface{

  AlienAgent creator;
  AlienBot currentBot;
  MachineInterface internal;
  String description;


  /**
   * Construct a machine.
   */
  public AlienMachine(AlienAgent creator, String description) throws IOException{
    this.creator=creator;
    this.description = description;
    internal = MachineFactory.generateMachine(description);
    currentBot = null;
  }

  /**
   * Start a bot on this machine to connect to a server
   * at addr and port. First, sends an ASSIGNMACHINE message,
   * and then an ASSIGNBOT message.
   * 
   * May requires AlienNode,AlienAgent token for suicide(), only called from server side
   */
  public void start(BotInterface bot, InetAddress addr, int port){
    currentBot = (AlienBot)bot;
    creator.sendAssignMachine(description);
    currentBot.startBot(addr,port);
  } 

  /**
   * Get the IP of this machine.
   * If the internal machine is local, returns the IP address of the alien client instead.
   */
  public InetAddress getIP(){
	if (internal instanceof LocalMachine){
		return creator.socket.getInetAddress();
	}
    return internal.getIP();
  }

  /**
   * Gets whether or not this bot could have come from a particular IP address.
   * 
   */
  public boolean isThisMachine(InetAddress addr) {
	  return getIP().equals(addr);
  }
  
  public String toString(){
	  return "AlienMachine Creator:"+creator.account.team+" internal:" + internal.toString();
  }
  
  /**
   * Clean this machine.
   * 
   * May requires AlienNode,AlienAgent token for suicide()
   */
  public void clean(){
    creator.sendCleanMachine(description);
  }
}
