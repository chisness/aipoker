package ca.ualberta.cs.poker.free.alien;
import java.io.*;
import ca.ualberta.cs.poker.free.tournament.*;
import java.net.*;
//import javax.mail.*;
//import javax.mail.internet.*;
//import java.util.Properties;

/**
 * An alien bot represents a bot that will be run
 * on the client side. When passed to an AlienMachine, 
 * the alien machine connects back to the AlienAgent (and
 * AlienClient) through the AlienBot.
 * 
 * All bots must be unique, ie not
 * be equal to anything else according to the equals() function.
 */
public class AlienBot implements BotInterface{
  BotInterface internal;
  /**
   * Construct an AlienBot with the relevant clientIP, name,
   * and creator.
   */
  public AlienBot(AlienAgent creator, String internal) throws IOException{
    this.creator = creator;
    this.internal = BotFactory.generateBot(internal);
  }

  /**
   * A pointer to its creator
   * Needed to send messages for termination.
   */
  public AlienAgent creator;

  /**
   * Tests if a particular machine supports this bot.
   * The machine must be submitted by the same agent
   * that submitted the bot.
   */
  public boolean machineSupports(MachineInterface mi){
    if (mi instanceof AlienMachine){
      AlienMachine am = (AlienMachine)mi;
      if (am.creator==this.creator){
    	  return internal.machineSupports(am.internal);
      }
    }
    return false;
  }


  /** 
   * Start the bot by sending a message to AlienClient through
   * AlienAgent.
   * 
   * May require AlienNode,AlienAgent token
   */
  public void startBot(InetAddress server, int port){
    creator.sendAssignBot(getName(),server,port);
  }


  public String getName(){
    return internal.getName();
  }

  public String toString(){
	  return "AlienBot Creator:" + creator.account.team+ "internal:"+internal.toString();
  }
  public boolean equals(Object obj){
	  if (obj instanceof AlienBot){
		  AlienBot ab = (AlienBot)obj;
		  if (ab.creator==this.creator){
			  return this.internal.equals(ab.internal);
		  }
	  }
	  return false;
  }
}
