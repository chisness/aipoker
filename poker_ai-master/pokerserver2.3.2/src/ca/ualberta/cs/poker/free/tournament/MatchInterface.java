package ca.ualberta.cs.poker.free.tournament;

import java.util.Vector;
import java.security.SecureRandom;

/**
 * An abstraction for matches, such that if new
 * match types are created in the future, it is possible to easily
 * add them into the framework.
 * @author Martin Zinkevich
 *
 */
public interface MatchInterface{
  /**
   * Get the bots for the match.
   * @return a vector of the bots used in this match.
   */
  public Vector<BotInterface> getBots();
  
  /**
   * Gets the machines used for the match.
   * getMachines().get(i) is the machine for the bot getBots().get(i).
   * If no machines have been assigned, may return null.
   * @return a vector of the machines
   */
  public Vector<MachineInterface> getMachines();
  
  /**
   * Generate a card file for the match.
   * Does nothing if card file already exists.
   * @param random A source of randomness.
   */
  public void generateCardFile(SecureRandom random);
  
  /**
   * Confirm that the card file is present and correct.
   */
  public boolean confirmCardFile();
  
  /**
   * This function starts using machines.
   * It is called from MachineRoom.assignMachines()
   */
  public void useMachines(Vector<MachineInterface> machines);

  /**
   * Starts the match.
   */
  public void startMatch();
  
  /**
   * Tests if the match is complete.
   * A match is complete if it terminated normally or if there
   * was an error in some bot.
   * @return true if the match is complete, false otherwise.
   */
  public boolean isComplete();
  
  /**
   * Gets the name of the match.
   * @return name
   */
  public String getName();
  
  /**
   * Gets the utilities of the bots.
   * getUtilities().get(i) is the utility of getBots().get(i) bot.
   * In the future, this should be in small blinds.
   * @return a vector of utilities.
   */
  public Vector<Integer> getUtilities();
  
  public int getHand();
  /**
   * Terminates the threads for this match.
   */
  public void terminate();
} 
