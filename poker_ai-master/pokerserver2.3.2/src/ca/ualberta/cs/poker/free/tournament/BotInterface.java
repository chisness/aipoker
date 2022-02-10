package ca.ualberta.cs.poker.free.tournament;

/**
 * NOTE: Bots must all be unique, ie not
 * be equal according to the equals() function.
 * The toString() function must return a description of the
 * bot readable by BotFactory.
 */
public interface BotInterface{

  /**
   * Tests if a particular machine supports this bot.
   */
  public boolean machineSupports(MachineInterface mi);

  /**
   * Get the name of the bot. Must be unique.
   */
  public String getName();
}
