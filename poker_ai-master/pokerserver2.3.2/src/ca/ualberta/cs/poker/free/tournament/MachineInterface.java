package ca.ualberta.cs.poker.free.tournament;
import java.net.InetAddress;

public interface MachineInterface{

  /**
   * Start a bot on this machine to connect to a server
   * at addr and port.
   */
  public void start(BotInterface bot, InetAddress addr, int port);

  /** 
   * Get the IP of this machine.
   */
  public InetAddress getIP();

  
  /**
   * Tests if the address could belong to this machine.
   * @param addr
   * @return
   */
  public boolean isThisMachine(InetAddress addr);
  
  /**
   * Clean this machine.
   */
  public void clean();
}
