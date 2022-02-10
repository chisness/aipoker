package ca.ualberta.cs.poker.free.tournament;

import java.io.*;
import java.net.*;
import java.util.*;


/*
 * This generates machines from a profile file line.
 * Currently, only RemoteMachine is supported.
 * @author Martin Zinkevich
 * 
 */
public class MachineFactory{

  /**
   * Throws a runtime exception if another token is not found
   */
  public static void checkTokens(StringTokenizer st, String str) throws IOException{
    if (!st.hasMoreTokens()){
      throw new IOException("Error: cannot parse line:"+str);
    }
  }
  
/**
 * Generates a new machine from a string.
 */
public static MachineInterface generateMachine(String str) throws IOException{
  StringTokenizer st = new StringTokenizer(str);
  checkTokens(st,str);
  String machineType = st.nextToken();
  if (machineType.equals("RemoteMachine")){
    return generateRemoteMachine(st,str);
  } else if (machineType.equals("LocalMachine")){
    return generateLocalMachine(st,str);
  }
  
  throw new IOException("Unrecognized machine type in " + str);
}

  /**
   * Generates a RemoteMachine.
   * The format is:<BR>
   * RemoteMachine &lt;IP&gt; &lt;username&gt; &lt;expansionLocation&gt;
   * [WINDOWS|LINUX]
   */
  public static RemoteMachine generateRemoteMachine(StringTokenizer st,
  String str) throws IOException{
    checkTokens(st,str);
    InetAddress address=null;
    try{
      address = InetAddress.getByName(st.nextToken());
    } catch(java.net.UnknownHostException E){
      throw new IOException("Unknown host in "+str);
    }
    checkTokens(st,str);
    String username = st.nextToken();
    checkTokens(st,str);
    String expansionLocation = st.nextToken();
    checkTokens(st,str);
    boolean isWindows = st.nextToken().equalsIgnoreCase("WINDOWS");
    boolean shouldClean = false;
    boolean shouldRestart = false;
    if (st.hasMoreTokens()){
    	shouldClean = st.nextToken().equalsIgnoreCase("SHOULDCLEAN");
    }
    if (st.hasMoreTokens()){
    	shouldRestart = st.nextToken().equalsIgnoreCase("SHOULDRESTART");
    }
    return new
    RemoteMachine(address,username,expansionLocation,isWindows,shouldClean,shouldRestart);
  }

  /**
   * Generates a LocalMachine.
   * The format is:<BR>
   * LocalMachine &lt;IP&gt; &lt;expansionLocation&gt;
   * [WINDOWS|LINUX]
   */
  public static LocalMachine generateLocalMachine(StringTokenizer st,
  String str) throws IOException{
    checkTokens(st,str);
    InetAddress address=null;
    try{
      address = InetAddress.getByName(st.nextToken());
    } catch(java.net.UnknownHostException E){
      throw new IOException("Unknown host in "+str);
    }
    checkTokens(st,str);
    String expansionLocation = st.nextToken();
    checkTokens(st,str);
    boolean isWindows = st.nextToken().equalsIgnoreCase("WINDOWS");
    return new
    LocalMachine(address,expansionLocation,isWindows);
  }
} 
