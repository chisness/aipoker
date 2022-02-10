package ca.ualberta.cs.poker.free.tournament;

import java.io.*;
import java.util.*;

/**
 * NOTE: Bots must all be unique, ie not
 * be equal according to the equals() function.
 */
public class BotFactory{


  /**
   * Throws a runtime exception if another token is not found
   */
  public static void checkTokens(StringTokenizer st, String str){
    if (!st.hasMoreTokens()){
      throw new RuntimeException("Error: cannot parse line:"+str);
    }
  }
  
/**
 * Generates a new bot from a string.
 * @see #generateBotTarFile(StringTokenizer, String)
 */
public static BotInterface generateBot(String str) throws IOException{
  StringTokenizer st = new StringTokenizer(str);
  checkTokens(st,str);
  String botType = st.nextToken();
  if (botType.equals("BotTarFile")){
    return generateBotTarFile(st,str);
  }
  throw new IOException("Unrecognized machine type in " + str);
}

  /**
   * Generates a BotTarFile.
   * The format is:<BR>
   * BotTarFile &lt;name&gt; &lt;location&gt; &lt;internalLocation&gt;
   * (WINDOWS|LINUX|LOCALLINUX|LOCALWINDOWS)+<BR>;
   * where + means one or more machines may be specified.
   */
  public static BotTarFile generateBotTarFile(StringTokenizer st,
  String str) throws IOException{
    checkTokens(st,str);
    String name = st.nextToken();
    checkTokens(st,str);
    String location = st.nextToken();
    checkTokens(st,str);
    String internalLocation = st.nextToken();
    checkTokens(st,str);
    boolean worksOnWindows = false;
    boolean worksOnLinux = false;
    boolean worksOnLocalWindows = false;
    boolean worksOnLocalLinux = false;

    do{
      String os = st.nextToken();
      if (os.equalsIgnoreCase("WINDOWS")){
        worksOnWindows = true;
      } else if (os.equalsIgnoreCase("LINUX")){
        worksOnLinux = true;
      } else if (os.equalsIgnoreCase("LOCALWINDOWS")){
        worksOnLocalWindows = true;
      } else if (os.equalsIgnoreCase("LOCALLINUX")){
        worksOnLocalLinux = true;
      } else {
        throw new IOException("Unexpected token in line "+str);
      }
    } while(st.hasMoreTokens());
    return new
    BotTarFile(name,location,internalLocation,worksOnWindows,worksOnLinux,
    worksOnLocalWindows,worksOnLocalLinux);
  }
}
