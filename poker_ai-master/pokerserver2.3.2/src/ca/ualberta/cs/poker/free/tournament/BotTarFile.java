package ca.ualberta.cs.poker.free.tournament;

/**
 * NOTE: Bots must all be unique, ie not
 * be equal according to the equals() function.
 * For this bot, the tar file must already be
 * distributed to the client machine before
 * the program is run.
 * TODO: make location a relative variable,
 * to increase mobility.
 */
public class BotTarFile implements BotInterface{
  String name;
  String location;
  String internalLocation;
  boolean worksOnWindows;
  boolean worksOnLinux;
  boolean worksOnLocalWindows;
  boolean worksOnLocalLinux;

  public BotTarFile(String name, String location,String internalLocation,
  boolean worksOnWindows, boolean worksOnLinux, boolean
  worksOnLocalWindows, boolean worksOnLocalLinux){
    this.name = name;
	this.location = location;
    this.internalLocation = internalLocation;
    this.worksOnWindows = worksOnWindows;
    this.worksOnLinux = worksOnLinux;
    
    this.worksOnLocalWindows = worksOnLocalWindows;
    this.worksOnLocalLinux = worksOnLocalLinux;
  }
    
  /**
   * Returns the location of the file containing
   * the bot. Should be a tar file.
   */
  public String getLocation(){
    return location;
  }

  public void setLocation(String location){
	  this.location = location;
	  this.internalLocation = location;
  }
  /**
   * Returns the file to execute within the expanded tar file.
   */
  public String getInternalLocation(){
    return internalLocation;
  }


  /**
   * Tests if a particular machine supports this bot.
   */
  public boolean machineSupports(MachineInterface mi){
    if (mi instanceof RemoteMachine){
      RemoteMachine machine = (RemoteMachine)mi;
      if (machine.isWindows){
        return worksOnWindows;
      } else {
        return worksOnLinux;
      }
    } else if (mi instanceof LocalMachine){
      LocalMachine machine = (LocalMachine)mi;
      if (machine.isWindows){
        return worksOnLocalWindows;
      } else {
        return worksOnLocalLinux;
      }
    }
    return false;
  }

  public String toString(){
    String result = "BotTarFile "+name +" "+location+" "+internalLocation+" ";
    if (worksOnWindows){
      result += " WINDOWS";
    }
    if (worksOnLocalWindows){
      result += " LOCALWINDOWS";
    }
    if (worksOnLinux){
      result += " LINUX";
    }

    if (worksOnLocalLinux){
      result += " LOCALLINUX";
    }
    return result;
  }

  public String getName(){
    return name;
  }
  
  public int hashCode(){
	  return name.hashCode();
  }
  public boolean equals(Object obj){
	  if (obj instanceof BotInterface){
		  return (getName().equals(((BotInterface)obj).getName()));
	  }
	  return false;
  }
  
  public void setName(String name){
	  this.name=name;
  }
}
