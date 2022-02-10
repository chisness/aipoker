package ca.ualberta.cs.poker.free.alien;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.InetAddress;

import ca.ualberta.cs.poker.free.server.TimeoutException;

public class CLAlienClientRing extends AlienClient {
	AlienProfile profile;
	public void checkUsername() throws IOException{
	      if (username==null){
	        System.out.print("Login:");
		BufferedReader r = new BufferedReader(new
		InputStreamReader(System.in));
		username = r.readLine();
	      }
	    }

	public void checkPassword() throws IOException{
	      if (password==null){
	        System.out.print("Password for "+username);
		BufferedReader r = new BufferedReader(new
		InputStreamReader(System.in));
		password = r.readLine();
	      }
	    }

	public void loadBotsFromProfile() throws TimeoutException{
		addBots(profile.bots);		
	}

	public void loadMachinesFromProfile() throws TimeoutException{
		addMachines(profile.machines);
	}
	
	public boolean connectAndLoginFromProfile() throws IOException, TimeoutException, InterruptedException{
	      this.username = profile.username;
	      checkUsername();
	      this.password = profile.password;
	      checkPassword();		
	      connect(InetAddress.getByName(profile.addr),profile.port);  
	      return login();
	}

    public void set(AlienProfile profile){
    	this.profile = profile;
    }
    
    public void loadMatchesFromProfile() throws TimeoutException{
    	addMatches(profile.matches);
    }
    
    public CLAlienClientRing(String filename) throws IOException{
    	super();
    	set(new AlienProfile(filename));
     }

    /**
	 * Start the client.
	 * Still needed?
     * @throws InterruptedException 
	 */
    public void startClient() throws IOException,TimeoutException, InterruptedException{
      if (!connectAndLoginFromProfile()){
    	  close();
    	  return;
      }
      loadMachinesFromProfile();
      loadBotsFromProfile();
      loadMatchesFromProfile();
      
      
      while(true){
        processMessage(receiveMessage());
        System.err.println("Completed:");
        for(int i=0;i<completedMatchStrings.size();i++){
        	System.err.println(completedMatchStrings.get(i));
        }
        System.err.println("Queued:");
        for(int i=0;i<matches.size();i++){
        	System.err.println(matches.get(i));
        }
        if (completedMatchStrings.size()==matches.size()*6){
        	if (runningMachines.size()==0){
				System.err.println("matches.size() == " + matches.size() + ", completedMatchStrings == " + completedMatchStrings.size());
				System.err.println("All matches complete");
        	  sendLogout();
           	  close();
           	  return;
        	}
        }
        Thread.sleep(1000);
      }
    }
    
    public void changePassword(String account, String newpassword) throws TimeoutException, IOException, InterruptedException{
    	if (!this.connectAndLoginFromProfile()){
    		close();
    		return;
    	}
    	sendChangePassword(account,newpassword);
    	String message = receiveMessage();
    	System.err.println(message);
    }
    
    public void addUser(String teamName, String username, String password, 
    		String email, String accountType) throws TimeoutException, IOException, InterruptedException{
    	if (!this.connectAndLoginFromProfile()){
    		close();
    		return;
    	}
    	
    	sendAddUser(teamName, username, password, email, accountType);
    	String message = receiveMessage();
    	System.err.println(message);	
    }
    
    public void terminateMatch(String matchName) throws TimeoutException, IOException, InterruptedException{
    	if (!this.connectAndLoginFromProfile()){
    		close();
    		return;
    	}
    	sendMatchTerminate(matchName);
    }

    public void shutdown() throws TimeoutException,IOException, InterruptedException{
    	if (!this.connectAndLoginFromProfile()){
    		close();
    		return;
    	}
    	sendShutdown();
    	
    }

    public static void showUsage(){
    	System.err.println("Usage:AlienClientRing <profile>");
    	System.err.println("Usage:AlienClientRing <profile> passwd <accountname> <newpassword>");
    	System.err.println("Usage:AlienClientRing <profile> terminate <matchName> ");
    	System.err.println("Usage:AlienClientRing <profile> shutdown");
    	System.err.println("Usage:AlienClientRing <profile> adduser <teamName> <username> <password> <email> <accountType>");
    	System.err.println("where <accountType> is NORMAL|SUPERUSER|TEAMLEADER");
    	System.exit(0);
    }

    /**
     * @see #showUsage()
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, InterruptedException, TimeoutException{
    	if (args.length==0){
    		showUsage();
    	}
    	CLAlienClientRing client = new CLAlienClientRing(args[0]);
    	
    	if (args.length==1){
    		client.startClient();
    	} else if (args[1].equals("passwd")){
    		if (args.length!=4){
    			showUsage();
    			return;
    		}
    		client.changePassword(args[2], args[3]);
    	} else if (args[1].equals("terminate")){
    		if (args.length!=3){
    			showUsage();
    			return;
    		}
    		client.terminateMatch(args[2]);
    	} else if (args[1].equals("shutdown")){
    		if (args.length!=2){
    			showUsage();
    			return;
    		}
    		client.shutdown();
    	} else if (args[1].equals("adduser")){
    		if (args.length!=7){
    			showUsage();
    			return;
    		}
    		String teamName = args[2];
    		String username = args[3];
    		String password = args[4];
    		String email = args[5];
    		String accountType = args[6];
    		client.addUser(teamName,username,password, 
    	    		email, accountType);
    	} else {
    		showUsage();
    	}
    }

}
