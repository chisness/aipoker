package ca.ualberta.cs.poker.free.alien;
import ca.ualberta.cs.poker.free.server.*;
import ca.ualberta.cs.poker.free.tournament.*;

import java.net.*;
import java.io.*;
import java.util.*;


/**
 * AlienAgent.java
 * The alien's "agent" on the server side. 
 * Sends messages to the alien and
 * receives messages from the alien.
 * Messages sent:<BR>
 * CLEANMACHINE:&lt;description&gt;<BR>
 * ASSIGNMACHINE:&lt;description&gt; (always sent before ASSIGNBOT)<BR>
 * ASSIGNBOT:&lt;name&gt;:&lt;serverIP&gt;:&lt;port&gt;(always sent after ASSIGNMACHINE)<BR>
 * MATCHSTARTED:&lt;matchname&gt;<BR>
 * MATCHTERMINATE:&lt;matchname&gt;<BR>
 * MATCHCOMPLETE:&lt;matchname&gt;<BR>
 * ERROR:&lt;message&gt;<BR>
 * SUCCESS<BR>
 * Messages received:<BR>
 * LOGIN:&lt;username&gt;:&lt;password&gt;<BR>
 * CREATEMACHINE:&lt;description&gt;<BR>
 * CREATEBOT:&lt;description&gt;<BR>
 * MATCHREQUEST:&lt;gametype&gt;:&lt;alienBotName&gt;:&lt;opponentBotName&gt;<BR>
 * MATCHTERMINATE:&lt;matchname&gt;<BR>
 * CHANGEPASSWORD:&lt;username&gt;:&lt;newpassword&gt;<BR>
 * LOGOUT<BR>
 * SHUTDOWN<BR>
 * ADDUSER:&lt;teamname&gt;:&lt;username&gt;:&lt;newpassword&gt;:&lt;email&gt;:&lt;accountType&gt;<BR>
 *
 * TODO Make sure the match queue is synchronized. Perhaps add a synchronized
 * method for adding matches?
 */
public class AlienAgent extends TimedSocket implements Runnable{

	static String tempRoot = "data/temp/";
	/**
	 * Unique agent name, used to append to file names.
	 */
	String agentName;

	/**
	 * Initialized to false
	 * When true, AlienAgent will be destroyed.
	 */
	boolean complete;

	/**
	 * Bots that have been introduced (in AlienBot form)
	 */
	Hashtable<String,AlienBot> bots;

	PrintStream out;

	/**
	 * Matches which have been completed.
	 */
	Vector<MatchInterface> completedMatches;

	/**
	 * Matches which have been queued.
	 */
	Vector<MatchInterface> queuedMatches;


	/**
	 * Machines which have been added.
	 */
	Vector<MachineInterface> queuedMachines;

	/**
	 * The user account associated with this agent.
	 * An account can have more than one agent associated with it.
	 */
	AlienAccount account;

	/**
	 * The parent node.
	 */
	AlienNode parent;

	/**
	 * Creates a new instance of AlienAgent;
	 */
	public AlienAgent(Socket socket, AlienNode parent) throws SocketException,
			 IOException {
				 super(socket);
				 System.err.println("A new AlienAgent has been created.");
				 this.parent = parent;
				 this.account = null;
				 this.complete = false;
				 this.agentName = parent.getNewAgentName();
				 completedMatches = new Vector<MatchInterface>();
				 queuedMatches = new Vector<MatchInterface>();
				 queuedMachines = new Vector<MachineInterface>();
				 bots = new Hashtable<String, AlienBot>();
				 out = new PrintStream(new FileOutputStream(agentName+".in.txt"));
				 Thread t = new Thread(this);
				 t.start();
			 }


	/**
	 * Sends an error message, then close the connection.
	 * 
	 * May requires AlienNode,AlienAgent token for suicide()
	 */
	public void sendError(String error){
		try{
			System.err.println("Sent error "+error);
			out.println("ERROR OBSERVED:"+error);
			sendMessage("ERROR:"+error);
		} catch(TimeoutException to){
		}

		suicide();
	}

	/**
	 * Sends a message to assign a bot.<BR>
	 * ASSIGNBOT:&lt;name&gt;:&lt;serverIP&gt;:&lt;port&gt;(always sent after ASSIGNMACHINE)<BR>
	 * @see AlienClient#processAssignBotMessage(String)
	 * @param name name of the bot
	 * @param serverIP the IP of the poker server for the match
	 * @param port the port of the poker server
	 * 
	 * May requires AlienNode,AlienAgent token for suicide()
	 */
	public void sendAssignBot(String name, InetAddress serverIP, int port){
		try{
			sendMessage("ASSIGNBOT:"+name+":"+serverIP.getHostAddress()+":"+port);
		} catch (TimeoutException to){
			out.println("SUICIDE:sendAssignBot");
			suicide();
		}
	}

	/**
	 * Sends a message to assign a machine.<BR>
	 * ASSIGNMACHINE:&lt;description&gt;<BR>
	 * Always followed by ASSIGNBOT.<BR>
	 * @see AlienClient#processAssignMachineMessage(String)
	 * @param description a description of the machine
	 * 
	 * May requires AlienNode,AlienAgent token for suicide()
	 */
	public void sendAssignMachine(String description){
		try {
			sendMessage("ASSIGNMACHINE:"+description);
		} catch (TimeoutException e) {
			out.println("SUICIDE:sendAssignMachine");
			suicide();
		}
	}
	/**
	 * NOTE: at present this function is not called.
	 * Send a message of the form:<BR>
	 * MATCHSTARTED:&lt;matchname&gt;<BR>
	 * @see AlienClient#processMatchStartMessage(String)
	 * @param matchName the name of the match
	 * 
	 * May requires AlienNode,AlienAgent token for suicide()
	 */
	public void sendMatchStarted(String matchName){
		try{
			sendMessage("MATCHSTARTED:"+matchName);
		} catch (TimeoutException te){
			out.println("SUICIDE:sendMatchStarted:"+te);
			suicide();
		}
	}

	/**
	 * Adds a queued match. However, if the Agent has been terminated,
	 * silently fails.
	 * @param match the match to add.
	 * 
	 * Requires AlienNode token, and AlienAgent token
	 */
	public boolean addQueuedMatch(MatchInterface match) {
		//System.out.println("AlienAgent.addQueuedMatch()");
		if (!complete) {
			if (!parent.generateCardFile(match)){
				return false;
			}

			addQueuedMatchHelper ( match );

			parent.pushBack(match);
		}
		return true;
	}

	/*
	 * Requires AlienAgent token
	 * 
	 */
	public synchronized void addQueuedMatchHelper ( MatchInterface match ) {
		queuedMatches.add(match);
	}

	/**
	 * Complete the match
	 * 
	 * @param match
	 *            the match that is completed
	 * 
	 * Require AlienAgent token
	 * 
	 */
	public synchronized boolean handleCompleteMatch(MatchInterface match){
		if (!completedMatches.contains(match)){
			completedMatches.add(match);
			return true;
		}else
			return false;


	}

	/**
	 * Sends a message that the match is complete.<BR>
	 * MATCHCOMPLETE:matchname<BR>
	 * @see AlienClient#processMatchComplete(String)
	 * @param matchname the name of the match
	 * 
	 * May requires AlienNode,AlienAgent token for suicide()
	 */ 
	public void sendMatchComplete(String matchname){
		try{
			sendMessage("MATCHCOMPLETE:"+matchname);
		} catch (TimeoutException te){
			out.println("SUICIDE:sendMatchComplete:"+te);
			suicide();
		}
	}

	/**
	 * Initialize account.
	 * If the message received is not a login message,
	 * or the login info is incorrect, then logout.
	 * LOGIN:&lt;username&gt;:&lt;password&gt;<BR>
	 * @throws TimeoutException 
	 * 
	 * May requires AlienAgent,AlienNode token
	 */
	public boolean login() throws TimeoutException, InterruptedException{
		// TODO Should this test three times or just once?
		for(int i=0;i<3;i++){
			String str = receiveMessage();
			Vector<String> data = parseByColons(str);
			/*System.out.println("data.size()="+data.size());
			  for(int j=0;j<data.size();j++){
			  System.out.println(data.get(j));
			  }*/
			if (data.size()!=3){
				sendError("Expected login:<username>:<password>, received "+str);
				return false;
			}
			if (!data.get(0).equals("LOGIN")){
				sendError("Expected login:<username>:<password>, received "+str);
				return false;
			}
			String username = data.get(1);
			String password = data.get(2);
			account = parent.testLogin(username,password);
			if (account!=null){
				sendMessage("SUCCESS");
				return true;
			}
			try{
				System.err.println("Failed login:"+str);
				sendMessage("ERROR:Login incorrect:please try again");
			} catch (TimeoutException te){
				out.println("Timeout error(login):"+te);
				suicide();
				return false;
			}
		}
		sendError("Too many attempts at a login");
		return false;
	}

	/*
	 * May requires AlienNode token for suicide()
	 */
	public boolean isLegalAlienBotName(String alienBotName){
		if (alienBotName.contains(".")){
			sendError("No periods allowed in bot names");
			return false;
		} else if (parent.getOpponent(alienBotName)!=null){
			sendError("Alien bots cannot have the same names as opponents");
			return false;
		}
		return true;
	}

	/**
	 * Processes a message in the middle.
	 * @param message
	 * 
	 * might require AlienAgent and AlienNode token for processMatchRequestMessage(message)
	 */
	public void processMessage(String message){
		if (message.equals("LOGOUT")){
			out.println("SUICIDE:processMessage");
			suicide();
			return;
		} else if (message.startsWith("MATCHREQUEST:")){
			processMatchRequestMessage(message);
		} else if (message.startsWith("CREATEBOT:")){
			processCreateBotMessage(message);
		} else if (message.startsWith("CREATEMACHINE:")){
			processCreateMachineMessage(message);
		} else if (message.startsWith("MATCHTERMINATE:")){
			processMatchTerminateMessage(message);
		}	else if (message.startsWith("CHANGEPASSWORD:")){
			processChangePasswordMessage(message);
		} else if (message.startsWith("ADDUSER:")){
			processAddUserMessage(message);
		} else if (message.equals("SHUTDOWN")){
			processShutdownMessage();
		} else {
			sendError("Unknown message");
		}
	}

	/**
	 * @see AlienClient#sendMatchTerminate(String)
	 * Process a request to terminate a match.
	 * The name of the match is prepended with the session name.<BR>
	 * MATCHTERMINATE:&lt;matchname&gt;<BR>
	 * @param message
	 * 
	 * Requires AlienAgent token, May require AlienNode token
	 */
	public void processMatchTerminateMessage(String message) {
		Vector<String> fields = parseByColons(message);
		String name = fields.get(1);
		name = account.username + "." + agentName+"."+name; 
		MatchInterface matchFound = processMatchTerminateMessageHelper(name);
		if (matchFound!=null){
			parent.killMatch(matchFound);
			sendMatchTerminate(name);
		}
	}

	/**
	 * Requires AlienAgent token
	 * @param name
	 * @return
	 */

	public synchronized MatchInterface processMatchTerminateMessageHelper(String name) {

		for(int i=0;i<queuedMatches.size();i++){
			MatchInterface match = queuedMatches.get(i);
			if (match.getName().equals(name)){
				queuedMatches.remove(i);
				return match;
			}
		}
		return null;
	}


	/*
	 * May requires AlienNode,AlienAgent token for suicide()
	 */
	private void sendMatchTerminate(String name) {
		try{
			sendMessage("MATCHTERMINATE:"+name);
		} catch (TimeoutException te){
			out.println("SUICIDE:sendMatchTerminate:"+te);
			suicide();
		}
	}


	/***
	 * Requires AlienAgent, AlienNode token
	 * @param machine
	 * 
	 */
	public void addMachine(AlienMachine machine){
		if (!complete){
			parent.add(machine);
			addMachineHelper(machine);
		}
	}

	public synchronized void addMachineHelper(AlienMachine machine){
		queuedMachines.add(machine);		
	}
	/**
	 * Process a request to create a machine.<BR>
	 * CREATEMACHINE:&lt;description&gt;
	 * 
	 * @param message
	 * 
	 * Requires AlienNode token
	 * Requires AlienAgent token
	 */
	public void processCreateMachineMessage(String message) {
		Vector<String> fields = parseByColons(message);
		String description = fields.get(1);
		try {
			AlienMachine machine = new AlienMachine(this, description);
			addMachine(machine);
		} catch (IOException io) {
			out.println("SUICIDE:processCreateMachineMessage");
			suicide();
		}
	}




	/**
	 * Process a request to create a bot.<BR>
	 * CREATEBOT:&lt;description&gt;
	 * 
	 * @param message
	 * 
	 * May requires AlienNode,AlienAgent token
	 * 
	 */
	public void processCreateBotMessage(String message) {
		Vector<String> fields = parseByColons(message);
		if (fields.size()!=2){
			sendError("Expected CREATEBOT:<description>, received "+message);
			out.println("SUICIDE:processCreateBotMessage:0");
			suicide();
			return;
		}
		String description = fields.get(1);
		try {
			AlienBot bot = new AlienBot(this, description);
			bots.put(bot.getName(), bot);
		} catch (IOException io) {
			out.println("SUICIDE:processCreateBotMessage:1");
			suicide();
		}
	}

	/*
	 * Requires AlienNode,AlienAgent token for parent.startShutdown();
	 */
	public void processShutdownMessage(){
		if (account.superuser){
			try{
				sendMessage("SUCCESS");
			} catch (TimeoutException to){

			}
			parent.startShutdown();
		} else {
			sendError("Not superuser: cannot shutdown system");
		} 
	}

	/**
	 * requires AlienNode token
	 * may require AlienAgent token
	 * @param message
	 */
	public void processChangePasswordMessage(String message){
		Vector<String> fields = parseByColons(message);
		System.err.println("CHANGEPASSWORD message received:"+message);
		if (fields.size()!=3){
			sendError("Expected CHANGEPASSWORD:<account>:<password>, received "+message);
			out.println("SUICIDE:processChangePasswordMessage:0");
			suicide();
			return;
		}
		String accountName = fields.get(1);
		String password = fields.get(2);
		AlienAccount otherAccount = parent.getAccount(accountName);
		if (otherAccount==null){
			sendError("No such user:"+accountName);
			out.println("SUICIDE:processChangePasswordMessage:1");
			suicide();
			return;
		}
		if (!account.superuser){
			if (!account.teamLeader){
				if (account!=otherAccount){
					sendError("Insufficient permission");
					out.println("SUICIDE:processChangePasswordMessage:2");
					suicide();
					return;
				}
			} else if (account.team.equals(otherAccount.team)){
				sendError("Different team");
				out.println("SUICIDE:processChangePasswordMessage:3");
				suicide();
				return;
			}
		}


		parent.changePassword(accountName,password);
		try{
			sendMessage("SUCCESS");
		} catch (TimeoutException te){
			out.println("SUICIDE:processChangePasswordMessage:4:"+te);
			suicide();
		}
	}

	/**
	 * Receive a message in the normal loop.
	 * Could be MATCHREQUEST, CREATEBOT, CREATEMACHINE,
	 * MATCHTERMINATE, or LOGOUT.
	 *
	 * might require AlienAgent token and AlienNode token for processMessage(str)
	 */
	public void receiveNormalMessage(){
		try{
			String str = receiveMessage();
			processMessage(str);
		} catch (TimeoutException te){
			out.println("SUICIDE:receiveNormalMessage:"+te);
			suicide();
			return;
		} catch (InterruptedException ie){
			suicide();
			return;
		}

	}


	/**
	 * Receive a match, and then send it to the AlienNode. using
	 * AlienNode.pushBack(). Note that the REVERSE of the match is also added.
	 * 
	 * Requires AlienAgent token, and AlienNode token
	 * 
	 */
	public void processMatchRequestMessage(String str){
		// System.err.println("AlienNode.processMatchRequestMessage("+str+")");
		// MATCHREQUEST:<gametype>:<matchname>:<alienbotname>:<opponentname>
		Vector<String> mess = parseByColons(str);
		if (!mess.get(0).equals("MATCHREQUEST")){
			sendError("Expected match request and received "+str);
			return;
		}
		if (mess.size()<2){
			sendError("Wrong number of parameters in " + str);
			return;    	  
		}
		String gameType = mess.get(1);
		if (gameType.equals("HEADSUPLIMIT")||gameType.equals("HEADSUP")){
			if (mess.size()!=5){
				sendError("Wrong number of parameters in " + str);
				return;
			}
			String matchName = mess.get(2);
			String alienBotName = mess.get(3);
			AlienBot alienBot = bots.get(alienBotName);
			if (alienBot==null){
				sendError("Unknown alien bot request:"+alienBotName);
				return;
			}
			String opponentBotName = mess.get(4);
			BotInterface opponentBot = parent.getOpponent(opponentBotName);
			if (opponentBot==null){
				sendError("Unknown opponent bot:"+opponentBotName);
				return;
			}

			InetAddress serverIP = parent.getServerIP();

			Vector<BotInterface> forwardBots = new Vector<BotInterface>();
			forwardBots.add(alienBot);
			forwardBots.add(opponentBot);
			String baseName = account.username +"." + agentName + "." + matchName;
			String cardFileName = matchName + ".crd";
			String forwardMatchName = baseName + "."+alienBotName+"."+opponentBotName;

			HeadsUpMatch forwardMatch = new
				HeadsUpMatch(forwardBots,cardFileName,serverIP,forwardMatchName,parent.getOpponentMatchType(opponentBot));
			//System.out.println("About to queue match");
			if (!addQueuedMatch(forwardMatch)){
				sendError("ERROR:card name used for a different type of game");
			}

			String reverseMatchName = baseName + "."
				+opponentBotName + "." + alienBotName;
			Vector<BotInterface> reverseBots = new Vector<BotInterface>();
			reverseBots.add(opponentBot);
			reverseBots.add(alienBot);
			HeadsUpMatch reverseMatch = new
				HeadsUpMatch(reverseBots,cardFileName,serverIP,reverseMatchName,parent.getOpponentMatchType(opponentBot));
			if (!addQueuedMatch(reverseMatch)){
				sendError("ERROR:card name used for a different type of game");
			}
		} else if (gameType.equals("RING2SERVER")) {
			if (mess.size()!=6){
				sendError("Wrong number of parameters in " + str);
				return;
			}
			String matchName = mess.get(2);
			String alienBotName = mess.get(3);
			AlienBot alienBot = bots.get(alienBotName);
			if (alienBot==null){
				sendError("Unknown alien bot request:"+alienBotName);
				return;
			}
			String opponentBotName = mess.get(4);
			BotInterface opponentBot = parent.getOpponent(opponentBotName);
			if (opponentBot==null){
				sendError("Unknown opponent bot:"+opponentBotName);
				return;
			}
			String opponentBotName2 = mess.get(5);
			BotInterface opponentBot2 = parent.getOpponent(opponentBotName2);
			if (opponentBot2==null){
				sendError("Unknown opponent bot:"+opponentBotName2);
				return;
			}

			InetAddress serverIP = parent.getServerIP();

			Vector<BotInterface> match1Bots = new Vector<BotInterface>();
			match1Bots.add(alienBot);
			match1Bots.add(opponentBot);
			match1Bots.add(opponentBot2);
			String baseName = account.username +"." + agentName + "." + matchName;
			String cardFileName = matchName + ".crd";
			String match1MatchName = baseName + "."+alienBotName+"."+opponentBotName+"."+opponentBotName2;
			RingLimitMatch match1 = new
				RingLimitMatch(match1Bots,cardFileName,serverIP,match1MatchName,parent.getOpponentMatchType(opponentBot2));
			//System.out.println("About to queue match");
			if (!addQueuedMatch(match1)){
				sendError("ERROR:card name used for a different type of game");
			}

			Vector<BotInterface> match5Bots = new Vector<BotInterface>();
			match5Bots.add(opponentBot2);
			match5Bots.add(alienBot);
			match5Bots.add(opponentBot);
			String match5MatchName = baseName + "."+opponentBotName2+"."+alienBotName+"."+opponentBotName;
			RingLimitMatch match5 = new
				RingLimitMatch(match5Bots,cardFileName,serverIP,match5MatchName,parent.getOpponentMatchType(opponentBot2));
			//System.out.println("About to queue match");
			if (!addQueuedMatch(match5)){
				sendError("ERROR:card name used for a different type of game");
			}

			Vector<BotInterface> match2Bots = new Vector<BotInterface>();
			match2Bots.add(alienBot);
			match2Bots.add(opponentBot2);
			match2Bots.add(opponentBot);
			String match2MatchName = baseName + "."+alienBotName+"."+opponentBotName2+"."+opponentBotName;
			RingLimitMatch match2 = new
				RingLimitMatch(match2Bots,cardFileName,serverIP,match2MatchName,parent.getOpponentMatchType(opponentBot2));
			//System.out.println("About to queue match");
			if (!addQueuedMatch(match2)){
				sendError("ERROR:card name used for a different type of game");
			}

			Vector<BotInterface> match3Bots = new Vector<BotInterface>();
			match3Bots.add(opponentBot);
			match3Bots.add(alienBot);
			match3Bots.add(opponentBot2);
			String match3MatchName = baseName + "."+opponentBotName+"."+alienBotName+"."+opponentBotName2;
			RingLimitMatch match3 = new
				RingLimitMatch(match3Bots,cardFileName,serverIP,match3MatchName,parent.getOpponentMatchType(opponentBot2));
			//System.out.println("About to queue match");
			if (!addQueuedMatch(match3)){
				sendError("ERROR:card name used for a different type of game");
			}

			Vector<BotInterface> match4Bots = new Vector<BotInterface>();
			match4Bots.add(opponentBot);
			match4Bots.add(opponentBot2);
			match4Bots.add(alienBot);
			String match4MatchName = baseName + "."+opponentBotName+"."+opponentBotName2+"."+alienBotName;
			RingLimitMatch match4 = new
				RingLimitMatch(match4Bots,cardFileName,serverIP,match4MatchName,parent.getOpponentMatchType(opponentBot2));
			//System.out.println("About to queue match");
			if (!addQueuedMatch(match4)){
				sendError("ERROR:card name used for a different type of game");
			}

			Vector<BotInterface> match6Bots = new Vector<BotInterface>();
			match6Bots.add(opponentBot2);
			match6Bots.add(opponentBot);
			match6Bots.add(alienBot);
			String match6MatchName = baseName + "."+opponentBotName2+"."+opponentBotName+"."+alienBotName;
			RingLimitMatch match6 = new
				RingLimitMatch(match6Bots,cardFileName,serverIP,match6MatchName,parent.getOpponentMatchType(opponentBot2));
			//System.out.println("About to queue match");
			if (!addQueuedMatch(match6)){
				sendError("ERROR:card name used for a different type of game");
			}

		} else if (gameType.equals("RING2CLIENT")) {
			if (mess.size()!=6){
				sendError("Wrong number of parameters in " + str);
				return;
			}
			String matchName = mess.get(2);
			String alienBotName = mess.get(3);
			AlienBot alienBot = bots.get(alienBotName);
			if (alienBot==null){
				sendError("Unknown alien bot request:"+alienBotName);
				return;
			}
			String opponentBotName = mess.get(4);
			AlienBot opponentBot = bots.get(opponentBotName);
			if (opponentBot==null){
				sendError("Unknown alien bot request:"+opponentBotName);
				return;
			}
			String opponentBotName2 = mess.get(5);
			BotInterface opponentBot2 = parent.getOpponent(opponentBotName2);
			if (opponentBot2==null){
				sendError("Unknown opponent bot:"+opponentBotName2);
				return;
			}

			InetAddress serverIP = parent.getServerIP();

			Vector<BotInterface> match1Bots = new Vector<BotInterface>();
			match1Bots.add(alienBot);
			match1Bots.add(opponentBot);
			match1Bots.add(opponentBot2);
			String baseName = account.username +"." + agentName + "." + matchName;
			String cardFileName = matchName + ".crd";
			String match1MatchName = baseName + "."+alienBotName+"."+opponentBotName+"."+opponentBotName2;
			RingLimitMatch match1 = new
				RingLimitMatch(match1Bots,cardFileName,serverIP,match1MatchName,parent.getOpponentMatchType(opponentBot2));
			//System.out.println("About to queue match");
			if (!addQueuedMatch(match1)){
				sendError("ERROR:card name used for a different type of game");
			}

			Vector<BotInterface> match5Bots = new Vector<BotInterface>();
			match5Bots.add(opponentBot2);
			match5Bots.add(alienBot);
			match5Bots.add(opponentBot);
			String match5MatchName = baseName + "."+opponentBotName2+"."+alienBotName+"."+opponentBotName;
			RingLimitMatch match5 = new
				RingLimitMatch(match5Bots,cardFileName,serverIP,match5MatchName,parent.getOpponentMatchType(opponentBot2));
			//System.out.println("About to queue match");
			if (!addQueuedMatch(match5)){
				sendError("ERROR:card name used for a different type of game");
			}

			Vector<BotInterface> match2Bots = new Vector<BotInterface>();
			match2Bots.add(alienBot);
			match2Bots.add(opponentBot2);
			match2Bots.add(opponentBot);
			String match2MatchName = baseName + "."+alienBotName+"."+opponentBotName2+"."+opponentBotName;
			RingLimitMatch match2 = new
				RingLimitMatch(match2Bots,cardFileName,serverIP,match2MatchName,parent.getOpponentMatchType(opponentBot2));
			//System.out.println("About to queue match");
			if (!addQueuedMatch(match2)){
				sendError("ERROR:card name used for a different type of game");
			}

			Vector<BotInterface> match3Bots = new Vector<BotInterface>();
			match3Bots.add(opponentBot);
			match3Bots.add(alienBot);
			match3Bots.add(opponentBot2);
			String match3MatchName = baseName + "."+opponentBotName+"."+alienBotName+"."+opponentBotName2;
			RingLimitMatch match3 = new
				RingLimitMatch(match3Bots,cardFileName,serverIP,match3MatchName,parent.getOpponentMatchType(opponentBot2));
			//System.out.println("About to queue match");
			if (!addQueuedMatch(match3)){
				sendError("ERROR:card name used for a different type of game");
			}

			Vector<BotInterface> match4Bots = new Vector<BotInterface>();
			match4Bots.add(opponentBot);
			match4Bots.add(opponentBot2);
			match4Bots.add(alienBot);
			String match4MatchName = baseName + "."+opponentBotName+"."+opponentBotName2+"."+alienBotName;
			RingLimitMatch match4 = new
				RingLimitMatch(match4Bots,cardFileName,serverIP,match4MatchName,parent.getOpponentMatchType(opponentBot2));
			//System.out.println("About to queue match");
			if (!addQueuedMatch(match4)){
				sendError("ERROR:card name used for a different type of game");
			}

			Vector<BotInterface> match6Bots = new Vector<BotInterface>();
			match6Bots.add(opponentBot2);
			match6Bots.add(opponentBot);
			match6Bots.add(alienBot);
			String match6MatchName = baseName + "."+opponentBotName2+"."+opponentBotName+"."+alienBotName;
			RingLimitMatch match6 = new
				RingLimitMatch(match6Bots,cardFileName,serverIP,match6MatchName,parent.getOpponentMatchType(opponentBot2));
			//System.out.println("About to queue match");
			if (!addQueuedMatch(match6)){
				sendError("ERROR:card name used for a different type of game");
			}
		} else {
			sendError("ERROR:Must be type HEADSUPLIMIT, HEADSUP or RING");
			return;
		}
	}

	/*
	 * 
	 * require AlienAgent Token and AlienNode token for SendMatchComplete
	 */
	public void testCompletedMatches(){

		Vector<MatchInterface> queuedMatchCopy = getQueuedMatches();

		for(MatchInterface m:queuedMatchCopy){
			if (m.isComplete()){
				if ( handleCompleteMatch(m) ) {
					sendMatchComplete(m.getName());
				}
			}
		}
	}

	public synchronized Vector<MatchInterface> getQueuedMatches() {
		return new Vector<MatchInterface> ( queuedMatches );
	}

	/**
	 * tar and e-mail completed matches to competitor. <BR>
	 * function.
	 */
	public synchronized void tarCompletedMatches(){
		Vector<String> files=new Vector<String>();
		for(MatchInterface m:completedMatches){
			if (m instanceof RingLimitMatch) {
				RingLimitMatch m2 = (RingLimitMatch)m;
				files.add(m2.resultFile);
				files.add(m2.logFile);
			} else {
				HeadsUpMatch m2 = (HeadsUpMatch)m;
				files.add(m2.resultFile);
				files.add(m2.logFile);
			}		
		}
		if (files.isEmpty()){
			return;
		}

		// XXX put back in, probably works?
		// DEPRECATED since we're using TarAndWeb
		String tempDirectory = agentName;
		String tarFile = agentName +".tar";
		String destinationAddress = account.email;
		String subject = "Poker Server "+agentName+" Results";
		String body  = "The following tar file has the results of "+completedMatches.size()+" matches.\n";
		body += "The units is small blinds (one small bet is two small blinds).\n";
		body += "- Martin Zinkevich and Christian Smith\n";
		TarAndEmail tae = new TarAndEmail(subject, body, destinationAddress, 
				tempDirectory, tarFile, files);

		TarAndWeb taw = new TarAndWeb( account.username, agentName, files);

		try{
			taw.execute();
		} catch (IOException io){
			System.err.println("Error sending e-mail");
			io.printStackTrace(System.err);
		}

	}
	/**
	 * Remove pointer from parent.
	 * 
	 * Requires AlienAgent,AlienNode token
	 */
	public void suicide(){
		complete = true;
		tarCompletedMatches();
		parent.removeAgent(this);

		try{
			out.println("SUICIDE");
			out.close();
			close();
		} catch (IOException io){
		}
	}

	/*
	 * 
	 * might require AlienAgent and AlienNode token for receiveNormalMessage();
	 * 
	 */
	public void run() {
		try {
			open();

			if (login() == false) {
				return;
			}
		} catch (IOException io) {
			return;
		} catch (TimeoutException te) {
			return;
		} catch (InterruptedException ie){
			return;
		}

		while (true) {
			if (complete){
				return;
			}
			receiveNormalMessage();
			if (complete) {
				return;
			}
			try {
				Thread.sleep(5000);
			} catch (InterruptedException ie) {
				out.println("SUICIDE:run");
				suicide();
				return;
			}
		}
	}

	/**
	 * Recieves a message: waits a day for a response.
	 */
	public String receiveMessage() throws TimeoutException,InterruptedException{
		if (!complete){
			setTimeRemaining(24*60*60*1000);
			String result = super.receiveMessage();
			out.println("message received:"+result);
			out.flush();
			return result;
		}
		return "LOGOUT";
	}

	/**
	 * Sends a message.
	 * Waits 30 seconds for reception.
	 */
	public void sendMessage(String message) throws TimeoutException{
		if (!complete){
			//setTimeRemaining(30000);
			out.println("server reply:"+message);
			super.sendMessage(message);
		}
	}


	/**
	 * Sends a message to clean a particular machine.<BR>
	 * CLEANMACHINE:&lt;description&gt;
	 * @see AlienClient#processCleanMachineMessage(String)
	 * @param description
	 * 
	 * May requires AlienNode,AlienAgent token for suicide()
	 */
	public void sendCleanMachine(String description) {
		try{
			sendMessage("CLEANMACHINE:"+description);
		} catch (TimeoutException te){
			out.println("SUICIDE:sendCleanMachine:"+te);
			suicide();
		}
	}
	/**
	 * Add a user. Tests for permission.
	 * ADDUSER:&lt;teamname&gt;:&lt;username&gt;:&lt;newpassword&gt;:&lt;email&gt;:&lt;accountType&gt;<BR>
	 * 
	 * May requires AlienAgent,AlienNode token
	 */
	public void processAddUserMessage(String message){
		Vector<String> fields = parseByColons(message);
		if (fields.size()!=6){
			sendError("Wrong format");
			out.println("SUICIDE:processAddUserMessage");
			// Why is this here? A second suicide?
			suicide();
		}
		String teamname = fields.get(1);
		String username = fields.get(2);
		String password = fields.get(3);
		String email = fields.get(4);
		String accountType = fields.get(5);

		boolean superuser = accountType.equalsIgnoreCase("superuser");
		boolean teamLeader = accountType.equalsIgnoreCase("teamleader");
		if (!account.superuser){
			if (!account.teamLeader){
				sendError("Cannot add user unless team leader or superuser");
				return;
			}
			if (!teamname.equals(account.team)){
				sendError("Cannot add user for other team");
				return;
			}
			if (superuser){
				sendError("Only superusers can create superuser accounts");
				return;
			}
			AlienAccount existing = parent.getAccount(username);
			if (existing!=null){
				if (!existing.team.equals(account.team)){
					sendError("User exists for other team");
					//suicide();
					return;
				}
			}
		}
		AlienAccount result = new AlienAccount(username,password,teamname,email,teamLeader,superuser);
		parent.addAccount(result);
		try{
			sendMessage("SUCCESS");
		} catch (TimeoutException to){
		}
	}
}
