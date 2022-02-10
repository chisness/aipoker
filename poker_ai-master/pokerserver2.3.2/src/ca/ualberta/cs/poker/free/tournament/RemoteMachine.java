package ca.ualberta.cs.poker.free.tournament;

import java.io.*;
import java.net.*;
import java.util.Vector;

/*
 * This class contains the code to start a process on a remote machine
 * and clean up afterwards.
 * NOTE: Currently the server MUST be Windows.
 * In order for this to work, ssh must be installed on the server
 * and sshd must be installed on the client. The system is not designed
 * to handle passwords or authentication, thus keypairs must be exchanged
 * prior to the execution of this code, and hosts must be known to one 
 * another.
 * @author Martin Zinkevich
 * 
 */
public class RemoteMachine implements MachineInterface {
	/**
	 * Track the loose threads from this machine.
	 */
	Vector<Thread> looseThreads;

	/**
	 * Processes that need to be cleaned out
	 * at the end.
	 */
	Vector<Process> looseProcesses;

	/**
	 * The address of the remote machine.
	 */
	private InetAddress address;

	/**
	 * The username for ssh on the remote machine.
	 */
	private String username;

	/**
	 * The location which can be used by the bot.
	 */
	private String expansionLocation;

	/**
	 * Is true if the remote machine is Windows.
	 */
	public boolean isWindows;

	/**
	 * True if the server is windows.
	 */
	public boolean serverIsWindows=false;
	
	/**
	 * True if an aggressive clean should be made.
	 */
	private boolean shouldClean;
	
	/**
	 * true if we are going to restart the machine, and wait for it to come back up
	 */
	private boolean shouldRestart;

	/**
	 * Construct a new RemoteMachine.
	 * @param shouldRestart 
	 */
	public RemoteMachine(InetAddress address, String username,
			String expansionLocation, boolean isWindows, boolean shouldClean, boolean shouldRestart) {
		this.address = address;
		this.username = username;
		this.expansionLocation = expansionLocation;
		if (expansionLocation.equals("")) {
			throw new RuntimeException("Empty expansionLocation");
		}
		if (!(expansionLocation.endsWith("/")||expansionLocation.endsWith("\\"))){
			throw new RuntimeException("expansionLocation does not end with a directory delimiter.");
		}
		this.isWindows = isWindows;
		this.shouldClean = shouldClean;
		this.shouldRestart = shouldRestart;
		this.looseThreads = new Vector<Thread>();
		this.looseProcesses = new Vector<Process>();
	}

	/**
	 * get the IP address of the RemoteMachine.
	 */
	public InetAddress getIP() {
		return address;
	}

	/**
	 * Tests if the address to check is exactly equal to the address of the
	 * remote machine. Hopefully, the remote machine has only one IP address.
	 */
	public boolean isThisMachine(InetAddress addr) {
		return address.equals(addr);
	}

	
	/**
	 * Gets the remote location of the bot
	 * @param bot
	 * @return
	 */
	public String getRemoteLocation(BotTarFile bot){
		String local = bot.getLocation();
		// If there are no separators, lastSeparatorIndex is -1.
		int lastSeparatorIndex = Math.max(local.lastIndexOf('/'),local.lastIndexOf('\\'));
		String name = local.substring(lastSeparatorIndex+1);
		return expansionLocation + name;
	}
	
	/**
	 * This copies the bot from the server to the remote machine.
	 * 
	 * Note that the current implementation does not send the bot's tar file to
	 * an agreed upon location. This can be fixed in the future.
	 */
	public void copyFromServer(BotTarFile bot) {
		String scpCommand = "scp -r " + bot.getLocation() + " "
				+ username + "@" + address.getHostAddress() + ":"+getRemoteLocation(bot);
		System.err.println("scp command:");
		try {
			Process p = Runtime.getRuntime().exec(scpCommand);
			FileOutputStream normalOut = new FileOutputStream("out.txt");
			FileOutputStream errOut = new FileOutputStream("err.txt");

			looseProcesses.add(p);
			StreamConnect sc = new StreamConnect(p.getInputStream(), normalOut);
			Thread tsc = new Thread(sc);
			tsc.start();
			looseThreads.add(tsc);
			StreamConnect scerr = new StreamConnect(p.getErrorStream(), errOut);
			Thread tscerr = new Thread(scerr);
			tscerr.start();
			looseThreads.add(tscerr);
			p.waitFor();
		} catch (InterruptedException e) {
		} catch (IOException io) {
		}
	}

	/**
	 * This extracts the bot and connects to the server. However, it does this
	 * in a separate thread.
	 */
	public void extractAndPlay(BotTarFile bot, InetAddress server, int port) {
		String redirOutErr = " > " + expansionLocation + "out.txt 2> "
		+ expansionLocation + "err.txt ";
		String serverIP = server.getHostAddress();

		if (bot.getLocation().endsWith("\\")||bot.getLocation().endsWith("/")){
			String internalLocation = bot.getLocation();
			String executable = internalLocation + "startme."+ ((isWindows) ? "bat" : "sh");
			String cdCommand = "cd " + internalLocation;
			String exCommand = executable + " " + serverIP + " " + port
			+ redirOutErr;
			String jointCommand = cdCommand + ";" + exCommand;
			// Note that this does NOT wait for the command
			// to complete.
			executeRemoteCommand(jointCommand, true);
		} else {

		String tarFile = getRemoteLocation(bot);//bot.getLocation()
		String internalLocation = expansionLocation + bot.getInternalLocation();
		String executable = internalLocation + "startme."
				+ ((isWindows) ? "bat" : "sh");


		String tarCommand = "tar -xf " + tarFile + " -C " + expansionLocation;// +
																				// redirOutErr;
		String cdCommand = "cd " + internalLocation;
		String exCommand = executable + " " + serverIP + " " + port
				+ redirOutErr;

		copyFromServer(bot);

		executeRemoteCommandAndWait(tarCommand, true);
		String jointCommand = cdCommand + ";" + exCommand;
		// Note that this does NOT wait for the command
		// to complete.
		executeRemoteCommand(jointCommand, true);
		}
	}

	/**
	 * Execute a remote command and wait for it to terminate.
	 */
	public void executeRemoteCommandAndWait(String command, boolean noQuotes) {
		try {
			Process p = executeRemoteCommand(command, noQuotes);
			if (p!=null){
			  p.waitFor();
			}
		} catch (InterruptedException e) {
		}
	}

	/**
	 * Begin execution of a remote command.
	 * NOTE: The addition of "noQuotes" is a bit of a hack, designed to let
	 * either a windows or linux computer act as the server.  Perhaps
	 * other commands have issues with the quotes as well
	 */
	public Process executeRemoteCommand(String command, boolean noQuotes) {
		String login = username + "@" + address.getHostAddress();
		String prefix = "ssh";

		String quote = (serverIsWindows || (!noQuotes)) ? "'" : "";
		String fullCommand = prefix + " " + login + " " + quote+command+quote;
		System.out.println("Executing " + command);
		System.out.println("Full command:" + fullCommand);

		try {
			Process p = Runtime.getRuntime().exec(fullCommand);
			if (p!=null){
				looseProcesses.add(p);
			}
			
			Thread scout = new Thread(new StreamConnect(p.getInputStream(),
					System.out));
			scout.start();
			looseThreads.add(scout);

			Thread scerr = new Thread(new StreamConnect(p.getErrorStream(),
					System.err));
			scerr.start();
			looseThreads.add(scerr);

			return p;
		} catch (IOException io) {
			System.err.println("I/O Exception executing a remote command");
			io.printStackTrace(System.err);
			return null;
		}
	}

	/**
	 * Aggressively kill a Linux bot.
	 */
	public void remoteKillLinux() {
		String command = "kill -9 -1";
		executeRemoteCommandAndWait(command, true);
	}

	/**
	 * Remotely kill a Windows bot. UNTESTED
	 */
	public void remoteKillWindows() {
		//String command = "cmd.exe /C taskkill.exe /F /FI \"USERNAME eq " + username + "\"";
		String command = "cmd.exe /C taskkill.exe /F /T /IM java.exe /IM bash.exe";
		executeRemoteCommandAndWait(command, true);
	}

	/**
	 * Clean files from the remote machine. 
	 * This appears to work:
	 * 1. from Linux to Linux
	 * 2. from Windows to Linux
	 * TODO Check if the quotes
	 * are needed.
	 */
	public void cleanFiles() {
		// Removed a slash. expansionLocation SHOULD HAVE A SLASH AT THE END
		//String command = "'rm -rf " + expansionLocation + "*'";
		// Removed quotes. Works for Linux to Linux
		String command = "rm -rf " + expansionLocation + "*";
		
		executeRemoteCommandAndWait(command, true);
	}

	/**
	 * Restart the machine. UNTESTED
	 */
	public void restartMachine() {

		if (shouldRestart) {
			if (isWindows) {
				// send restart
				//executeRemoteCommandAndWait("shutdown -r now");
				executeRemoteCommandAndWait("shutdown -r -t 0", true);
			} else {
				// send linux restart
				executeRemoteCommandAndWait("shutdown -r now", true);
			}

			int timeOut = 3000; // timeout for the ping to say its down 
			int attempts = 0;
			int maxAttempts = 180; // at 1 second each, 3 minutes
			boolean status = true;

			// wait 3 minutes for machine to go down
			while( status == true && attempts < maxAttempts ) {
				try {
					status = getIP().isReachable(timeOut); // ping machine
				} catch (UnknownHostException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
				attempts++;
			}
			
			// status is now false, or the machine didn't restart after MAX_ATTEMPTS
			attempts = 0;
			
			// now wait 3 minutes for it to come back up
			while (status == false && attempts < maxAttempts) {
				try {
					status = getIP().isReachable(timeOut);
				} catch (UnknownHostException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
				attempts++;
			}
		}

	}

	/**
	 * Start a bot.
	 */
	public void start(BotInterface bot, InetAddress server, int port) {
		System.out.println("Starting machine " + address);
		extractAndPlay((BotTarFile) bot, server, port);
	}

	/**
	 * Kills all threads and processes "owned" by this object
	 * on the server machine
	 *
	 */
	public void cleanThreads(){
		for (Thread t : looseThreads) {
			t.interrupt();
		}
		looseThreads.clear();
		for(Process p:looseProcesses){
		    p.destroy();
		  }
		looseProcesses.clear();
	}
	
	public void clean() {
		// Kills all user processes.
		cleanThreads();
		System.out.println("shouldClean="+shouldClean);
		if (shouldClean) {
			// Agressively shuts down anything on the machine
			if (isWindows) {
				remoteKillWindows();
			} else {
				remoteKillLinux();
			}
			// Erases the files on the remote machine
			cleanFiles();
			// Cleans all threads created while the files were
			// cleaned.
			cleanThreads();	
		}
	}

	/**
	 * Output the IP as a string.
	 */
	public String toString() {
		return address.toString();
	}
}
