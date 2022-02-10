package ca.ualberta.cs.poker.free.tournament;

import java.io.*;
import java.net.*;
import java.util.Vector;



/*
 * This class contains the code to start a process on a remote machine
 * and clean up afterwards.
 * 
 * @author Martin Zinkevich
 * 
 */
public class LocalMachine implements MachineInterface {

	/**
	 * Threads that need to be cleaned out
	 * at the end.
	 */
	Vector<Thread> looseThreads;

	/**
	 * Processes that need to be cleaned out
	 * at the end.
	 */
	Vector<Process> looseProcesses;

	/** Address of the server itself */
	private InetAddress address;

	/** Location to expand bot into */
	private String expansionLocation;

	/**
	 * Is the local machine a window machine?
	 */
	public boolean isWindows;

	/**
	 * Should the clean be more aggressive?
	 */
	private boolean shouldClean;

	private boolean allowMoreThreads;
	
	/**
	 * Create a new LocalMachine with the address (must
	 * be the server IP or 127.0.0.1), expansionLocation
	 * and if this is Windows.
	 */
	public LocalMachine(InetAddress address, String expansionLocation,
			boolean isWindows) {
		this.address = address;
		this.expansionLocation = expansionLocation;
		if (expansionLocation.equals("")) {
			throw new RuntimeException("Empty expansionLocation");
		}
		this.isWindows = isWindows;
		this.shouldClean = false;
		this.looseThreads = new Vector<Thread>();
		this.looseProcesses = new Vector<Process>();
		allowMoreThreads = true;
	}

	/**
	 * Get the IP of this machine.
	 */
	public InetAddress getIP() {
		return address;
	}

	/**
	 * Conservatively tests if a bot connecting from addr could be this
	 * machine. Runs into all sorts of issues in the case of local machines,
	 * such as whether the machine is connecting via a local connection
	 * or over the internet.
	 */
	public boolean isThisMachine(InetAddress addr) {
		try {
			if (addr.equals(address)) {
				return true;
			}
			if (addr.equals(InetAddress.getLocalHost())) {
				return true;
			}
			if (addr.equals(InetAddress.getByName("127.0.0.1"))) {
				return true;
			}

		} catch (UnknownHostException unk) {
			unk.printStackTrace();
			System.err.println(unk);
		}
		return false;
	}

	public String toWindows(String filename) {
		return filename.replace('/', '\\');
	}

	public String toLinux(String filename) {
		return filename.replace('\\', '/');
	}

	/**
	 * This extracts the bot and connects to the server. However, it does
	 * this in a separate thread.
	 */
	public void extractAndPlay(BotTarFile bot, InetAddress server, int port) {
		String tarFile = bot.getLocation();
		String windowsPrefix = "cmd.exe /C ";
		String executable = "cmd.exe /C startme.bat";
		String copyCommand = "cmd.exe /C copy /Y " + tarFile + " "
				+ expansionLocation + "temp.jar";
		String tarCommand = windowsPrefix + "tar -xf " + tarFile + " -C "
				+ expansionLocation;

		// Windows
		if (isWindows) {
			tarFile = toWindows(tarFile);
			expansionLocation = toWindows(expansionLocation);
		} else {
			// Linux
			tarFile = toLinux(tarFile);
			expansionLocation = toLinux(expansionLocation);
			tarCommand = "tar -xf " + tarFile + " -C " + expansionLocation;
			copyCommand = "cp " + tarFile + " " + expansionLocation
					+ "temp.jar";
			executable = "startme.sh";
		}

		// Copy and expand if bot is a tarball
		if (tarFile.endsWith(".tar")) {
			executeCommandAndWait(tarCommand);
		}

		// Copy & expand bot if bot is a jarball
		if (tarFile.endsWith(".jar")) {
			// Y prefix is required to suppress prompting
			executeCommandAndWait(copyCommand);
			if (!isWindows) {
				executeCommandAndWait("chmod -R u+rwx .", expansionLocation,
						expansionLocation);
			}
			String jarCommand = "jar -xf temp.jar";
			executeCommandAndWait(jarCommand, expansionLocation,
					expansionLocation);
		}

		String internalLocation = expansionLocation + bot.getInternalLocation();
		if (tarFile.endsWith("/") || tarFile.endsWith("\\")) {
			internalLocation = tarFile;
		}

		// Start bot
		String serverIP = server.getHostAddress();
		String exCommand = executable + " " + serverIP + " " + port;

		// convert to linux format if necessary
		if (!isWindows) {
			internalLocation = toLinux(internalLocation);
			// This used to say executeCommand
			// Moreover, there was a superfluous forward slash.
			exCommand = internalLocation + exCommand;
			// the exec() command refused to run ./startme.sh, but adding an absolute path makes
			// it work, so be it
		}
		// This appears redundant for Linux.
		executeCommand(exCommand, internalLocation, expansionLocation);
	}

	/**
	 * Execute a command on the local machine and wait for it to complete.
	 */
	public void executeCommandAndWait(String command) {
		try {
			Process p = executeCommand(command);
			StreamConnect sc = new StreamConnect(p.getInputStream(), System.out);
			Thread tsc = new Thread(sc);
			tsc.start();
			
			//looseThreads.add(tsc);
			addThread(tsc);
			
			StreamConnect scerr = new StreamConnect(p.getErrorStream(),
					System.err);
			Thread tscerr = new Thread(scerr);
			tscerr.start();
			//looseThreads.add(tscerr);
			addThread(tsc);
			p.waitFor();
		} catch (InterruptedException e) {
		}
	}

	/**
	 * Execute a command on the local machine and wait for it to complete.
	 */
	public void executeCommandAndWait(String command, String directory,
			String outputDir) {
		try {
			Process p = executeCommand(command, directory, outputDir);
			StreamConnect sc = new StreamConnect(p.getInputStream(), System.out);
			Thread tsc = new Thread(sc);
			tsc.start();
			//looseThreads.add(tsc);
			addThread(tsc);
			StreamConnect scerr = new StreamConnect(p.getErrorStream(),
					System.err);
			Thread tscerr = new Thread(scerr);
			tscerr.start();
			//looseThreads.add(tscerr);
			addThread(tsc);
			p.waitFor();
		} catch (InterruptedException e) {
		}
	}

	/**
	 * Process a local command which may have a lot of output.
	 */
	public Process executeCommand(String command, String directory,
			String outputDir) {
		System.out.println("Executing locally:" + command);
		System.out.println("In directory:" + directory);
		System.out.println("With output:" + outputDir);
		try {
			FileOutputStream normalOut = new FileOutputStream(outputDir
					+ "out.txt");
			FileOutputStream errOut = new FileOutputStream(outputDir
					+ "err.txt");

			Process p = Runtime.getRuntime().exec(command, null,
					new File(directory));
			looseProcesses.add(p);
			StreamConnect sc = new StreamConnect(p.getInputStream(), normalOut);
			Thread tsc = new Thread(sc);
			tsc.start();
			//looseThreads.add(tsc);
			addThread(tsc);
			
			StreamConnect scerr = new StreamConnect(p.getErrorStream(), errOut);
			Thread tscerr = new Thread(scerr);
			tscerr.start();
			//looseThreads.add(tscerr);
			addThread(tsc);
			return p;
		} catch (IOException io) {
			System.err.println("I/O Exception executing a local command: "
					+ command + " in " + directory);
			io.printStackTrace();
			return null;
		}
	}

	/**
	 * Execute a command locally and return the process.
	 */
	public Process executeCommand(String command) {
		System.out.println("Executing locally:" + command);

		try {
			Process p = Runtime.getRuntime().exec(command);
			looseProcesses.add(p);
			return p;
		} catch (IOException io) {
			System.err.println("I/O Exception executing a local command: "
					+ command);
			io.printStackTrace();
			return null;
		}
	}

	/**
	 * UNTESTED
	 */
	public void cleanFiles() {
		String command = "'rm -rf " + expansionLocation + "/*'";
		executeCommandAndWait(command);
	}

	/**
	 * UNTESTED
	 */
	public void restartMachine() {
		throw new RuntimeException("Not implemented");
	}

	/**
	 * Start a new bot on this machine.
	 */
	public void start(BotInterface bot, InetAddress server, int port) {
		System.out.println("Starting machine " + address);
		extractAndPlay((BotTarFile) bot, server, port);
	}

	/**
	 * If clean() is called and a Thread is being added,
	 * interupt the thread. This will only occur if clean()
	 * has been called and a thread is added in the mean time
	 * 
	 * @param thread
	 */
	public synchronized void  addThread( Thread thread ) {
		if( allowMoreThreads ) {
			looseThreads.add(thread);
		} else
			thread.interrupt();
	}
	
	/**
	 * Interupt all the threads in a synchronized method
	 * to avoid a concurrent modification
	 *
	 */
	public synchronized void  removeThreads() {
		for (Thread t : looseThreads) {
			t.interrupt();
		}
		looseThreads.clear();
	}
	
	//public synchronized void removeThread( Thread )
	
	/**
	 * We needed to synchronize the vector operations to avoid
	 * concurrentmodification exceptions when a machine is trying
	 * to clean() while threads were being added
	 */
	public void clean() {
		
		allowMoreThreads = false;
		
		removeThreads();
		
		for (Process p : looseProcesses) {
			p.destroy();
		}
		if (shouldClean) {
			cleanFiles();
		}
		
		allowMoreThreads = true;
	}

	/**
	 * Prints the IP.
	 */
	public String toString() {
		return address.getHostAddress() + " " + this.expansionLocation + " "
				+ (isWindows ? "WINDOWS" : "LINUX");
	}
}
