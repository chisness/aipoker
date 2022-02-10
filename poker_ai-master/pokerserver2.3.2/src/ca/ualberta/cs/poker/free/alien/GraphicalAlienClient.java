package ca.ualberta.cs.poker.free.alien;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.io.IOException;
import java.net.InetAddress;
import java.net.SocketException;
import java.util.Hashtable;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.UIManager;
import javax.swing.UnsupportedLookAndFeelException;

import ca.ualberta.cs.poker.free.alien.graphics.CreateMatchDialog;
import ca.ualberta.cs.poker.free.alien.graphics.LocalBotDialog;
import ca.ualberta.cs.poker.free.alien.graphics.LoginDialog;
import ca.ualberta.cs.poker.free.server.TimeoutException;
import ca.ualberta.cs.poker.free.tournament.BotInterface;
import ca.ualberta.cs.poker.free.tournament.BotTarFile;


public class GraphicalAlienClient extends JFrame implements ActionListener,WindowListener, AlienClientListener{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	JMenu currentMenu;
	JMenuBar bar;
	public Hashtable<String,BotTarFile> localBots;
	public Vector<String> opponents;
	
	//public Hashtable<String,BotTarFile> localBotsSent;
	public AlienClient client;
	public AlienProfile profile;
	JList queuedMatches;
	Vector<String> queuedMatchStrings;
	JList completedMatches;
	JOptionPane about;
	JDialog aboutDialog;
	public InetAddress machineName;
	public int port;
	static String defaultProfile = "graphicalalienclient.prf";
	String dialogFinished=null;
	private Vector<String> completedMatchStrings;
	private HelpViewer helpv;
	public final static int STARTING_WIDTH = 200;
	public final static int STARTING_HEIGHT = 200;
	
	public Vector<String> fixedNameBots;
	private boolean loggedIn;
	Thread clientThread;
	public GraphicalAlienClient(){
		this(defaultProfile);
	}
	
	public GraphicalAlienClient(String profileLocation){
		super("Benchmark Server Access");
		client = new AlienClient();
		client.listener = this;
		localBots = new Hashtable<String,BotTarFile>();
		fixedNameBots = new Vector<String>();
		loggedIn = false;
		init();
		try{
		profile = new AlienProfile(profileLocation);
		loadProfile();
		} catch (IOException io){
			System.err.println("I/O error with profile "+profileLocation+":");
			System.err.println(io.getMessage());
		} catch (TimeoutException to){
			System.err.println("Problem connecting to server");
		}
		
		
	}
	
	public void loadProfile() throws TimeoutException{
		for(BotInterface bot:profile.bots){
			if (bot instanceof BotTarFile){
				localBots.put(bot.getName(),(BotTarFile)bot);
			}
		}
		for(ClientMatch match:profile.matches){
			addQueuedMatch(match);
		}
	}

	public JPanel getRunningPanel(){
		JPanel runningPanel = new JPanel(new BorderLayout());
		runningPanel.add(new JLabel("Running Matches"),BorderLayout.NORTH);
		queuedMatches = new JList();
		queuedMatchStrings = new Vector<String>();
		runningPanel.add(new JScrollPane(queuedMatches),BorderLayout.CENTER);
		JButton killButton = new JButton("Terminate");
		killButton.setActionCommand("TERMINATEMATCHES");
		killButton.addActionListener(this);
		killButton.setToolTipText("Terminate the match selected above");
		runningPanel.add(killButton,BorderLayout.SOUTH);
		queuedMatches.setToolTipText("Queued matches waiting to be run are listed here");
		return runningPanel;
	}

	public void manageLocalBots(){
		LocalBotDialog lbd = new LocalBotDialog(this);
		lbd.init();
		
	}

	public Vector<ClientMatch> getMatches(){
		while(localBots.isEmpty()){
			int result = JOptionPane.showConfirmDialog(this,"There are no local bots. Would you like to create some now?");
			if (result==JOptionPane.YES_OPTION){
				manageLocalBots();
			} else {
				showError("Cannot create matches without creating local bots.");
				return null;
			}
		}
		while(!loggedIn){
			int result = JOptionPane.showConfirmDialog(this,"You are not logged in. Would you like to login now?");
			if (result==JOptionPane.YES_OPTION){
				login();
			} else {
				showError("Cannot create matches without being logged in");
				return null;
			}
		}

		CreateMatchDialog cmd=new CreateMatchDialog(this);
		return cmd.getMatches();
	}
	
	public JPanel getCompletedPanel(){
		JPanel completedPanel = new JPanel(new BorderLayout());
		setLayout(new GridLayout(2,1));
		completedPanel.add(new JLabel("Completed Matches"),BorderLayout.NORTH);
		completedMatchStrings = new Vector<String>();
		completedMatches = new JList();
		completedPanel.add(new JScrollPane(completedMatches),BorderLayout.CENTER);
		completedMatches.setToolTipText("Completed matches will be listed here");
		
		return completedPanel;
	}
	
	public void newMenu(String name){
		currentMenu = new JMenu(name);
		bar.add(currentMenu);
	}
	
	public void newMenuItem(String name, String action, String tooltip){
		JMenuItem item = new JMenuItem(name);
		item.setActionCommand(action);
		item.setToolTipText(tooltip);
		item.addActionListener(this);
		currentMenu.add(item);
	}
	public void showError(String message){
		JOptionPane.showMessageDialog(this,message,"Error",JOptionPane.ERROR_MESSAGE);
	}
	public boolean validName(String name, String nameType){
		if (name==null){
			showError(nameType+" name is null");
			return false;
		} else if (name.equals("")){
			showError(nameType+" name is empty");
			return false;
		} else if (name.contains(".")||name.contains(" ")){
			showError(nameType+" name cannot contain periods or spaces");
			return false;
		}
		return true;
	}
	



	public void init(){
		JPanel runningPanel = getRunningPanel();
		JPanel completedPanel = getCompletedPanel();
		// xian - set position for the main jframe
		// maz - setLocationRelativeTo centers it below
		//setBounds(STARTING_WIDTH, STARTING_HEIGHT, this.getWidth(), this.getHeight());
		setLayout(new FlowLayout());
		add(runningPanel);
		add(completedPanel);
		addWindowListener(this);
		bar = new JMenuBar();
		setJMenuBar(bar);
		newMenu("Match");
		newMenuItem("Create","CREATEMATCHES","Create matches and send them to the server");
		newMenuItem("Terminate","TERMINATEMATCHES","Terminate selected matches running on the server");
		newMenuItem("Analyze","ANALYZEMATCHES","Analyze matches that have completed (not implemented)");
		currentMenu.addSeparator();
		newMenuItem("Quit", "QUITGUI","Exit the program (kills all matches!)");
		newMenu("Local Bots");
		newMenuItem("Manage","MANAGELOCALBOTS","Edit old bots and create new ones");
		newMenu("Login");
		newMenuItem("Login","LOGIN","Login to the server");
		newMenuItem("Logout","LOGOUT","Logout of the server");
		newMenu("Help");
		newMenuItem("About", "ABOUT", "Information about the software");
		newMenuItem("Help", "HELP","Help about the software");
		pack();
		validate();
		// From http://www.rgagnon.com/javadetails/java-0223.html
		// Also from http://java.sun.com/docs/books/tutorial/uiswing/components/frame.html
		//TODO is this copyrighted in a way such that I can't use it?
		setLocationRelativeTo(null);
		setVisible(true);
	}
	
	public boolean login(){
		if (!getLoginInfo()){
			return false;
		}
		try{
		client.connect(machineName,port);
		loggedIn = client.login();
		if (!loggedIn){
			client.close();
			showError("Can't Login!");
		} else {
			// Do something about confirmation here
			if (profile!=null && profile.machines!=null){
				client.addMachines(profile.machines);
			}
			if (clientThread!=null){
				clientThread.interrupt();
			}
			clientThread = new Thread(client);
			clientThread.start();
			JOptionPane.showMessageDialog(this,"Login successful!","Login",JOptionPane.INFORMATION_MESSAGE);
			//box.showMessage();
		}
		return loggedIn;
		} catch (SocketException so){
			so.printStackTrace(System.err);
			showError("Can't Login!"+so.getMessage());
			return false;
		} catch (IOException e) {
			e.printStackTrace(System.err);
			showError("Can't Login! "+e.getMessage());
			return false;
		} catch (TimeoutException toe){
			toe.printStackTrace(System.err);
			showError("Can't Login! Timeout error!");
			return false;
		} catch (InterruptedException ie){
			System.err.println("Interruption during login: exiting now");
			System.exit(-1);
			// That's weird: technically, this is unreachable.
			return false;
		}
	}
	/**
	 * Called when the menu item about is selected
	 *
	 */
	public void about() {
		
		String aboutText = "This graphical client was written for " +
							"use with the University of Alberta free poker " +
							"server by Christian Smith and Martin Zinkevich. " +
							"\n\n Details can be found at http://www.cs.ualberta.ca/~pokert";
		
		about = new JOptionPane(aboutText, JOptionPane.YES_NO_CANCEL_OPTION );
		aboutDialog = about.createDialog(getRunningPanel(), "About");	
		aboutDialog.setVisible(true);
	}
	
	/**
	 * Called when the menu item help is selected
	 *
	 */
	public void help() {
		helpv = new HelpViewer();
		helpv.setLocationRelativeTo(this);
		//helpv.setBounds(STARTING_WIDTH, STARTING_HEIGHT, helpv.getWidth(), helpv.getHeight());
	}
	public boolean getLoginInfo(){
		LoginDialog ld = new LoginDialog(this);
		return ld.getLoginInfo();
		
	}
	
	public static void main(String[] args){
		// From "Java Tutorial"
		// http://java.sun.com/docs/books/tutorial/information/license.html
		try{
			// Ugly
			//UIManager.setLookAndFeel("com.sun.java.swing.plaf.motif.MotifLookAndFeel");
			//Not found
			//UIManager.setLookAndFeel("com.sun.java.swing.plaf.gtk.GTKLookAndFeel" );
            //Puts in top-left corner
			UIManager.setLookAndFeel("com.sun.java.swing.plaf.windows.WindowsLookAndFeel");
			//UIManager.setLookAndFeel("javax.swing.plaf.metal.MetalLookAndFeel");
		} catch (UnsupportedLookAndFeelException e) {
			
			// this is needed for linux, otherwise it gets unsupported exception
			try {
				UIManager.setLookAndFeel("com.sun.java.swing.plaf.gtk.GTKLookAndFeel" );
			} catch (Exception e1) {
				e1.printStackTrace();
			} 
			
		}
		catch (Exception e){
			
			e.printStackTrace(System.err);
		}

		if (args.length==0){
			new GraphicalAlienClient();
		} else {
			new GraphicalAlienClient(args[0]);
		}
		//frame.init();
		/*for(int i=0;i<10;i++){
			frame.addQueuedMatch("Match"+i);
			Thread.sleep(50);
		}*/
		//BotTarFile bot = frame.getBotInterface();
		//frame.editBot(bot);
		//frame.manageLocalBots();
		
		//System.exit(0);
	}

	/**
	 * This logs out of the system and exits the program.
	 * TODO: Handle 
	 */
	public void logout(){
		if (!queuedMatchStrings.isEmpty()){
			int result = JOptionPane.showConfirmDialog(this,"Warning! Logging out will cause all unfinished matches to terminate and the program to exit! Are you sure you want to do this?","DANGER!",JOptionPane.YES_NO_CANCEL_OPTION);
			if (result!=JOptionPane.YES_OPTION){
				showError("Logout aborted.");
			}
		}
		if (loggedIn){
		  loggedIn = false;
/*		  try{
			  clientThread.interrupt();
			  while(clientThread.isAlive()){

				  Thread.currentThread().sleep(1000);
			  }
		  client.close();
		  } catch (IOException io){
			  System.err.println("NOTE:Currently logging out causes a timeout exception.");
			  showError(io.getMessage());
		  } catch (InterruptedException ie){
			System.err.println("Hard kill");
		  } catch (TimeoutException to){
			  System.err.println("NOTE:Currently logging out causes a timeout exception.");
		  }
		  */
		System.exit(0);
		}
	}
	public void addQueuedMatch(ClientMatch match) throws TimeoutException{
		if (!localBots.containsKey(match.bot)){
			showError("Bot "+match.bot+ " unknown in Match:\n"+match+".");
			return;
		}
		while(!loggedIn){
			int result = JOptionPane.showConfirmDialog(this,"You are not logged in. Would you like to login now?");
			if (result==JOptionPane.YES_OPTION){
				login();
			} else {
				showError("Cannot queue matches without being logged in");
				return;
			}
		}
		if (!fixedNameBots.contains(match.bot)){
			fixedNameBots.add(match.bot);
			client.addBot(localBots.get(match.bot));
		}
		client.addMatch(match);
		Vector<String> miniMatches = match.getMatchNames();
		for(String miniMatch:miniMatches){
			addQueuedMatch(miniMatch);
		}
	}
	
	public void addQueuedMatch(String match){
		queuedMatchStrings.add(match);
		queuedMatches.setListData(queuedMatchStrings);
		pack();
		validate();
		repaint();
	}
	
	public void actionPerformed(ActionEvent arg0) {
		String command = arg0.getActionCommand();
		if (command.equals("TERMINATEMATCHES")){
			Object[] toDelete=queuedMatches.getSelectedValues();
			if (toDelete.length==0){
				showError("No matches to terminate!!!");
			} else {
			try{
			
			String finalMessage = "Matches:\n";
			for(Object obj:toDelete){
				String str = (String)obj;
				finalMessage+=str+"\n";
				client.sendMatchTerminate(str);
			}
			showError(finalMessage+"appear to have been terminated correctly. However, they will not be removed locally from the queue in case this is not true.");
			} catch (TimeoutException to){
				showError("Error sending match terminate message");
				System.err.println(to);
			}
			}
		
		} else if (command.equals("CREATEMATCHES")){
			Vector<ClientMatch> matches = getMatches();
			if (matches==null){
				System.err.println("No matches returned");
			} else {
				try{
				for(ClientMatch match:matches){
					System.err.println(match);
					addQueuedMatch(match);
				}
				} catch (TimeoutException to){
					System.err.println("Error connecting to server");
				}
			}
			
		} else if (command.equals("ANALYZEMATCHES")){
			
		} else if (command.equals("MANAGELOCALBOTS")){
			manageLocalBots();
		} else if (command.equals("LOGIN")){
			login();
		} else if (command.equals("LOGOUT")){
			logout();
		} else if (command.equals("ABOUT")){
			about();
		} else if (command.equals("HELP")){
			help();
		} else if (command.equals("QUITGUI")){
			System.exit(0);
		}
		
	}

	public void windowOpened(WindowEvent e) {		
	}

	public void windowClosing(WindowEvent e) {
		System.exit(0);
	}

	public void windowClosed(WindowEvent e) {
		System.exit(0);
	}

	public void windowIconified(WindowEvent e) {
		
	}

	public void windowDeiconified(WindowEvent e) {
		
	}

	public void windowActivated(WindowEvent e) {		
	}

	public void windowDeactivated(WindowEvent e) {		
	}

	public static String getLocalMatchName(String matchName){
		int firstPeriod = matchName.indexOf('.');
		int secondPeriod = matchName.indexOf('.',firstPeriod+1);
		return matchName.substring(secondPeriod+1);
		
	}
	public void handleMatchCompleted(String matchName) {
		matchName = getLocalMatchName(matchName);
		queuedMatchStrings.remove(matchName);
		queuedMatches.setListData(queuedMatchStrings);
		completedMatchStrings.add(matchName);
		completedMatches.setListData(completedMatchStrings);		
		repaint();
	}

	public void handleMatchTerminated(String matchName) {
		// TODO Auto-generated method stub
		System.err.println("GraphicalAlienClient.handleMatchTerminated");
		matchName = getLocalMatchName(matchName);
		queuedMatchStrings.remove(matchName);
		queuedMatches.setListData(queuedMatchStrings);
		repaint();
	}
	
}
