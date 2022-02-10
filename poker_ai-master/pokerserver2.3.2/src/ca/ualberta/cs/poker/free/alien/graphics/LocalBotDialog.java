package ca.ualberta.cs.poker.free.alien.graphics;

import java.awt.BorderLayout;
import java.awt.FileDialog;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Hashtable;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;

import ca.ualberta.cs.poker.free.alien.GraphicalAlienClient;
import ca.ualberta.cs.poker.free.tournament.BotTarFile;

public class LocalBotDialog extends JDialog implements ActionListener{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	GraphicalAlienClient parent;
	JTextField directory;
	Hashtable<String,BotTarFile> localBots;
	JList localBotList;
	JDialog editDialog;
	
	public LocalBotDialog(GraphicalAlienClient parent){
		super(parent,"Manage Local Bots",true);
		//setBounds(GraphicalAlienClient.STARTING_WIDTH, GraphicalAlienClient.STARTING_HEIGHT, this.getWidth(), this.getHeight());
		this.parent = parent;
		this.localBots = parent.localBots;
		directory = null;
	}
	
	public boolean isNewNameLegal(String newName, BotTarFile bot){
		if (bot.getName().equals(newName)){
			return true;
		}
		
		if (newName.contains(".")||newName.contains(" ")){
			JOptionPane.showMessageDialog(this.parent,"Name cannot contain periods or spaces.","Error",JOptionPane.ERROR_MESSAGE);
			return false;
		}

		if (parent.fixedNameBots.contains(bot.getName())){
			JOptionPane.showMessageDialog(this.parent,"Name has been submitted to the server and cannot be changed.","Error",JOptionPane.ERROR_MESSAGE);
			return false;
		}
		if (localBots.containsKey(newName)){
			JOptionPane.showMessageDialog(this.parent,"Another bot with the same name already exists.","Error",JOptionPane.ERROR_MESSAGE);
			return false;			
		}
		return true;
	}
	
	public String getUntitledName(){
		if (!localBots.contains("untitled")){
			return "untitled";
		} 
		int i=1;
		String candidate = null;
		do{
			i++;
			candidate = "untitled-"+i;
		}
		while(localBots.contains(candidate));
		return candidate;
	}
	public void init(){
		Vector<String> localBotNames = new Vector<String>(localBots.keySet());
		localBotList = new JList(localBotNames);
		setLayout(new BorderLayout());
		//JPanel center = new JPanel(new FlowLayout());
		//Vector<String> sentBotNames = new Vector<String>(parent.localBotsSent.keySet());
		//JList localBotsSentList = new JList(sentBotNames);
		//center.add(new JScrollPane(localBotList));
		//center.add(new JScrollPane(localBotsSentList));
		add(new JScrollPane(localBotList),BorderLayout.CENTER);
		JPanel buttons = new JPanel(new FlowLayout());
		JButton editBot = new JButton("Edit");
		editBot.setActionCommand("EDITBOTLBD");
		editBot.addActionListener(this);
		buttons.add(editBot);
		JButton newBot = new JButton("New");
		newBot.setActionCommand("NEWBOTLBD");
		newBot.addActionListener(this);
		buttons.add(newBot);
		JButton copyBot = new JButton("Copy");
		copyBot.setActionCommand("COPYBOTLBD");
		copyBot.addActionListener(this);
		buttons.add(copyBot);
		add(buttons,BorderLayout.SOUTH);
		JButton okButton = new JButton("OK");
		okButton.setActionCommand("OKLBD");
		okButton.addActionListener(this);
		buttons.add(okButton);
		add(buttons,BorderLayout.SOUTH);
		
		pack();
		validate();
		setLocationRelativeTo(parent);
		setVisible(true);
		
		
	}
	
	public String getBotLocation(){
		FileDialog fd = new FileDialog(this,"Bot Location",FileDialog.LOAD);
		fd.setVisible(true);
		return fd.getDirectory();
		
	}
	
	public boolean newBot(){
		BotTarFile bot = new BotTarFile(getUntitledName(), "","",false, false, true, true);
		boolean result = editBot("New Bot",bot);
		if (result){
			localBots.put(bot.getName(),bot);
			updateLocalBots();
		}
		return result;
	}
	
	public void updateLocalBots(){
		localBotList.setListData(new Vector<String>(localBots.keySet()));
		validate();
		repaint();
	}
	public boolean editExistingBot(BotTarFile bot){
		String oldName = bot.getName();
		boolean result = editBot("Edit Bot",bot);
		if (result){
			localBots.remove(oldName);
			localBots.put(bot.getName(),bot);
			updateLocalBots();
		}
		return result;
	}
	
	public boolean editBot(String title,BotTarFile bot){
		EditBotDialog ebd = new EditBotDialog(this,title,bot);
		return ebd.runDialog();
/*		dialogFinished = null;
		editDialog = new JDialog(this,title,true);
		editDialog.setLayout(new GridLayout(3,3));
		editDialog.add(new JLabel("Name:"));
		JTextField name = new JTextField(20);
		name.setText(bot.getName());
		editDialog.add(name);
		editDialog.add(new JLabel());
		editDialog.add(new JLabel("Directory:"));
		directory = new JTextField(20);
		directory.setText(bot.getLocation());
		editDialog.add(directory);
		// Placeholder for browse button
		JButton browseButton = new JButton("Browse");
		browseButton.setActionCommand("BROWSEEDITBOT");
		browseButton.addActionListener(this);
		editDialog.add(browseButton);
		JButton okButton = new JButton("OK");
		okButton.setActionCommand("OKEDITBOT");
		okButton.addActionListener(this);
		editDialog.add(okButton);
		JButton cancelButton = new JButton("Cancel");
		cancelButton.setActionCommand("CANCELEDITBOT");
		cancelButton.addActionListener(this);
		editDialog.add(cancelButton);
		editDialog.validate();
		editDialog.pack();
		editDialog.setVisible(true);
		if (dialogFinished.equals("OKEDITBOT")){
			String newDirectory = directory.getText();
			String newName = name.getText();
			bot.setName(newName);
			bot.setLocation(newDirectory);
			if (newName.equals("")){
				editDialog.setVisible(false);
				return false;
			}
			if (!(new File(newDirectory).exists())){
				editDialog.setVisible(false);
				return false;
			}
		}
		
		editDialog.setVisible(false);
		//dialog.dispose();
		return dialogFinished.equals("OKEDITBOT");*/
	}
	
	public void actionPerformed(ActionEvent e) {
		String command = e.getActionCommand();
		if (command.equals("OKLBD")){
			setVisible(false);
		} else if (command.equals("EDITBOTLBD")){
			String val = (String)localBotList.getSelectedValue();
			if (val!=null){
				BotTarFile bot = localBots.get(val);
				if (bot!=null){
					editExistingBot(bot);
				}
			}
		} else if (command.equals("NEWBOTLBD")){
			newBot();
		} else if (command.equals("COPYBOTLBD")){
			
		}
	}
	
}
