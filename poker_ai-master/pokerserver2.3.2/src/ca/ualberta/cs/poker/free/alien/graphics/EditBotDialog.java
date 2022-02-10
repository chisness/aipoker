package ca.ualberta.cs.poker.free.alien.graphics;

import java.awt.FileDialog;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JTextField;

import ca.ualberta.cs.poker.free.tournament.BotTarFile;

public class EditBotDialog extends JDialog implements ActionListener{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	BotTarFile bot;
	private String dialogFinished;
	JTextField directory;
	JTextField name;
	String newName;
	String newDirectory;
	LocalBotDialog parent;
	
	public EditBotDialog(LocalBotDialog lbd, String title,BotTarFile bot){
		super(lbd,title,true);
		this.parent = lbd;
		this.bot = bot;
	}
	
	
	public boolean runDialog(){
		dialogFinished = null;
		setLayout(new GridLayout(3,3));
		add(new JLabel("Name:"));
		name = new JTextField(20);
		name.setText(bot.getName());
		add(name);
		add(new JLabel());
		add(new JLabel("Directory:"));
		directory = new JTextField(20);
		directory.setText(bot.getLocation());
		add(directory);
		// Placeholder for browse button
		JButton browseButton = new JButton("Browse");
		browseButton.setActionCommand("BROWSEEDITBOT");
		browseButton.addActionListener(this);
		add(browseButton);
		JButton okButton = new JButton("OK");
		okButton.setActionCommand("OKEDITBOT");
		okButton.addActionListener(this);
		add(okButton);
		JButton cancelButton = new JButton("Cancel");
		cancelButton.setActionCommand("CANCELEDITBOT");
		cancelButton.addActionListener(this);
		add(cancelButton);
		validate();
		pack();
		setLocationRelativeTo(parent);
		setVisible(true);
		if (dialogFinished.equals("OKEDITBOT")){
			//newDirectory = directory.getText();
			bot.setName(newName);
			bot.setLocation(newDirectory);
			
		}
		// maz-Why was this here?
		//setBounds(GraphicalAlienClient.STARTING_WIDTH, GraphicalAlienClient.STARTING_HEIGHT, this.getWidth(), this.getHeight());
		setVisible(false);
		//dialog.dispose();
		return dialogFinished.equals("OKEDITBOT");
	}

	public String getBotLocation(){
		FileDialog fd = new FileDialog(this,"Bot Location",FileDialog.LOAD);
		fd.setVisible(true);
		return fd.getDirectory();
		
	}

	public void actionPerformed(ActionEvent e) {
		String command = e.getActionCommand();
		if (command.equals("OKEDITBOT")){
			newName = name.getText();
			newDirectory = directory.getText();
			if (!parent.isNewNameLegal(newName,bot)){
				return;
			}
			if (!(new File(newDirectory).exists())){
				return;
			}

			dialogFinished = command;
			setVisible(false);
		} else if (command.equals("CANCELEDITBOT")){
			dialogFinished = command;
			setVisible(false);
		} else if (command.equals("BROWSEEDITBOT")){
			String dir = getBotLocation();
			if (dir!=null){
				directory.setText(dir);
			}
		}
	}
}
