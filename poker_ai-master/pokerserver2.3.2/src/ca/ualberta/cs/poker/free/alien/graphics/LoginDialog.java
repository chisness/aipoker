package ca.ualberta.cs.poker.free.alien.graphics;

import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.net.InetAddress;
import java.net.UnknownHostException;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPasswordField;
import javax.swing.JTextField;

import ca.ualberta.cs.poker.free.alien.GraphicalAlienClient;

public class LoginDialog extends JDialog implements ActionListener {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	GraphicalAlienClient parent;
	String dialogFinished;
	private JTextField machineNameField;
	private JTextField portField;
	private JTextField usernameField;
	private JPasswordField passwordField;
	
	public LoginDialog(GraphicalAlienClient gac){
		super(gac,"Login",true);
		parent = gac;
		addComponents();
	}
	
	public void addComponents(){
		//setBounds(GraphicalAlienClient.STARTING_WIDTH, GraphicalAlienClient.STARTING_HEIGHT, this.getWidth(), this.getHeight());
		setLayout(new GridLayout(5,2));
		add(new JLabel("Machine:"));
		machineNameField = new JTextField(20);
		
		if (parent.profile!=null){
			if (parent.profile.addr!=null){
				machineNameField.setText(parent.profile.addr);
			} else {
				machineNameField.setText("willingdon.cs.ualberta.ca");				
			}
		} else {
			machineNameField.setText("willingdon.cs.ualberta.ca");
		}
		add(machineNameField);
		add(new JLabel("Port:"));
		portField = new JTextField(20);
		portField.setText(""+parent.profile.port);
		//portField.setText("5000");
		add(portField);
		add(new JLabel("Username:"));
		usernameField = new JTextField(20);
		if (parent.profile!=null&&parent.profile.username!=null){
			usernameField.setText(parent.profile.username);
		}
		add(usernameField);
		add(new JLabel("Password:"));
		passwordField = new JPasswordField(20);
		if (parent.profile!=null&&parent.profile.password!=null){
			passwordField.setText(parent.profile.password);
		}
		
		add(passwordField);
		
		JButton okButton = new JButton("OK");
		okButton.setActionCommand("OKLOGIN");
		okButton.addActionListener(this);
		add(okButton);
		JButton cancelButton = new JButton("Cancel");
		cancelButton.setActionCommand("CANCELLOGIN");
		cancelButton.addActionListener(this);
		add(cancelButton);
		validate();
		pack();
		setLocationRelativeTo(parent);
	}
	public void actionPerformed(ActionEvent e) {
		String command = e.getActionCommand();
		if (command.equals("OKLOGIN")){
			dialogFinished = command;
			setVisible(false);
		} else if (command.equals("CANCELLOGIN")){
			dialogFinished = command;
			setVisible(false);
		}

	}

	public boolean getLoginInfo(){
		dialogFinished = null;
		setVisible(true);
		if (dialogFinished.equals("OKLOGIN")){
			parent.client.username = usernameField.getText();
			parent.client.password = new String(passwordField.getPassword());
			try{
				parent.machineName = InetAddress.getByName(machineNameField.getText());
			
				parent.port = Integer.parseInt(portField.getText());
			} catch (NumberFormatException nfe){
				parent.port = 0;
			} catch (UnknownHostException e) {
				parent.port = 0;
			}
			//System.err.println("username:"+client.username);
			//System.err.println("password:"+client.password);
			
		}
		//loginDialog.dispose();
		if (dialogFinished.equals("OKLOGIN")&&parent.port==0){
			return false;
			//return getLoginInfo();
		}
		return dialogFinished.equals("OKLOGIN");
	}

}
