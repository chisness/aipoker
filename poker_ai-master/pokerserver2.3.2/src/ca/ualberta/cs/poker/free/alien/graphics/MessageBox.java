package ca.ualberta.cs.poker.free.alien.graphics;

import java.awt.BorderLayout;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;

public class MessageBox extends JDialog implements ActionListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public static void showError(JDialog owner,String error){
		MessageBox mess = new MessageBox(owner,"Error",error);
		mess.showMessage();
	}

	public static void showError(Frame owner,String error){
		MessageBox mess = new MessageBox(owner,"Error",error);
		mess.showMessage();
	}

	public static void showMessage(Frame owner, String title, String message){
		MessageBox mess = new MessageBox(owner,title,message);
		mess.showMessage();
	}

	public static void showMessage(JDialog owner, String title, String message){
		MessageBox mess = new MessageBox(owner,title,message);
		mess.showMessage();
	}

	public void actionPerformed(ActionEvent e) {
		setVisible(false);
	}
	public MessageBox(Frame owner,String title,String message){
		super(owner,title,true);
		layoutMessage(message);
	}
	
	public MessageBox(JDialog owner,String title,String message){
		super(owner,title,true);
		layoutMessage(message);
	}
	
	public void layoutMessage(String message){
		setLayout(new BorderLayout());
		add(new JLabel(message),BorderLayout.CENTER);
		JButton okButton = new JButton("OK");
		okButton.addActionListener(this);
		add(okButton,BorderLayout.SOUTH);
	}
	
	public void showMessage(){
		pack();
		validate();
		setVisible(true);
	}
	
	

}
