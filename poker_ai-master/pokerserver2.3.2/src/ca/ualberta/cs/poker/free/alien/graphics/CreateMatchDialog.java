package ca.ualberta.cs.poker.free.alien.graphics;

import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextField;

import ca.ualberta.cs.poker.free.alien.ClientMatch;
import ca.ualberta.cs.poker.free.alien.GraphicalAlienClient;

public class CreateMatchDialog extends JDialog implements ActionListener{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	String dialogFinished;
	GraphicalAlienClient parent;
	private JList opponentList;
	private JList localBotList;
	private JTextField matchName;
	private JTextField repeatCount;
	int numRepeats;
	Object[] selectedOpponents;
	Object[] selectedLocalBots;
	String matchNameString;
	
	public CreateMatchDialog(GraphicalAlienClient parent){
		super(parent,"Create New Matches",true);
		this.parent = parent;
		//setBounds(GraphicalAlienClient.STARTING_WIDTH, GraphicalAlienClient.STARTING_HEIGHT, this.getWidth(), this.getHeight());
		addComponents();
	}
	
	public void addComponents(){
		// load the possible opponents from the profile
		Vector<String> opponents = parent.profile.opponents;
		
		// in the old versions, there will be no opponents, so add the old
		if ( opponents.size() == 0 ) {
			opponents.add("BankrollHyperborean");opponents.add("SeriesHyperborean");
			opponents.add("Monash-BPP");opponents.add("BluffBot");opponents.add("Teddy");
		}
		opponentList = new JList(opponents);
		opponentList.setToolTipText("Possible opponent bots for the match");
		Vector<String> localBotNames = new Vector<String>(parent.localBots.keySet());
		localBotList = new JList(localBotNames);
		localBotList.setToolTipText("Local bots on your machine for the match");
		setLayout(new FlowLayout());
		add(new JScrollPane(opponentList));
		add(new JScrollPane(localBotList));
		JPanel buttons = new JPanel(new GridLayout(3,2));
		buttons.add(new JLabel("Name:"));
		matchName = new JTextField(20);
		buttons.add(matchName);
		buttons.add(new JLabel("Repeat:"));
		repeatCount = new JTextField(20);
		repeatCount.setText("1");
		buttons.add(repeatCount);
		JButton okButton = new JButton("OK");
		okButton.addActionListener(this);
		okButton.setActionCommand("OKGETMATCHES");
		buttons.add(okButton);
		JButton cancelButton = new JButton("Cancel");
		cancelButton.setActionCommand("CANCELGETMATCHES");
		cancelButton.addActionListener(this);
		buttons.add(cancelButton);
		add(buttons);
		pack();
		validate();
		setLocationRelativeTo(parent);
	}
	public Vector<ClientMatch> getMatches(){
		dialogFinished = null;
		setVisible(true);
		if (dialogFinished.equals("OKGETMATCHES")){			
			Vector<ClientMatch> result = new Vector<ClientMatch>();
			for(Object opponent:selectedOpponents){
				for(Object localBot:selectedLocalBots){
					//ClientMatch cm = new ClientMatch();
					ClientMatch cm = new ClientMatch("HEADSUP",matchNameString, (String)localBot, (String)opponent);
					
					if (numRepeats==1){
						result.add(cm);
					} else {
						for(int i=0;i<numRepeats;i++){
							result.add(new ClientMatch(cm,i));
						}
					}
				}
			}
			return result;
		}
		return null;
	}
	
	public void showError(String message){
		JOptionPane.showMessageDialog(this.parent,message,"Error",JOptionPane.ERROR_MESSAGE);		
	}
	public void actionPerformed(ActionEvent e) {
		String command = e.getActionCommand();
		
		if (command.equals("OKGETMATCHES")){
			matchNameString = matchName.getText();
			if (!parent.validName(matchNameString,"Match")){
				return;
			}
			selectedOpponents = opponentList.getSelectedValues();
			selectedLocalBots = localBotList.getSelectedValues();
			try{
			numRepeats = Integer.parseInt(repeatCount.getText());
			if (numRepeats<=0){
				showError("Repeats must be a positive number");
				return;				
			}
			if( numRepeats > 1000) {
				showError("Repeats must be below 1000. Note each repeat is a duplicate match of 3000 hands, " +
						"and may take up to 6 hours depending on opponent");
				return;
			}
			if (selectedOpponents.length==0){
				showError("Must select at least one opponent");
				return;
			} else if (selectedLocalBots.length==0){
				showError("Must select at least one local bot");
				return;
			}
			
			dialogFinished = command;
			setVisible(false);
			} catch (NumberFormatException nfe){
				JOptionPane.showMessageDialog(this.parent,"Repeats must be a positive number","Error",JOptionPane.ERROR_MESSAGE);
				return;
			}
			
		} else if (command.equals("CANCELGETMATCHES")){
			dialogFinished = command;
			setVisible(false);
		}
	}
	
}
