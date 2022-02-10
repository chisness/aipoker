package ca.ualberta.cs.poker.free.alien;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.io.*;
//import java.net.URL;
//import java.nio.channels.FileChannel;

/**
 * A window for providing help to the user on how to use the
 * project manager.  
 *
 * @author Jeffery Grajkowski
 */
public class HelpViewer extends JFrame implements HyperlinkListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private JEditorPane content;
	private JButton closeButton;
	    
	private final static int WINDOW_WIDTH = 800;
	private final static int WINDOW_HEIGHT = 500;
	
	
	/**
     * Tries to find the location of the help menu, then sets it to the first page determined by whether manager is true or false.
     */
	public HelpViewer() {
		super.getContentPane().setLayout(new BorderLayout());
		
		content = new JEditorPane();
		content.setEditable(false);
		content.setEditorKit(new javax.swing.text.html.HTMLEditorKit());
		content.addHyperlinkListener(this);
		
		try {
			File helpfile = new File("manual/help.html");
			content.setPage( "file:///" + helpfile.getCanonicalPath());
		}
		catch (IOException ioe) {
			if (JOptionPane.showConfirmDialog(null, "The file help.html can not be found in the working directory. Would you like to search for it manually?", "Help Not Found", JOptionPane.YES_NO_OPTION) == JOptionPane.YES_OPTION) {
				while (true) {
					final JFileChooser fc = new JFileChooser();
					fc.setDialogType(JFileChooser.OPEN_DIALOG);
					fc.setApproveButtonMnemonic(KeyEvent.VK_S);
					fc.setApproveButtonText("Select");
					fc.setApproveButtonToolTipText("Select Help Index");
					fc.setDialogTitle("Select Help Index File");
					int returnVal = fc.showOpenDialog(new Panel());
					try {
						if (returnVal == JFileChooser.APPROVE_OPTION) {
							
							content.setPage("file:///" + fc.getSelectedFile().toString() ); //.replace('\\', '/'));
							break;
						}
					}
					catch (IOException ioe2) {//ioe2.getMessage()
						if (JOptionPane.showConfirmDialog(null, "Unable to open file: " + "file:///" + fc.getSelectedFile().toString().replace('\\', '/') + ".  Try another file?", "Help Not Found", JOptionPane.YES_NO_OPTION) != JOptionPane.YES_OPTION)
							return;
					}
				}
			}
			else
				return;
		}
		
		JScrollPane contentWrapper = new JScrollPane(content);
		contentWrapper.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
		contentWrapper.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
		contentWrapper.setPreferredSize(new Dimension(WINDOW_WIDTH, WINDOW_HEIGHT));
		
		super.getContentPane().add(contentWrapper, BorderLayout.CENTER);
		
		closeButton = new JButton("Close");
		closeButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				dispose();
			}
		});
		JPanel closeWrapper = new JPanel();
		closeWrapper.setLayout(new FlowLayout(FlowLayout.RIGHT));
		closeWrapper.add(closeButton);
		closeWrapper.add(new Box.Filler(new Dimension(100, 1), new Dimension(100, 1), new Dimension(100, 1)));
		super.getContentPane().add(closeWrapper, BorderLayout.SOUTH);
		
		pack();
		setTitle("Help System");
		setResizable(true);
		setVisible(true);
	}
	
	
	public void hyperlinkUpdate(HyperlinkEvent e) {
		// MAZ: I made this able to open remote files as well.
		//System.err.println(e.getURL());
		//System.err.println(e.getEventType() == HyperlinkEvent.EventType.ACTIVATED);
		//System.err.println(e.getURL().getProtocol().equals("file"));
		if (e.getEventType() == HyperlinkEvent.EventType.ACTIVATED) {
			try {
			//	if (e.getURL().getProtocol().equals("file"))	// only open local files
					content.setPage(e.getURL());
			} catch (IOException ioe) {
				
			}
		}
	}
}

