package ca.ualberta.cs.poker.free.alien;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.channels.FileChannel;
import java.util.Date;
import java.util.Properties;
import java.util.Vector;
import java.io.*;
import javax.mail.*;
import javax.mail.internet.*;
import javax.activation.*;

import ca.ualberta.cs.poker.free.tournament.StreamConnect;

/**
 * This class will take the result files from a match/tournament and tar the files
 * and email the log/results/any file to the recipient. Intended use is for informing
 * poker competitors of the results of the matches.
 * 
 * Note that the directory delimiter should be a forward slash "/".
 * 
 * @author Christian Smith
 * @author Martin Zinkevich
 *
 */

public class TarAndEmail {
	Vector<String> files;
	String subject;
	String body;
	String destinationAddress;
	String tempDirectory;
	String tempTarFile;

	
	/**
	 * Send from a particular address.
	 * TODO initialize during AlienNode initialization.
	 */
	static String senderAddress;
	
	
	/**
	 * Server for e-mail
	 * TODO initialize during AlienNode initialization.
	 */
	static InetAddress server=null;
	
	/**
	 * Eventually
	 * What port should we use? Do we even need to specify it?
	 * TODO initialize during AlienNode initialization.
	 */
	static int port=0; 

	public TarAndEmail( String _subject, String _body, String _destinationAddress, 
			String _tempDirectory, String _tempTarFile, Vector<String> _files) {
		
		/*try {
			server=InetAddress.getByName("mail.cs.ualberta.ca");
		} catch (UnknownHostException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		senderAddress = "chsmith@cs.ualberta.ca";
		*/
		
		subject = _subject;
		body = _body;
		destinationAddress = _destinationAddress;
		tempDirectory = _tempDirectory;
		tempTarFile = _tempTarFile;
		files = _files;
		
		port = 0;
	}
	
	public static void initStatic(){
		try{
			server=InetAddress.getByName("mail.cs.ualberta.ca");
		} catch (UnknownHostException unk){
			
		}
	
		senderAddress = "chsmith@cs.ualberta.ca";
		port = 0;
	}

	
	public void execute() throws IOException{
		//System.err.println("Executing TarAndEmail");
		//System.err.println(this.toString());
		createDirectory(tempDirectory);
		for(String filename:files){
			copy(filename,tempDirectory);
		}
		tarDirectory(tempDirectory,tempTarFile);
		emailFile(tempTarFile,subject,body,destinationAddress);
		
	}
	/**
	 * Copies a file. Throws an IOException if an error occurs.
	 * @param origin filename of the origin
	 * @param destination filename of the destination (or parent directory)
	 */
	public static void copy(String origin, String destinationdir) throws IOException{
	    
		// we need the actual filename, despite how many dir deep it is
		// to construct the destination filename
		File file = new File(origin);
		String originFileName = file.getName();

		// Create channel on the source
		FileChannel srcChannel = new FileInputStream(origin).getChannel();

		// Create channel on the destination
		FileChannel dstChannel = new FileOutputStream(destinationdir + "/"
				+ originFileName).getChannel();

		// Copy file contents from source to destination
		dstChannel.transferFrom(srcChannel, 0, srcChannel.size());

		// Close the channels
		srcChannel.close();
		dstChannel.close();
	}
	
	/**
	 * Creates a directory. Note that this may fail because the directory already
	 * exists.
	 * 
	 * @param directory
	 * @throws IOException if directory creation failed
	 */
	public static void createDirectory(String directory) throws IOException{
		File directoryFile = new File(directory);
		boolean success = directoryFile.mkdir();
		
		if ( !success ) {
			if (directoryFile.exists()){
				throw new IOException("Directory already exists!");
			}
			throw new IOException("Could not create directory");	
		}
		
	}
	
	/**
	 * Tar a directory. Note that this operation should be done from the
	 * parent directory of the directory -- this will be the parents dir by
	 * default since we just created a directory for the temp files from
	 * the working directory.
	 * @param directory
	 * @param result the resulting tar file.
	 * @throws IOException if the directory does not exist.
	 */
	public static void tarDirectory(String directory, String result) throws IOException{
		String command = "tar -cf " + result + " " + directory;
		//System.err.println("TAR COMMAND:"+command);
		Vector<Thread> looseThreads=new Vector<Thread>();
		try{
			    Process p = Runtime.getRuntime().exec(command);
			    StreamConnect sc = new StreamConnect(
			    p.getInputStream(),System.out);
			    Thread tsc = new Thread(sc);
			    tsc.start();
			    looseThreads.add(tsc);
			    StreamConnect scerr = new StreamConnect(
			    p.getErrorStream(),System.err);
			    Thread tscerr = new Thread(scerr);
			    tscerr.start();
			    looseThreads.add(tscerr);
			    p.waitFor();
			    
			  } catch (InterruptedException e){
			  }
		for(Thread t:looseThreads){
			t.interrupt();
		}
		// this would make the system wait for tar to finish
//		try { tarProcess.waitFor();
//		} catch (InterruptedException e) {
//			e.printStackTrace();  }
	}
	
	/**
	 * E-mails a file
	 * @param file
	 * @param subject the subject of the message
	 * @param body the body of the message
	 * @param recipient
	 * @throws IOException
	 */
	public static void emailFile(String file, String subject, String body, String recipient) throws IOException{
		
		try {
			// Get system properties
			Properties props = System.getProperties();

			// -- Attaching to default Session, or we could start a new one --
			props.put("mail.smtp.host", server.getHostName());

			//	Get session
			Session session = Session.getDefaultInstance(props, null);

			// Create a new message
			Message msg = new MimeMessage(session);

			// Set the FROM and TO fields 
			msg.setFrom(new InternetAddress(senderAddress));
			msg.setRecipients(Message.RecipientType.TO, InternetAddress.parse( recipient, false));
			
			// Set the subject and body text --
			msg.setSubject(subject);

			// -- We could include CC recipients too --
			// if (cc != null) msg.setRecipients(Message.RecipientType.CC,InternetAddress.parse(cc, false));
			
			
			// -- Set some other header information --
			msg.setHeader("X-Mailer", "PokerServerMail");
			msg.setSentDate(new Date());

			// this would set the body normally, but we're using an attachment
			//msg.setText(body);

			// since we're attaching a file we will use a multipart message
			//create the message part 
			MimeBodyPart messageBodyPart = new MimeBodyPart();

			// fill message body with body text
			messageBodyPart.setText(body);

			// multipart for attachments
			Multipart multipart = new MimeMultipart();
			multipart.addBodyPart(messageBodyPart);

			// Part two is attachment
			messageBodyPart = new MimeBodyPart();
			DataSource source = new FileDataSource(file);
			messageBodyPart.setDataHandler(new DataHandler(source));
			messageBodyPart.setFileName(file);
			multipart.addBodyPart(messageBodyPart);

			// Put parts including attachment in message
			msg.setContent(multipart);

			// -- Send the message --
			Transport.send(msg);

			//System.out.println("Message sent OK.");

		} catch (Exception ex) {
			ex.printStackTrace();
		}
	
	}
	
	public String toString(){
		String result = "files:\n";
		for(String file:files){
			result +=file+"\n";
		}
		result+= "Subject:"+subject+"\n";
		result+= "Body:"+body+"\n";
		result+= "Destination Address:"+destinationAddress+"\n";
		result+= "Temp Directory:" + tempDirectory+"\n";
		result+= "Temp Tar File:"+ tempTarFile+"\n";
		return result;
	}
}
