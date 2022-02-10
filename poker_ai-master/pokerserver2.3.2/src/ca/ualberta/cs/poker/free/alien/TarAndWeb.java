package ca.ualberta.cs.poker.free.alien;

import java.nio.channels.FileChannel;
import java.util.Vector;
import java.io.*;

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

public class TarAndWeb {
	Vector<String> files;
	//String tempDirectory;
	String tempTarFile;
	String user;
	String destDirectory;
	String directoryToCopyFiles;
	String finishedTarBallDirAndName;
	String sessionNumber;
	
	static String staticUser;
	static String webDirectory = "/local/data/web/htdocs/";
	static String dataTempDirectory = "data/temp/";
	static String staticDestDir;

	public TarAndWeb( String _user, String _sessionNumber, Vector<String> _files ) {

		user = _user;
		files = _files;
		sessionNumber =_sessionNumber;
		
		directoryToCopyFiles = dataTempDirectory + sessionNumber;   // data/temp/session82
		finishedTarBallDirAndName = "/local/data/web/htdocs/" + _user + "/" + sessionNumber + ".tar";
		
	}
	
	public static void initStatic(){
		staticUser = "test";
		staticDestDir = "/local/data/web/htdocs/" + staticUser + "/";
	}

	
	public void execute() throws IOException{
		//System.err.println("Executing TarAndEmail");
		//System.err.println(this.toString());
		
		createDirectory(directoryToCopyFiles);
		for(String filename:files){
			System.out.println("TarAndWeb.execute():filename="+filename);
			if (new File(filename).exists()){
				copy(filename,directoryToCopyFiles);
			}
		}
		
		// tar -cf /www/session82.tar -C data/temp session82
		tarDirectory( finishedTarBallDirAndName,  dataTempDirectory, sessionNumber );
		//tarDirectory( destDirectory, tempDirectory, tempDir, tempTarFile);
		
		
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
		// boolean success = 
		directoryFile.mkdirs();
		
		/*if ( !success ) {
			if (directoryFile.exists()){
				throw new IOException("Directory already exists!");
			}
			throw new IOException("Could not create directory");	
		}*/
		
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
	public static void tarDirectory(String finalTarballDirAndName, String dirToTarFrom, String directoryToTarRelativeToCurrent) throws IOException{
		String command = "tar -cf " + finalTarballDirAndName + " -C " + dirToTarFrom + " " + directoryToTarRelativeToCurrent;
		
		System.out.println( "Execing " + command );
		// this will place the resulting tarfile in the 
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
	

	
	public String toString(){
		String result = "files:\n";
		for(String file:files){
			result +=file+"\n";
		}
		result+= "User: " + user;
		//result+= "Temp Directory:" + tempDirectory+"\n";
		result+= "Temp Tar File:"+ tempTarFile+"\n";
		return result;
	}
}
