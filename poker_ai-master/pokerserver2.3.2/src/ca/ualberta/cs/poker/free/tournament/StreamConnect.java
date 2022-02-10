package ca.ualberta.cs.poker.free.tournament;
import java.io.*;

/**
 * If a program called by Process sends data to stdout
 * which is not read, it blocks.
 * This class allows you to connect stdout of a process
 * to an inputstream
 */
public class StreamConnect implements Runnable{
  /** 
   * The stream to be read from.
   * Usually, the stream from the process.
   */
  InputStream is;

  /**
   * The stream to be written to.
   * Usually a file or stdout or stderr.
   */
  OutputStream os;

  /**
   * Generate a new StreamConnect object.
   */
  public StreamConnect(InputStream is, OutputStream os){
    this.is = is;
    this.os = os;
  }

  /**
   * A thread which takes bytes from the input stream
   * to the output stream.
   */
  public void run(){
    while(true){
      try{
        int bytesAvailable = is.available();
       
        if (bytesAvailable>0){
          byte[] buffer = new byte[bytesAvailable];
          is.read(buffer);
          os.write(buffer);
        }
      } catch (Exception e){
      }
      try{
        Thread.sleep(20);
      } catch(InterruptedException e){
	return;
      }
    }
  }
}
