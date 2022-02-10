package ca.ualberta.cs.poker.free.statistics;

import java.awt.FileDialog;
import java.awt.Frame;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Iterator;
import java.util.Vector;
import java.util.Hashtable;

public class GraphicalMatchStatistics {
	public static Vector<MatchStatistics> loadFile(String file) throws IOException{
		Vector<MatchStatistics> result=new Vector<MatchStatistics>();
		File f = new File(file);
		if (!(f.exists())){
			System.err.println("File not found:"+file);
		} else if (f.isDirectory()){
			System.err.println("Descending into directory "+file);
			String[] files = f.list();
			if (!file.endsWith(File.separator)){
				file+=File.separator;
			}
			
			for(String subFile:files){
				result.addAll(loadFile(file+subFile));
			}
		} else if (file.endsWith(".res")){
			System.err.println("Result file "+file+" passed over.");
		} else {
		  System.err.println("Loading match "+file+"...");
		  MatchStatistics m = new MatchStatistics(file);
		  result.add(m);
		}
		return result;
	}
	
	public static void convertFile(String file) throws IOException{
		File f = new File(file);
		if (!(f.exists())){
			System.err.println("File not found:"+file);
		} else if (f.isDirectory()){
			System.err.println("Descending into directory "+file);
			String[] files = f.list();
			if (!file.endsWith(File.separator)){
				file+=File.separator;
			}
			
			for(String subFile:files){
				convertFile(file+subFile);
			}
		} else if (file.endsWith(".res")){
			System.err.println("Result file "+file+" passed over.");
		} else {
		  System.err.println("Loading match "+file+"...");
		  MatchStatistics m = new MatchStatistics(file);
		  m.normalizeHandNumbers();
		  File f1 = new File("uofa"+file);
		  FileOutputStream fos = new FileOutputStream(f1);
		  PrintStream ps = new PrintStream(fos);
          for(HandStatistics hand:m.hands){
				ps.println(hand.toUofAString());
		  }
		  ps.close();
		}
	}

	public static void computeStandardDeviation(String file) throws IOException{
			Vector<MatchStatistics> matches = loadFile(HandStatistics.remove(file,'"'));
			SeriesStatistics series = new SeriesStatistics(matches);
			MatchStatistics firstMatch = matches.firstElement();
			Hashtable<String,Double> stddev = series.getStandardDeviation(firstMatch.getFirstHandNumber(),firstMatch.getLastHandNumber());
			//System.out.println("Per game (in small blinds):");
			//System.out.println(MapOperationsD.mapToString(stddev));
			System.out.println("Standard deviation of utility per hand (in small bets):");
			Hashtable<String,Double> stddevSmallBetsPerHand = MapOperationsD.divide(stddev, firstMatch.getNumberOfHands()*firstMatch.getSmallBlindsInASmallBet()*2);
			System.out.println(MapOperationsD.mapToString(stddevSmallBetsPerHand));
			Hashtable<String,Integer> utility = series.getUtilities(firstMatch.getFirstHandNumber(),firstMatch.getLastHandNumber());
			
			Hashtable<String,Double> utilitySmallBets = MapOperationsD.cast(utility).divide(firstMatch.getNumberOfHands()*firstMatch.getSmallBlindsInASmallBet()*matches.size());
			System.out.println("Utility per hand (in Small Bets):");
			System.out.println(utilitySmallBets);
		}
	
	
	public static void main(String[] args) throws IOException{
		Frame f = new Frame();
		FileDialog fd = new FileDialog(f,"Load match statistics",FileDialog.LOAD);
		fd.setVisible(true);
		String dir = fd.getDirectory();
		///String file = fd.getFile();
		computeStandardDeviation(dir);//);+file);
		System.exit(0);
	}
	public static void considerPosition(String[] args) throws IOException{
		if (args.length!=0){
			String file = args[0];		
			Vector<MatchStatistics> match = loadFile(HandStatistics.remove(file,'"'));
			Iterator<MatchStatistics> iterator = match.iterator();
			MatchStatistics first = iterator.next();
			Vector<Hashtable<String, Integer> > data = first.getUtilityMapBySeatInSmallBlinds();
			int totalHands = first.hands.size();
			while(iterator.hasNext()){
				MatchStatistics nextMatch = iterator.next();
				nextMatch.addUtilityBySeat(data, nextMatch.getFirstHandNumber(), nextMatch.getLastHandNumber());
				totalHands += nextMatch.hands.size();
			}
			System.out.println("Total number of hands:"+totalHands);
			for(int i=0;i<data.size();i++){
				for(String s:data.get(i).keySet()){
					System.out.println("Player "+s+" in seat "+i+" received "+data.get(i).get(s));
				}
			}
		}
	}
		
	
}
