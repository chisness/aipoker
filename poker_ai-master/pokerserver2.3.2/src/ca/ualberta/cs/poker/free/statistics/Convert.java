package ca.ualberta.cs.poker.free.statistics;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;

public class Convert {
	String inputType;
	String format;
	String outputDir;
	Vector<String> inputDir;
	int handIndex;
	
	public Convert(){
		format = "PA";
		outputDir = "converted";
		inputType = "normal";
		inputDir = new Vector<String>();
		handIndex = 0;
	}
	public static void showUsage(){
		System.err.println("Usage:Convert [-o <OUTPUTDIRECTORY> |<INPUTDIRECTORY>| -f <FORMAT>| -h <HANDINDEX> | -t <INPUTTYPE>]*");
		System.exit(-1);
	}

	public void handleArgs(String[] args){
		for(int i=0;i<args.length;i++){
			if (args[i].equals("-o")){
				i++;
				if (i>=args.length){
					showUsage();
				}
				outputDir=args[i];
			} else if (args[i].equals("-f")){
				i++;
				if (i>=args.length){
					showUsage();
				}
				format=args[i];				
			} else if (args[i].equals("-h")){
				i++;
				if (i>=args.length){
					showUsage();
				}
				handIndex=Integer.parseInt(args[i]);				
			} else if (args[i].equals("-i")){
				i++;
				if (i>=args.length){
					showUsage();
				}
				inputType = args[i];
			} else {
				inputDir.add(args[i]);
			}
		}		
	}
	
	public static void main(String[] args) throws FileNotFoundException, IOException{
		Convert c = new Convert();
		c.handleArgs(args);
		c.convert();
	}
	
	public void convert() throws FileNotFoundException, IOException{
		if (inputDir.isEmpty()){
			inputDir.add("data\\results");
		}
		
		for(String file:inputDir){
			convert(file);
		}
	}
	
	public void convert(String file) throws FileNotFoundException, IOException{
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
				convert(file+subFile);
			}
		} else if (!file.endsWith(".log")){
			System.err.println("Result file "+file+" passed over.");
		} else {
		  System.err.println("Loading match "+file+"...");
		  MatchStatistics m;
		  if (inputType.equals("normal")){
			  m = new MatchStatistics(file);
		  } else {
			  m = new MatchStatistics();
			  m.readPokerAcademy(new BufferedReader(new FileReader(file)));
		  }
		  System.err.println("Loaded match:"+file);
		  File inputFile = new File(file);
		  String name = inputFile.getName();
		  File outputDirFile = new File(outputDir);
		  File outputFile = new File(outputDir,name);
		  outputDirFile.mkdirs();
		  BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
		  String heroName = m.hands.firstElement().names.firstElement();
		  double initialPAStack = 100000;
		  double heroStackBlinds = initialPAStack;
		  double adversaryStackBlinds = initialPAStack;
		  for(HandStatistics hand:m.hands){
			  if (format.equals("PA")){
				  writer.write(hand.toPAString(heroName,handIndex,heroStackBlinds,adversaryStackBlinds)+"\n");
				  double heroStackChange = hand.getNetSmallBlinds(heroName);
				  heroStackBlinds += heroStackChange/2;
				  adversaryStackBlinds -= heroStackChange/2;
				  handIndex++;
			  } else if (format.equals("uofa")){
				  writer.write(hand.toUofAString()+"\n");
			  }
		  }
		  writer.close();
		}
	}
}
