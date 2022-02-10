package ca.ualberta.cs.poker.free.statistics;

import java.util.Vector;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 * This code was never completed. The idea was to have a general-purpose interface
 * to GnuPlot.
 * @author maz
 *
 */
public class Chart{
	String chartName;
	String xAxisName;
	String yAxisName;
	
	Vector<ChartSeries> data;
	
	public void setDatafileIndices(){
		for(int i=0;i<data.size();i++){
			data.get(i).datafileIndex=i;
		}
	}
	
	public String getDataFile(){
		return chartName+".dat";
	}
	
	public String getPostscriptFile(){
		return chartName+".ps";
	}
	
	public String getGnuplotFile(){
		return chartName+".gnu";
	}
	
	public String getGifFile(){
		return chartName+".gif";
	}
	
	public void writeDatafile(BufferedWriter writer) throws IOException{
		for(ChartSeries series:data){
			series.writeDatafile(writer);
			writer.write("\n\n");
		}
	}
	
	public void writeGnuplotPostscript(BufferedWriter writer) throws IOException{
		writer.write("set terminal postscript\n");
		writer.write("set output "+getPostscriptFile()+"\n");
		writeGnuplotPlot(writer);
	}
	
	public void writeGnuplotPlot(BufferedWriter writer) throws IOException{
		writer.write("plot ");
		for(int i=0;i<data.size()-1;i++){
			data.get(i).writeDisplayCommand(writer, getDataFile());
			writer.write(",");
		}
		data.lastElement().writeDisplayCommand(writer, getDataFile());
		writer.write("\n");
	}
	
	public void executeGnuplot() throws IOException{
	  //Process p = 
		Runtime.getRuntime().exec("gnuplot < " +getGnuplotFile());
	}
	
	public void createPostscriptFile() throws IOException{
		setDatafileIndices();
		BufferedWriter gnuplot = new BufferedWriter(new FileWriter(getGnuplotFile()));
		writeGnuplotPostscript(gnuplot);
		gnuplot.close();
		BufferedWriter datafile = new BufferedWriter(new FileWriter(getDataFile()));
		writeDatafile(datafile);
		datafile.close();
		
	}
}
