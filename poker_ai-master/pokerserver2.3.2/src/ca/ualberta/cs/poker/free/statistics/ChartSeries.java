package ca.ualberta.cs.poker.free.statistics;

import java.awt.geom.*;
import java.awt.geom.Point2D.Double;
import java.awt.*;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Vector; 

/**
 * This code was never completed. It was designed to hold a particular series for a
 * gnuplot chart.
 * @author maz
 *
 */
public class ChartSeries {
	public Vector<Point2D.Double> data;
	
	/**
	 * When a multi-set data file is created, what is the index of this series?
	 */
	public int datafileIndex;
	String name;
	Color color;
	String lineFormat;
	String pointFormat;
	
	
	
	public ChartSeries(Vector<Double> data, String name, Color color, String lineFormat, String pointFormat) {
		this.data = data;
		this.datafileIndex = 0;
		this.name = name;
		this.color = color;
		this.lineFormat = lineFormat;
		this.pointFormat = pointFormat;
	}

	public void writeDatafile(BufferedWriter writer) throws IOException{
		for(Double point:data){
			writer.write(""+point.x+" "+point.y+"\n");
		}
	}
	
	public void writeDisplayCommand(BufferedWriter writer, String datafile) throws IOException{
		writer.write("\""+datafile+"\" index "+datafileIndex+" using 1:2");
		if (name!=null){
			writer.write(" title "+name);
		}
		/*if (lineFormat!=null||pointFormat!=null){
			writer.writ
		}*/
	}

	public ChartSeries(Vector<Point2D.Double> data){
		this(data,null,null,null,null);		
	}
	
	/**
	 * Gets the maximum y-value.
	 * @return the maximum value, or Double.NEGATIVE_INFINITY if there is no data.
	 */
	public double getDataMaxOrdinate(){
		double maxSoFar = java.lang.Double.NEGATIVE_INFINITY;
		for(Point2D.Double point:data){
			if (point.y>maxSoFar){
				maxSoFar=point.y;
			}
		}
		return maxSoFar;
	}
	
	/**
	 * Gets the minimum x-value.
	 * @return the minimum value, or Double.POSITIVE_INFINITY if there is no data.
	 */
	public double getDataMinOrdinate(){
		double minSoFar = java.lang.Double.POSITIVE_INFINITY;
		for(Point2D.Double point:data){
			if (point.y<minSoFar){
				minSoFar=point.y;
			}
		}
		return minSoFar;
	}
	
	public void createRepresentation(){
		
	}
	
	
	/**
	 * Gets the maximum x-value.
	 * @return the maximum value, or Double.NEGATIVE_INFINITY if there is no data.
	 */
	public double getDataMaxAbscissa(){
		double maxSoFar = java.lang.Double.NEGATIVE_INFINITY;
		for(Point2D.Double point:data){
			if (point.x>maxSoFar){
				maxSoFar=point.x;
			}
		}
		return maxSoFar;
	}
	
	/**
	 * Gets the minimum x-value.
	 * @return the minimum value, or Double.POSITIVE_INFINITY if there is no data.
	 */
	public double getDataMinAbscissa(){
		double minSoFar = java.lang.Double.POSITIVE_INFINITY;
		for(Point2D.Double point:data){
			if (point.x<minSoFar){
				minSoFar=point.x;
			}
		}
		return minSoFar;
	}
	
	
	
}