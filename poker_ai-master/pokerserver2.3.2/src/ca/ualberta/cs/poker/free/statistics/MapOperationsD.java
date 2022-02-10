package ca.ualberta.cs.poker.free.statistics;

import java.util.Hashtable;

public class MapOperationsD extends Hashtable<String,Double>{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public MapOperationsD(Hashtable<String,Double> copy){
		super(copy);
	}
	
	
	
	public MapOperationsD(){
		super();
	}
	
	public static MapOperationsD cast(Hashtable<String,Integer> initial){
		return new MapOperationsD(castD(initial));
	}
	
	public MapOperationsD square(){
		return new MapOperationsD(square(this));
	}
	
	public MapOperationsD subtract(Hashtable<String,Double> subtrahend){
		return new MapOperationsD(subtract(this,subtrahend));
	}
	

	public MapOperationsD divide(double divisor){
		return new MapOperationsD(divide(this,divisor));
	}

	public MapOperationsD multiply(double divisor){
		return new MapOperationsD(multiply(this,divisor));
	}
	
	public void increment(Hashtable<String,Double> addend){
		increment(this,addend);
	}
	

	public void decrement(Hashtable<String,Double> addend){
		decrement(this,addend);
	}
	
	
	/*
	public static void incrementI(Hashtable<String,Integer> initial,Hashtable<String,Integer> addend){
		assert(initial.keySet().equals(addend.keySet()));
		
		for(String key:initial.keySet()){
			initial.put(key, initial.get(key)+addend.get(key));
		}
	}
	*/
	public static void increment(Hashtable<String,Double> initial,Hashtable<String,Double> addend){
		if (initial.isEmpty()){
			for(String key:addend.keySet()){
				initial.put(key,+addend.get(key));
			}
		} else {
			assert(initial.keySet().equals(addend.keySet()));
			
			for(String key:initial.keySet()){
			initial.put(key, initial.get(key)+addend.get(key));
		}
		}
	}
	
	public static void decrement(Hashtable<String,Double> initial,Hashtable<String,Double> addend){
		if (initial.isEmpty()){
			for(String key:addend.keySet()){
				initial.put(key,-addend.get(key));
			}
		} else {		
			assert(initial.keySet().equals(addend.keySet()));

			for(String key:initial.keySet()){
			initial.put(key, initial.get(key)-addend.get(key));
		}
		}
	}
	
	public static Hashtable<String, Double> subtract(Hashtable<String,Double> initial,Hashtable<String,Double> addend){
		assert(initial.keySet().equals(addend.keySet()));
		Hashtable<String,Double> result = new Hashtable<String,Double>();

		for(String key:initial.keySet()){
			result.put(key, initial.get(key)-addend.get(key));
		}
		return result;
	}
	
	public static Hashtable<String,Double> square(Hashtable<String,Double> initial){
		Hashtable<String,Double> result = new Hashtable<String,Double>();

		for(String key:initial.keySet()){
			result.put(key, initial.get(key)*initial.get(key));
		}
		return result;

	}
	
	public static Hashtable<String,Double> divide(Hashtable<String,Double> initial, double divisor){
		Hashtable<String,Double> result = new Hashtable<String,Double>();

		for(String key:initial.keySet()){
			result.put(key, initial.get(key)/divisor);
		}
		return result;

	}

	public static Hashtable<String,Double> multiply(Hashtable<String,Double> initial, double multiplier){
		Hashtable<String,Double> result = new Hashtable<String,Double>();

		for(String key:initial.keySet()){
			result.put(key, initial.get(key)*multiplier);
		}
		return result;

	}

	
	public static Hashtable<String,Double> sqrt(Hashtable<String,Double> initial){
		Hashtable<String,Double> result = new Hashtable<String,Double>();
		for(String key:initial.keySet()){
			result.put(key, Math.sqrt(initial.get(key)));
		}
		return result;
	}
	
	public String toString(){
		return mapToString(this);
	}
	
	public static String mapToString(Hashtable<String,Double> initial){
		String result="";
		for(String key:initial.keySet()){
			result += key + ":"+initial.get(key)+"\n";
		}
		return result;
	}

	/*public static String mapToString(Hashtable<String,Double> initial, int precision){
		String result="";
		for(String key:initial.keySet()){
			result += key + ":"+initial.get(key)+"\n";
		}
		return result;
	}*/

	public static Hashtable<String,Double> castD(Hashtable<String,Integer> initial){
		Hashtable<String,Double> result= new Hashtable<String,Double>();
		for(String key:initial.keySet()){
			result.put(key, new Double(initial.get(key)));
		}
		return result;
	}
	
	/**
	 * Returns a rounded version of the map
	 * @param initial the initial map
	 * @param precision the number of significant digits past zero
	 * @return the rounded version of the map
	 */
	
}
