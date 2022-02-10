package ca.ualberta.cs.poker.free.statistics;

import java.util.Vector;

public class WeightedRandomVariable extends RandomVariable {
	Vector<RandomVariable> variables;
	Vector<Double> weights;
	
	public WeightedRandomVariable(DataSet ds){
		super(ds);
		variables = new Vector<RandomVariable>();
		weights = new Vector<Double>();
	}
	
	public void add(double weight, RandomVariable variable){
		variables.add(variable);
		weights.add(weight);
	}
	
	public void add(RandomVariable variable){
		add(1,variable);
	}
	
	public void averageIn(RandomVariable variable){
		add(1,variable);
		double everyWeight = 1.0/variables.size();
		for(int i=0;i<weights.size();i++){
			weights.set(i, everyWeight);
		}
		
	}
	
	public static WeightedRandomVariable getDifference(DataSet ds, RandomVariable a, RandomVariable b){
		WeightedRandomVariable result = new WeightedRandomVariable(ds);
		result.add(1,a);
		result.add(-1,b);
		return result;
	}
	@Override
	public Double getValue(DataPoint dp) {
		double result = 0;
		for(int i=0;i<variables.size();i++){
			result += (double)(weights.get(i))*(double)(variables.get(i).getValue(dp));
		}
		return result;
	}

	@Override
	public boolean isDefined(DataPoint dp) {
		for(RandomVariable rv:variables){
			if (!rv.isDefined(dp)){
				return false;
			}
		}
		// TODO Auto-generated method stub
		return true;
	}

}
