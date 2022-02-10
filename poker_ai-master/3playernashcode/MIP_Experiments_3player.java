package experiments;

import config.Config_3player;
import general.HelperFunctions;
import gurobi.GRB;
import gurobi.GRBException;
import solvers.MIP_3player;

// Main file for the random game experiments
public class MIP_Experiments_3player implements Config_3player {
	
	public static void main(String[] args) throws GRBException {
		solve(NUM_STRATEGIES, NUM_TRIALS, NUM_CORES, PRINT_INTERVAL, TIME_LIMIT);
	}

	public static void solve(int numStrategies, int numTrials, int numThreads, int printInterval, int timeLimit) throws GRBException {
		
		int[] numStr = {numStrategies, numStrategies, numStrategies};

		long numInfeasible = 0;
		long numOptimal = 0;
		long numOverTime = 0;
		long numSubOptimal = 0;

		double timeInfeasible = 0;
		double timeOptimal = 0;
		double timeOverTime = 0;
		double timeOverall = 0;
		double timeSubOptimal = 0;

		double[] times = new double[numTrials];
		
		double nextTime = 0;
		
		for (int t = 0; t < numTrials; ++t) {
			if (t % printInterval == 0) {
				System.out.println(t);
				System.out.println(nextTime);
			}

			double[][][][] matrix = Game.makeRandom_3player(numStr);
			
			MIP_3player nash = new MIP_3player();
			//long startTime = System.currentTimeMillis();
			int nextStatus = nash.solve(matrix, numStr, numThreads, timeLimit);
			//long endTime = System.currentTimeMillis();
			//double nextTime = (endTime - startTime)/1000.0;
			nextTime = nash.getRunningTime();
			if (nextTime > timeLimit)
				nextTime = timeLimit;
			times[t] = nextTime;

			if (nextStatus == GRB.Status.OPTIMAL) {
				++numOptimal;
				timeOptimal += nextTime;
			}
			else if (nextStatus == GRB.Status.TIME_LIMIT) {
				++numOverTime;
				timeOverTime += nextTime;
			}
			else if (nextStatus == GRB.Status.SUBOPTIMAL) {
				//System.out.println("SUBOPTIMAL: " + nextStatus);
				++numSubOptimal;
				timeSubOptimal += nextTime;
			}
			else {
				System.out.println("EHEHE: " + nextStatus);
				System.exit(0);
				++numInfeasible;
				timeInfeasible += nextTime;
			}
			timeOverall += nextTime;

		}
		System.out.println("NUM INFEASIBLE: " + numInfeasible);
		System.out.println("NUM OVER TIME: " + numOverTime);
		System.out.println("NUM OPTIMAL: " + numOptimal);
		System.out.println("NUM SUBOPTIMAL: " + numSubOptimal);
		System.out.println("TOTAL NUMBER: " + (numInfeasible + numOverTime + numOptimal + numSubOptimal));

		System.out.println("TIME INFEASIBLE: " + timeInfeasible);
		System.out.println("TIME OVER TIME: " + timeOverTime);
		System.out.println("TIME OPTIMAL: " + timeOptimal);
		System.out.println("TIME SUBOPTIMAL: " + timeSubOptimal);
		System.out.println("TIME OVERALL: " + timeOverall);

		System.out.println("AVERAGE TIME INFEASIBLE: " + timeInfeasible/numInfeasible);
		System.out.println("AVERAGE TIME OVERTIME: " + timeOverTime/numOverTime);
		System.out.println("AVERAGE TIME OPTIMAL: " + timeOptimal/numOptimal);
		System.out.println("AVERAGE TIME SUBOPTIMAL: " + timeOptimal/numOptimal);
		System.out.println("AVERAGE TIME OVERALL: " + timeOverall/(numInfeasible + numOverTime + numOptimal));
		System.out.println("MEDIAN TIME OVERALL: " + HelperFunctions.findMedian(times, times.length));
	}
}