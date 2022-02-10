package solvers;

import java.util.Random;

import config.Config_3player;
import general.Constants;
import gurobi.GRB;
//import gurobi.GRBConstr;
import gurobi.GRBEnv;
import gurobi.GRBException;
import gurobi.GRBLinExpr;
import gurobi.GRBModel;
import gurobi.GRBQuadExpr;
import gurobi.GRBVar;

// Uses new Gurobi 9.0 nonconvex qcp solver
public class MIP_3player implements Constants, Config_3player {

	private double[][] strategies;

	//private double[] brValues = new double[NUM_PLAYERS];

	private double[] epsilons = new double[NUM_PLAYERS];

	private double[] neValues = new double[NUM_PLAYERS];
	
	private double runningTime = 0;

	public double[][] getStrategies() {
		return strategies;
	}

	public double[] getEpsilons() {
		return epsilons;
	}

	public double[] getNEValues() {
		return neValues;
	}
	
	public double getRunningTime() {
		return runningTime;
	}

	public static void main(String[] args) throws GRBException {

		int numThreads = 0;
		if (args.length > 0) { 
			numThreads = Integer.parseInt(args[0]);
		}

		int[] numStrategies = {NUM_STRATEGIES,NUM_STRATEGIES,NUM_STRATEGIES};
		Random r = new Random();

		double[][][][] matrix = new double[NUM_PLAYERS][numStrategies[0]][numStrategies[1]][numStrategies[2]];
		for (int player = 0; player < NUM_PLAYERS; ++player) {
			for (int i = 0; i < numStrategies[0]; ++i) {
				for (int j = 0; j < numStrategies[1]; ++j) {
					for (int k = 0; k < numStrategies[2]; ++k) {
						matrix[player][i][j][k] = r.nextDouble();
						//System.out.println(matrix[player][i][j][k]);
					}
				}
			}
		}

		MIP_3player Nasher = new MIP_3player();
		//Nasher.solve(matrix, numStrategies, numThreads);
		Nasher.solve(matrix, numStrategies, numThreads, TIME_LIMIT);
	}

	public int solve(double[][][][] matrix, int[] numStrategies, int numThreads, int timeLimit) throws GRBException {
		//int result = Nasher(3, numStrategies, matrix);

		strategies = new double[NUM_PLAYERS][];
		for (int i = 0; i < NUM_PLAYERS; ++i)
			strategies[i] = new double[numStrategies[i]];

		int result = Nasher(NUM_PLAYERS, numStrategies, matrix, numThreads, timeLimit);

		/*
		computeBestResponse(0, NUM_STRATEGIES, strategies[1], strategies[2], matrix);
		computeBestResponse(1, NUM_STRATEGIES, strategies[0], strategies[2], matrix);
		computeBestResponse(2, NUM_STRATEGIES, strategies[0], strategies[1], matrix);

		neValues = computeNeValues(strategies[0], strategies[1], strategies[2], matrix);
		epsilons = computeEpsilons(brValues, neValues);
		*/

		/*
		System.out.println("NE VALUES");
		for (int i = 0; i < NUM_PLAYERS; ++i)
			System.out.println(neValues[i]);

		System.out.println("BR VALUES");
		for (int i = 0; i < NUM_PLAYERS; ++i)
			System.out.println(brValues[i]);

		for (int i = 0; i < NUM_PLAYERS; ++i) {
			System.out.println("PLAYER " + i + ": ");
			for (int j = 0; j < numStrategies[i]; ++j)
				System.out.println(strategies[i][j]);
		}

		System.out.println("EPSILONS");
		for (int i = 0; i < NUM_PLAYERS; ++i)
			System.out.print(epsilons[i] + " ");
			*/
		
		return result;
	}

	private int Nasher (int numPlayers, int[] numStrategies, double[][][][] payoffs, int numThreads, int timeLimit) throws GRBException {
		GRBEnv env = new GRBEnv();
		GRBModel model = new GRBModel(env);
		//model.set(GRB.IntParam.OutputFlag, 0);
		model.set(GRB.IntParam.Threads, numThreads);
		model.set(GRB.IntParam.NonConvex, 2);
		model.set(GRB.IntParam.OutputFlag, GUROBI_VERBOSE);
		model.set(GRB.DoubleParam.FeasibilityTol, GUROBI_FEAS_TOL);
		model.set(GRB.DoubleParam.TimeLimit, timeLimit);
		model.set(GRB.IntParam.NumericFocus, 2);
		
		//System.out.println(model.get(GRB.IntParam.Threads));

		GRBVar[][] b = new GRBVar[numPlayers][];
		GRBVar[][] p = new GRBVar[numPlayers][];
		GRBVar[][] u_strategy = new GRBVar[numPlayers][];
		GRBVar[][] r = new GRBVar[numPlayers][];
		GRBVar[] u_player = new GRBVar[numPlayers];

		GRBVar[][][] p_prod_vars = new GRBVar[numPlayers][][];
		p_prod_vars[0] = new GRBVar[numStrategies[1]][numStrategies[2]];
		p_prod_vars[1] = new GRBVar[numStrategies[0]][numStrategies[2]];
		p_prod_vars[2] = new GRBVar[numStrategies[0]][numStrategies[1]];

		double[] MAX = new double[numPlayers];
		double[] U = computeU(numPlayers, numStrategies, payoffs, MAX); // max differences in utility
		//for (int i = 0; i < U.length; ++i)
		//U[i] = 9;

		for (int i = 0; i < numPlayers; ++i) {
			b[i] = new GRBVar[numStrategies[i]];
			p[i] = new GRBVar[numStrategies[i]];
			u_strategy[i] = new GRBVar[numStrategies[i]];
			r[i] = new GRBVar[numStrategies[i]];

			u_player[i] = model.addVar(0, MAX[i], 0, GRB.CONTINUOUS, "u_player");

			for (int j = 0; j < numStrategies[i]; ++j) {
				b[i][j] = model.addVar(0, 1, 0, GRB.BINARY, "b");
				p[i][j] = model.addVar(0, 1, 0, GRB.CONTINUOUS, "p");
				u_strategy[i][j] = model.addVar(0, MAX[i], 0, GRB.CONTINUOUS, "u_strategy");
				r[i][j] = model.addVar(0, MAX[i], 0, GRB.CONTINUOUS, "r");
			}

			for (int j = 0; j < p_prod_vars[i].length; ++j) {
				for (int k = 0; k < p_prod_vars[i][j].length; ++k) {
					p_prod_vars[i][j][k] =  model.addVar(0, 1, 0, GRB.CONTINUOUS, "p_prod_vars");
				}
			}
		}

		GRBLinExpr[] sum_constraint = new GRBLinExpr[numPlayers];
		//GRBLinExpr[][] u_constraint = new GRBLinExpr[numPlayers][];
		GRBLinExpr[][] r_constraint = new GRBLinExpr[numPlayers][];
		GRBLinExpr[][] p_constraint = new GRBLinExpr[numPlayers][];
		GRBLinExpr[][] r_ub_constraint = new GRBLinExpr[numPlayers][];
		GRBLinExpr[][] p_prod_constraint = new GRBLinExpr[numPlayers][];
		
		GRBQuadExpr[][][] p_prod_vars_constraint = new GRBQuadExpr[numPlayers][][];

		for (int i = 0; i < numPlayers; ++i) {
			sum_constraint[i] = new GRBLinExpr();
			//u_constraint[i] = new GRBLinExpr[numStrategies[i]]; // Technically this is redundant, consider eliminating it.
			r_constraint[i] = new GRBLinExpr[numStrategies[i]];
			p_constraint[i] = new GRBLinExpr[numStrategies[i]];
			r_ub_constraint[i] = new GRBLinExpr[numStrategies[i]];
			p_prod_constraint[i] = new GRBLinExpr[numStrategies[i]];

			for (int j = 0; j < numStrategies[i]; ++j) {
				sum_constraint[i].addTerm(1, p[i][j]);

				/*u_constraint[i][j] = new GRBLinExpr();
				u_constraint[i][j].addTerm(1, u_player[i]);
				u_constraint[i][j].addTerm(-1, u_strategy[i][j]);
				//model.addConstr(u_constraint[i][j], GRB.GREATER_EQUAL, -EPSILON, "u_constraint " + i + " " + j); // don't need this
				model.addConstr(u_constraint[i][j], GRB.GREATER_EQUAL, 0, "u_constraint " + i + " " + j); // don't need this
				 */
				r_constraint[i][j] = new GRBLinExpr();
				r_constraint[i][j].addTerm(1, r[i][j]);
				r_constraint[i][j].addTerm(-1, u_player[i]);
				r_constraint[i][j].addTerm(1, u_strategy[i][j]);
				//model.addConstr(r_constraint[i][j], GRB.GREATER_EQUAL, -EPSILON, "r_constraint gr " + i + " " + j);
				//model.addConstr(r_constraint[i][j], GRB.LESS_EQUAL, EPSILON, "r_constraint le " + i + " " + j);
				model.addConstr(r_constraint[i][j], GRB.EQUAL, 0, "r_constraint gr " + i + " " + j);

				p_constraint[i][j] = new GRBLinExpr();
				p_constraint[i][j].addTerm(1, p[i][j]);
				p_constraint[i][j].addTerm(1, b[i][j]);
				//model.addConstr(p_constraint[i][j], GRB.LESS_EQUAL, 1+EPSILON, "p_constraint " + i + " " + j);
				model.addConstr(p_constraint[i][j], GRB.LESS_EQUAL, 1, "p_constraint " + i + " " + j);

				r_ub_constraint[i][j] = new GRBLinExpr();
				r_ub_constraint[i][j].addTerm(1, r[i][j]);
				r_ub_constraint[i][j].addTerm(-U[i], b[i][j]);
				//model.addConstr(r_ub_constraint[i][j], GRB.LESS_EQUAL, EPSILON, "r_ub_constraint " + i + " " + j);
				model.addConstr(r_ub_constraint[i][j], GRB.LESS_EQUAL, 0, "r_ub_constraint " + i + " " + j);

				p_prod_constraint[i][j] = new GRBLinExpr();
				p_prod_constraint[i][j].addTerm(1, u_strategy[i][j]);
			}
			//model.addConstr(sum_constraint[i], GRB.GREATER_EQUAL, 1-EPSILON, "sum_constraint gr " + i);
			//model.addConstr(sum_constraint[i], GRB.LESS_EQUAL, 1+EPSILON, "sum_constraint le " + i);
			model.addConstr(sum_constraint[i], GRB.EQUAL, 1, "sum_constraint le " + i);

			for (int s1 = 0; s1 < numStrategies[0]; ++s1) {
				for (int s2 = 0; s2 < numStrategies[1]; ++s2) {
					for (int s3 = 0; s3 < numStrategies[2]; ++s3) {
						int j,m,n;
						switch(i) {
						case 0: j = s1; m = s2; n = s3;
						break;
						case 1: j = s2; m = s1; n = s3;
						break;
						case 2: j = s3; m = s1; n = s2;
						break;
						default: j = -1; m = -1; n = -1;
						break;
						}
						p_prod_constraint[i][j].addTerm(-1 * payoffs[i][s1][s2][s3], p_prod_vars[i][m][n]);
					}
				}
			}
			for (int j = 0; j < numStrategies[i]; ++j) {
				//model.addQConstr(p_prod_constraint[i][j], GRB.GREATER_EQUAL, -EPSILON, "p_prod_constraint " + i + " " + j);
				//model.addQConstr(p_prod_constraint[i][j], GRB.LESS_EQUAL, EPSILON, "p_prod_constraint " + i + " " + j);
				model.addConstr(p_prod_constraint[i][j], GRB.EQUAL, 0, "p_prod_constraint " + i + " " + j);
			}
		}

		p_prod_vars_constraint[0] = new GRBQuadExpr[numStrategies[1]][numStrategies[2]];
		p_prod_vars_constraint[1] = new GRBQuadExpr[numStrategies[0]][numStrategies[2]];
		p_prod_vars_constraint[2] = new GRBQuadExpr[numStrategies[0]][numStrategies[1]];
		
		for (int s2 = 0; s2 < numStrategies[1]; ++s2) {
			for (int s3 = 0; s3 < numStrategies[2]; ++s3) {
				p_prod_vars_constraint[0][s2][s3] = new GRBQuadExpr();
				p_prod_vars_constraint[0][s2][s3].addTerm(1, p_prod_vars[0][s2][s3]);
				p_prod_vars_constraint[0][s2][s3].addTerm(-1, p[1][s2], p[2][s3]);
				model.addQConstr(p_prod_vars_constraint[0][s2][s3], GRB.EQUAL, 0, "p_prod_vars_constraint " + 0 + " " + s2 + " " + s3);
			}
		}
		
		for (int s1 = 0; s1 < numStrategies[0]; ++s1) {
			for (int s3 = 0; s3 < numStrategies[2]; ++s3) {
				p_prod_vars_constraint[1][s1][s3] = new GRBQuadExpr();
				p_prod_vars_constraint[1][s1][s3].addTerm(1, p_prod_vars[1][s1][s3]);
				p_prod_vars_constraint[1][s1][s3].addTerm(-1, p[0][s1], p[2][s3]);
				model.addQConstr(p_prod_vars_constraint[1][s1][s3], GRB.EQUAL, 0, "p_prod_vars_constraint " + 1 + " " + s1 + " " + s3);
			}
		}
		
		for (int s1 = 0; s1 < numStrategies[0]; ++s1) {
			for (int s2 = 0; s2 < numStrategies[1]; ++s2) {
				p_prod_vars_constraint[2][s1][s2] = new GRBQuadExpr();
				p_prod_vars_constraint[2][s1][s2].addTerm(1, p_prod_vars[2][s1][s2]);
				p_prod_vars_constraint[2][s1][s2].addTerm(-1, p[0][s1], p[1][s2]);
				model.addQConstr(p_prod_vars_constraint[2][s1][s2], GRB.EQUAL, 0, "p_prod_vars_constraint " + 2 + " " + s1 + " " + s2);
			}
		}

		model.optimize();

		//System.out.println("RUN TIME: "
			//	+ model.get(GRB.DoubleAttr.Runtime));
		
		//double nextTime = model.get(GRB.DoubleAttr.Runtime);
		//System.out.println("TIME LIMIT: "
		//		+ model.getEnv().get(GRB.DoubleParam.TimeLimit));

		int optimStatus = model.get(GRB.IntAttr.Status);
		
		runningTime = model.get(GRB.DoubleAttr.Runtime);

		//System.out.println("TIME limit status: "
		//		+ GRB.Status.TIME_LIMIT);
		//System.out.println("Optim status: " + optimStatus);

		/*
		if (optimStatus == GRB.Status.INF_OR_UNBD) {
			System.out.println("STATUS = INF OR UNB!");
		}
		if (optimStatus == GRB.Status.OPTIMAL) {
			System.out.println("STATUS = OPTIMAL");
		}
		if (optimStatus == GRB.Status.INFEASIBLE) {
			System.out.println("STATUS = INFEASIBLE :(");

			model.computeIIS();
			System.out.println("\nThe following constraint(s) "
					+ "cannot be satisfied:");

			for (GRBConstr c : model.getConstrs()) {
				if (c.get(GRB.IntAttr.IISConstr) == 1) {
					System.out.println(c.get(GRB.StringAttr.ConstrName));
				}
			}

			model.dispose();
			env.dispose();

			return optimStatus;
		}
		*/
		
		if (optimStatus == GRB.Status.OPTIMAL) {
			for (int i = 0; i < numPlayers; ++i) {
				//System.out.print("PLAYER " + i + ": ");
				for (int j = 0; j < numStrategies[i]; ++j) {
					strategies[i][j] = p[i][j].get(GRB.DoubleAttr.X);
					//System.out.println(strategies[i][j]);
				}
			}
		}

		model.dispose();
		env.dispose();

		return optimStatus;
	}

	public double[] computeU(int numPlayers, int[] numStrategies, double[][][][] payoffs, double[] m) {
		double[] U = new double[numPlayers];

		for (int player = 0; player < numPlayers; ++player) {
			double max = -Double.MAX_VALUE;
			double min = Double.MAX_VALUE;
			for (int i = 0; i < numStrategies[0]; ++i) {
				for (int j = 0; j < numStrategies[1]; ++j) {
					for (int k = 0; k < numStrategies[2]; ++k) {
						double next = payoffs[player][i][j][k];
						if (next > max) 
							max = next;
						if (next < min)
							min = next;
					}
				}
			}
			U[player] = max - min;
			m[player] = max;
			//System.out.println("U: " + U[player]);
		}
		return U;
	}

	/*
	private double[] computeNeValues(double[] str1, double[] str2, double[] str3, double[][][][] payoffs) {
		double[] result = new double[NUM_PLAYERS];
		for (int i = 0; i < NUM_PLAYERS; ++i)
			result[i] = computePayoff(i, str1, str2, str3, payoffs);
		return result;
	}

	private double computePayoff(int player, double[] str1, double[] str2, double[] str3, double[][][][] payoffs) {
		double result = 0;
		for (int i = 0; i < str1.length; ++i) {
			for (int j = 0; j < str2.length; ++j) {
				for (int k = 0; k < str3.length; ++k) {
					result += str1[i] * str2[j]*str3[k]*payoffs[player][i][j][k];
				}
			}
		}
		return result;
	}

	private int computeBestResponse(int player, int numStrategies, double[] opp1Str, double[] opp2Str, double[][][][] payoffs) {
		double[][] oppProds = new double[opp1Str.length][opp2Str.length];
		for (int i = 0; i < opp1Str.length; ++i) {
			for (int j = 0; j < opp2Str.length; ++j) {
				oppProds[i][j] = opp1Str[i] * opp2Str[j];
			}
		}

		double br_payoff = (-1)*Double.MAX_VALUE;
		int br = -1;
		for (int i = 0; i < numStrategies; ++i) {
			double payoff = 0;
			for (int j = 0; j < oppProds.length; ++j) {
				for (int k = 0; k < oppProds[j].length; ++k) {
					int index1, index2, index3;
					switch(player) {
					case 0: 
						index1 = i;
						index2 = j;
						index3 = k;
						break;
					case 1:
						index1 = j;
						index2 = i;
						index3 = k;
						break;
					case 2:
						index1 = j;
						index2 = k;
						index3 = i;
						break;
					default:
						index1 = -1;
						index2 = -1;
						index3 = -1;
						break;
					}
					payoff += oppProds[j][k] * payoffs[player][index1][index2][index3];
				}
			}
			if (payoff > br_payoff) {
				br_payoff = payoff;
				br = i;
			}
		}
		brValues[player] = br_payoff;
		return br; 
	}

	private double[] computeEpsilons(double[] br, double[] ne) {
		double[] result = new double[NUM_PLAYERS];

		for (int i = 0; i < NUM_PLAYERS; ++i)
			result[i] = br[i] - ne[i];

		return result;
	}
	*/
}
