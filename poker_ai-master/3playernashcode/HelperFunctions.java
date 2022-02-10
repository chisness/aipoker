package general;

import java.util.Arrays;
import java.util.Random;

public class HelperFunctions {

	public static void normalize(double[][] d) {
		double max = -Double.MAX_VALUE;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < d.length; ++i) {
			for (int j = 0; j < d[i].length; ++j) {
				if (d[i][j] > max) 
					max = d[i][j];
				if (d[i][j] < min)
					min = d[i][j];
			}
		}
		if (min < 0) {
			max = max - min;
			for (int i = 0; i < d.length; ++i) {
				for (int j = 0; j < d[i].length; ++j) {
					d[i][j] -= min;
					d[i][j] /= max;
				}
			}
		}
		else {
			for (int i = 0; i < d.length; ++i) {
				for (int j = 0; j < d[i].length; ++j) 
					d[i][j] /= max;
			}
		}
	}

	public static void normalize(double[][][][] d) {
		double max = -Double.MAX_VALUE;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < d.length; ++i) {
			for (int j = 0; j < d[i].length; ++j) {
				for (int k = 0; k < d[i][j].length; ++k) {
					for (int m = 0; m < d[i][j][k].length; ++m) {
						if (d[i][j][k][m] > max) 
							max = d[i][j][k][m];
						if (d[i][j][k][m] < min)
							min = d[i][j][k][m];
					}
				}
			}
		}
		//System.out.println(min + " " + max);
		if (min < 0) {
			max = max - min;
			for (int i = 0; i < d.length; ++i) {
				for (int j = 0; j < d[i].length; ++j) {
					for (int k = 0; k < d[i][j].length; ++k) {
						for (int m = 0; m < d[i][j][k].length; ++m) {
							d[i][j][k][m] -= min;
							d[i][j][k][m] /= max;
						}
					}
				}
			}
		}
		else {
			for (int i = 0; i < d.length; ++i) {
				for (int j = 0; j < d[i].length; ++j) {
					for (int k = 0; k < d[i][j].length; ++k) {
						for (int m = 0; m < d[i][j][k].length; ++m) {
							d[i][j][k][m] /= max;
						}
					}
				}
			}
		}
	}

	public static void normalize(double[][][][][] d) {
		double max = -Double.MAX_VALUE;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < d.length; ++i) {
			for (int j = 0; j < d[i].length; ++j) {
				for (int k = 0; k < d[i][j].length; ++k) {
					for (int m = 0; m < d[i][j][k].length; ++m) {
						for (int n = 0; n < d[i][j][k][m].length; ++n) {
							if (d[i][j][k][m][n] > max) 
								max = d[i][j][k][m][n];
							if (d[i][j][k][m][n] < min)
								min = d[i][j][k][m][n];
						}
					}
				}
			}
		}
		if (min < 0) {
			max = max - min;
			for (int i = 0; i < d.length; ++i) {
				for (int j = 0; j < d[i].length; ++j) {
					for (int k = 0; k < d[i][j].length; ++k) {
						for (int m = 0; m < d[i][j][k].length; ++m) {
							for (int n = 0; n < d[i][j][k][m].length; ++n) {
								d[i][j][k][m][n] -= min;
								d[i][j][k][m][n] /= max;
							}
						}
					}
				}
			}
		}
		else {
			for (int i = 0; i < d.length; ++i) {
				for (int j = 0; j < d[i].length; ++j) {
					for (int k = 0; k < d[i][j].length; ++k) {
						for (int m = 0; m < d[i][j][k].length; ++m) {
							for (int n = 0; n < d[i][j][k][m].length; ++n) {
								d[i][j][k][m][n] /= max;
							}
						}
					}
				}
			}
		}
	}

	public static void normalize(double[][][][][][] d) {
		double max = -Double.MAX_VALUE;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < d.length; ++i) {
			for (int j = 0; j < d[i].length; ++j) {
				for (int k = 0; k < d[i][j].length; ++k) {
					for (int m = 0; m < d[i][j][k].length; ++m) {
						for (int n = 0; n < d[i][j][k][m].length; ++n) {
							for (int o = 0; o < d[i][j][k][m][n].length; ++o) {
								if (d[i][j][k][m][n][o] > max) 
									max = d[i][j][k][m][n][o];
								if (d[i][j][k][m][n][o] < min)
									min = d[i][j][k][m][n][o];
							}
						}
					}
				}
			}
		}
		if (min < 0) {
			max = max - min;
			for (int i = 0; i < d.length; ++i) {
				for (int j = 0; j < d[i].length; ++j) {
					for (int k = 0; k < d[i][j].length; ++k) {
						for (int m = 0; m < d[i][j][k].length; ++m) {
							for (int n = 0; n < d[i][j][k][m].length; ++n) {
								for (int o = 0; o < d[i][j][k][m][n].length; ++o) {
									d[i][j][k][m][n][o] -= min;
									d[i][j][k][m][n][o] /= max;
								}
							}
						}
					}
				}
			}
		}
		else {
			for (int i = 0; i < d.length; ++i) {
				for (int j = 0; j < d[i].length; ++j) {
					for (int k = 0; k < d[i][j].length; ++k) {
						for (int m = 0; m < d[i][j][k].length; ++m) {
							for (int n = 0; n < d[i][j][k][m].length; ++n) {
								for (int o = 0; o < d[i][j][k][m][n].length; ++o) {
									d[i][j][k][m][n][o] /= max;
								}
							}
						}
					}
				}
			}
		}
	}

	public static int Pure_ESSPM(double[][] payoffs) {
		int numStrategies = payoffs.length;
		for (int i = 0; i < numStrategies; ++i) {
			boolean valid = true;
			for (int j = 0; j < numStrategies; ++j) {
				if (j == i)
					continue;
				if (payoffs[j][i] < payoffs[i][i])
					continue;
				if ((payoffs[j][i] == payoffs[i][i]) && payoffs[j][j] < payoffs[i][j])
					continue;
				valid = false;
				break;
			}
			if (valid)
				return i;
		}
		return -1;
	}

	public static int getSupportSize(double[] x) {
		int result = 0;
		for (int i = 0; i < x.length; ++i)
			if (x[i] != 0)
				++result;
		return result;
	}

	public static double randDouble(double min, double max, Random r) {
		double diff = max - min;
		return (diff * r.nextDouble() + min);
	}

	public static double max(double[] arr) {
		double max = -Double.MAX_VALUE;
		for (int i = 0; i < arr.length; ++i) {
			if (arr[i] > max)
				max = arr[i];
		}
		return max;
	}

	public static double computeStdError(double[] values, double average) {
		double stdDev = 0;
		for (int i = 0; i < values.length; ++i) 
			stdDev += Math.pow(values[i] - average, 2);
		stdDev /= values.length;
		stdDev = Math.sqrt(stdDev);
		double stdErr = stdDev / Math.sqrt(values.length);
		return stdErr;
	}

	public static double computeStdError(double[] values, double average, int num) {
		double stdDev = 0;
		for (int i = 0; i < num; ++i) 
			stdDev += Math.pow(values[i] - average, 2);
		stdDev /= num;
		stdDev = Math.sqrt(stdDev);
		double stdErr = stdDev / Math.sqrt(num);
		return stdErr;
	}

	public static double findMedian(double a[], int n) { 
		// First we sort the array 
		Arrays.sort(a); 

		// check for even case 
		if (n % 2 != 0) 
			return (double)a[n / 2]; 

		return (double)(a[(n - 1) / 2] + a[n / 2]) / 2.0; 
	} 
}
