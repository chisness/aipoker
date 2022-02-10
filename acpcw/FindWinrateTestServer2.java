package playingAgent.computeWinrates;
import java.io.*;
import java.util.*;

public class FindWinrateTestServer2 {

	static String MATCH_NAME_FWD;
	static String MATCH_NAME_REV;
	static String BOT1;
	static String BOT2;
	static String DIRECTORY;
	static int NUM_MATCHES;
	static int NUM_HANDS;
	static int BB_SIZE;

	// Computes payoff to BOT 1 of next line
	static double compute_next_payoff(StringTokenizer str) {
		String line, first_bot;
		int index, payoff_first_index, name_first_index;
		double temp;

		line = str.nextToken();
		index = 0;

		for (int k = 0; k < 4; ++k) {
			while (line.charAt(index) != ':') 
				++index;
			++index;
		}
		payoff_first_index = index;

		while (line.charAt(index) != '|') 
			++index;
		temp = Double.parseDouble(line.substring(payoff_first_index,index)); 

		while (line.charAt(index) != ':') 
			++index;
		++index;
		name_first_index = index;
		while (line.charAt(index) != '|') 
			++index;
		first_bot = line.substring(name_first_index,index);

		if (first_bot.equals(BOT1))
			return temp;
		else if (first_bot.equals(BOT2)) 
			return (-1 * temp);
		else {
			System.out.println("ERROR: INVALID BOT NAME ENTERED");
			System.out.println(first_bot);
			System.exit(0);
			return -1;
		}
	}

	public static void main(String[] args) throws IOException {

		MATCH_NAME_FWD = args[0];
		MATCH_NAME_REV = args[1];
		BOT1 = args[2];
		BOT2 = args[3];
		DIRECTORY = args[4];
		NUM_MATCHES = Integer.parseInt(args[5]);
		NUM_HANDS = Integer.parseInt(args[6]);
		BB_SIZE = Integer.parseInt(args[7]);
		
		String FILE_FWD, FILE_REV;
		BufferedReader br_fwd, br_rev;
		StringTokenizer str_fwd, str_rev;
		int array_index;

		int num_bot_1_wins = 0;
		double match_payoff;

		double current_profit = 0;
		double total_profit = 0;

		double [] payoffs = new double[NUM_MATCHES*NUM_HANDS];

		double fwd_payoff, rev_payoff;

		for (int i = 0; i < NUM_MATCHES; ++i) {
			match_payoff = 0;
			FILE_FWD = DIRECTORY + MATCH_NAME_FWD + "_" + i + ".log";
			//FILE_FWD = DIRECTORY + MATCH_NAME + "_" + (i+1) + ".log";
			br_fwd = new BufferedReader(new FileReader(FILE_FWD));
			for (int j = 0; j < 4; ++j)
				br_fwd.readLine();

			FILE_REV = DIRECTORY + MATCH_NAME_REV + "_" + i + ".log";
			//FILE_REV = DIRECTORY + MATCH_NAME + "_" + (i+1) + "_reverse.log";
			br_rev = new BufferedReader(new FileReader(FILE_REV));
			for (int j = 0; j < 4; ++j)
				br_rev.readLine();

			for (int j = 0; j < NUM_HANDS; ++j) {
				str_fwd = new StringTokenizer(br_fwd.readLine());
				fwd_payoff = compute_next_payoff(str_fwd);
				str_rev = new StringTokenizer(br_rev.readLine());
				rev_payoff = compute_next_payoff(str_rev);

				current_profit = fwd_payoff + rev_payoff;
				match_payoff += current_profit;

				// now aggregate fwd and rev
				current_profit /= (2.0 * BB_SIZE);
				array_index = i*NUM_HANDS + j;
				payoffs[array_index] = current_profit;
				total_profit += current_profit; 
			}
			br_fwd.close();
			br_rev.close();

			if (match_payoff >= 0)
				++num_bot_1_wins;
		}

		double winrate = total_profit / (NUM_MATCHES * NUM_HANDS);

		double std_dev = 0;
		for (int i = 0; i < payoffs.length; ++i) 
			std_dev += Math.pow(payoffs[i] - winrate, 2);
		std_dev /= payoffs.length;
		std_dev = Math.sqrt(std_dev);
		double std_err = std_dev / Math.sqrt(NUM_MATCHES * NUM_HANDS);

		System.out.println(BOT1 + " won at a rate of " + winrate + " big blinds per hand.");
		System.out.println("The standard error was " + std_err + ".");
		System.out.println(BOT1 + " won " + num_bot_1_wins + " matches.");
	}
}

