import java.util.Arrays;
import java.util.Random;
import java.util.TreeMap;

public class KTCFRChanceFBRFIX {
    public long nodecount = 0;
    public int brcount = 0;
    public double[][] brex = new double[4][200];
    public int itf = 0;
    public static final int PASS = 0, BET = 1, NUM_ACTIONS = 2;
    public static final Random random = new Random();
    public TreeMap<String, Node> nodeMap = new TreeMap<String, Node>();

    class Node {
        String infoSet;
        double[] regretSum = new double[NUM_ACTIONS], 
                 strategy = new double[NUM_ACTIONS], 
                 strategySum = new double[NUM_ACTIONS];
        

        private double[] getStrategy(double realizationWeight) {
            double normalizingSum = 0;
            for (int a = 0; a < NUM_ACTIONS; a++) {
                strategy[a] = regretSum[a] > 0 ? regretSum[a] : 0;
                normalizingSum += strategy[a];
            }
            for (int a = 0; a < NUM_ACTIONS; a++) {
                if (normalizingSum > 0)
                    strategy[a] /= normalizingSum;
                else
                    strategy[a] = 1.0 / NUM_ACTIONS;
                strategySum[a] += realizationWeight * strategy[a];
            }
            return strategy;
        }
        

        public double[] getAverageStrategy() {
            double[] avgStrategy = new double[NUM_ACTIONS];
            double normalizingSum = 0;
            for (int a = 0; a < NUM_ACTIONS; a++)
                normalizingSum += strategySum[a];
            for (int a = 0; a < NUM_ACTIONS; a++) 
                if (normalizingSum > 0)
                    avgStrategy[a] = strategySum[a] / normalizingSum;
                else
                    avgStrategy[a] = 1.0 / NUM_ACTIONS;
            return avgStrategy;
        }
        

        public String toString() {
                return String.format("%4s: %s", infoSet, Arrays.toString(getAverageStrategy()));
        }

    }
    

    public void train(int iterations, int decksize, int buckets) {
        int[] cards = new int[decksize];
        //int[] card = null;
        long starttime = System.currentTimeMillis();
        for (int i = 0; i < decksize; i++) {
          cards[i]=i;
        }
        double util1 = 0;
        for (int i = 0; i <= iterations; i++) {
            for (int c1 = cards.length - 1; c1 > 0; c1--) { 
                int c2 = random.nextInt(c1 + 1);
                int tmp = cards[c1];
                cards[c1] = cards[c2];
                cards[c2] = tmp;
                
            }
            util1 += cfr(cards, "", 1, 1, decksize, starttime, i, buckets, iterations);
        }
        System.out.println("Average game value: " + util1 / iterations);
        long elapsedtime1 = System.currentTimeMillis() - starttime;
        System.out.println("Total time: " + elapsedtime1);
        
        for (int i = 0; i < brcount; i++) {
          System.out.println("Exploitability: " + brex[0][i] + " at nodecount: " + brex[1][i] + " at time: " + brex[2][i] + " at iteration: " + brex[3][i]);
        }
        
        for (Node n : nodeMap.values())
            System.out.println(n);
    }
    

    private double cfr(int[] cards, String history, double p0, double p1, int decksize, long starttime, int currit, int buckets, int iterations) {
              //System.out.println(cards[0]);  
              //System.out.println(cards[1]); 
              //System.out.println(cards[2]); 
        int plays = history.length();
        int player = plays % 2;
        int opponent = 1 - player;
        if (plays > 1) {
            boolean terminalPass = history.charAt(plays - 1) == 'p';
            boolean doubleBet = history.substring(plays - 2, plays).equals("bb");
            boolean isPlayerCardHigher = cards[player] > cards[opponent];
            if (terminalPass)
                if (history.equals("pp"))
                    return isPlayerCardHigher ? 1 : -1;
                else
                    return 1;
            else if (doubleBet)
                return isPlayerCardHigher ? 2 : -2;
        }  
        
        String infoSet = history;
        
        if (buckets > 0) {
          int bucket = 0;
          //for (int i = buckets; i => 1; i--){
          for (int i = 0; i < buckets; i++){
            if (cards[player] < (decksize/buckets)*(i+1)) {
               bucket = i;
               break;
               }
          }
          infoSet = bucket + history;
        }
        else {
         infoSet = cards[player] + history; 
        }
        //System.out.println(infoSet);
        
        nodecount = nodecount + 1;
        if (currit == iterations) {
          itf=itf+1;
        }
    if ((nodecount % 1000000) == 0)
      System.out.println("nodecount: " + nodecount);
    if ((itf==1) || (nodecount == 32) || (nodecount == 64) || (nodecount == 128) || (nodecount == 256) || (nodecount == 512) || (nodecount == 1024) || (nodecount == 2048) || (nodecount == 4096) || (nodecount == 8192) || (nodecount == 16384) || (nodecount == 32768) || (nodecount == 65536) || (nodecount == 131072) || (nodecount == 262144) || (nodecount == 524288) || (nodecount == 1048576) || (nodecount == 2097152) || (nodecount == 4194304) || (nodecount == 8388608) || (nodecount == 16777216) || (nodecount == 33554432) || (nodecount == 67108864) || (nodecount == 134217728) || (nodecount == 268435456)  || (nodecount == 536870912) || (nodecount == 1073741824) || (nodecount == 2147483648l) || (nodecount == 4294967296l) || (nodecount == 8589934592l) || (nodecount == 17179869184l) || (nodecount == 100000) || (nodecount == 1000000) || (nodecount == 10000000) || (nodecount == 100000000) || (nodecount == 1000000000)  || (nodecount % 100000000)==0) {//|| (nodecount == 10000000000)) {
        double[] oppreach = new double[decksize];
         double br0 = 0;
         double br1 = 0;
        
         
         for (int c=0; c < decksize; c++) {
           for (int j = 0; j < decksize; j++) {
             if (c==j)
               oppreach[j] = 0;
             else
               oppreach[j] = 1./(oppreach.length-1);
           }
           System.out.println("br iter: " + brf(c, "", 0, oppreach, buckets)); 
           br0 += brf(c, "", 0, oppreach, buckets);
         }
  
         for (int c=0; c < decksize; c++) {
           for (int j = 0; j < decksize; j++) {
             if (c==j)
               oppreach[j] = 0;
             else
               oppreach[j] = 1./(oppreach.length-1);
           }
           //System.out.println("br iter: " + brf(c, "", 1, oppreach));
           br1 += brf(c, "", 1, oppreach, buckets);
         }
         
         long elapsedtime = System.currentTimeMillis() - starttime;
         System.out.println("br0 " + br0);
         System.out.println("br1 " + br1);
         //System.out.println("Average game value: " + util0 / currit); //empirical, should also get game value based on average strategy expected value
         System.out.println("Exploitability: " + (br0+br1)/(2));
         System.out.println("Number of nodes touched: " + nodecount);
         System.out.println("Time elapsed in milliseconds: " + elapsedtime);
         System.out.println("Iterations: " + currit);
         //System.out.println("Iteration player: " + player_iteration);
         brex[0][brcount] = (br0+br1)/(2);
           brex[1][brcount] = nodecount;
           brex[2][brcount] = elapsedtime;
           brex[3][brcount] = currit;
         brcount = brcount + 1;
        }
        
        Node node = nodeMap.get(infoSet);
        if (node == null) {
            node = new Node();
            node.infoSet = infoSet;
            nodeMap.put(infoSet, node);
        }

        double[] strategy = node.getStrategy(player == 0 ? p0 : p1);
        double[] util = new double[NUM_ACTIONS];
        double nodeUtil = 0;
        
        for (int a = 0; a < NUM_ACTIONS; a++) {
            String nextHistory = history + (a == 0 ? "p" : "b");
            util[a] = player == 0 
                ? - cfr(cards, nextHistory, p0 * strategy[a], p1, decksize, starttime, currit, buckets, iterations)
                : - cfr(cards, nextHistory, p0, p1 * strategy[a], decksize, starttime, currit, buckets, iterations);
            nodeUtil += strategy[a] * util[a];
        }

        for (int a = 0; a < NUM_ACTIONS; a++) {
            double regret = util[a] - nodeUtil;
            node.regretSum[a] += (player == 0 ? p1 : p0) * regret;
        }

        return nodeUtil;
    }
    
    private double brf(int player_card, String history, int player_iteration, double[] oppreach, int buckets)
    {
      // System.out.println("oppreach_toploop: " + oppreach[0] + " " + oppreach[1] + " " + oppreach[2]);
  
      // same as in CFR, these evaluate how many plays and whose turn it is
      // player is whose turn it is to act at the current action
      // we know player based on history.length() since play switches after each action
      int plays = history.length();
      int player = plays % 2;
  
      // check for terminal pass
      // possible sequences in kuhn poker: 
      // pp (terminalpass), bb (doublebet), bp (terminalpass), pbp (terminalpass), pbb (doublebet)
      if (plays > 1) {
        double exppayoff = 0;
        boolean terminalPass = history.charAt(plays - 1) == 'p'; //check for last action being a pass
        boolean doubleBet = history.substring(plays - 2, plays).equals("bb");
        if (terminalPass || doubleBet) { //hand is terminal
          // System.out.println("opp reach: " + oppreach[0] + " " + oppreach[1] + " " + oppreach[2]); 
          // oppdist = normalize(oppreach)
          double[] oppdist = new double[oppreach.length];
          double oppdisttotal = 0;
          for (int i = 0; i < oppreach.length; i++) {
            oppdisttotal += oppreach[i]; //compute sum of distribution for normalizing later
          }
      /*if (terminalPass)
            System.out.println("terminal pass history: " + history);
        if (doubleBet)
            System.out.println("terminal doublebet history: " + history); */
          for (int i = 0; i < oppreach.length; i++) { //entire opponent distribution
            oppdist[i] = oppreach[i]/oppdisttotal; //normalize opponent distribution
            double payoff = 0;
            boolean isPlayerCardHigher = player_card > i;
            // System.out.println("opponent dist pre normalized: " + oppdist[i] + " for card: " + i + " (main card: " + ci + ")");
            // System.out.println("current player: " + player);
            // System.out.println("main player: " + player_iteration);
            if (terminalPass) {
              if (history.equals("pp")) {
                //if (player == player_iteration)
                  payoff = isPlayerCardHigher ? 1 : -1;
                //else
                  //payoff = isPlayerCardHigher ? -1 : 1;
              }
              else {
                if (player == player_iteration)
                  payoff = 1;
                else
                  payoff = -1;
              }
              //else {
                //payoff = 1;
               /* if (player == player_iteration)
                  return 1;
                else
                  return -1;*/
              }
            //}
            else if (doubleBet) {
              
              //if (player == player_iteration)
                payoff = isPlayerCardHigher ? 2 : -2;
              //else
                //payoff = isPlayerCardHigher ? -2 : 2;
            }      
            exppayoff += oppdist[i]*payoff; //adding weighted payoffs
            //  }
          }
          //System.out.println("exppayoff: " + exppayoff);
          return exppayoff;
        }
      }
  
      /*
       if (plays==0 && player == player_iteration) { //chance node main (i) player
       //System.out.println("CHANCE NODE PLAYER i");
       double brv = 0;
       for (int a = 0; a < NUM_ACTIONS; a++) {
       String nextHistory = history + (a == 0 ? "p" : "b");
       brv += brf(player_card, nextHistory, player_iteration, oppreach);
       }
       return brv; 
       }
  
  if (plays==0 && player != player_iteration) { //chance node opponent (-i) player
  //System.out.println("CHANCE NODE PLAYER -i");
  String dummyHistory = history + "p";
  return brf(player_card, dummyHistory, player_iteration, oppreach); //give opponent player dummy card of 1 that is never used
  }*/
      //System.out.println("beginning of br iteration, player: " + player);
      //System.out.println("beg of iteration oppreach: " + oppreach[0] + " " + oppreach[1] + " " + oppreach[2]);
      double[] d = new double[NUM_ACTIONS];  //opponent action dist
      d[0] = 0;
      d[1] = 0;
      //double[] new_oppreach = new double[oppreach.length];
  
      double[] new_oppreach = new double[oppreach.length]; //new opponent card distribution
      for (int i = 0; i < oppreach.length; i++) {
        new_oppreach[i] = oppreach[i]; 
      }
      //System.out.println("new_oppreach_after_define: " + new_oppreach[0] + " " + new_oppreach[1] + " " + new_oppreach[2]);
  
      double v = -100000; //initialize node utility
      double[] util = new double[NUM_ACTIONS]; //initialize util value for each action
      util[0] = 0; 
      util[1] = 0;
      double[] w = new double[NUM_ACTIONS]; //initialize weights for each action
      w[0] = 0;
      w[1] = 0;
      String infoSet = history;
      for (int a = 0; a < NUM_ACTIONS; a++) { 
        //System.out.println("in loop action: " + a + ", oppreach: " + oppreach[0] + " " + oppreach[1] + " " + oppreach[2]);
        if (player != player_iteration) {
          //System.out.println("REGULAR NODE PLAYER -i");
          for (int i = 0; i < oppreach.length; i++) {
            if (buckets > 0) {
              int bucket1 = 0;
              //for (int j = buckets; j => 1; j--) {
              for (int j = 0; j < buckets; j++) {
                if (i < (oppreach.length/buckets)*(j+1)) {
                  bucket1 = j;
                  break;
                }
              }
            infoSet = bucket1 + history;
            }
            //System.out.println("oppreach: " + i + " " + oppreach[i]);
            //System.out.println("oppreach: " + oppreach.length);
            else {
            infoSet = i + history; //read info set, which is hand + play history
            }
            //System.out.println("infoset: " + infoSet);
            //for (Node n : nodeMap.values())
            //System.out.println(n);
            Node node = nodeMap.get(infoSet);
            if (node == null) {
             node = new Node();
             node.infoSet = infoSet;
             nodeMap.put(infoSet, node);
             System.out.println("infoset: " + infoSet);
             }
          
            double[] strategy = node.getAverageStrategy(); //read strategy (same as probability)
            //System.out.println("oppreach: " + oppreach[i]);
            new_oppreach[i] = oppreach[i]*strategy[a]; //update reach probability
            //System.out.println("after newoppreach, original: " + oppreach[0] + " " + oppreach[1] + " " + oppreach[2]);
            //System.out.println("strategy[a]: " + strategy[0] + "  strategy[b] :" + strategy[1]);
            w[a] += new_oppreach[i]; //sum weights over all possibilities of the new reach
            //System.out.println("getting strategy and weight: " + w[a]);
          }
      
        }
        //System.out.println("before brf call oppreach: " + oppreach[0] + " " + oppreach[1] + " " + oppreach[2]);
        String nextHistory = history + (a == 0 ? "p" : "b"); 
        //System.out.println("new_oppreach: " + new_oppreach[0] + " " + new_oppreach[1] + " " + new_oppreach[2]);
        util[a] = brf(player_card, nextHistory, player_iteration, new_oppreach, buckets); //recurse for each action
        if (player == player_iteration && util[a] > v) {
          v = util[a]; //this action is better than previously best action
        }
      }
  
      if (player != player_iteration) {
        // D_(-i) = Normalize(w)
        // d is action distribution that = normalized w
        // System.out.println("weight 0: " + w[0]);
        // System.out.println("weight 1: " + w[1]);
        d[0] = w[0]/(w[0]+w[1]);
        d[1] = w[1]/(w[0]+w[1]);
        v = d[0]*util[0] + d[1]*util[1];
      }
      return v;
  
    }
    

    public static void main(String[] args) {
        int iterations = 1000000000;
        int decksize = 100;
        int buckets = 25; //options: 0, 3, 10, 25
        new KTCFRChanceFBRFIX().train(iterations, decksize, buckets);
    }

}