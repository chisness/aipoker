package ca.ualberta.cs.poker.free.graph;

import java.util.Vector;
import java.util.Random;

public class BipartiteGraph<A,B>{
  TestConnection<A,B> connect;
  Vector<A> partA;
  Vector<B> partB;
  boolean[] AReachable;
  boolean[] BReachable;
  int[] currentAtoBMatching;
  int[] currentBtoAMatching;
  public BipartiteGraph(TestConnection<A,B> connect,
    Vector<A> partA, Vector<B> partB){
    this.connect = connect;
    this.partA = partA;
    this.partB = partB;
  }

  public void setMatch(int aIndex, int bIndex){
    //System.out.println("setMatch("+aIndex+","+bIndex +")");
    currentAtoBMatching[aIndex]=bIndex;
    currentBtoAMatching[bIndex]=aIndex;
  }

  public void initMatchingFields(){
    currentAtoBMatching = new int[partA.size()];
    currentBtoAMatching = new int[partB.size()];
    for(int i=0;i<currentAtoBMatching.length;i++){
      currentAtoBMatching[i]=-1;
    }
    for(int i=0;i<currentBtoAMatching.length;i++){
      currentBtoAMatching[i]=-1;
    }
    initReachable();
  }

  public void initReachable(){
    AReachable = new boolean[partA.size()];
    BReachable = new boolean[partB.size()];
    
    for(int i=0;i<currentAtoBMatching.length;i++){
      AReachable[i]=(currentAtoBMatching[i]==-1);
    }
    for(int j=0;j<currentBtoAMatching.length;j++){
      BReachable[j]=false;
    }
  }
  public boolean canConnect(int aIndex, int bIndex){
      return connect.canConnect(partA.get(aIndex),partB.get(bIndex));
  }

  /**
   * Finds an augmenting path and applies it to the current matching.
   */
  public boolean findAugmentingPath(int aIndexOrigin){
    //System.out.println("findAugmentingPath("+aIndexOrigin+")");
    for(int bIndex=0;bIndex<currentBtoAMatching.length;bIndex++){
      if (BReachable[bIndex]){
        continue;
      }
      //System.out.println("Is there a path from a"+aIndexOrigin+" to b"+bIndex);
      if (canConnect(aIndexOrigin,bIndex)){
        BReachable[bIndex]=true;
        //System.out.println("Looking at a path from a"+aIndexOrigin+" to b"+bIndex);
        if (currentBtoAMatching[bIndex]==-1){
	  setMatch(aIndexOrigin,bIndex);
	  return true;
        }
	int aIndex = currentBtoAMatching[bIndex];
	//System.out.println("Looking to go back from b"+bIndex+ " to a"+aIndex);
	if (AReachable[aIndex]){
	  //System.out.println("a"+aIndex+" is already reachable.");
	  //System.out.println("That's funny!");
	  return false;
	}
	AReachable[aIndex]=true;
	if (findAugmentingPath(aIndex)){
	  setMatch(aIndexOrigin,bIndex);
	  return true;
        }
      }
    }
    return false;
  }

  public Vector<B> getMatching(){
    initMatchingFields();
    boolean foundAugmentingPath = false;
    do{
      foundAugmentingPath = false;
      initReachable();
      for(int aIndex=0;aIndex<currentAtoBMatching.length;aIndex++){
        if (currentAtoBMatching[aIndex]!=-1){
	  continue;
	}

        foundAugmentingPath = findAugmentingPath(aIndex);
	if (foundAugmentingPath){
	  break;
	}
      }
    } while(foundAugmentingPath);
    Vector<B> result = new Vector<B>(currentAtoBMatching.length);
    for(int i=0;i<currentAtoBMatching.length;i++){
      if (currentAtoBMatching[i]!=-1){
        result.add(partB.get(currentAtoBMatching[i]));
      } else {
        result.add(null);
      }
    }
    return result;
  }


  static class MatrixTest implements TestConnection<Integer,Integer>{
    boolean[][] adjacent;
    public MatrixTest(boolean [][]adjacent){
      this.adjacent = adjacent;
    }
    public boolean canConnect(Integer a, Integer b){
      return adjacent[a][b];
    }
  }
    

  public static Vector<Integer> testMatrix(boolean[][] adjacent){
    System.out.println("Bipartite matching test");
    for(int i=0;i<adjacent.length;i++){
      for(int j=0;j<adjacent[i].length;j++){
        System.out.print(((adjacent[i][j]) ? "1" : "0"));
      }
      System.out.println();
    }
    Vector<Integer> alice=new Vector<Integer>();
    Vector<Integer> bob =new Vector<Integer>();
    for(int i=0;i<adjacent.length;i++){
      alice.add(i);
    }

    for(int j=0;j<adjacent[0].length;j++){
      bob.add(j);
    }
    MatrixTest matrixTest = new MatrixTest(adjacent);
    BipartiteGraph<Integer,Integer> g = new BipartiteGraph<Integer,Integer>(matrixTest,alice,bob);
    Vector<Integer> result = g.getMatching();
    for(int i=0;i<result.size();i++){
      System.out.println(""+i+" connects to "+result.get(i));
    }
    return result;
  }


  public static int[] getKofN(int k, int n, Random r){
    if (k<0){
      throw new RuntimeException("k must be non-negative:"+k);
    }

    if (k==0){
      return new int[0];
    }
      
    if (n<=0){
      throw new RuntimeException("n must be positive:"+n);
    }
    int[] dummy = new int[n];
    int[] result = new int[k];

    for(int i=0;i<n;i++){
      dummy[i]=i;
    }

    for(int i=0;i<k;i++){
      int selection = r.nextInt(n-i)+i;
      result[i] = dummy[selection];
      dummy[selection] = dummy[i];
    }
    return result;
  }
    

  public static void testRandom(Random r){
    int numRows = r.nextInt(10)+1;
    int numCols = r.nextInt(10)+1;
    int minDim = (numRows<numCols) ? numRows : numCols;
    int numPairs = r.nextInt(minDim);
    double density = r.nextDouble();
    boolean[][] adjacent = new boolean[numRows][numCols];

    for(int i=0;i<adjacent.length;i++){
      for(int j=0;j<adjacent[i].length;j++){
        if (r.nextDouble()<density){
	  adjacent[i][j]=true;
	}
      }
    }

    int[] rowOfPair = getKofN(numPairs,numRows,r);
    int[] columnOfPair = getKofN(numPairs,numCols,r);
    for(int i=0;i<numPairs;i++){
      System.out.println("("+rowOfPair[i]+","+columnOfPair[i]+")");
      adjacent[rowOfPair[i]][columnOfPair[i]]=true;
    }
    Vector<Integer> result = testMatrix(adjacent);
    if (result.size()!=numRows){
      throw new RuntimeException("Number of rows in solution is not correct"); 
    }
    int countMatches=0;
    boolean[] claimed = new boolean[numCols];
    for(int i=0;i<result.size();i++){
      if (result.get(i)!=null){
        countMatches++;
	if (!adjacent[i][result.get(i)]){
	  throw new RuntimeException("Matched two numbers that were not adjacent:"+i+ " and " +result.get(i));
	}
	if (claimed[result.get(i)]){
	  throw new RuntimeException("Matched element "+result.get(i)+" twice.");
	}
	claimed[result.get(i)]=true;
      }
    }
    if (countMatches<numPairs){
      throw new RuntimeException("Failed to find a maximum matching");
    }
  }

  /**
   * This is for testing purposes.
   */
  public static void main(String[] args){
    Random r = new Random(0);
    for(int i=0;i<1000;i++){
      testRandom(r);
    }
    boolean[][] adjacent1 =
    {{false,true,true},{true,false,true},{true,true,false}};
    testMatrix(adjacent1);

    boolean[][] adjacent2 =
    {{false,false,true},{true,false,true},{true,true,false}};
    testMatrix(adjacent2);

    boolean[][] adjacent3 =
    {{false,false,false},{true,false,false},{true,true,false}};
    testMatrix(adjacent3);

    boolean[][] adjacent4 =
    {{false,false,true,true},{true,false,false,false},{true,false,false,false}};
    testMatrix(adjacent4);

    boolean[][] adjacent5 =
    {{false,false,true,true},{true,false,false,false},{true,false,true,false}};
    testMatrix(adjacent5);
  }
}
