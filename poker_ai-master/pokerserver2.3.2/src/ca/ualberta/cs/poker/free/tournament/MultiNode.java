package ca.ualberta.cs.poker.free.tournament;

import java.security.SecureRandom;
import java.util.Vector;

/**
 * When we want to load multiple tournaments from the profile, create a tree
 * where the children are nodes, and the methods call the all member functions
 * of the children nodes
 * 
 * @author Christian Smith
 *
 */
public class MultiNode implements Node {

	Vector<Node> nodes;
	
	public MultiNode( Vector<Node> nodes_) {
		nodes = nodes_;
	}
	
	public void addNode( Node node ) {
		nodes.add( node );
	}
	
	/**
	 * confirm all the children nodes
	 */
	public boolean confirmCardFiles() {
		boolean val = true;
		for( Node n:nodes) {
			if ( n.confirmCardFiles() == false ) {
				val = false;
			}
		}
		
		return val;
	}

	public void generateCardFiles(SecureRandom random) {
		for( Node n:nodes) {
			n.generateCardFiles(random);
		}
		
	}

	/**
	 * Return all the winners of the children nodes
	 */
	public Vector<BotInterface> getWinners() {
		Vector<BotInterface> winners = new Vector<BotInterface>();
		for( Node n: nodes ) {
			winners.addAll(n.getWinners());
		}
		return winners;
	}

	/**
	 * Are all nodes complete?
	 */
	public boolean isComplete() {
		boolean allDone = true;
		
		for( Node n: nodes) {
			if( n.isComplete() == false) 
				allDone = false;
		}
		return allDone;
	}

	/**
	 * load all the children nodes
	 */
	public void load(Forge w) {
		for( Node n: nodes) {
			n.load(w);
		}
	}

	public void showStatistics() {
		for( Node n: nodes ) {
			n.showStatistics();
		}
	}

}
