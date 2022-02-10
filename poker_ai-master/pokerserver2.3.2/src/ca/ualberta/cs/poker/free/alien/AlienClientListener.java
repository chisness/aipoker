package ca.ualberta.cs.poker.free.alien;

public interface AlienClientListener {
	/** 
	 * Called after matchName is added to the completedMatchStrings
	 * @param matchName the match to be added.
	 */
	public void handleMatchCompleted(String matchName);
	public void handleMatchTerminated(String matchName);

	//public void handleMatchStarted(String matchName);
}
