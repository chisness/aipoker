package ca.ualberta.cs.poker.free.server;

import java.io.IOException;
import java.net.Socket;
import java.net.SocketException;

public class RingAgent extends TimedSocket {
	/**
	 * The protocol of the agent.
	 */
	String protocol;
	
	/**
	 * Is this bot kicked out of the match?
	 */
	boolean inGoodStanding;
	
	/**
	 * Construct a new agent for a new player
	 * @param socket the client socket
	 * @param playerIndex the index of the player
	 * @throws SocketException 
	 * @throws IOException
	 */
	public RingAgent(Socket socket, int playerIndex) throws SocketException,
			IOException {
		super(socket, playerIndex);
		protocol = null;
		inGoodStanding = true;
		open();
	}
	
	

}
