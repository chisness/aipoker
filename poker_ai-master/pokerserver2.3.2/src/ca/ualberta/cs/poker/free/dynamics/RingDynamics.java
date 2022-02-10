// RingDynamics.java

package ca.ualberta.cs.poker.free.dynamics;

import java.security.SecureRandom;
import java.util.*;
import java.io.*;

/**
 * 
 * <P>
 * This class provides the mechanics for three games. A fourth (no-limit
 * ring) is also given, but in a more untested form.</P>
 * <UL>
 * <LI>RING LIMIT:2/4 limit texas hold'em with 3-10 players</LI>
 * <LI>HEADS-UP LIMIT:2/4 limit reverse blinds texas hold'em with 2 players</LI>
 * <LI>HEADS-UP NO-LIMIT:No-limit reverse blinds texas hold'em with 2 players (NOTE: This will not be used for the competition: Doyle's game will be used instead)</LI>
 * <LI>HEADS-UP DOYLES:1000 chip limit reverse blinds texas hold'em with 2 players</LI>
 * </UL>
 * <P>DISCLAIMER: Although there are notes throughout this implementation, THE
 * IMPLEMENTATION IS THE FORMAL STATEMENT OF THE RULES, NOT THE COMMENTS.</P>
 * 
 * 
 * 
 * 
 * <H2>Ring Limit (More than 2 players)</H2>
 * 
 * 
 * <P>If there are k players, seats are numbered 0 through k-1.</P>
 * <P>The <I>i</I>th player on the <I>j</I>th round is in seat <I>i-j</I> MOD <I>k</I>.</P>
 * <P> The highest number seat is the <B>button</B>.  Seat 0
 * is the <b>small blind seat</b>. Seat 1 is the <b>big blind seat</b>. The first to act (voluntarily) in the
 * first round is Seat 2, and thereafter the lowest index seat is the first
 * to act.</P>
 *
 * <P>In a hand of poker, a player can put money in the <B>pot</B> from her <B>stack</B>. Whenever it is said that
 * a player puts money in the pot, it is assumed it is from her stack (and is therefore removed from her stack). 
 * The money in
 * the pot goes to the winner or winners of the pot. We assume that the stack of each
 * player is quite large (at least 48 chip for each hand played) and therefore no player
 * goes bankrupt.</P>
 * 
 * <P>At all times, it is maintained how much each player has put in the pot. The maximum amount in the
 * pot is the maximum of all these amounts.
 * At the beginning of a hand, all players are <B>active</B>, meaning they are
 * still eligible for the money in the pot.</P>
 * 
 * <P>
 * The small blind amount is 1 chip, the big blind amount is 2 chips.
 * The <B>bet amount</B> on the preflop and flop is 2 chips, and on the turn and river is 4 chips.
 * </P>
 * 
 * <P>The hand begins by the player in the small blind seat putting 1 chip into the pot. The player
 * in the big blind seat then puts 2 chips in the pot. </P>
 * 
 * <P>Two private cards are then dealt to each player. These are only revealed to that player.</P>
 * 
 * <H3>Preflop Betting</H3>
 * <P>Seat 2 is the first to make a voluntary
 * act. She has three choices:
 * </P>
 * <UL>
 * <LI><B>fold</B>: Place no more money in the pot, and become <B>inactive</B> (ineligible for the pot).</LI>
 * <LI><B>call</B>: Bring the amount she has in the pot to the maximum amount in the pot.</LI>
 * <LI><B>raise</B>: Bring the amount she has in the pot to the maximum amount in the pot, and then put in the bet amount (2 chips) </LI>.
 * </UL>
 * <P>
 * For any Seat k (except the button), the <B>next seat</B> is Seat k+1. The next
 * seat after the button is Seat 0. The <B>next active player</B> is determined by
 * going from one seat to the next until an active player is found.
 * </P>
 * 
 * <P>A round ends after both everyone has made a voluntary action (not a blind) and if there was a raise,
 * all (but the last to raise) active players have made a voluntary action thereafter.</P>
 * 
 * 
 * <P>After a player has made a voluntary act, if the round has not ended, the next active player 
 * makes an voluntary action. She can always be fold or call.
 * She can raise if no more than two raises have already been made.</P>
 * 
 * <H3>Flop</H3>
 * <P>After the preflop betting has ended, three flop cards are revealed to all the players.</P>
 * 
 * A round of betting similar to the preflop betting, except:
 * <UL>
 * <LI>The first player to act is the active player in the seat with lowest index.</LI>
 * <LI>A player may raise if there have been less than four raises made previously on the round.</LI>
 * </UL>
 * 
 * <h3>Turn</h3>
 * <P>After the flop betting has ended, the turn card is revealed to all the players.</P>
 * <P>A betting round follows similar to the flop betting, except the betting amount is now 4 chips.
 * 
 * <h3>River</h3>
 * <P>After the river betting has ended, the river card is revealed to all the players.</P>
 * <P>A betting round follows identical to the turn betting.</P>
 * 
 * <h3>Showdown</h3>
 * <P>After the river betting has ended, any players who are active (yet to fold this hand) are 
 * eligible for the pot. All active players reveal their cards: by considering each players' two private
 * cards and the five public cards (three flop, the turn, the river), one can determine, given two
 * players, which has better cards or if their cards are of equal quality. See <A href="HandAnalysis.html">HandAnalysis</A>.
 * Thus, we consider all players who have the best cards, who we refer to as <B>winners</B> of the hand.
 * If the number of winners divides the number of chips in the pot, it is divided evenly. If the number
 * of winners does not divide the number of chips in the pot, then extra chips are given to winners in seat number
 * ordering.
 * </P>
 *
 * <H2>Heads-Up Limit</H2>
 * If there are TWO PLAYERS, then it is REVERSE BLINDS: the button (Seat 1) 
 * plays the small blind, the other player (Seat 0) the big blind. Seat 1 goes first
 * on the first round, and Seat 2 thereafter. Otherwise, the rules are the same
 * as ring limit.
 * 
 * <H2>Heads-Up No Limit</H2>
 * <P>The game of heads-up no limit is similar to heads-up limit, but there are more actions available to the players.</P>
 * <P>It is also the case that the players have a stack size limited to (1000?) chips. If a player has all of her chips in
 * the pot, and loses the hand, the match ends.
 * <P>If a player is in Seat 0 (the big blind seat) and only has 1 chip, she can still put in that 1 chip as the big blind.
 * Seat 1 must still put a second chip in the pot to call the big blind.</P>
 * <P>In particular, a player can still fold or call, and any time it is a player's turn to act and her stack is not zero, 
 * she can raise (no limit on the number of raises per round).</P>
 * <P>The bet amount is, to some degree, the choice of the player raising. If it is the first bet on a round, it
 * must be at least the size of the big blind (2 chips). If it is not, then it must at least be the size of the previous bet.
 * The exception to this rule is that a player can always go all in, putting all her remaining stack in the pot. Of course,
 * the player cannot have a negative stack after a raise, which places an upper limit upon how much she can raise.</P>
 * 
 * <P>
 * If one player puts more money in the pot than the other is capable due to her stack, then the other is only required to go all in
 *  to stay active. However, the player with more in the pot gets back the difference of the amounts in the pot before the rest of
 *  the pot is given to the winner(s).
 * </P>
 * 
 * 
 * 
 * <H2>Implementation of the Protocol</H3>
 * <P>
 * Before every action of a player, and at the end of the hand, all players receive the state of the hand (see
 *   {@link #getMatchState(int)}).
 * It is of the form:<BR>
 * {@code MATCHSTATE:<seat>:<handNumber>:<bettingSequence>:<cards>}<BR>
 * 
 * <P>In a simple betting sequence (for limit games), the betting sequence is the list of raises &quot;r&quot;, calls &quot;c&quot;, and folds &quot;f&quot;.
 * with the ends of rounds specified by slashes. In an advanced betting sequence, the total amount the player has in the
 * pot that round follows a call or raise.
 * <P>Actions are preformed by echoing the match state as it was sent and appending a colon and an action string.</P>
 * A fold is indicated by &quot;f&quot;, a call by &quot;c&quot;, and a raise by &quot;r&quot; and possibly an amount. The
 * amount is not not required in the limit game.
 * </P>
 * <H3>Illegal actions</H3>
 * <P>If an illegal action is submitted, there is a mapping from all actions (strings) to legal ones. In particular, if a player
 * makes an action that does not begin with 'r', 'f', or 'c', the player calls. If a player
 * attempts to raise and cannot raise, the player calls. If a player gives a raise that is too high, it is changed to the maximum
 * raise. If a player gives a raise that is too low or does not specify an amount, it is the minimum raise.</P>
 * 
 * <P>Blinds have been added to the betting sequence in the more generic language.
 * <P>Current stack sizes has not been added to the match state.
 * <H3>Acknowledgements</H3>
 * 
 * The rules for heads-up no-limit and ring limit are based largely upon the rules of the US Poker 
 * Association.<BR>
 * The rules for limit are based upon the rules found in Poker Academy.<BR>
 * 
 * Darse Billings clarified some of the rules in the no-limit ring implementation, which
 * will not be in the competition this year.<BR>
 * 
 * @author Martin Zinkevich
 */
public class RingDynamics {
	public MatchType info;
    /**
     * The number of players
     */
    public int numPlayers;

    /**
     * The number of seats, i.e. active players.
     */
    public int numSeats;

    /**
     * player[i] is the player in seat i
     */
    int[] player;
    
    /**
     * The stack size of a particular player.
     * Note that this is NOT indexed by seat, but by player
     */
    public int[] stack;
    
    /**
     * The name of the bots playing
     * Indexed by player, not seat.
     */
    public String[] botNames;
    
    /**
     * inPot[i] is the contribution to the pot of the 
     * player in seat i (in CHIPS).
     */
    public int[] inPot;
    /**
     * amountWon[i] is the net amount won by the player in
     * seat i (in CHIPS).
     */
    public int[] amountWon;

    /**
     * grossWon[i] is the gross amount won by the player
     * in seat i (in CHIPS).
     */
    public int[] grossWon;
    
    /**
     * active[i] is true if the player in seat i is 
     * still active (has not folded).
     */
    public boolean[] active;

    /**
     * canRaiseNextTurn[i] is true if the player in seat i can
     * still raise her next turn this round (unless four bets 
     * are made before play reaches him in a limit game, she has no
     * money, or she has already folded).
     */
    public boolean[] canRaiseNextTurn;
    
    /**
     * Round bets is the number of bets made in this round
     * so far. This is to keep betting capped at 4.
     */
    public int roundBets;

    /**
     * The sequence of backslash-delimited bets. f indicates
     * a fold, c a call, r a raise.
     */
    public String bettingSequence;

    /**
     * The seat next to act.
     */
    public int seatToAct;
    
    /**
     * The last player to have bet, or the first player to have made
     * a voluntary action in the round.
     */
    public int lastActionSeat;
    
    
    /**
     * Most recent bet size. Initially (on every round), it is the big blind.
     */
    public int lastBetSize;
    
    /** 
     * Round index incremented when the cards for that round are dealt. 
     * Preflop is 0, Flop is 1, Turn is 2, River is 3, Showdown is 4. */
    public int roundIndex;
    
    /** 
     * The next action will be the first action on the round. 
     */
    public boolean firstActionOnRound;
    
    /**
     * The amount of the last partial bet
     */
    int lastPartialBetSize;
    
    /** The hand is over */
    public boolean handOver;
    
    /** True if the player has won, false otherwise */
    //public boolean[] winner;
    
    /** Cards in the hole */
    public Card[][] hole;
    
    /** Full board (may not have been revealed) */
    public Card[] board;
    
    
    /**
     * the number of the hand.
     */
    public int handNumber;
    
    /** 
     * Creates a new instance of PokerServer, a limit game. 
     */
    /*
    public RingDynamics(SecureRandom random,int numPlayers) {
      this(numPlayers,0,LimitType.LIMIT,false,null);
    }
    */
    
    public RingDynamics(int numPlayers, MatchType info, String[] botNames){
    	this.info = info;
    	handNumber = 0;
	    this.numPlayers = numPlayers;
	    this.numSeats = numPlayers;
		stack=new int[numPlayers];
		for(int i=0;i<stack.length;i++){
			stack[i]=info.initialStackSize;
		}
		player = new int[numSeats];
		for(int i=0;i<player.length;i++){
			player[i]=i;
		}
		
		board = null;
		hole = null;
		this.botNames = botNames;
    }
    /*
    public RingDynamics(int numPlayers, int initialStack, 
    		LimitType limitGame, boolean stackBoundGame,
    		String[] botNames) {
    	this(numPlayers,new GameInfo(limitGame,stackBoundGame,initialStack),botNames);
    }
    */
    
    
    public void startHand(SecureRandom random){
        initializeBets();
        dealCards(random);
    }
    
    
    /*
     * This will load the cards from a file instead of securerandom
     */
    public void startHand(BufferedReader cardFile) {

		initializeBets();
		dealCards(cardFile);
    }
    
    /**
     * Deals cards from a file. The cards are stored in a way such that
     * if the same file is used twice, then for the 13th hand in both matches:</P>
     * <UL>
     * <LI>The board cards will be the same in both matches.
     * <LI>The kth player will receive the same cards.
     * </UL>
     * 
     * 
     * <P>
     * The cards are permuted, as if no one has left the game.
     * This is necessary to agree with the card formats from the first year.
     * Reconstructing card files is certainly difficult, but avoidable if
     * they are saved.
     * </P>
     * 
     * @param cardFile the source of the cards, from which one line is
     * read
     */
    public void dealCards(BufferedReader cardFile){
		String cards = "";
		hole = new Card[numSeats][2];
		board = new Card[5];

		try {
			cards = cardFile.readLine();
		} catch (IOException e) {
			System.err.println("Error reading from specified card file");
		}
		if (cards==null){
			System.err.println("Could not read line in RingDynamics.dealCards");
		}

		if (cards.length() != (10 + numPlayers * 4)) {
			System.err.println("***** RingDynamics: Wrong line length in card file");
			System.err.println("File length:"+cards.length());
			System.err.println("Expected:"+(10+numPlayers * 4));
		}

		/**
		 * This draws the cards as if no players had ever exited 
		 * the game.
		 */
		for (int seat = 0; seat < numSeats; seat++) {
			int playerIndex = seatToPlayer(seat);
			// A player's seat decreases by one every hand (modulo numPlayers)
			int shiftsLeft = handNumber % numPlayers;
			// We can flip this to make the shift positive.
			int shiftsRight = numPlayers - shiftsLeft;
			int canonicalSeatIndex = (playerIndex + shiftsRight) % numPlayers;
			hole[seat][0] = new Card(cards.substring(4 * canonicalSeatIndex, 4 * canonicalSeatIndex + 2));
			hole[seat][1] = new Card(cards.substring(4 * canonicalSeatIndex + 2, 4 * canonicalSeatIndex + 4));
		}

		int boardBegin = 4 * numPlayers;
		for (int i = 0; i < 5; i++) {
			board[i] = new Card(cards.substring(boardBegin + (2 * i),
					boardBegin + 2 + (2 * i)));
		}
		// System.out.println( "HAND = " + cards );

	}
    
    
    
    /** 
     * Sets all cards from the SecureRandom device */
    public void dealCards(SecureRandom random) {
		Card[] dealt = Card.dealNewArray(random, 5+(numSeats*2));
		hole = new Card[numSeats][2];
		board = new Card[5];
		for (int i = 0; i < numSeats; i++) {
			hole[numSeats][0] = dealt[i * 2];
			hole[numSeats][1] = dealt[(i * 2) + 1];
		}

		for (int i = 0; i < 5; i++) {
			board[i] = dealt[(numSeats * 2) + i];
		}
	}

    /**
     * Map a player index to a seat index
     * @param player a player index
     * @return the respective seat index
     */
    public int playerToSeat(int playerIndex){
    	for(int i=0;i<player.length;i++){
    		if (player[i]==playerIndex){
    			return i;
    		}
    	}
    	return -1;
    }
    
    /**
     * Map a seat index to a player index
     * @param seat a seat index
     * @return the respective player index
     */
    public int seatToPlayer(int seat){
    	return player[seat];
    }
    
    /**
     * Gets the next seat, given the current one.
     * Loops around the "end" of the table.
     * @param seat
     * @return the next seat
     */
    public int getNextSeat(int seat){
        return (seat+1<numPlayers) ? (seat+1) : 0;
    }

    /**
     * Gets the next seat of an active player.
     * Note an active player may NOT be able to act this round.
     * @param seat
     * @return
     */
    public int getNextActiveSeat(int seat) {
		if (getNumActivePlayers()==0){
			throw new RuntimeException("No active players!!!");
		}
    	do {
			seat = getNextSeat(seat);
		} while (!active[seat]);
		return seat;
	}

    public int getNextSeatCanActThisRound(int seat){
    	if (getNumPlayersLeftToAct()==0){
    		System.err.println(this);
    		throw new RuntimeException("No players can act this round");
    	}
    	do{
    		seat = getNextSeat(seat);
    	} while (!canActThisRound(seat));
    	return seat;
    }
	/**
	 * Returns the maximum in pot, or bigBlindSize
	 * if the maximum is below the big blind size.
	 * 
	 * This is because the "big blind", even if not
	 * completed, must be matched by other players, and
	 * that the next raise must start from the big blind (i.e.,
	 * a bet must leave the max in pot to 4 small blinds).
	 * @return the maximum amount in the pot in CHIPS
	 */
	public int getMaxInPot() {
		int maxSoFar = info.bigBlindSize;
		for (int i = 0; i < numPlayers; i++) {
			if (inPot[i] > maxSoFar) {
				maxSoFar = inPot[i];
			}
		}
		return maxSoFar;
	}
    
    /**
	 * Get the stack size of the player in a particular seat
	 * 
	 * @param seat the seat of the player
	 * @return the size of the stack in CHIPS
	 */
    public int getSeatStack(int seat){
    	return stack[seatToPlayer(seat)];
    }
    
    /**
     * A player is all in if it is a stack bound game and her stack is zero.
     * @param seat the seat of the player.
     * @return true if the player is all in.
     */
    public boolean isAllIn(int seat){
    	return info.stackBoundGame && (getSeatStack(seat)==0);
    }
    
    /**
     * Gets the amount required for a certain seat to
     * call. If it is a stack bound game, it is the minimum 
     * of the stack size of the player
     * and the difference between the maximum in the pot
     * versus the pot of that player. Otherwise, it is
     * just the difference.
     * @param seat
     * @return the amount in CHIPS required to call
     */
    public int getAmountToCall(int seat){
    	// CHIPS
    	int maxSoFar=getMaxInPot();
    	int normalAmountToCall = maxSoFar-inPot[seat];
    	if (info.stackBoundGame){
    		int stackBound = getSeatStack(seat);
    		if (stackBound<normalAmountToCall){
    			return stackBound;
    		}
    	}
    	return normalAmountToCall;
    }

    
    /**
     * Gets the standard limit bet.
     * @return the amount of the limit bet.
     */
    public int getLimitBet(){
    	if (roundBets==4){
    		return 0;
    	}
		return (roundIndex<2) ? info.smallBetSize : info.bigBetSize;    	
    }
    
    /**
     * Get the maximum amount the player in seat can raise.
     * The player must have enough to call, and then can apply
     * the remainder of her stack to raise. This does not take
     * into account the amount that other players will be able
     * to call. There is no test as to whether
     * the player in that seat will actually have the opportunity to 
     * raise.
     * @param seat the seat of the player
     * @return the maximum amount to seat in CHIPS
     */
    public int getMaxRaise(int seat){
    	switch(info.limitGame){
    	case LIMIT:
    	default:
    	int maxNormalBet = getLimitBet()-lastPartialBetSize;
    		if (info.stackBoundGame){
    			int stackBound = getSeatStack(seat)-getAmountToCall(seat);
    			if (stackBound<maxNormalBet){
    				return stackBound;
    			}
    		}
    		return maxNormalBet;
    	case NOLIMIT:
    		// Here we assume it is a stack bound game.
    		return getSeatStack(seat)-getAmountToCall(seat);
    	case POTLIMIT:
    		return Math.min(getSeatStack(seat)-getAmountToCall(seat),getSumPotsAfterCall(seat)-getAmountToCall(seat));
    	case DOYLE:
    		return info.doyleLimit-(getAmountToCall(seat)+inPot[seat]);
    	}
    }
    
    public int getSumPotsAfterCall(int seat){
    	int initial = getAmountToCall(seat);
    	for(int potContribution:inPot){
    		initial+=potContribution;
    	}
    	return initial;
    }

    /**
     * An amount which is considered a full raise.
     * @return the full raise amount
     */
    public int getFullRaiseAmount(){
    	switch(info.limitGame){
    	case LIMIT:
    	default:
    		return getLimitBet();
    	case NOLIMIT:
    	case POTLIMIT:
    	case DOYLE:
    		return lastBetSize;    		
    	}
    }
    
    /**
     * The minimum amount for a legitimate raise. In a limit game,
     * this is the current bet size. In a no-limit game, there is 
     * no limit to the current bet size. There is no test as to whether
     * the player in that seat will actually have the opportunity to 
     * raise.<BR>
     * 
     * One can &quot;complete&quot; a bet
     * of another player who has made a partial bet. In this implementation,
     * suppose the last bet was 10 dollars, then someone went all in, and to match 
     * the all in bet requires only 4 dollars. Then the minimum bet is now 6 dollars.<BR>
	 * 
     * Moreover, making a minimum raise is not 
     * necessarily a "true" raise, unless it is the size of the previously made 
     * raise. If one goes all in with a min raise below the "normal" raise value, this
     * does not "re-open" betting.
     * @param seat the seat of the player
     * @return the minimum raise
     */
    public int getMinRaise(int seat){
    	int normalMinRaise = 0;
    	switch(info.limitGame){
    	case LIMIT:
    	default:
    		// Only one value in limit game.
    		normalMinRaise = getMaxRaise(seat)-lastPartialBetSize;
    		break;
    	case POTLIMIT:
    	case NOLIMIT:
    		normalMinRaise = lastBetSize-lastPartialBetSize;
    		break;
    	case DOYLE:
    		normalMinRaise = lastBetSize-lastPartialBetSize;
    		int boundRaiseD = info.doyleLimit-getAmountToCall(seat);
    		if (boundRaiseD<normalMinRaise){
    			return boundRaiseD;
    		}
    		return normalMinRaise;
    	}
    	
		if (info.stackBoundGame){
			int boundRaise = getSeatStack(seat)-getAmountToCall(seat);
			if (boundRaise<normalMinRaise){
				return boundRaise;
			}
		}
		return normalMinRaise;
    }

    /**
     * <P>
     * This returns the amount that be in the pot after a raise (from
     * the player in seat i).
     * This includes the amount for the call and what is already in the pot.
     * Return a valid raise close to the amount given to raise.
     * If the raise is too high, lower it to an acceptable amount.
     * If the raise is too low, increase it to an acceptable amount.<BR>
     * </P>
     * 
     * <P>
     * Note that there is no test to see if the seat is actually
     * active. If a seat is NOT active, then this is the amount that one
     * would put in upon the seat's next turn if no raises are made before
     * then.</P>
     * @param amount the amount in CHIPS the player wants to put in the pot
     * @param seat the seat from which the raise is tested.
     * 
     * @return the amount in CHIPS the player can put in the pot
     */
    public int makeValidPotAfterRaise(int amount, int seat){
    	int amountToCall = getAmountToCall(seat);
    	int amountInPot = inPot[seat];
    	//System.err.println("amountToCall:"+amountToCall);
    	int amountToRaise = amount - (amountToCall+amountInPot);
    	//System.err.println("amountToRaise:"+amountToRaise);
    	
    	return Math.min(getMaxRaise(seat), Math.max(getMinRaise(seat),amountToRaise))+(amountToCall+amountInPot);
    }
    
    /**
     * Returns true if it is legal to raise the next turn, 
     * there have not been four bets this round, and the 
     * player is not all in.
     * @param seat
     * @return
     */
    public boolean canRaise(int seat){
    	return canRaiseNextTurn[seat]&&(getMaxRaise(seat)>0);
    }
    
    
    /**
     * Move chips from the stack into the pot. Is assumed
     * that this amount is less than or equal to the amount in
     * the stack.
     * @param amount in CHIPS to add to seat
     * @param seat 
     */
    public void addToPot(int amount, int seat){
    	stack[seatToPlayer(seat)]-=amount;
    	inPot[seat]+=amount;
    }
    
   
    /**
     * Initialize the pots and the variables for 
     * the next hand. Pay the blinds, and set the
     * next seat to act.
     * 
     * 
     *
     */
    public void initializeBets(){
      bettingSequence = "";
      handOver = false;
      amountWon = null;
      grossWon = null;
      //winner = null;
      roundIndex = 0;
      firstActionOnRound = true;
      inPot = new int[numSeats];
      active = new boolean[numSeats];
	  canRaiseNextTurn=new boolean[numSeats];
      for(int i=0;i<numSeats;i++){
    	  canRaiseNextTurn[i]=true;
    	  inPot[i]=0;
    	  active[i]=true;
      }
      lastBetSize = 2;
      lastPartialBetSize = 0;
      roundBets = 1;
      int smallBlindSeat = 0;
      int bigBlindSeat = 1;
      seatToAct = 2;
      if (numSeats==2){
    	  // Reverse blinds implemented here
    	  smallBlindSeat = 1;
    	  bigBlindSeat = 0;
    	  seatToAct = 1;
      }
      int smallBlindThisHand=info.smallBlindSize;
      int bigBlindThisHand=info.bigBlindSize;
      if (info.stackBoundGame){
        smallBlindThisHand = Math.min(info.smallBlindSize,getSeatStack(smallBlindSeat));
        bigBlindThisHand= Math.min(info.bigBlindSize,getSeatStack(bigBlindSeat));
      }
      
      if (isBettingSequenceComplex()){
    	  bettingSequence += ("b"+smallBlindThisHand);
    	  bettingSequence += ("b"+bigBlindThisHand);
      }
      addToPot(bigBlindThisHand,bigBlindSeat);
      addToPot(smallBlindThisHand,smallBlindSeat);
      lastActionSeat = seatToAct;
    }
    
    public void incrementRound(){
        roundIndex++;
        if (roundIndex<4){
            for(int i=0;i<numSeats;i++){
          	  canRaiseNextTurn[i]=true;
            }
        	bettingSequence += '/';
            firstActionOnRound = true;
            roundBets=0;
            lastBetSize=info.bigBlindSize;
            lastPartialBetSize = 0;
	    // gets the lowest active index
	    seatToAct = getNextActiveSeat(numSeats-1);
	    //System.err.println("First seat to act on round "+roundIndex+":"+seatToAct);
        } else {
            //winner = getWinners();
            endHand();
        }
    }
    
    public boolean canActThisRound(int seat){
    	return active[seat] && (canRaiseNextTurn[seat] || (inPot[seat]<getMaxInPot()));
    }
    
    public int getNumPlayersLeftToAct(){
    	int initialSeat = getNextActiveSeat(0);
    	int seat=initialSeat;
    	int numCanAct = 0;
    	do{
    		if (canActThisRound(seat)){
    			numCanAct++;
    		}
    		seat = getNextActiveSeat(seat);
    	} while(seat!=initialSeat);
    	return numCanAct;
    }
    
    /**
     * Return the match state.
     * @param seat the seat of the player observing the state
     * @return the state from perspective of the player in seat seat.
     */
    public String getMatchState(int seat){
    		return "MATCHSTATE:" + seat + ":" + handNumber + ":" + bettingSequence + ":" + getCardState(seat);   	
    }
    
    
    
    /**
     * Return the global state at the end of a hand
     * @return a string that includes the bot names (backslash-delimited) in seat order, hand number, the betting sequence,
     * the card information, and the bankroll changes in seat order (in small blinds)
     */
    public String getGlobalState(){
    	String result = "";
    	if (botNames!=null){
    		result = botNames[seatToPlayer(0)];
    		for(int seat=1;seat<numSeats;seat++){
    			result += "|" + botNames[seatToPlayer(seat)];
    		}
    	}
    	result += (":" + handNumber + ":"+bettingSequence +":"+getCardState(-1));
    	if (amountWon!=null){
    		result += ":"+amountWon[0];
    		for(int i=1;i<numSeats;i++){
    			result+=("|"+amountWon[i]);
    		}
    	}
    	return result;
    	
    }
    
    /**
     * Cards are visible if the viewingSeat and the viewedSeat are
     * the same, or it is the showdown and the viewedSeat did not
     * fold.
     */
    public boolean isSeatVisible(int viewingSeat, int viewedSeat){
        return (viewingSeat==viewedSeat)||(roundIndex==4 &&
	active[viewedSeat]);
    }
    
    /**
     * Should the betting sequence have amounts of calls and raises?
     * True if no-limit or stack bound game.
     */
    public boolean isBettingSequenceComplex(){
    	return (info.limitGame!=LimitType.LIMIT)||info.stackBoundGame;
    }
    
    /**
     * If seat==-1, we want all the card info for the logs, this assumes
     * we will only use seat==-1 when the server wants logging info, otherwise
     * seat>=0
     * 
     * @param seat
     * @return the card state of the game, suitable to be sent over TCP/IP
     */
    public String getCardState(int seat) {
		String result = "";
		for (int i = 0; i < numSeats; i++) {
			if (i != 0) {
				result = result + "|";
			}
			if (isSeatVisible(seat, i) || seat == -1) {
				result = result + hole[i][0] + hole[i][1];
			}
		}

		if (roundIndex > 0) {
			result = result + "/" + board[0] + board[1] + board[2];
		}
		if (roundIndex > 1) {
			result = result + "/" + board[3];
		}
		if (roundIndex > 2) {
			result = result + "/" + board[4];
		}

		return result;
	}
    
    /**
	 * Updates the state when a call is made. If the game is no-limit or is no
	 * stack bound, the amount put in the pot IN CHIPS is placed in the betting
	 * sequence.
	 */
    public void handleCall(){
      bettingSequence = bettingSequence + 'c';
      // CHIPS
      int amountToCall = getAmountToCall(seatToAct);
      int amount = amountToCall+inPot[seatToAct];
      if (isBettingSequenceComplex()){
    	  bettingSequence += amount;
      }

      if (firstActionOnRound){
    	  lastActionSeat = seatToAct;
      }
      addToPot(amountToCall, seatToAct);
      canRaiseNextTurn[seatToAct]=false;
      if (firstActionOnRound){
    	  firstActionOnRound=false;
      } else {
          if (getNumPlayersLeftToAct()==0){
            incrementRound();
            return;
          }
      }
      seatToAct = getNextSeatCanActThisRound(seatToAct);
    }
    
    /**
     * Updates the state when a (legal) raise is made.
     * The amount is the amount put in the pot, i.e., the call amount
     * plus the raise amount.
     * Uses more general calls, where the call is followed by the
     * amount added to the pot.<BR>
     * 
     * Observe that this handles legitimate bids (bids which
     * are at least as large as the last bid) and illegitimate
     * bids (bids that are all-in, but smaller than the last bid).<BR>
     * 
     * If a bid is illegitimate, then it does not refresh the ability
     * of people to raise this round.
     */
    public void handleRaise(int amount){
    	//System.err.println("handleRaise()");
    	firstActionOnRound = false;
    	bettingSequence +=  'r';
    	if (isBettingSequenceComplex()){
    	  bettingSequence += amount;
    	}
    	int betSize = amount-(getAmountToCall(seatToAct)+inPot[seatToAct]);        
		lastActionSeat=seatToAct;
    	if (betSize+lastPartialBetSize>=getFullRaiseAmount()){
    		// A full bet is made.
   			roundBets++;
   			for(int i=0;i<numSeats;i++){
   				canRaiseNextTurn[i]=true;
   			}
   			canRaiseNextTurn[seatToAct]=false;
   			lastBetSize = betSize + lastPartialBetSize;
   			lastPartialBetSize = 0;
    	} else {
    		// A partial bet is made.
    		lastPartialBetSize+=betSize;
    		canRaiseNextTurn[seatToAct]=false;
    	}
    	addToPot(amount-inPot[seatToAct], seatToAct);
        if (getNumPlayersLeftToAct()==0){
        	incrementRound();
        } else {
        	seatToAct = getNextSeatCanActThisRound(seatToAct);
        }
    }
    
    
    /**
     * Updates the state when a (legal) raise is made.
     */
    public void handleRaise(){
    	int amount = makeValidPotAfterRaise(0,seatToAct);
    	handleRaise(amount);
    }
    
    /**
     * Updates the state when a (legal) fold is made.
     */
    public void handleFold(){
    	//System.err.println("handleFold():"+getGlobalState());
       bettingSequence=bettingSequence + 'f';
       firstActionOnRound = false;
       active[seatToAct]=false;
       if (getNumActivePlayers()<2){
         endHand();
       } else {
         if (getNumPlayersLeftToAct()==0){
    	     incrementRound();
         } else {
           seatToAct = getNextSeatCanActThisRound(seatToAct);
         }
       }
    }

    public void handleAction(char action){
    	//System.err.println("handleAction():"+this);
        handleAction(""+action);
    	/*switch(action){
            case 'c':
                handleCall();
                break;
            case 'r':
                if (roundBets<4){
                    handleRaise();
                    break;
                }
                // Fall through if illegal to raise
            default:
            case 'f':
                handleFold();
        }*/
    }
    
    public void handleAction(String action){
    	char actionChar = action.charAt(0);
    	//System.err.println("handleAction(): "+action+" seatToAct: "+seatToAct);
    	int amount = 0;
    	try{
    		if (action.length()>1){
    			amount = Integer.parseInt(action.substring(1));
    		}
    	} catch (NumberFormatException e){
    		amount = 0;
    	}
    	switch(actionChar){
    	case 'r':
    		if (canRaise(seatToAct)){
    			handleRaise(makeValidPotAfterRaise(amount,seatToAct));
    		} else {
    			handleCall();
    		}
    		break;
    	case 'f':
    		handleFold();
    		break;
    	default:
    	case 'c':
    		handleCall();
    		break;    	
    	}
    }
    
    /**
     * @deprecated The concept of winning a hand is no longer a boolean
     * situation, because several players can win or lose a hand. In fact,
     * it is possible to win a side pot while getting negative net winnings!<BR>
     * 
     * Gets who won on this hand (only valid on the showdown).
     * @return  a boolean array, indicating true for every winner in
     * the hand.
     */
    public boolean[] getWinners(){
    	throw new RuntimeException("Not implemented");
     }
    
    
    /**
     * Place money from a side pot into grossWon. Should NOT be
     * called until showdown. 
     *
	 * According to the US Poker Association, the
	 * odd chips should be distributed to winners
	 * starting left of the button until they run out.
     * @param playersIn the players that are eligible for this pot (put enough in the pot and did not fold)
     * @param potSize the size of the pot in CHIPS
     */
    public void handleSidePot(boolean[] playersIn, int potSize){
        boolean[] potWinners = new boolean[numSeats];
        for(int i=0;i<numSeats;i++){
            potWinners[i]=false;
          }
   
        int firstIndex = 0;
        while(playersIn[firstIndex]==false){
        	firstIndex++;
        }
        
        Card[][] playersToCompare = new Card[2][];
        playersToCompare[0]=hole[firstIndex];
        potWinners[firstIndex]=true;
        
        for(int i=firstIndex+1;i<numSeats;i++){
          if (playersIn[i]){
              playersToCompare[1] = hole[i];
          	int winnerIndex = HandAnalysis.determineWinner(playersToCompare,board);
          	if (winnerIndex==-1){
          		// tie
          	  potWinners[i]=true;
          	} else if (winnerIndex==1){
          		// new winner
          	  potWinners[i]=true;
          	  for(int j=0;j<i;j++){
          	    potWinners[j]=false;
          	  }
          	  playersToCompare[0]=hole[i];
          	}
          }
        }
        
        int numWinners = 0;
        for(int i=0;i<numSeats;i++){
        	if (potWinners[i]){
        		numWinners++;
        	}
        }
        
        // Amount given to every winner.
        int baseAmountWon = potSize/numWinners;
        int remainder = potSize-(baseAmountWon*numWinners);
        
        for(int i=0;i<numSeats;i++){
        	if (potWinners[i]){
        		grossWon[i]+=baseAmountWon;
        		if (remainder>0){
        			grossWon[i]++;
        			remainder--;
        		}
        	}
        }
    }
    
    
    /**
     * After the game is over, determine the gross amount from the pot
     * won by each player.
     * This function assumes that the game is terminated.
     * 
     * It splits the pot into a "main pot" and "side pots". Every
     * player still in the hand is eligible for the main pot, and
     * the side pots are only for players with a sufficient amount
     * in the pot.
     * 
     * <P>
     * A simple way to describe how the main pot (which below is just
     * the first "side pots") and side pots can be constructed is:
     * </P>
     * 
     * <UL>
     * <LI>Find the smallest amount (A) put in the pot by an active player.
     * <LI>Form the "main pot" by taking
     * <UL>
     * <LI>All the money from players with less (these players have folded).
     * <LI>An amount (A) from any player with an amount greater than or equal to (A).
     * </UL>
     * <UL>Find the next smallest amount (NA) put in the pot by an active player.
     * <LI>Form the next side pot by taking:
     * <UL>
     * <LI>All the money from players with less than (NA) in the pot that isn't
     * in an earlier pot.
     * <LI>An amount (NA-A) from players with an amount greater than or equal to (NA).
     * </UL>
     * </UL>
     */
    public void determineGrossWon(){
    	/*System.err.println("determineGrossWon():"+getGlobalState());
    	System.err.println("determineGrossWon():"+this);*/
    	grossWon = new int[numSeats];
    	TreeMap<Integer,Vector<Integer>> mapPotSizeToSeats = new TreeMap<Integer,Vector<Integer>>();
    	for(int i=0;i<numSeats;i++){
    		Vector<Integer> result = mapPotSizeToSeats.get(inPot[i]);
    		if (result==null){
    			result = new Vector<Integer>();
    			mapPotSizeToSeats.put(inPot[i],result);
    		}
    		result.add(i);
    	}
    	Vector<Integer> sidePotSizes=new Vector<Integer>();
    	Vector<boolean []> inSidePots=new Vector<boolean []>();
    	
    	
    	// The amount per player in the last side pot
    	int lastSidePotAmount = 0;
    	
    	// The current size of the side pot. Computed
    	// by finding the number of players who folded.
    	int currentSidePotSize = 0;
    	int playersRemaining = numSeats;
    	for(int potAmount:mapPotSizeToSeats.keySet()){
    		Vector<Integer> seats = mapPotSizeToSeats.get(potAmount);
    		for(int seat:seats){
    		if (active[seat]){
    			for(boolean[] inSidePot:inSidePots){
    				inSidePot[seat]=true;
    			}
    			if (lastSidePotAmount<potAmount){
    			  int potAfter = (potAmount-lastSidePotAmount)*playersRemaining;
    			  int potSize = potAfter + currentSidePotSize;
    			  sidePotSizes.add(potSize);
    			  boolean[] inSidePot = new boolean[numSeats];
    			  for(int i=0;i<numSeats;i++){
    				  inSidePot[i]=false;
    			  }
    			  inSidePot[seat]=true;
    			  inSidePots.add(inSidePot);
    			  currentSidePotSize = 0;
    			  lastSidePotAmount = potAmount;
    			}
    		} else {
    			currentSidePotSize += (potAmount-lastSidePotAmount);
    		}
    		playersRemaining--;
    		}
    	}
    	
    	int lastSidePot = sidePotSizes.size()-1;
    	// In the rare event that the person with the most in the pot folds for an obscure
    	// reason, need to add his extra to the last pot.
    	sidePotSizes.set(lastSidePot, sidePotSizes.get(lastSidePot)+currentSidePotSize);
    	assert(inSidePots.size()==sidePotSizes.size());
    	for(int i=sidePotSizes.size()-1;i>=0;i--){
    		handleSidePot(inSidePots.get(i),sidePotSizes.get(i));
    	}
    	//System.err.println("params:"+this);
    }
    
    /**
     * We end the hand if all but one player folds, or
     * we go to the showdown.
     */
    public void endHand(){
    	determineGrossWon();
        amountWon=new int[numSeats];
        for(int i=0;i<numSeats;i++){
        	amountWon[i]=grossWon[i]-inPot[i];
        	stack[seatToPlayer(i)]+=grossWon[i];
        }
        handOver = true;
    }
    
    /**
     * Remove a player from the table.
     * @param seat
     */
    public void removePlayer(int seat){
    	int[] newPlayer = new int[numSeats-1];
    	for(int i=0;i<seat;i++){
    		newPlayer[i]=player[i];
    	}
    	
    	for(int i=seat;i<numSeats-1;i++){
    		newPlayer[i]=player[i+1];
    	}
    	player = newPlayer;
    	numSeats--;
    }
    
    /**
     * Remove any players that went bust last 
     * hands and put the players in the next seat
     *
     */
    public void nextSeats(){
    	if (info.stackBoundGame){
        for(int i=0;i<numSeats;i++){
        	if (getSeatStack(i)==0){
        		removePlayer(i);
        		i--;
        	}
        }
    	}
        int lastPlayer = player[0];
        for(int i=0;i<numSeats-1;i++){
        	player[i]=player[i+1];
        }
        player[player.length-1]=lastPlayer;    	
    }
    
    /**
     * Sets the hand number.
     * @param handNumber
     */
    public void setHandNumber(int handNumber){
        this.handNumber = handNumber;
    }
    
    /**
     * Goes to the next hand. If the next hand is the first hand, leaves the hand
     * number and seats as is. Otherwise, increments them.
     * @param reader
     */
    public void nextHand(BufferedReader reader){
    	if (hole!=null){
    	  setHandNumber(handNumber+1);
    	  nextSeats();
    	}
    	startHand(reader);
    }
    
    /**
     * Return the number of players that have not folded.
     * A player is active if she has not folded.
     * @return the number of players that has not folded
     */
    public int getNumActivePlayers(){
    	int result = 0;
    	for(boolean activeSeat:active){
    		if (activeSeat){
    			result++;
    		}
    	}
    	return result;
    }
    
    /**
     * The game is over if it is the showdown or everyone but one
     * person has folded.
     * @return true if the game is over, false otherwise.
     */
    public boolean isGameOver(){
    	return (roundIndex==4)||(getNumActivePlayers()==1);
    }

    public int getNumActivePlayersNotAllIn(){
    	int result = 0;
      for(int i=0;i<numSeats;i++){
    	  if (active[i] && !isAllIn(i)){
    		  result++;
    	  }
      }
      return result;
    }
    
    /**
     * Gets the header for the log file. Must be called before the game begins
     * to get the accurate stack size.
     * @return the header as a string
     */
    public String getHeader(){
    	String result = "## GAMESTATE Version 2.0\n";
    	result+= "## type: "+ (info.limitGame) + (info.stackBoundGame ? " STACKBOUND":" NOSTACKBOUND")+"\n";
    	if (info.stackBoundGame){
    		result+= "## stacksize: "+stack[0];
    	}
    	result+="# Outcomes of hand are shown in the form:\n";
    	result+="# <PLAYERS>:<HANDNUMBER>:<BETTING>:<CARDS>:<NETONHAND>\n";
    	result+="# Players are listed in seat order:";
    	if (numPlayers>2){
    		result+="small blind, then big blind, then first to act.\n";
    		if (info.stackBoundGame){
    			result+="# If the number of players falls to two, then the ";
    		}
    	}
    	if ((numPlayers==2)||info.stackBoundGame){
    		result+="big blind (or button or dealer) is listed first.\n";
    	}
    	result+="# Cards on the preflop are in seat order, divided by vertical lines |.\n";
    	result+="# The net on won or lost on a hand (in small blinds) is last, and is in seat order.\n";
    	return result;
    }
    public String toString(){
    	String result = "type:"+ (info.limitGame) + (info.stackBoundGame ? " STACKBOUND":"");
    	result += "bettingSequence:"+bettingSequence+"\n";
    	result += "\nlastActionSeat:"+lastActionSeat+"\n";
    	result += "seatToAct:"+seatToAct+"\n";
    	result += "roundBets:"+roundBets+"\n";
    	result += "canRaise(seatToAct):"+canRaise(seatToAct)+"\n";
    	result += "getMinRaise(seatToAct):"+getMinRaise(seatToAct)+"\n";
    	
    	result += "active (by seat):";
    	for(int i=0;i<active.length;i++){
    		result+=" "+active[i];
    	}
    	
    	result += "\n"+"inPot (by seat):";
    	for(int i=0;i<inPot.length;i++){
    		result+=" "+inPot[i];
    	}
    	
    	result+= "\n"+"stack (by player):";
    	for(int i=0;i<stack.length;i++){
    		result+=" "+stack[i];
    	}
    	
    	result+= "\n"+"canRaise (by seat):";
    	for(int i=0;i<this.canRaiseNextTurn.length;i++){
    		result+=" "+canRaiseNextTurn[i];
    	}
    	
    	if (grossWon!=null){
    		result+="\n"+"gross won:";
    		for(int i=0;i<grossWon.length;i++){
    			result+=" "+grossWon[i];
    		}
    	}
    	return result;
    }
    
}
