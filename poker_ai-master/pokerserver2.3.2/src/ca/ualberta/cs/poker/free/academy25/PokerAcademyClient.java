/*
 * PokerAcademyClient.java
 *
 * Created on April 21, 2006, 11:33 AM
 */

package ca.ualberta.cs.poker.free.academy25;


import ca.ualberta.cs.poker.free.client.PokerClient;
import ca.ualberta.cs.poker.free.dynamics.HandAnalysis;
import ca.ualberta.cs.poker.free.dynamics.MatchStateMessage;
import com.biotools.meerkat.*;
import java.io.*;
import java.net.*;
/**
 * This class allows for a Player from PokerAcademy to be plugged in.
 *
 * @author Martin Zinkevich
 */
public class PokerAcademyClient extends PokerClient {
    private GameInfoDynamics dynamics;
    private GameInfoImpl gameinfo;
    private Player player;
    /** Creates a new instance of PokerAcademyClient 
     * Pass a pointer to the player you want to use.
     */
    public PokerAcademyClient(Player p) {
        gameinfo = null;
        dynamics = null;
        player = p;
    }
    

    /**
     * Not sure why I can't just run new Hand(str),
     * but this will work for now.
     */
    public static Hand getHand(String str){
      Hand h = new Hand();
      for(int i=0;i<str.length();i+=2){
        Card c = new Card(str.substring(i,i+2));
        h.addCard(c);
      }
      return h;
    }

    /**
     * Called at the start of the game
     */
    private void handleStartGame(){
        MatchStateMessage message = new MatchStateMessage(this.currentGameStateString);
        dynamics.doNewGame(message.handNumber,(message.seatTaken==1) ? 0 : 1);
        player.gameStartEvent(gameinfo);
        player.stageEvent(0);
        // Small blind 
        player.actionEvent(gameinfo.getSmallBlindSeat(),Action.smallBlindAction(gameinfo.getSmallBlindSize()));
        dynamics.doPostSmallBlind();
        player.gameStateChanged();
        // Big blind
        dynamics.currentPlayerSeat = dynamics.getOtherSeat(dynamics.button);
        player.actionEvent(dynamics.getOtherSeat(gameinfo.getSmallBlindSeat()),Action.bigBlindAction(gameinfo.getBigBlindSize()));
        dynamics.doPostBigBlind();
        player.gameStateChanged();
        dynamics.currentPlayerSeat = dynamics.button;
        player.dealHoleCardsEvent();
        
        //System.out.println("Hole cards:"+message.hole[message.seatTaken]);
         
        //Hand hole = new Hand(message.hole[message.seatTaken]);
        Hand hole = getHand(message.hole[message.seatTaken]);
        //System.out.println("Hole cards converted:"+hole);
        player.holeCards(hole.getFirstCard(),hole.getLastCard(),0);
    }
    
    
    /**
     * Called whenever an action is sent FROM the server.
     */
    private void handleAction(){
        MatchStateMessage message = new MatchStateMessage(this.currentGameStateString);
        int index = message.getLastAction();
        switch(index){
            case 0:
                handleFold();
                break;
            case 1:
                handleCall();
                break;
            case 2:
                handleRaise();
                break;
            default:
            break;
        }
    }
    
    
    /**
     * Called whenever a call action is sent FROM the server.
     */
    private void handleCall(){
        player.actionEvent(gameinfo.getCurrentPlayerSeat(),Action.callAction(gameinfo));
        dynamics.doPostCheckOrCall();
        player.gameStateChanged();
        if (gameinfo.getNumToAct()==0){
            if (gameinfo.getStage()==Holdem.RIVER){
                handleShowdown();
            } else {
                handleStage();
            }
        } else {
            dynamics.changeCurrentSeat();
        }
    }
    
    /**
     * Called whenever a raise action is sent FROM the server.
     */
    private void handleRaise(){
        player.actionEvent(gameinfo.getCurrentPlayerSeat(),Action.raiseAction(gameinfo));
        dynamics.doPostBetOrRaise();
        player.gameStateChanged();
        dynamics.changeCurrentSeat();
    }
    
    
    
    private void handleFold(){
        player.actionEvent(gameinfo.getCurrentPlayerSeat(),Action.foldAction(gameinfo));
        dynamics.doPostFold();
        player.gameStateChanged();
        dynamics.doPreWinEvent(dynamics.getOtherSeat(gameinfo.getCurrentPlayerSeat()));
        player.winEvent(gameinfo.getCurrentPlayerSeat(),gameinfo.getTotalPotSize(),null);
        dynamics.doPreGameOver();
        player.gameOverEvent();
    }
    
    private void handleStage(){
        MatchStateMessage message = new MatchStateMessage(currentGameStateString);
        dynamics.setBoard(message.board);
        dynamics.doPreStageEvent(dynamics.stage+1);
        player.stageEvent(dynamics.stage);
    }
    
    /**
     * At present, an empty string is sent with each win event.
     */
    private void handleShowdown(){
        // System.out.println("handleShowdown:Client:"+getClientID()+currentGameStateString+":stage:"+gameinfo.getStage());
        MatchStateMessage message = new MatchStateMessage(this.currentGameStateString);
        handleShowCardsAtShowdown(0);
        handleShowCardsAtShowdown(1);
        int winner = HandAnalysis.determineWinner(message.hole,message.board);
        if (winner==-1){
            dynamics.doPreTieEvent(0);
            player.winEvent(0,gameinfo.getTotalPotSize()/2.0,"");
            dynamics.doPreTieEvent(1);
            player.winEvent(1,gameinfo.getTotalPotSize()/2.0,"");
        } else {
            // Need to flip winner if we are in a different seat
            dynamics.doPreWinEvent((message.seatTaken==0) ? winner : (1-winner));
            player.winEvent(gameinfo.getCurrentPlayerSeat(),gameinfo.getTotalPotSize(),"");            
        }

        dynamics.doPreGameOver();
        player.gameOverEvent();

    }
        
    /**
     * Show a particular player's card at the showdown.
     * Note: there is no mucking.
     */
    private void handleShowCardsAtShowdown(int seat){
        MatchStateMessage message = new MatchStateMessage(currentGameStateString);
        int serverSeat = (message.seatTaken==0) ? seat : (1-seat);
        Hand hole = getHand(message.hole[serverSeat]);
        dynamics.hole[serverSeat]=new Hand(hole);
        player.showdownEvent(seat,hole.getFirstCard(),hole.getLastCard());
    }
    
    /**
     * Called whenever the state is changed.
     */
    public void handleStateChange() throws IOException, SocketException{
        if (gameinfo==null){
            dynamics = new GameInfoDynamics();
            gameinfo = new GameInfoImpl(dynamics);
            handleStartGame();
        } else {
            long oldHandNumber = gameinfo.getGameID();
            //int oldStage = gameinfo.getStage();
            MatchStateMessage message = new MatchStateMessage(currentGameStateString);
            if (oldHandNumber!=message.handNumber){
                handleStartGame();
            } else {
                handleAction();
            }
        }
        if (gameinfo.getCurrentPlayerSeat()==0){
            
            // System.out.println("ACT:Client:"+getClientID()+currentGameStateString+":roundBets:"+dynamics.roundBets);
            
            Action a = player.getAction();
            if (a==null){
                sendFold();
            } else if (a.isCheckOrCall()){
                sendCall();
            } else if (a.isBetOrRaise()){
                sendRaise();
            } else {
                sendFold();
            }
        }
    }
    
    /**
     * NOT WORKING YET
     */
    /*public static Player getPlayerFromLoadedJarFile(String botDescriptionFile)
            throws ClassNotFoundException, NoSuchMethodException, 
            InstantiationException, IllegalAccessException,
            InvocationTargetException{

        System.out.println("prefs file name:"+botDescriptionFile);
        Preferences prefs = new Preferences(botDescriptionFile);
        String className = prefs.getPreference("BOT_PLAYER_CLASS");
        System.out.println("class name:"+className);
        Class playerClass = Class.forName(className);
        Class[] paramClasses = new Class[0];
        Constructor constructor = playerClass.getConstructor(paramClasses);
        Object playerObject = constructor.newInstance(null);
        Player result = (Player)playerObject;
        result.init(prefs);
        return result;
    }*/

    
    /**
     * NOT WORKING YET
     */
    /*public static Player getPlayerFromBotFile(String botDescriptionFile) 
            throws ClassNotFoundException, NoSuchMethodException, 
            InstantiationException, IllegalAccessException,
            InvocationTargetException{
        System.out.println("java.library.path="+System.getProperty("java.library.path"));
        System.out.println("prefs file name:"+botDescriptionFile);
        Preferences prefs = new Preferences(botDescriptionFile);
        String jarFile = prefs.getPreference("PLAYER_JAR_FILE");
        System.out.println("jar file name:"+jarFile);
        String className = prefs.getPreference("BOT_PLAYER_CLASS");
        System.out.println("class name:"+className);
        System.loadLibrary(jarFile);
        Class playerClass = Class.forName(className);
        Class[] paramClasses = new Class[0];
        Constructor constructor = playerClass.getConstructor(paramClasses);
        Object playerObject = constructor.newInstance(null);
        Player result = (Player)playerObject;
        result.init(prefs);
        return result;
    }*/
    
}
