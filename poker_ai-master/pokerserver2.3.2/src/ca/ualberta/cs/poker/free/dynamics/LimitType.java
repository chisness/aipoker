package ca.ualberta.cs.poker.free.dynamics;

public enum LimitType {
	LIMIT,POTLIMIT,NOLIMIT,DOYLE;
	public static LimitType parse(String str){
		for(LimitType lim:LimitType.values()){
			if (lim.toString().equals(str)){
				return lim;
			}
		}
		return LIMIT;
	}
}
