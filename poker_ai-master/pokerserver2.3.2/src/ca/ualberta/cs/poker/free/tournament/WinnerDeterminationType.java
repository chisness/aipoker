package ca.ualberta.cs.poker.free.tournament;


public enum WinnerDeterminationType {
	INSTANTRUNOFFBANKROLL,INSTANTRUNOFFSERIES,TRUNCATEDBANKROLL;
	public static WinnerDeterminationType parse(String str){
		for(WinnerDeterminationType lim:WinnerDeterminationType.values()){
			if (lim.toString().equals(str)){
				return lim;
			}
		}
		return INSTANTRUNOFFBANKROLL;
	}
}
