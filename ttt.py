from typing import List, Optional, Dict
from abc import ABC, abstractmethod


class GamePosition(ABC):
    @abstractmethod
    def value(self) -> Optional[int]:
        pass

    @abstractmethod
    def player(self) -> int:
        pass

    @abstractmethod
    def available_moves(self) -> List['GamePosition']:
        pass


class TicTacToe(GamePosition):
    def __init__(self, positions: List[int], curplayer: int) -> None:
        self.positions = tuple(positions)
        self.curplayer = curplayer

    def __repr__(self) -> str:
        return f"({self.positions}, {self.curplayer})"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, another: object) -> bool:
        return repr(self) == repr(another)

    def value(self) -> Optional[int]:
        for (a, b, c) in [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]:
            if self.positions[a] == self.positions[b] == self.positions[c] != 0:
                return self.positions[a]
        if all(x != 0 for x in self.positions):
            return 0
        return None

    def available_moves(self) -> List['TicTacToe']:  # type: ignore
        res = []
        for i in range(9):
            t = list(self.positions)
            if t[i] == 0:
                t[i] = self.curplayer
                res.append(TicTacToe(t, 0-self.curplayer))
        return res

    def player(self) -> int:
        return self.curplayer


class MinMaxSearch:
    def __init__(self):
        self.memo: Dict[GamePosition, int] = {}

    def find_value(self, state: GamePosition):
        if state not in self.memo:
            val = state.value()
            if val is None:
                val = max(self.find_value(s) * state.player() for s in state.available_moves())
                self.memo[state] = val * state.player()
            else:
                self.memo[state] = val
        return self.memo[state]


if __name__ == "__main__":
    mms = MinMaxSearch()
    print(mms.find_value(TicTacToe([0, 0, 0, 0, 0, 0, 0, 0, 0], 1)))