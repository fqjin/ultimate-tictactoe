## Selfplay reinforcement learning <br/> for Ultimate Tic-Tac-Toe (UTTT)
<p>
<img src="./images/Game_Mouse.png" width="392" height="412">
</p>

**Run `gui.py` to play against a neural network.**

## Rules:
* The game board consists of a 3x3 grid of mini-boards, each having 3x3 tiles.
* Player X begins by playing in any tile. Play then alternates between players O and X.
* **Players must play in the mini-board corresponding to the location of the previous move in its respective mini-board.** For example, if the previous move was an X in the upper-right corner of the lower-right miniboard, then player O must play in the upper-right miniboard.
* A mini-board is filled when it is won by a player (3 tiles in a row) or all tiles are taken (tie). When sent to a filled mini-board, a player can play in any of the unfilled mini-boards.
* The game is won when a player wins the big board (3 mini-boards in a row). Tied mini-boards do not count for either player. The game is tied if neither player can win the big board.

## Selfplay strength estimate
<p>
<img src="./images/Ordo.png" width="390" height="260">
</p>
