## A selfplay reinforcement learning MCTS neural network that plays Ultimate Tic-Tac-Toe (UTTT)
<p>
<img src="./images/Game_Mouse.png" width="393" height="412">
</p>

## Rules:
* Player must play in the mini-board corresponding to the location of the previous move in its mini-board.
* A mini-board is filled when it is won by a player or all tiles are taken (tie).
* A player can play in any mini-board if the one pointed to is filled.
* Game is won when a player wins the big board of mini-boards. 
* Tied mini-boards do not count toward any player.
* Game is tied when no player can win the big board.

Run `stats.py` to play against 1000-node MCTS