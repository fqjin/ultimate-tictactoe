from engine import BigBoard, decode_dict
from players import BasePlayer
from net_player import NetPlayer
from alphabeta import ABPlayer
from time import time


def play(player0: BasePlayer,
         player1: BasePlayer,
         game=None,
         verbose=False,
         press_enter=False,
         give_moves0=True,
         give_moves1=True,
         temp=None,
         moves=None,
         timing=False,
         ):
    if game is None:
        game = BigBoard()
    if moves is None:
        moves = []
    evals = []
    times = []
    if verbose:
        game.draw()

    player0.reset()
    player1.reset()
    while True:
        if press_enter:
            input()

        if temp and len(moves) < temp[0]:
            args = {'invtemp': temp[1]}
        else:
            args = {}
        # TODO: Rewrite keep_tree logic better
        if timing:
            time1 = time()
        if game.mover:
            if give_moves1 and len(moves) > 3:
                sector, tile = player1.get_move(game, moves=moves[-2:], **args)
            else:
                sector, tile = player1.get_move(game, **args)
        else:
            if give_moves0 and len(moves) > 2:
                sector, tile = player0.get_move(game, moves=moves[-2:], **args)
            else:
                sector, tile = player0.get_move(game, **args)
        if timing:
            times.append(time()-time1)
        game.move(sector, tile)
        moves.append((sector, tile))
        if game.mover and isinstance(player0, (NetPlayer, ABPlayer)):
            evals.append(player0.t.v)
        elif not game.mover and isinstance(player1, (NetPlayer, ABPlayer)):
            evals.append(player1.t.v)
        else:
            evals.append(0.0)

        if verbose:
            print(f'{decode_dict[2-game.mover]} @ {sector}, {tile}')
            if evals:
                print(f'{decode_dict[2-game.mover]} eval: {evals[-1]}')
            game.draw()

        if game.result:
            evals[-1] = ((game.result + 1) % 3) - 1
            player0.resulted(game, moves)
            player1.resulted(game, moves)
            break

    if verbose:
        print('GAME OVER')
        if game.result == 3:
            print('Tie Game')
        else:
            print(f'{decode_dict[game.result]} Wins')

    if timing:
        return game.result, moves, evals, times
    else:
        return game.result, moves, evals
