from engine import BigBoard, decode_dict
from players import BasePlayer
from time import time


def play(player0: BasePlayer,
         player1: BasePlayer,
         game=None,
         verbose=False,
         press_enter=False,
         give_moves0=True,
         give_moves1=True,
         temp=None,
         ):
    if game is None:
        game = BigBoard()
    moves = []
    if verbose:
        game.draw()

    while True:
        if press_enter:
            input()

        if temp and len(moves) < temp[0]:
            args = {'invtemp': temp[1]}
        else:
            args = {}
        # time1 = time()
        # TODO: Rewrite keep_tree logic better
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
        # print(f'Time elapsed: {time()-time1:.2f}')
        game.move(sector, tile)
        moves.append((sector, tile))

        if verbose:
            print(f'{decode_dict[2-game.mover]} @ {sector}, {tile}')
            game.draw()

        if game.result:
            player0.resulted(game, moves)
            player1.resulted(game, moves)
            break

    if verbose:
        print('GAME OVER')
        if game.result == 3:
            print('Tie Game')
        else:
            print(f'{decode_dict[game.result]} Wins')

    return game.result, moves
