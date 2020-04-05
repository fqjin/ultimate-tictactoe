from engine import BigBoard, decode_dict
from players import BasePlayer


def play(player0: BasePlayer,
         player1: BasePlayer,
         game=None,
         verbose=True,
         press_enter=True):
    if game is None:
        game = BigBoard()
    moves = []
    if verbose:
        game.draw()

    while True:
        if press_enter:
            input()

        if game.mover:
            sector, tile = player1.get_move(game)
        else:
            sector, tile = player0.get_move(game)
        game.move(sector, tile)
        moves.append((sector, tile))

        if verbose:
            print(f'{decode_dict[2-game.mover]} @ {sector}, {tile}')
            game.draw()

        if game.result:
            break

    if verbose:
        print('GAME OVER')
        if game.result == 3:
            print('Tie Game')
        else:
            print(f'{decode_dict[game.result]} Wins')

    return game.result, moves
