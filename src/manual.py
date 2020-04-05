"""Play UTTT Manually"""
import numpy as np
from random import randrange
from engine import BigBoard, decode_dict


def play_manual(auto_random=False, verbose=True, moves=81):
    game = BigBoard()
    if verbose:
        game.draw()

    for _ in range(moves):
        while True:
            if len(game.sectors) > 1:
                if auto_random:
                    sector = randrange(9)
                else:
                    sector = int(input('Sector: '))
            else:
                sector = game.sectors[0]
                if not auto_random:
                    print(f'Sector {sector}')
            if auto_random:
                tile = randrange(9)
            else:
                tile = int(input('Tile: '))
                if tile == 'q':
                    break

            try:
                game.move(sector, tile)
                break
            except ValueError:
                if not auto_random:
                    print('Illegal Move')

        if verbose:
            print(f'{decode_dict[2-game.mover]} @ {sector}, {tile}')
            game.draw()

        if game.result:
            print('GAME OVER')
            if game.result == 3:
                print('Tie Game')
            else:
                print(f'{decode_dict[game.result]} Wins')
            break
    return game


if __name__ == '__main__':
    # game = play_manual(auto_random=True)
    results = [play_manual(auto_random=True, verbose=False) for _ in range(100)]
    print(results)
    print(np.histogram(results, bins=3))
