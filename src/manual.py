"""Play UTTT Manually"""
from engine import *


if __name__ == '__main__':
    game = BigBoard()
    game.draw()

    while True:
        print(f'Mover: {decode_dict[game.mover+1]}')
        if len(game.sectors) > 1:
            sector = int(input('Sector: '))
        else:
            sector = game.sectors[0]
            print(f'Sector {sector}')
        tile = int(input('Tile: '))

        try:
            game.move(sector, tile)
            game.draw()
            print(game.states)
            print(game.secret_states)
        except ValueError:
            print('Illegal Move')

        if game.result:
            print('GAME OVER')
            if game.result == 3:
                print('Tie Game')
            elif game.result == 1:
                print(f'{decode_dict[game.result]} Wins')
            break
