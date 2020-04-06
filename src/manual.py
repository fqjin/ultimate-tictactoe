"""Play UTTT Manually"""
from engine import BigBoard, decode_dict


def play_manual():
    """Play manual game

    Will ask for sector and tile numbers
    To quit, give -1 for tile
    """
    game = BigBoard()
    game.draw()

    while True:
        while True:
            if len(game.sectors) > 1:
                sector = int(input('Sector: '))
            else:
                sector = game.sectors[0]
                print(f'Sector {sector}')
            tile = int(input('Tile: '))
            if tile == -1:
                print('QUIT')
                return game

            try:
                game.move(sector, tile)
                break
            except ValueError:
                print('Illegal Move')

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
    g = play_manual()
