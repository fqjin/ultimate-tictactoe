import tkinter as tk
import tkinter.font as tkFont
import numpy as np
from players import BasePlayer
from engine import decode_dict, BigBoard


coords_dict = {}
for sector, tile in np.ndindex(9, 9):
    coords_dict[(sector, tile)] = (3*(sector//3)+tile//3, 3*(sector%3)+tile%3)

color_dict = {
    1: '#99ccff',  # light blue
    2: '#ff9999',  # light red
    3: '#99ff99',  # light green
}


class GuiPlayer(BasePlayer):
    def __init__(self, x=0, y=0):
        self.return_value = None
        self.root = None
        self.x = x
        self.y = y

    def pressed(self, sector, tile):
        self.return_value = (sector, tile)
        self.root.destroy()

    def get_move(self, board: BigBoard, moves=None, invtemp=None):
        # TODO: Refactor board: BigBoard -> bigboard: BigBoard
        self.root = tk.Tk()
        self.root.geometry(f"+{self.x}+{self.y}")
        self.root.title(f'UTTT Gui Player - {decode_dict[board.mover+1]} to move')
        font = tkFont.Font(root=self.root, family='Helvetica', size=14, weight=tkFont.BOLD)
        frame = tk.Frame(self.root)
        frame.pack(fill='both', expand=1)
        buttons = {}
        for sector, tile in np.ndindex(9, 9):
            if board.states[sector]:
                color = color_dict[board.states[sector]]
                state = tk.DISABLED
            else:
                # light yellow
                color = '#ffff99' if sector in board.sectors else 'white'
                if sector not in board.sectors or board.boards[sector][tile]:
                    state = tk.DISABLED
                else:
                    state = tk.NORMAL
            if moves:
                if (sector, tile) == moves[-1]:
                    color = '#ff9966'  # light orange
            b = tk.Button(frame,
                          text=decode_dict[board.boards[sector][tile]],
                          command=lambda sector=sector, tile=tile: self.pressed(sector, tile),
                          state=state,
                          disabledforeground='black',
                          background=color,
                          height=2,
                          width=4,
                          font=font,
                          )
            rr, cc = coords_dict[(sector, tile)]
            b.grid(row=rr, column=cc)
            buttons[(sector, tile)] = b
        frame.grid_columnconfigure(3, weight=1)
        frame.grid_rowconfigure(3, weight=1)
        self.root.mainloop()

        return self.return_value

    def resulted(self, board: BigBoard, moves):
        """Show the finished game board in gui"""
        self.get_move(board, moves)


def print_result(result, moves, evals):
    if result == 3:
        print('Tie Game')
    else:
        print(f'{decode_dict[result]} Wins')
    print(moves)
    print(evals)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=100, 
                        help='Number of nodes to search. Default 100')
    parser.add_argument('--v_mode', type=int, default=1, 
                        help='Select move with best value rather than visits. Default 1')
    parser.add_argument('--noise', type=int, default=1, 
                        help='Add Dirichlet noise to policy. Default 0')
    args = parser.parse_args()

    from play import play
    from net_player import BatchNetPlayer
    from selfplay import load_model

    string1 = f'Using {args.nodes} nodes in '
    string2 = 'V mode ' if args.v_mode else 'N mode '
    string3 = 'with noise' if args.noise else 'without noise'
    print(string1 + string2 + string3)

    m = load_model()
    p1 = GuiPlayer(x=600)
    p2 = BatchNetPlayer(nodes=args.nodes, v_mode=args.v_mode, model=m, noise=args.noise)

    print_result(*play(p1, p2))
    print_result(*play(p2, p1))
    # from stats import make_stats
    # make_stats(p1, p2, 'Felix', name, 'Felix_vs_net30000',
    #            num=4, temp=(5, 1.0), save=True)
