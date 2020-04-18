import tkinter as tk
import tkinter.font as tkFont
import numpy as np
from players import BasePlayer, BigBoard
from engine import bit2board_table, decode_dict


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
        self.root = tk.Tk()
        self.root.geometry(f"+{self.x}+{self.y}")
        font = tkFont.Font(root=self.root, family='Helvetica', size=14, weight=tkFont.BOLD)
        frame = tk.Frame(self.root)
        frame.pack(fill='both', expand=1)
        boards = [bit2board_table[b] for b in board.bits]
        buttons = {}
        for sector, tile in np.ndindex(9, 9):
            if board.states[sector]:
                color = color_dict[board.states[sector]]
                state = tk.DISABLED
            else:
                # light yellow
                color = '#ffff99' if sector in board.sectors else 'white'
                if sector not in board.sectors or boards[sector][tile]:
                    state = tk.DISABLED
                else:
                    state = tk.NORMAL
            if moves:
                if (sector, tile) == moves[-1]:
                    color = '#ff9966'  # light orange
            b = tk.Button(frame,
                          text=decode_dict[boards[sector][tile]],
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


def print_result(result, moves):
    if result == 3:
        print('Tie Game')
    else:
        print(f'{decode_dict[result]} Wins')
    print(moves)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=100)
    parser.add_argument('--v_mode', type=int, default=1)
    parser.add_argument('--noise', type=int, default=0)
    args = parser.parse_args()

    import torch
    from network import UTTTNet
    from play import play
    from net_player import NetPlayer

    name = '350_3500bs2048lr0.1d0.001e5'
    print(f'Using {name} network')
    string1 = f'Using {args.nodes} nodes in '
    string2 = 'V mode ' if args.v_mode else 'N mode '
    string3 = 'with noise' if args.noise else 'without noise'
    print(string1 + string2 + string3)

    m = UTTTNet()
    m.load_state_dict(torch.load(f'../models/{name}.pt',
                                 map_location='cpu'))
    m = m.eval()
    p1 = NetPlayer(nodes=args.nodes, v_mode=args.v_mode, model=m, noise=args.noise)
    p2 = GuiPlayer(x=600)

    print_result(*play(p1, p2))
    print_result(*play(p2, p1))
