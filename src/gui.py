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

    def get_move(self, board: BigBoard, moves=None):
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
