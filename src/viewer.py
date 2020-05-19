"""Gui game viewer"""
import numpy as np
import tkinter as tk
import tkinter.font as tkFont
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from engine import decode_dict, BigBoard
from gui import coords_dict, color_dict


class GameViewer:
    # TODO: Load game dialog
    # TODO: Playout lines
    def __init__(self, moves, evals=None, name='', x=0, y=0):
        self.moves = moves.tolist()
        if evals is not None:
            self.evals = evals.tolist()
        else:
            self.evals = [0.0] * len(moves)
        self.x = x
        self.y = y

        self.boards = [BigBoard()]
        for m in moves:
            new_board = self.boards[-1].copy()
            new_board.move(*m)
            self.boards.append(new_board)
        self.moves.insert(0, None)
        self.evals.append(self.evals[-1])
        self.zeros = np.zeros(len(self.evals))
        self.arange = np.arange(len(self.evals))
        self.evals0 = self.evals[::2]
        self.arange0 = self.arange[::2]
        self.evals1 = self.evals[1::2]
        self.arange1 = self.arange[1::2]

        self.root = tk.Tk()
        self.root.geometry(f"+{self.x}+{self.y}")
        self.root.title('UTTT Game Analysis')
        self.font = tkFont.Font(root=self.root, family='Helvetica',
                                size=14, weight=tkFont.BOLD)

        self.frame = tk.Frame(self.root)
        self.frame_t = tk.Frame(self.root)
        self.frame_p = tk.Frame(self.root)
        self.frame_s = tk.Frame(self.root)
        self.frame.grid(row=0, column=0, rowspan=5)
        self.frame_t.grid(row=0, column=1)
        self.frame_p.grid(row=1, column=1, rowspan=3)
        self.frame_s.grid(row=4, column=1)

        self.label = tk.Label(self.frame_t, text=name, font=self.font)
        self.label.pack()

        self.slider = tk.Scale(self.frame_s,
                               to=len(self.boards) - 1,
                               orient=tk.HORIZONTAL,
                               length=400,
                               command=lambda i: [
                                   self.plot_eval(int(i)),
                                   self.draw_board(int(i))
                               ]
                               )
        idx = 1
        self.slider.set(idx)
        self.slider.pack(fill=tk.X)

        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.fig.set_tight_layout(True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self.frame_p)
        self.canvas.get_tk_widget().pack()
        self.plot_eval(idx)

        self.buttons = {}
        for sector, tile in np.ndindex(9, 9):
            b = tk.Button(self.frame,
                          text='',
                          state=tk.DISABLED,
                          disabledforeground='black',
                          height=2,
                          width=4,
                          font=self.font,
                          )
            rr, cc = coords_dict[(sector, tile)]
            b.grid(row=rr, column=cc)
            self.buttons[(sector, tile)] = b
        self.draw_board(idx)

        self.root.mainloop()

    def plot_eval(self, idx):
        self.ax.clear()
        self.ax.plot(self.arange0, self.evals0)
        self.ax.plot(self.arange1, self.evals1)
        self.ax.plot(self.zeros, zorder=0)
        self.ax.plot([idx, idx], [-1.0, 1.0], zorder=0)
        self.ax.set_ylabel('Eval')
        self.ax.set_xlabel('Move')
        self.ax.set_xlim([0, self.arange[-1]])
        self.ax.set_ylim([-1.0, 1.0])
        self.canvas.draw()

    def draw_board(self, idx):
        move = self.moves[idx]
        board = self.boards[idx]
        for sector, tile in np.ndindex(9, 9):
            if board.states[sector]:
                color = color_dict[board.states[sector]]
            else:
                # light yellow
                color = '#ffff99' if sector in board.sectors else 'white'
            if move:
                if sector == move[0] and tile == move[1]:
                    color = '#ff9966'  # light orange
            b = self.buttons[(sector, tile)]
            b.configure(text=decode_dict[board.boards[sector][tile]],
                        background=color,
                        )


if __name__ == '__main__':
    name = 'net5000_vs_net15000_10k_game0'
    game = np.load('../misc/' + name + '.npz')
    GameViewer(game['moves'], game['evals'], name)
