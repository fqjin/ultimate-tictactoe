from time import time
from engine import BigBoard
from tree import Tree, Root
from net_player import NetPlayer, BatchNetPlayer
from selfplay import load_model, device

# Warm up
g = BigBoard()
t = Tree(g, Root(), noise=True)
for _ in range(1000):
    t.explore()

# Tree
# Pre terminal merge: 2.99, 3.00, 3.03
# Postterminal merge: 3.19, 3.15, 3.21
g = BigBoard()
t = Tree(g, Root(), noise=False)
t0 = time()
for _ in range(10000):
    t.explore()
print(t.N)
print(time()-t0)

# NetTree
# Pre terminal merge: 3.92, 3.87, 3.89
# Postterminal merge: 3.89, 3.89, 3.89
model = load_model()
g = BigBoard()
p = NetPlayer(nodes=1000, model=model, device=device, noise=False)
t0 = time()
move = p.get_move(g)
print(time()-t0)
print(move)

# BatchNetTree
# Pre terminal merge: 9.07, 9.07, 9.22
# Postterminal merge: 9.36, 9.38, 9.43
g = BigBoard()
p = BatchNetPlayer(nodes=10000, model=model, device=device, noise=False)
t0 = time()
move = p.get_move(g)
print(time()-t0)
print(move)
