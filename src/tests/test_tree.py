import pytest
from tree import *


def test_tree_init():
    r = Root()
    x = BigBoard(sectors=(2, 3))
    t = Tree(x, parent=r, parent_index=0)

    assert t.v == 0.0
    assert t.P.sum() == pytest.approx(1.0, abs=1e-9)
    assert len(t.P) == 18
    assert len(t.N) == 18
    assert len(t.Q) == 18
    assert len(t.children) == 18
    assert t.children[0][0] == 2
    assert t.children[0][1] == 0
    assert 1 not in t.terminal


# def test_terminal():
#     r = Root()
#     x = BigBoard(sectors=(0,), mover=0)
#     t = Tree(x, parent=r, parent_index=0)
#     t.Q = np.random.randn(9)
#     assert t.check_terminal() == False
#     t.terminal += 1
#     assert t.check_terminal() == True
#     assert r.terminal[0] == 1
#     assert r.Q[0] == np.max(t.Q)


def test_explore():
    r = Root()
    x = BigBoard(sectors=(0,), mover=0)
    t = Tree(x, parent=r, parent_index=0)
    t.explore()
    assert t.N.sum() == 9+1


## Other Useful Test Code:
# from tree import *
# b = BigBoard()
# b.move(4,4)
# t = Tree(b, Root())
# for _ in range(20000):
#     t.explore()
#
# t.__dict__
#
# c = t
# while True:
#     c.board.draw()
#     c = c.goto(np.argmax(c.N))
#
# c.__dict__
# #
# while True:
#     print(f'{c.v}, {c.sign}, {c.N}, {np.max(c.Q/c.N)}')
#     c = c.parent
