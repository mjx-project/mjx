from typing import List


def rankings(tens) -> List[int]:
    assert len(tens) == 4
    order = list(range(4))
    order.sort(key=lambda i: tens[i] - i)
    order.reverse()
    ret = [0] * 4
    for i, p in enumerate(order):
        ret[p] = i

    # e.g tens = [20000, 35000, 35000, 10000]
    #     => order = [1, 2, 0, 3]
    #        ret   = [2, 0, 1, 3]

    return ret
