from collections import defaultdict

import numpy as np


class DistCache:
    def __init__(self):
        """
        Args:
            n: Number of slots for the cache
        """
        self.cache = defaultdict(lambda: 0)

    def log(self, **key_vals):
        """
        Args:
            slots: ids for the array
            **key_vals:
        """
        for k, v in key_vals.items():
            count = self.cache[k + '@counts'] + 1
            self.cache[k + '@counts'] = count
            self.cache[k] = v + (count - 1) * self.cache[k]
            self.cache[k] /= count

    def get_summary(self):
        ret = {
            k: v
            for k, v in self.cache.items()
            if not k.endswith("@counts")
        }
        self.cache.clear()
        return ret


if __name__ == '__main__':
    cl = DistCache()
    lin_vel = np.ones((11, 11))
    ang_vel = np.zeros((5, 5))
    cl.log(lin_vel=lin_vel, ang_vel=ang_vel)
    lin_vel = np.zeros((11, 11))
    ang_vel = np.zeros((5, 5))
    cl.log(lin_vel=lin_vel, ang_vel=ang_vel)
    print(cl.get_summary())


class SlotCache:
    def __init__(self, n):
        """
        Args:
            n: Number of slots for the cache
        """
        self.n = n
        self.cache = defaultdict(lambda: np.zeros([n]))

    def log(self, slots=None, **key_vals):
        """
        Args:
            slots: ids for the array
            **key_vals:
        """
        if slots is None:
            slots = range(self.n)

        for k, v in key_vals.items():
            counts = self.cache[k + '@counts'][slots] + 1
            self.cache[k + '@counts'][slots] = counts
            self.cache[k][slots] = v + (counts - 1) * self.cache[k][slots]
            self.cache[k][slots] /= counts

    def get_summary(self):
        ret = {
            k: v
            for k, v in self.cache.items()
            if not k.endswith("@counts")
        }
        self.cache.clear()
        return ret


if __name__ == '__main__':
    cl = SlotCache(100)
    reset_env_ids = [2, 5, 6]
    lin_vel = [0.1, 0.5, 0.8]
    ang_vel = [0.4, -0.4, 0.2]
    cl.log(reset_env_ids, lin_vel=lin_vel, ang_vel=ang_vel)

    cl.log(lin_vel=np.ones(100))
