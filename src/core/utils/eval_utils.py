
class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.history = []
        self.last = None
        self.val = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.last = self.mean()
        self.history.append(self.last)
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def mean(self):
        if self.count == 0:
            return 0.
        return self.sum / self.count
    
class Averager:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, val, nrep = 1):
        self._val = val
        self._sum += val * nrep
        self._count += nrep
        self._avg = self._sum / self._count

    @property
    def val(self): return self._val
    @property
    def avg(self): return self._avg
    @property
    def sum(self): return self._sum
    @property
    def count(self): return self._count