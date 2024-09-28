from math import exp, log

import torch


class DiscreteDepth():
    def __init__(self,
                 dbound,
                 dbins,
                 dmode='UD',
                 log_shift=1.0):
        """
        Args:
            dbound (list[float, float]): Min and max depth
            dbins (int): Number of depth bins
            dmode (str): Depth discretization mode. Can be only choose from follows:
                - 'UD': Uniform discretization.
                - 'SID': Spacing-increasing discretization (log increasing).
                - 'LID': Linear-increasing discretization.
                Default: 'UD'.
            log_shift (float): Used by SID only.
        """
        self.dbound = dbound
        self.dmin = dbound[0]
        self.dmax = dbound[1]
        self.dbins = dbins
        self.dmode = dmode
        self.log_shift = log_shift

        self.ticks = torch.tensor(self._get_ticks(), dtype=torch.float)
        self.bin_depths = self._get_bin_depths()

    def _get_ticks(self):
        if self.dmode == 'UD':
            bin_size = (self.dmax - self.dmin) / self.dbins
            ticks = [self.dmin + i * bin_size for i in range(self.dbins + 1)]
        elif self.dmode == 'SID':
            # shift depth to adjust initial increase rate
            dmin = self.dmin + self.log_shift
            dmax = self.dmax + self.log_shift
            ticks = torch.tensor(
                [exp(log(dmin) + log(dmax / dmin) * i / self.dbins)
                    for i in range(self.dbins + 1)],
                dtype=torch.float)
        elif self.dmode == 'LID':
            first_bin_size = (self.dmax - self.dmin) / (
                self.dbins * (self.dbins + 1) / 2)
            ticks = [self.dmin + first_bin_size * (1 + i) * i / 2
                for i in range(self.dbins + 1)]
        else:
            raise NotImplementedError
        
        return ticks
    
    def _get_bin_depths(self):
        if self.dmode == 'SID':
            depths = (self.ticks[:-1] + self.ticks[1:]) / 2 - self.log_shift
        else:
            depths = (self.ticks[:-1] + self.ticks[1:]) / 2
        
        return depths
    
    def get_indices(self, depth_map, target=True):
        """
        Args:
            target (bool): Whether the depth bins indices will be used for
                a target tensor in loss comparison. Default: True.
        """
        if self.dmode == 'UD':
            bin_size = (self.dmax - self.dmin) / self.dbins
            indices = (depth_map - self.dmin) / bin_size
        elif self.dmode == 'SID':
            dmin = self.dmin + self.log_shift
            dmax = self.dmax + self.log_shift
            indices = self.dbins * \
                (torch.log(depth_map + self.log_shift) - log(dmin)) / \
                log((dmax / dmin))
        elif self.dmode == 'LID':
            first_bin_size = (self.dmax - self.dmin) / (
                self.dbins * (self.dbins + 1) / 2)
            indices = -0.5 + 0.5 * torch.sqrt(
                1 + 8 * (depth_map - self.dmin) / first_bin_size)
        else:
            raise NotImplementedError

        if target:
            # 0.0 in depth_map means no supervise or out of range
            indices = indices.floor().long().clip(min=0, max=self.dbins - 1)
        else:
            # Shift indices to the middle of a bin (only accurate when 'UD')
            indices = indices - 0.5

        return indices
