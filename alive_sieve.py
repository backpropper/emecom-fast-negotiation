import torch


class AliveSieve(object):
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self.alive_mask = torch.ones(batch_size, dtype=torch.uint8, device=device)
        self.alive_idxes = self.mask_to_idxes(self.alive_mask)
        self.out_idxes = self.alive_idxes.clone()

    @staticmethod
    def mask_to_idxes(mask):
        return mask.view(-1).nonzero().long().view(-1)

    def mark_dead(self, dead_mask):
        if dead_mask.max() == 0:
            return
        dead_idxes = self.mask_to_idxes(dead_mask)
        self.alive_mask[dead_idxes] = 0
        self.alive_idxes = self.mask_to_idxes(self.alive_mask)

    def get_dead_idxes(self):
        dead_mask = 1 - self.alive_mask
        return dead_mask.nonzero().long().view(-1)

    def any_alive(self):
        return self.alive_mask.max() == 1

    def all_dead(self):
        return self.alive_mask.max() == 0

    def set_dead_global(self, target, v):
        dead_idxes = self.get_dead_idxes()
        if len(dead_idxes) == 0:
            return
        target[self.out_idxes[dead_idxes]] = v

    def self_sieve_(self):
        self.out_idxes = self.out_idxes[self.alive_idxes]
        self.batch_size = self.alive_mask.int().sum()
        self.alive_mask = torch.ones(self.batch_size, dtype=torch.uint8, device=self.device)
        self.alive_idxes = self.mask_to_idxes(self.alive_mask)

    def sieve_tensor(self, t):
        return t[self.alive_idxes]

    def sieve_list(self, alist):
        return [alist[b] for b in self.alive_idxes]


class SievePlayback(object):
    def __init__(self, alive_masks, device):
        self.alive_masks = alive_masks
        self.device = device

    def __iter__(self):
        batch_size = self.alive_masks[0].size()[0]
        global_idxes = torch.ones(batch_size, dtype=torch.uint8, device=self.device).nonzero().long().view(-1)
        T = len(self.alive_masks)
        for t in range(T):
            self.batch_size = len(global_idxes)
            yield t, global_idxes
            mask = self.alive_masks[t]
            if mask.max() == 0:
                return
            global_idxes = global_idxes[mask.nonzero().long().view(-1)]
