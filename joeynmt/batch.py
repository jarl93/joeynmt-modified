# coding: utf-8

"""
Implementation of a mini-batch.
"""


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, torch_batch, pad_index, use_cuda=False, use_tpu=False):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        :param use_tpu:
        """
        self.src, self.src_lengths = torch_batch.src
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.nseqs = self.src.size(0)
        self.trg_input = None
        self.trg = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None
        self.use_cuda = use_cuda
        self.use_tpu = use_tpu

        if hasattr(torch_batch, "trg"):
            trg, trg_lengths = torch_batch.trg
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]
            self.trg_lengths = trg_lengths
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]
            # we exclude the padded areas from the loss computation
            self.trg_mask = (self.trg_input != pad_index).unsqueeze(1)
            self.ntokens = (self.trg != pad_index).data.sum().item()

        if use_cuda:
            if not use_tpu:
                self._make_cuda()
        if use_tpu:
            if not use_cuda:
                self._make_tpu()

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()

        if self.trg_input is not None:
            self.trg_input = self.trg_input.cuda()
            self.trg = self.trg.cuda()
            self.trg_mask = self.trg_mask.cuda()

    def _make_tpu(self):
        """
        Move the batch to TPU

        :return:
        """
        #device = torch_xla.core.xla_model.xla_device()
        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)

        if self.trg_input is not None:
            self.trg_input = self.trg_input.to(device)
            self.trg = self.trg.to(device)
            self.trg_mask = self.trg_mask.to(device)

    def sort_by_src_lengths(self):
        """
        Sort by src length (descending) and return index to revert sort

        :return:
        """
        _, perm_index = self.src_lengths.sort(0, descending=True)
        rev_index = [0]*perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        sorted_src_lengths = self.src_lengths[perm_index]
        sorted_src = self.src[perm_index]
        sorted_src_mask = self.src_mask[perm_index]
        if self.trg_input is not None:
            sorted_trg_input = self.trg_input[perm_index]
            sorted_trg_lengths = self.trg_lengths[perm_index]
            sorted_trg_mask = self.trg_mask[perm_index]
            sorted_trg = self.trg[perm_index]

        self.src = sorted_src
        self.src_lengths = sorted_src_lengths
        self.src_mask = sorted_src_mask

        if self.trg_input is not None:
            self.trg_input = sorted_trg_input
            self.trg_mask = sorted_trg_mask
            self.trg_lengths = sorted_trg_lengths
            self.trg = sorted_trg

        if self.use_cuda:
            if not self.use_tpu:
                self._make_cuda()
        if self.use_tpu:
            if not self.use_cuda:
                self._make_tpu()

        return rev_index
