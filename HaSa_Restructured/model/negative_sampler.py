import random
from collections import OrderedDict
from typing import Dict, Set

import numpy as np
import torch
import torch.nn.functional as F

from ..setting.logger_config import logger
from ..utils.dict_hub import get_entity_dict, get_train_triplet_dict, get_link_graph
from ..utils.doc import encode_entities
from ..utils.utils import get_model_obj


class HaSaNegativeSampler:
    """Embedding-bank based negative sampling of HaSa.

    * bank        : (num_entities x em_dim) stale embeddings, initialised with the current encoder and
                    refreshed every step with the in-batch head / tail / hard / false-negative vectors.
    * hard negs   : top-scoring entities for the (h, r) query that are not known true tails.
    * hardness p  : softmax over L2-normalised scores of [tails | heads | hard negs]; p[i, i] = 1, other known
                    true tails of (h, r) are pushed to ~0 (method hasa / hasa_hard_bias).
    * false negs  : (method hasa) hard negatives that lie within 2 hops of the head, padded with random 2-hop
                    neighbours (or known true tails); q = softmax(normalize(score)).
    * hasa_wohard_bias: random negatives, no weights.
    """

    TWO_HOP = 2
    CACHE_SIZE = 50000
    CHUNK = 1024

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.method = args.method
        self.num_hard = args.num_hard_neg
        self.num_false = args.num_false_neg
        self.rng = random.Random(args.seed)
        self.np_rng = np.random.RandomState(args.seed)

        self.entity_dict = get_entity_dict()
        self.num_entities = len(self.entity_dict)
        self.train_triplets = get_train_triplet_dict()
        self.link_graph = get_link_graph() if self.method == 'hasa' else None

        self.bank = None
        self._near_cache: 'OrderedDict[str, Set[int]]' = OrderedDict()

    # ------------------------------------------------------------------ bank
    @torch.no_grad()
    def init_bank(self, model) -> None:
        raw = get_model_obj(model)
        was_training = raw.training
        raw.eval()

        chunks = []
        for start in range(0, self.num_entities, self.CHUNK):
            enc = encode_entities(list(range(start, min(start + self.CHUNK, self.num_entities))))
            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                vec = raw.embed_entities(enc['token_ids'].to(self.device), enc['mask'].to(self.device))
            chunks.append(vec.float())

        self.bank = torch.cat(chunks, dim=0)
        raw.train(was_training)
        logger.info(f'Initialised embedding bank: {tuple(self.bank.shape)}')

    @torch.no_grad()
    def update_bank(self, batch_dict: Dict, output_dict: Dict) -> None:
        """Write the (train-mode) embeddings produced in this step back into the bank."""
        self.bank[batch_dict['head_idx']] = output_dict['head_vector'].detach().float()
        self.bank[batch_dict['tail_idx']] = output_dict['tail_vector'].detach().float()
        self.bank[batch_dict['hard_idx'].reshape(-1)] = output_dict['hard_vectors'].detach().float()
        if 'false_vectors' in output_dict:
            self.bank[batch_dict['false_idx'].reshape(-1)] = output_dict['false_vectors'].detach().float()

    # -------------------------------------------------------------- helpers
    def _tokens(self, indices, prefix: str) -> Dict[str, torch.Tensor]:
        enc = encode_entities(indices)
        return {
            f'{prefix}_token_ids': enc['token_ids'].to(self.device, non_blocking=True),
            f'{prefix}_mask': enc['mask'].to(self.device, non_blocking=True),
            f'{prefix}_token_type_ids': enc['token_type_ids'].to(self.device, non_blocking=True),
        }

    def _near(self, head_id: str) -> Set[int]:
        """Entity indices within 2 hops of the head (head excluded), LRU cached."""
        if head_id in self._near_cache:
            self._near_cache.move_to_end(head_id)
            return self._near_cache[head_id]

        near = set(self.link_graph.get_n_hop_entity_indices(
            head_id, self.entity_dict, n_hop=self.TWO_HOP))
        near.discard(self.entity_dict.entity_to_idx(head_id))

        self._near_cache[head_id] = near
        if len(self._near_cache) > self.CACHE_SIZE:
            self._near_cache.popitem(last=False)
        return near

    def _mine_hard(self, gx: torch.Tensor, exs, tail_idx) -> torch.Tensor:
        scores = gx @ self.bank.t()                                    # a x N
        rows, cols = [], []
        for i, ex in enumerate(exs):
            known = [self.entity_dict.entity_to_idx(e)
                     for e in self.train_triplets.get_neighbors(ex.head_id, ex.relation)]
            known.append(tail_idx[i])
            rows.extend([i] * len(known))
            cols.extend(known)
        scores[torch.LongTensor(rows).to(self.device), torch.LongTensor(cols).to(self.device)] = float('-inf')
        return scores.topk(self.num_hard, dim=1).indices                # a x hard

    def _pick_false(self, exs, hard_idx_cpu: torch.Tensor) -> torch.Tensor:
        false_idx = torch.empty(len(exs), self.num_false, dtype=torch.long)
        for i, ex in enumerate(exs):
            near = self._near(ex.head_id)
            chosen = [h for h in hard_idx_cpu[i].tolist() if h in near]
            need = self.num_false - len(chosen)
            if need > 0:
                if len(near) >= need:
                    chosen += self.rng.sample(list(near), need)
                else:
                    known = [self.entity_dict.entity_to_idx(e)
                             for e in self.train_triplets.get_neighbors(ex.head_id, ex.relation)]
                    chosen += self.rng.choices(known, k=need)
            false_idx[i] = torch.LongTensor(chosen[:self.num_false])
        return false_idx

    # ----------------------------------------------------------------- main
    @torch.no_grad()
    def sample(self, model, batch_dict: Dict) -> Dict:
        """Return extra batch entries: negative tokens (model inputs) + indices / weights (loss inputs)."""
        raw = get_model_obj(model)
        exs = batch_dict['batch_data']
        a = len(exs)
        ar = torch.arange(a, device=self.device)

        e2i = self.entity_dict.entity_to_idx
        tail_list = [e2i(ex.tail_id) for ex in exs]
        head_list = [e2i(ex.head_id) for ex in exs]
        tail_idx = torch.LongTensor(tail_list).to(self.device)
        head_idx = torch.LongTensor(head_list).to(self.device)
        extra = {'tail_idx': tail_idx, 'head_idx': head_idx}

        if self.method == 'hasa_wohard_bias':
            hard_idx = torch.from_numpy(
                self.np_rng.randint(self.num_entities, size=(a, self.num_hard))).to(self.device)
            extra['hard_idx'] = hard_idx
            extra.update(self._tokens(hard_idx.reshape(-1).tolist(), 'hard'))
            return extra

        # (h, r) query vectors in eval mode (no dropout), as in the original sampler
        was_training = raw.training
        raw.eval()
        with torch.cuda.amp.autocast(enabled=self.args.use_amp):
            gx = raw.encode_hr(batch_dict['head_token_ids'], batch_dict['head_mask'],
                               batch_dict['relation_token_ids'], batch_dict['relation_mask'])
        raw.train(was_training)
        gx = gx.float()

        hard_idx = self._mine_hard(gx, exs, tail_list)
        extra['hard_idx'] = hard_idx
        extra.update(self._tokens(hard_idx.reshape(-1).tolist(), 'hard'))

        # hardness weights over [tails | heads | hard negatives]
        cand = self.bank[torch.cat([tail_idx, head_idx, hard_idx.reshape(-1)])]
        p = F.normalize(gx @ cand.t(), dim=1)
        p[ar, ar] = -10
        p[:, :a].masked_fill_(~batch_dict['triplet_mask'], -10)      # other true tails of (h, r) in the batch
        prob = F.softmax(p, dim=1)
        prob[ar, ar] = 1
        extra['prob'] = prob

        if self.method == 'hasa':
            false_idx = self._pick_false(exs, hard_idx.cpu()).to(self.device)
            extra['false_idx'] = false_idx
            extra.update(self._tokens(false_idx.reshape(-1).tolist(), 'false'))

            f_scores = torch.einsum('ad,akd->ak', gx, self.bank[false_idx])
            extra['f_prob'] = F.softmax(F.normalize(f_scores, dim=1), dim=1)

        return extra
