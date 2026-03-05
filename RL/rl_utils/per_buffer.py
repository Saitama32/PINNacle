import torch
import random
from collections import namedtuple
import math

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'action', 'reward', 'done', 'model_reward', 'opt_model_i'))
    

# ---------- Prioritized Experience Replay (proportional) ----------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.memory, self.prior, self.pos = [], [], 0

        # --- success replay ---
        # Порог по model_reward, выше которого терминальный переход считаем "успехом"
        self.success_threshold = 0.0
        # Набор индексов переходов, которые являются успешными терминалами (done == 1)
        self.success_indexes = set()

    def _warn_renorm(self, where: str, reason: str):
        print(f"WARNING[PER:{where}] fallback renorm triggered: {reason}")

    def __len__(self):
        return len(self.memory)

    def push(self, *args, priority=None, coeff=1.0):
        """
        args: (state, next_state, action, reward, done, model_reward, opt_model_i)
        """
        tr = Transition(*args)

        # --- базовый приоритет ---
        if priority is None:
            if self.prior:
                finite_prior = [x for x in self.prior if math.isfinite(x) and x > 0]
                base_p = max(finite_prior) if finite_prior else 1.0
                p = base_p * coeff
            else:
                p = 1.0
        else:
            p = float(priority)
        if (not math.isfinite(p)) or p <= 0:
            p = self.eps

        # --- индекс, в который пишем ---
        if len(self.memory) < self.capacity:
            idx = len(self.memory)
            self.memory.append(tr)
            self.prior.append(p)
        else:
            idx = self.pos
            self.memory[idx] = tr
            self.prior[idx] = p
            self.pos = (self.pos + 1) % self.capacity

        # --- обновляем success_indexes для этого индекса ---
        # считаем успешным терминалом: done == 1 и model_reward > success_threshold
        done = getattr(tr, "done", 0)
        model_reward = float(getattr(tr, "model_reward", 0.0))

        is_success = (done == 1) and (model_reward > self.success_threshold)

        if is_success:
            self.success_indexes.add(idx)
        else:
            # если в этом слоте раньше был успех, а теперь нет — убираем
            self.success_indexes.discard(idx)


    def sample(self, batch_size, beta=0.4, device='cpu'):
        pr = torch.tensor(self.prior, dtype=torch.float, device=device)
        pr = torch.nan_to_num(pr, nan=self.eps, posinf=1.0, neginf=self.eps).clamp_min(self.eps)
        probs = (pr + self.eps) ** self.alpha
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        if probs.sum() <= 0:
            self._warn_renorm("sample", "probs.sum() <= 0 after sanitize")
            probs = torch.ones_like(probs)
        probs = probs / probs.sum()

        idxs = torch.multinomial(
            probs, batch_size,
            replacement=len(self.memory) < batch_size
        )

        # w_j = (N * P(j))^{-β} / max_i w_i
        N = len(self.memory)
        weights = (N * probs[idxs]).pow(-beta)
        weights = (weights / weights.max()).float()

        batch = [self.memory[int(i)] for i in idxs]
        return batch, idxs, weights

    def sample_uniform(self, batch_size, device='cpu'):
        """
        Равномерная выборка без смещения (для warmup). Веса = 1.
        """
        N = len(self.memory)
        if N == 0:
            raise RuntimeError("Buffer is empty")
        idxs = torch.randint(0, N, (batch_size,), device=device)
        batch = [self.memory[int(i)] for i in idxs]
        weights = torch.ones(batch_size, dtype=torch.float, device=device)
        return batch, idxs, weights

    def update_priorities(self, idxs, new_p):
        for i, p in zip(idxs.tolist(), new_p.tolist()):
            p = float(p)
            if (not math.isfinite(p)) or p <= 0:
                p = self.eps
            self.prior[int(i)] = p
    
    # per_buffer.py  --- ДОБАВИТЬ внутрь класса PrioritizedReplayBuffer
    def _build_sequence_from_start(self, start_idx: int, L: int):
        """
        Собираем переходы [start_idx .. start_idx+L-1], обрываем на done
        и не даём вылезти за конец буфера. Без циклического wrap-around.
        """
        seq = []
        i = start_idx
        N = len(self.memory)
        steps = 0
        while i < N and steps < L:
            tr = self.memory[i]
            seq.append(tr)
            steps += 1
            # если эпизод закончился — выходим (не включаем следующий)
            if getattr(tr, "done", 0) != 0:
                break
            i += 1
        return seq
    
    def _build_sequence_ending_at(self, end_idx: int, L: int):
        """
        Строим последовательность, которая:
        - всегда заканчивается на end_idx (обычно done=1),
        - идёт назад максимум на L-1 шагов,
        - НИКОГДА не пересекает границу эпизода (т.е. не включает предыдущий done!=0).
        """
        seq_rev = []
        i = end_idx
        steps = 0
        N = len(self.memory)

        while i >= 0 and steps < L:
            tr = self.memory[i]

            # если это не самый правый шаг и у него done != 0,
            # значит мы дошли до конца предыдущего эпизода -> дальше не идём
            if steps > 0 and getattr(tr, "done", 0) != 0:
                break

            seq_rev.append(tr)
            steps += 1
            i -= 1

        seq_rev.reverse()
        return seq_rev
    

    def sample_sequences(self, batch_size: int, L: int, beta=None, uniform=False, device='cpu'):
        """
        Возвращает:
        - seqs: list[list[Transition]] длиной B, каждая — последовательность длиной ≤L,
        - idxs: Tensor[B] стартовых индексов (их и обновляем в update_priorities),
        - is_w: Tensor[B] importance-sampling веса.

        ВАЖНО:
        - стартовые индексы выбираются только среди нетерминальных переходов,
          у которых есть хотя бы один шаг вперёд (idx < N-1) -> цепочки не единичные.
        - никаких добиваний батча случайными терминалами.
        """
        N = len(self.memory)
        if N == 0:
            raise RuntimeError("Buffer is empty")

        # --- строим пул валидных стартовых индексов: done == 0 и есть следующий шаг ---
        valid_start_idxs = [
            i for i, tr in enumerate(self.memory[:-1])  # до N-1 включительно только N-2
            if getattr(tr, "done", 0) == 0
        ]
        # если вообще нет валидных стартов — fallback: позволяем всё как раньше
        no_valid_starts = (len(valid_start_idxs) == 0)

        if uniform:
            # --- РАВНОМЕРНЫЙ СЭМПЛИНГ ---
            if no_valid_starts:
                # всё плохо, берём как раньше: любые индексы
                idxs = torch.randint(0, N, (batch_size,), device=device)
            else:
                # выбираем только из valid_start_idxs, с повторениями при необходимости
                if len(valid_start_idxs) >= batch_size:
                    # без повторов можно, если пул большой
                    chosen = random.sample(valid_start_idxs, batch_size)
                else:
                    # пул маленький — разрешаем повторы
                    print("⚠️ PrioritizedReplayBuffer: uniform sampling with repeats due to small valid start pool")
                    chosen = [random.choice(valid_start_idxs) for _ in range(batch_size)]
                idxs = torch.tensor(chosen, dtype=torch.long, device=device)

            is_w = torch.ones(batch_size, dtype=torch.float, device=device)

        else:
            # --- PER СЭМПЛИНГ ПО СТАРТОВЫМ ЭЛЕМЕНТАМ ---
            pr = torch.tensor(self.prior, dtype=torch.float, device=device)
            pr = torch.nan_to_num(pr, nan=self.eps, posinf=1.0, neginf=self.eps).clamp_min(self.eps)
            probs = (pr + self.eps) ** self.alpha
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

            if no_valid_starts:
                # нет нетерминальных стартов -> классический PER по всем
                if probs.sum() <= 0:
                    self._warn_renorm("sample_sequences/main", "no_valid_starts and probs.sum() <= 0")
                    probs = torch.ones_like(probs)
                probs = probs / probs.sum()
            else:
                # обнуляем вероятность для НЕвалидных стартов
                mask = torch.zeros(N, dtype=torch.float, device=device)
                mask[valid_start_idxs] = 1.0
                probs = probs * mask
                # если вдруг все веса обнулились (на всякий случай) — fallback к исходным
                if probs.sum() <= 0:
                    self._warn_renorm("sample_sequences/masked", "masked probs sum to zero; fallback to unmasked")
                    probs = (pr + self.eps) ** self.alpha
                    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                    if probs.sum() <= 0:
                        self._warn_renorm("sample_sequences/masked", "unmasked probs sum to zero; fallback to uniform")
                        probs = torch.ones_like(probs)
                probs = probs / probs.sum()

            support = int((probs > 0).sum().item())
            replacement = support < batch_size
            idxs = torch.multinomial(probs, batch_size, replacement=replacement)

            assert beta is not None, "beta must be provided for PER sampling"
            # IS-веса считаем по ИСХОДНЫМ probs (как в классическом PER)
            base_probs = (pr + self.eps) ** self.alpha
            base_probs = torch.nan_to_num(base_probs, nan=0.0, posinf=0.0, neginf=0.0)
            if base_probs.sum() <= 0:
                self._warn_renorm("sample_sequences/isw", "base_probs sum to zero; fallback to uniform")
                base_probs = torch.ones_like(base_probs)
            base_probs = base_probs / base_probs.sum()
            weights = (N * base_probs[idxs]).pow(-beta)
            is_w = (weights / weights.max()).float()

        # --- сбор последовательностей ---
        seqs = [self._build_sequence_from_start(int(i), L) for i in idxs.tolist()]

        return seqs, idxs, is_w


    
    def sample_success_sequences(self, batch_size: int, L: int, device='cpu'):
        """
        Сэмплирует batch_size последовательностей длиной ≤L,
        КАЖДАЯ из которых заканчивается успешным терминальным переходом
        (индекс в self.success_indexes).

        Возвращает:
          - seqs: list[list[Transition]]
          - idxs: Tensor[batch_size] стартовых индексов последовательностей
          - is_w: Tensor[batch_size] весов (здесь просто 1.0)
        """
        if not self.success_indexes:
            # если пока нет ни одного успеха — просто fallback на uniform sequences
            seqs, idxs, is_w = self.sample_sequences(batch_size, L, beta=None, uniform=True, device=device)
            return seqs, idxs, is_w

        success_list = list(self.success_indexes)
        N_succ = len(success_list)

        seqs = []
        idxs = []

        # чтобы не зависнуть, если данные очень "кривые"
        max_tries_per_seq = 100

        for _ in range(batch_size):
            seq = None
            start_idx_for_this_seq = None

            for _try in range(max_tries_per_seq):
                end_idx = success_list[random.randint(0, N_succ - 1)]

                # Строим последовательность, заканчивающуюся этим success
                candidate = self._build_sequence_ending_at(end_idx, L)

                if len(candidate) <= 1:
                    continue

                seq = candidate

                # индекс первого элемента, зная end_idx и длину
                start_idx_for_this_seq = end_idx - (len(seq) - 1)
                start_idx_for_this_seq = max(start_idx_for_this_seq, 0)

                break  # выходим из цикла попыток, последовательность найдена

            if seq is None:
            # Не смогли найти нормальную success-цепочку.
            # Крайний случай: добиваем батч обычной последовательностью.
                fallback_seqs, fallback_idxs, _ = self.sample_sequences(
                    1, L, beta=None, uniform=True, device=device
                )
                seq = fallback_seqs[0]
                start_idx_for_this_seq = int(fallback_idxs[0])

            seqs.append(seq)
            idxs.append(start_idx_for_this_seq)

        idxs = torch.tensor(idxs, dtype=torch.long, device=device)
        is_w = torch.ones(batch_size, dtype=torch.float, device=device)

        return seqs, idxs, is_w



