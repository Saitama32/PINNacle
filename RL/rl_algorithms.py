# Separate file for RL algorithms (e.g., rl_algorithms.py)

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import math
from copy import copy
import matplotlib.pyplot as plt
from collections import defaultdict
from math import ceil
import statistics
from pathlib import Path
from RL.rl_utils.DQN_classes import DQN_optim, DQN_params
from comet_ml.integration.pytorch import watch
from RL.rl_utils.per_buffer import PrioritizedReplayBuffer, Transition, UniformReplayBuffer
from RL.rl_utils.per_offline import recalc_all_priorities_batched
from RL.rl_utils.logger import log_priority_to_comet
from RL.rl_utils.metrics import collect_policy_metrics_by_seq_position
from RL.rl_utils.informativity_metrics import collect_chain_mc_metrics


EPS_START = 0.5
EPS_END = 0.05
EPS_DECAY = 200
TAU = 0.01

class DQNAgent:
    def __init__(self, n_observation=None, n_action=None, optimizer_dict=None, lr=1e-3, gamma=0.98, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, memory_size=50000, batch_size=128, n_transitions_reinit = 2000, per_alpha =  0.6, per_beta0 = 0.4, device='cpu', exp=None,
                 warmup_updates: int = 200, recalc_batch_size: int = 32, success_frac = 0.2,
                 include_terminal_starts: bool = False,
                 use_prioritized_replay: bool = True,
                 use_soft_watkins: bool = True,
                 use_trust_region_masking: bool = True,
                 model_snapshot_dir="rl_model_snapshots"):
        self.n_observation = n_observation
        self.n_action = n_action
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.include_terminal_starts = include_terminal_starts
        self.use_prioritized_replay = use_prioritized_replay
        self.use_soft_watkins = use_soft_watkins
        self.use_trust_region_masking = use_trust_region_masking
        self.n_transitions_reinit = n_transitions_reinit
        self.steps_done = 0
        self.epsilon_step_offset = 0
        self.opt_count = 0
        self.opt_count_out = 0
        self.opt_count_for_reinit = 5
        self.optimizer_dict = optimizer_dict
        self.i2opt = {v: k for v, k in enumerate(optimizer_dict.keys())}
        uniq_params = list(set([x for xs in optimizer_dict.values() for x in xs]))
        self.i2params = {k: v for v, k in enumerate(uniq_params)}
        self.huberloss = nn.HuberLoss(reduction='none')
        self.opt_step = 0

        # e - greedly 
        self.slot_bootstrap_steps = 20     # первые N шагов нового запуска делаем повышенное ε
        self.slot_bootstrap_eps = 0.5

        # TD
        self.lambda_ = 0.95     # λ
        self.kappa  = 0.7      # tolerance κ (0=жёсткий Watkins)
        self.seq_len = 12   

        # --- TD-нормализация для параметров ---
        self.param_td_running_std = {}   # dict: key -> EMA(std)
        self.param_td_mom = 0.99
        self.param_td_eps = 1e-6

        # ---- Trust-region гиперпараметры ----
        self.tr_alpha = 2.0         # ширина «бокса» в σ TD-ошибки (2.0–3.0 ок)
        self.tr_eps   = 1e-6        # численная защита
        self.tr_mom   = 0.99        # EMA для бегущего std TD-ошибки
        self.td_running_std = 0.0   # буфер EMA(std(|δ|)) по батчам

        # PER
        self.per_alpha = per_alpha
        self.per_beta  = per_beta0
        self.per_beta_inc = (1.0 - per_beta0) / 100000.0
        if self.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(memory_size, alpha=per_alpha)
        else:
            self.replay_buffer = UniformReplayBuffer(memory_size)

        # --- Success replay (один буфер, логическая подселекция) ---
        # Доля последовательностей, которые берем из успешных эпизодов
        self.success_frac = success_frac         # например, 30% батча
        # Порог по model_reward, > которого done==1 считаем успехом
        self.success_reward_threshold = 0.0
        # прокидываем порог в буфер
        self.replay_buffer.success_threshold = self.success_reward_threshold

        #warmup
        self.warmup_updates_total = warmup_updates
        self.warmup_updates_done = 0
        self.warmup_active = self.use_prioritized_replay and warmup_updates > 0
        self.recalc_done = not self.use_prioritized_replay
        self.recalc_batch_size = recalc_batch_size
        self.transition_counter = 0


        self.device = device
        self.exp = exp
        self.model_snapshot_dir = Path(model_snapshot_dir)
        epsilon_and_warmap_params = {
            "slot_bootstrap_steps": self.slot_bootstrap_steps,
            "slot_bootstrap_eps": self.slot_bootstrap_eps,
            "warmup_updates": warmup_updates,
            "recalc_batch_size": recalc_batch_size,
            "EPS_START": EPS_START,
            "EPS_END": EPS_END,
            "EPS_DECAY": EPS_DECAY,
            "TAU": TAU,
            "use_prioritized_replay": self.use_prioritized_replay,
            "use_soft_watkins": self.use_soft_watkins,
            "use_trust_region_masking": self.use_trust_region_masking,
        }
        if self.exp is not None:
            self.exp.log_parameters(epsilon_and_warmap_params)

        self.model_optim = DQN_optim(len(self.i2opt)).to(device)
        self.model_params = DQN_params(self.optimizer_dict).to(device)
        # if self.exp is not None:
        #     watch(self.model_optim, log_step_interval = 200)
        #     watch(self.model_params, log_step_interval = 200)

        self.reinit_target()

        self.optimizer_opt = optim.Adam(self.model_optim.parameters(), lr=lr)
        self.optimizer_params = optim.Adam(self.model_params.parameters(), lr=lr)

    def reinit_target(self):
        self.target_model_optim = DQN_optim(len(self.i2opt)).to(self.device)
        self.target_model_params = DQN_params(self.optimizer_dict).to(self.device)
        for param in self.target_model_optim.parameters():
            param.requires_grad = False
        for param in self.target_model_params.parameters():
            param.requires_grad = False
        self.target_model_optim.load_state_dict(self.model_optim.state_dict())
        self.target_model_optim.eval()
        self.target_model_params.load_state_dict(self.model_params.state_dict())
        self.target_model_params.eval()

    def start_epsilon_schedule(self):
        """Start epsilon decay from the current global metric step."""
        self.epsilon_step_offset = self.steps_done

    def get_epsilon_threshold(self):
        interaction_steps = max(0, self.steps_done - self.epsilon_step_offset)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
            -1.0 * interaction_steps / EPS_DECAY
        )
        if interaction_steps < self.slot_bootstrap_steps:
            return self.slot_bootstrap_eps
        return eps_threshold

    def detach_transition(self, transition):
        def detach_item(item):
            if isinstance(item, torch.Tensor):
                return item.detach().clone()
            elif isinstance(item, tuple):
                return tuple(detach_item(subitem) for subitem in item)
            elif isinstance(item, dict):
                return {k: detach_item(v) for k, v in item.items()}
            elif isinstance(item, list): 
                return [detach_item(v) for v in item]
            else:
                return item

        return Transition(
            state=detach_item(transition.state),
            next_state=detach_item(transition.next_state),
            action=detach_item(transition.action),
            reward=detach_item(transition.reward),
            done=detach_item(transition.done),
            model_reward=detach_item(transition.model_reward),
            opt_model_i=detach_item(transition.opt_model_i)
        )
    
    def push_memory(self, rl_params, priority=None):
        tr = self.detach_transition(Transition(*rl_params))
        self.replay_buffer.push(
            tr.state, tr.next_state, tr.action, tr.reward, tr.done, tr.model_reward, tr.opt_model_i, coeff=1.5
        )

    def _stack_state(self, st):
        total = st['loss_total'].to(self.device)
        oper  = st['loss_oper'].to(self.device)
        bnd   = st['loss_bnd'].to(self.device)

        if 'delta' in st:
            delta = st['delta'].to(self.device)
        else:
            delta = torch.zeros_like(total)

        x = torch.stack((total, oper, bnd, delta), dim=0)   # (4,26,26)
        return x

    def _get_param_act_idx(self, action_i, pname):
        """
        Унифицируем разные форматы action:
        либо (optim_idx, {'epochs': idx, ...}), либо (optim_idx, epochs_idx, {param: idx})
        """
        if isinstance(action_i[1], dict):
            return int(action_i[1][pname])
        if pname == 'epochs':
            return int(action_i[1])
        return int(action_i[2][pname])

    # def deepcopy_replay_buffer_without_graph(self, buffer):
    #     clean_buffer = PrioritizedReplayBuffer(capacity=len(buffer.memory))
    #     for transition in buffer.memory:
    #         clean_buffer.push(*self.detach_transition(transition))
    #     return clean_buffer

    def _sample_sequences(self, batch_size, L, uniform: bool, beta=None):
        """
        Сэмплируем последовательности из буфера.
        - При uniform=True (warmup) берём только обычные sequence-выборки.
        - При uniform=False (основное обучение) мешаем:
            * часть батча из обычного PER (по приоритетам)
            * часть батча из успешных эпизодов (success_sequences),
              если такие есть и success_frac > 0.
        """
        rb = self.replay_buffer

        # --- Warmup: только uniform-выборка ---
        # PER warmup deliberately excludes success replay. In the no-PER
        # ablation uniform sampling is the main policy, so success replay is
        # still mixed below exactly as in the full agent.
        if uniform and self.use_prioritized_replay:
            seqs, idxs, is_w = rb.sample_sequences(
                batch_size,
                L,
                beta=None,
                uniform=True,
                device=self.device,
                include_terminal_starts=self.include_terminal_starts,
            )
            return seqs, idxs, is_w

        # --- Основной режим: PER + при необходимости success-эпизоды ---
        main_uniform = not self.use_prioritized_replay
        main_beta = None if main_uniform else beta

        if self.success_frac <= 0.0 or not rb.success_indexes:
            # если success-режим выключен или ещё нет успешных эпизодов
            seqs, idxs, is_w = rb.sample_sequences(
                batch_size,
                L,
                beta=main_beta,
                uniform=main_uniform,
                device=self.device,
                include_terminal_starts=self.include_terminal_starts,
            )
            return seqs, idxs, is_w

        # Сколько последовательностей взять из success-эпизодов
        n_succ = int(batch_size * self.success_frac)
        n_succ = max(1, n_succ)           # минимум одна
        n_succ = min(n_succ, batch_size)  # но не больше батча

        n_main = batch_size - n_succ
        if n_main <= 0:
            # крайний случай: весь батч из success
            n_main = 0
            n_succ = batch_size

        seqs_all = []
        idxs_all = []
        isw_all  = []

        # 1) Основная часть — обычный PER по стартовым индексам
        if n_main > 0:
            main_seqs, main_idxs, main_is_w = rb.sample_sequences(
                n_main,
                L,
                beta=main_beta,
                uniform=main_uniform,
                device=self.device,
                include_terminal_starts=self.include_terminal_starts,
            )
            seqs_all.extend(main_seqs)
            idxs_all.append(main_idxs.to(torch.long))
            isw_all.append(main_is_w.to(self.device))

        # 2) Success-последовательности (равномерно по success_indexes)
        if n_succ > 0:
            succ_seqs, succ_idxs, succ_is_w = rb.sample_success_sequences(
                n_succ,
                L,
                device=self.device,
                include_terminal_starts=self.include_terminal_starts,
            )
            seqs_all.extend(succ_seqs)
            idxs_all.append(succ_idxs.to(torch.long))
            isw_all.append(succ_is_w.to(self.device))

        # 3) Склеиваем индексы и веса в тензоры
        idxs_cat = torch.cat(idxs_all, dim=0)
        isw_cat  = torch.cat(isw_all, dim=0)

        # На всякий случай контролируем длину
        assert len(seqs_all) == batch_size, f"Expected {batch_size} seqs, got {len(seqs_all)}"

        return seqs_all, idxs_cat, isw_cat

    

    def _greedy_mask(self, s_batch, a_batch):
    # s_batch: Tensor[B, ..., 26,26], a_batch: LongTensor[B]
        with torch.no_grad():
            _, q_all = self.model_optim(s_batch)       # [B, A]
            a_star = q_all.argmax(dim=1)               # [B]
        return (a_batch == a_star)                     # [B] bool

    def _soft_watkins_targets(self, seq, gamma):
        states      = torch.stack([self._stack_state(tr.state)      for tr in seq])
        next_states = torch.stack([self._stack_state(tr.next_state) for tr in seq])
        actions     = torch.tensor([tr.action[0] for tr in seq], dtype=torch.long, device=self.device)
        rewards     = torch.tensor([tr.reward   for tr in seq], dtype=torch.float, device=self.device)
        dones       = torch.tensor([(tr.done!=0) for tr in seq], dtype=torch.bool, device=self.device)

        # --- ПРАВИЛЬНОЕ накапливание G^{(n)} ---
        Gn = []
        ret = torch.zeros((), device=self.device)
        pow_ = torch.tensor(1.0, device=self.device)            # = γ^0 на старте
        sum_w = torch.zeros((), device=self.device)
        for n in range(len(seq)):                               # n=0..N-1  => (n+1)-step
            ret = ret + pow_ * rewards[n]                       # += γ^n * r_{t+1+n}
            if not dones[n]:
                with torch.no_grad():
                    _, q_on_next  = self.model_optim(next_states[n:n+1])
                    a_star        = q_on_next.argmax(dim=1)
                    _, q_tg_next  = self.target_model_optim(next_states[n:n+1])
                    boot = q_tg_next.gather(1, a_star.view(-1,1)).squeeze()  
            else:
                boot = torch.zeros((), device=self.device)
            Gn.append(ret + (pow_ * gamma) * boot)              # + γ^{n+1} * boot
            pow_ = pow_ * gamma                                 # γ^{n+1} к следующему шагу

        greedy_mask = self._greedy_mask(states, actions)
        g_vals = torch.where(greedy_mask, torch.ones_like(rewards), torch.full_like(rewards, self.kappa))
        G_lambda = torch.zeros((), device=self.device)
        g_prefix = torch.tensor(1.0, device=self.device)
        for n in range(len(seq)):  # n=0..N-1  => (n+1)-step
            if n > 0:
                # включаем g для шага t+(n-1) — т.е. для промежуточных шагов после текущего
                g_prefix = g_prefix * g_vals[n-1]
            w_n = (1.0 - self.lambda_) * (self.lambda_ ** n) * g_prefix
            G_lambda = G_lambda + w_n * Gn[n]
            sum_w = sum_w + w_n

        G_lambda = G_lambda / sum_w.clamp_min(1e-3)
        return G_lambda.detach(), sum_w.detach()

    def _one_step_optimizer_targets(self, next_state, reward, done):
        """Historical one-step Double-DQN target used before soft Watkins."""
        with torch.no_grad():
            _, q_next_online = self.model_optim(next_state)
            next_actions = q_next_online.argmax(dim=1)
            _, q_next_target = self.target_model_optim(next_state)
            q_next = q_next_target.gather(1, next_actions.view(-1, 1)).squeeze(1)
            return reward + (1.0 - done) * self.gamma * q_next

    def _trust_region_keep_mask(self, q_sa, y_opt, state, action_o):
        """Return the optimizer-loss keep mask and TD scale diagnostics."""
        delta = (y_opt - q_sa).detach()
        sigma_batch = delta.std().clamp_min(self.tr_eps).item()
        self.td_running_std = (
            self.tr_mom * self.td_running_std
            + (1.0 - self.tr_mom) * sigma_batch
        )
        sigma = max(sigma_batch, self.td_running_std, self.tr_eps)
        sigma_t = torch.full_like(q_sa, fill_value=sigma)

        if not self.use_trust_region_masking:
            drop = torch.zeros_like(q_sa, dtype=torch.bool)
            return drop, torch.ones_like(q_sa), sigma_t, sigma

        with torch.no_grad():
            _, q_opt_target = self.target_model_optim(state)
        q_target_sa = q_opt_target.gather(1, action_o.view(-1, 1)).squeeze(1)
        gap = q_sa.detach() - q_target_sa
        too_far = gap.abs() > (self.tr_alpha * sigma_t)
        moving_away = torch.sign(gap) != torch.sign(q_sa.detach() - y_opt.detach())
        drop = too_far & moving_away
        return drop, (~drop).float(), sigma_t, sigma

    
    def optim_(self, iters=1):
        """
        PER + Double DQN + Dueling.
        За один вызов делает `iters` батч-обновлений из приоритезированного буфера.
        Возвращает два списка средних лоссов (голова оптимизатора и суммарно по параметрам).
        """
        loss_arr_optim_class, loss_arr_param = [], []
        all_rewards, all_dones = [], []
        model_reward_i_ar = []
        chain_mc_metric_values = defaultdict(list)

        for _ in range(iters):
            if len(self.replay_buffer) < self.batch_size:
                break

            if self.warmup_active:
                seqs, idxs, is_w = self._sample_sequences(self.batch_size, self.seq_len, uniform=True, beta=None)
                is_w = is_w.to(self.device)           # единицы
            else:
                seqs, idxs, is_w = self._sample_sequences(
                    self.batch_size,
                    self.seq_len,
                    uniform=not self.use_prioritized_replay,
                    beta=self.per_beta if self.use_prioritized_replay else None,
                )
                if self.use_prioritized_replay:
                    self.per_beta = min(1.0, self.per_beta + self.per_beta_inc)
                is_w = is_w.to(self.device)

            policy_position_metrics = collect_policy_metrics_by_seq_position(self, seqs)    

            first_trs = [seq[0] for seq in seqs]

            state, next_state, action, reward, done, model_reward, opt_model_i = zip(*[
                (tr.state, tr.next_state, tr.action, tr.reward, tr.done, tr.model_reward, tr.opt_model_i)
                for tr in first_trs
            ])

            B = len(first_trs)

            state  = torch.stack([self._stack_state(s)  for s in state])      # (B,2,26,26)
            next_state = torch.stack([self._stack_state(s2) for s2 in next_state])
            reward   = torch.tensor(reward, dtype=torch.float, device=self.device)              # (B,)
            done_raw = torch.tensor(done, dtype=torch.int8, device=self.device)    # сохраняем знак для метрик
            done = (done_raw != 0).float()        
            action_o = torch.tensor([a[0] for a in action], dtype=torch.long, device=self.device)
            model_reward = torch.FloatTensor(model_reward).to(self.device)
            opt_model_i = torch.IntTensor(opt_model_i).to(self.device)

            # --- OPTIMIZER HEAD: текущие Q(s_t,a_t)
            flat, q_opt_cur = self.model_optim(state)
            q_sa = q_opt_cur.gather(1, action_o.view(-1,1)).squeeze(1)

            # --- SOFT/WATKINS G^{λ,κ} на каждый элемент батча из своей последовательности ---
            if self.use_soft_watkins:
                with torch.no_grad():
                    soft_watkins_targets = [
                        self._soft_watkins_targets(seq, self.gamma) for seq in seqs
                    ]
                    y_opt_list, lambda_weight_list = zip(*soft_watkins_targets)
                y_opt = torch.stack(y_opt_list, dim=0)
                lambda_weight = torch.stack(lambda_weight_list, dim=0)
            else:
                y_opt = self._one_step_optimizer_targets(next_state, reward, done)
                lambda_weight = torch.ones(B, dtype=torch.float, device=self.device)

            tr_mask_drop, tr_keep, sigma_t, sigma = self._trust_region_keep_mask(
                q_sa, y_opt, state, action_o
            )

            # --- применяем маску к лоссу оптимизаторной головы ---
            per_sample_loss_opt = self.huberloss(input=q_sa, target=y_opt) * is_w
            loss_opt = (per_sample_loss_opt * tr_keep).sum() / tr_keep.sum().clamp_min(1.0)


            td_opt_abs = (q_sa - y_opt).abs().detach()
            td_opt_abs = td_opt_abs * tr_keep + self.tr_eps  

            # --- PARAM HEADS: Double per-parameter ---
            opt_names = [self.i2opt[int(i.item())] for i in action_o]
            q_params_cur = self.model_params(flat, opt_names)
            with torch.no_grad():
                q_params_next_on = self.model_params(next_state, opt_names)
                q_params_next_tg = self.target_model_params(next_state, opt_names)

            loss_param_items, td_param_items = [], []
            for i in range(B):
                lp_sum = torch.tensor(0.0, device=self.device)
                td_sum = 0.0
                for pname in self.optimizer_dict[opt_names[i]]:
                    act_idx = self._get_param_act_idx(action[i], pname)
                    q_curr  = q_params_cur[i][pname][act_idx]

                    q_next_on  = q_params_next_on[i][pname]         # (n_choices,)
                    a_next_p   = int(q_next_on.argmax().item())
                    q_next_tg  = q_params_next_tg[i][pname][a_next_p]
                    y_p = reward[i] + (1.0 - done[i]) * self.gamma * q_next_tg
                    delta_p_raw = (y_p - q_curr).detach()  

                    sigma_batch_p = float(delta_p_raw.abs().clamp_min(self.param_td_eps).item())
                    key = (opt_names[i], pname) 
                    prev = self.param_td_running_std.get(key, sigma_batch_p)
                    ema  = self.param_td_mom * prev + (1.0 - self.param_td_mom) * sigma_batch_p
                    self.param_td_running_std[key] = ema
                    sigma_p = max(ema, sigma_batch_p, self.param_td_eps)

                    delta_p_norm = (y_p - q_curr) / sigma_p
                    lp = self.huberloss(input=delta_p_norm, target=torch.zeros_like(delta_p_norm)) * is_w[i]
                    lp_sum = lp_sum + (lp * tr_keep[i])
                    td_sum += float((q_curr - y_p).abs().item() / sigma_p)
                loss_param_items.append(lp_sum)
                td_param_items.append(td_sum)

            loss_param = torch.stack(loss_param_items).sum() / tr_keep.sum().clamp_min(1.0)

            # --- шаг оптимизации ---
            self.optimizer_opt.zero_grad()
            self.optimizer_params.zero_grad()
            (loss_opt + loss_param).backward()
            torch.nn.utils.clip_grad_norm_(self.model_optim.parameters(), 10.0)
            torch.nn.utils.clip_grad_norm_(self.model_params.parameters(), 10.0)
            self.optimizer_opt.step()
            self.optimizer_params.step()

            new_priors = None
            if self.use_prioritized_replay:
                with torch.no_grad():
                    td_param_abs = torch.as_tensor(
                        td_param_items, dtype=torch.float, device=self.device
                    )
                    td_param_abs = td_param_abs * tr_keep + self.tr_eps
                    new_priors = td_opt_abs + td_param_abs
                self.replay_buffer.update_priorities(idxs, new_priors.cpu())

            if self.warmup_active:
                self.warmup_updates_done += 1
                if self.warmup_updates_done >= self.warmup_updates_total and not self.recalc_done:
                    print(f"Warmup finished: {self.warmup_updates_done} updates. Recalculating priorities offline...")
                    recalc_all_priorities_batched(self, batch_size=self.recalc_batch_size)
                    self.recalc_done = True
                    self.warmup_active = False

            # --- периодическое обновление таргет-сетей (как у тебя) ---
            print("\nRL optimization is complete!\n")
            self.transition_counter += self.batch_size
            

            # периодическая реинициализация таргет-сетей (как у тебя)
            if self.transition_counter >= self.n_transitions_reinit:
                print("REINIT TARGET")
                self.reinit_target()
                self.transition_counter = 0

            loss_arr_optim_class.append(float(loss_opt.item()))
            loss_arr_param.append(float(loss_param.item()))

            with torch.no_grad():
                delta_raw  = (y_opt - q_sa).detach()
                delta_norm = (delta_raw / sigma_t)

                mean_abs_delta_norm = float(delta_norm.abs().mean().item())
                sigma_td = float(sigma)
                q_abs_mean = float(q_sa.abs().mean().item())
                y_opt_mean = float(y_opt.abs().mean().item())
                lambda_weight_mean = float(lambda_weight.mean().item())
                lambda_weight_min = float(lambda_weight.min().item())
                lambda_weight_max = float(lambda_weight.max().item())

                # tr_drop_frac у тебя уже есть как drop_frac
                # seq_avg_len у тебя уже есть как avg_len

                prio_p95 = None
                if new_priors is not None:
                    prio_p95 = float(
                        torch.quantile(new_priors.detach().float(), 0.95).item()
                    )

                greedy_actions = q_opt_cur.argmax(dim=1)
                action_diagnostics = {}
                for action_idx, optim_name in self.i2opt.items():
                    action_mask = action_o == action_idx
                    action_count = int(action_mask.sum().item())
                    action_diagnostics[
                        f"train_policy/greedy_frac/{optim_name}"
                    ] = float((greedy_actions == action_idx).float().mean().item())
                    action_diagnostics[
                        f"train_q/{optim_name}/mean_all_states"
                    ] = float(q_opt_cur[:, action_idx].mean().item())

                    if action_count == 0:
                        continue

                    action_diagnostics[
                        f"train_action/{optim_name}/target_mean"
                    ] = float(y_opt[action_mask].mean().item())

                chain_mc_metrics = collect_chain_mc_metrics(self, seqs, q_sa, self.gamma)
                for metric_name, metric_value in chain_mc_metrics.items():
                    chain_mc_metric_values[metric_name].append(metric_value)


            print(f"Loss for params: {loss_param}")
            print(f"Loss for optim: {loss_opt}")
            print(f"Loss for both: {loss_opt + loss_param}")

            all_rewards.append(reward.detach().cpu())
            all_dones.append(done_raw.detach().cpu())

            agent_action_mask = opt_model_i >= 0
            model_reward_i_ar += model_reward[agent_action_mask].reshape(-1).tolist()


            dropped  = int(tr_mask_drop.sum().item())
            kept     = int(tr_keep.sum().item())
            drop_frac = dropped / max(dropped + kept, 1)

            delta_raw = (y_opt - q_sa).detach()
            mean_abs_delta = delta_raw.abs().mean().item()

            lens = [len(seq) for seq in seqs]
            frac_len_gt1 = sum(l > 1 for l in lens) / max(len(lens), 1)
            avg_len = (sum(lens) / max(len(lens), 1))


            # seqs: список последовательностей, каждая <= L
            # Посчитаем, сколько из них заканчиваются success-терминалом

            count_seq_success = 0
            count_seq_total   = len(seqs)

            for seq in seqs:
                last = seq[-1]
                # тот же критерий успеха, что использует буфер
                if (last.done == 1) and (last.model_reward > self.success_reward_threshold):
                    count_seq_success += 1

            frac_seq_success = count_seq_success / max(count_seq_total, 1)


        metrics_to_log = {
            "mean_abs_delta_norm": mean_abs_delta_norm,
            "sigma_td": sigma_td,
            "q_abs_mean": q_abs_mean,
            "y_opt_mean": y_opt_mean,
            "tr_drop_frac": drop_frac,
            "mean_abs_delta": mean_abs_delta,
            "seq_frac_len_gt1": frac_len_gt1,
            "seq_avg_len": avg_len,
            "seq_frac_success": frac_seq_success,
            "lambda_weight_mean": lambda_weight_mean,
            "lambda_weight_min": lambda_weight_min,
            "lambda_weight_max": lambda_weight_max,
        }
        if prio_p95 is not None:
            metrics_to_log["prio_p95"] = prio_p95

        metrics_to_log.update({
            metric_name: statistics.mean(values) if values else 0.0
            for metric_name, values in chain_mc_metric_values.items()
        })
        metrics_to_log.update(policy_position_metrics)
        metrics_to_log.update(action_diagnostics)

        if self.exp is not None:
            self.exp.log_metrics(metrics_to_log, step=self.steps_done)

        self.opt_step += 1

        reward_tensor = torch.cat(all_rewards)
        done_tensor = torch.cat(all_dones)

        # Подсчёт: хорошее завершение — done == 1 и reward > 0
        count_good_end = torch.sum((done_tensor == 1) & (reward_tensor > 0)).item()

        # Подсчёт: плохое завершение — done == -1 и reward < 0
        count_bad_end = torch.sum((done_tensor == -1) & (reward_tensor < 0)).item()

        # матрица сопряжённости, чтобы сразу увидеть, почему пересечений нет
        sign_reward = torch.sign(reward_tensor).clamp(min=-1, max=1)  # -1, 0, 1
        for d in (-1, 0, 1):
            row_mask = (done_tensor == d)
            c_neg = ((row_mask) & (sign_reward == -1)).sum().item()
            c_zero = ((row_mask) & (sign_reward ==  0)).sum().item()
            c_pos = ((row_mask) & (sign_reward ==  1)).sum().item()
            print(f"done={d}: reward<0={c_neg}, reward==0={c_zero}, reward>0={c_pos}")


        print(f"Count of good ends: {count_good_end}")
        print(f"Count of bad ends: {count_bad_end}") 

        print("done counts:", (done_tensor == 1).sum().item(), (done_tensor == -1).sum().item())
        print("reward>0:", (reward_tensor > 0).sum().item(), "reward<0:", (reward_tensor < 0).sum().item())
   
        
        # mean_batch_loss_optim_class = 0
        # for el in loss_arr_optim_class:
        #     mean_batch_loss_optim_class += el
        # mean_batch_loss_optim_class = mean_batch_loss_optim_class / len(loss_arr_optim_class)
        # if loss_arr_param != []:
        #     mean_batch_loss_param = 0
        #     for el in loss_arr_param:
        #         mean_batch_loss_param += el
        #     mean_batch_loss_param = mean_batch_loss_param / len(loss_arr_param)
        agent_reward_count = len(model_reward_i_ar)
        bad_action = [el for el in model_reward_i_ar if el <= 0]

        optim_batch_loss_mean = statistics.mean(loss_arr_optim_class)
        param_batch_loss_mean = statistics.mean(loss_arr_param) 

        print(f"Mean batch loss optim class: {optim_batch_loss_mean}")
        print(f"Mean batch loss param: {param_batch_loss_mean}")

        if self.exp is not None:

            self.exp.log_metric("optim_batch_loss_mean", optim_batch_loss_mean, step=self.steps_done)
            self.exp.log_metric("optim_batch_loss_median", statistics.median(loss_arr_optim_class), step=self.steps_done)
            self.exp.log_metric("param_batch_loss_mean", param_batch_loss_mean, step=self.steps_done)
            self.exp.log_metric("param_batch_loss_median", statistics.median(loss_arr_param), step=self.steps_done)
            self.exp.log_metric("steps_done", self.steps_done, step=self.steps_done)
            self.exp.log_metric("all_rewards_mean", statistics.mean(reward_tensor.tolist()), step=self.steps_done)
            self.exp.log_metric("agent_reward_count", agent_reward_count, step=self.steps_done)
            if agent_reward_count > 0:
                self.exp.log_metric("agent_reward_mean", statistics.mean(model_reward_i_ar), step=self.steps_done)
                self.exp.log_metric("agent_reward_median", statistics.median(model_reward_i_ar), step=self.steps_done)
                self.exp.log_metric("bad_action_procent", len(bad_action)/agent_reward_count, step=self.steps_done)
            self.exp.log_metric("count_good_end", count_good_end, step=self.steps_done)
            self.exp.log_metric("count_bad_end", count_bad_end, step=self.steps_done)
            # Логируем список приоритетов

            if self.use_prioritized_replay:
                log_priority_to_comet(self.exp, self.replay_buffer.prior, step=self.steps_done)
            # self.exp.log_parameter('priority', self.replay_buffer.prior)



            # Save model snapshots locally and, optionally, log them to Comet as assets.
            # We use log_asset instead of log_model to avoid the Comet model-element limit.
            self.model_snapshot_dir.mkdir(parents=True, exist_ok=True)
            optim_path = self.model_snapshot_dir / f"model_optim_step_{self.steps_done}.pt"
            params_path = self.model_snapshot_dir / f"model_params_step_{self.steps_done}.pt"

            torch.save(self.model_optim.state_dict(), optim_path)
            torch.save(self.model_params.state_dict(), params_path)

            self.exp.log_asset(
                str(optim_path),
                file_name=f"rl_model_snapshots/model_optim_step_{self.steps_done}.pt",
                step=self.steps_done,
                overwrite=True,
            )
            self.exp.log_asset(
                str(params_path),
                file_name=f"rl_model_snapshots/model_params_step_{self.steps_done}.pt",
                step=self.steps_done,
                overwrite=True,
            )
            self.exp.log_other("model_snapshot_step", self.steps_done)
            self.exp.log_other("model_snapshot_local_dir", str(self.model_snapshot_dir))


        return loss_arr_optim_class, loss_arr_param

    
    def post_proc_model(self, optim_class, epochs_class, param_class):
        class_name = self.i2opt[optim_class]
        epochs = self.optimizer_dict[class_name]['epochs'][epochs_class]
        params = {}
        for param_name, param_val in param_class.items():
            params[param_name] = self.optimizer_dict[class_name][param_name][param_val]
        action_dict = {
            'type': class_name,
            'epochs': epochs,
            'params': params
        }
        return action_dict


    def get_random_action(self):
        optim_class = random.randint(0, len(self.i2opt) - 1)
        class_name = self.i2opt[optim_class]
        param_class = {}
        optim_class_dict = self.optimizer_dict[class_name]

        for key in optim_class_dict:
            if key == 'epochs': epochs_class = random.randint(0, len(optim_class_dict['epochs']) - 1)
            else:
                param_class[key] = random.randint(0, len(optim_class_dict[key]) - 1)

        return optim_class, epochs_class, param_class
    
    # Action function stub
    def select_action(self, state):

        # собрать 4-канальное состояние
        if "delta" not in state:
            delta = torch.zeros_like(state["loss_total"])
        else:
            delta = state["delta"]

        state_tensor = torch.stack([
            state["loss_total"],
            state["loss_oper"],
            state["loss_bnd"],
            delta
        ], dim=0).to(self.device)

        # сделать батч: (1,4,26,26)
        state_tensor = state_tensor.unsqueeze(0)

        # eps-greedy
        sample = random.random()
        eps_threshold = self.get_epsilon_threshold()
        self.steps_done += 1

        # --- GREEDY ---
        if sample > eps_threshold:
            with torch.no_grad():
                liner_out, q_opt = self.model_optim(state_tensor)
                optim_class = int(torch.argmax(q_opt).item())

                optim_name = self.i2opt[optim_class]

                param_class = {}
                param_dict = self.model_params(liner_out, [optim_name])[0]

                for key in param_dict:
                    if key == 'epochs':
                        epochs_class = int(torch.argmax(param_dict[key]).item())
                    else:
                        param_class[key] = int(torch.argmax(param_dict[key]).item())

        # --- EPSILON RANDOM ---
        else:
            optim_class, epochs_class, param_class = self.get_random_action()

        # оформить action в формате твоего пайплайна
        action_dict = self.post_proc_model(optim_class, epochs_class, param_class)

        return action_dict, (optim_class, epochs_class, param_class), sample > eps_threshold


    def render_Q_function(self):            
        
        get_weights = lambda model_lair: nn.Sigmoid()(torch.sum(model_lair.weight, dim=1)).detach().cpu().numpy()
        optim_weights = get_weights(self.model_optim.fc_optim_class)

        plt.figure(figsize=(10, 6))
        plt.title('Optimizers')
        x = [i for i in range(len(optim_weights))]
        plt.bar(x, optim_weights, align='center')
        labes_optim = [self.i2opt[i] for i in range(len(self.i2opt))]
        plt.xticks(x, labes_optim)
        plt.savefig(f'Optimizers_{self.opt_count_out}.png')

        for optim_name in self.model_params.fc_param_by_opt:
            n_params = len(self.model_params.fc_param_by_opt[optim_name])
            n_rows = ceil(n_params/2)
            fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(10, 6))
            fig.suptitle(f'{optim_name}_params')
            i_subplot = 0
            for param_name in self.model_params.fc_param_by_opt[optim_name]:
                fc_ = self.model_params.fc_param_by_opt[optim_name][param_name][-1]
                weights = get_weights(fc_)
                x = [i for i in range(len(weights))]
                labes_optim = [str(el) for el in self.optimizer_dict[optim_name][param_name]]
                if n_rows > 1:
                    i_subplot_cord = (ceil(i_subplot/2), i_subplot%2)
                else:
                    i_subplot_cord = i_subplot
                axes[i_subplot_cord].bar(x, weights, align='center')
                axes[i_subplot_cord].set_xticks(x, labes_optim)
                axes[i_subplot_cord].set_title(param_name)
                i_subplot += 1
            fig.savefig(f'{optim_name}_params_{self.opt_count_out}.png')
