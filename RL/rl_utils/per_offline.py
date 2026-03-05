# per_offline.py
import torch

@torch.no_grad()
def recalc_all_priorities_batched(agent, batch_size: int = 32):
    """
    Пересчитывает приоритеты для ВСЕГО буфера agent.replay_buffer по |TD|.
    Учитывает:
      - Double DQN для головы оптимизатора
      - Double для параметрических голов выбранного оптимизатора
    Выполняется батчами, чтобы не съесть память.

    Ожидает, что у agent есть:
      - replay_buffer (из per_buffer.PrioritizedReplayBuffer)
      - model_optim, target_model_optim
      - model_params, target_model_params
      - _stack_state(dict)->Tensor[2,26,26]
      - _get_param_act_idx(action, pname) -> int
      - i2opt, optimizer_dict, device, gamma
    """
    buf = agent.replay_buffer
    N = len(buf)
    if N == 0:
        print("Replay buffer is empty; skip recalc.")
        return

    agent.model_optim.eval(); agent.target_model_optim.eval()
    agent.model_params.eval(); agent.target_model_params.eval()

    dev = agent.device
    B = int(batch_size)
    new_priors = []

    def _stack_batch(transitions):
        state  = torch.stack([agent._stack_state(tr.state)      for tr in transitions]).to(dev)   # (b,2,26,26)
        nstate = torch.stack([agent._stack_state(tr.next_state) for tr in transitions]).to(dev)
        reward = torch.tensor([float(tr.reward) for tr in transitions], dtype=torch.float, device=dev)
        # Terminal flag: both done==1 and done==-1 must stop bootstrap.
        done   = torch.tensor([1.0 if int(tr.done) != 0 else 0.0 for tr in transitions], dtype=torch.float, device=dev)
        a_optim = torch.tensor([int(tr.action[0]) for tr in transitions], dtype=torch.long, device=dev)
        opt_names = [agent.i2opt[int(i.item())] for i in a_optim]
        return state, nstate, reward, done, a_optim, opt_names

    for start in range(0, N, B):
        end = min(start+B, N)
        chunk = buf.memory[start:end]
        state, nstate, reward, done, a_optim, opt_names = _stack_batch(chunk)

        # ----- OPTIMIZER HEAD TD -----
        flat, q_opt_cur = agent.model_optim(state)                       # (b,A)
        q_sa = q_opt_cur.gather(1, a_optim.view(-1,1)).squeeze(1)        # (b,)

        _, q_opt_next_on  = agent.model_optim(nstate)
        a_next = q_opt_next_on.argmax(dim=1)
        _, q_opt_next_tg  = agent.target_model_optim(nstate)
        q_next = q_opt_next_tg.gather(1, a_next.view(-1,1)).squeeze(1)
        y_opt  = reward + (1.0 - done) * agent.gamma * q_next
        td_opt = (q_sa - y_opt).abs()                                    # (b,)

        # ----- PARAM HEADS TD (сумма по параметрам выбранного оптимизатора) -----
        q_params_cur      = agent.model_params(flat, opt_names)
        q_params_next_on  = agent.model_params(nstate, opt_names)
        q_params_next_tg  = agent.target_model_params(nstate, opt_names)

        td_param = torch.zeros(len(chunk), dtype=torch.float, device=dev)

        for i, opt_name in enumerate(opt_names):
            for pname in agent.optimizer_dict[opt_name]:
                act_idx = agent._get_param_act_idx(chunk[i].action, pname)
                q_curr  = q_params_cur[i][pname][act_idx]

                q_next_on  = q_params_next_on[i][pname]          # (n_choices,)
                a_next_p   = int(q_next_on.argmax().item())
                q_next_tg  = q_params_next_tg[i][pname][a_next_p]
                y_p = reward[i] + (1.0 - done[i]) * agent.gamma * q_next_tg
                td_param[i] += (q_curr - y_p).abs()

        new_p_chunk = (td_opt + td_param + buf.eps).detach().cpu().tolist()
        new_priors.extend(new_p_chunk)
        idxs_chunk = torch.arange(start, end, dtype=torch.long)
        buf.update_priorities(idxs_chunk, torch.tensor(new_p_chunk, dtype=torch.float))

    print(f"✅ Recalculated priorities for {N} transitions. "
          f"mean={float(torch.tensor(new_priors).mean()):.4f}, "
          f"min={min(new_priors):.4f}, max={max(new_priors):.4f}")
