import os
import torch
from collections import deque, namedtuple
from sklearn.model_selection import train_test_split
import traceback
import random
import pickle
import numpy as np

from RL.rl_algorithms import PrioritizedReplayBuffer, Transition

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.memory = deque()

#     def push(self, *args):
#         self.memory.append(Transition(*args))

#     def sample(self, batch_size):
#         current_sample = random.sample(self.memory, batch_size)
#         current_sample_tuples = [tuple(t) for t in current_sample]
#         return current_sample_tuples

#     def __len__(self):
#         return len(self.memory)
    
# Transition = namedtuple('Transition',
#                         ('state', 'next_state', 'action', 'reward', 'done',  'model_reward', 'opt_model_i'))

rename_map = {
    'action_raw': 'action',
    'env_raw_reward': 'reward',
    'abs_done': 'done',
    'reward_model_i': 'reward_model',
}

def to_cpu(obj):
    if torch.is_tensor(obj):
        # detach на всякий случай, чтобы оторваться от графа вычисления
        return obj.detach().to('cpu')
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(to_cpu(v) for v in obj)
    return obj

def load_transitions_to_replay_buffer(replay_buffer, source, learn_or_analyze="learn", prev_tol=0.0, current_tol=0.0):
    """
    Загружает переходы в replay_buffer.
    Может принимать:
      1️⃣ path (str): путь к директории с файлами .pt/.pickle;
      2️⃣ list[dict]: список уже загруженных структур (например, из Comet).
    """
    count = 0
    count_done_1 = 0
    count_done_minus_1 = 0

    def process_transition_dict(data, file_label="memory"):
        nonlocal count, count_done_1, count_done_minus_1

        # Переименование ключей
        renamed = {}
        for old_key, value in data.items():
            new_key = rename_map.get(old_key, old_key)
            renamed[new_key] = value
        data = renamed

        # Проверяем обязательные ключи
        required_keys = ['state', 'next_state', 'action', 'reward', 'done', 'reward_model', 'opt_model_i']
        if not all(k in data for k in required_keys):
            print(f"⚠️ Пропуск {file_label}: отсутствуют обязательные ключи {set(required_keys) - set(data.keys())}")
            return

        # Перенос на CPU
        state_cpu      = to_cpu(data['state'])
        next_state_cpu = to_cpu(data['next_state'])
        action_cpu     = to_cpu(data['action'])

        # reward / model_reward — сразу CPU float32
        BLOCKED_ROUNDED = {round(x, 4) for x in [-1.3047, -1.3186, -1.0238]}
        reward_val = float(data['reward'])
        if round(reward_val, 4) in BLOCKED_ROUNDED:
            print(f"⚠️ Фильтр Burgers: reward={reward_val}")
            return
        if prev_tol != 0.0:
            data = modify_transition(data, prev_tol=prev_tol, current_tol=current_tol)

        reward_t       = torch.tensor(data['reward'], dtype=torch.float32, device='cpu')
        model_reward_t = torch.tensor(data['reward_model'], dtype=torch.float32, device='cpu')

        # --- нормализация флагов done и наград ---
        # if reward_t < -1.0 and data['done'] == 1:
        #     data['done'] = -1
        #     if model_reward_t > 10: 
        #         model_reward_t = reward_t
        #     if reward_t < -10:
        #         reward_t = -10
        # if data['done'] == 1 and model_reward_t > 50:
        #     model_reward_t = model_reward_t - 90
        #     if model_reward_t > 40:
        #         model_reward_t = model_reward_t - 40
        # if data['done'] == -1 and model_reward_t < -50:
        #     if model_reward_t < -120:
        #         model_reward_t = -10
        #     else:
        #         model_reward_t = model_reward_t + 90
        # if data['done'] == -1 and model_reward_t > 0:
        #     model_reward_t = -10
        # if np.isclose(-1.0/reward_t, model_reward_t, atol=1e-4):
        #     model_reward_t = reward_t

        # --- создаём Transition ---
        transition = Transition(
            state=state_cpu,
            next_state=next_state_cpu,
            action=action_cpu,
            reward=model_reward_t if learn_or_analyze == 'learn' else reward_t,
            done=int(data['done']),
            model_reward=model_reward_t,
            opt_model_i=data['opt_model_i']
        )

        replay_buffer.push(*transition)
        count += 1
        if data['done'] == 1:
            count_done_1 += 1
        elif data['done'] == -1:
            count_done_minus_1 += 1

    # === 1️⃣ Если передан список структур ===
    if isinstance(source, list):
        print(f"📥 Загружаем из списка структур: {len(source)} элементов")
        for i, item in enumerate(source):
            if isinstance(item, dict):
                process_transition_dict(item, f"list[{i}]")
            else:
                print(f"⚠️ Пропуск list[{i}] — не dict ({type(item)})")

    # === 2️⃣ Если передан путь к директории ===
    elif isinstance(source, str) and os.path.isdir(source):
        print(f"📁 Загружаем переходы из директории: {source}")
        for root, _, files in os.walk(source):
            for file in sorted(files, key=lambda f: os.path.getmtime(os.path.join(root, f))):
                if not (file.endswith('.pickle') or file.endswith('.pt')):
                    continue
                file_path = os.path.join(root, file)
                try:
                    if file.endswith('.pickle'):
                        with open(file_path, 'rb') as f:
                            data_load = pickle.load(f)
                    else:
                        data_load = torch.load(file_path, map_location="cpu")

                    if isinstance(data_load, dict):
                        process_transition_dict(data_load, file_path)
                    elif isinstance(data_load, list):
                        for i, d in enumerate(data_load):
                            if isinstance(d, dict):
                                process_transition_dict(d, f"{file_path}[{i}]")
                    elif isinstance(data_load, dict) and "memory" in data_load:
                        for i, d in enumerate(data_load["memory"]):
                            process_transition_dict(d, f"{file_path}::memory[{i}]")
                    else:
                        print(f"⚠️ {file_path} — неизвестный формат ({type(data_load)})")

                except Exception as e:
                    print(traceback.format_exc())
                    print(f"❌ Ошибка при чтении {file_path}: {e}")

    else:
        raise ValueError("Аргумент source должен быть либо путём к директории, либо списком структур (list[dict])")

    print(f"✅ Загружено {count} переходов ({count_done_1} успешных, {count_done_minus_1} неуспешных)")
    return replay_buffer


def modify_transition(data, prev_tol=0.0, current_tol=0.0):
    """
    Модифицирует reward_model и done в зависимости от значения reward.
    Условия:
      - если abs(reward) < 0.040956 или abs(reward) > 0.041 — ничего не делаем
      - если 0.040956 < abs(reward) < 0.041 — заменяем reward_model = reward, done = 0 (если done был 1)
    """
    reward = float(data.get('reward', 0.0))
    abs_r = abs(reward)
    if current_tol < prev_tol:
        if prev_tol < abs_r <= current_tol:
            print("\n=== ⚙️ Data before modification ===")
            print({
                'reward': data.get('reward'),
                'reward_model': data.get('reward_model'),
                'done': data.get('done'),
                'opt_model_i': data.get('opt_model_i')
            })

            data['reward_model'] = reward
            if data.get('done') == 1:
                data['done'] = 0

            print("=== ✅ Data after modification ===")
            print({
                'reward': data.get('reward'),
                'reward_model': data.get('reward_model'),
                'done': data.get('done'),
                'opt_model_i': data.get('opt_model_i')
            })
            print("=" * 50)

    return data



def split_buffer(orig_buffer: PrioritizedReplayBuffer, test_ratio: float = 0.2):
    """
    Делает стратифицированное разбиение ReplayBuffer по grad_i с использованием sklearn.
    """
    data = list(orig_buffer.memory)
    labels = [tr.done for tr in data]

    train_data, test_data = train_test_split(
        data,
        test_size=test_ratio,
        stratify=labels,
    )

    train_buffer = PrioritizedReplayBuffer(capacity=len(train_data) + 100)
    test_buffer  = PrioritizedReplayBuffer(capacity=len(test_data) + 100)

    for tr in train_data:
        train_buffer.push(*tr)
    for tr in test_data:
        test_buffer.push(*tr)

    return train_buffer, test_buffer


# Пример использования:
# agent = DQNAgent(...)  # предполагается, что агент уже создан
# load_transitions_to_replay_buffer(agent.replay_buffer, "/path/to/transitions")

# replay_buffer = ReplayBuffer(10000)
# path = r"C:\Users\Рустам\Documents\GitHub\torch_DE_solver_local\test\RL_experiments\Article_exp\data\burg_state"
# load_transitions_to_replay_buffer(replay_buffer, path)
# print(replay_buffer.__len__())