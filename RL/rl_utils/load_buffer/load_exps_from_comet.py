from comet_ml import API
import torch
import io, os
from datetime import datetime
from RL.rl_algorithms import PrioritizedReplayBuffer
from RL.rl_utils.load_buffer.load_transitions_into_buffer_pickle import load_transitions_to_replay_buffer


# === Настройки ===
WORKSPACE = "saitama32"
PROJECT_NAME = "rlpinn-grayscott-farm-transitions"
# MAX_EXPERIMENTS = 15  # можно изменить при необходимости

api = API(api_key="aP71fQTYPNqfsYWvudPPmoBl5")  # или просто API()


# === Вспомогательные функции ===
def get_metadata_field(exp, field, default=None):
    try:
        meta = exp.get_metadata()
        return meta.get(field, default)
    except Exception:
        return default
    

def get_param_value(exp, param_name, default=None):
    try:
        params = exp.get_parameters_summary()
        params_dict = {p["name"]: p["valueCurrent"] for p in params}
        return params_dict.get(param_name, default)
    except Exception:
        return default



def get_end_time(exp):
    end_ms = get_metadata_field(exp, "endTimeMillis")
    if end_ms:
        return datetime.fromtimestamp(end_ms / 1000)
    return datetime.min


def get_duration_hours(exp):
    """Возвращает длительность эксперимента в часах."""
    start_ms = get_metadata_field(exp, "startTimeMillis", 0)
    end_ms = get_metadata_field(exp, "endTimeMillis", 0)
    if not start_ms or not end_ms:
        return 0.0
    duration_h = (end_ms - start_ms) / (1000 * 60 * 60)
    return duration_h


def is_crashed(exp):
    return get_metadata_field(exp, "hasCrashed", False) is True


# === Основная функция ===
def collect_all_comet_transitions(replay_buffer=None, max_exps_last=10, duration_grater_hours = 1, 
                                  save_dir=None, tolerance = 0.0, prev_tol=0.0, use_tol=True, new_tol=False,
                                  use_log_state=False, proj_name=None, mark_states=None) -> PrioritizedReplayBuffer:
    """Собирает все переходы из не-crashed экспериментов проекта и возвращает заполненный PrioritizedReplayBuffer."""
    print("🔍 Получаем эксперименты из Comet...")
    if proj_name is not None:
        experiments = list(api.get_experiments(workspace=WORKSPACE, project_name=proj_name))
    else:
        experiments = list(api.get_experiments(workspace=WORKSPACE, project_name=PROJECT_NAME))
    # valid_experiments = [exp for exp in experiments if not is_crashed(exp)]
    experiments_sorted = sorted(experiments, key=get_end_time, reverse=True)
    experiments_sorted_duration = [
        exp for exp in experiments_sorted
        if get_duration_hours(exp) >= duration_grater_hours
    ]
    # experiments_sorted = [api.get_experiment(workspace=WORKSPACE, project_name=PROJECT_NAME, experiment='751c7ca595dd4dafb22a0cfe61c26b6f')]

    experiments_sorted_duration = experiments_sorted_duration[:max_exps_last]
    if prev_tol > 0.0 and use_tol:

        experiments_sorted_tol = [
            exp for exp in experiments_sorted_duration 
            if float(get_param_value(exp, "tolerance", 0.0)) >= prev_tol
        ]
    elif prev_tol == 0 and use_tol:
        experiments_sorted_tol = [
            exp for exp in experiments_sorted_duration 
            if float(get_param_value(exp, "tolerance", 0.0)) >= tolerance
        ]
    else:
        experiments_sorted_tol = experiments_sorted_duration 



    print(f"✅ Найдено {len(experiments_sorted_tol)} активных экспериментов для загрузки буферов.\n")

    all_transitions = []  # сюда соберём всё

    for i, exp in enumerate(experiments_sorted_tol, 1):
        meta = exp.get_metadata()
        exp_id = meta.get("experimentKey")
        exp_name = meta.get("experimentName")
        print(f"[{i:2d}] {exp_name} ({exp_id})")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            print(f"💾 Сохранение включено — файлы будут сохраняться в {save_dir}")

        assets = exp.get_asset_list()
        # --- фильтруем и сортируем по step ---
        pt_assets = [a for a in assets if a["fileName"].endswith(".pt") and "entry_step" in a["fileName"]]

        def get_step(asset):
            if "step" in asset and isinstance(asset["step"], (int, float)):
                return int(asset["step"])
            fname = asset.get("fileName", "")
            try:
                return int(fname.split("entry_step_")[-1].split(".")[0])
            except Exception:
                return 0

        pt_assets = sorted(pt_assets, key=get_step)

        if not pt_assets:
            print("   ⚠️ Нет файлов entry_step_*.pt — пропускаем.")
            continue

        for asset in pt_assets:
            filename = asset["fileName"]

            try:
                file_bytes = exp.get_asset(asset["assetId"], return_type="binary")
                buffer_stream = io.BytesIO(file_bytes)
                data_load = torch.load(buffer_stream, map_location="cpu")

                if save_dir is not None:
                    safe_name = f"{exp_name}_{filename}".replace("/", "_")
                    save_path = os.path.join(save_dir, safe_name)
                    torch.save(data_load, save_path)
                    print(f"   💾 Сохранено локально: {save_path}")

                if isinstance(data_load, dict):
                    if "memory" in data_load:
                        all_transitions.extend(data_load["memory"])
                    else:
                        all_transitions.append(data_load)
                elif isinstance(data_load, list):
                    all_transitions.extend(data_load)
                else:
                    print(f"   ⚠️ Формат {filename} не распознан ({type(data_load)}), пропуск.")
                    continue

                print(f"   ⬇️ {filename} загружен ({len(all_transitions)} переходов накоплено).")

            except Exception as e:
                print(f"   ❌ Ошибка при чтении {filename}: {e}")
    # tolerance =0.0608023 
    # prev_tol= 0.060776
    if tolerance > 0.0 and prev_tol == 0.0 and new_tol:
        all_transitions = truncate_failure_chains_by_tol(
            all_transitions,
            tol=tolerance,
            shift_reward=10.0,
        )
    elif tolerance > prev_tol and prev_tol != 0.0:
        all_transitions = truncate_success_chains(all_transitions, current_tol=tolerance, prev_tol= prev_tol)

    if mark_states:
        all_transitions = add_proj_mark(all_transitions, proj_name)

    # --- Сдвиг наград для успешных переходов ---
    all_transitions = shift_done_rewards(all_transitions,  done = -1, shift_value= -5)
    # --- Добавление delta loss ---
    all_entries = add_delta_to_all_entries(all_transitions)

    if use_log_state:
        apply_log_transform_to_transitions(all_entries)

    print(f"\n🚀 Всего собрано {len(all_entries)} переходов из {len(experiments_sorted_duration)} экспериментов.")
    if not all_entries:
        print("⚠️ Не найдено переходов для загрузки — возвращаем пустой буфер.")
        return PrioritizedReplayBuffer(capacity=1)

    # === Заполняем буфер ===
    replay_buffer = load_transitions_to_replay_buffer(replay_buffer, all_entries, prev_tol=prev_tol, current_tol=tolerance)

    # print(f"\n✅ Финальный буфер содержит {len(replay_buffer)} переходов.")
    return replay_buffer


def compute_delta_map(loss_t, loss_t1, eps=1e-6):
    """
    Нормализованная дельта, как мы делаем в онлайне:
    raw = loss_t1 - loss_t
    delta = sign(raw) * log(1 + |raw|)
    затем нормируем на max|delta| и режем в [-1,1]
    """
    raw_delta = loss_t1 - loss_t
    delta = torch.sign(raw_delta) * torch.log1p(torch.abs(raw_delta))
    delta = delta / (delta.abs().max() + eps)
    delta = delta.clamp(-1, 1)
    return delta


def add_delta_to_sequence(seq, eps=1e-6):
    """
    seq: список переходов (dict), у каждого:
      tr["state"]["loss_total"], tr["next_state"]["loss_total"] — тензоры
    Модифицирует seq in-place, проставляя state["delta"] и next_state["delta].

    Если во ВСЕХ переходах уже есть и state["delta"], и next_state["delta"],
    НИЧЕГО не делаем (идемпотентность).
    """
    if not seq:
        return

    # --- 0) Проверяем, не всё ли уже размечено delta ---
    already_has_delta_everywhere = all(
        ("delta" in tr.get("state", {}) and "delta" in tr.get("next_state", {}))
        for tr in seq
    )
    if already_has_delta_everywhere:
        # все состояния уже имеют delta → ничего не трогаем
        return

    # --- 1) Сначала считаем delta_t для каждого перехода и кладём в next_state["delta"] ---
    for tr in seq:
        s  = tr["state"]
        ns = tr["next_state"]

        total_t  = s["loss_total"]
        total_t1 = ns["loss_total"]

        delta_t = compute_delta_map(total_t, total_t1, eps=eps)
        ns["delta"] = delta_t

    # --- 2) Теперь проставляем delta в state ---
    # Для самого первого state в эпизоде — delta = 0
    first_state = seq[0]["state"]
    first_state["delta"] = torch.zeros_like(first_state["loss_total"])

    # Для остальных state берём delta из предыдущего next_state
    for i in range(1, len(seq)):
        prev_ns = seq[i - 1]["next_state"]
        curr_s  = seq[i]["state"]
        curr_s["delta"] = prev_ns["delta"]



def add_delta_to_all_entries(entries):
    """
    entries: список всех переходов всех экспериментов, уже отсортированный по времени внутри эксперимента.
    Если у тебя есть явные границы экспериментов (experiment_id), можно группировать и по ним.
    """
    sequences = []
    curr_seq = []

    for tr in entries:
        curr_seq.append(tr)

        # конец эпизода: done == 1 или done == -1
        if tr["done"] in (1, -1):
            sequences.append(curr_seq)
            curr_seq = []

    # хвост, если закончился на done == 0 (неполный эпизод)
    if curr_seq:
        sequences.append(curr_seq)

    # Обрабатываем каждую последовательность
    for seq in sequences:
        add_delta_to_sequence(seq)

    # entries модифицированы in-place, можно просто вернуть для удобства
    return entries



def shift_done_rewards(transitions, done = 1, shift_value= -5):
    """
    Увеличивает model_reward на shift_value для всех переходов, где done == 1 или -1.
    Возвращает изменённый список transitions.
    """
    print(f"\n🔧 Сдвигаем reward_model на {shift_value} для всех переходов (done={done})...")
    count = 0

    for tr in transitions:
        if done == 1:
            if int(tr.get("done", 0)) == done:
                # Убедиться, что reward_model существует
                if "reward_model" in tr:
                    try:
                        tr["reward_model"] = float(tr["reward_model"]) + shift_value
                        count += 1
                    except:
                        print("⚠️ Не удалось преобразовать reward_model в float:", tr["reward_model"])
                else:
                    print("⚠️ У перехода нет поля reward_model", tr)
        if done == -1:
             if int(tr.get("done", 0)) == done:
                # Убедиться, что reward_model существует
                if "reward_model" in tr:
                    try:
                        tr["reward_model"] =  shift_value
                        count += 1
                    except:
                        print("⚠️ Не удалось преобразовать reward_model в float:", tr["reward_model"])
                else:
                    print("⚠️ У перехода нет поля reward_model", tr)


    print(f"✅ Сдвинуто reward_model для {count} успешных переходов.")

    return transitions



def truncate_failure_chains_by_tol(transitions, tol=0.0, shift_reward=10.0):
    """
    Разбивает поток переходов на цепочки, которые оканчиваются done == -1.
    Внутри каждой такой цепочки ищет первый переход, где reward <= tol.

    Если такой переход найден:
    - помечает его как done = 1
    - увеличивает reward_model на reward_bonus
    - отбрасывает все последующие переходы в этой цепочке, включая исходный done == -1

    Если в цепочке подходящего перехода нет, цепочка сохраняется без изменений.
    Уже успешные эпизоды (done == 1) сохраняются как есть и сбрасывают текущую цепочку.
    """
    print(f"\n🔧 Обрезаем failure-цепочки по tol={tol}")

    cleaned = []
    current_chain = []
    truncated_chains = 0

    def flush_chain(chain):
        nonlocal truncated_chains
        if not chain:
            return

        cut_idx = None
        for idx, tr in enumerate(chain):
            reward = float(tr.get("reward", 0.0))
            done = int(tr.get("done", 0))
            if done == 0 and abs(reward) <= tol:
                cut_idx = idx
                break

        if cut_idx is None:
            cleaned.extend(chain)
            return

        tr = chain[cut_idx]
        print("\n=== ⚙️ Failure chain before modification ===")
        print({
            "reward": tr.get("reward"),
            "reward_model": tr.get("reward_model"),
            "done": tr.get("done"),
            "opt_model_i": tr.get("opt_model_i"),
        })

        tr["done"] = 1
        tr["reward_model"] = float(tr["reward_model"]) + shift_reward
        cleaned.extend(chain[:cut_idx + 1])
        truncated_chains += 1

        print("=== ✅ Failure chain after modification ===")
        print({
            "reward": tr.get("reward"),
            "reward_model": tr.get("reward_model"),
            "done": tr.get("done"),
            "opt_model_i": tr.get("opt_model_i"),
        })
        print("=" * 50)

    for tr in transitions:
        done = int(tr.get("done", 0))

        if done == 1:
            flush_chain(current_chain)
            current_chain = []
            cleaned.append(tr)
            continue

        current_chain.append(tr)

        if done == -1:
            flush_chain(current_chain)
            current_chain = []

    flush_chain(current_chain)
    print(f"✅ Обрезано failure-цепочек: {truncated_chains}")
    return cleaned


def truncate_success_chains(transitions, current_tol=0.0608023, prev_tol= 0.060776):
    """
    transitions: общий список переходов, отсортированный последовательно.
    Каждый эпизод заканчивается done = 1.
    Нужно: если reward < threshold → done = 1 + удалить все последующие в эпизоде.
    
    Возвращает новый список переходов.
    """


    cleaned = []
    episode = []
    flag_is_tail = False  # флаг, что мы в "хвосте" после успешного перехода

    for tr in transitions:
        if not flag_is_tail:
            episode.append(tr)
        else:
            print("⚠️ Пропускаем переход в хвосте после успешного завершения.")
            print(tr['reward'], tr['done'])

        reward = float(tr["reward"])
        done = int(tr["done"])

        if done == 1:
            cleaned.extend(episode)
            episode = []  # конец эпизода
            flag_is_tail = False
            continue

        # --- Успешный переход ---
        if prev_tol < abs(reward) <= current_tol:
            print("\n=== ⚙️ Data before modification ===")
            print({
                'reward': tr.get('reward'),
                'reward_model': tr.get('reward_model'),
                'done': tr.get('done'),
                'opt_model_i': tr.get('opt_model_i')
            })


            tr["done"] = 1
            tr["reward_model"] += 10
            cleaned.extend(episode)
            episode = []  # начать новый эпизод
            print("=== ✅ Data after modification ===")
            print({
                'reward': tr.get('reward'),
                'reward_model': tr.get('reward_model'),
                'done': tr.get('done'),
                'opt_model_i': tr.get('opt_model_i')
            })
            print("=" * 50)
            flag_is_tail = True
            continue

        # --- Конец эпизода ---
        if done == -1:
            if not flag_is_tail:
                cleaned.extend(episode)
                episode = []
            else:
                episode = []
            flag_is_tail = False

    # Если последний эпизод не завершился done=-1 — отбрасываем "хвост"
    # (позиционные ошибки уровня tolerance точно не должны жить вечно)
    
    return cleaned


def _safe_log1p_signed(x, eps=1e-12):
    """
    sign(x) * log(1 + |x|) для torch.Tensor или чисел.
    """

    return torch.sign(x) * torch.log1p(torch.abs(x) + eps)


def _apply_log_transform_to_state_dict_noextra(state_dict, keys=None, eps=1e-12):
    """
    Модифицирует state_dict in-place.
    НЕ добавляет новых ключей, не меняет структуру.
    - keys=None => логарифмируем ВСЕ числовые/тензорные поля.
    """
    if not isinstance(state_dict, dict):
        return

    if keys is None:
        keys = [k for k, v in state_dict.items()
                if torch.is_tensor(v) or isinstance(v, (int, float))]

    for k in keys:
        if k in state_dict:
            v = state_dict[k]
            if torch.is_tensor(v) or isinstance(v, (int, float)):
                state_dict[k] = _safe_log1p_signed(v, eps=eps)


def apply_log_transform_to_transitions(transitions, state_keys=None, eps=1e-12):
    """
    transitions: list[dict] где каждый dict имеет 'state' и 'next_state'
    Лог-трансформ к state/next_state. Никаких новых ключей не добавляем.
    """
    print(f"\n🔧 Применяем лог-трансформацию к состояниям")
    for tr in transitions:
        _apply_log_transform_to_state_dict_noextra(tr.get("state"), keys=state_keys, eps=eps)
        _apply_log_transform_to_state_dict_noextra(tr.get("next_state"), keys=state_keys, eps=eps)


def add_proj_mark(all_transitions, proj_name):
    marked = []
    for tr in all_transitions:
        tr["pde"] = proj_name

    return all_transitions

# === Точка входа ===
# if __name__ == "__main__":
#     buffer = collect_all_comet_transitions(PrioritizedReplayBuffer(capacity=100000), 1)
    # torch.save(buffer.memory, "merged_replay_buffer.pt")
    # print("💾 Буфер сохранён в merged_replay_buffer.pt")
    # exp = api.get_experiment(workspace=WORKSPACE, project_name=PROJECT_NAME, experiment='751c7ca595dd4dafb22a0cfe61c26b6f')
    # meta = exp.get_metadata()
    # exp_id = meta.get("experimentKey")
    # exp_name = meta.get("experimentName")

    # assets = exp.get_asset_list()
    # pt_assets = [a for a in assets if a["fileName"].endswith(".pt") and "entry_step" in a["fileName"]]

    # for asset in pt_assets:
    #     filename = asset["fileName"]
    #     print(asset)
