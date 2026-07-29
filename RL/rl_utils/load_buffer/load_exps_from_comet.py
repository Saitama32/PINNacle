from comet_ml import API
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import torch
from dotenv import load_dotenv
from RL.rl_algorithms import PrioritizedReplayBuffer
from RL.rl_utils.load_buffer.load_transitions_into_buffer_pickle import (
    load_transitions_to_replay_buffer,
)
from RL.rl_utils.load_buffer.rebuild_states_from_solver_models import (
    rebuild_transitions_states_from_solver_models,
)


WORKSPACE = "saitama32"
PROJECT_NAME = "rlpinn-grayscott-farm-transitions"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
api = API(api_key=os.getenv("COMET_API_KEY"))



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


def get_asset_step(asset):
    if "step" in asset and isinstance(asset["step"], (int, float)):
        return int(asset["step"])
    fname = asset.get("fileName", "")
    try:
        return int(fname.split("entry_step_")[-1].split(".")[0])
    except Exception:
        return 0


def _extract_transitions_from_payload(data_load):
    if isinstance(data_load, dict):
        if "memory" in data_load:
            return list(data_load["memory"])
        return [data_load]
    if isinstance(data_load, list):
        return list(data_load)
    return None


@dataclass
class ExperimentLoadResult:
    index: int
    exp_id: str
    exp_name: str
    transitions: list
    loaded_files: list
    skipped_files: list
    error: str = None


def load_single_experiment_transitions(exp, index, save_dir=None):
    meta = exp.get_metadata()
    exp_id = meta.get("experimentKey")
    exp_name = meta.get("experimentName")

    assets = exp.get_asset_list()
    pt_assets = [
        asset for asset in assets
        if asset["fileName"].endswith(".pt") and "entry_step" in asset["fileName"]
    ]
    pt_assets = sorted(pt_assets, key=get_asset_step)

    if not pt_assets:
        return ExperimentLoadResult(
            index=index,
            exp_id=exp_id,
            exp_name=exp_name,
            transitions=[],
            loaded_files=[],
            skipped_files=["NO_ENTRY_STEP_ASSETS"],
        )

    experiment_transitions = []
    loaded_files = []
    skipped_files = []

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

            transitions = _extract_transitions_from_payload(data_load)
            if transitions is None:
                skipped_files.append(
                    f"{filename}: unsupported format {type(data_load).__name__}"
                )
                continue

            experiment_transitions.extend(transitions)
            loaded_files.append(filename)
        except Exception as exc:
            skipped_files.append(f"{filename}: {exc}")

    return ExperimentLoadResult(
        index=index,
        exp_id=exp_id,
        exp_name=exp_name,
        transitions=experiment_transitions,
        loaded_files=loaded_files,
        skipped_files=skipped_files,
    )


def _log_experiment_result(result, running_total):
    print(f"[{result.index:2d}] {result.exp_name} ({result.exp_id})")

    if result.error:
        print(f"   ERROR loading experiment: {result.error}")
        return

    # if result.loaded_files:
    #     for filename in result.loaded_files:
    #         print(f"   loaded {filename}")
    # else:
    #     print("   no transition files were loaded")

    for skipped in result.skipped_files:
        if skipped == "NO_ENTRY_STEP_ASSETS":
            print("   no entry_step_*.pt files, skipping")
        else:
            print(f"   skipped {skipped}")

    print(
        f"   appended as one block: "
        f"{len(result.transitions)} transitions ({running_total} total)"
    )


def _resolve_num_workers(num_workers, total_experiments):
    if total_experiments <= 0:
        return 1
    if num_workers is None:
        return min(5, total_experiments)
    return max(1, min(int(num_workers), total_experiments))


def _done_value(tr):
    done = tr.get("done")
    if torch.is_tensor(done):
        if done.numel() != 1:
            return None
        done = done.detach().cpu().item()
    try:
        return int(done)
    except (TypeError, ValueError):
        return None


def _filter_terminal_without_active_chain(transitions):
    filtered = []
    chain_active = False
    skipped = 0

    for tr in transitions:
        done = _done_value(tr)
        if not chain_active and done in (1, -1):
            skipped += 1
            continue

        filtered.append(tr)
        chain_active = done not in (1, -1)

    if skipped:
        print(f"Skipped {skipped} terminal transitions without active chain.")
    return filtered


def _state_loss_matches_next_state(state, next_state, loss_key="loss_total"):
    if not isinstance(state, dict) or not isinstance(next_state, dict):
        return False
    if loss_key not in state or loss_key not in next_state:
        return False

    state_loss = state[loss_key]
    next_loss = next_state[loss_key]
    if torch.is_tensor(state_loss) and torch.is_tensor(next_loss):
        if state_loss.shape != next_loss.shape:
            return False
        return bool(torch.allclose(state_loss, next_loss, equal_nan=True))

    try:
        return float(state_loss) == float(next_loss)
    except (TypeError, ValueError):
        return False


def _zero_state_like(state):
    if not isinstance(state, dict) or "loss_total" not in state:
        return state

    zero = {}
    for key in ("loss_total", "loss_oper", "loss_bnd"):
        value = state.get(key)
        if torch.is_tensor(value):
            zero[key] = torch.zeros_like(value)
    return zero


def _repair_equal_states_from_previous_next_states(seq, loss_key="loss_total"):
    repaired = 0
    for i, tr in enumerate(seq):
        if _state_loss_matches_next_state(
            tr.get("state"),
            tr.get("next_state"),
            loss_key=loss_key,
        ):
            if i == 0:
                tr["state"] = _zero_state_like(tr.get("next_state"))
            else:
                tr["state"] = seq[i - 1]["next_state"]
            repaired += 1
    return repaired


def repair_equal_states_in_all_entries(entries, loss_key="loss_total"):
    sequences = []
    curr_seq = []

    for tr in entries:
        curr_seq.append(tr)
        if _done_value(tr) in (1, -1):
            sequences.append(curr_seq)
            curr_seq = []

    if curr_seq:
        sequences.append(curr_seq)

    repaired = 0
    for seq in sequences:
        repaired += _repair_equal_states_from_previous_next_states(
            seq,
            loss_key=loss_key,
        )

    if repaired:
        print(f"Repaired {repaired} transitions where state matched next_state.")
    return entries


def drop_chains_longer_than(transitions, max_chain_len=12, block_label=None):
    if max_chain_len is None or max_chain_len <= 0:
        return transitions

    cleaned = []
    chain = []
    dropped_chains = 0
    dropped_transitions = 0
    max_seen_len = 0

    def flush_current_chain():
        nonlocal dropped_chains, dropped_transitions, max_seen_len
        if not chain:
            return

        chain_len = len(chain)
        max_seen_len = max(max_seen_len, chain_len)
        if chain_len > max_chain_len:
            dropped_chains += 1
            dropped_transitions += chain_len
        else:
            cleaned.extend(chain)

    for tr in transitions:
        chain.append(tr)
        if _done_value(tr) in (1, -1):
            flush_current_chain()
            chain = []

    flush_current_chain()

    if dropped_chains:
        label = f" in {block_label}" if block_label else ""
        print(
            f"Dropped {dropped_chains} chains{label} with length > {max_chain_len} "
            f"({dropped_transitions} transitions dropped, max seen length {max_seen_len})."
        )

    return cleaned


def collect_all_comet_transitions(
    replay_buffer=None,
    max_exps_last=10,
    duration_grater_hours=0,
    save_dir=None,
    tolerance=0.0,
    prev_tol=0.0,
    use_tol=True,
    new_tol=False,
    use_log_state=False,
    proj_name=None,
    mark_states=None,
    num_workers=None,
    rebuild_states_from_solver_models=False,
    AE_model_params=None,
    AE_train_params=None,
    loss_surface_params=None,
    max_chain_len=12,
) -> PrioritizedReplayBuffer:
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

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Local save enabled: {save_dir}")

    transition_blocks = []
    worker_count = _resolve_num_workers(num_workers, len(experiments_sorted_tol))
    indexed_experiments = list(enumerate(experiments_sorted_tol, 1))

    if worker_count <= 1:
        experiment_results = [
            load_single_experiment_transitions(exp, index, save_dir=save_dir)
            for index, exp in indexed_experiments
        ]
    else:
        print(f"Parallel loading with {worker_count} workers.")
        experiment_results = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(load_single_experiment_transitions, exp, index, save_dir): index
                for index, exp in indexed_experiments
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    experiment_results.append(future.result())
                except Exception as exc:
                    experiment_results.append(
                        ExperimentLoadResult(
                            index=index,
                            exp_id="unknown",
                            exp_name="unknown",
                            transitions=[],
                            loaded_files=[],
                            skipped_files=[],
                            error=str(exc),
                        )
                    )

    experiment_results.sort(key=lambda result: result.index)

    for result in experiment_results:
        if result.error:
            _log_experiment_result(
                result,
                sum(len(block) for block in transition_blocks),
            )
            continue
        transition_blocks.append(result.transitions)
        _log_experiment_result(
            result,
            sum(len(block) for block in transition_blocks),
        )

    all_entries = []
    for block_index, transitions in enumerate(transition_blocks, 1):
        if not transitions:
            continue

        block_transitions = _filter_terminal_without_active_chain(transitions)

        if tolerance > 0.0 and prev_tol == 0.0 and new_tol:
            block_transitions = truncate_failure_chains_by_tol(
                block_transitions,
                tol=tolerance,
                shift_reward=10.0,
            )
        elif tolerance > prev_tol and prev_tol != 0.0:
            block_transitions = truncate_success_chains(
                block_transitions,
                current_tol=tolerance,
                prev_tol=prev_tol,
            )

        if mark_states:
            block_transitions = add_proj_mark(block_transitions, proj_name)

    # --- Сдвиг наград для успешных переходов ---
        block_transitions = shift_done_rewards(block_transitions, done=-1, shift_value=-5)
    # --- Добавление delta loss ---
        if rebuild_states_from_solver_models:
            block_entries = rebuild_transitions_states_from_solver_models(
                block_transitions,
                AE_model_params=AE_model_params,
                AE_train_params=AE_train_params,
                loss_surface_params=loss_surface_params,
            )
        else:
            block_entries = repair_equal_states_in_all_entries(block_transitions)
            block_entries = add_delta_to_all_entries(block_entries)

        block_entries = drop_chains_longer_than(
            block_entries,
            max_chain_len=max_chain_len,
            block_label=f"experiment block {block_index}",
        )
        all_entries.extend(block_entries)

    if use_log_state and not rebuild_states_from_solver_models:
        apply_log_transform_to_transitions(
            all_entries,
            state_keys=("loss_total", "loss_oper", "loss_bnd", "loss_ic"),
        )


    print(f"\n🚀 Всего собрано {len(all_entries)} переходов из {len(experiments_sorted_duration)} экспериментов.")
    if not all_entries:
        print("⚠️ Не найдено переходов для загрузки — возвращаем пустой буфер.")
        return PrioritizedReplayBuffer(capacity=1)

    replay_buffer = load_transitions_to_replay_buffer(
        replay_buffer,
        all_entries,
        prev_tol=prev_tol,
        current_tol=tolerance,
    )
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
        s = tr["state"]
        ns = tr["next_state"]

        total_t = s["loss_total"]
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
        curr_s = seq[i]["state"]
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


def shift_done_rewards(transitions, done=1, shift_value=-5):
    """
    Увеличивает model_reward на shift_value для всех переходов, где done == 1 или -1.
    Возвращает изменённый список transitions.
    """
    print(f"\n🔧 Сдвигаем reward_model на {shift_value} для всех переходов (done={done})...")
    count = 0

    for tr in transitions:
        if done == 1:
            if int(tr.get("done", 0)) == done:
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

        print("=== Failure chain after modification ===")
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
    print(f"Truncated failure chains: {truncated_chains}")
    return cleaned


def truncate_success_chains(transitions, current_tol=0.0608023, prev_tol=0.060776):
    """
    transitions: общий список переходов, отсортированный последовательно.
    Каждый эпизод заканчивается done = 1.
    Нужно: если reward < threshold → done = 1 + удалить все последующие в эпизоде.
    
    Возвращает новый список переходов.
    """
    cleaned = []
    episode = []
    flag_is_tail = False

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
            episode = []
            flag_is_tail = False
            continue

        if prev_tol < abs(reward) <= current_tol:
            print("\n=== Data before modification ===")
            print({
                "reward": tr.get("reward"),
                "reward_model": tr.get("reward_model"),
                "done": tr.get("done"),
                "opt_model_i": tr.get("opt_model_i"),
            })

            tr["done"] = 1
            tr["reward_model"] += 10
            cleaned.extend(episode)
            episode = []
            print("=== Data after modification ===")
            print({
                "reward": tr.get("reward"),
                "reward_model": tr.get("reward_model"),
                "done": tr.get("done"),
                "opt_model_i": tr.get("opt_model_i"),
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
        keys = [
            k for k, v in state_dict.items()
            if torch.is_tensor(v) or isinstance(v, (int, float))
        ]

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
    for tr in all_transitions:
        tr["pde"] = proj_name
    return all_transitions
