from comet_ml import API
import torch
import io
from dotenv import load_dotenv
import os
import re
from typing import Optional
# === Настройки ===
WORKSPACE = "saitama32"
PROJECT_NAME = "rlpinn"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
api_key = os.getenv("COMET_API_KEY")


api = API(api_key=api_key)  # или просто API()
# experiment_key = "9da803bf471942d68069d835e2f95651"
def _asset_step(asset) -> int:
    step = asset.get("step")
    if step is None:
        step = asset.get("metadata", {}).get("step")
    if step is None:
        match = re.search(r"_step_(\d+)", asset.get("fileName", ""))
        if match:
            step = match.group(1)
    return int(step or 0)


def load_rl_agent_from_comet(
    experiment_key,
    step: Optional[int] = None,
    map_location: str = "cpu",
):
    """
    Загружает веса RL-агента (model_optim и model_params) из эксперимента Comet ML.
    
    Args:
        rl_agent: экземпляр твоего DQNAgent
        experiment_key (str): ключ эксперимента Comet (например, 'c4e3a8ff9112457d8c674fb68e3817c0')
        step (int|None): шаг, для которого загрузить модели (если None — берётся последний)
        workspace, project: имя workspace и проекта
        api_key: API ключ (если None — берётся из конфига)
    """
    exp = api.get_experiment(workspace=WORKSPACE, project_name=PROJECT_NAME, experiment=experiment_key)
    assets = exp.get_asset_list()

    # --- фильтруем по подпапкам ---
    # --- старая структура ---
    # optim_assets = [a for a in assets if a.get("dir") == "models/rl_agent_optim"]
    # params_assets = [a for a in assets if a.get("dir") == "models/rl_agent_params"]


    # новая схема
    # if not optim_assets or not params_assets:
    optim_assets = [
        a for a in assets
        if "model_optim_step_" in a.get("fileName", "")
    ]

    params_assets = [
        a for a in assets
        if "model_params_step_" in a.get("fileName", "")
    ]

    if not optim_assets or not params_assets:
        raise ValueError(
            "❌ Не найдены модели ни в старой структуре "
            "(models/rl_agent_optim, models/rl_agent_params), "
            "ни в новой (others/rl_model_snapshots)"
        )

    # --- сортируем по step ---
    optim_assets.sort(key=_asset_step)
    params_assets.sort(key=_asset_step)

    # --- выбираем нужный шаг ---
    if step is None:
        optim_asset = optim_assets[-1]
        params_asset = params_assets[-1]
        print(f"⬇️ Загружаем последние версии моделей {experiment_key}: step={_asset_step(optim_asset)}/{_asset_step(params_asset)}")
    else:
        # ищем ближайшие по step
        optim_asset = min(optim_assets, key=lambda a: abs(_asset_step(a) - step))
        params_asset = min(params_assets, key=lambda a: abs(_asset_step(a) - step))
        print(f"⬇️ Загружаем модели для шага, ближайшего к {step}: "
              f"optim_step={_asset_step(optim_asset)}, params_step={_asset_step(params_asset)}")

    # === загрузка model_optim ===
    print(f"📦 Загрузка {optim_asset['fileName']} ...")
    optim_bytes = exp.get_asset(optim_asset["assetId"], return_type="binary")
    model_optim_state = torch.load(io.BytesIO(optim_bytes), map_location=map_location)
    print(f"✅ model_optim ({optim_asset['fileName']}) загружен.")

    # === загрузка model_params ===
    print(f"📦 Загрузка {params_asset['fileName']} ...")
    params_bytes = exp.get_asset(params_asset["assetId"], return_type="binary")
    model_params_state = torch.load(io.BytesIO(params_bytes), map_location=map_location)
    print(f"✅ model_params ({params_asset['fileName']}) загружен.")

    print("🎯 Оба state_dict успешно загружены из Comet.")
    return model_optim_state, model_params_state

# if __name__ == "__main__":

#     from tedeous.rl_algorithms import DQNAgent

#     rl_agent = DQNAgent(optimizer_dict=my_opt_dict, device="cuda:0")


    
