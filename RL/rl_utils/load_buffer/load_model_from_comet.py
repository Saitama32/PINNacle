from comet_ml import API
import torch
import io

# === Настройки ===
WORKSPACE = "saitama32"
PROJECT_NAME = "rlpinn"


api = API(api_key="aP71fQTYPNqfsYWvudPPmoBl5")  # или просто API()
# experiment_key = "9da803bf471942d68069d835e2f95651"
step=None

def load_rl_agent_from_comet(experiment_key, map_location: str = "cpu"):
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
    optim_assets = [a for a in assets if a.get("dir") == "models/rl_agent_optim"]
    params_assets = [a for a in assets if a.get("dir") == "models/rl_agent_params"]

    if not optim_assets or not params_assets:
        raise ValueError("❌ В эксперименте нет моделей rl_agent_optim или rl_agent_params")

    # --- сортируем по step ---
    optim_assets.sort(key=lambda a: a.get("step", 0))
    params_assets.sort(key=lambda a: a.get("step", 0))

    # --- выбираем нужный шаг ---
    if step is None:
        optim_asset = optim_assets[-1]
        params_asset = params_assets[-1]
        print(f"⬇️ Загружаем последние версии моделей {experiment_key}: step={optim_asset['step']}/{params_asset['step']}")
    else:
        # ищем ближайшие по step
        optim_asset = min(optim_assets, key=lambda a: abs(a.get("step", 0) - step))
        params_asset = min(params_assets, key=lambda a: abs(a.get("step", 0) - step))
        print(f"⬇️ Загружаем модели для шага, ближайшего к {step}: "
              f"optim_step={optim_asset['step']}, params_step={params_asset['step']}")

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


    