import torch
import tempfile

def log_priority_to_comet(exp, priority, step):
    """
    Логирует список или тензор priority как временный .pt файл в Comet.
    """
    if not isinstance(priority, torch.Tensor):
        priority = torch.tensor(priority)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        file_path = tmp_file.name

    torch.save(priority, file_path)
    exp.log_asset(
        file_path,
        file_name=f"priority_step_{step}.pt",
        step=step,
        metadata={"type": "priority_tensor", "agent_step": step}
    )
