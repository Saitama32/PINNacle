import random


class HybridHeadPolicy:
    """Greedy policy combining optimizer and parameter decisions by name."""

    def __init__(
        self,
        optimizer_agent,
        parameter_agent,
        optimizer_source,
        parameter_source,
    ):
        optimizer_names = tuple(optimizer_agent.optimizer_dict)
        parameter_names = tuple(parameter_agent.optimizer_dict)
        if optimizer_names != parameter_names:
            raise ValueError(
                "Hybrid checkpoints use different optimizer orderings: "
                f"{optimizer_names} != {parameter_names}"
            )
        self.optimizer_agent = optimizer_agent
        self.parameter_agent = parameter_agent
        self.optimizer_source = str(optimizer_source)
        self.parameter_source = str(parameter_source)

    def select_action(self, state, epsilon=0.05):
        self.optimizer_agent.steps_done += 1
        if random.random() < float(epsilon):
            action_raw = self.optimizer_agent.get_random_action()
            action = self.optimizer_agent.post_proc_model(*action_raw)
            diagnostics = {
                "optimizer_source": "epsilon-random",
                "parameter_source": "epsilon-random",
                "optimizer": action["type"],
                "optimizer_q_values": None,
            }
            return action, action_raw, False, diagnostics

        optim_class, optim_name, q_values = (
            self.optimizer_agent.select_optimizer_greedy(state)
        )
        epochs_class, param_class = (
            self.parameter_agent.select_parameters_greedy(state, optim_name)
        )
        action_raw = (optim_class, epochs_class, param_class)
        action = self.optimizer_agent.post_proc_model(*action_raw)
        diagnostics = {
            "optimizer_source": self.optimizer_source,
            "parameter_source": self.parameter_source,
            "optimizer": optim_name,
            "optimizer_q_values": q_values,
        }
        return action, action_raw, True, diagnostics
