import re

import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune


class ZOCGE(torch.optim.Optimizer):
    """Zeroth-order coordinate gradient estimator with optional pruning."""

    def __init__(
        self,
        params,
        lr=1e-3,
        mu=1e-3,
        weight_decay=0.0,
        sparsity=0.0,
        prune_method="random",
        remask_interval=0,
        feature_reuse=False,
        grasp_sample_size=0,
    ):
        defaults = dict(
            lr=lr,
            mu=mu,
            weight_decay=weight_decay,
            sparsity=sparsity,
            prune_method=prune_method,
            remask_interval=remask_interval,
            feature_reuse=feature_reuse,
            grasp_sample_size=grasp_sample_size,
        )
        super().__init__(params, defaults)
        self.model = None
        self.losses = None
        self._mask_ready = False
        self._mask_version = 0
        self._global_step = 0

    def attach_model(self, model):
        self.model = model

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.zero_()

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("ZOCGE requires a closure.")
        if self.model is None:
            raise RuntimeError("ZOCGE must be attached to a model before training.")

        with torch.enable_grad():
            if (not self._mask_ready) or self._should_remask():
                self._prepare_mask(closure)

            base_loss, cache = self._evaluate_closure(
                closure,
                return_intermediates=self._feature_reuse_enabled(),
            )
            base_value = base_loss.detach()
            params_info = self._iter_active_coordinates()

            self.zero_grad(set_to_none=True)
            for name, param, coord_indices in params_info:
                grad_flat = torch.zeros(param.numel(), device=param.device, dtype=param.dtype)
                start_idx = self._parameter_to_layer_index(name)
                for coord_idx in coord_indices:
                    coord = int(coord_idx)
                    with torch.no_grad():
                        param.view(-1)[coord] += self._option("mu")
                    perturbed_loss = self._evaluate_perturbed_closure(
                        closure,
                        cache,
                        start_idx,
                    )
                    grad_flat[coord] = (
                        perturbed_loss.detach() - base_value
                    ) / self._option("mu")
                    with torch.no_grad():
                        param.view(-1)[coord] -= self._option("mu")
                param.grad = grad_flat.view_as(param)

            self._apply_update()
            self._global_step += 1
            return base_loss.detach()

    def _option(self, name):
        return self.param_groups[0].get(name, self.defaults[name])

    def _feature_reuse_enabled(self):
        return bool(self._option("feature_reuse"))

    def _should_remask(self):
        remask_interval = int(self._option("remask_interval"))
        if remask_interval <= 0 or self._global_step == 0:
            return False
        return self._global_step % remask_interval == 0

    def _prepare_mask(self, closure):
        dense_weights = self._capture_dense_linear_weights()
        self._remove_existing_pruning()
        sparsity = float(self._option("sparsity"))
        prune_method = self._option("prune_method")
        if sparsity > 0:
            if prune_method == "random":
                self._apply_random_pruning(sparsity)
            elif prune_method == "zo_grasp":
                self._apply_zo_grasp_pruning(sparsity, closure)
            else:
                raise ValueError(f"Unknown ZOCGE prune_method: {prune_method}")
            self._restore_dense_linear_weights(dense_weights)
        self._refresh_param_groups()
        self._mask_ready = True
        self._mask_version += 1
        print(
            f"ZOCGE mask prepared: method={prune_method}, "
            f"sparsity={sparsity:.3f}, version={self._mask_version}"
        )

    def _remove_existing_pruning(self):
        for module in self._linear_modules():
            if torch_prune.is_pruned(module) and hasattr(module, "weight_orig"):
                torch_prune.remove(module, "weight")

    def _capture_dense_linear_weights(self):
        dense_weights = {}
        for module_name, module in self.model.net.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if hasattr(module, "weight_orig"):
                dense_weights[module_name] = module.weight_orig.detach().clone()
            else:
                dense_weights[module_name] = module.weight.detach().clone()
        return dense_weights

    def _restore_dense_linear_weights(self, dense_weights):
        with torch.no_grad():
            for module_name, dense_weight in dense_weights.items():
                module = self._get_module_by_name(module_name)
                if hasattr(module, "weight_orig"):
                    module.weight_orig.copy_(dense_weight)
                else:
                    module.weight.copy_(dense_weight)

    def _apply_random_pruning(self, sparsity):
        parameters = [(module, "weight") for module in self._linear_modules()]
        if not parameters:
            return
        torch_prune.global_unstructured(
            parameters=parameters,
            pruning_method=torch_prune.RandomUnstructured,
            amount=sparsity,
        )

    def _apply_zo_grasp_pruning(self, sparsity, closure):
        scores = self._compute_zo_grasp_scores(closure)
        parameters = list(scores.keys())
        if not parameters:
            return
        torch_prune.global_unstructured(
            parameters=parameters,
            pruning_method=torch_prune.L1Unstructured,
            amount=sparsity,
            importance_scores=scores,
        )

    def _compute_zo_grasp_scores(self, closure):
        grasp_sample_size = int(self._option("grasp_sample_size"))
        if grasp_sample_size <= 0:
            grasp_sample_size = 1

        params = self._extract_linear_weight_params()
        base_value = self._evaluate_params_loss(closure, params)
        g0 = self._rge_estimate(closure, params, grasp_sample_size, base_value)
        modified_params = {
            key: value.detach().clone() + g0[key] * self._option("mu")
            for key, value in params.items()
        }
        modified_base = self._evaluate_params_loss(closure, modified_params)
        g1 = self._rge_estimate(closure, modified_params, grasp_sample_size, modified_base)

        scores = {}
        for module_name, module in self.model.net.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            param_key = f"{module_name}.weight"
            hessian_grad = (g1[param_key] - g0[param_key]) / self._option("mu")
            scores[(module, "weight")] = (-module.weight.detach() * hessian_grad).abs()
        return scores

    def _extract_linear_weight_params(self):
        params = {}
        for module_name, module in self.model.net.named_modules():
            if isinstance(module, nn.Linear):
                params[f"{module_name}.weight"] = module.weight.detach().clone()
        return params

    def _evaluate_params_loss(self, closure, params_dict):
        backup = {}
        with torch.no_grad():
            for key, value in params_dict.items():
                module_name, attr = key.rsplit(".", 1)
                module = self._get_module_by_name(module_name)
                param = getattr(module, attr)
                backup[key] = param.detach().clone()
                param.copy_(value)
        try:
            return closure(skip_backward=True).detach()
        finally:
            with torch.no_grad():
                for key, value in backup.items():
                    module_name, attr = key.rsplit(".", 1)
                    module = self._get_module_by_name(module_name)
                    getattr(module, attr).copy_(value)

    def _rge_estimate(self, closure, params_dict, sample_size, base_value):
        grads = {key: torch.zeros_like(value) for key, value in params_dict.items()}
        for _ in range(sample_size):
            perturb = {}
            perturbed_params = {}
            for key, value in params_dict.items():
                direction = torch.randn_like(value)
                direction = direction / (torch.norm(direction) + 1e-8)
                direction = direction * self._option("mu")
                perturb[key] = direction
                perturbed_params[key] = value + direction
            directional_derivative = (
                self._evaluate_params_loss(closure, perturbed_params) - base_value
            ) / self._option("mu")
            for key in grads:
                grads[key] += perturb[key] * directional_derivative / sample_size
        return grads

    def _evaluate_closure(self, closure, return_intermediates=False):
        result = closure(
            skip_backward=True,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            loss, cached_intermediates = result
            return loss, cached_intermediates
        return result, None

    def _evaluate_perturbed_closure(self, closure, cached_intermediates, start_idx):
        if (
            cached_intermediates is not None
            and start_idx is not None
            and self._feature_reuse_enabled()
        ):
            return closure(
                skip_backward=True,
                cached_intermediates=cached_intermediates,
                starting_id=start_idx,
            )
        return closure(skip_backward=True)

    def _refresh_param_groups(self):
        params = list(self.model.net.parameters()) + list(
            getattr(self.model, "external_trainable_variables", [])
        )
        for group in self.param_groups:
            group["params"] = params

    def _iter_active_coordinates(self):
        buffers = dict(self.model.net.named_buffers())
        params_info = []

        for name, param in self.model.net.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("weight_orig"):
                mask_name = name.replace("weight_orig", "weight_mask")
                mask = buffers.get(mask_name)
                if mask is None:
                    raise RuntimeError(f"Missing pruning mask for parameter {name}")
                coord_indices = mask.reshape(-1).nonzero(as_tuple=False).reshape(-1).tolist()
            else:
                coord_indices = list(range(param.numel()))
            params_info.append((name, param, coord_indices))

        external_trainables = getattr(self.model, "external_trainable_variables", [])
        for idx, param in enumerate(external_trainables):
            if not getattr(param, "requires_grad", False):
                continue
            params_info.append((f"__external_{idx}", param, list(range(param.numel()))))
        return params_info

    def _apply_update(self):
        lr = float(self._option("lr"))
        weight_decay = float(self._option("weight_decay"))
        with torch.no_grad():
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    grad = param.grad
                    if weight_decay > 0:
                        grad = grad + weight_decay * param
                    param.add_(grad, alpha=-lr)

    def _linear_modules(self):
        return [
            module
            for module in self.model.net.modules()
            if isinstance(module, nn.Linear)
        ]

    def _parameter_to_layer_index(self, param_name):
        if hasattr(self.model.net, "parameter_to_layer_index"):
            return self.model.net.parameter_to_layer_index(param_name)
        match = re.search(r"linears\.(\d+)\.(weight|weight_orig|bias)$", param_name)
        if match:
            return int(match.group(1))
        return None

    def _get_module_by_name(self, module_name):
        module = self.model.net
        if not module_name:
            return module
        for part in module_name.split("."):
            module = getattr(module, part)
        return module
