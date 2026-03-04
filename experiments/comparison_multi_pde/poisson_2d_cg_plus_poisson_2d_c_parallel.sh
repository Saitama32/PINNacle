#!/bin/bash
# Автоматический запуск poisson_2d_cg_chain.py с распределением по GPU

SCRIPT="examples/examples_PINNacle/comparison_multi_pde/poisson_2d_cg_pinnacle_rl_comparison.py"
SCRIPT_2="examples/examples_PINNacle/comparison_multi_pde/poisson_2d_claasic_pinnacle_rl_comparison.py"

# Проверяем, сколько доступно GPU
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

log_enable="True"
log_unenable="False"
exp_key_1="7f7a91cef55d4aeba0e509024977456b"
echo "Обнаружено GPU: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "❌ Не найдено ни одного CUDA-устройства. Выходим."
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Запускаем 2 процесса на одной GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$log_enable" --exp_key "$exp_key_1"&
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_2" --log_key "$log_enable" --exp_key "$exp_key_1"&
elif [ "$NUM_GPUS" -ge 2 ]; then
    echo "Запускаем по 2 процесса на каждую из двух GPU..."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" --log_key "$log_enable" --exp_key "$exp_key_1"&
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_2" --log_key "$log_enable" --exp_key "$exp_key_1"&
    # CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key "$log_enable" --exp_key "$exp_key_3"&
    # CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" --log_key "$log_enable" --exp_key "$exp_key_4"&
else
    echo "⚠️ Найдено более 2 GPU, но используется только первые две."
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" &
    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" &
    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" &
fi

# Ждём завершения всех процессов
wait
echo "✅ Все процессы завершены."
