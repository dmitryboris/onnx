from ast import mod
import torch
import torch_tensorrt as t_trt
import os
from typing import Tuple
from torch.utils.data import Dataset

try:
    from core.model import Resnet18
    from core.datasets import CustomImageDataset, RandomImageDataset
    from core.utils import run_test
except ImportError:
    from model import Resnet18
    from datasets import CustomImageDataset, RandomImageDataset
    from utils import run_test


def convert_to_torch_trt(
    model_path: str,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    precision: str = 'fp16',
    workspace_size: int = 1 << 30,
    min_batch_size: int = 1,
    max_batch_size: int = 1,
    opt_batch_size: int = 1,
    **kwargs
):
    """
    Конвертирует PyTorch модель в TensorRT через torch-tensorrt
    
    Args:
        model_path: Путь к сохраненной PyTorch модели
        output_path: Путь для сохранения TensorRT модели
        input_shape: Форма входного тензора
        precision: Точность (fp32, fp16, int8)
        workspace_size: Размер рабочего пространства в байтах
    
    Returns:
        Путь к сохраненной TensorRT модели
    """
    # Загружаем модель
    model = Resnet18()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model = model.to('cuda')
    model.eval()
    
    # Создаем директорию если не существует
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    precision = torch.float16 if precision == 'fp16' else torch.float32
    min_shape = (min_batch_size, *input_shape)
    opt_shape = (opt_batch_size, *input_shape)
    max_shape = (max_batch_size, *input_shape)
    
    # Конвертируем модель
    trt_model = t_trt.compile(
        model,
        inputs=[t_trt.Input(
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
            dtype=torch.float32
        )],
        enabled_precisions={precision},
        workspace_size=workspace_size
    )

    inputs = [
        torch.randn(min_shape, device='cuda'),
        torch.randn(opt_shape, device='cuda'),
        torch.randn(max_shape, device='cuda')
    ]
    
    # Сохраняем модель
    t_trt.save(trt_model, model_path.replace('.pth', '.trt'), inputs=inputs)
    
    print(f"Модель успешно конвертирована в TensorRT (torch-tensorrt): {model_path.replace('.pth', '.trt')}")
    return trt_model

def test_torch_trt_model(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int] = (3, 224, 224),
    num_runs: int = 1000,
    min_batch_size: int = 1,
    max_batch_size: int = 1,
    batch_step: int = 1,
    dataset: Dataset = None,
    **kwargs
) -> dict[Tuple[int, int, int], float]:
    """
    Тестирует torch-tensorrt модель
    
    Args:
        model_path: Путь к модели
        input_shape: Форма входного тензора
        num_runs: Количество прогонов
        min_batch_size: Минимальный размер батча
        max_batch_size: Максимальный размер батча
        dataloader: Даталоадер для тестирования
    
    Returns:
        Словарь с результатами тестирования
    """
    return run_test(
        model_wrapper=model,
        input_shape=input_shape,
        num_runs=num_runs,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        batch_step=batch_step,
        dataset=dataset,
        timer_type='cuda'
    )