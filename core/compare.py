import torch
from typing import Dict, Tuple, List
from torch.utils.data import Dataset
from tabulate import tabulate
from torchvision import transforms

# Импорты для тестирования
try:
    from core.torch_onnx import test_onnx_model_cpu_timer, test_onnx_model_cuda_timer, convert_to_onnx
    from core.torch_trt import test_torch_trt_model, convert_to_torch_trt
    from core.datasets import CustomImageDataset, RandomImageDataset
    from core.utils import run_test, gpu_mem_usage, cpu_mem_usage
    from core.model import Resnet18
except ImportError:
    from torch_onnx import test_onnx_model_cpu_timer, test_onnx_model_cuda_timer, convert_to_onnx
    from torch_trt import test_torch_trt_model, convert_to_torch_trt
    from datasets import CustomImageDataset, RandomImageDataset
    from utils import run_test, gpu_mem_usage, cpu_mem_usage
    from model import Resnet18


def make_res_table(results: List[Dict], test_fn_names: Dict[str, str]):
    headers = [
        'function',
        'dataloader_type',
        'precision',
        'device',
        'shape',
        'time (batch)',
        'time (per image)',
        'allocated_memory',
        'speedup',
        'FLOPs',
        'GPU util %'
    ]

    table = []
    base_time = None
    for result in results:
        if base_time is None and result['timer_type'] == 'cpu' and result['precision'] == 'fp32':
            base_time = list(result['results'].values())[0]

    for result in results:
        for shape, time_res in result['results'].items():
            batch_size = shape[0]
            per_image_time = time_res / batch_size
            speedup = base_time / time_res if base_time is not None else 0

            table.append([
                test_fn_names[result['test_function']],
                result['dataloader'],
                result['precision'],
                result['timer_type'],
                shape,
                f'{time_res:.3f} ms',
                f'{per_image_time:.3f} ms',
                f'{result["allocated_memory"]:.1f} MB',
                f'{speedup:.1f}x',
                f'{result.get("flops", 0) / 1e9:.2f} G' if 'flops' in result else 'N/A',
                f'N/A'
            ])

    print(table)
    with open('res1.md', 'w') as f:
        f.write(tabulate(table, headers=headers, tablefmt='github'))
    return table


def test_torch_model(
        model: torch.nn.Module,
        dataset: Dataset,
        batch_step: int = 1,
        num_runs: int = 50,
        min_batch_size: int = 1,
        max_batch_size: int = 1,
        precision: str = 'fp16',
        timer_type: str = 'cuda',
        **kwargs
):
    model.eval()
    model = model.to('cuda')
    dtype = torch.float16 if precision == 'fp16' else torch.float32

    def model_wrapper(input_data):
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=dtype):
            return model(input_data)

    if timer_type == 'cuda':
        result = run_test(
            model_wrapper=model_wrapper,
            input_shape=(3, 224, 224),
            batch_step=batch_step,
            dataset=dataset,
            num_runs=num_runs,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
        )

        return result


def test_onnx(
        model_path: str,
        dataset: Dataset,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        num_runs: int = 50,
        min_batch_size: int = 1,
        opt_batch_size: int = 1,
        max_batch_size: int = 1,
        batch_step: int = 1,
        precision: str = 'fp32',
        timer_type: str = 'cuda',
        **kwargs
):
    convert_to_onnx(
        model_path=model_path,
        output_path=model_path.replace('.pth', '.onnx'),
        input_shape=input_shape,
        precision=precision,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        opt_batch_size=opt_batch_size,
    )

    if timer_type == 'cuda':
        result = test_onnx_model_cuda_timer(
            onnx_path=model_path.replace('.pth', '.onnx'),
            input_shape=input_shape,
            batch_step=batch_step,
            dataset=dataset,
            num_runs=num_runs,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
        )

        return result

    elif timer_type == 'cpu' and precision == 'fp32':
        return test_onnx_model_cpu_timer(
            onnx_path=model_path.replace('.pth', '.onnx'),
            input_shape=input_shape,
            batch_step=batch_step,
            dataset=dataset,
            num_runs=num_runs,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
        )


def test_torch_trt(
        model_path: str,
        dataset: Dataset,
        batch_step: int = 1,
        num_runs: int = 50,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        min_batch_size: int = 1,
        opt_batch_size: int = 1,
        max_batch_size: int = 1,
        precision: str = 'fp32',
        timer_type: str = 'cuda',
        **kwargs
):
    model = convert_to_torch_trt(
        model_path=model_path,
        input_shape=input_shape,
        precision=precision,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        opt_batch_size=opt_batch_size,
    )

    if timer_type == 'cuda':
        result = test_torch_trt_model(
            model=model,
            input_shape=input_shape,
            batch_step=batch_step,
            dataset=dataset,
            num_runs=num_runs,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
        )

        return result


def benchmark_models(
        model_path: str,
        num_runs: int = 50,
        image_size: int = 224,
        min_batch_size: int = 64,
        max_batch_size: int = 64,
        opt_batch_size: int = 64,
        batch_step: int = 4,
):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    real_dataset = CustomImageDataset(root_dir='../data/test', transform=transform)

    dummy_target_size = (3, image_size, image_size)
    dummy_dataset = RandomImageDataset(target_size=dummy_target_size)

    model = Resnet18()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    static_kwargs = {
        'model': model,
        'onnx_path': model_path.replace('.pth', '.onnx'),
        'input_shape': dummy_target_size,
        'model_path': model_path,
        'min_batch_size': min_batch_size,
        'max_batch_size': max_batch_size,
        'opt_batch_size': opt_batch_size,
        'batch_step': batch_step,
        'num_runs': num_runs,
    }

    kwargs = {
        'datasets': [real_dataset],
        'precisions': ['fp16', 'fp32'],
        'timer_types': ['cuda', 'cpu']
    }

    test_functions = [
        test_torch_model,
        test_onnx,
        test_torch_trt
    ]

    test_fn_names = {
        test_torch_model.__name__: 'torch',
        test_onnx.__name__: 'onnx',
        test_torch_trt.__name__: 'torch_trt'
    }

    results = []
    for precision in kwargs['precisions']:
        for test_function in test_functions:
            for dataset in kwargs['datasets']:
                for timer_type in kwargs['timer_types']:
                    mem_usage = gpu_mem_usage if timer_type == 'cuda' else cpu_mem_usage
                    print(
                        f'test params: test_function: {test_function.__name__}, dataloader: {dataset.__class__.__name__}, precision: {precision}, timer_type: {timer_type}')

                    result, allocated_memory, *extra = mem_usage(test_function)(
                        **static_kwargs,
                        dataset=dataset,
                        precision=precision,
                        timer_type=timer_type
                    )

                    if result is None:
                        continue

                    data = {
                        'test_function': test_function.__name__,
                        'dataloader': 'real' if dataset == real_dataset else 'dummy',
                        'precision': precision,
                        'timer_type': timer_type,
                        'results': result,
                        'allocated_memory': allocated_memory
                    }
                    results.append(data)

    make_res_table(results, test_fn_names)


if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"AMP supported: {torch.cuda.is_bf16_supported()}")


    sizes = [224, 256, 512]

    for i, size in enumerate(sizes):
        benchmark_models(
            model_path=f'../weights/best_resnet18_{size}.pth',
            num_runs=30,
            image_size=size,
            min_batch_size=16,
            max_batch_size=64,
            opt_batch_size=32,
            batch_step=16,
        )