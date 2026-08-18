import torch
print('torch      :', torch.__version__)
print('xpu 가용   :', torch.xpu.is_available())
print('디바이스 수:', torch.xpu.device_count() if torch.xpu.is_available() else 0)
if torch.xpu.is_available():
    for i in range(torch.xpu.device_count()):
        print(f'  [{i}]', torch.xpu.get_device_name(i))
    a = torch.randn(4096, 4096, device='xpu'); b = torch.randn(4096, 4096, device='xpu')
    c = (a @ b).sum().item(); torch.xpu.synchronize()
    print('matmul 통과:', c)
    