import torch

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Capability: {torch.cuda.get_device_capability(0)}')
    
    # 测试简单操作
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ CUDA operations work!")
    except RuntimeError as e:
        print(f"✗ CUDA error: {e}")
else:
    print("CUDA not available - GPU not detected or not supported")
