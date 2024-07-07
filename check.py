import torch

def check_cuda():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Number of CUDA devices:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("CUDA device capability:", torch.cuda.get_device_capability(0))
        
        # Set the device to GPU
        device = torch.device("cuda")
        
        # Create large random tensors on the GPU
        x = torch.rand(10000, 10000, device=device)
        y = torch.rand(10000, 10000, device=device)
        w = torch.rand(10000, 10000, device=device)
        v = torch.rand(10000, 10000, device=device)
        
        print("Starting multiple matrix multiplications on GPU...")
        # Perform multiple matrix multiplications
        z1 = torch.matmul(x, y)
        z2 = torch.matmul(z1, w)
        z3 = torch.matmul(z2, v)
        
        # Additional element-wise operations
        z4 = torch.sin(z3) + torch.cos(z3)
        
        # Confirm computation is done
        print("Multiple matrix multiplications and element-wise operations completed.")
        print("Tensor z4 (result of operations) shape:", z4.shape)
    else:
        print("CUDA is not available. Please check your CUDA installation.")

if __name__ == "__main__":
    check_cuda()
