import modal

app = modal.App("diffusion-model-test")

image = modal.Image.from_registry(f"nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04", add_python="3.11")\
        .run_commands("apt update")\
        .apt_install("nvtop", "git", "git-lfs")\
        .pip_install(
            "sympy",
            "tqdm",
            "torch",
            "torchvision",
            "ipdb",
            "diffusers",
            "accelerate",
            "git+https://github.com/fal-ai-community/alphabet-dataset",
            "datasets",
        )\
        .add_local_dir("mindiffusion", remote_path="/root/mindiffusion")\
        .add_local_file("train_mnist.py", remote_path="/root/train_mnist.py")

content_volume = modal.Volume.from_name(
    "diffusion-model-test-content3", create_if_missing=True
)

@app.function(
    gpu="T4",
    image=image,
    timeout=24000,
    volumes={"/root/contents": content_volume},
)
def train():
    from train_mnist import train_mnist
    modal.interact()
    train_mnist(device="cuda:0")
