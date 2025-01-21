def hello_torch() -> None:
    import torch

    _ = torch.zeros(1337)
    print("Hello, PyTorch")


if __name__ == "__main__":
    hello_torch()
