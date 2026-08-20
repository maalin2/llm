import torch

def main():
    print("pytorch: ", torch.__version__)
    print("cuda available: ", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    x = torch.tensor([1,2,3])
    y = torch.tensor([10,20,30])

    print("cpu result: ", x + y)

main()
