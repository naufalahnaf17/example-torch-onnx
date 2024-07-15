import torch
from torch import nn
from datasets import get_datasets,transform_to_dataLoader
from train import train_model
from model import get_model
from visualize import visualize_data

def main():
    # Memuat Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Memuat Dataset
    train_data,test_data = get_datasets()

    # # # Visualisasikan data jika diperlukan
    # # visualize_data(train_data)
    # # visualize_data(test_data)
    
    # Membuat dataset jadi format DataLoader [batch_size,c,h,w]
    train_loader = transform_to_dataLoader(train_data,64,True)
    test_loader = transform_to_dataLoader(test_data,64,False)

    # Memuat Stuktur Model
    model = get_model()
    print(f"Struktur Model : {model}")

    # Test input model apakah sudah sesuai atau belum
    dummy_input = torch.rand(1,1,28,28).to(device)
    output = model(dummy_input)
    print(f"Dummy Output : {output}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    # Training and Validation
    trained_model,akurasi_model = train_model(
        model,
        train_loader,
        test_loader,
        epochs=15,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    print(f"Akurasi Model : {akurasi_model:.4f}%")

    # Save Model
    torch.save(trained_model.state_dict(),"./output/mnist_model.pth")

    # Load Model
    serve_model = get_model()
    serve_model.load_state_dict(torch.load('./output/mnist_model.pth'))
    print(f"Model Loaded : {serve_model}")

    # Save Model to ONNX Format
    torch_model = serve_model
    torch_input = torch.rand(1,1,28,28)
    scripted_model = torch.jit.trace(torch_model, torch_input)
    torch.onnx.export(scripted_model, torch_input, "./output/mnist_model.onnx", input_names=["input"], output_names=["output"])
    print(f"Model Saved to ONNX Format")

if __name__ == "__main__":
    # Load Dataset
    main()