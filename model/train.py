import torch 

def train_model(model,train_loader,test_loader,epochs,criterion,optimizer,device):
    # Training Loop
    for epoch in range(epochs):
        model.train()
        print(f"----------- Epoch : {epoch + 1} ------------------")

        for batch,(image,label) in enumerate(train_loader):
            optimizer.zero_grad()
            image,label = image.to(device),label.to(device)
            pred = model(image)
            loss = criterion(pred,label)

            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(f"Batch : [{batch}/{len(train_loader)}] Loss : {loss.item():.5f}")

        model.eval()
        correct = 0
        with torch.no_grad():
            for image, label in test_loader:
                image, label = image.to(device), label.to(device)
                pred = model(image)
                correct += (pred.argmax(1) == label).type(torch.float).sum().item()
        
        correct /= len(test_loader.dataset)
        akurasi = 100 * correct
        print(f"Akurasi : {akurasi:.4f}%")
    
    return model,akurasi
