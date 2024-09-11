import torch

# FGSM attack code
def fgsm_input(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, -1, 1)
    # Return the perturbed image
    return perturbed_image

def fgsm_attack(model, data_loader, criterion, epsilon):
    correct = 0
    model.eval()
    for x,y in data_loader:
        x.requires_grad = True
        output = model(x)
        
        init_pred = output.max(1, keepdim=True)[1]
        #print(init_pred)
    
        if init_pred.item() != y.max(1,keepdim=True)[1].item():
            continue

        _, classtype = torch.max(y,1)
        loss = criterion(output, classtype)
        model.zero_grad()
        loss.backward()
    
        # TODO: denorm
        perturbed_data = fgsm_input(x, epsilon, x.grad.data)
        # TODO: normalize

        output = model(perturbed_data)
        prediction = torch.argmax(output, dim=1)
        target = torch.argmax(y, dim=1)
        if prediction.item()==target.item():
            correct+=1

    return correct

