import torch


def pgd_attack(model, data_loader, criterion, eps, alpha, iters):
    correct = 0
    model.eval()
    for x, y in data_loader:
        x.requires_grad = True
        output = model(x)

        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != y.max(1, keepdim=True)[1].item():
            continue
        _, classtype = torch.max(y, 1)

        original_x = x.data

        for _ in range(iters):
            x.requires_grad = True
            output = model(x)
            model.zero_grad()
            loss = criterion(output, classtype)
            loss.backward()

            adv_x = x + alpha * x.grad.sign()
            eta = torch.clamp(adv_x - original_x, min=-eps, max=eps)
            x = torch.clamp(original_x + eta, min=-1, max=1).detach_()

        output = model(x)
        prediction = torch.argmax(output, dim=1)
        target = torch.argmax(y, dim=1)
        if prediction.item() == target.item():
            correct += 1

    return correct
