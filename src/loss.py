def yolo_loss(input, target, mask):
    loss = ((input - target) ** 2 * mask).sum()
    return loss
