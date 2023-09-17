def create_forward(opt, *argv):
    from .trainer import Net
    model = Net()
    model.initialize(opt)
    return model

