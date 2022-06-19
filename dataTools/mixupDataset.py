import torch

class MixupDataset():
    def __init__(self) -> None:
        self.psudo_labels = None
        pass

    def update_psudos(self, data_loader, model, device):
        self.indexes, self.psudo_labels = _get_predicted_scores(data_loader, model, device)


def _get_predicted_scores(data_loader, model, device):
    model.eval()
    predicted_scores = []
    indexes = []

    with torch.no_grad():
        for _, (index, Xs, Ys) in enumerate(data_loader):
            Xs = Xs.to(device)
            Ys = Ys.to(device)
            outputs = model(Xs).squeeze()

            outputs = torch.sigmoid(outputs)

            predicted_scores.append(outputs)
            indexes.append(index.squeeze())
    
    predicted_scores = torch.cat(predicted_scores)
    indexes = torch.cat(indexes)

    return indexes, predicted_scores