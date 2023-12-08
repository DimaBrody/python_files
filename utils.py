import torch
import numpy as np

# print("HELLO")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += np.sum(val)
        self.count += n
        self.avg = self.sum / self.count


def get_score(y_val, y_pred, metrics, scalers, target_name="target", is_scaled=True):
    all_scores = {}
    for metric in metrics:
        metric_scores = []
        for i in range(len(y_val)):
            #             print("VAL:", torch.round(y_val[i], decimals=3))
            #             print("PRED:", scalers["target"].inverse_transform(y_pred[i].reshape(-1, 1)).round(3))
            if is_scaled:
                score = metric(scalers[target_name].inverse_transform(y_val[i].reshape(-1, 1)),
                               scalers[target_name].inverse_transform(y_pred[i].reshape(-1, 1)))
            else:
                score = metric(y_val[i], y_pred[i])
            metric_scores.append(score)
        all_scores[str(metric.__name__)] = np.mean(metric_scores)
    return all_scores


def train_fn(fold, train_dataloader, model, loss_fn, optimizer, epoch, device):
    losses = AverageMeter()
    model.train()
    for step, (inputs, seq_lens, image, inputs_stocks, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(inputs, seq_lens, image, inputs_stocks)
        loss = loss_fn(outputs, labels)
        losses.update(loss.item(), labels.size(0))
        loss.backward()
        optimizer.step()
    return losses.avg


def valid_fn(valid_dataloader, model, loss_fn, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    for step, (input_ids, seq_lens, image, inputs_stocks, labels) in enumerate(valid_dataloader):
        batch_size = labels.size(0)
        with torch.no_grad():
            outputs = model(input_ids, seq_lens, image, inputs_stocks)
            loss = loss_fn(outputs, labels)
        losses.update(loss.item(), batch_size)
        preds.append(outputs.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return losses.avg, predictions
