from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch
from evaluation import *
from loss import *
import cv2
from torch.utils.tensorboard import SummaryWriter

def traverseDataset(model: nn.Module, loader: DataLoader,
                    description, device, funcLoss,
                    log_writer: SummaryWriter, log_section, optimizer=None,
                    show_result=False, thresh_value=None, ):
    is_training = (optimizer != None)

    import time
    time_start = time.time()
    with tqdm(loader, unit="batch") as tepoch:
        total_loss = 0
        ls_eval_result = []
        model.train(is_training)
        for i, (name, data, label) in enumerate(tepoch):
            tepoch.set_description(description)
            data = data.to(device)
            label = label.to(device)

            eval_result = {}
            if is_training:
                out = model(data)
                if type(out) != list:
                    out = [out]
                loss = 0
                for x in out:
                    loss += funcLoss(x, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:  # eval
                with torch.no_grad():
                    out = model(data)
                    loss = funcLoss(out, label)
                for index in range(loader.batch_size):
                    pred = out[index][0].detach().cpu().numpy()
                    gt = label[index][0].detach().cpu().numpy()
                    id = name[0].split(".")[0]
                    eval_result = calc_result(pred, gt, thresh_value=thresh_value)

            eval_result["loss"] = float(loss) / loader.batch_size
            ls_eval_result.append(eval_result)

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            tepoch.set_postfix(avg_loss='{:.3f}'.format(avg_loss), curr_loss='{:.3f}'.format(loss.item()))

    time_end = time.time()
    avg_ms = (time_end - time_start) * 1000 / len(loader) / loader.batch_size

    num_params = sum([param.nelement() for param in model.parameters()])

    result = avg_result(ls_eval_result)
    result['avg_ms'] = avg_ms
    result['num_params'] = num_params

    return result
