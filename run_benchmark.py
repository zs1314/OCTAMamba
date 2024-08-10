import shutil
import torch
from dataset import *
from utils import *
from settings_benchmark import *
from dataset import writer
from torch.utils.tensorboard import SummaryWriter

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # torch.use_deterministic_algorithms(True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    print("seet seed：",seed)

seed(0)

all_dataset = prepareDatasets()
print(f"Models: {[name for name in models]}")
print(f"Datasets: {[name for name in all_dataset]}")

# Self-test: tries to load each model once to make sure that each model is loaded
print("Trying to load each our_model...")
for name_model in models:
    model: nn.Module = models[name_model]()

root_result = "result"
if not os.path.exists(root_result):
    os.mkdir(root_result)

id_card = 0
# Manual Graphics Card Selection
count_card = torch.cuda.device_count()
if count_card > 1:
    while True:
        s = input(f"Please choose a video card number (0-{count_card - 1}): ")
        if s.isdigit():
            id_card = int(s)
            if id_card >= 0 and id_card < count_card:
                break
        print("Invalid input!")
        continue
device_cuda = torch.device(f'cuda:{id_card}' if torch.cuda.is_available() else 'cpu')
print(f"\n\nVideo Card {id_card} will be used.")

for name_model in models:
    root_result_model = os.path.join(root_result, name_model)
    if not os.path.exists(root_result_model):
        os.mkdir(root_result_model)
    # foo = models[name_model]()
    # total = sum([param.nelement() for param in foo.parameters()])
    # print("Model:{}, Number of parameter: {:.3f}M".format(name_model, total/1e6))
    # continue

    # Training on individual training sets
    for name_dataset in all_dataset:
        dataset = all_dataset[name_dataset]
        trainLoader = DataLoader(dataset=dataset['train'], batch_size=2, shuffle=True, drop_last=False, num_workers=3)
        valLoader = DataLoader(dataset=dataset['val'])
        testLoader = DataLoader(dataset=dataset['test'])
        model: nn.Module = models[name_model]().to(device_cuda)

        root_result_model_dataset = os.path.join(root_result_model, name_dataset)
        path_flag = os.path.join(root_result_model_dataset, f"finished.flag")
        if os.path.exists(path_flag):
            continue
        if os.path.exists(root_result_model_dataset):
            shutil.rmtree(root_result_model_dataset)
        os.mkdir(root_result_model_dataset)

        print(f"\n\n\nCurrent Model:{name_model}, Current training dataset: {name_dataset}")

        log_section = f"{name_model}_{name_dataset}"

        funcLoss = DiceLoss() if 'loss' not in dataset else dataset['loss']
        thresh_value = None if 'thresh' not in dataset else dataset['thresh']
        optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad],
                                     lr=0.0001, weight_decay=0.001)

        NUM_MAX_EPOCH = 400
        bestResult = {"epoch": -1, "dice": -1}
        ls_best_result = []
        for epoch in range(NUM_MAX_EPOCH):
            torch.cuda.empty_cache()
            """trainset"""
            log_section_parent = f"{log_section}"
            result_train = traverseDataset(model=model, loader=trainLoader,
                                           thresh_value=thresh_value,
                                           log_section=f"{log_section_parent}_{epoch}_train",
                                           log_writer=writer if epoch % 5 == 0 else None,
                                           description=f"Train Epoch {epoch}", device=device_cuda,
                                           funcLoss=funcLoss, optimizer=optimizer)

            for key in result_train:
                writer.add_scalar(tag=f"{log_section}/{key}_train",
                                  scalar_value=result_train[key],
                                  global_step=epoch
                                  )

            """Validation - validation is performed on the validation set, and the correlation results between the training set and the validation set are saved in the tensorboard"""
            result = traverseDataset(model=model, loader=valLoader,
                                     thresh_value=thresh_value,
                                     log_section=f"{log_section_parent}_{epoch}_val",
                                     log_writer=writer if epoch % 5 == 0 else None,
                                     description=f"Val Epoch {epoch}", device=device_cuda,
                                     funcLoss=funcLoss, optimizer=None)
            for key in result:
                writer.add_scalar(tag=f"{log_section}/{key}_val",
                                  scalar_value=result[key],
                                  global_step=epoch
                                  )

            """Test - on the test set, whenever the best is reached on the validation set, the metrics are run on the 
            test set, and the best results of each run are saved in a json file, and training is stopped if, 
            on the test set, there is no improvement for 40 epochs in succession"""
            dice = result['dice']
            print(f"val dice:{dice}. ({name_model} on {name_dataset})")
            if dice > bestResult['dice']:
                # 这里是验证集上最好的
                bestResult['dice'] = dice
                bestResult['epoch'] = epoch
                ls_best_result.append("epoch={}, val_dice={:.3f}".format(epoch, dice))
                print("best dice found. evaluating on testset...")
                ls_best_result.append("test-result!!!:")
                result = traverseDataset(model=model, loader=testLoader,
                                         thresh_value=thresh_value,
                                         log_section=None,
                                         log_writer=None,
                                         description=f"Test Epoch {epoch}", device=device_cuda,
                                         funcLoss=funcLoss, optimizer=None)
                ls_best_result.append(result)

                path_json = os.path.join(root_result_model_dataset, "best_result.json")
                with open(path_json, "w") as f:
                    json.dump(ls_best_result, f, indent=2)
                path_model = os.path.join(root_result_model_dataset, 'model_best.pth')
                torch.save(model.state_dict(), path_model)
            else:
                # If there is no change in DICE for 40 epochs on the validation set, training is stopped
                threshold = 40
                if epoch - bestResult['epoch'] >= threshold:
                    print(f"Precision didn't improve in recent {threshold} epoches, stop training.")
                    break

        with open(path_flag, "w") as f:
            f.write("training and testing finished.")
