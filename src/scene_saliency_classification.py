import argparse
import os
import pickle
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import logging
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import tempfile
import shutil
import random
import numpy as np
from torch import nn, Tensor
import math
from torch.optim import AdamW
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
import warnings

warnings.filterwarnings('always')


def setup_simple_logger(args):
    if not os.path.exists(os.path.join(args.output_dir, "log")):
        os.makedirs(os.path.join(args.output_dir, "log"))

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO, handlers=[
            logging.FileHandler(os.path.join(args.output_dir, f"log/debug_{args.exp_name}.log")),
            logging.StreamHandler()])

    logger.setLevel(logging.INFO)

    return logger


def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


class SaliencyDataset(Dataset):

    def __init__(self, split_name, args):

        if split_name == "train":
            self.file_path = os.path.join(args.input_file_path, "train.pkl")
        elif split_name == "val":
            self.file_path = os.path.join(args.input_file_path, "val.pkl")
        elif split_name == "test":
            self.file_path = os.path.join(args.input_file_path, "test.pkl")

        with open(self.file_path, "rb") as f:
            self.data = pickle.load(f)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def collate_batch(batch):
    max_len = -1
    for data in batch:
        if max_len < data["scenes_embeddings"].shape[0]:
            max_len = data["scenes_embeddings"].shape[0]

    scenes = torch.zeros(len(batch), max_len, batch[0]["scenes_embeddings"].shape[-1])
    mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    labels = torch.ones(len(batch), max_len) * -1

    for idx, data in enumerate(batch):
        # label_list.append(data["labels"])
        _embedding = data["scenes_embeddings"]
        scenes[idx, :len(_embedding), :] = _embedding
        mask[idx, :len(_embedding)] = torch.zeros(len(_embedding), dtype=torch.bool)
        labels[idx, :len(_embedding)] = data["labels"]

    return scenes.to(device), mask.to(device), labels.to(device)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if self.batch_first:
            pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:

        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0)]

        return x


class SceneSaliency(nn.Module):
    def __init__(self, input_dimensions, nhead, num_layers, output_dim, position_dropout=0.1):
        super().__init__()

        self.pos_encoder = PositionalEncoding(input_dimensions, position_dropout, batch_first=True)
        self.encoder_layer = TransformerEncoderLayer(d_model=input_dimensions, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(input_dimensions, output_dim)

    def forward(self, scene_embeddings, embed_mask):
        scene_embeddings_pos = self.pos_encoder(scene_embeddings)
        enc_output = self.transformer_encoder(scene_embeddings_pos, mask=None, src_key_padding_mask=embed_mask)
        logits = self.linear(enc_output)
        return logits


def checkpoint(epoch, model, optimizer, scheduler, directory, filename='checkpoint.pt', max_checkpoints=5):
    '''
    Save a checkpoint
    Args:
        epoch - current epoch
        step - current step
        modules - a dict of name to object that supports the method state_dict
        directory - the directory to save the checkpoint file
        filename - the filename of the checkpoint
        max_checkpoints - how many checkpoints to keep
    '''
    if not os.path.isdir(directory):
        os.makedirs(directory)

    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    # 'scheduler': scheduler.state_dict()}

    with tempfile.NamedTemporaryFile() as temp_checkpoint_file:
        torch.save(state, temp_checkpoint_file)

        checkpoint_path = os.path.join(directory, filename)
        if os.path.exists(checkpoint_path):
            root, ext = os.path.splitext(filename)
            for i in range(max_checkpoints - 2, -1, -1):
                previous_path = os.path.join(directory, f'{root}{i}{ext}') if i else checkpoint_path
                if os.path.exists(previous_path):
                    backup_path = os.path.join(directory, f'{root}{i + 1}{ext}')
                    if os.path.exists(backup_path):
                        os.replace(previous_path, backup_path)
                    else:
                        os.rename(previous_path, backup_path)

        shutil.copy(temp_checkpoint_file.name, f'{checkpoint_path}.incomplete')
        os.rename(f'{checkpoint_path}.incomplete', checkpoint_path)

    return checkpoint_path


def save_checkpoint(epoch, model, optimizer, checkpoint_path, scheduler=None, best=False):
    checkpoint_path = checkpoint(epoch, model, optimizer, scheduler, checkpoint_path, max_checkpoints=3)

    if best:
        dirname = os.path.dirname(checkpoint_path)
        basename = os.path.basename(checkpoint_path)
        best_checkpoint_path = os.path.join(dirname, f'best_{basename}')
        shutil.copy2(checkpoint_path, best_checkpoint_path)



def add_model_specific_args(parser):
   
    parser.add_argument("--seed", type=int, default=1234, help="Seed")
    parser.add_argument("--lr", type=float, default=8e-5, help="Maximum learning rate")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--output_dir", type=str,default='./outputs/logs/classification/',
                        help="Location of output dir")
    parser.add_argument("--checkpoint_every", default=5, type=int, help='Checkpoint every epoch')
    parser.add_argument("--input_file_path", default="./outputs/scene_classification_data/",
                        type=str,
                        help='Dataset path')

   
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--exp_name", type=str, default="classification", help="Experiment Name")
    parser.add_argument("--checkpoint_path", type=str,
                        default="./outputs/scene_classification_checkpoints/",
                        help="Checkpoint Path")
    parser.add_argument("--max_checkpoints", type=int, default=3, help="Maximum number of checkpoints to be stored")
    parser.add_argument("--log_every_epoch", default=5, type=int, help='Logging verbosity')
    parser.add_argument("--input_dimensions", default=1024, type=int, help='Transformer Input Dimensions')
    parser.add_argument("--num_head", default=16, type=int, help='Transformer Number of head')
    parser.add_argument("--num_layers", default=10, type=int, help='Transformer number of layers')
    parser.add_argument("--output_dim", default=1, type=int, help='Transformer output dimension')
    parser.add_argument("--summ_data_output",
                        default="./outputs/training_data_using_prediction/",
                        type=str, help='Output path for pickles to input in summarization model')
    parser.add_argument("--threshold", type=float, default=0.5, help="Classifier Threshold")
    return parser


def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")

def logTest(metrics):
    logger.info(f" Final test metrics: {metrics}")


def compute_metrics(pred, target):
    pred = pred.long().detach().cpu()
    target = target.long().detach().cpu()

    Precision, Recall, f1, _ = precision_recall_fscore_support(target, pred, average='macro')

    acc = balanced_accuracy_score(target, pred)

    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': Precision,
        'Recall': Recall
    }


def averageBatch(metrics):
    acc = []
    f1=[]
    pre=[]
    recall=[]
    for idx in metrics:
        acc.append(idx['Accuracy'])
        f1.append(idx['F1'])
        pre.append(idx['Precision'])
        recall.append(idx['Recall'])

    mean_acc = np.mean(acc)
    mean_f1 = np.mean(f1)
    mean_pre = np.mean(pre)
    mean_recall = np.mean(recall)



    return {"Average Accuracy":mean_acc,"Average f1":mean_f1,"Average precision":mean_pre,"Average recall":mean_recall
            }

def train_loop(model, optimizer, criterion, dataloader,args):
    model.train()
    total_loss = 0
    batch_acc = []

    for scenes, mask, labels_padded in dataloader:
        scenes = scenes.to(device)
        mask = mask.to(device)
        labels_padded = labels_padded.to(device)
        output_padded = model(scenes, mask)

        loss_mask = ~mask
        loss_padded = criterion(output_padded.squeeze(-1), labels_padded)

        loss_unpadded = torch.masked_select(loss_padded, loss_mask)
        loss = loss_unpadded.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()

        output = torch.masked_select(output_padded.squeeze(-1), loss_mask)
        pred = torch.sigmoid(output) > args.threshold

        target = torch.masked_select(labels_padded, loss_mask)

        batch_acc.append(compute_metrics(pred, target))

    return total_loss / len(dataloader), batch_acc


def validation_loop(model, optimizer, criterion, dataloader,args):
    model.eval()
    total_loss = 0
    batch_acc = []
    with torch.no_grad():
        for scenes, mask, labels_padded in dataloader:
            scenes = scenes.to(device)
            mask = mask.to(device)
            labels_padded = labels_padded.to(device)

            output_padded = model(scenes, mask)

            loss_mask = ~mask
            loss_padded = criterion(output_padded.squeeze(-1), labels_padded)

            loss_unpadded = torch.masked_select(loss_padded, loss_mask)
            loss = loss_unpadded.mean()

            total_loss += loss.detach().item()

            pred = torch.masked_select(output_padded.squeeze(-1), loss_mask)
            pred = torch.sigmoid(pred) > args.threshold

            target = torch.masked_select(labels_padded, loss_mask)
            batch_acc.append(compute_metrics(pred, target))

    return total_loss / len(dataloader), batch_acc


def load_checkpoint(model, optimizer, checkpoint_path, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def train_model(model, optimizer, criterion, args, trainLoader, valLoader):
    progress_bar = tqdm(range(args.epochs))
    best_val_loss = 9999
    best_val_acc = -99
    print(device)
    for epoch in range(args.epochs):

        train_loss, train_metrics = train_loop(model, optimizer, criterion, trainLoader,args)
        log_metrics(epoch, {'train_loss': train_loss, 'epochs': epoch})

        val_loss, val_metrics_batch = validation_loop(model, optimizer, criterion, valLoader,args)
        val_average_metric = averageBatch(val_metrics_batch)
        log_metrics(epoch, {'val_loss': val_loss, 'epochs': epoch})
        log_metrics(epoch, val_average_metric)

        if val_average_metric['Average f1']>best_val_acc:
        #if val_loss < best_val_loss:
            logger.info(f'Metric improved, saving checkpoint')
            #best_val_loss = val_loss
            best_val_acc = val_average_metric['Average f1']
            save_checkpoint(epoch, model, optimizer, args.checkpoint_path, scheduler=None, best=True)

        if epoch % args.checkpoint_every == 0:
            logger.info(f'Saving checkpoint at epoch:{epoch}')
            save_checkpoint(epoch, model, optimizer, args.checkpoint_path, scheduler=None, best=False)

        progress_bar.update(1)
        # model.train()
        # total_loss = 0.0
        # batch_acc = []

    logger.info(f'End of training')


def getPositiveWeight(trainDataset):
    labels = []
    for d in range(len(trainDataset)):
        labels += trainDataset[d]["labels"]

    ones = sum(labels)
    zeros = len(labels) - ones
    positive_weight = torch.FloatTensor([zeros / ones]).to(device)
    return positive_weight


def logTest(metrics):
    logger.info(f" Final test metrics: {metrics}")


def loadModel(args):
    model = SceneSaliency(input_dimensions=args.input_dimensions, nhead=args.num_head, num_layers=args.num_layers,
                          output_dim=args.output_dim)
    return model


def generateData(model, dataloader):
    model.eval()
    dataset = []

    with torch.no_grad():
        for scenes, mask, labels_padded in dataloader:
            movieDict = {}

            scenes = scenes.to(device)
            mask = mask.to(device)
            labels_padded = labels_padded.to(device)

            output_padded = model(scenes, mask)
            loss_mask = ~mask
            pred = torch.masked_select(output_padded.squeeze(-1), loss_mask)
            pred = pred > 0.5

            movieDict["prediction_labels"] = pred.int()
            dataset.append(movieDict)
    return dataset


def savePickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def loadPickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def prepareDataForSummarization(pred_data, script_data):
    dataForSummarization = []
    for i in range(len(script_data)):
        movieDict = {}
        scriptTextDict = script_data[i]["scenes"]
        pred_labels = pred_data[i]["prediction_labels"].detach().cpu().tolist()

        movieDict["script"] = " ".join(scriptTextDict[idx] for idx, label in enumerate(pred_labels) if label == 1)

        movieDict["summary"] = script_data[i]["summary"]
        dataForSummarization.append(movieDict)
    return dataForSummarization


if __name__ == '__main__':

    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = add_model_specific_args(main_arg_parser)
    args = parser.parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(os.path.join(args.summ_data_output)):
        os.makedirs(os.path.join(args.summ_data_output))

    logger = setup_simple_logger(args)

    trainDataset = SaliencyDataset("train", args)
    valDataset = SaliencyDataset("val", args)
    testDataset = SaliencyDataset("test", args)

    trainLoader = DataLoader(trainDataset, batch_size=8, shuffle=True, collate_fn=collate_batch)
    valLoader = DataLoader(valDataset, batch_size=8, shuffle=False, collate_fn=collate_batch)
    testLoader = DataLoader(testDataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

    model = loadModel(args)
    optimizer = AdamW(model.parameters(),  lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=getPositiveWeight(trainDataset))
    model.to(device)

    train_model(model, optimizer,criterion,args,trainLoader,valLoader)

    logger.info(f'Loading the last best model and testing')
    model, optimizer, epoch = load_checkpoint(model, optimizer, args.checkpoint_path + "/best_checkpoint.pt")

    test_loss, test_metrics = validation_loop(model, optimizer, criterion, testLoader,args)
    final_test_metrics = averageBatch(test_metrics)
    print("Testing", final_test_metrics)
    logTest(final_test_metrics)

    genTrainLoader = DataLoader(trainDataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    genValLoader = DataLoader(valDataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    genTestLoader = DataLoader(testDataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

    logger.info(f'Generating data for summarization')
    trainPred = generateData(model, genTrainLoader)
    valPred = generateData(model, genValLoader)
    testPred = generateData(model, genTestLoader)

    


    trainSummData = prepareDataForSummarization(trainPred,trainDataset)
    valSummData = prepareDataForSummarization(valPred,valDataset)
    testSummData = prepareDataForSummarization(testPred,testDataset)


    savePickle(os.path.join(args.summ_data_output, "train.pkl"), trainSummData)
    savePickle(os.path.join(args.summ_data_output, "val.pkl"), valSummData)
    savePickle(os.path.join(args.summ_data_output, "test.pkl"), testSummData)

    logger.info(f'Experiment completed')