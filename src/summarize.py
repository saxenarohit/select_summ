import argparse
import os
import pickle
import datasets
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM, set_seed, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import logging
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import tempfile
import shutil
import random
import numpy as np
from torch.cuda.amp import GradScaler
from bert_score import score as bert_score_func

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

class SummarizationDataset(Dataset):
    def __init__(self, tokenizer,split_name, args):

        if split_name=="train":
            self.file_path = os.path.join(args.file_path,"train.pkl")
        elif split_name=="val":
            self.file_path = os.path.join(args.file_path,"val.pkl")
        elif split_name=="test":
            self.file_path = os.path.join(args.file_path,"test.pkl")


        self.model_name = args.model_name

        self.tokenizer = tokenizer
        with open(self.file_path,"rb") as f:
            self.data = pickle.load(f)

        
        self.max_input_len = args.max_input_len
        self.max_output_len = args.max_output_len

    def __len__(self):
        """Returns length of the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Gets an example from the dataset. The input and output are tokenized and limited to a certain seqlen."""
        entry = self.data[idx]
        input_ids = self.tokenizer.encode(entry['script'], truncation=True, max_length=self.max_input_len,
                                          padding='max_length')  # padding to max seqlen for const memory/example
        output_ids = self.tokenizer.encode(entry['summary'], truncation=True, max_length=self.max_output_len,
                                           padding='max_length')  # padding to max seqlen for const memory/example

        if self.model_name =="allenai/led-large-16384":
            output_ids = output_ids[1:]
        return torch.tensor(input_ids), torch.tensor(output_ids)

    @staticmethod
    def collate_fn(batch):
        """
        Groups multiple examples into one batch with padding and tensorization.
        The collate function is called by PyTorch DataLoader
        """
        pad_token_id = 1
        input_ids, output_ids = list(zip(*batch))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=pad_token_id)
        return input_ids, output_ids

def set_global_attention_mask(input_ids,tokenizer,args):
    """Configure the global attention pattern based on the task"""

    global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)

    # The following line puts global attention on the <s> token to make sure all model
    # parameters which is necessery for gradient checkpointing to work.
    global_attention_mask[:, 0] = 1

    # # Global attention on the first 100 tokens
    # global_attention_mask[:, :100] = 1

    # # Global attention on scene starting
    if args.use_global_attn:
        global_attention_mask[(input_ids == tokenizer.convert_tokens_to_ids('INT'))] = 1
        global_attention_mask[(input_ids == tokenizer.convert_tokens_to_ids('EXT'))] = 1
        global_attention_mask[(input_ids == tokenizer.convert_tokens_to_ids('ĠINT'))] = 1
        global_attention_mask[(input_ids == tokenizer.convert_tokens_to_ids('ĠEXT'))] = 1

    return global_attention_mask


def setup_simple_logger(args):
    if not os.path.exists(os.path.join(args.log_dir, "log")):
        os.makedirs(os.path.join(args.log_dir, "log"))

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO, handlers=[
            logging.FileHandler(os.path.join(args.log_dir, f"log/debug_{args.exp_name}.log")),
            logging.StreamHandler()])

    logger.setLevel(logging.INFO)

    return logger

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight","layer_norm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [{'params': params_with_wd, 'weight_decay': args.weight_decay},
            {'params': params_without_wd, 'weight_decay': 0.0}]


def add_model_specific_args(parser):
    parser.add_argument("--seed", type=int, default=1234, help="Seed")
    parser.add_argument("--lr", type=float, default=5e-5, help="Maximum learning rate")
    parser.add_argument("--warmup", type=int, default=1024, help="Number of warmup steps")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--max_output_len", type=int, default=1024, help="maximum num of wordpieces in the summary")
    parser.add_argument("--output_dir", type=str, default='./outputs/summarization_checkpoints/',
                        help="Location of output dir")
    parser.add_argument("--log_dir", type=str, default='./outputs/logs/summarization/',
                        help="Location of output dir")
    parser.add_argument("--val_every", default=200, type=int, help='Validation every')
    parser.add_argument("--file_path", default="./outputs/training_data_using_prediction/", type=str,
                        help='Dataset path')

    parser.add_argument("--max_input_len", type=int, default=16384, help="maximum num of wordpieces in the input")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="number of gradient accumulation steps")
    parser.add_argument("--fp16", action='store_true', help="Use fp16 ")
    parser.add_argument("--grad_ckpt", action='store_true', help='Enable gradient checkpointing to save memory')
    parser.add_argument("--attention_window", type=int, default=1024, help="Attention window")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment Name")
    parser.add_argument("--checkpoint_path", type=str, default="/rds/user/co-saxe1/hpc-work/output/logging_test/", help="Experiment Name")
    parser.add_argument("--max_checkpoints", type=int, default=3, help="Maximum number of checkpoints to be stored")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--log_every_step", default=10, type=int,  help='Logging verbosity')
    parser.add_argument("--model_name", default="allenai/led-large-16384", type=str, help='Hugging face model to be used')
    parser.add_argument("--use_global_attn", action='store_true', help='Enable global attention')
    parser.add_argument("--resume_training", action='store_true', help='Resume training from best checkpoint')
    return parser

def get_dataloader( split_name, is_train):
    """Get training and validation dataloaders"""

    #sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
    if split_name =="train":
        loader =  DataLoader(trainDataset, batch_size=args.batch_size, shuffle=is_train,
                          num_workers=args.num_workers,
                          collate_fn=SummarizationDataset.collate_fn)
    elif split_name =="validation":
        loader = DataLoader(valDataset, batch_size=args.batch_size, shuffle=is_train,
                            num_workers=args.num_workers,
                            collate_fn=SummarizationDataset.collate_fn)
    elif split_name =="test":
        loader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=is_train,
                            num_workers=args.num_workers,
                            collate_fn=SummarizationDataset.collate_fn)
    return loader


def loadModel(args):
    config = AutoConfig.from_pretrained(args.model_name,
                                        cache_dir="/rds/user/co-saxe1/hpc-work/huggingface_cache")

    if args.model_name=="allenai/led-large-16384":
        print("Setting bos token id to 0")
        config.forced_bos_token_id = 0

    if args.grad_ckpt:
        # model.gradient_checkpointing_enable()
        config.gradient_checkpointing = True
        # use cache cannot be used with gradient checkpointing
        config.use_cache = False
    config.attention_window = [args.attention_window] * len(config.attention_window)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, config=config,
                                                  cache_dir="/rds/user/co-saxe1/hpc-work/huggingface_cache")

    # Load tokenizer and metric
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True,
                                              cache_dir="/rds/user/co-saxe1/hpc-work/huggingface_cache")

    if args.model_name == "allenai/led-large-16384":
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer,model,config

def get_lr():
    return optimizer.param_groups[0]['lr']

def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")


def checkpoint(epoch, step, model,config, fp16,scaler,directory,scheduler,completed_steps, filename='checkpoint.pt', max_checkpoints=5):
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
             'steps': step,
             'config': config,
             'completed_steps': completed_steps,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict()}

    if fp16:
        state['scaler'] = scaler.state_dict()

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


def save_checkpoint(epoch, step, model, args, config,scaler,scheduler,completed_steps, best=False):
    checkpoint_path = checkpoint(epoch, step, model,config, args.fp16,scaler, args.checkpoint_path,scheduler,completed_steps, max_checkpoints=args.max_checkpoints)

    if best:
        dirname = os.path.dirname(checkpoint_path)
        basename = os.path.basename(checkpoint_path)

        if args.resume_training:
            best_checkpoint_path = os.path.join(dirname, f'best_resume_{basename}')
        else:
            best_checkpoint_path = os.path.join(dirname, f'best_{basename}')
        shutil.copy2(checkpoint_path, best_checkpoint_path)


def load_checkpoint(model, optimizer,fp16,scaler, checkpoint_path,scheduler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    config = checkpoint['config']
    epoch = checkpoint['epoch']
    steps = checkpoint['steps']
    completed_steps =  checkpoint['completed_steps']
    scheduler.load_state_dict(checkpoint['scheduler'])

    if fp16:
        scaler.load_state_dict(checkpoint['scaler'])
    return model, optimizer,scaler,config,epoch,steps,completed_steps,scheduler


def logTest(metrics):
    logger.info(f" Final test metrics: {metrics}")


def evaluate_step(model,args,tokenizer, test=False):
    model.eval()

    if test:
        data_loader = test_dataloader
        all_predictions = []
        all_references = []
    else:
        data_loader = validation_dataloader

    metricsDict = {}
    count=0
    for batch in data_loader:
        input_ids = batch[0].to(device)
        output_ids = batch[1].to(device)
        with torch.no_grad():
            generated_ids = model.generate(input_ids=input_ids,
                                           attention_mask=(input_ids != tokenizer.pad_token_id),
                                           global_attention_mask=set_global_attention_mask(input_ids,tokenizer,args),
                                           use_cache=True, max_length=args.max_output_len, num_beams=args.beam_size)

        # Convert predicted and gold token ids to strings
        predictions = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        references = tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
        if test:
            all_predictions += predictions
            all_references += references

        # Compute rouge
        metric_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        if test:
            results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
            bertscore.add_batch(predictions=predictions, references=references)

        else:
            results = rouge.compute(predictions=predictions, references=references)

        for metric_name in metric_names:

            metric_val = input_ids.new_zeros(1) + results[metric_name].mid.fmeasure
            if metric_name in metricsDict:
                metricsDict[metric_name].append(metric_val)
            else:
                metricsDict[metric_name] = [metric_val]

        count+=1


    metricsDictReturn = {}
    for metric_name in metric_names:
        if test:
            metricsDictReturn["test_" + metric_name] = torch.mean(torch.cat(metricsDict[metric_name])).item()
        else:
            metricsDictReturn["val_" + metric_name] = torch.mean(torch.cat(metricsDict[metric_name])).item()

    if test:
        bert_result = bertscore.compute(lang="en",batch_size=args.batch_size)
        metricsDictReturn["test_bert_p"] = np.mean(bert_result["precision"])
        metricsDictReturn["test_bert_r"] = np.mean(bert_result["recall"])
        metricsDictReturn["test_bert_f"] = np.mean(bert_result["f1"])

    if test:
        with open(os.path.join(args.test_summaries,"pred.pkl"), "wb") as f:
            pickle.dump(all_predictions, f)

        with open(os.path.join(args.test_summaries,"ref.pkl"), "wb") as f:
            pickle.dump(all_references, f)

    return metricsDictReturn


def train_model(num_training_steps,model,optimizer,config,tokenizer,scaler):
    progress_bar = tqdm(range(num_training_steps))
    completed_steps = 0
    step = 0
    best_val_r1 = -1
    print(device)
    for epoch in range(args.epochs):

        model.train()
        train_loss = 0.0
        for batch in train_dataloader:

            input_ids = batch[0].to(device)
            output_ids = batch[1].to(device)
            outputs = model(input_ids, attention_mask=(input_ids != tokenizer.pad_token_id),  # mask padding tokens
                            global_attention_mask=set_global_attention_mask(input_ids,tokenizer,args),  # set global attention
                            labels=output_ids, use_cache=False)

            loss = outputs.loss
            loss = loss / args.grad_accum

            train_loss += loss.item()
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.log_every_step ==0:
                log_metrics(step, {'lr': get_lr(), 'steps': step,'epochs': epoch, "optimize_steps": completed_steps,
                                   'loss/train': loss.item(), "running_train_loss": train_loss})

            if step % args.grad_accum == 0:

                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

                completed_steps += 1

                train_loss = 0.0

            if step % args.val_every == 0:

                logger.info(f'Evaluating and saving model at epoch:{epoch} step: {step}')
                log_metrics(step, {'lr': get_lr(), 'steps': step,'epochs': epoch, "optimize_steps": completed_steps,
                                   'loss/train': loss.item(), "running_train_loss": train_loss})
                val_metrics = evaluate_step(model,args,tokenizer)
                val_metrics["steps"] = step
                log_metrics(step, val_metrics)
                if val_metrics["val_rouge1"] > best_val_r1:

                    logger.info(f'Metric improved')
                    save_checkpoint(epoch, step, model, args,config,scaler,lr_scheduler,completed_steps, best=True)
                    best_val_r1 = val_metrics["val_rouge1"]
                else:
                    save_checkpoint(epoch, step, model, args,config,scaler,lr_scheduler,completed_steps, best=False)

                model.train()

            step += 1
            progress_bar.update(1)
        logger.info(f'Saving model checkpoint at end of epoch:{epoch} step: {step - 1}')
        save_checkpoint(epoch, step, model, args,config,scaler,lr_scheduler,completed_steps, best=False)

    logger.info(f'End of training')


def resume_train_model(steps_completed,epochs_completed,optimized_steps,num_training_steps,model,optimizer,config,tokenizer,scaler):
    progress_bar = tqdm(range(steps_completed, num_training_steps))
    completed_steps = optimized_steps
    step = steps_completed
    best_val_r1 = -1
    print(device)
    for epoch in range(epochs_completed,args.epochs):

        model.train()
        train_loss = 0.0
        for batch in train_dataloader:

            input_ids = batch[0].to(device)
            output_ids = batch[1].to(device)
            outputs = model(input_ids, attention_mask=(input_ids != tokenizer.pad_token_id),  # mask padding tokens
                            global_attention_mask=set_global_attention_mask(input_ids,tokenizer,args),  # set global attention
                            labels=output_ids, use_cache=False)

            loss = outputs.loss
            loss = loss / args.grad_accum

            train_loss += loss.item()
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.log_every_step ==0:
                log_metrics(step, {'lr': get_lr(), 'steps': step,'epochs': epoch, "optimize_steps": completed_steps,
                                   'loss/train': loss.item(), "running_train_loss": train_loss})

            if step % args.grad_accum == 0:

                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

                completed_steps += 1

                train_loss = 0.0

            if step % args.val_every == 0:

                logger.info(f'Evaluating and saving model at epoch:{epoch} step: {step}')
                log_metrics(step, {'lr': get_lr(), 'steps': step,'epochs': epoch, "optimize_steps": completed_steps,
                                   'loss/train': loss.item(), "running_train_loss": train_loss})
                val_metrics = evaluate_step(model,args,tokenizer)
                val_metrics["steps"] = step
                log_metrics(step, val_metrics)
                if val_metrics["val_rouge1"] > best_val_r1:

                    logger.info(f'Metric improved')
                    save_checkpoint(epoch, step, model, args,config,scaler,lr_scheduler,completed_steps, best=True)
                    best_val_r1 = val_metrics["val_rouge1"]
                else:
                    save_checkpoint(epoch, step, model, args,config,scaler,lr_scheduler,completed_steps, best=False)

                model.train()

            step += 1
            progress_bar.update(1)
        logger.info(f'Saving model checkpoint at end of epoch:{epoch} step: {step - 1}')
        save_checkpoint(epoch, step, model, args, config, scaler, lr_scheduler, completed_steps)

    logger.info(f'End of training')

if __name__ == '__main__':

    main_arg_parser = argparse.ArgumentParser(description="summarization")
    parser = add_model_specific_args(main_arg_parser)
    args = parser.parse_args()
    args.test_summaries = os.path.join(args.output_dir, "test_summaries")


    set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.test_summaries):
        os.makedirs(args.test_summaries)

    logger = setup_simple_logger(args)

    tokenizer,model,config = loadModel(args)

    rouge = datasets.load_metric('rouge')
    bertscore = datasets.load_metric("bertscore")
    logger.info(f'Using model: {args.model_name}')

    trainDataset = SummarizationDataset(tokenizer=tokenizer, split_name="train", args=args)
    valDataset = SummarizationDataset(tokenizer=tokenizer, split_name="val", args=args)
    testDataset = SummarizationDataset(tokenizer=tokenizer, split_name="test", args=args)

    train_dataloader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers,
                                  collate_fn=SummarizationDataset.collate_fn)
    validation_dataloader = DataLoader(valDataset, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers,
                                       collate_fn=SummarizationDataset.collate_fn)
    test_dataloader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=SummarizationDataset.collate_fn)

    num_training_steps = args.epochs * len(train_dataloader)


    scaler = GradScaler(enabled=args.fp16)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup,
                                                   num_training_steps=num_training_steps)

    model.to(device)
    if args.resume_training:
        model, optimizer, scaler, config, epoch, steps,optimized_steps,lr_scheduler = load_checkpoint(model, optimizer, args.fp16, scaler,
                                                                         args.checkpoint_path + "/best_checkpoint.pt",lr_scheduler)


        logger.info(f'Resuming training from the last best model')
        resume_train_model(steps,epoch,optimized_steps,num_training_steps, model, optimizer, config,tokenizer,scaler)
    else:
        train_model(num_training_steps, model, optimizer, config,tokenizer,scaler)

    logger.info(f'Loading the last best model and testing')
    model, optimizer,scaler,config,epoch,steps,optimized_steps,lr_scheduler = load_checkpoint(model, optimizer,args.fp16, scaler, args.checkpoint_path + "/best_checkpoint.pt",lr_scheduler)
    test_metric = evaluate_step(model,args,tokenizer, test=True)
    print("Testing", test_metric)
    logTest(test_metric)
    logger.info(f'Training completed')