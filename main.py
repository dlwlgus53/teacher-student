import os
import argparse
from tagger import tag
import torch
from dataset import Dataset
from base_logger import logger
from trainer import train_loop, test
from utils import calculate_diff
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor
from base_logger import logger
from collections import OrderedDict
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--train_rate' ,  type = float, default=0.01)
parser.add_argument('--test_batch_size' , type = int, default=32)
parser.add_argument('--train_batch_size' , type = int, default=10)
parser.add_argument('--max_iter' ,  type = int, default=10)
parser.add_argument('--base_trained', type = str, default = "google/t5-small-ssm-nq", help =" pretrainned model from 🤗")
parser.add_argument('--diff_end' , type = int,  help = 'when loop end?', default = 0.01)
parser.add_argument('--test_device', type = int, default = 0)
parser.add_argument('--train_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/train_data.json')
parser.add_argument('--val_path' ,  type = str,  default = '../woz-data/MultiWOZ_2.1/dev_data.json')
parser.add_argument('--test_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/test_data.json')


args = parser.parse_args()

def model_load():
    model_path = f"./model/woz{args.train_rate}.pt"
    logger.info(f"load model from {model_path}")
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.','') # remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v
    teacher = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True)
    teacher.load_state_dict(new_state_dict)
    logger.info(f"load finished")
    
    return teacher

def data_loader_load(tokenizer):
    "NAVIES CALLING"
    train_dataset =Dataset(args.train_path, 'train', tokenizer)
    val_dataset =Dataset(args.val_path, 'val', tokenizer)
    test_dataset =Dataset(args.test_path, 'test',tokenizer)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.train_batch_size, pin_memory=True,
        num_workers=0, shuffle=False, collate_fn=train_dataset.collate_fn)
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.test_batch_size, pin_memory=True,
        num_workers=0, shuffle=False, collate_fn=val_dataset.collate_fn)
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.test_batch_size, pin_memory=True,
        num_workers=0, shuffle=False, collate_fn=test_dataset.collate_fn)
    
    return train_loader, val_loader, test_loader

def main():
    logger.info('load the model')
    model = model_load()
    tokenizer = T5Tokenizer.from_pretrained(args.base_trained)

    teacher_model, student_model, new_student = model.to(f'cuda:{args.test_device}'), copy.deepcopy(model), copy.deepcopy(model)
    train_loader, val_loader, test_loader = data_loader_load(tokenizer)
    optimizer = Adafactor(model.parameters(),lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False)
    logger.info('start loop')
    for iter in range(args.max_iter):
        logger.info(f'iter number : {iter}')
        # 전체 데이터에 대해서 태깅합니다.
        logger.info("1. Teacher Tagging")
        if iter ==0:
            teacher_tagged = tag(args, teacher_model, train_loader)
        else:
            teacher_tagged = student_tagged
        # 그걸로 학습합니다.
        logger.info("2. Student Training")
        train_loop(args, student_model, optimizer, train_loader, teacher_tagged)
        # 학습 후 태깅합니다.
        logger.info("Student Tagging")
        student_tagged = tag(args,student_model,train_loader)
        # 차이를 봅니다.
        logger.info("3. Student calculate diff")
        diff_rate = calculate_diff(args, teacher_tagged, student_tagged)
        
        logger.info('4. Difference Rate : %0.4f' % diff_rate)
        if diff_rate < args.diff_end:
            logger.info('Break the loop')
            break
        else:
            teacher_model = student_model
            student_model =  copy.deepcopy(new_student) # 이부분은 실험을 해 보아야 겠군
    
    torch.save(student_model.state_dict(), f"model/woz{args.train_rate}_loop.pt")
    score = test(args,student_model, test_loader, tokenizer)
    logger.info(score)

if __name__ == '__main__':
    main()