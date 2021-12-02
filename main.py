import utils
import copy
import torch
import argparse
import logging
from tagger import tag
from dataset import Dataset
from trainer import train, test
from log_conf import init_logger
from collections import OrderedDict

from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor



parser = argparse.ArgumentParser()
parser.add_argument('--model_rate' ,  type = float, default=0.01, help='not a datarate mode_trained data rate')
parser.add_argument('--test_batch_size' , type = int, default=32)
parser.add_argument('--train_batch_size' , type = int, default=10)
parser.add_argument('--max_iter' ,  type = int, default=10)
parser.add_argument('--base_trained', type = str, default = "google/t5-small-ssm-nq", help =" pretrainned model from 🤗")
parser.add_argument('--diff_end' , type = int,  help = 'when loop end?', default = 0.01)
parser.add_argument('--test_device', type = int, default = 0)
parser.add_argument('--train_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/train_data.json') # for all data
parser.add_argument('--test_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/test_data.json') # for all data
parser.add_argument('--save_prefix' , type = str,  default = '') # for all data

# parser.add_argument('--test_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/train_data0.01.json')

args = parser.parse_args()

init_logger(f'{args.save_prefix}{args.data_rate}.log')
logger = logging.getLogger("my")



def model_load():
    model_path = f"./model/woz{args.model_rate}.pt"
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

# def data_loader_load(tokenizer):
#     train_dataset =Dataset(args.train_path, 'train', tokenizer)
#     test_dataset =Dataset(args.test_path, 'test',tokenizer)
    
#     train_loader = torch.utils.data.DataLoader(
#         dataset=train_dataset, batch_size=args.train_batch_size, pin_memory=True,
#         num_workers=0, shuffle=False, collate_fn=train_dataset.collate_fn)
    
#     test_loader = torch.utils.data.DataLoader(
#         dataset=test_dataset, batch_size=args.test_batch_size, pin_memory=True,
#         num_workers=0, shuffle=False, collate_fn=test_dataset.collate_fn)
    
#     return train_loader, test_loader


def get_loader(dataset, batch_size):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    shuffle = False
    pin_memory = True
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, pin_memory=pin_memory,
        num_workers=0, shuffle=shuffle, sampler=train_sampler,  collate_fn=dataset.collate_fn)
    return loader       



def main():
    utils.makedir('temp')
    logger.info('load the model')
    batch_size = int(args.batch_size / args.gpus)
    tokenizer = T5Tokenizer.from_pretrained(args.base_trained)
    train_dataset =Dataset(args.train_path, 'train', tokenizer)
    test_dataset =Dataset(args.test_path, 'test', tokenizer)        
    train_loader = get_loader(train_dataset, batch_size)
    test_loader = get_loader(test_dataset, batch_size)

    model = model_load()


    teacher_model, student_model, new_student = model.to(f'cuda:{args.test_device}'), copy.deepcopy(model), copy.deepcopy(model)
    train_loader,  test_loader = data_loader_load(tokenizer)
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
        logger.info("2.1 Student Training")
        train(args, student_model, optimizer, train_loader, teacher_tagged)
        # 학습 후 태깅합니다.
        logger.info("2.2 Student Tagging")
        student_tagged = tag(args,student_model,train_loader)
        # 차이를 봅니다.
        logger.info("3. Student calculate diff")
        diff_rate = utils.calculate_diff(args, teacher_tagged, student_tagged)
        
        logger.info('4. Difference Rate : %0.4f' % diff_rate)
        if diff_rate < args.diff_end:
            logger.info('Break the loop')
            break
        else:
            teacher_model = student_model
            student_model =  copy.deepcopy(new_student)
    
    torch.save(student_model.state_dict(), f"model/woz{args.save_prefix}{args.train_rate}.pt")
    joint_goal_acc, slot_acc, loss = test(args,student_model, test_loader, tokenizer)
    logger.info(f'JGA : {joint_goal_acc} Slot Acc : {slot_acc} Loss : {loss}')
    
if __name__ == '__main__':
    main()