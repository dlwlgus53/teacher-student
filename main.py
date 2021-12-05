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


#for  DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor


parser = argparse.ArgumentParser()
parser.add_argument('--model_rate' ,  type = float, default=0.01, help='not a datarate mode_trained data rate')
parser.add_argument('--tag_batch_size' , type = int, default=32)
parser.add_argument('--train_batch_size' , type = int, default=8)
parser.add_argument('--max_iter' ,  type = int, default=10)
parser.add_argument('--base_trained', type = str, default = "google/t5-small-ssm-nq", help =" pretrainned model from 🤗")
parser.add_argument('--pretrained_model', type = str, default = "google/t5-small-ssm-nq", help =" pretrainned model from 🤗")

parser.add_argument('--diff_end' , type = int,  help = 'when loop end?', default = 0.01)
parser.add_argument('--train_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/train_data.json') # for all data
parser.add_argument('--test_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/test_data.json') # for all data
parser.add_argument('--save_prefix' , type = str,  default = '') # for all data
parser.add_argument('--model_path' , type = str,  default = './model/woz0.1.pt') # for all data
parser.add_argument('-n', '--nodes', default=1,type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=2, type=int,help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,help='ranking within the nodes')
parser.add_argument('--port' ,  type = int, default=17365, help='port number')
parser.add_argument('--temp_folder' ,  type = str, default='temp', help='temp folder')



# parser.add_argument('--test_path' , type = str,  default = '../woz-data/MultiWOZ_2.1/train_data0.01.json')

args = parser.parse_args()

init_logger(f'{args.save_prefix}{args.model_rate}.log')
logger = logging.getLogger("my")



def model_load(model, pretrained_path):
    model_path = f"./model/woz{args.model_rate}.pt"
    logger.info(f"load model from {model_path}")
    state_dict = torch.load(pretrained_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.','') # remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v
        
        
    model.load_state_dict(new_state_dict)
    logger.info(f"load finished")
    return model

def get_loader(dataset, batch_size):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    shuffle = False
    pin_memory = True
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, pin_memory=pin_memory,
        num_workers=0, shuffle=shuffle, sampler=train_sampler,  collate_fn=dataset.collate_fn)
    return loader       


def main(gpu, args):
    logger.info(f'{gpu} starts')
    min_diff = float('inf')
    torch.distributed.init_process_group(
        backend='nccl',
        init_method=f'tcp://127.0.0.1:{args.port}',
        world_size=args.gpus,
        rank=gpu)

    tokenizer = T5Tokenizer.from_pretrained(args.base_trained)

    train_batch_size = int(args.train_batch_size / args.gpus)
    tag_batch_size = int(args.tag_batch_size / args.gpus)
    
    train_dataset =Dataset(args.train_path, 'train', tokenizer)
    
    train_loader = get_loader(train_dataset, train_batch_size)
    tag_loader =  get_loader(train_dataset, tag_batch_size)
    
    
    logger.info('load the model')

    teacher_model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True).to(gpu)
    student_model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True).to(gpu)
    newbie_model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True).to(gpu)
    
    teacher_model = model_load(teacher_model, args.pretrained_model)
    student_model = model_load(student_model, args.pretrained_model)
    newbie_model = model_load(newbie_model, args.pretrained_model)
    
    
    teacher_model = DDP(teacher_model, device_ids=[gpu])
    student_model = DDP(student_model, device_ids=[gpu])
    newbie_model = DDP(newbie_model, device_ids=[gpu])
    
    optimizer = Adafactor(student_model.parameters(),lr=1e-3,
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
        if gpu ==0 :logger.info(f'iter number : {iter}')
        if gpu ==0 :logger.info("1. Teacher Tagging")

        if iter ==0:
            # tag(args,gpu, teacher_model, tag_loader, prefix = 't')
            dist.barrier()
            teacher_tagged = utils.load_tag(args, prefix = 't')
        else:
            teacher_tagged = utils.load_tag(args, prefix = 't') # 여기서 합쳐서 원본으로 만들어야겠네
        # 그걸로 학습합니다.
        
        
        if gpu ==0 : logger.info("2.1 Student Training")
        train(args, gpu, student_model, optimizer, train_loader, teacher_tagged)
        # 학습 후 태깅합니다.
        if gpu ==0 :logger.info("2.2 Student Tagging")
        tag(args, gpu, student_model,tag_loader, prefix = 's')
        dist.barrier()
        student_tagged = utils.load_tag(args, prefix = 's')
        # 차이를 봅니다.
        if gpu ==0 :
            logger.info("3. Student calculate diff")
            diff_rate = utils.calculate_diff(args, teacher_tagged, student_tagged, tokenizer)
            logger.info('>> Difference Rate : %0.4f' % diff_rate)
            
            if min_diff>diff_rate:
                min_diff = diff_rate
                torch.save(student_model.state_dict(), f"model/woz{args.save_prefix}{args.model_rate}.pt")
                logger.info('Save new student')
                
        teacher_model = student_model
        student_model =  copy.deepcopy(newbie_model)
        optimizer = Adafactor(student_model.parameters(),lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False)


def evaluate():
    tokenizer = T5Tokenizer.from_pretrained(args.base_trained)
    test_dataset =Dataset(args.test_path, 'test',tokenizer)
    
    loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.test_batch_size, pin_memory=True,
        num_workers=0, shuffle=False, collate_fn=test_dataset.collate_fn)
    
    if args.pretrained_model:
        logger.info(f"User pretrained model{args.pretrained_model}")
        state_dict = torch.load(args.pretrained_model)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
            new_state_dict[name] = v
        model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True)
        model.to(f'cuda')
        model.load_state_dict(new_state_dict)
    
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True)
        model.to(f'cuda')
        
        
    joint_goal_acc, slot_acc, schema_acc, loss = test(args, model, loader, tokenizer)
    logger.info(f'JGA : {joint_goal_acc} Slot Acc : {slot_acc} Loss : {loss}')
    logger.info(f'schema_acc : {schema_acc}')
    
    schema_acc['JGA'] = joint_goal_acc
    schema_acc['schema_acc'] = slot_acc
    schema_acc['loss'] = loss
    
    utils.dict_to_csv(schema_acc, f'{args.save_prefix}{args.model_rate}.csv')
    
    joint_goal_acc, slot_acc, loss = test(args,student_model, test_loader, tokenizer)
    logger.info(f'JGA : {joint_goal_acc} Slot Acc : {slot_acc} Loss : {loss}')
    
if __name__ == '__main__':
    utils.makedirs(args.temp_folder)
    world_size = args.gpus * args.nodes 
    mp.spawn(main,
        nprocs=world_size,
        args=(args,),
        join=True)
    evaluate()