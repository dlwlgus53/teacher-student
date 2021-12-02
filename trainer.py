import torch
import logging
import pdb 
import json
import ontology
from utils import*
from collections import defaultdict

from utils import evaluate_metrics

logger = logging.getLogger("my")

def train(args, model, optimizer, train_loader, teacher_tagged):
    model.train()
    gpu = 0
    if gpu==0: logger.info("Train start")
    for iter, batch in enumerate(train_loader):
        optimizer.zero_grad()
        dial_ids = batch['dial_id']
        turn_ids = batch['turn_id']
        schemas = batch['schema']
        input_ids = batch['input']['input_ids'].to('cuda:0')
        labels = [ teacher_tagged[d][t][s] for (d,t,s) in zip(dial_ids, turn_ids, schemas)]
        labels = torch.stack(labels).to('cuda:0')
        outputs = model(input_ids=input_ids, labels=labels)
        loss =outputs.loss
        loss.backward()
        optimizer.step()
    
        if (iter + 1) % 50 == 0 and gpu==0:
            logger.info('Student training gpu {} step : {}/{} Loss: {:.4f}'.format(
                gpu,
                iter, 
                str(len(train_loader)),
                loss.detach())
            )

def test(args, model, test_loader, tokenizer):
    belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id # schema
    model.eval()
    loss_sum = 0
    logger.info("Test start")
    with torch.no_grad():
        for iter,batch in enumerate(test_loader):
            outputs = model(input_ids=batch['input']['input_ids'].to('cuda:0'), labels=batch['target']['input_ids'].to('cuda:0'))
            outputs_text = model.generate(input_ids=batch['input']['input_ids'].to('cuda:0'))
            outputs_text = [tokenizer.decode(o).replace('</s>','').replace('<pad>','').strip() for o in outputs_text]
            
            
            for idx in range(len(outputs_text)):
                if outputs_text[idx] == ontology.QA['NOT_MENTIONED'] : continue
                dial_id = batch['dial_id'][idx]
                turn_id = batch['turn_id'][idx]
                schema = batch['schema'][idx]
                belief_state[dial_id][turn_id][schema] = outputs_text[idx]

            if (iter + 1) % 50 == 0:
                logger.info('step : {}/{}'.format(
                iter+1, 
                str(len(test_loader)),
                ))
                loss_sum += outputs.loss.to('cpu')
                
    test_file = json.load(open(args.test_path , "r"))

    joint_goal_acc, slot_acc = evaluate_metrics(belief_state, test_file , ontology.QA['all-domain'])
    

    return  joint_goal_acc, slot_acc, loss_sum/iter

