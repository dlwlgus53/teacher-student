import torch, logging
import ontology

from collections import defaultdict
logger = logging.getLogger("my")

def tag(args, model, train_loader):
    '''
    train_loader에 있는 데이터를 load 가져와서 tagging 합니다.
    '''
    
    number_belief= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id # schema
    model.eval()
    with torch.no_grad():
        for iter,batch in enumerate(train_loader):
            input_ids = batch['input']['input_ids'].to(f'cuda:{args.test_device}')
            outputs_text = model.generate(input_ids=input_ids).to('cpu')
            
            for idx in range(len(outputs_text)):
                dial_id = batch['dial_id'][idx]
                turn_id = batch['turn_id'][idx]
                schema = batch['schema'][idx]
                number_belief[dial_id][turn_id][schema] = outputs_text[idx]

            if (iter + 1) % 50 == 0:
                logger.info('step : {}/{}'.format(
                iter+1, 
                str(len(train_loader)),
                ))
    
    # 파일에 쓰도록 하자!
    return  number_belief