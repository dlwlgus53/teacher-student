import torch, logging
import pickle
import ontology

from collections import defaultdict
logger = logging.getLogger("my")

def tag(args, gpu, model, train_loader, prefix):
    '''
    train_loader에 있는 데이터를 load 가져와서 tagging 합니다.
    '''
    belief_state = defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id # schema
    model.eval()
    with torch.no_grad():
        for iter,batch in enumerate(train_loader):
            input_ids = batch['input']['input_ids'].to(f'cuda:{gpu}')
            outputs_text = model.module.generate(input_ids=input_ids).to('cpu')
            
            for idx in range(len(outputs_text)):
                dial_id = batch['dial_id'][idx]
                turn_id = batch['turn_id'][idx]
                schema = batch['schema'][idx]
                belief_state[dial_id][turn_id][schema] = outputs_text[idx]

            if gpu == 0 and (iter + 1) % 50 == 0:
                logger.info('step : {}/{}'.format(
                iter+1, 
                str(len(train_loader)),
                ))
                
    for k in belief_state.keys():
        belief_state[k] = dict(belief_state[k])
    
    pickle_path = f'./temp/{prefix}{args.save_prefix}{gpu}.pickle'
    with open(pickle_path, 'wb') as f:
        pickle.dump(dict(belief_state), f, pickle.HIGHEST_PROTOCOL)
        
    # 파일에 쓰도록 하자!
    return  belief_state