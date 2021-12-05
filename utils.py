import os, csv
import pdb, logging
import pickle
from collections import defaultdict

logger = logging.getLogger("my")

    
def load_tag(args, prefix):
    tags = []
    
    path_dir = f'./{args.temp_folder}'
    file_list = os.listdir(path_dir)

    for pickle_path in file_list:
        if pickle_path[0] == prefix:
            
            with open(f'./{path_dir}/{pickle_path}', 'rb') as f:
                item = pickle.load(f)
                tags.append(item)
            
    return tags

def _get_label(dial_ids, turn_ids, schemas, tagged):
    label = ''
    for tag in tagged:
        try:
            label = tag[dial_ids][turn_ids][schemas]
        except KeyError as e:
            continue
    return label
   
def calculate_diff(args, teacher_tagged, student_tagged, tokenizer):
    diff_count = 0.0
    diff_all =0.0
    for tag in teacher_tagged:
        for dial_id in tag:
            for idx, turn_id in enumerate(tag[dial_id]):
                texts = []
                for schema in tag[dial_id][turn_id]:
                    teacher_tag = tag[dial_id][turn_id][schema]
                    student_tag = _get_label(dial_id, turn_id, schema, student_tagged)
                    diff_all += 1
                    teacher_text = tokenizer.decode(teacher_tag).replace('</s>','').replace('<pad>','').strip()
                    student_text = tokenizer.decode(student_tag).replace('</s>','').replace('<pad>','').strip()
              
                    if teacher_text == student_text: diff_count +=1
                    
                    if idx == len(tag[dial_id])-1:
                        texts.append(teacher_text +' : ' + student_text)
                        
            if idx == len(tag[dial_id])-1:
                logger.info(dial_id)
                logger.info(texts)
                    
    return (diff_count/diff_all)

def evaluate_metrics(all_prediction, raw_file, slot_temp):
    turn_acc, joint_acc, turn_cnt, joint_cnt = 0, 0, 0, 0
    
    for key in raw_file.keys():
        if key not in all_prediction.keys(): continue
        dial = raw_file[key]['log']
        for turn_idx, turn in enumerate(dial):
            belief_label = turn['belief']
            belief_pred = all_prediction[key][turn_idx]
            
            belief_label = [f'{k} : {v}' for (k,v) in belief_label.items()] 
            belief_pred = [f'{k} : {v}' for (k,v) in belief_pred.items()] 
            if turn_idx == len(dial)-1:
                logger.info(key)
                logger.info(f'label : {sorted(belief_label)}')
                logger.info(f'pred : {sorted(belief_pred)}')
                
            if set(belief_label) == set(belief_pred):
                joint_acc += 1
            joint_cnt +=1
            
            turn_acc += compute_acc(belief_label, belief_pred, slot_temp)
            turn_cnt += 1
            
    return joint_acc/joint_cnt, turn_acc/turn_cnt

def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.split(" : ")[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.split(" : ")[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC

def makedirs(path): 
   try: 
        os.makedirs(path) 
   except OSError: 
       if not os.path.isdir(path): 
           raise
       
       
       
def dict_to_csv(data, file_name):
    w = csv.writer(open(f'./logs/csvs/{file_name}', "a"))
    for k,v in data.items():
        w.writerow([k,v])
    w.writerow(['===============','================='])