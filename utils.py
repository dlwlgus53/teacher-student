import pdb
from base_logger import logger
def calculate_diff(args, teacher_tagged, student_tagged):
    diff_count = 0.0
    diff_all =0.0
    for dial_id in teacher_tagged:
        for turn_id in teacher_tagged[dial_id]:
            for schema in teacher_tagged[dial_id][turn_id]:
                teacher_tag = teacher_tagged[dial_id][turn_id][schema]
                student_tag = student_tagged[dial_id][turn_id][schema]
                diff_all += 1
                if  (teacher_tag == student_tag).all(): diff_count +=1
    return 1- (diff_count/diff_all)

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

