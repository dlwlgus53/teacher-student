import re
import pdb
import json
import torch
import pickle
import ontology
import tokenizer_config as tc
from tqdm import tqdm
from base_logger import logger

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, type, tokenizer):
        self.tokenizer = tokenizer
        
        pickle_path = f'./data/preprocessed_train0.01.pickle'
        raw_path = f'{data_path[:-5]}.json'

        if type == 'train':
            pickle_path = f'./data/preprocessed_train0.001.pickle'
            raw_path = f'{data_path[:-5]}.json'
            
        try:
            logger.info(f"load {pickle_path}")
            with open(pickle_path, 'rb') as f:
                item = pickle.load(f)
            
            self.source = item['source']
            self.target = item['target']
            self.schema = item['schema']
            self.dial_id = item['dial_id']
            self.turn_id = item['turn_id']
                            
        except:
            logger.info("Failed to load processed file. Start processing")
            raw_dataset = json.load(open(raw_path , "r"))
            context, question, answer,  belief, dial_id, turn_id, schema = self.seperate_data(raw_dataset)
            # TODO belief에  mltiple 번호 나온다
            assert len(context)==len(question) == len(schema) == len(belief) == len(dial_id) == len(turn_id)
            
            input_text = [f"question: {q} context: {c} belief: {b}" for (q,c,b) in zip(question, context, belief)]

            logger.info("Encoding dataset (it will takes some time)")
            logger.info("encoding input text")
            self.source = self.encode(input_text)
            logger.info("encoding answer")
            self.target = self.encode(answer)
            self.schema = schema
            self.dial_id = dial_id
            self.turn_id = turn_id
            
            item = {
                'source' : self.source,
                'target' : self.target,
                'schema' : self.schema,
                'dial_id' : self.dial_id,
                'turn_id' : self.turn_id,
            }
            
            with open(pickle_path, 'wb') as f:
                pickle.dump(item, f, pickle.HIGHEST_PROTOCOL)
            
            
    def encode(self, texts ,return_tensors="pt"):
        examples = []
        for i, text in enumerate(texts):
            if i%100 == 0:
                logger.info(f'{i}/{len(texts)}')
            
            while True:
                tokenized = self.tokenizer.batch_encode_plus([text], padding=False, return_tensors=return_tensors) # TODO : special token
                if len(tokenized)> self.tokenizer.model_max_length:
                    idx = [m.start() for m in re.finditer("\[user\]", text)]
                    text = text[:idx[0]] + text[idx[1]:] # delete one turn
                else:
                    break
                
            examples.append(tokenized)
        return examples

    def __getitem__(self, index):
        source = {k:v.squeeze() for (k,v) in self.source[index].items()}
        target = {k:v.squeeze() for (k,v) in self.target[index].items()}
            
        return {"source": source, "target": target, \
                "turn_id" : (self.turn_id[index]), "dial_id" : (self.dial_id[index]), "schema":(self.schema[index])}
    
    def __len__(self):
        return len(self.source)

    def seperate_data(self, dataset):
        context = []
        question = []
        belief = []
        answer = []
        schema = []
        dial_id = []
        turn_id = []
        
        print(f"preprocessing data")
        for id in dataset.keys():
            dialogue = dataset[id]['log']
            dialogue_text = ""
            b = {}
            
            for i, turn in enumerate(dialogue):
                dialogue_text += '[user] '
                dialogue_text += turn['user']
                for key_idx, key in enumerate(ontology.QA['all-domain']): # TODO
                    q = ontology.QA[key]['description']
                    c = dialogue_text
                    
                    if key in turn['belief']: # 언급을 한 경우
                        a = turn['belief'][key]
                        if isinstance(a, list) : a= a[0] # in muptiple type, a == ['sunday',6]
                    else:
                        a = ontology.QA['NOT_MENTIONED']
                    
                    schema.append(key)
                    answer.append(a)
                    context.append(c)
                    question.append(q)
                    belief.append(b)
                    dial_id.append(id)
                    turn_id.append(i)
                    
                b = turn['belief'] #  하나씩 밀려서 들어가야함.! 유저 다이얼처럼
                dialogue_text += '[sys] '
                dialogue_text += turn['response']
        
        return context, question, answer,  belief, dial_id, turn_id, schema
    
    

    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.

        Collate function is applied to the output of a DataLoader as it is yielded.
        """
        # 여기다 패딩하고 마스크도 다시다시
        # "source": self.source[index], "target": self.target[index],
        
        pad_source = self.tokenizer.pad([x["source"] for x in batch],padding=True)
        pad_target = self.tokenizer.pad([x["target"] for x in batch],padding=True)
        schema = [x["schema"] for x in batch]
        dial_id = [x["dial_id"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        
        
        return {"input": pad_source, "target": pad_target,\
                 "schema":schema, "dial_id":dial_id, "turn_id":turn_id}
        
        # return pad_source
    
    

if __name__ == '__main__':
    data_path = '../woz-data/MultiWOZ_2.1/train_data.json'
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    type = 'train'
    
    dataset = Dataset(data_path, type, data_rate = 0.01, tokenizer= tokenizer) 
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, collate_fn=dataset.collate_fn)
        
    for batch in loader:
        pdb.set_trace()
    
    