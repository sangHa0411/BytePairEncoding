import os
import json

def read_data(file_path) :
    with open(file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    return json_data         
              
def get_data(json_data) :
    text_data = []
    doc_data = json_data['document']
    for doc in doc_data :
        data_list = doc['utterance']
        dialogue = [data['form'] for data in data_list]
        text_data.extend(dialogue)
    return text_data

def load_data(dir_path) :
    data_list = os.listdir(dir_path)
    json_data = []
    for data in data_list :
        if data.endswith('.json') :
            file_path = os.path.join(dir_path, data)
            try :
                json_file = read_data(file_path)
                json_data.append(json_file)
            except UnicodeDecodeError :
                continue
                
    text_data = []
    for json_file in json_data :
        text_list = get_data(json_file)
        text_data.extend(text_list)
    return text_data
