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
