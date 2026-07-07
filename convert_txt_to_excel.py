import os
#from striprtf.striprtf import rtf_to_text
import argparse

from striprtf import striprtf
rtf_to_text = striprtf.rtf_to_text

import argparse

#import xlsx
import pandas as pd

def write_np_str_array_to_excel(array, file, header):
    df = pd.DataFrame(array)
    df.to_excel(file, header=header, index=False)

def read_excel_to_np_str_array(file, header):
    df = pd.read_excel(file, header=header)
    return df.values

def get_date_str(val_lines):
    datum_str = val_lines.split(')')[-1][1:11].replace('-','_')
    datum = datum_str[8:10] + '_' + datum_str[5:7] + '_' + datum_str[0:4]
    return datum

def get_title_str(text_lines):
    title = text_lines.split('(')[1].split(')')[0]
    title_sub = title[:min(len(title), 25)]
    return title_sub, title

def get_text_str(content):
    text = content
    return text

def get_paragraphs(text):
    paragraphs = text.split('\n')
    return paragraphs
    
def filter_paragraphs(paragraphs, keywords=['ai', 'ki', 'intelligenz', 'chatgpt'], forbidden_words=[]):
    filtered_paragraphs = []
    for paragraph in paragraphs:
      this_key_words = [word for word in keywords if word in paragraph]
      this_forbidden_words = [word for word in forbidden_words if word in paragraph]
      if len(this_key_words) > 0 and len(this_forbidden_words) == 0:
        filtered_paragraphs.append(paragraph)
    #print(filtered_paragraphs)
    return filtered_paragraphs 


def parse_date(file):
    with open(file, 'r', encoding="utf8") as f:
        file_data = f.read()
        all = file_data.split('-----------------------------------------------------------------------------')
        print(len(all))
        val = all[0]
        
        text = get_text_str(all[1])
        paragraphs = get_paragraphs(text)
        #filtered_paragraphs = filter_paragraphs(paragraphs, ['ai', 'ki', 'Intelligenz', 'intelligenz','AI','KI','ChatGPT'], ['http'])   
        # 以下一行是用于中文文本
        filtered_paragraphs = filter_paragraphs(paragraphs, ['人工智能', '智能', 'AI','ChatGPT'], ['http'])    
        print(len(filtered_paragraphs), len(paragraphs))
        title_str, title = get_title_str(val)
        print(title, title_str)
        datum_str = get_date_str(val)
        print(datum_str)
        return datum_str, title_str, title, text, filtered_paragraphs

def rename_files(folder, label_file):
    print(folder)
    paths = [os.path.join(folder, file) for file in os.listdir(folder)]
    print('paths', paths)


    header = ['file', 'datum', 'title', 'text', 'label1', 'label2', 'label3', 'label4','label5','label6']
    label_values = []
    cnt = 0
    for path in paths:
        try:
            datum_str, title_str, full_title, text, filtered_paragraphs = parse_date(path)
            #print(datum_str, title_str)
            new_path = os.path.join(folder, datum_str + '_' + title_str + '.rtf')
            #if new_path == path:
            #    continue
            for paragraph in filtered_paragraphs:
                label_value = [path, datum_str.replace('_', " "), full_title.replace("_", " "), paragraph, '0', '0', '0', '0', '0', '0']
                label_values.append(label_value)
            '''
            cnt = 0
            while os.path.exists(new_path):
                new_path = os.path.join(folder, datum_str + '_' + title_str + '_' + str(cnt) + '.rtf')
                cnt += 1
            '''
            #os.rename(path, new_path)
            #if cnt > 10:
            #    break
            cnt += 1
        except Exception as e:
            print(str(e))
    write_np_str_array_to_excel(label_values, label_file, header)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='.\\chinacorpusArticle1\\xinminwanbao')
    parser.add_argument('--labels_file', type=str, default='labelsxinminwanbao.xlsx')
    args = parser.parse_args()
    print('hello sunshine')
    #all_folders = [os.path.join(args.folder, folder) for folder in os.listdir(args.folder)] 
    #for folder in all_folders:
        #if os.path.isdir(folder):
rename_files(args.folder, args.labels_file)