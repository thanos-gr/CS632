import os
import re
import pandas as pd
import gensim
import keras
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from matplotlib import pyplot

path = '/home/thanos_kats/DeepLearning/PROJECT/ohsumed-all/asdffa/'

path_flag = (False, True)[os.path.exists(path)]
csv_flag = (False, True)[os.path.exists(os.path.join(os.getcwd(),"Health.csv"))]
printos.pah.join(os.getcwd(),"Health.csv"))
stop_words = set(stopwords.words('engish'))

dic_mapping = {
	'C01':'Bacterial Infections and Mycoses',
	'C02':'Virus Diseases',
	'C03':'Parasitic Diseases',
	'C04':'Neoplasms',
	'C05':'Musculoskeletal Diseases',
	'C06':'Digestive System Diseases',
	'C07':'Stomatognathic Diseases',
	'C08':'Respiratory Tract Diseases',
	'C09':'Otorhinolaryngologic Diseases',
	'C10':'Nervous System Diseases',
	'C11':'Eye Diseases',
	'C12':'Urologic and Male Genital Diseases',
	'C13':'Female Genital Diseases and Pregnancy Complications',
	'C14':'Cardiovascular Diseases',
	'C15':'Hemic and Lymphatic Diseases',
	'C16':'Neonatal Diseases and Abnormalities',
	'C17':'Skin and Connective Tissue Diseases',
	'C18':'Nutritional and Metabolic Diseases',
	'C19':'Endocrine Diseases',
	'C20':'Immunologic Diseases',
	'C21':'Disorders of Environmental Origin',
	'C22':'Animal Diseases',
	'C23':'Pathological Conditions, Signs and Symptoms' }

class MySentences(object):
    def __init__(self, dirname, fname):
        self.dirname = dirname
	self.fname = fname
 
    def __iter__(self):
    	df = pd.read_csv(os.path.join(self.dirname, self.fname), sep=',')
	for line in df['Text']:
                yield line.split(' ')

def stopWords(text):
        filtered_sent = []
	for sentence in text:
		word_tokens = word_tokenize(sentence)
		filt_sent = [w for w in word_tokens if not w in stop_words and w not in \
			    ['%','=','(',')',','] and not re.search('[0-9]+',w)]
		filtered_sent.extend(filt_sent)
						
	text_string = ' '.join(filtered_sent)
	return text_string

def create_df():
        if path_flag:
                df = pd.DataFrame(columns=['Text','Labels'])
                text_list=[]
		label_list = []
		for _dir in os.listdir(path):
			if os.path.isdir(os.path.join(path, _dir)):
				rel_path = os.path.join(path, _dir)
				files = [os.path.join(rel_path, f) for f in os.listdir(rel_path)]
				for _file in files:
					print('Reading ',_file)
					label_list.append(dic_mapping[_dir])
					with open(_file ,'r') as input_file:
						raw_text = input_file.readlines()
						text_stop = stopWords(raw_text)
						text_list.append(text_stop)				
		df['Text'] = text_list
		df['Labels'] = label_list
                
		print(df.head())
		df.to_csv('Health.csv',sep=',',index=False)
	elif csv_flag:
		print(True)
		sentences = MySentences(os.getcwd(),'Health.csv')
		print('Creating Model')
		w2v = gensim.models.Word2Vec(sentences,iter=5, min_count=10, size=200, workers =5)
		print(w2v)
		print('Saving Model')
		w2v.save('w2v_health.bin')
	
	
if __name__ == '__main__':
	df = pd.read_csv(os.path.join(os.getcwd(),'Health.csv'), sep=',')
	print(len(df))
