import time 
from tqdm import tqdm

class SparseRetriever:
    
    def __init__(self, collection, bow, ranking_model):
        self.collection = collection
        self.backend = self.collection.backend
        self.bow = bow
        self.ranking_model = ranking_model
        
        # apply ranking_model?
        
        # send collection_to_gpu?
        
    def retrieve(self, questions_iterator, profile=False):
        
        for question in tqdm(questions_iterator):
            question = self.ranking_model.question_transform(self.bow(question))
            