import time 
from tqdm import tqdm

class SparseRetriever:
    
    def __init__(self, collection, bow, weighting_model):
        # apply ranking_model
        self.collection = self.weighting_model.transform_collection(collection)
        self.backend = self.collection.backend
        self.bow = bow
        self.weighting_model = weighting_model
        
        # apply ranking_model?

        
    def retrieve(self, questions_iterator, top_k=1000):
        
        sparse_collection_matrix = self.collection.get_sparce_matrix()
        
        for question in tqdm(questions_iterator):

            question = self.weighting_model.question_transform(self.bow(question))
            
            # perform search
            query = self.backend.create_dense_tensor_from_bow(question)
            
            
            