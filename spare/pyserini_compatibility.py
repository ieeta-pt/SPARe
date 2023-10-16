"""
Auxiliar classes to load pyserini indexes
"""
from spare.text2vec import BagOfWords
from pyserini.index.lucene import IndexReader
import jnius

def bm25_pyserini_iterator(index_reader, k1=0.9, b=0.7):

    token2id = {term.term:i for i,term in enumerate(index_reader.terms())}
    
    for i in range(index_reader.stats()['documents']):
        docid = index_reader.convert_internal_docid_to_collection_docid(i)
        bm25_vector = {}
        try:
            for term in index_reader.get_document_vector(docid):
                bm25_vector[token2id[term]] = index_reader.compute_bm25_term_weight(docid, term, analyzer=None, k1=k1, b=b)
        except jnius.JavaException as e:
            print("empty doc", docid)
            # index a empty vec this can probably be avoided, but the shape is currently set before. Maybe add pruning methods
            
        yield docid, bm25_vector

class PyseriniBOW(BagOfWords):
    
    def __init__(self, index_folder):
        
        index_reader = IndexReader(index_folder)
        self.token2id = {term.term:i for i,term in enumerate(index_reader.terms())}
        
        def tokenizer(text):
            tokens_ids = []
            for token in index_reader.analyze(text.lower()):
                #token_id=token2id[token]
                if token in self.token2id:
                    tokens_ids.append(self.token2id[token])
                #if token in token2id:
                #    token_id=token2id[token]
                #    if token_id is not None:
                #        tokens_ids.append(token_id)
            return tokens_ids
        
        super().__init__(tokenizer, index_reader.stats()["unique_terms"])
        
