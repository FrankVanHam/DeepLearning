import numpy as np

class Processor:
    def __init__(self, word_to_vec_map):
        self.word_to_vec_map = word_to_vec_map

    def cosine_similarity(self, u, v):
        """
        Cosine similarity reflects the degree of similarity between vector u and v.  It is the similarity as the normalized dot product of X and Y.
        """
        
        # Special case.
        if np.all(u == v):
            return 1
        
        # dot product
        dot = np.dot(u,v) 
        
        # Compute the L2 norm of u
        norm_u = np.sqrt(np.sum(u*u))
        
        # Compute the L2 norm of v
        norm_v = np.sqrt(np.sum(v*v))
        
        # Avoid division by 0
        if np.isclose(norm_u * norm_v, 0, atol=1e-32):
            return 0
        else:
            return dot/(norm_u * norm_v)

    def complete_analogy(self, word_a, word_b, word_c):
        """
        Performs the word analogy task. Returns the best_word as measured by cosine similarity
        """
        
        # convert words to lowercase
        word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
        
        # Get the word embeddings e_a, e_b and e_c
        e_a, e_b, e_c = self.word_to_vec_map[word_a], self.word_to_vec_map[word_b], self.word_to_vec_map[word_c]
        
        words = self.word_to_vec_map.keys()
        max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
        best_word = None
        
        # loop over the whole word vector set
        for w in words: 
            if w == word_c: continue
            # Compute cosine similarity
            cosine_sim = self.cosine_similarity(e_b-e_a, self.word_to_vec_map[w]-e_c)
            if cosine_sim > max_cosine_sim:
                max_cosine_sim = cosine_sim
                best_word = w
            
        return best_word