import pandas as pd
import numpy as np
import json
from os.path import dirname , abspath
import heapq
from scipy.sparse import csr_matrix , coo_matrix


ITEM_FILE =  'item_properties_part1.csv'
USER_FILE =  'events.csv'

TARGET_CATEGORY_ID = '888'



def create_ratings(user_file_loc , item_file_loc):
    print("Reading User File...")
    user_df = pd.read_csv(user_file_loc , dtype={'visitorid':'object' , 'itemid':'object'})
    print(user_df.dtypes)

    data , rows , cols = [] , [] , []
    user_dict , item_dict = {} , {}
    rowcols = set([])

    print("Creating Matrix...")
    for idx , row , in user_df.iterrows():
        user_id , item_id = row['visitorid'] , row['itemid']

        if user_id not in user_dict:
            user_dict[user_id] = len(user_dict)

        if item_id not in item_dict:
            item_dict[item_id] = len(item_dict)

        row_idx , col_idx = item_dict[item_id] , user_dict[user_id]

        if (row_idx , col_idx) not in rowcols:
            rows.append(row_idx)
            cols.append(col_idx)
            data.append(1)
            rowcols.add((row_idx , col_idx))

    shapi = (len(item_dict) , len(user_dict))
    print(shapi)
    data = np.array(data)
    xc = pd.DataFrame(data)
    print(xc)
    matrix = coo_matrix((data , (rows , cols)) , shape=shapi)
    matrix = matrix.tocsr()
    return matrix , user_dict , item_dict


def get_similarity(rating_matrix):
    print("Creating item similarity matrix...")
    A = rating_matrix.astype(np.int64)
    m = sparse_corrcoef(A)

    return m 

def sparse_corrcoef(A, B=None):
    A = A.astype(np.float64)
    A = A - A.mean(1)
    norm = A.shape[1] - 1.
    C = A.dot(A.T.conjugate()) / norm
    d = np.diag(C)
    coeffs = C / np.sqrt(np.outer(d, d))
    return coeffs


class Recommendations(object):
    INSTANCE = None
    valid_items = set([])
    target_category_id = ''
    item_file = ''

    def __init__(self , target_category_id , item_file):
        Recommendations.target_category_id = target_category_id
        Recommendations.item_file = item_file
        self.find_valid_items()

    @classmethod
    def find_valid_items(cls):
        df = pd.read_csv(cls.item_file)
        items = df['itemid'].tolist()

        for c in items:
            if ((df['itemid'] == c) & (df['property'] == cls.target_category_id)).any():
                cls.valid_items.add(c)

    @classmethod
    def get_instance(cls , target_category_id , item_file):
        if not cls.INSTANCE:
            cls.INSTANCE = Recommendations(target_category_id , item_file)
            return cls.INSTANCE

    
    def get_top_k_recommendations(self, rating_matrix, similarity_matrix, user_id, user_dict, item_dict, k=5):
        print("Recommending for " + str(user_id))

        inv_item_dict = {v: k for k, v in item_dict.items()}
        col_idx = user_dict[user_id]
        user_col = rating_matrix.getcol(col_idx).toarray().flatten()

        rated_item_indexes = [idx for idx, val in enumerate(user_col) if val == 1]
        predicted_ratings = []

        for row_idx, val in enumerate(user_col):
            item_id = inv_item_dict[row_idx]

            # Ignore the challenge if it is not a part of Target Contest
            if item_id not in self.valid_items:
                continue 

            if val != 1:
                similarities = []
                for idx in rated_item_indexes:
                    similarities.append(similarity_matrix[row_idx, idx])

                most_similar_items = heapq.nlargest(3, similarities)
                predicted_ratings.append((np.mean(most_similar_items), row_idx))

        inv_item_dict = {v: k for k, v in item_dict.items()}

        recommendations = [inv_item_dict[row_idx] for score, row_idx in heapq.nlargest(k, predicted_ratings)]
        return recommendations


if __name__ == '__main__':
    rating_matrix , user_dict , item_dict = create_ratings(USER_FILE , ITEM_FILE)
    similarity_matrix = get_similarity(rating_matrix)

    print("Getting Recommendations...")
    all_recommendations = {}
    for i  in user_dict:
        R = Recommendations.get_instance(TARGET_CATEGORY_ID , ITEM_FILE)
        recommenda = R.get_top_k_recommendations(rating_matrix , similarity_matrix ,i, user_dict , item_dict , 10)
        all_recommendations[i] = recommenda

    with open('all_recommendations.json' , 'w') as f:
        json.dump(all_recommendations , f)