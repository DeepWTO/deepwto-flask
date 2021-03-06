from typing import Union, List

import requests
import json

import numpy as np

from tqdm import tqdm


from invokabilities import generate_sorted_dict_for_invokabilities
import pickle

def load_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        python_obj = pickle.load(f)
    return python_obj

available_ds = [2, 18, 22, 31, 34, 46, 56, 58, 60, 62, 67, 68, 69, 75, 76,
                87, 90, 98, 103, 108, 121, 122, 135, 136, 139, 141, 146, 152,
                155, 161, 162, 165, 166, 174, 175, 177, 184, 202, 207, 212,
                217, 219, 221, 231, 234, 238, 244, 245, 246, 248, 257, 264,
                265, 266, 267, 268, 269, 276, 282, 283, 286, 290, 294, 295,
                296, 301, 302, 308, 312, 315, 316, 320, 321, 322, 332, 336,
                339, 343, 344, 345, 350, 353, 360, 363, 366, 371, 379, 381,
                384, 392, 394, 396, 397, 399, 400, 406, 412, 414, 415, 422,
                425, 427, 429, 430, 431, 435, 436, 437, 440, 442, 447, 449,
                453, 454, 456, 457, 461, 464, 468, 471, 472, 473, 475, 476,
                477, 479, 480, 482, 483, 484, 485, 486, 488, 490, 492, 493,
                495, 499, 504, 505, 513, 518, 523]

available_articles = ['Article I', 'Article I:1', 'Article II', 'Article II:1', 'Article II:1(a)', 'Article II:1(b)', 'Article II:2', 'Article II:3', 'Article III', 'Article III:1', 'Article III:2', 'Article III:4', 'Article III:5', 'Article III:7', 'Article IV', 'Article IX', 'Article IX:2', 'Article V', 'Article V:1', 'Article V:2', 'Article V:3', 'Article V:3(a)', 'Article V:4', 'Article V:5', 'Article V:6', 'Article V:7', 'Article VI', 'Article VI:1', 'Article VI:2', 'Article VI:2(a)', 'Article VI:2(b)', 'Article VI:3', 'Article VI:5(a)', 'Article VI:6', 'Article VII', 'Article VII:1', 'Article VII:2', 'Article VII:5', 'Article VIII', 'Article VIII:1', 'Article VIII:3', 'Article VIII:4', 'Article X', 'Article X:1', 'Article X:2', 'Article X:3', 'Article X:3(a)', 'Article XI', 'Article XI:1', 'Article XIII', 'Article XIII:1', 'Article XIII:2', 'Article XIII:3(b)', 'Article XIX', 'Article XIX:1', 'Article XIX:2', 'Article XIX:3', 'Article XV', 'Article XVI', 'Article XVI:1', 'Article XVI:4', 'Article XVII', 'Article XVII:1', 'Article XVII:1(c)', 'Article XVIII', 'Article XVIII:10', 'Article XVIII:11', 'Article XX', 'Article XXI', 'Article XXII', 'Article XXII:1', 'Article XXIII', 'Article XXIII:1', 'Article XXIII:1(a)', 'Article XXIII:1(b)', 'Article XXIV', 'Article XXIV:12', 'Article XXIV:5(b)', 'Article XXIV:6', 'Article XXVIII']

class AppSyncClient:
    latest_version = "1.0.0"

    def __init__(self, api_key, endpoint_url):
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.headers = {
            "Content-Type": "application/graphql",
            "x-api-key": api_key,
            "cache-control": "no-cache",
        }

    def execute_gql(self, query):
        payload_obj = {"query": query}
        payload = json.dumps(payload_obj)
        response = requests.request(
            "POST", self.endpoint_url, data=payload, headers=self.headers
        )
        return response

    def create_invokabilities(self, ds: int, split: str, scores: List[dict], version: str = "1.0.0"):
        ds_split = '"{}_{}"'.format(str(ds), split)
        version = '"{}"'.format(version)
        scores = '"{}"'.format(scores)
        query = """
                mutation CreateInvokabilities {{
                    createInvokabilities(input: 
                        {{
                        ds_split: {0}, 
                        version: {1},
                        scores: {2}
                        }}
                        ) {{
                               ds_split
                               version
                               scores
                            }}
                        }}
                """.format(
            ds_split, version, scores
        )

        res = self.execute_gql(query).json()
        return res["data"]["createInvokabilities"]["ds_split"]

    def create_gov_tokens(self, ds: int, tokens: List[str], version: str = "1.0.0"):
        ds = '{}'.format(ds)
        version = '"{}"'.format(version)
        tokens = '{}'.format(tokens)
        query = """
                   mutation CreateGovTokenized {{
                       createGovTokenized(input: 
                           {{
                           ds: {0},
                           version: {1},
                           tokens: {2}
                           }}
                           ) {{
                                  ds
                                  version
                                  tokens
                               }}
                           }}
                   """.format(
            ds, version, tokens
        )

        res = self.execute_gql(query).json()
        print(res)
        return res["data"]["createGovTokenized"]["ds"]

    def create_gov_gradcam(self, ds_art: str, weights: List[float], version: str = "1.0.0"):
        print(weights)
        ds_art = '"{}"'.format(ds_art)
        version = '"{}"'.format(version)
        weights = '{}'.format(weights)
        query = """
                   mutation CreateGovGradCam {{
                       createGovGradCAM(input: 
                           {{
                           ds_art: {0}, 
                           version: {1},
                           weights: {2}
                           }}
                           ) {{
                                  ds_art
                                  version
                                  weights
                               }}
                           }}
                   """.format(
            ds_art, version, weights
        )

        res = self.execute_gql(query).json()
        print(res)
        return res["data"]["createGovGradCAM"]

    def create_article_tokens(self, article: int, tokens: List[str], version: str = "1.0.0"):
        article = '"{}"'.format(article)
        version = '"{}"'.format(version)
        tokens = '{}'.format(tokens)
        query = """
                      mutation CreateArticleTokenized {{
                          createArticleTokenized(input: 
                              {{
                              article: {0},
                              version: {1},
                              tokens: {2}
                              }}
                              ) {{
                                     article
                                     version
                                     tokens
                                  }}
                              }}
                      """.format(
            article, version, tokens
        )
        res = self.execute_gql(query).json()
        print(res)
        return res["data"]["createArticleTokenized"]

    def create_article_gradcam(self, ds_art: str, weights: List[float], version: str = "1.0.0"):
        ds_art = '"{}"'.format(ds_art)
        version = '"{}"'.format(version)
        weights = '{}'.format(weights)
        query = """
                   mutation CreateArticleGradCam {{
                       createArticleGradCAM(input: 
                           {{
                           ds_art: {0}, 
                           version: {1},
                           weights: {2}
                           }}
                           ) {{
                                  ds_art
                                  version
                                  weights
                               }}
                           }}
                   """.format(
            ds_art, version, weights
        )

        res = self.execute_gql(query).json()
        print(res)
        return res["data"]["createArticleGradCAM"]


if __name__ == "__main__":
    client = AppSyncClient(api_key="da2-ojotqixqrff5tdaarm7zpn5cfa",
                           endpoint_url="https://jpnk5vptq5djzmshkevn57a72y.appsync-api.us-east-1.amazonaws.com/graphql")

    # for ds in available_ds:
    #     scores = generate_sorted_dict_for_invokabilities(ds=ds, invokability_dict_pkl="invokability_dict_train.pkl")
    #     res = client.create_invokabilities(ds=ds, split="train", scores=scores )
    #     print(res)

    # fp = 'factual_dict_tokenized.pkl'
    # data = load_pickle(fp)
    # for ds in available_ds:
    #     tokens = json.dumps(data[ds])
    #     res = client.create_gov_tokens(ds=ds, tokens=tokens)
    #     print(res)

    # fp = 'result_dict_train.pkl'
    # data = load_pickle(fp)
    # keys = list(data.keys())
    #
    # for i, key in enumerate(keys):
    #     print(i)
    #     print(key)
    #     grad_cam_gov = list(data[key]['grad_cam_gov'])
    #     res = client.create_gov_gradcam(ds_art=key, weights=grad_cam_gov)
    #     print(res)


    # fp = 'article_dict_tokenized.pkl'
    # data = load_pickle(fp)
    # print(len(available_articles))
    # for article in available_articles:
    #     tokens = json.dumps(data[article])
    #     res = client.create_article_tokens(article=article, tokens=tokens)

    fp = 'result_dict_test.pkl'
    data = load_pickle(fp)
    keys = list(data.keys())

    for i, key in tqdm(enumerate(keys)):
        print(key)
        grad_cam_art = list(data[key]['grad_cam_art'])
        res = client.create_article_gradcam(ds_art=key, weights=grad_cam_art)
        print(res)