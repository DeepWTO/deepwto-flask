from typing import Union, List

import requests
import json

from invokabilities import generate_sorted_dict_for_invokabilities


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

    def create_invokabilities(self, ds: int, split: str, scores: str, version: str = "1.0.0"):
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


if __name__ == "__main__":
    client = AppSyncClient(api_key="da2-ojotqixqrff5tdaarm7zpn5cfa",
                           endpoint_url="https://jpnk5vptq5djzmshkevn57a72y.appsync-api.us-east-1.amazonaws.com/graphql")

    for ds in available_ds:
        scores = generate_sorted_dict_for_invokabilities(ds=ds, invokability_dict_pkl="invokability_dict_train.pkl")
        res = client.create_invokabilities(ds=ds, split="train", scores=scores )
        print(res)