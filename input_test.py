import json
import pickle

input_file = "/Users/zachary/Downloads/test_data.json"


def pickle_object(python_obj, pickle_path):
    with open(pickle_path, "wb") as f:
        pickle.dump(python_obj, f)
    return True


def load_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        python_obj = pickle.load(f)
    return python_obj


factual_dict_tokenized = dict()
article_dict_tokenized = dict()

# with open(input_file) as fin:
#     for each_line in fin:
#         data = json.loads(each_line)
#         ds_art = data['testid']
#         ds = int(ds_art.split("_")[0])
#         art = ds_art.split("_")[1][1:-1]
#         print(ds, art)
#         print(article_dict_tokenized.keys())
#
#         if ds not in article_dict_tokenized.keys():
#             article_dict_tokenized[art] = data['art']
#         else:
#             pass
#
#     pickle_object(article_dict_tokenized, "article_dict_tokenized.pkl")

if __name__ == "__main__":
    # fp = 'factual_dict_tokenized.pkl'
    # data = load_pickle(fp)
    # print(data.keys())
    # prev = None
    # prev_key = None
    #
    # for key in data.keys():
    #     curr = data[key]
    #     print(key)
    #
    #     if prev == curr:
    #         print("duplicated at", prev_key, key)
    #
    #     prev = curr
    #     prev_key = key

    # fp = 'article_dict_tokenized.pkl'
    # data = load_pickle(fp)
    # print(data.keys())
    # print(len(data.keys()))
    # prev = None
    # prev_key = None
    #
    # for key in data.keys():
    #     curr = data[key]
    #     print(key)
    #     print(curr)
    #
    #     if prev == curr:
    #         print("duplicated at", prev_key, key)
    #
    #     prev = curr
    #     prev_key = key

    # facts_pkl = 'factual_dict_tokenized.pkl'
    # arts_pkl = 'article_dict_tokenized.pkl'
    # facts = load_pickle(facts_pkl)
    # arts = load_pickle(arts_pkl)
    #
    # data_dicts = dict()
    # data_dicts["gov"] = facts
    # data_dicts["art"] = arts
    #
    # pickle_object(data_dicts, "data.pkl")

    data = load_pickle("data.pkl")
    print(data.keys())
    print(len(data["gov"].keys()))
    print(len(data["art"].keys()))
    pass
