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


# factual_dict_tokenized = dict()
# article_dict_tokenized = dict()

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

    # data = load_pickle("data.pkl")
    # print(data.keys())
    # print(len(data["gov"].keys()))
    # print(len(data["art"].keys()))

    # result_dict_test = load_pickle("result_dict_train.pkl")
    # print(len(result_dict_test.keys()))
    # keys = list(result_dict_test.keys())
    # inst_num = 4
    # example_instance =result_dict_test[keys[inst_num]]
    # print(keys[inst_num])
    # print(example_instance)
    # print(example_instance['grad_cam_art'].shape)

    # invokability_dict_test = dict()
    # result_dict_test = load_pickle("result_dict_train.pkl")
    # keys = result_dict_test.keys()
    # for key in keys:
    #     ds = int(key.split('_')[0])
    #     art = key.split('_')[1]
    #     if ds in invokability_dict_test.keys():
    #         invokability_dict_test[ds][art] = (result_dict_test[key]["score"], result_dict_test[key]["input_label"])
    #     else:
    #         invokability_dict_test[ds] = dict()
    #         invokability_dict_test[ds][art] = (result_dict_test[key]["score"], result_dict_test[key]["input_label"])
    #
    # pickle_object(invokability_dict_test, "invokability_dict_train.pkl")


    def generate_sorted_dict_for_invokabilities(ds=18, invokability_dict_pkl="invokability_dict_test.pkl"):
        invokability_dict = load_pickle(invokability_dict_pkl)
        # print(invokability_dict_test.keys())
        # print(invokability_dict_test[ds].keys())
        print(invokability_dict[ds])
        # print(len(invokability_dict_test[ds].keys()))

        # sorted = {k: v for k, v in sorted(invokability_dict_test[ds].items(), key=lambda item: item[1][0])}
        sorted_list = sorted(invokability_dict[ds].items(), key=lambda item: item[1][0], reverse=True)
        # print(list(sorted))
        print(sorted_list)

        sorted_dicts = []
        for item in sorted_list:
            temp_dict = dict()
            temp_dict["name"] = item[0]
            temp_dict["pred"] = item[1][0]
            temp_dict["label"] = item[1][1]
            sorted_dicts.append(temp_dict)

        print(sorted_dicts)
        dump = json.dumps(str(sorted_dicts))
        print(dump)

        class mydict(dict):
            def __str__(self):
                return json.dumps(self)

    generate_sorted_dict_for_invokabilities(ds=2, invokability_dict_pkl="invokability_dict_test.pkl")
    # invokability_dict_train = load_pickle("invokability_dict_train.pkl")
    # print(invokability_dict_train.keys())
    # print(invokability_dict_train[ds].keys())
    # print(invokability_dict_train[ds])
    # print(len(invokability_dict_train[ds].keys()))

    pass
