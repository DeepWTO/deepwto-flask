import json
import pickle

input_file = "/home/zachary/test_data.json"


def pickle_object(python_obj, pickle_path):
    with open(pickle_path, "wb") as f:
        pickle.dump(python_obj, f)
    return True


def load_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        python_obj = pickle.load(f)
    return python_obj


# with open(input_file) as fin:
#     # print(fin.keys())
#     for each_line in fin:
#         data = json.loads(each_line)
#         # print(data.keys())
#         # print(data['testid'])
#         if data['testid'] == '139_[Article III:4]' :
#             print('yes')
#             pickle_object(data, '139_[Article III:4].pkl')
#             break

if __name__ == "__main__":
    fp = '/tmp/pycharm_project_271/139_[Article III:4].pkl'
    data = load_pickle(fp)
    print(data.keys())
    print(data['gov'])
    print(data['art'])
    pass
