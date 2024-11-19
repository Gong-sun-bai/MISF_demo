import json
import os
def load_file_list_recursion(fpath, result):
    allfilelist = os.listdir(fpath)
    for file in allfilelist:
        filepath = os.path.join(fpath, file)
        if os.path.isdir(filepath):
            load_file_list_recursion(filepath, result)
        else:
            result.append(filepath)
            print(len(result))



def scan(input_path, out_put):
    result_list = []
    load_file_list_recursion(input_path, result_list)
    result_list.sort()

    for i in range(len(result_list)):
        print('{}_{}'.format(i, result_list[i]))

    with open(out_put, 'w') as j:
        json.dump(result_list, j)

scan('/root/autodl-tmp/misf-main/data/image/test', './data/face_test.txt')
scan('/root/autodl-tmp/misf-main/data/image/train', './data/face_train.txt')
scan('/root/autodl-tmp/misf-main/data/image/val', './data/face_val.txt')
scan('/root/autodl-tmp/misf-main/data/mask/train', './data/mask_train.txt')
scan('/root/autodl-tmp/misf-main/data/mask/eval', './data/mask_eval.txt')
scan('/root/autodl-tmp/misf-main/data/mask/test_20', './data/mask_test_20.txt')
scan('/root/autodl-tmp/misf-main/data/mask/test_40', './data/mask_test_40.txt')
scan('/root/autodl-tmp/misf-main/data/mask/test_60', './data/mask_test_60.txt')