import json

def getData(filename):
    filename = filename
    # 1. Read file contents
    with open(filename, "r") as file:
        datos = json.load(file)

    return datos

def clasesNumero(data, clases):
    super_array = []

    for i in data:
        super_array.extend(i['labels'])
    print('Numero de objetos: ', len(super_array))
    for k, v in zip(clases.keys(), clases.values()):
        print('{} : {}'.format(k, super_array.count(v)))

class_list = getData('JSONfiles_G/label_map_G.json')

test_objects = getData('JSONfiles_G/TEST_objects_G.json')
train_objects = getData('JSONfiles_G/TRAIN_objects_G.json')
print('test')
print(len(test_objects))
clasesNumero(test_objects, class_list)
print('train')
print(len(train_objects))
clasesNumero(train_objects, class_list)