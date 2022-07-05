import json

filename = '/home/fp/Escritorio/LuisBringas/FCOS/results/Results_TransferLearning_CPU.json' #TransferLearning
#filename = '/home/fp/Escritorio/LuisBringas/FCOS/results/Losses_FineTuning.json' #FineTuning
#filename = '/home/fp/Escritorio/LuisBringas/FCOS/results/Losses_FromScratch.json' #FromScratch

with open(filename, "r") as file:
    datos = json.load(file)

def parserFloat(array):
    return [float(v) for v in array] 

print("NMS: ", len(datos['NMS']))
print("El tiempo promedio NMS es: {}".format(sum(parserFloat(datos['NMS']))/len(datos['NMS'])))

print("DRAW: ", len(datos['Draw']))
print("El tiempo promedio del draw es: {}".format(sum(parserFloat(datos['Draw']))/len(datos['Draw'])))

print("RED: ", len(datos['Red']))
print("El tiempo promedio de la red es: {}".format(sum(parserFloat(datos['Red']))/len(datos['Red'])))

print("GENERAL: ", len(datos['General']))
print("El tiempo promedio general es: {}".format(sum(parserFloat(datos['General']))/len(datos['General'])))