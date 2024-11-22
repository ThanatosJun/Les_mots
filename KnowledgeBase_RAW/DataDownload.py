import kagglehub
# French Dictionary
path = kagglehub.dataset_download("kartmaan/dictionnaire-francais")
print("Path to dataset files:", path)
# English French Dictionary
path = kagglehub.dataset_download("abdallahwagih/english-france-dictionary")
print("Path to dataset files:", path)
# English French Translation Dataset
path = kagglehub.dataset_download("dhruvildave/en-fr-translation-dataset")
print("Path to dataset files:", path)