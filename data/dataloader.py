import pandas as pd

def read_csv():
    data = pd.read_csv(r"C:\Users\nivy1\Documents\Engineering\MS.c\Natural Language Processing\ner-food-recipies\data\TASTEset.csv")
    return data

d = read_csv()
print(d['ingredients'][0])