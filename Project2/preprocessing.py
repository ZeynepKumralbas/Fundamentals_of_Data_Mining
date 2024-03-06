import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv
from mlxtend.preprocessing import TransactionEncoder


class preprocessing:

    def __init__(self):

        data = pd.read_csv("seeds_dataset.txt", delim_whitespace=True, lineterminator='\n',
                           names=["Area", "Perimeter", "Compactness", "Length of kernel", "Width of kernel",
                                  "Asymmetry coefficient", "Length of kernel groove", "Label"])

        data_x = data[["Area", "Perimeter", "Compactness", "Length of kernel", "Width of kernel",
                                  "Asymmetry coefficient", "Length of kernel groove"]]

        self.data_y = data["Label"]

        self.data_x = self.normalization(data_x)

     #   print(self.data_x)
     #   print(self.data_y)

        dataset = []
        with open('cse4063-spring2020-project-2-dataset-fpm.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                row_data = row[:-1]  # delete the last (empty) element
                dataset.append(row_data)
                line_count += 1

        #print("dataset")
        #print(dataset)

        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        pd.set_option('display.max_columns', 20)
 #       print("Transaction Encoder")
  #      print(df)
        self.FPM_data = df


    def normalization(self, data_x):
        scaler = MinMaxScaler()
        data_x = pd.DataFrame(data=scaler.fit_transform(data_x.iloc[:, :]), columns=data_x.columns)
        return data_x

    def get_data(self):
        return self.data_x, self.data_y

    def get_FPM_data(self):
        return self.FPM_data
