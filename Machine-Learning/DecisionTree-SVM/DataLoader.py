import numpy as np

class DataLoader:
    @staticmethod
    def load_credit(file_path):
        dataset = []
        labels = []
        file = open(file_path, "r")
        for line in file:
            line = line.strip("\n\r")
            # to get rid of extra lines that do not contain any information
            if len(line) < 5:
                continue
            parts = line.split(" ")
            data = []
            numeric_transformer = float
            # Attribute 1: (qualitative) - Status of existing checking account
            attr_1 = {'A11': 0, 'A12': 1, 'A13': 2, 'A14': 3}
            data.append(attr_1[parts[0]]) #
            # Attribute 2: (numerical) - Duration in month
            data.append(numeric_transformer(parts[1]))
            # Attribute 3: (qualitative) Credit history
            attr_3 = {'A30':0, 'A31':1, 'A32':2, 'A33':3, 'A34':4}
            data.append(attr_3[parts[2]])
            # Attribute 4: (qualitative) Purpose
            attr_4 = {'A40': 0, 'A41': 1, 'A42': 2, 'A43': 3, 'A44': 4, 'A45': 5, 'A46': 6, 'A47': 7, 'A48': 8, 'A49': 9, 'A410': 10}
            data.append(attr_4[parts[3]])
            # Attribute 5: (numerical) # Credit amount
            data.append(numeric_transformer(parts[4]))
            # Attibute 6: (qualitative) Savings account / bonds
            attr_6 = {'A61': 0, 'A62': 1,'A63': 2,'A64': 3,'A65': 4}
            data.append(attr_6[parts[5]])
            # Attribute 7: (qualitative) Present employment since
            attr_7 = {'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4}
            data.append(attr_7[parts[6]])
            # Attribute 8: (numerical) Installment rate in percentage of disposable income
            data.append(numeric_transformer(parts[7]))
            # Attribute 9: (qualitative) Personal status and sex
            attr_9 = {'A91': 0, 'A92': 1, 'A93': 2, 'A94': 3, 'A95': 4}
            data.append(attr_9[parts[8]])
            # Attribute 10: (qualitative) Other debtors / guarantors
            attr_10 = {'A101': 0, 'A102': 1, 'A103': 2}
            data.append(attr_10[parts[9]])
            # Attribute 11: (numerical) Present residence since
            data.append(numeric_transformer(parts[10]))
            # Attribute 12: (qualitative) Property
            attr_12 = {'A121': 0, 'A122': 1, 'A123': 2, 'A124': 3}
            data.append(attr_12[parts[11]])
            # Attribute 13: (numerical) Age in years
            data.append(numeric_transformer(parts[12]))
            # Attribute 14: (qualitative) Other installment plans
            attr_14 = {'A141': 0, 'A142': 1, 'A143': 2}
            data.append(attr_14[parts[13]])
            # Attribute 15: (qualitative) Housing
            attr_15 = {'A151': 0, 'A152': 1, 'A153': 2}
            data.append(attr_15[parts[14]])
            # Attribute 16: (numerical) Number of existing credits at this bank
            data.append(numeric_transformer(parts[15]))
            # Attribute 17: (qualitative) Job
            attr_17 = {'A171': 0, 'A172': 1, 'A173': 2, 'A174': 3}
            data.append(attr_17[parts[16]])
            # Attribute 18: (numerical) Number of people being liable to provide maintenance for
            data.append(numeric_transformer(parts[17]))
            # Attribute 19: (qualitative) # Telephone
            attr_19 = {'A191': 0, 'A192': 1}
            data.append(attr_19[parts[18]])
            # Attribute 20: (qualitative) foreign worker
            attr_20 = {'A201': 0, 'A202': 1}
            data.append(attr_20[parts[19]])
            dataset.append(data)
            # 1 good, 2 bad for credit application
            labels.append(1 if parts[20] == "1" else 0)
        file.close()
        return np.array(dataset, dtype=np.float32), np.array(labels, dtype=np.int32)

    @staticmethod
    def load_credit_with_onehot(file_path):
        dataset = []
        labels = []
        file = open(file_path, "r")
        for line in file:
            line = line.strip("\n\r")
            # to get rid of extra lines that do not contain any information
            if len(line) < 5:
                continue
            parts = line.split(" ")
            data = []
            numeric_transformer = float
            # Attribute 1: (qualitative) - Status of existing checking account
            attr_1 = {'A11': [1,0,0,0], 'A12': [0,1,0,0], 'A13': [0,0,1,0], 'A14': [0,0,0,1]}
            data.extend(attr_1[parts[0]])  #
            # Attribute 2: (numerical) - Duration in month
            data.append(numeric_transformer(parts[1]))
            # Attribute 3: (qualitative) Credit history
            attr_3 = {'A30': [1,0,0,0,0], 'A31': [0,1,0,0,0], 'A32': [0,0,1,0,0], 'A33': [0,0,0,1,0], 'A34': [0,0,0,0,1]}
            data.extend(attr_3[parts[2]])
            # Attribute 4: (qualitative) Purpose
            attr_4 = {'A40': [1,0,0,0,0,0,0,0,0,0,0], 'A41': [0,1,0,0,0,0,0,0,0,0,0], 'A42': [0,0,1,0,0,0,0,0,0,0,0],
                      'A43': [0,0,0,1,0,0,0,0,0,0,0], 'A44': [0,0,0,0,1,0,0,0,0,0,0], 'A45': [0,0,0,0,0,1,0,0,0,0,0],
                      'A46': [0,0,0,0,0,0,1,0,0,0,0], 'A47': [0,0,0,0,0,0,0,1,0,0,0], 'A48': [0,0,0,0,0,0,0,0,1,0,0],
                      'A49': [0,0,0,0,0,0,0,0,0,1,0], 'A410': [0,0,0,0,0,0,0,0,0,0,1]}
            data.extend(attr_4[parts[3]])
            # Attribute 5: (numerical) # Credit amount
            data.append(numeric_transformer(parts[4]))
            # Attibute 6: (qualitative) Savings account / bonds
            attr_6 = {'A61': [1,0,0,0,0], 'A62': [0,1,0,0,0], 'A63': [0,0,1,0,0], 'A64': [0,0,0,1,0], 'A65': [0,0,0,0,1]}
            data.extend(attr_6[parts[5]])
            # Attribute 7: (qualitative) Present employment since
            attr_7 = {'A71': [1,0,0,0,0], 'A72': [0,1,0,0,0], 'A73': [0,0,1,0,0], 'A74': [0,0,0,1,0], 'A75': [0,0,0,0,1]}
            data.extend(attr_7[parts[6]])
            # Attribute 8: (numerical) Installment rate in percentage of disposable income
            data.append(numeric_transformer(parts[7]))
            # Attribute 9: (qualitative) Personal status and sex
            attr_9 = {'A91': [1,0,0,0,0], 'A92': [0,1,0,0,0], 'A93': [0,0,1,0,0], 'A94': [0,0,0,1,0], 'A95': [0,0,0,0,1]}
            data.extend(attr_9[parts[8]])
            # Attribute 10: (qualitative) Other debtors / guarantors
            attr_10 = {'A101': [1,0,0], 'A102': [0,1,0], 'A103': [0,0,1]}
            data.extend(attr_10[parts[9]])
            # Attribute 11: (numerical) Present residence since
            data.append(numeric_transformer(parts[10]))
            # Attribute 12: (qualitative) Property
            attr_12 = {'A121': [1,0,0,0], 'A122': [0,1,0,0], 'A123': [0,0,1,0], 'A124': [0,0,0,1]}
            data.extend(attr_12[parts[11]])
            # Attribute 13: (numerical) Age in years
            data.append(numeric_transformer(parts[12]))
            # Attribute 14: (qualitative) Other installment plans
            attr_14 = {'A141': [1,0,0], 'A142': [0,1,0], 'A143': [0,0,1]}
            data.extend(attr_14[parts[13]])
            # Attribute 15: (qualitative) Housing
            attr_15 = {'A151': [1,0,0], 'A152': [0,1,0], 'A153': [0,0,1]}
            data.extend(attr_15[parts[14]])
            # Attribute 16: (numerical) Number of existing credits at this bank
            data.append(numeric_transformer(parts[15]))
            # Attribute 17: (qualitative) Job
            attr_17 = {'A171': [1,0,0,0], 'A172': [0,1,0,0], 'A173': [0,0,1,0], 'A174': [0,0,0,1]}
            data.extend(attr_17[parts[16]])
            # Attribute 18: (numerical) Number of people being liable to provide maintenance for
            data.append(numeric_transformer(parts[17]))
            # Attribute 19: (qualitative) # Telephone
            attr_19 = {'A191': [1,0], 'A192': [0,1]}
            data.extend(attr_19[parts[18]])
            # Attribute 20: (qualitative) foreign worker
            attr_20 = {'A201': [1,0], 'A202': [0,1]}
            data.extend(attr_20[parts[19]])
            dataset.append(data)
            # 1 good, 2 bad for credit application
            labels.append(1 if parts[20] == "1" else 0)
        file.close()
        return np.array(dataset, dtype=np.float32), np.array(labels, dtype=np.int32)
