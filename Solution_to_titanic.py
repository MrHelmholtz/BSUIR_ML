import random

import pandas as pd
import numpy as np
from pandas import isnull
import csv

# 0 - passenger id
# 1 - Survived or not
# 2 - person class
# 3 - person name
# 4 - sex
# 5 - age
# 6 - siblings and spouses
# 7 - parents and children
# 8 - ticket
# 9 - fare
# 10 - cabin
# 11 - embarked


data = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\titanic.csv")
test = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\test.csv")
examp = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\gender_submission.csv")
columns_used_in_train = np.array([1, 2, 4, 5, 6, 7, 9, 11])
columns_used_in_test = [a - 1 for a in columns_used_in_train if a > 1]
columns_labels = [data.columns[x] for x in columns_used_in_train]


def view_groups(table):
      groups_by_labels = [data.groupby(label).size() for label in columns_labels]
      print(groups_by_labels)

def view_stats(table):
      stats = table.iloc[:, columns_used_in_train].describe().to_string()
      print(stats)

def get_new_mean_data(column):
      return data.iloc[:, column].mean()

def get_new_median_data(column):
      if column == 11:
            return "S"
      elif column == 4:
            return "male"
      return data.iloc[:, column].median()

def view_similar_people_stats(person):
      if len(person.values) == 11:
            ilabels = columns_used_in_test
      else:
            ilabels = columns_used_in_train[1:]

      pers_class = person.values[ilabels[0]]
      sex = person.values[ilabels[1]]
      if person.values[ilabels[2]] is None:
            age = get_new_mean_data(ilabels[2])
      else:
            age = person.values[ilabels[2]]

      sib_sp = person.values[ilabels[3]]
      par_ch = person.values[ilabels[4]]
      fare = person.values[ilabels[5]]
      if person.values[ilabels[6]] is None:
            emb = get_new_median_data(ilabels[6])
      else:
            emb = person.values[ilabels[6]]

      print(data.iloc[:, columns_used_in_train]
            [(data['Personclass'] >= pers_class - 1) & (data['Personclass'] <= pers_class + 1)]
            [(data['Sex'] == sex)]
            [(data['Age'] >= age - 14.526497) & (data['Age'] <= age + 14.526497)]
            [(data['SibSp'] >= sib_sp - 1.102743) & (data['Parch'] <= sib_sp + 1.102743)]
            [(data['Parch'] >= par_ch - 1) & (data['Parch'] <= par_ch + 1)]
            [(data['Fare'] >= fare - 49.693429) & (data['Fare'] <= fare + 49.693429)]
            [(data['Embarked'] == emb)]
            # .describe()
            .to_string()
            )

def get_probability_of_survival(person):
      if len(person.values) == 11:
            ilabels = columns_used_in_test
      else:
            ilabels = columns_used_in_train[1:]

      pers_class = person.values[ilabels[0]]
      sex = person.values[ilabels[1]]
      if isnull(person.values[ilabels[2]]):
            age = get_new_median_data(ilabels[2]+1)
      else:
            age = person.values[ilabels[2]]
      sib_sp = person.values[ilabels[3]]
      par_ch = person.values[ilabels[4]]
      fare = person.values[ilabels[5]]
      if person.values[ilabels[6]] is None:
            emb = get_new_median_data(ilabels[6]+1)
      else:
            emb = person.values[ilabels[6]]

      chan = (data.iloc[:, 1][(data['Personclass'] >= pers_class -1) & (data['Personclass'] <= pers_class +1)]
                             [(data['Sex'] == sex)]
                             [(data['Age'] >= age - 14.526497) & (data['Age'] <= age + 14.526497)]
                             [(data['SibSp'] >= sib_sp - 1.102743) & (data['Parch'] <= sib_sp + 1.102743)]
                             [(data['Parch'] >= par_ch-2 ) & (data['Parch'] <= par_ch +2)]
                             [(data['Fare'] >= fare - 49.693429) & (data['Fare'] <= fare + 49.693429)]
                             [(data['Embarked'] == emb)]
                             .mean())



      # if isnull(chan):
      #       chan = (data.iloc[:, 1][(data['Personclass'] >= pers_class - 1) & (data['Personclass'] <= pers_class + 1)]
      #               [(data['Sex'] == sex)]
      #               [(data['Age'] >= age - 14.526497) & (data['Age'] <= age + 14.526497)]
      #               [(data['SibSp'] >= sib_sp - 1.102743) & (data['Parch'] <= sib_sp + 1.102743)]
      #               [(data['Parch'] >= par_ch - 1) & (data['Parch'] <= par_ch + 1)]
      #               [(data['Fare'] >= fare - 49.693429) & (data['Fare'] <= fare + 49.693429)]
      #               .mean())
      #       if isnull(chan):
      #             chan = (data.iloc[:, 1][
      #                           (data['Personclass'] >= pers_class - 1) & (data['Personclass'] <= pers_class + 1)]
      #                     [(data['Sex'] == sex)]
      #                     [(data['Age'] >= age - 14.526497) & (data['Age'] <= age + 14.526497)]
      #                     [(data['SibSp'] >= sib_sp - 1.102743) & (data['Parch'] <= sib_sp + 1.102743)]
      #                     [(data['Parch'] >= par_ch - 1) & (data['Parch'] <= par_ch + 1)]
      #                     .mean())
      #             if isnull(chan):
      #                   chan = (data.iloc[:, 1][
      #                                 (data['Personclass'] >= pers_class - 1) & (data['Personclass'] <= pers_class + 1)]
      #                           [(data['Sex'] == sex)]
      #                           [(data['Age'] >= age - 14.526497) & (data['Age'] <= age + 14.526497)]
      #                           [(data['SibSp'] >= sib_sp - 1.102743) & (data['Parch'] <= sib_sp + 1.102743)]
      #                           .mean())
      #                   if isnull(chan):
      #                       chan = (data.iloc[:, 1][
      #                               (data['Personclass'] >= pers_class - 1) & (data['Personclass'] <= pers_class + 1)]
      #                           [(data['Sex'] == sex)]
      #                           [(data['Age'] >= age - 14.526497) & (data['Age'] <= age + 14.526497)]
      #                           .mean())
      #
      #                   return chan
      #             else:
      #                 return chan
      #
      #       else :
      #             return chan
      # # elif chan == 0.5:
      # #       return chan + np.random.normal(0, 0.3)
      # else:
      return chan

def ro(person, data_person):
      diff = 1
      crw = 25
      class_ro = np.abs(person.values[columns_used_in_test[0]] - data_person.values[columns_used_in_train[0+diff]])
      srw = 0
      sex_ro = np.array([person.values[columns_used_in_test[1]] == data_person.values[columns_used_in_train[1+diff]]])
      if(sex_ro == 0): srw = 70
      arw = 2
      age_ro = np.abs(person.values[columns_used_in_test[2]] - data_person.values[columns_used_in_train[2+diff]])
      if(isnull(age_ro)): age_ro = 14.526497
      sbsprw = 3
      sb_sp_ro = np.abs(person.values[columns_used_in_test[3]] - data_person.values[columns_used_in_train[3+diff]])
      parchrw = 3
      par_ch_ro = np.abs(person.values[columns_used_in_test[4]] - data_person.values[columns_used_in_train[4+diff]])
      frw = 0.1
      fare_ro = np.abs(person.values[columns_used_in_test[5]] - data_person.values[columns_used_in_train[5+diff]])
      if (isnull(fare_ro)): fare_ro = 49.693429
      erw = 0
      emb_ro = np.array([person.values[columns_used_in_test[6]] == data_person.values[columns_used_in_train[6+diff]]])
      if(emb_ro == 0): erw = 3
      total_ro = (class_ro * crw + srw + age_ro * arw + sb_sp_ro * sbsprw + par_ch_ro * parchrw + fare_ro * frw + erw)
      return total_ro

def K_gauss(r):
      return 1/(np.sqrt(np.pi * 2)) * np.exp(-1 *(r ** 2)/2)

def K_triang(r):
      return  np.abs( 1 - r) * np.array([np.abs(r) <= 1])

def K_rectan(r):
      return 1/2 * np.array([np.abs(r) <= 1])

def K_Epanch(r):
      return 3/4 * (1 - r ** 2) * np.array([np.abs(r) <= 1])

def K_quadratic(r):
      return 15/16 * (1 - r ** 2) ** 2 * np.array([np.abs(r) <= 1])
# pers_class = 3
# sex = 'male'
# age = 40
# sib_sp = 0
# par_ch = 0
# fare = 30
# emb = 'S'

# print(data.iloc[:,[1,2,4,5,6,7,9,11 ]]
# data.groupby().size()

# print((test.iloc[24]))
# print(get_probability_of_survival(test.iloc[24]))
# print(view_similar_people_stats(test.iloc[24]))

# [177, 194, 306, 326, 392]
# [24, 64, 69, 81, 114, 132, 139, 142, 152, 174, 178, 188, 196, 202, 214, 342, 343, 360, 365, 407]
# print(test.isnull().sum())
# print(get_new_median_data(9))
# print(get_new_mean_data(9))


# print(data.columns[columns_used_in_train])
# print(test.columns[columns_used_in_test])
# view_groups(data)
# print(len(test))
# print(columns_used_in_test)

res = []
unknown = []
nones = []
# print(data.isnull().sum() )
# print(test.isnull().sum() )
# print(test.iloc[69, :].values[columns_used_in_test])
h = 45

for x in range(len(test)):
      chances_death = np.sum([K_Epanch(ro(test.iloc[x,:], data.iloc[i, :])/h) for i in range(len(data))
                              if data.iloc[i, 1] == 0])
      chances_survival = np.sum([K_Epanch(ro(test.iloc[x,:], data.iloc[i, :])/h) for i in range(len(data))
                              if data.iloc[i, 1] == 1])
      chances = (chances_survival - chances_death)
      if chances == 0:
            unknown.append(x)
      elif isnull(chances):
            nones.append(x)
      # res.append([test.iloc[x,0], 1]) if chances > 0.5 else res.append([test.iloc[x,0], 0]) if chances < 0.5 else res.append(-1.0)
      # if chances == 0.5 or isnull(chances):
      #       chances = random.gauss(0.5,0.1)
      # if chances == 0.5:
      #       unknown.append(x)
      res.append([test.iloc[x,0], 1]) if chances > 0 else res.append([test.iloc[x,0], 0]) if chances < 0\
            else res.append([test.iloc[x,0], 0])
      # res.append(1) if chances > 0.5 else res.append( 0) if chances < 0.5 else res.append(-1.0)

print((unknown))
print((nones))
# print(sum([res[i] == -1 for i in range(len(res))]))
print(res)
# print(view_similar_people_stats(test.iloc[177,:]))
# print(view_similar_people_stats(test.iloc[194,:]))
# res = np.array(res)
# answer = pd.DataFrame(res, columns=['PassengerId', 'Survived'])
# answer = answer.to_csv()

# answer.to_csv("C:\\Users\\Lenovo\\Desktop")
# print(examp)
# print(answer)

answer = [['PassengerId', 'Survived']] + res
file_path = "C:\\Users\\Lenovo\\PycharmProjects\\PythonProject\\BSUIR\\BSUIR_ML\\answer.csv"

f = open(file_path, "w", newline="")
writer = csv.writer(f)
for row in answer:
      writer.writerow(row)
f.close()
