import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('train.csv')

print(df.head())

print(df['Survived'].value_counts())

print(df['Pclass'].value_counts())

print(df['Sex'].value_counts())

#describe count, mean, min, max of column age
print(df['Age'].describe())

print(df['SibSp'].value_counts())

print(df['Parch'].value_counts())

#describe count, mean, min, max of column fare
print(df['Fare'].describe())

print(df[df['Cabin'].isnull()]['Survived'].count())
#there are too many null values for Cabin therfore we can remove it in final feature selection

print(df[df['Cabin']=='E25'])

#print solo female passengers
print(df[(df['Parch']==0) & (df['SibSp']==0) & (df['Sex']=='female')]['Survived'].count())

#passengers above 30

print(df[(df['Age'] > 30)]['Survived'].count())

#passengers above 30 and survived
print(df[(df['Age'] > 30) & (df['Survived'] == 1)]['Survived'].count())


print(df['Cabin'].value_counts().head())


#avg fare in each class

print(df.groupby('Pclass')['Fare'].mean())

sns.boxplot(x="Pclass", y="Fare", data=df, palette='rainbow')
#mb.boxplot(x="Pclass", y="Fare")
plt.show()

'''fig, ax =plt.subplots(1,3 , figsize=(10, 6) , sharex='col', sharey='row')
a = sns.countplot(x = 'Sex' , data=df , ax = ax[0] , order=['male' , 'female'])
b = sns.countplot(x = 'Sex' , data= df['Survived' == 1] , ax = ax[1] , order=['male' , 'female'])
c = sns.countplot(x = 'Sex' , data= df[ ((df['Age'] < 21) & ('Survived' == 1)) ] , order=['male' , 'female'])
ax[0].set_title('All passenger')
ax[1].set_title('Survived passenger')
ax[2].set_title('Survived passenger under age 21')'''

sns.distplot(df[df['Fare']<200]['Fare'],bins=50)

plt.show()