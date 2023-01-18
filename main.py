import numpy as np
import math
import pandas as pd
import visual as vi
import functions as fun

# import the pyplot library

import matplotlib.pyplot as plotter
import utile as utl
import seaborn as sns

import aef.AEF as aef
import matplotlib.pyplot as plt
import factor_analyzer as fa
from sklearn.preprocessing import StandardScaler

# read the Project_python_crime.xlsx file
theftData = pd.read_excel('dataIN/Project_python_crime.xlsx', sheet_name='Theft',na_values=':')
print(theftData)

rapeData = pd.read_excel('dataIN/Project_python_crime.xlsx', sheet_name='Rape',na_values=':')
# print(rapeData)

sexualViolenceData = pd.read_excel('dataIN/Project_python_crime.xlsx', sheet_name='Sexual violence',na_values=':')
# print(sexualViolenceData)

# read the Intentional_homicide.csv file
#homicideData= pd.read_csv("dataIN/Intentional_homicide.csv", index_col=0, na_values=':',encoding= 'unicode_escape')
# print(homicideData)

homicideData1=pd.read_excel("dataIN/2011-2016.xlsx",na_values=':')
homicideData2=pd.read_excel("dataIN/2016-2020.xlsx",na_values=':')

homicideData=pd.merge(homicideData1,homicideData2,on=["TIME"])
print(homicideData)

# taking the values from my table
numericalValuesHomicide=homicideData.values
# deleting first column to only have the numerical values
numericalValuesHomicide=np.delete(arr=numericalValuesHomicide, axis=1, obj=0)
print(numericalValuesHomicide)
dfHomicide = pd.DataFrame(numericalValuesHomicide)
# dfHomicide.fillna(0)
dfHomicide = dfHomicide.replace(np.nan, 0)
homicideData['sum']=dfHomicide.sum(axis=1)
print(homicideData)
homicideData.to_excel("dataOUT/HomData.xlsx")

# taking the values from my table
numericalValuesTheft=theftData.values
# deleting first column to only have the numerical values
numericalValuesTheft=np.delete(arr=numericalValuesTheft,axis=1,obj=0)
print(numericalValuesTheft)
dfTheft = pd.DataFrame(numericalValuesTheft)
# dfTheft.fillna(0)
theftData['sum']=dfTheft.sum(axis=1)
print(theftData)
theftData.to_excel("dataOUT/TheftData.xlsx")

# res=pd.merge(homicideData,theftData,on=("sum"))
# res.to_excel("dataOUT/TABLE.xlsx")

df1=pd.DataFrame(homicideData['TIME'])
df1['Sum theft']=theftData['sum'];
df1['Sum homicide']=homicideData['sum']
# df1.rename(columns = {'sum':'Sum homicide'}, inplace = True)
print(df1)
df1.to_excel("dataOUT/see.xlsx")


#dictionary with Belgium sum of theft and homicides in the years 2011-2020
theftBel = df1[df1['TIME'] == 'Belgium']
# print(theftBel)
thisdict = {"Country" : 'Belgium',
            "Theft":theftBel['Sum theft'],
            "Homicide":theftBel['Sum homicide']}
print(thisdict)


# First graph

df=pd.read_excel("dataIN/TheftData2.xlsx")
plt.figure(figsize=(8,5))

#PLOT the number of sexual violence and rape by country in 2020
font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}
plt.title("Sexual violence and rape correlation", fontdict = font1)
plt.xlabel("Sexual violence", fontdict = font2)
plt.ylabel("Rape", fontdict = font2)

plt.scatter(sexualViolenceData['2020'], rapeData['2020'],color = 'hotpink',alpha=0.5, cmap='nipy_spectral')
plt.show()

# 2nd graph : safety index and crime index by country
df2=pd.read_csv("dataIN/indexes.csv")
# df2=df2.filter(regex='Crime Index')
df2=df2.dropna()
df2.head(5)
df2.plot()
# plt.show()

#3rd graph - PIE Belgium types of crimes

# belgium=df[df['Country']=='Belgium']
# print(belgium)
#
# y = np.asarray(belgium.values)
# ynorm=y[1:].flatten()
# # print(type(y))
# print(ynorm)
# print(ynorm.ndim)
valuesY=np.array([120.4, 182, 28090, 80.7, 17.1])
mylabels = np.array(["Violence1", "Homicides", "Robberies", "Domestic burglaries", "Motor vehicle thefts"])
# print(belgium.columns)
# mylabels=belgium.columns[1:]
myl=list(mylabels)
print(mylabels)
# mylabels = np.asarray(belgium.columns)
figureObject, axesObject = plotter.subplots(figsize=(8,8))

# Draw the pie chart
length = [0, 0.5, 0.5, 0.5, 1.2]
axesObject.pie(valuesY, labels=mylabels,  startangle=0,pctdistance=0.6,labeldistance=1.2,explode=length,autopct='%1.1f%%')
plt.show()
# Aspect ratio - equal means pie is a circle

axesObject.axis('equal')
# plt.pie(valuesY,labels=mylabels,autopct='%1.2f', startangle = 90)
# axesObject.axis('equal')
# plt.show()



# 4th graph : Romania Crime Rate & Statistics 1990-2022
df5=pd.read_excel("dataIN/romania_crime_rate.xlsx")
print(df5)
x=np.array(df5['Year'])
print(x)
y=np.array(df5['Per 100K Population'])
print(y)

plt.subplot(1, 2, 1)
plt.plot(x,y)
plt.title("Romania Crime Rate & Statistics 1990-2022", fontdict = font1)
plt.xlabel("Per 100k Population", fontdict = font2)
plt.ylabel("Year", fontdict = font2)
plt.show()




#Exploratory Factor Analysis
tabel = pd.read_csv('dataIN/tipuri_crime_romania.csv', index_col=0, na_values=':', encoding= 'unicode_escape')
numeObs = tabel.index[:]
numeVar = tabel.columns

n = len(numeObs)
m = len(numeVar)

Xbrut = tabel.values
print(Xbrut)

X = fun.inlocuireNAN(Xbrut)
aefModel = aef.AEF(X)
Xstd = aefModel.Xstd

XTab = pd.DataFrame(data=Xstd, index=numeObs, columns=numeVar)
XTab.to_csv('dataOUT/XTab.csv')

sphericityBarlett = fa.calculate_bartlett_sphericity(XTab)
print(sphericityBarlett)

kmo = fa.calculate_kmo(XTab)
print(kmo)

if(sphericityBarlett[0]>sphericityBarlett[1]):
    print("There is at least one common factor")
else:
    print("There are no factors")
    exit(-1)

print(kmo[0])
vector = kmo[0]
matrice = vector[:, np.newaxis]
print(matrice)
kmoTab = pd.DataFrame(data=matrice, index=numeVar, columns=['KMO indices'])

chi2TabMin = 1
noOfSignificantFactors = 2
for q in range(2, m):
    faModel = fa.FactorAnalyzer(n_factors=q)
    faModel.fit(XTab)
    factorLoadings = faModel.loadings_
    specificFactors = faModel.get_uniquenesses()

    chi2Calc, chi2TAb = aefModel.calculTestBartlett(factorLoadings, specificFactors)
    print(chi2Calc, chi2TAb)

    if np.isnan(chi2Calc) or np.isnan(chi2TAb):
        break
    if chi2TabMin > chi2TAb:
        chi2TabMin = chi2TAb
        noOfSignificantFactors = q

print('Number of significant factors extracted:', noOfSignificantFactors)

faFit = fa.FactorAnalyzer(n_factors=noOfSignificantFactors)
faFit.fit(XTab)

factorCorel = faFit.loadings_
print(factorCorel)
factorCorelTab = pd.DataFrame(data=factorCorel, index=numeVar,columns=('F' + str(k+1) for k in range(noOfSignificantFactors)))
print(XTab)
print(factorCorelTab)

vi.correlogram(kmoTab, titlu='Kaiser-Meyer-Olkin indices')
vi.correlogram(factorCorelTab, titlu='Correlogram of correlation factors')
vi.show()

#Communalities
factorCorelTab=np.square(factorCorelTab)
# print(factorCorelTab)
df2 = factorCorelTab.sum(axis=0)
print(df2)