import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


dataframe=pd.read_csv("CocaCola_Sales_Rawdata.csv")

dataframe['quarterstrim']=0
for i in range(42):
     p=    dataframe["Quarter"][i]
     print(p)
     dataframe["quarterstrim"][i]   =p[0:2]

print(dataframe["quarterstrim"][i])

afterdummy=pd.get_dummies(dataframe['quarterstrim'])
print(afterdummy)
dataframe=pd.concat([dataframe,afterdummy],axis=1)
print(dataframe)


dataframe["t"] = np.arange(1,43)
print(dataframe)
dataframe["t_sqaure"]=dataframe["t"]*dataframe["t"]
dataframe["log_sales"]=np.log(dataframe["Sales"])

dataframe.Sales.plot()
plt.show()

Train = dataframe.head(30)
Test = dataframe.tail(12)

import statsmodels.formula.api as smf

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
print(rmse_linear)


