## EXNO-3-DS
## Register no:212224230114
## Date:19/04/25

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
~~~
import pandas as pd
df = pd.read_csv('/content/Encoding Data.csv')
df
~~~
![Screenshot 2025-04-19 101239](https://github.com/user-attachments/assets/0a9497b6-0630-4b8a-aacf-2bcf91e9f142)
~~~
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
~~~
![Screenshot 2025-04-19 101435](https://github.com/user-attachments/assets/e0e4d512-3675-4ac7-9901-94eb0bc7067f)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2025-04-19 101623](https://github.com/user-attachments/assets/b2cfad72-ee12-47f9-8947-962e6ea56bb6)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2025-04-19 101704](https://github.com/user-attachments/assets/3ecc6a69-dd28-4c30-bcc3-629bdb3f07ed)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2025-04-19 101748](https://github.com/user-attachments/assets/7b3a6884-a1b4-42b9-ba7e-82167f2580ba)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2025-04-19 101847](https://github.com/user-attachments/assets/ebf63995-64bc-42ed-95ac-178d97265de4)
```
!pip install --upgrade category_encoders
```
![Screenshot 2025-04-19 101943](https://github.com/user-attachments/assets/ee03c061-7186-449d-9021-0255a0761770)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/9a51104e-e05b-4f27-bd65-1f742d4d29ab)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/cc626f7e-d439-486e-a0a4-4dd58d0d519f)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/c212fbae-56a8-40e6-8f20-995848b73d05)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/66bac5a7-dc4c-4a9e-b310-c0af93fbc8eb)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/a600ac30-2e51-4b9f-809d-c325071ac044)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/4ca5dc66-e222-4993-b010-c88fe9e20935)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/c8ef34a7-3b7d-40d9-a526-979821478547)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/a647c0dc-a396-4835-a629-759abee6531f)
```
 df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
 df
```
![image](https://github.com/user-attachments/assets/f1ba374a-0a55-495b-88f5-f9522c6222ae)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/8f570d59-bdd3-4e42-a2de-162f0e4e0767)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/84388881-fd0d-4fd0-95af-60d911f26d6c)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/aa07eab3-89de-46b3-b484-ca3501717af4)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/5f0a437b-e2e5-484e-a53d-0e6dec924da3)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/a1367e62-420a-4789-addf-ab5461df023a)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/09b48a8f-6c47-498b-9457-7cd49a2b716c)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/e89caee3-4893-4f85-9fd6-7abe4809d8d4)
```
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c1aedde8-a3a5-4f38-b6af-78ba8af3f26b)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/11083a8c-a37e-4a13-9b58-b701d8fd850a)


# RESULT:
    Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully
      

       
