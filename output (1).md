#INX Future Inc Employee Performance for HR analysis

##I.Import libraries


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('darkgrid')  # setting up background
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,classification_report,confusion_matrix
# TO avoid warnings
import warnings
warnings.filterwarnings('ignore')
```

##II.Data import


```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Load the dataset
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "eshwarganta/employee-performance-analysis-inx-future-inc",
    "INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls",
    pandas_kwargs={"sheet_name": "INX_Future_Inc_Employee_Perform"}
)

print("First 5 records:")
df.head()

```

    Downloading from https://www.kaggle.com/api/v1/datasets/download/eshwarganta/employee-performance-analysis-inx-future-inc?dataset_version_number=1&file_name=INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls...


    100%|██████████| 401k/401k [00:00<00:00, 1.90MB/s]


    First 5 records:






  <div id="df-fb48e35b-5a39-4cbb-8262-c8ca62e1c6ee" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EmpNumber</th>
      <th>Age</th>
      <th>Gender</th>
      <th>EducationBackground</th>
      <th>MaritalStatus</th>
      <th>EmpDepartment</th>
      <th>EmpJobRole</th>
      <th>BusinessTravelFrequency</th>
      <th>DistanceFromHome</th>
      <th>EmpEducationLevel</th>
      <th>...</th>
      <th>EmpRelationshipSatisfaction</th>
      <th>TotalWorkExperienceInYears</th>
      <th>TrainingTimesLastYear</th>
      <th>EmpWorkLifeBalance</th>
      <th>ExperienceYearsAtThisCompany</th>
      <th>ExperienceYearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
      <th>Attrition</th>
      <th>PerformanceRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E1001000</td>
      <td>32</td>
      <td>Male</td>
      <td>Marketing</td>
      <td>Single</td>
      <td>Sales</td>
      <td>Sales Executive</td>
      <td>Travel_Rarely</td>
      <td>10</td>
      <td>3</td>
      <td>...</td>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>2</td>
      <td>10</td>
      <td>7</td>
      <td>0</td>
      <td>8</td>
      <td>No</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E1001006</td>
      <td>47</td>
      <td>Male</td>
      <td>Marketing</td>
      <td>Single</td>
      <td>Sales</td>
      <td>Sales Executive</td>
      <td>Travel_Rarely</td>
      <td>14</td>
      <td>4</td>
      <td>...</td>
      <td>4</td>
      <td>20</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
      <td>No</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E1001007</td>
      <td>40</td>
      <td>Male</td>
      <td>Life Sciences</td>
      <td>Married</td>
      <td>Sales</td>
      <td>Sales Executive</td>
      <td>Travel_Frequently</td>
      <td>5</td>
      <td>4</td>
      <td>...</td>
      <td>3</td>
      <td>20</td>
      <td>2</td>
      <td>3</td>
      <td>18</td>
      <td>13</td>
      <td>1</td>
      <td>12</td>
      <td>No</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E1001009</td>
      <td>41</td>
      <td>Male</td>
      <td>Human Resources</td>
      <td>Divorced</td>
      <td>Human Resources</td>
      <td>Manager</td>
      <td>Travel_Rarely</td>
      <td>10</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>23</td>
      <td>2</td>
      <td>2</td>
      <td>21</td>
      <td>6</td>
      <td>12</td>
      <td>6</td>
      <td>No</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E1001010</td>
      <td>60</td>
      <td>Male</td>
      <td>Marketing</td>
      <td>Single</td>
      <td>Sales</td>
      <td>Sales Executive</td>
      <td>Travel_Rarely</td>
      <td>16</td>
      <td>4</td>
      <td>...</td>
      <td>4</td>
      <td>10</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>No</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-fb48e35b-5a39-4cbb-8262-c8ca62e1c6ee')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-fb48e35b-5a39-4cbb-8262-c8ca62e1c6ee button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-fb48e35b-5a39-4cbb-8262-c8ca62e1c6ee');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-f578d28b-dd1c-404c-b776-74748f277c73">
      <button class="colab-df-quickchart" onclick="quickchart('df-f578d28b-dd1c-404c-b776-74748f277c73')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-f578d28b-dd1c-404c-b776-74748f277c73 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




##III.EDA


```python
df.shape
```




    (1200, 28)



-The data collected included 1200 employee’s performance appraisal records, described by 28 parameters.


```python
df.columns
```




    Index(['EmpNumber', 'Age', 'Gender', 'EducationBackground', 'MaritalStatus',
           'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency',
           'DistanceFromHome', 'EmpEducationLevel', 'EmpEnvironmentSatisfaction',
           'EmpHourlyRate', 'EmpJobInvolvement', 'EmpJobLevel',
           'EmpJobSatisfaction', 'NumCompaniesWorked', 'OverTime',
           'EmpLastSalaryHikePercent', 'EmpRelationshipSatisfaction',
           'TotalWorkExperienceInYears', 'TrainingTimesLastYear',
           'EmpWorkLifeBalance', 'ExperienceYearsAtThisCompany',
           'ExperienceYearsInCurrentRole', 'YearsSinceLastPromotion',
           'YearsWithCurrManager', 'Attrition', 'PerformanceRating'],
          dtype='object')



-Object describes:

1. **EmpNumber**: Unique employee ID.  
2. **Age**: Employee's age in years.  
3. **Gender**: Employee's gender [Male/Female].  
4. **EducationBackground**: High school or post-secondary degree.  
5. **MaritalStatus**: Marital or civil status.  
6. **EmpDepartment**: Employee's department.  
7. **EmpJobRole**: Main job role/responsibility.  
8. **BusinessTravelFrequency**: Frequency of business travel.  
9. **DistanceFromHome**: Distance from home to office.  
10. **EmpEducationLevel**: Education level (e.g., Diploma, Degree, Master’s).  
11. **EmpEnvironmentSatisfaction**: Satisfaction with work environment.  
12. **EmpHourlyRate**: Pay rate per hour.  
13. **EmpJobInvolvement**: Level of job involvement.  
14. **EmpJobLevel**: Job grade or level.  
15. **EmpJobSatisfaction**: Job satisfaction level.  
16. **NumCompaniesWorked**: Number of previous companies worked at.  
17. **OverTime**: Works overtime or not [Yes/No].  
18. **EmpLastSalaryHikePercent**: Last year’s salary increase percentage.  
19. **EmpRelationshipSatisfaction**: Satisfaction with workplace relationships.  
20. **TotalWorkExperienceInYears**: Total years of work experience.  
21. **TrainingTimesLastYear**: Number of trainings last year.  
22. **EmpWorkLifeBalance**: Balance between work and personal life.  
23. **ExperienceYearsAtThisCompany**: Years at current company.  
24. **ExperienceYearsInCurrentRole**: Years in current job role.  
25. **YearsSinceLastPromotion**: Years since last promotion.  
26. **YearsWithCurrManager**: Years with current manager.  
27. **Attrition**: Whether the employee left the company.  
28. **PerformanceRating**: Overall performance rating.
-



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1200 entries, 0 to 1199
    Data columns (total 28 columns):
     #   Column                        Non-Null Count  Dtype 
    ---  ------                        --------------  ----- 
     0   EmpNumber                     1200 non-null   object
     1   Age                           1200 non-null   int64 
     2   Gender                        1200 non-null   object
     3   EducationBackground           1200 non-null   object
     4   MaritalStatus                 1200 non-null   object
     5   EmpDepartment                 1200 non-null   object
     6   EmpJobRole                    1200 non-null   object
     7   BusinessTravelFrequency       1200 non-null   object
     8   DistanceFromHome              1200 non-null   int64 
     9   EmpEducationLevel             1200 non-null   int64 
     10  EmpEnvironmentSatisfaction    1200 non-null   int64 
     11  EmpHourlyRate                 1200 non-null   int64 
     12  EmpJobInvolvement             1200 non-null   int64 
     13  EmpJobLevel                   1200 non-null   int64 
     14  EmpJobSatisfaction            1200 non-null   int64 
     15  NumCompaniesWorked            1200 non-null   int64 
     16  OverTime                      1200 non-null   object
     17  EmpLastSalaryHikePercent      1200 non-null   int64 
     18  EmpRelationshipSatisfaction   1200 non-null   int64 
     19  TotalWorkExperienceInYears    1200 non-null   int64 
     20  TrainingTimesLastYear         1200 non-null   int64 
     21  EmpWorkLifeBalance            1200 non-null   int64 
     22  ExperienceYearsAtThisCompany  1200 non-null   int64 
     23  ExperienceYearsInCurrentRole  1200 non-null   int64 
     24  YearsSinceLastPromotion       1200 non-null   int64 
     25  YearsWithCurrManager          1200 non-null   int64 
     26  Attrition                     1200 non-null   object
     27  PerformanceRating             1200 non-null   int64 
    dtypes: int64(19), object(9)
    memory usage: 262.6+ KB



```python
df.describe().T
```





  <div id="df-7839a00a-8085-4a79-9b5f-0109d7dc02a1" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1200.0</td>
      <td>36.918333</td>
      <td>9.087289</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>36.0</td>
      <td>43.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>DistanceFromHome</th>
      <td>1200.0</td>
      <td>9.165833</td>
      <td>8.176636</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>14.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>EmpEducationLevel</th>
      <td>1200.0</td>
      <td>2.892500</td>
      <td>1.044120</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>EmpEnvironmentSatisfaction</th>
      <td>1200.0</td>
      <td>2.715833</td>
      <td>1.090599</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>EmpHourlyRate</th>
      <td>1200.0</td>
      <td>65.981667</td>
      <td>20.211302</td>
      <td>30.0</td>
      <td>48.0</td>
      <td>66.0</td>
      <td>83.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>EmpJobInvolvement</th>
      <td>1200.0</td>
      <td>2.731667</td>
      <td>0.707164</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>EmpJobLevel</th>
      <td>1200.0</td>
      <td>2.067500</td>
      <td>1.107836</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>EmpJobSatisfaction</th>
      <td>1200.0</td>
      <td>2.732500</td>
      <td>1.100888</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>NumCompaniesWorked</th>
      <td>1200.0</td>
      <td>2.665000</td>
      <td>2.469384</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>EmpLastSalaryHikePercent</th>
      <td>1200.0</td>
      <td>15.222500</td>
      <td>3.625918</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>14.0</td>
      <td>18.0</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>EmpRelationshipSatisfaction</th>
      <td>1200.0</td>
      <td>2.725000</td>
      <td>1.075642</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>TotalWorkExperienceInYears</th>
      <td>1200.0</td>
      <td>11.330000</td>
      <td>7.797228</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>TrainingTimesLastYear</th>
      <td>1200.0</td>
      <td>2.785833</td>
      <td>1.263446</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>EmpWorkLifeBalance</th>
      <td>1200.0</td>
      <td>2.744167</td>
      <td>0.699374</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>ExperienceYearsAtThisCompany</th>
      <td>1200.0</td>
      <td>7.077500</td>
      <td>6.236899</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>ExperienceYearsInCurrentRole</th>
      <td>1200.0</td>
      <td>4.291667</td>
      <td>3.613744</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>YearsSinceLastPromotion</th>
      <td>1200.0</td>
      <td>2.194167</td>
      <td>3.221560</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>YearsWithCurrManager</th>
      <td>1200.0</td>
      <td>4.105000</td>
      <td>3.541576</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>PerformanceRating</th>
      <td>1200.0</td>
      <td>2.948333</td>
      <td>0.518866</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7839a00a-8085-4a79-9b5f-0109d7dc02a1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7839a00a-8085-4a79-9b5f-0109d7dc02a1 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7839a00a-8085-4a79-9b5f-0109d7dc02a1');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-fb085de1-ddb3-45f5-99ac-1d3fd6c3cd8a">
  <button class="colab-df-quickchart" onclick="quickchart('df-fb085de1-ddb3-45f5-99ac-1d3fd6c3cd8a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-fb085de1-ddb3-45f5-99ac-1d3fd6c3cd8a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




##IV.Basic analysis

###1.Age


```python
#Vẽ biểu đồ histplot với x=Age, màu xanh lá đậm
plt.figure(figsize=(10,5))
sns.histplot(df['Age'], palette='Dark2')
plt.title('Age distribution')
plt.axvline(df['Age'].mean(), color='green', linestyle='dashed', linewidth=2)
plt.show()
```


    
![png](output_14_0.png)
    



```python
def categorize_age(age):
    if age < 28:
        return '<28'
    elif 28 <= age <= 40:
        return '28-40'
    else:
        return '>40'
df_age = df.copy()
df_age['Age_group'] = df_age['Age'].apply(categorize_age)
age_group_counts = df_age['Age_group'].value_counts().sort_index()

```


```python
plt.figure(figsize=(3,3))
plt.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Age distribute')
plt.show()

```


    
![png](output_16_0.png)
    


- Age distribute range is between 18 to 60
- Most of the employee age is between 28  to 40 (~53%)



###2.Employee Hourly Rate


```python
#Vẽ biểu đồ histplot với x=Age, màu xanh lá đậm
plt.figure(figsize=(10,5))
sns.histplot(df['EmpHourlyRate'])
plt.title('EmpHourlyRate distribution')
plt.axvline(df['EmpHourlyRate'].mean(), color='green', linestyle='dashed', linewidth=2)
plt.show()
```


    
![png](output_19_0.png)
    


- Employee hourly rate distribute range is between \$30 to \$100
- Most of the employee hourly rate is between \$45

### 3.Year Experience


```python
#Vẽ biểu đồ histplot với x=Age, màu xanh lá đậm
plt.figure(figsize=(10,5))
sns.histplot(df['TotalWorkExperienceInYears'])
plt.title('TotalWorkExperienceInYears distribution')
plt.axvline(df['TotalWorkExperienceInYears'].mean(), color='green', linestyle='dashed', linewidth=2)
plt.show()
```


    
![png](output_22_0.png)
    


- The majority of the compay is in the range of less than 10 years of experience.

- The number of people with experience from 0 to about 12 years is very high, then gradually decreases.

**Most common group (~11 years)**:

- There is a prominent column around **11 years of experience** — this is the **highest point (mode)**, which shows the group with the highest number of employees.

**There are many outliers with high experience (>30 years)**:

- Some people have up to **35–40 years of experience**, but very few — indicating that these are special cases (outliers).


###4.Experience Years At This Company


```python
#Vẽ biểu đồ histplot với x=Age, màu xanh lá đậm
plt.figure(figsize=(10,5))
sns.histplot(df['ExperienceYearsAtThisCompany'])
plt.title('ExperienceYearsAtThisCompany distribution')
plt.axvline(df['ExperienceYearsAtThisCompany'].mean(), color='green', linestyle='dashed', linewidth=2)
plt.show()
```


    
![png](output_25_0.png)
    


- Most employees have **less than 10 years** working at the company.

- In particular, the **highest column** is in the range of **4–5 years**, indicating that this is the most common number of years of work.

**There are many new employees (0–3 years)**:

- A large number of employees have only worked from **0 to 3 years** ⇒ The company may be in the expansion phase

**A small number of long-term employees**:
- There is still a small number of employees working >20 years, even 30–40 years ⇒ They may be senior, veteran employees.

**Green dashed line**:
- It may be the average number of years working at the company, around **7 years**.

- This average is skewed up due to some outliers with very long working years.

**PLOTS MULTIPLE FEATURE**


```python
count = df[['Gender', 'EducationBackground', 'MaritalStatus','BusinessTravelFrequency','DistanceFromHome',
              'EmpEducationLevel', 'EmpEnvironmentSatisfaction','EmpJobInvolvement', 'EmpJobLevel',
              'EmpJobSatisfaction', 'NumCompaniesWorked', 'OverTime']]

plt.figure(figsize=(20, 25))
plotno = 1

for column in count:
    if plotno <= 13:
        plt.subplot(3, 4, plotno)

        # Lấy số lượng các giá trị duy nhất trong cột để tạo bảng màu tương ứng
        unique_vals = count[column].nunique()
        colors = sns.color_palette("Dark2", unique_vals)  # HSV để có màu đa dạng

        sns.countplot(x=count[column], palette=colors)
        plt.xlabel(column, fontsize=10)
        plt.xticks(rotation=30)  # Xoay nhãn cho dễ đọc nếu cần
    plotno += 1

plt.tight_layout()
plt.show()
```


    
![png](output_28_0.png)
    


###5.Gender

- The number of **male** employees is higher than that of female employees.
- The workforce is predominantly male.

###6.EducationBackground

- The most common backgrounds are **Life Scienes** and **Medical**.
- **Human Resources** has the fewest employees.
- This suggests the company has a focus on Life Scienes and Medical roles.

###7.MaritalStatus

- Most employees are **Married**, followed by **Single**, with the fewest being **Divorced**.
- This may reflect a more mature or stable workforce demographic.

###8.BusinessTravelFrequency

- The majority of employees **Travel Rarely** for work.
- A small number travel **Frequently**, and very few do **Not Travel** at all.
- This indicates a moderate level of business travel in most roles.

###9.DistanceFromHome

- Distance varies widely, but the most common range is between **1 and 5 km**.
- The distribution is quite spread out with no sharp peak.
- The commute distance may not be a major issue for most employees.

###10.EmpEducationLevel

- **Most employees hold a Bachelor's (3) or Master's (4) degree.**
- Very few have only **Below College (1)** or the highest level **Doctorate (5)**.
- This reflects a well-educated workforce with a strong academic background.

###11.EmpEnvironmentSatisfaction

- The majority of employees report **High (3)** or **Very High (4)** satisfaction.
- Few report **Low (1)** or **Medium (2)**.
- This indicates a generally positive work environment.

###12.EmpJobInvolvement

- Most employees rate their involvement as **High (3)** or **Very High (4)**.
- Few report **Low (1)**.
- This suggests employees are actively engaged in their work.

###13.EmpJobLevel

- Most employees are at **Level 1 or 2**, suggesting many are in entry or mid-level roles.
- Higher levels (4 and 5) have far fewer people — a typical organizational hierarchy.

###14.EmpJobSatisfaction

- Job satisfaction is predominantly **High (3)** and **Very High (4)**.
- Very few employees report **Low (1)** satisfaction.
- This implies that most employees are content with their jobs.

###15.NumCompaniesWorked

- The most common number is **1**, meaning many employees are in their **first job**.
- Less no of employee work in more than 5 companies
- The number gradually declines with more companies worked.
- This may reflect employee loyalty or a younger workforce.

###16.OverTime

Most no of employee on doing over time and less than 350 employee doing overtime in company.


```python
count = df[['EmpLastSalaryHikePercent', 'EmpRelationshipSatisfaction','TrainingTimesLastYear','EmpWorkLifeBalance',
               'ExperienceYearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager', 'Attrition',
               'PerformanceRating','EmpDepartment']]

plt.figure(figsize=(25, 20))
plotno = 1

for column in count:
    if plotno <= 13:
        plt.subplot(5, 5, plotno)

        # Lấy số lượng các giá trị duy nhất trong cột để tạo bảng màu tương ứng
        unique_vals = count[column].nunique()
        colors = sns.color_palette("Dark2", unique_vals)  # HSV để có màu đa dạng

        sns.countplot(x=count[column], palette=colors)
        plt.xlabel(column, fontsize=15)
        plt.xticks(rotation=45)  # Xoay nhãn cho dễ đọc nếu cần
    plotno += 1

plt.tight_layout()
plt.show()
```


    
![png](output_53_0.png)
    


###17.EmpLastSalaryHikePercent

- Most common salary hikes are in the **11–14%** range.
- The frequency decreases steadily beyond 15%, with very few receiving **22% or more**.
- Suggests that most salary increments are conservative.

###18.EmpRelationshipSatisfaction

*(Mapped: 1 – Low, 2 – Medium, 3 – High, 4 – Very High)*
- Majority of employees report **High (3)** or **Very High (4)** relationship satisfaction.
- Very few are in the **Low (1)** category.
- Indicates healthy workplace relationships overall.

###19.TrainingTimesLastYear

- Most employees received training **2 or 3 times** last year.
- Very few had **no training**, and a small group received **4–6 training sessions**.
- The organization invests moderately in upskilling.

###20. EmpWorkLifeBalance  


*(Mapped: 1 – Bad, 2 – Good, 3 – Better, 4 – Best)*
- Majority rate their work-life balance as **Better (3)**.
- A smaller but significant group rates it as **Good (2)**.
- Few rate it as **Bad (1)** or **Best (4)**.
- Indicates room for improvement but generally positive perceptions.

###21.ExperienceYearsInCurrentRole

- Most employees have **2–4 years** of experience in their current role.
- Very few have more than **10 years**.
- Suggests either role rotation, promotions, or turnover after a few years.

###22.YearsSinceLastPromotion

- Most employees were promoted **within the last 0–1 year**.
- Promotion frequency drops drastically after 2 years.
- This could reflect either a recent promotion wave or fast-track career progression.

###23.YearsWithCurrManager

- Most employees have worked with their current manager for **2 years**, followed by **0–1 years**.
- Relationships lasting **8 years or more** are rare.
- May indicate frequent reorganization or management shifts.

###24.Attrition

- Majority of employees are still with the company (**No Attrition**).
- A smaller segment has left (**Yes**).
- This hints at a relatively **low attrition rate**, though deeper analysis may reveal influencing factors.

###25.PerformanceRating

*(Mapped: 1 – Low, 2 – Good, 3 – Excellent, 4 – Outstanding)*
- Most employees are rated **Excellent (3)**.
- Some receive **Good (2)**, and fewer are **Outstanding (4)**.
- This skew could suggest inflated ratings or a high-performing workforce.

###26.EmpDepartment

- The largest departments are **Sales**, **Development**, and **Research & Development**.
- **Human Resources**, **Finance**, and especially **Data Science** are much smaller.
- Reflects organizational focus on technical and customer-facing functions.

###27.EmpJobRole


```python
#Vẽ biểu đồ histplot với x=Age, màu xanh lá đậm
plt.figure(figsize=(10,5))
sns.histplot(df['EmpJobRole'])
plt.title('EmpJobRole distribution')
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_75_0.png)
    


- **Top Roles**:
  - **Sales Executive** and **Developer** are by far the most common roles in the company.
    - Sales Executive has the **highest count**, suggesting a sales-driven business model.
    - Developers follow closely, indicating a strong tech or product development team.
  
- **Mid-Level Representation**:
  - Roles like **Manager**, **Research Scientist**, **Laboratory Technician**, and **Manager R&D** are moderately represented.
  - Suggests a balanced organizational structure with reasonable presence in both managerial and R&D functions.

- **Specialized Roles (Lower Count)**:
  - Positions such as **Data Scientist**, **Senior Developer**, **Technical Architect**, and **Business Analyst** appear in **smaller numbers**.
  - These could be **niche roles** or relatively **newer functions** within the company.

- **Least Common Roles**:
  - **Healthcare Representative**, **Delivery Manager**, and **Research Director** have the **lowest counts**.
  - This might reflect either a low demand or specific needs fulfilled by a small team.

- **Insights**:
  - The high number of employees in executional and technical roles aligns with earlier findings (e.g., dominance of Sales and Development departments).
  - The role distribution shows that while the company is **execution-heavy**, it still maintains **strategic and technical leadership** with various managerial and specialist roles.

##V.2-variable analysis

Focus on:\
1.Relation Between age & Experience Years At This Company\
2.Relation Between experiance year at this company & total work experiance\
3.Relation between Employee last salary hike and number of company worked\
4.Relation between Years Since Last Promotion and Experience Years In CurrentRole\
5.Relation between Employee Hourly Rate and Years With Current Manager\
6.Relation between Distance From Home and Employee Last Salary Hike Percent


```python

plt.figure(figsize=(18, 15))  # Set overall figure size

# 1. Age vs TotalWorkExperienceInYears
plt.subplot(3, 3, 1)
sns.lineplot(x='Age', y='TotalWorkExperienceInYears', data=df)
plt.title('Age vs Total Experience')
plt.xlabel('Age')
plt.ylabel('Total Experience')

# 2. ExperienceYearsAtThisCompany vs TotalWorkExperienceInYears
plt.subplot(3, 3, 2)
sns.lineplot(x='ExperienceYearsAtThisCompany', y='TotalWorkExperienceInYears', data=df)
plt.title('Company Exp vs Total Exp')
plt.xlabel('Experience at Company')
plt.ylabel('Total Experience')

# 3. EmpLastSalaryHikePercent vs NumCompaniesWorked
plt.subplot(3, 3, 3)
sns.lineplot(x='EmpLastSalaryHikePercent', y='NumCompaniesWorked', data=df)
plt.title('Salary Hike vs Companies Worked')
plt.xlabel('Salary Hike (%)')
plt.ylabel('Companies Worked')

# 4. YearsSinceLastPromotion vs ExperienceYearsInCurrentRole
plt.subplot(3, 3, 4)
sns.lineplot(x='YearsSinceLastPromotion', y='ExperienceYearsInCurrentRole', data=df)
plt.title('Years Since Promotion vs Current Role Exp')
plt.xlabel('Years Since Last Promotion')
plt.ylabel('Experience in Current Role')

# 5. EmpHourlyRate vs YearsWithCurrManager
plt.subplot(3, 3, 5)
sns.lineplot(x='EmpHourlyRate', y='YearsWithCurrManager', data=df)
plt.title('Hourly Rate vs Manager Tenure')
plt.xlabel('Hourly Rate')
plt.ylabel('Years with Current Manager')

# 6. DistanceFromHome vs EmpLastSalaryHikePercent
plt.subplot(3, 3, 6)
sns.lineplot(x='DistanceFromHome', y='EmpLastSalaryHikePercent', data=df)
plt.title('Distance from Home vs Salary Hike')
plt.xlabel('Distance From Home')
plt.ylabel('Salary Hike (%)')

plt.tight_layout()
plt.show()

```


    
![png](output_79_0.png)
    



```python
plt.figure(figsize=(18, 15))  # Tổng kích thước

# 1. Age vs TotalWorkExperienceInYears - Blue
plt.subplot(3, 3, 1)
sns.lineplot(x='Age', y='TotalWorkExperienceInYears', data=df, color='royalblue')
plt.title('Age vs Total Experience')
plt.xlabel('Age')
plt.ylabel('Total Experience')

# 2. ExperienceYearsAtThisCompany vs TotalWorkExperienceInYears - Green
plt.subplot(3, 3, 2)
sns.lineplot(x='ExperienceYearsAtThisCompany', y='TotalWorkExperienceInYears', data=df, color='seagreen')
plt.title('Company Exp vs Total Exp')
plt.xlabel('Experience at Company')
plt.ylabel('Total Experience')

# 3. EmpLastSalaryHikePercent vs NumCompaniesWorked - Orange
plt.subplot(3, 3, 3)
sns.lineplot(x='EmpLastSalaryHikePercent', y='NumCompaniesWorked', data=df, color='darkorange')
plt.title('Salary Hike vs Companies Worked')
plt.xlabel('Salary Hike (%)')
plt.ylabel('Companies Worked')

# 4. YearsSinceLastPromotion vs ExperienceYearsInCurrentRole - Purple
plt.subplot(3, 3, 4)
sns.lineplot(x='YearsSinceLastPromotion', y='ExperienceYearsInCurrentRole', data=df, color='mediumpurple')
plt.title('Years Since Promotion vs Current Role Exp')
plt.xlabel('Years Since Last Promotion')
plt.ylabel('Current Role Experience')

# 5. EmpHourlyRate vs YearsWithCurrManager - Red
plt.subplot(3, 3, 5)
sns.lineplot(x='EmpHourlyRate', y='YearsWithCurrManager', data=df, color='firebrick')
plt.title('Hourly Rate vs Manager Tenure')
plt.xlabel('Hourly Rate')
plt.ylabel('Years with Current Manager')

# 6. DistanceFromHome vs EmpLastSalaryHikePercent - Teal
plt.subplot(3, 3, 6)
sns.lineplot(x='DistanceFromHome', y='EmpLastSalaryHikePercent', data=df, color='teal')
plt.title('Distance from Home vs Salary Hike')
plt.xlabel('Distance From Home')
plt.ylabel('Salary Hike (%)')

plt.tight_layout()
plt.show()

```


    
![png](output_80_0.png)
    


###**1. Age vs Total Work Experience**

- There is a clear **linear relationship**: the older the employee, the more total years of experience they have.
- Most employees aged <25 years have about 5 years of experience in this company

###**2. Experience at Company vs Total Work Experience**

- Total work experience **increases** with years at the current company, but not all long-tenured employees have high total experience.
- There’s a slight divergence: some employees have high total experience but **shorter tenure** at the current company → likely due to **job-hopping** in the past.
- This reflects a diverse career background among the workforce.

###**3. Salary Hike % vs Number of Companies Worked**

- The chart shows strong fluctuations and **no clear trend** between last salary hike % and the number of companies worked for.
- This suggests that **salary increments** might not be influenced by how many companies an employee has worked at.
- Interestingly, employees with the **highest salary hikes** seem to have worked at **fewer companies**.

###**4. Years Since Last Promotion vs Experience In Current Role**

- There is a **mild positive correlation**: the longer it's been since a promotion, the longer the employee has been in their current role.
- This is intuitive — employees who haven't been promoted recently likely **stay in the same position**.
- Still, there are some large fluctuations, possibly due to **varying promotion policies** across departments.

###**5. Hourly Rate vs Years With Current Manager**

- The chart is **highly scattered**, showing no strong correlation.
- This suggests that **hourly rate** isn't closely tied to the **tenure with the current manager**.
- It may depend more on technical skills, job role, or other compensation structures.

###**6. Distance From Home vs Last Salary Hike %**

- There’s a slight upward trend: employees who **live farther from work** might be receiving **higher salary hikes**.
- This could be due to the company offering **incentives or compensation** to retain employees who commute long distances.
- However, the wide spread of data suggests this relationship is **not very strong** and could vary depending on job type, location, etc.

##VI.ANALYSIS ON CATEGORICAL FEATURE


```python
categorical = df[['Gender', 'EducationBackground', 'MaritalStatus', 'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition']]

plt.figure(figsize=(25, 25))
plotno = 1

for column in categorical:
    if plotno <= 13:
        plt.subplot(4, 4, plotno)

        # Lấy số lượng các giá trị duy nhất trong cột để tạo bảng màu tương ứng
        unique_vals = categorical[column].nunique()
        colors = sns.color_palette("Dark2", unique_vals)  # HSV để có màu đa dạng

        sns.countplot(x=categorical[column], hue=df.PerformanceRating, palette=colors)
        plt.xlabel(column, fontsize=15)
        plt.ylabel('PerformanceRating')
        plt.xticks(rotation=90)  # Xoay nhãn cho dễ đọc nếu cần
    plotno += 1

plt.tight_layout()
plt.show()
```


    
![png](output_88_0.png)
    




##VII.ANALYSIS ON DISCERETE FEATURE


```python
numeric = df[['EmpEducationLevel','EmpEnvironmentSatisfaction','EmpJobInvolvement','EmpJobLevel','EmpJobSatisfaction',
                 'EmpWorkLifeBalance']]

plt.figure(figsize=(25, 15))
plotno = 1

for column in numeric:
    if plotno <= 13:
        plt.subplot(2, 3, plotno)

        # Lấy số lượng các giá trị duy nhất trong cột để tạo bảng màu tương ứng
        unique_vals = numeric[column].nunique()
        colors = sns.color_palette("Dark2", unique_vals)  # HSV để có màu đa dạng

        sns.countplot(x=numeric[column], hue=df.PerformanceRating, palette=colors)
        plt.xlabel(column, fontsize=15)
        plt.ylabel('PerformanceRating')
        plt.xticks(rotation=45)  # Xoay nhãn cho dễ đọc nếu cần
    plotno += 1

plt.tight_layout()
plt.show()
```


    
![png](output_91_0.png)
    


##VIII.Multivariate analysis


```python
import matplotlib.pyplot as plt
import seaborn as sns

palette = sns.color_palette("Dark2")

fig = plt.figure(figsize=(24, 16))
fig.suptitle('HR Multivariate Analysis with Performance Rating', fontsize=22)

# 1. Age vs Total Experience (chiếm 2 ô đầu hàng 1)
ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=4)
sns.lineplot(ax=ax1, x='Age', y='TotalWorkExperienceInYears', hue='PerformanceRating', data=df, palette=palette)
ax1.set_title('Age vs Total Experience')
ax1.set_xlabel('Age')
ax1.set_ylabel('Total Experience')

# 2. Gender vs Companies Worked
ax2 = plt.subplot2grid((3, 4), (1, 0))
sns.barplot(ax=ax2, x='Gender', y='NumCompaniesWorked', hue='PerformanceRating', data=df, palette=palette)
ax2.set_title('Gender vs Companies Worked')
ax2.set_xlabel('Gender')
ax2.set_ylabel('Num Companies Worked')
ax2.tick_params(axis='x', rotation=90)

# 3. Marital Status vs Salary Hike
ax3 = plt.subplot2grid((3, 4), (1, 1))
sns.barplot(ax=ax3, x='MaritalStatus', y='EmpLastSalaryHikePercent', hue='PerformanceRating', data=df, palette=palette)
ax3.set_title('Marital Status vs Salary Hike %')
ax3.set_xlabel('Marital Status')
ax3.set_ylabel('Salary Hike %')
ax3.tick_params(axis='x', rotation=90)

# 4. Business Travel vs Env Satisfaction
ax4 = plt.subplot2grid((3, 4), (1, 2))
sns.barplot(ax=ax4, x='BusinessTravelFrequency', y='EmpEnvironmentSatisfaction', hue='PerformanceRating', data=df, palette=palette)
ax4.set_title('Business Travel vs Env Satisfaction')
ax4.set_xlabel('Business Travel')
ax4.set_ylabel('Environment Satisfaction')
ax4.tick_params(axis='x', rotation=90)

# 5. Attrition vs Years With Manager
ax5 = plt.subplot2grid((3, 4), (1, 3))
sns.barplot(ax=ax5, x='Attrition', y='YearsWithCurrManager', hue='PerformanceRating', data=df, palette=palette)
ax5.set_title('Attrition vs Years With Manager')
ax5.set_xlabel('Attrition')
ax5.set_ylabel('Years With Manager')
ax5.tick_params(axis='x', rotation=90)

# 6. Education vs Current Role Experience
ax6 = plt.subplot2grid((3, 4), (2, 0))
sns.barplot(ax=ax6, x='EducationBackground', y='ExperienceYearsInCurrentRole', hue='PerformanceRating', data=df, palette=palette)
ax6.set_title('Education vs Role Experience')
ax6.set_xlabel('Education Background')
ax6.set_ylabel('Current Role Experience')
ax6.tick_params(axis='x', rotation=90)

# 7. OverTime vs Distance From Home
ax7 = plt.subplot2grid((3, 4), (2, 1))
sns.barplot(ax=ax7, x='OverTime', y='DistanceFromHome', hue='PerformanceRating', data=df, palette=palette)
ax7.set_title('OverTime vs Distance From Home')
ax7.set_xlabel('OverTime')
ax7.set_ylabel('Distance From Home')
ax7.tick_params(axis='x', rotation=90)

# 8. Department vs Training
ax8 = plt.subplot2grid((3, 4), (2, 2))
sns.barplot(ax=ax8, x='EmpDepartment', y='TrainingTimesLastYear', hue='PerformanceRating', data=df, palette=palette)
ax8.set_title('Department vs Training')
ax8.set_xlabel('Department')
ax8.set_ylabel('Trainings Last Year')
ax8.tick_params(axis='x', rotation=90)

# 9. Attrition vs Companies Worked
ax9 = plt.subplot2grid((3, 4), (2, 3))
sns.barplot(ax=ax9, x='Attrition', y='NumCompaniesWorked', hue='PerformanceRating', data=df, palette=palette)
ax9.set_title('Attrition vs Companies Worked')
ax9.set_xlabel('Attrition')
ax9.set_ylabel('Companies Worked')
ax9.tick_params(axis='x', rotation=90)


# Layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

```


    
![png](output_93_0.png)
    


**Department with Performance rating**


```python
plt.figure(figsize=(20, 14))

# 1. Boxplot: Performance Rating by Department & Gender
plt.subplot(2, 1, 1)
sns.violinplot(x='EmpDepartment',y='PerformanceRating',hue=df.Gender,data=df,palette='Dark2')
plt.title('Performance Rating by Department and Gender', fontsize=20)
plt.xlabel('EmpDepartment', fontsize=16)
plt.ylabel('PerformanceRating', fontsize=16)
plt.xticks(rotation=90)

# 2. Countplot: Count of Employees by Department and Performance Rating
plt.subplot(2, 1, 2)
ax = sns.countplot(x='EmpDepartment', hue='PerformanceRating', data=df, palette='Dark2')
plt.title('Employee Count by Department and Performance Rating', fontsize=20)
plt.xlabel('EmpDepartment', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.xticks(rotation=90)

# Add annotations to count bars
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.0f}', (p.get_x() + p.get_width() / 2., height + 1),
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

```


    
![png](output_95_0.png)
    



```python
# PERCENT COUNT IN EMPLOYEE DEPARTMENT WITH PERFORMANCE RATING
percent_table = pd.crosstab(
    index=df['PerformanceRating'],
    columns=df['EmpDepartment'],
    normalize=True,
    margins=True
)*100
round(percent_table,2)
```





  <div id="df-19f06ce4-f8dd-4a89-8315-f7252ba8810c" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>EmpDepartment</th>
      <th>Data Science</th>
      <th>Development</th>
      <th>Finance</th>
      <th>Human Resources</th>
      <th>Research &amp; Development</th>
      <th>Sales</th>
      <th>All</th>
    </tr>
    <tr>
      <th>PerformanceRating</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0.08</td>
      <td>1.08</td>
      <td>1.25</td>
      <td>0.83</td>
      <td>5.67</td>
      <td>7.25</td>
      <td>16.17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.42</td>
      <td>25.33</td>
      <td>2.50</td>
      <td>3.17</td>
      <td>19.50</td>
      <td>20.92</td>
      <td>72.83</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.17</td>
      <td>3.67</td>
      <td>0.33</td>
      <td>0.50</td>
      <td>3.42</td>
      <td>2.92</td>
      <td>11.00</td>
    </tr>
    <tr>
      <th>All</th>
      <td>1.67</td>
      <td>30.08</td>
      <td>4.08</td>
      <td>4.50</td>
      <td>28.58</td>
      <td>31.08</td>
      <td>100.00</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-19f06ce4-f8dd-4a89-8315-f7252ba8810c')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-19f06ce4-f8dd-4a89-8315-f7252ba8810c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-19f06ce4-f8dd-4a89-8315-f7252ba8810c');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-f51ad55b-1431-4ecb-bcca-91ad7b3cc1ff">
  <button class="colab-df-quickchart" onclick="quickchart('df-f51ad55b-1431-4ecb-bcca-91ad7b3cc1ff')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-f51ad55b-1431-4ecb-bcca-91ad7b3cc1ff button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




##IX.Factor affecting to the employee performance


```python
# Chỉ lấy các cột số
numeric_cols = df.select_dtypes(include=['float64', 'int64'])

# Tính tương quan với PerformanceRating
correlation = numeric_cols.corr()['PerformanceRating'].drop('PerformanceRating').sort_values(key=abs, ascending=False)
correlation=correlation.sort_values(ascending=False)
# Hiển thị top 3
top3_corr = correlation.head(3)
print("Top 3 correlated numeric features with PerformanceRating:")
print(top3_corr)
# Vẽ Biểu đồ
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation.index, y=correlation.values, palette='Dark2')
plt.title('Correlated Numeric Features with PerformanceRating')
plt.xlabel('Feature')
plt.ylabel('Correlation')
plt.xticks(rotation=90)
plt.show()

```

    Top 3 correlated numeric features with PerformanceRating:
    EmpEnvironmentSatisfaction    0.395561
    EmpLastSalaryHikePercent      0.333722
    EmpWorkLifeBalance            0.124429
    Name: PerformanceRating, dtype: float64



    
![png](output_98_1.png)
    




##X.Data Preprocessing

###1.Handling missing value


```python
df.isnull().sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>EmpNumber</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Gender</th>
      <td>0</td>
    </tr>
    <tr>
      <th>EducationBackground</th>
      <td>0</td>
    </tr>
    <tr>
      <th>MaritalStatus</th>
      <td>0</td>
    </tr>
    <tr>
      <th>EmpDepartment</th>
      <td>0</td>
    </tr>
    <tr>
      <th>EmpJobRole</th>
      <td>0</td>
    </tr>
    <tr>
      <th>BusinessTravelFrequency</th>
      <td>0</td>
    </tr>
    <tr>
      <th>DistanceFromHome</th>
      <td>0</td>
    </tr>
    <tr>
      <th>EmpEducationLevel</th>
      <td>0</td>
    </tr>
    <tr>
      <th>EmpEnvironmentSatisfaction</th>
      <td>0</td>
    </tr>
    <tr>
      <th>EmpHourlyRate</th>
      <td>0</td>
    </tr>
    <tr>
      <th>EmpJobInvolvement</th>
      <td>0</td>
    </tr>
    <tr>
      <th>EmpJobLevel</th>
      <td>0</td>
    </tr>
    <tr>
      <th>EmpJobSatisfaction</th>
      <td>0</td>
    </tr>
    <tr>
      <th>NumCompaniesWorked</th>
      <td>0</td>
    </tr>
    <tr>
      <th>OverTime</th>
      <td>0</td>
    </tr>
    <tr>
      <th>EmpLastSalaryHikePercent</th>
      <td>0</td>
    </tr>
    <tr>
      <th>EmpRelationshipSatisfaction</th>
      <td>0</td>
    </tr>
    <tr>
      <th>TotalWorkExperienceInYears</th>
      <td>0</td>
    </tr>
    <tr>
      <th>TrainingTimesLastYear</th>
      <td>0</td>
    </tr>
    <tr>
      <th>EmpWorkLifeBalance</th>
      <td>0</td>
    </tr>
    <tr>
      <th>ExperienceYearsAtThisCompany</th>
      <td>0</td>
    </tr>
    <tr>
      <th>ExperienceYearsInCurrentRole</th>
      <td>0</td>
    </tr>
    <tr>
      <th>YearsSinceLastPromotion</th>
      <td>0</td>
    </tr>
    <tr>
      <th>YearsWithCurrManager</th>
      <td>0</td>
    </tr>
    <tr>
      <th>Attrition</th>
      <td>0</td>
    </tr>
    <tr>
      <th>PerformanceRating</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



###2.Encoding


```python
print('Categorical features: ', list(df.select_dtypes('object')))
```

    Categorical features:  ['EmpNumber', 'Gender', 'EducationBackground', 'MaritalStatus', 'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition']



```python
#Gender
print('Before:')
print(df['Gender'].value_counts())
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
print('----------------------')
#print sau khi đổi
print('After:')
print(df['Gender'].value_counts())
```

    Before:
    Gender
    Male      725
    Female    475
    Name: count, dtype: int64
    ----------------------
    After:
    Gender
    1    725
    0    475
    Name: count, dtype: int64



```python
#EducationBackground
print('Before:')
print(df['EducationBackground'].value_counts())
df['EducationBackground'] = df['EducationBackground'].map({'Life Sciences': 0, 'Medical': 1, 'Marketing': 2, 'Technical Degree': 3, 'Human Resources': 4, 'Other': 5})
print('----------------------')
#print sau khi đổi
print('After:')
print(df['EducationBackground'].value_counts())
```

    Before:
    EducationBackground
    Life Sciences       492
    Medical             384
    Marketing           137
    Technical Degree    100
    Other                66
    Human Resources      21
    Name: count, dtype: int64
    ----------------------
    After:
    EducationBackground
    0    492
    1    384
    2    137
    3    100
    5     66
    4     21
    Name: count, dtype: int64



```python
#MaritalStatus
print('Before:')
print(df['MaritalStatus'].value_counts())
df['MaritalStatus'] = df['MaritalStatus'].map({'Married': 2, 'Single': 1, 'Divorced': 0})
print('----------------------')
#print sau khi đổi
print('After:')
print(df['MaritalStatus'].value_counts())
```

    Before:
    MaritalStatus
    Married     548
    Single      384
    Divorced    268
    Name: count, dtype: int64
    ----------------------
    After:
    MaritalStatus
    2    548
    1    384
    0    268
    Name: count, dtype: int64



```python
#EmpDepartment
print('Before:')
print(df['EmpDepartment'].value_counts())
df['EmpDepartment'] = df['EmpDepartment'].map({'Sales': 5, 'Development':4, 'Research & Development':3, 'Human Resources':2, 'Finance':1, 'Data Science':0})
print('----------------------')
#print sau khi đổi
print('After:')
print(df['EmpDepartment'].value_counts())
#
```

    Before:
    EmpDepartment
    Sales                     373
    Development               361
    Research & Development    343
    Human Resources            54
    Finance                    49
    Data Science               20
    Name: count, dtype: int64
    ----------------------
    After:
    EmpDepartment
    5    373
    4    361
    3    343
    2     54
    1     49
    0     20
    Name: count, dtype: int64



```python
#EmpJobRole
print('Before:')
print(df['EmpJobRole'].value_counts())
df['EmpJobRole']=df['EmpJobRole'].map({'Sales Executive':18,
                                        'Developer': 17,
                                        'Manager R&D':16,
                                        'Research Scientist':15,
                                        'Sales Representative':14,
                                        'Laboratory Technician':13,
                                        'Senior Developer':12,
                                        'Manager':11,
                                        'Finance Manager':10,
                                        'Human Resources':9,
                                        'Technical Lead':8,
                                        'Manufacturing Director':7,
                                       'Healthcare Representative':6,
                                       'Data Scientist':5,
                                       'Research Director':4,
                                       'Business Analyst':3,
                                       'Senior Manager R&D':2,
                                       'Delivery Manager':1,
                                       'Technical Architect':0})

print('----------------------')
#print sau khi đổi
print('After:')
print(df['EmpJobRole'].value_counts())
```

    Before:
    EmpJobRole
    Sales Executive              270
    Developer                    236
    Manager R&D                   94
    Research Scientist            77
    Sales Representative          69
    Laboratory Technician         64
    Senior Developer              52
    Manager                       51
    Finance Manager               49
    Human Resources               45
    Technical Lead                38
    Manufacturing Director        33
    Healthcare Representative     33
    Data Scientist                20
    Research Director             19
    Business Analyst              16
    Senior Manager R&D            15
    Delivery Manager              12
    Technical Architect            7
    Name: count, dtype: int64
    ----------------------
    After:
    EmpJobRole
    18    270
    17    236
    16     94
    15     77
    14     69
    13     64
    12     52
    11     51
    10     49
    9      45
    8      38
    7      33
    6      33
    5      20
    4      19
    3      16
    2      15
    1      12
    0       7
    Name: count, dtype: int64



```python
#BusinessTravelFrequency
print('Before:')
print(df['BusinessTravelFrequency'].value_counts())
df['BusinessTravelFrequency'] = df['BusinessTravelFrequency'].map({'Travel_Rarely':2, 'Travel_Frequently':1,'Non-Travel':0})
print('----------------------')
#print sau khi đổi
print('After:')
print(df['BusinessTravelFrequency'].value_counts())
```

    Before:
    BusinessTravelFrequency
    Travel_Rarely        846
    Travel_Frequently    222
    Non-Travel           132
    Name: count, dtype: int64
    ----------------------
    After:
    BusinessTravelFrequency
    2    846
    1    222
    0    132
    Name: count, dtype: int64



```python
#OverTime
print('Before:')
print(df['OverTime'].value_counts())
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
print('----------------------')
#print sau khi đổi
print('After:')
print(df['OverTime'].value_counts())
```

    Before:
    OverTime
    No     847
    Yes    353
    Name: count, dtype: int64
    ----------------------
    After:
    OverTime
    0    847
    1    353
    Name: count, dtype: int64



```python
#Attrition
print('Before:')
print(df['Attrition'].value_counts())
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
print('----------------------')
print('After:')
print(df['Attrition'].value_counts())
```

    Before:
    Attrition
    No     1022
    Yes     178
    Name: count, dtype: int64
    ----------------------
    After:
    Attrition
    0    1022
    1     178
    Name: count, dtype: int64


###3.Check duplicate


```python
print(df.duplicated().sum())
```

    0


Their is no Duplicates is present in data.

###4.Check skew


```python
print('1.Distance From Home Feature Skewness:',df.DistanceFromHome.skew())
print('2.Employee Hourly Rate Feature Skewness:',df.EmpHourlyRate.skew())
print('3.Employee Last Salary Hike Percent Feature Skewness:',df.EmpLastSalaryHikePercent.skew())
print('4.Total Work Experiance In Year Feature Skewness:',df.TotalWorkExperienceInYears.skew())
print('5.Experiance Year At This Company Feature Skewness:',df.ExperienceYearsAtThisCompany.skew())
print('6.Experiance Year In Current Role Feature Skewness:',df.ExperienceYearsInCurrentRole.skew())
print('7.Year Since Last Promotion Feature Skewness:',df.YearsSinceLastPromotion.skew())
print('8.Years With Current Manager Feature Skewness:',df.YearsWithCurrManager.skew())
```

    1.Distance From Home Feature Skewness: 0.9629561160828001
    2.Employee Hourly Rate Feature Skewness: -0.035164888157941436
    3.Employee Last Salary Hike Percent Feature Skewness: 0.8086536332261228
    4.Total Work Experiance In Year Feature Skewness: 1.0868618597364565
    5.Experiance Year At This Company Feature Skewness: 1.789054979919473
    6.Experiance Year In Current Role Feature Skewness: 0.8881586703270758
    7.Year Since Last Promotion Feature Skewness: 1.9749315589155791
    8.Years With Current Manager Feature Skewness: 0.8131582957766446


|Feature | Skewness | Note|
|--------|----------|----------|
|Distance From Home | 0.96 | Moderate|
|Employee Hourly Rate | -0.035 | Almost standard|
|Employee Last Salary Hike Percent | 0.81 | Moderate|
|Total Work Experience In Years | 1.09 | **Strong moderate**|
|Experience Years At This Company | 1.79 | **Strong moderate**|
|Experience Years In Current Role | 0.89 | Moderate|
|Years Since Last Promotion | 1.97 | **Strong moderate**|
|Years With Current Manager | 0.81 | Moderate|

**Tree-based models** like Random Forest, XGBoost; Neural network model then skew processing is not necessary because it **don't affect** by skew.

###5.Check Outlier


```python
outlier = [
    'Age','DistanceFromHome','EmpHourlyRate','EmpLastSalaryHikePercent',
    'TotalWorkExperienceInYears','TrainingTimesLastYear','ExperienceYearsAtThisCompany',
    'ExperienceYearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager'
]

plt.figure(figsize=(20, 10))
for i, outlier in enumerate(outlier, 1):
    plt.subplot(2, 5, i)
    sns.boxplot(y=df[outlier], palette='Dark2')
    plt.title(outlier, fontsize=12)
    plt.tight_layout()
plt.show()
```


    
![png](output_121_0.png)
    



```python
cols = [
    'Age','DistanceFromHome','EmpHourlyRate','EmpLastSalaryHikePercent',
    'TotalWorkExperienceInYears','TrainingTimesLastYear',
    'ExperienceYearsAtThisCompany','ExperienceYearsInCurrentRole',
    'YearsSinceLastPromotion','YearsWithCurrManager'
]

df = df.copy()
replaced_counts = {}

for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    median = df[col].median()

    # Tạo mặt nạ outlier
    mask_outlier = (df[col] < lower) | (df[col] > upper)

    # Đếm số lượng giá trị bị thay thế
    replaced_counts[col] = mask_outlier.sum()

    # Thay thế bằng median
    df[col] = np.where(mask_outlier, median, df[col])

# In kết quả
print("The number of outliers is replaced by the median:")
for col, count in replaced_counts.items():
    print(f"{col}: {count} value.")

```

    The number of outliers is replaced by the median:
    Age: 0 value.
    DistanceFromHome: 0 value.
    EmpHourlyRate: 0 value.
    EmpLastSalaryHikePercent: 0 value.
    TotalWorkExperienceInYears: 51 value.
    TrainingTimesLastYear: 188 value.
    ExperienceYearsAtThisCompany: 56 value.
    ExperienceYearsInCurrentRole: 16 value.
    YearsSinceLastPromotion: 88 value.
    YearsWithCurrManager: 11 value.



```python
outlier = [
    'Age','DistanceFromHome','EmpHourlyRate','EmpLastSalaryHikePercent',
    'TotalWorkExperienceInYears','TrainingTimesLastYear','ExperienceYearsAtThisCompany',
    'ExperienceYearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager'
]

plt.figure(figsize=(20, 10))
for i, outlier in enumerate(outlier, 1):
    plt.subplot(2, 5, i)
    sns.boxplot(y=df[outlier], palette='Dark2')
    plt.title(outlier, fontsize=12)
    plt.tight_layout()
plt.show()
```


    
![png](output_123_0.png)
    



✅ **Why replace with the median instead of removing?**

1. **Preserves important data**
- If you **remove rows containing outliers**, you lose the **entire row’s information**.
- These columns might be directly related to **target variables** like *PerformanceRating*, *Attrition*, etc., so keeping the row is important.

🧠 **Instead of deleting the row → just replace the “abnormal” value with the median**

---
2. **Median is not affected by outliers**
- The mean can be **skewed** by extreme values.
- In contrast, the **median is a more stable representation of central tendency**, making it a better candidate for replacement.

Example:
```text
Average experience: 4.1 years
But one person has 50 years → mean increases to 6 years
→ Median of 4 years is more reasonable
```

---

3. **These columns often have skewed distributions or realistic extreme values**
- For instance:
  - Someone might have just joined the company (0 years)
  - Someone hasn't been promoted in 15 years
  - Someone has worked with the same manager for 20 years

➡️ If you remove all these values, you **lose important real-world cases**.

---
4. **Preserves row count for statistics and models**
- For ML models or statistical analysis: keeping the number of rows consistent improves reliability.

---

📌 Conclusion:
> Replacing outliers with the **median** is a safe, effective, and appropriate approach, especially when dealing with **continuous variables related to time or experience**, and when **data retention is a priority**.



```python
#Drop unique employee ID column
df_1=df.iloc[:,1:]
df_1
```





  <div id="df-eb4ba542-9722-4a78-b083-c5e3d25885c0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>EducationBackground</th>
      <th>MaritalStatus</th>
      <th>EmpDepartment</th>
      <th>EmpJobRole</th>
      <th>BusinessTravelFrequency</th>
      <th>DistanceFromHome</th>
      <th>EmpEducationLevel</th>
      <th>EmpEnvironmentSatisfaction</th>
      <th>...</th>
      <th>EmpRelationshipSatisfaction</th>
      <th>TotalWorkExperienceInYears</th>
      <th>TrainingTimesLastYear</th>
      <th>EmpWorkLifeBalance</th>
      <th>ExperienceYearsAtThisCompany</th>
      <th>ExperienceYearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
      <th>Attrition</th>
      <th>PerformanceRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>18</td>
      <td>2</td>
      <td>10.0</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>4</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>10.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>18</td>
      <td>2</td>
      <td>14.0</td>
      <td>4</td>
      <td>4</td>
      <td>...</td>
      <td>4</td>
      <td>20.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>18</td>
      <td>1</td>
      <td>5.0</td>
      <td>4</td>
      <td>4</td>
      <td>...</td>
      <td>3</td>
      <td>20.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>18.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41.0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>11</td>
      <td>2</td>
      <td>10.0</td>
      <td>4</td>
      <td>2</td>
      <td>...</td>
      <td>2</td>
      <td>23.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>18</td>
      <td>2</td>
      <td>16.0</td>
      <td>4</td>
      <td>1</td>
      <td>...</td>
      <td>4</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1195</th>
      <td>27.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>18</td>
      <td>1</td>
      <td>3.0</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>37.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>12</td>
      <td>2</td>
      <td>10.0</td>
      <td>2</td>
      <td>4</td>
      <td>...</td>
      <td>1</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1197</th>
      <td>50.0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>12</td>
      <td>2</td>
      <td>28.0</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>3</td>
      <td>20.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>20.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1198</th>
      <td>34.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>9.0</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>4</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1199</th>
      <td>24.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>18</td>
      <td>2</td>
      <td>3.0</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>1200 rows × 27 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-eb4ba542-9722-4a78-b083-c5e3d25885c0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-eb4ba542-9722-4a78-b083-c5e3d25885c0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-eb4ba542-9722-4a78-b083-c5e3d25885c0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-dfec0219-6e58-4090-8584-0eca54b37e11">
      <button class="colab-df-quickchart" onclick="quickchart('df-dfec0219-6e58-4090-8584-0eca54b37e11')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-dfec0219-6e58-4090-8584-0eca54b37e11 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_370bb822-2847-4b7a-98be-6d7ed03b6f34">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_1')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_370bb822-2847-4b7a-98be-6d7ed03b6f34 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_1');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
# Standardize dữ liệu (bắt buộc trước khi PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_1)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Vẽ biểu đồ tỷ lệ phương sai tích luỹ
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

plt.figure(figsize=(10, 6))
sns.lineplot(x=range(1, len(cumulative_variance)+1), y=cumulative_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.axhline(y=0.9, color='red', linestyle='--', label='90% threshold')
plt.legend()
plt.show()

```


    
![png](output_126_0.png)
    


From above PCA it shows the 20 feature has less varaince loss, so we are going to select 20 feature


```python
pca = PCA(n_components=20)
new_data = pca.fit_transform(df_1)
new_data
```




    array([[-1.11557584e+01, -2.58826089e+00,  1.32985129e+00, ...,
             2.07026915e-01, -4.35644718e-01, -1.02845409e+00],
           [-2.34156251e+01,  1.48138727e+01,  4.00581301e+00, ...,
            -2.06487716e-02, -1.41521941e+00, -2.40086126e-01],
           [-1.78064522e+01,  1.19035626e+01, -3.94800090e+00, ...,
             8.70677019e-01, -3.60873061e-01,  1.74116774e-01],
           ...,
           [ 8.65694612e+00,  1.98652416e+01,  1.78974519e+01, ...,
             1.78836515e-01, -2.41689050e+00,  6.57992875e-01],
           [-2.01686051e+01, -6.10325222e-01,  2.18266360e-01, ...,
             7.23238277e-02,  1.96896080e+00,  9.88947866e-01],
           [-1.49160338e+00, -1.54366829e+01, -5.31660357e+00, ...,
            -2.62861880e-01,  8.04796999e-01,  1.64666106e-01]])




```python
principle_df = pd.DataFrame(data=new_data,columns=['pca1','pca2','pca3','pca4','pca5','pca6','pca7','pca8','pca9','pca10',
                            'pca11','pca12','pca13','pca14','pca15','pca16','pca17','pca18','pca19','pca20'])

# Add target veriable
principle_df['PerformanceRating']=df_1.PerformanceRating

principle_df.head()
```





  <div id="df-3f595f4f-acd9-4e4c-9fca-c83ad1ccf5e1" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pca1</th>
      <th>pca2</th>
      <th>pca3</th>
      <th>pca4</th>
      <th>pca5</th>
      <th>pca6</th>
      <th>pca7</th>
      <th>pca8</th>
      <th>pca9</th>
      <th>pca10</th>
      <th>...</th>
      <th>pca12</th>
      <th>pca13</th>
      <th>pca14</th>
      <th>pca15</th>
      <th>pca16</th>
      <th>pca17</th>
      <th>pca18</th>
      <th>pca19</th>
      <th>pca20</th>
      <th>PerformanceRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-11.155758</td>
      <td>-2.588261</td>
      <td>1.329851</td>
      <td>6.957677</td>
      <td>4.357053</td>
      <td>-1.564258</td>
      <td>-3.239375</td>
      <td>-0.590075</td>
      <td>0.094515</td>
      <td>1.597066</td>
      <td>...</td>
      <td>0.672242</td>
      <td>0.813835</td>
      <td>0.824532</td>
      <td>1.813584</td>
      <td>0.648844</td>
      <td>0.061707</td>
      <td>0.207027</td>
      <td>-0.435645</td>
      <td>-1.028454</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-23.415625</td>
      <td>14.813873</td>
      <td>4.005813</td>
      <td>1.277256</td>
      <td>5.078435</td>
      <td>3.015044</td>
      <td>-3.089418</td>
      <td>-2.071479</td>
      <td>-1.762426</td>
      <td>0.538671</td>
      <td>...</td>
      <td>1.162620</td>
      <td>1.296272</td>
      <td>-1.561136</td>
      <td>0.193382</td>
      <td>1.465217</td>
      <td>-0.065494</td>
      <td>-0.020649</td>
      <td>-1.415219</td>
      <td>-0.240086</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-17.806452</td>
      <td>11.903563</td>
      <td>-3.948001</td>
      <td>15.247922</td>
      <td>4.756587</td>
      <td>-1.678096</td>
      <td>6.029107</td>
      <td>3.363880</td>
      <td>0.190706</td>
      <td>0.608704</td>
      <td>...</td>
      <td>-0.768847</td>
      <td>1.564376</td>
      <td>-1.371808</td>
      <td>-0.655582</td>
      <td>1.363135</td>
      <td>0.411471</td>
      <td>0.870677</td>
      <td>-0.360873</td>
      <td>0.174117</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.307404</td>
      <td>9.984291</td>
      <td>0.315620</td>
      <td>3.743315</td>
      <td>-3.265847</td>
      <td>7.703459</td>
      <td>0.372131</td>
      <td>-2.086110</td>
      <td>-2.447636</td>
      <td>0.104541</td>
      <td>...</td>
      <td>2.538227</td>
      <td>-1.403603</td>
      <td>1.708467</td>
      <td>-0.066858</td>
      <td>0.231003</td>
      <td>-1.057545</td>
      <td>-0.600487</td>
      <td>1.968561</td>
      <td>-1.328891</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18.972383</td>
      <td>17.493753</td>
      <td>4.775380</td>
      <td>-15.038408</td>
      <td>6.157293</td>
      <td>-3.247555</td>
      <td>-1.561450</td>
      <td>3.556415</td>
      <td>-1.285795</td>
      <td>0.147145</td>
      <td>...</td>
      <td>1.208907</td>
      <td>-1.473301</td>
      <td>-2.065350</td>
      <td>0.133920</td>
      <td>0.192423</td>
      <td>0.696976</td>
      <td>-0.107711</td>
      <td>-0.183127</td>
      <td>-0.692336</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3f595f4f-acd9-4e4c-9fca-c83ad1ccf5e1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-3f595f4f-acd9-4e4c-9fca-c83ad1ccf5e1 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3f595f4f-acd9-4e4c-9fca-c83ad1ccf5e1');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-520a7066-af3c-4b16-ba4b-5c61fa990caf">
      <button class="colab-df-quickchart" onclick="quickchart('df-520a7066-af3c-4b16-ba4b-5c61fa990caf')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-520a7066-af3c-4b16-ba4b-5c61fa990caf button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




##XI.ML model to prediction

###1.Define independant and dependant features (target variable)


```python
X=principle_df.drop('PerformanceRating',axis=1)
y=principle_df['PerformanceRating']
```

###2.Upsampling

**What is SMOTE Upsampling Used For?**

**SMOTE** (Synthetic Minority Over-sampling Technique) is used to address the problem of **imbalanced datasets**, especially in **classification tasks** where one class (usually the "positive" or rare class) is underrepresented.

---
**Why Use SMOTE?**

Instead of simply duplicating the minority class data (which may lead to overfitting), **SMOTE generates new, synthetic samples** by:

* Selecting a data point from the minority class.
* Finding its *k* nearest neighbors (also from the minority class).
* Creating a new synthetic point by interpolating between the selected point and one of its neighbors.


```python
from collections import Counter
from imblearn.over_sampling import SMOTE
sm = SMOTE(k_neighbors=5) # obeject creation
print("unbalanced data   :  ",Counter(y))
X_sm,y_sm = sm.fit_resample(X,y)
print("balanced data:    :",Counter(y_sm))
```

    unbalanced data   :   Counter({3: 874, 2: 194, 4: 132})
    balanced data:    : Counter({3: 874, 4: 874, 2: 874})



```python
#encoding PerformanceRating in y_sm
y_sm = y_sm.map({2: 0, 3: 1, 4: 2})
print('----------------------')
#print sau khi đổi
print('After:')
print(y_sm.value_counts())
```

    ----------------------
    After:
    PerformanceRating
    1    874
    2    874
    0    874
    Name: count, dtype: int64



```python
X_sm
```





  <div id="df-be707f50-c778-410f-ba40-ef7c3a8ad1e3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pca1</th>
      <th>pca2</th>
      <th>pca3</th>
      <th>pca4</th>
      <th>pca5</th>
      <th>pca6</th>
      <th>pca7</th>
      <th>pca8</th>
      <th>pca9</th>
      <th>pca10</th>
      <th>pca11</th>
      <th>pca12</th>
      <th>pca13</th>
      <th>pca14</th>
      <th>pca15</th>
      <th>pca16</th>
      <th>pca17</th>
      <th>pca18</th>
      <th>pca19</th>
      <th>pca20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-11.155758</td>
      <td>-2.588261</td>
      <td>1.329851</td>
      <td>6.957677</td>
      <td>4.357053</td>
      <td>-1.564258</td>
      <td>-3.239375</td>
      <td>-0.590075</td>
      <td>0.094515</td>
      <td>1.597066</td>
      <td>-2.121582</td>
      <td>0.672242</td>
      <td>0.813835</td>
      <td>0.824532</td>
      <td>1.813584</td>
      <td>0.648844</td>
      <td>0.061707</td>
      <td>0.207027</td>
      <td>-0.435645</td>
      <td>-1.028454</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-23.415625</td>
      <td>14.813873</td>
      <td>4.005813</td>
      <td>1.277256</td>
      <td>5.078435</td>
      <td>3.015044</td>
      <td>-3.089418</td>
      <td>-2.071479</td>
      <td>-1.762426</td>
      <td>0.538671</td>
      <td>-1.667084</td>
      <td>1.162620</td>
      <td>1.296272</td>
      <td>-1.561136</td>
      <td>0.193382</td>
      <td>1.465217</td>
      <td>-0.065494</td>
      <td>-0.020649</td>
      <td>-1.415219</td>
      <td>-0.240086</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-17.806452</td>
      <td>11.903563</td>
      <td>-3.948001</td>
      <td>15.247922</td>
      <td>4.756587</td>
      <td>-1.678096</td>
      <td>6.029107</td>
      <td>3.363880</td>
      <td>0.190706</td>
      <td>0.608704</td>
      <td>-2.769329</td>
      <td>-0.768847</td>
      <td>1.564376</td>
      <td>-1.371808</td>
      <td>-0.655582</td>
      <td>1.363135</td>
      <td>0.411471</td>
      <td>0.870677</td>
      <td>-0.360873</td>
      <td>0.174117</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.307404</td>
      <td>9.984291</td>
      <td>0.315620</td>
      <td>3.743315</td>
      <td>-3.265847</td>
      <td>7.703459</td>
      <td>0.372131</td>
      <td>-2.086110</td>
      <td>-2.447636</td>
      <td>0.104541</td>
      <td>-1.575280</td>
      <td>2.538227</td>
      <td>-1.403603</td>
      <td>1.708467</td>
      <td>-0.066858</td>
      <td>0.231003</td>
      <td>-1.057545</td>
      <td>-0.600487</td>
      <td>1.968561</td>
      <td>-1.328891</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18.972383</td>
      <td>17.493753</td>
      <td>4.775380</td>
      <td>-15.038408</td>
      <td>6.157293</td>
      <td>-3.247555</td>
      <td>-1.561450</td>
      <td>3.556415</td>
      <td>-1.285795</td>
      <td>0.147145</td>
      <td>0.504912</td>
      <td>1.208907</td>
      <td>-1.473301</td>
      <td>-2.065350</td>
      <td>0.133920</td>
      <td>0.192423</td>
      <td>0.696976</td>
      <td>-0.107711</td>
      <td>-0.183127</td>
      <td>-0.692336</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2617</th>
      <td>-24.447766</td>
      <td>1.370678</td>
      <td>-3.919854</td>
      <td>0.371372</td>
      <td>4.387509</td>
      <td>-4.292830</td>
      <td>8.596996</td>
      <td>-0.808563</td>
      <td>-1.060622</td>
      <td>1.766453</td>
      <td>4.404359</td>
      <td>0.080562</td>
      <td>0.365913</td>
      <td>-1.695946</td>
      <td>0.170724</td>
      <td>-0.525441</td>
      <td>0.532728</td>
      <td>-0.482127</td>
      <td>-0.039673</td>
      <td>0.107278</td>
    </tr>
    <tr>
      <th>2618</th>
      <td>25.291405</td>
      <td>9.960037</td>
      <td>-9.506878</td>
      <td>-12.820355</td>
      <td>4.039808</td>
      <td>-6.175239</td>
      <td>4.904472</td>
      <td>-0.368234</td>
      <td>1.873364</td>
      <td>-1.224382</td>
      <td>0.350872</td>
      <td>-0.782441</td>
      <td>-0.521117</td>
      <td>-2.426350</td>
      <td>0.177001</td>
      <td>-0.745127</td>
      <td>-0.631842</td>
      <td>0.170057</td>
      <td>-0.246990</td>
      <td>1.227279</td>
    </tr>
    <tr>
      <th>2619</th>
      <td>-32.952264</td>
      <td>-8.317763</td>
      <td>14.215831</td>
      <td>-2.480954</td>
      <td>-2.854570</td>
      <td>-0.027414</td>
      <td>4.296947</td>
      <td>-0.562651</td>
      <td>-0.280475</td>
      <td>-0.019928</td>
      <td>-0.650865</td>
      <td>0.561862</td>
      <td>0.073832</td>
      <td>0.033062</td>
      <td>1.253768</td>
      <td>-0.642340</td>
      <td>-0.482317</td>
      <td>-0.580589</td>
      <td>0.076923</td>
      <td>-0.283740</td>
    </tr>
    <tr>
      <th>2620</th>
      <td>-8.894778</td>
      <td>14.666064</td>
      <td>-7.912222</td>
      <td>-1.918736</td>
      <td>-5.154779</td>
      <td>-7.768683</td>
      <td>5.536061</td>
      <td>-0.420602</td>
      <td>0.861352</td>
      <td>0.111241</td>
      <td>-1.784363</td>
      <td>-0.972519</td>
      <td>0.115302</td>
      <td>0.403997</td>
      <td>-0.901090</td>
      <td>0.338397</td>
      <td>-0.283819</td>
      <td>-0.723201</td>
      <td>0.023374</td>
      <td>-0.087338</td>
    </tr>
    <tr>
      <th>2621</th>
      <td>6.421476</td>
      <td>15.526773</td>
      <td>-8.649424</td>
      <td>-0.515494</td>
      <td>-5.827684</td>
      <td>11.032611</td>
      <td>4.230049</td>
      <td>0.767248</td>
      <td>-1.900872</td>
      <td>2.261519</td>
      <td>-0.205350</td>
      <td>0.260916</td>
      <td>0.988150</td>
      <td>0.751180</td>
      <td>-0.729725</td>
      <td>0.119440</td>
      <td>1.403213</td>
      <td>0.575325</td>
      <td>0.108833</td>
      <td>0.444062</td>
    </tr>
  </tbody>
</table>
<p>2622 rows × 20 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-be707f50-c778-410f-ba40-ef7c3a8ad1e3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-be707f50-c778-410f-ba40-ef7c3a8ad1e3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-be707f50-c778-410f-ba40-ef7c3a8ad1e3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-8850916c-b449-4f21-88c9-a7447d46d9d6">
      <button class="colab-df-quickchart" onclick="quickchart('df-8850916c-b449-4f21-88c9-a7447d46d9d6')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-8850916c-b449-4f21-88c9-a7447d46d9d6 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

  <div id="id_59b3c925-5712-4e51-85b9-f3d2bc98d41f">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('X_sm')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_59b3c925-5712-4e51-85b9-f3d2bc98d41f button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('X_sm');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
y_sm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PerformanceRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2617</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2618</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2619</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2620</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2621</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>2622 rows × 1 columns</p>
</div><br><label><b>dtype:</b> int64</label>



###3.Split training and testing data


```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_sm,y_sm,random_state=42,test_size=0.20)
# Check shape of train and test
X_train.shape, X_test.shape, y_train.shape, y_test.shape

```




    ((2097, 20), (525, 20), (2097,), (525,))



###4.Modeling XGBoost

Training and cross-validation model


```python
import xgboost
xgb = xgboost.XGBClassifier(n_estimators=300,
                            criterion="gini",
                            max_depth=11,
                            max_feature=None,
                            random_state=42,
                            objective="multi:softmax")
```


```python
kfold = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(xgb, X_train, y_train, cv=kfold)
```


```python
scores
```




    array([0.92619048, 0.92142857, 0.92840095, 0.94749403, 0.94749403])




```python
scores.mean()
```




    np.float64(0.9342016138197522)




```python
xgb.fit(X_train, y_train)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, criterion=&#x27;gini&#x27;, device=None,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, feature_types=None, gamma=None,
              grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=11, max_feature=None,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=300,
              n_jobs=None, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>XGBClassifier</div></div><div><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, criterion=&#x27;gini&#x27;, device=None,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, feature_types=None, gamma=None,
              grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=11, max_feature=None,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=300,
              n_jobs=None, ...)</pre></div> </div></div></div></div>




```python
# Prediction on testing data
xgb_test_predict = xgb.predict(X_test)

# Prediction on training data
xgb_train_predict = xgb.predict(X_train)
```


```python
xgb_train_accuracy = accuracy_score(xgb_train_predict,y_train)
print("Training accuracy of XGBoost",xgb_train_accuracy)
print("Classification report of training: \n",classification_report(xgb_train_predict,y_train))
```

    Training accuracy of XGBoost 1.0
    Classification report of training: 
                   precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       690
               1       1.00      1.00      1.00       701
               2       1.00      1.00      1.00       706
    
        accuracy                           1.00      2097
       macro avg       1.00      1.00      1.00      2097
    weighted avg       1.00      1.00      1.00      2097
    


The model performs perfectly on the training set, with no errors.

This is a sign of overfitting – that is, the model learns too much from the training data, and may not generalize well to new data.


```python
xgb_test_accuracy = accuracy_score(xgb_test_predict,y_test)
print("Testing accuracy of XGBoost",xgb_test_accuracy*100)
print("Precision Score:", precision_score(xgb_test_predict, y_test, average='weighted')*100)
print("Classification report of testing: \n",classification_report(xgb_test_predict,y_test))
```

    Testing accuracy of XGBoost 95.42857142857143
    Precision Score: 95.46733873242687
    Classification report of testing: 
                   precision    recall  f1-score   support
    
               0       0.96      0.95      0.95       185
               1       0.92      0.95      0.94       169
               2       0.98      0.96      0.97       171
    
        accuracy                           0.95       525
       macro avg       0.95      0.95      0.95       525
    weighted avg       0.95      0.95      0.95       525
    


Despite the overfitting, the model still performs very well on the test data.

High accuracy (>95%) and F1-scores above 0.94 for all classes.

No classes are missed or too weak, the classification is quite balanced.
