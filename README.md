# HR-Analytics-Performance-Forecasting

## 📊 Overview

This project aims to analyze employee performance data from **INX Future Inc.**, a global leader in data analytics and automation solutions. Leveraging statistical analysis, machine learning models, and HR insights, we explore the key factors affecting employee performance and build a predictive model to assist HR in decision-making.

---

## 🏢 About INX Future Inc.

INX Future Inc. (INX) is a renowned data analytics and automation company with a presence in global markets for over 15 years. The organization is consistently ranked among the **Top 20 Best Employers** for the past five years, thanks to its dynamic work culture and strong focus on employee satisfaction.

---

## ⚠️ Current Challenge: Declining Employee Performance

Despite a thriving company culture, **INX** has observed a **decline in employee performance metrics** in recent years. This concerning trend has prompted a deep-dive analysis into the root causes, without resorting to harsh disciplinary actions that might lower employee morale. The goal is to identify actionable insights and solutions.

---

## 🎯 Project Goals

1. **Identify top 3 critical factors affecting employee performance**
2. **Develop a machine learning model to predict employee performance**
3. **Recommend strategic actions to improve performance**

---

## 🧠 Project Resources

* 📘 **Jupyter Notebook (Google Colab)**:
  [View full notebook here](https://colab.research.google.com/drive/1vhvW9R7sDLYAKj1x2DKFIXYLzOltbSzG?usp=sharing)

* 📑 **Detailed Analysis Report**:
  [Access the full report here](https://www.canva.com/design/DAGm4BXrZic/XiQMjaRJ1iWhhSAYt6aUSw/edit?ui=eyJIIjp7IkEiOnRydWV9fQ&fbclid=IwY2xjawKRmZtleHRuA2FlbQIxMABicmlkETFFdjF4c0h3clFLUjdHS2h5AR7SSmyS3FDadLyVr1U_QL50fIpBGybFRDKIC-gzDbtkhKWsPAcnpGFg398WKw_aem_32ErDi3UDKvu6Gk4ZL0Ysw)

---

## 🔍 Project Workflow

### I. Import Libraries

Standard Python libraries for data analysis and modeling (e.g., pandas, numpy, matplotlib, seaborn, sklearn, xgboost)

### II. Data Import

* Load the employee performance dataset

### III. Exploratory Data Analysis (EDA)

* Overview of the dataset structure and key variables

### IV. Basic Univariate Analysis

Analyzing individual features:

* Age
* Hourly Rate
* Total and Company-specific Experience
* Gender
* Education Background
* Marital Status
* Travel Frequency
* Distance From Home
* Education Level, Job Involvement, Job Satisfaction
* Number of Companies Worked
* Overtime Status
* Salary Hike %
* Relationship Satisfaction
* Work-Life Balance
* Years Since Last Promotion
* Years with Current Manager
* Attrition Status
* Performance Rating
* Department and Job Role

### V. Two-Variable (Bivariate) Analysis

* Age vs. Total Experience
* Company Experience vs. Total Experience
* Salary Hike % vs. Companies Worked
* Promotion Timeline vs. Role Experience
* Hourly Rate vs. Manager Tenure
* Commute Distance vs. Salary Hike %

### VI. Categorical Feature Analysis

* Breakdown and comparison of categorical variables by performance

### VII. Discrete Feature Analysis

* Frequency distribution and performance impact

### VIII. Multivariate Analysis

* Correlation and interaction between multiple variables

### IX. Key Performance Drivers

* Identification of top factors influencing performance using feature importance techniques

### X. Data Preprocessing

* Handling missing values
* Encoding categorical variables
* Checking for duplicates
* Detecting skewness
* Outlier detection and treatment

### XI. Predictive Modeling

* Defining dependent and independent variables
* Upsampling to balance the dataset
* Train-test data split
* Model training using **XGBoost**
* Performance evaluation using appropriate metrics (e.g., accuracy, precision, recall)

---

## 🤖 Machine Learning Outcome

A trained **XGBoost model** capable of **predicting employee performance**, helping HR to proactively identify high and low performers and take appropriate measures.

### 🔹 Training Results of XGBoost 1.0

| Class | Precision | Recall | F1-score | Support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 1.00      | 1.00   | 1.00     | 690     |
| 1     | 1.00      | 1.00   | 1.00     | 701     |
| 2     | 1.00      | 1.00   | 1.00     | 706     |

| Metric       | Score |
| ------------ | ----- |
| Accuracy     | 1.00  |
| Macro Avg    | 1.00  |
| Weighted Avg | 1.00  |

---

### 🔹 Testing Results of XGBoost 1.0

| Class | Precision | Recall | F1-score | Support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.96      | 0.95   | 0.95     | 185     |
| 1     | 0.92      | 0.95   | 0.94     | 169     |
| 2     | 0.98      | 0.96   | 0.97     | 171     |

| Metric          | Score  |
| --------------- | ------ |
| Accuracy        | 0.9543 |
| Precision Score | 0.9547 |
| Macro Avg       | 0.95   |
| Weighted Avg    | 0.95   |


---

## 📝 Key Insights & Recommendations

1. **Top 3 Factors Influencing Performance**:

   * Job Involvement
   * Last Salary Hike %
   * Work Life Balance

2. **Recommendations**:

* **Enhance employee environment satisfaction to improve overall performance**

  * A positive and supportive work environment is essential for boosting employee performance.
  * The company should invest more in creating a comfortable, inclusive, and motivating workplace culture.

* **Implement regular salary hikes to encourage better performance**

  * A well-structured salary increase system serves as a strong motivator for employees to perform well.
  * Ensure that salary hikes are performance-based and transparent to promote fairness and morale.

* **Promote employees every 6 months to maintain motivation**

  * Establishing a biannual promotion policy helps set clear career paths and encourages consistent effort.
  * Use clear performance metrics and fair evaluation processes to support promotion decisions.

* **Improve employees’ work-life balance to enhance performance ratings**

  * Work-life balance has a direct impact on employee satisfaction and their overall performance.
  * Offer flexible working hours, wellness programs, and sufficient leave policies to support a healthy balance.

* **Prioritize female candidates when hiring for HR roles**

  * Data indicates that female employees tend to outperform male counterparts in Human Resources roles.
  * Companies should consider gender diversity as a strategic advantage when recruiting for HR positions.

* **Recognize higher performance in Development and Sales departments**

  * Employees in Development and Sales consistently show higher performance compared to other departments.
  * Continue to invest in these departments through training, tools, and incentive programs to maximize their potential.

* **Focus on employees with low to medium satisfaction but excellent performance**

  * A significant number of employees with low or medium scores in Job Satisfaction and Relationship Satisfaction still deliver excellent performance.
  * These employees should not be overlooked; instead, management should engage with them to understand their needs and provide tailored support to retain and motivate them.

---


## 🛠️ Tech Stack

* Python 3.x
* Pandas, NumPy, Seaborn, Matplotlib
* Scikit-learn
* XGBoost
* Jupyter Notebook

---

## 📌 Conclusion

This project provides a comprehensive HR analytics framework to analyze, understand, and predict employee performance. By combining data-driven insights with practical HR strategies, **INX Future Inc.** can improve productivity while maintaining high morale and workplace satisfaction.

