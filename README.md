# Employee_Burnout_Prediction

# Model Used
I used multiple regression models like Linear Regression, K-Nearest Neighbours and Random Forest and after evaluation Random Forest performs better with the r2 score of 89%

# What data do we have ?
- Employee ID: A unique identifier for each employee. It can be a number, a combination of numbers and letters.
- Date of Joining: The date on which the employee joined the organization. It can be in the format of YYYY-MM-DD or DD-MM-YYYY.
- Gender: The gender of the employee. It can be "Male" or "Female".
- Company Type: The type of company where the employee is working. It can be "Service" or "Product".
- WFH Setup Available: Whether the employee has access to a work from home setup. It can be "Yes" or "No".
- Designation: The job title of the employee.

# EDA(Exploratory data analysis)
The data is being analyzed to examine the observations (rows) and characteristics (columns) of the dataset. It contains both categorical and numerical features, with a majority of the categorical features having a low number of distinct values. Upon closer examination, I noticed that there are missing values in our dataset. Additionally, I was surprised to find that the Target variable, which is the Burn Rate, also has missing values.
