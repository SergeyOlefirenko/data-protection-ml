# Introduction
## 1. Why I Chose This Project/Idea
#### The choice of this idea is based on the need to use new methods of data protection in future work projects. This idea allows us to develop skills in data encryption and decryption using machine learning algorithms, which can be applied in the future for developing payment system algorithms.

## 2. What I Did, The Process
#### This project is an application for analyzing and verifying data from a CSV file related to insurance payouts. It also provides data encryption and decryption functionality based on machine learning algorithms, with a graphical interface for loading, processing, and displaying data to the user.

##### 2.1. DataProtectionApp: 

This class defines the GUI application. It includes methods for loading a CSV file, displaying seaborn plots, and validating data. The user interface consists of labels, buttons, and a text widget for displaying results.

##### 2.2. load_file(): 

This method is triggered when the user clicks the "Browse" button. It opens a dialog for selecting a CSV file. If a file is selected, the method updates the UI label with the chosen file path.

##### 2.3. display_seaborn_plots(): 

This method displays seaborn plots for data visualization. Specifically, it shows a histogram of the 'age' column from the loaded dataset.

##### 2.4. validate_data():

This method checks the loaded data. It verifies if a file is selected and if the input checksum code is correct, loads the data, preprocesses it, splits it into features and target values, displays seaborn plots, performs linear regression on the original data, encrypts the data using matrix multiplication, decrypts the data, checks data integrity, and finally displays the results on the user interface.

##### Setting up the main window and main loop: 

The main window is created using tkinter, an instance of DataProtectionApp is created, and the main event loop is started.

### Auxiliary functions for loading, preprocessing, splitting, and decrypting data:

##### 2.5. def preprocess_data(df):

### This function performs the following steps:

##### 2.5.1. Renames columns in the dataframe for convenience.
##### 2.5.2. Converts the 'age' column to int64 data type.
##### 2.5.3. Applies the get_dummies method to encode the categorical 'gender' variable.
##### 2.5.4. Checks for the presence of 'gender_female' and 'gender_male' columns in the original data and adds them with default values if they are missing.
##### 2.5.5. Removes duplicate rows from the dataframe.
##### 2.5.6. Returns the processed dataframe.

*This process prepares the data for prediction, ensuring the correct format and data types, and accounts for the absence of some categories that may be encountered in the data.*

##### 2.6. Maschine learning:

##### 2.6.1. Linear regression on original data:

lin_reg = LinearRegression()
lin_reg.fit(features, target)
r2_original = r2_score(target, lin_reg.predict(features))

##### 2.6.2. Data encryption using matrix multiplication:

attr_adder = MatrixMultiplication(multi=True)
new_features, matrix = attr_adder.transform(features, matrix=True)

##### 2.6.3. Data decryption:

decrypted_features = decryptor(new_features, matrix, features.columns, checksum)

##### 2.6.4. Linear regression on encrypted data:

lin_reg_rnd = LinearRegression()
lin_reg_rnd.fit(new_features, target)
predict_rnd = lin_reg_rnd.predict(new_features)
r_2_rnd = r2_score(target, predict_rnd)

##### 2.6.5. Linear regression on decrypted data:

lin_reg_decrypted = LinearRegression()
lin_reg_decrypted.fit(decrypted_features, target)
predict_decrypted = lin_reg_decrypted.predict(decrypted_features)
r_2_decrypted = r2_score(target, predict_decrypted)

##### 2.6.6. Splitting data into training and testing sets:

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

##### 2.6.7. Linear regression on training and testing sets:

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2_pred = r2_score(y_test, y_pred)

##### 2.6.8. Encrypting training and testing sets of data:

X_train_tr, matrix_1 = attr_adder.transform(X_train, matrix=True)
X_test_tr = X_test @ matrix_1

##### 2.6.9. Linear regression on encrypted training and testing sets of data:

model_tr = LinearRegression()
model_tr.fit(X_train_tr, y_train)
y_pred_tr = model_tr.predict(X_test_tr)
r2_tr = r2_score(y_test, y_pred_tr)

##### 2.6.10. Decrypting the testing dataset:

X_test_decr = decryptor(X_test_tr, matrix_1, features.columns, checksum)

### This code performs the following machine learning operations:

##### 1. Training models;
##### 2. Encrypting and decrypting data;
##### 3. Evaluating the quality of models on encrypted and decrypted data.


## 3. Description of the Obtained Result

|                   | count |    mean |         std |   50% |   75% |     max |
|-------------------|-------|---------|-------------|-------|-------|---------|
| age               |  10.0 |    35.3 |    7.972871 |  34.0 |  39.5 |    50.0 |
| salary            |  10.0 | 73100.0 | 20360.364546| 69500.0| 78750.0| 120000.0|
| family_members    |  10.0 |     3.1 |    1.197219 |   3.0 |   4.0 |     5.0 |
| insurance_payments|  10.0 |   334.0 |   92.159282 | 305.0 | 387.5 |   500.0 |

### Summary Statistics for the Main Columns of the Dataset:

##### count: 
*Number of samples (10 for all parameters).*
##### mean: 
*Mean value (mean age is 35.3 years).*
##### std: 
*Standard deviation (for age, it is 7.97).*
##### min, 25%, 50%, 75%, max: 
*Minimum, first quartile, median, third quartile, and maximum values respectively for each parameter. Quartile values (25%, 50%, 75%).*

### Coefficients of Determination (R²):

Coefficient of Determination: 0.8238609574330558

Coefficient of Determination (encrypted data): 0.8238609574330472

Coefficient of Determination (decrypted data): 0.8365445371678424

*The coefficient of determination (R²) indicates the proportion of variance in the dependent variable explained by the model. Values close to 1 indicate a good fit of the model to the data:*

##### Coefficient of Determination: 
For the original data — 0.8238609574330558.

##### Coefficient of Determination (encrypted data): 
After data encryption — 0.8238609574330472, which is very close to the original value, indicating that encryption did not affect the model.

##### Coefficient of Determination (decrypted data): 
After data decryption — 0.8365445371678424, which is slightly higher, possibly due to rounding during decryption.

### Information on Samples

Number of training samples: 7
Number of validation samples: 3
Total number of samples: 10
Number of prepared samples (checksum): 10

### Information on Data Splitting into Training and Testing Sets:

Number of training samples: 7.
Number of validation samples: 3.
Total number of samples: 10 (all data used).
Number of prepared samples (checksum): 10 (confirmation that all data is accounted for).

### Coefficients of Determination for the Testing Set

Coefficient of Determination (test data): -46.85153781939563

Coefficient of Determination (encrypted test data): -46.851537819819114

*These R² coefficients for the testing set indicate how well the model predicts data not used during training:*

Coefficient of Determination (test data): -46.85153781939563. 

*A negative value indicates that the model poorly predicts the test data.*

Coefficient of Determination (encrypted test data): -46.851537819819114, which is equal to the previous value, indicating that encryption did not affect the model.

### Checksum Verification and Matches

***Checksum for verification: 3***

***Number of matches: At least 3***

|                | Count |
|----------------|-------|
| age            |     3 |
| salary         |     3 |
| family_members |     3 |
| gender_female  |     3 |
| gender_male    |     3 |


### Decryption Process:

The message "Decryption started..." indicates the beginning of the decryption process.
The message "Decryption completed..." indicates its completion.

### Checksum:
"Actual checksum: 10" and "Expected checksum: 3" show that the actual checksum value does not match the expected one, causing an error: "Error: Checksum mismatch. Expected: 3, Actual: 10".
A checksum mismatch indicates that the data was corrupted or altered during transmission or storage.

### Statistical Description of Data:

A table with parameters (age, salary, family_members, insurance_payments, gender_female) shows a statistical description of the data, including the count, mean, standard deviation (std), median (50%), third quartile (75%), and maximum (max) values.
### Coefficients of Determination:
Coefficient of Determination: for the original data — 0.8238609574330558 shows (R²) for the original data, indicating a high degree of explained variance. 

### Values close to 1 indicate a good fit of the model.

***Coefficient of Determination (encrypted data)***:
After data encryption — 0.8238609574330472, which is very close to the original value, indicating that encryption did not affect the model.

***Coefficient of Determination (decrypted data)***:
After data decryption — 0.8365445371678424, which is slightly higher, possibly due to rounding during decryption.

## 4. Difficulties and Obstacles

##### Coefficient of Determination (test data): -46.85153781939563 and Coefficient of Determination (encrypted test data): -46.851537819819114 
indicate extremely poor model performance on test data, which might suggest an issue with the modeling or the data. 

***However, the value -46.851537819819114 is equal to the previous value, indicating that encryption did not affect the model.***

#### Second Decryption Process:

Re-running the decryption shows that the actual checksum matches the expected one:

"Actual checksum: 3" and "Expected checksum: 3".

The message "Checksum for verification: 3" confirms that the checksum matches.

## 5. Conclusion - Possible Improvements to the Project in the Future or How It Can Be Used

### Necessary Improvements:

##### 1. Achieving a coefficient of determination (R²) close to 1.
##### 2. Eliminating negative values of the coefficient of determination (test data) for accurate prediction on test data.
##### 3. Potential Uses of the Project: 
*The methodology described in this project can serve as a foundation for future projects related to secure data transmission.*


