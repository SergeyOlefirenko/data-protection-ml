import pandas as pd
import numpy as np


def load_data(file_path):
    return pd.read_csv(file_path)


def split_data(data, target_column):
    return data.drop(columns=[target_column], axis=1), data[target_column]


def preprocess_data(df):
    df.columns = ['gender', 'age', 'salary', 'family_members', 'insurance_payments']
    df['age'] = df['age'].astype('int64')
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)

    # If the 'gender_female' and 'gender_male' column is missing in the original data, we add it.
    if 'gender_female' not in df.columns:
        df['gender_female'] = 0  # Here we can set any default value.

    if 'gender_male' not in df.columns:
        df['gender_male'] = 0  # Here we can set any default value.

    df.drop_duplicates(inplace=True)
    return df


class MatrixMultiplication():
    def __init__(self, multi=True):
        self.multi = multi

    def fit(self, X, y=None):
        return self

    def transform(self, X, matrix=True, y=None):
        matrix_rnd = np.random.rand(X.shape[1], X.shape[1]) * 42
        X_new = X @ matrix_rnd
        if matrix:
            return X_new, matrix_rnd
        else:
            return X_new

    # Data cleaning function for prediction preparation

    def preprocess_data(df):
        df.columns = ['gender', 'age', 'salary', 'family_members', 'insurance_payments']
        df['age'] = df['age'].astype('int64')
        df = pd.get_dummies(df, columns=['gender'], drop_first=True)

        # Save the count of unique values for gender_male and gender_female.
        num_male_values = df['gender_male'].sum()
        num_female_values = df['gender_female'].sum()

        # If the 'gender_female' and 'gender_male' column is missing in the source data, we add it.
        if 'gender_female' not in df.columns:
            df['gender_female'] = 0  # Here we can set any default value.

        if 'gender_male' not in df.columns:
            df['gender_male'] = 0  # Here we can set any default value.

        df.drop_duplicates(inplace=True)
        return df, num_male_values, num_female_values

    def decryptor(X_new, matrix, column_names, checksum, num_male_values, num_female_values):
        X = X_new.dot(np.linalg.inv(matrix))
        X.columns = column_names
        X = X.round().astype('int64')

        actual_checksum = X.shape[0]
        if actual_checksum != checksum:
            print(
                f"Error: Checksum does not match. Expected value:: {checksum}, Actual value: {actual_checksum}")

        # Filter only the correct quantity of values for gender_male and gender_female.
        decrypted_gender_male = X['gender_male'].head(num_male_values)
        decrypted_gender_female = X['gender_female'].head(num_female_values)

        # Сreate a DataFrame to append to the original data.
        decrypted_gender_data = pd.DataFrame({'gender_male': decrypted_gender_male,
                                              'gender_female': decrypted_gender_female})

        # Аdd the columns gender_male and gender_female to the original data.
        X = pd.concat([X, decrypted_gender_data], axis=1)

        return X


def decryptor(X_new, matrix, column_names, checksum):
    X = X_new.dot(np.linalg.inv(matrix))
    X.columns = column_names
    X = X.round().astype('int64')

    actual_checksum = X.shape[0]
    if actual_checksum != checksum:
        print(
            f"Error: Checksum does not match. Expected value.: {checksum}, Actual value: {actual_checksum}")

    return X


def decryptor(X_new, matrix, column_names, checksum):
    print("Decryption started...")
    X = X_new.dot(np.linalg.inv(matrix))
    print("Decryption completed...")
    X.columns = column_names
    X = X.round().astype('int64')

    actual_checksum = X.shape[0]
    print("Actual checksum:", actual_checksum)
    print("Expected checksum:", checksum)  # Add output of the expected checksum.
    if actual_checksum != checksum:
        print(
            f"Error: Checksum does not match. Expected value.: {checksum}, Actual value: {actual_checksum}")

    return X

#
# def decryptor(X_new, matrix, column_names, checksum, gender):
#     print("Decryption started...")
#     X = X_new.dot(np.linalg.inv(matrix))
#     print("Decryption completed...")
#     X.columns = column_names
#     X = X.round().astype('int64')
#
#     actual_checksum = X.shape[0]
#     print("Actual checksum:", actual_checksum)
#     print("Expected checksum:", checksum)  # Добавляем вывод ожидаемой контрольной суммы
#     if actual_checksum != checksum:
#         print(
#             f"Ошибка: Контрольная сумма не совпадает. Ожидаемое значение: {checksum}, Фактическое значение: {actual_checksum}")
#
#     # Filter decrypted features by gender
#     decrypted_gender_features = X[X['gender_' + gender]]
#
#     return decrypted_gender_features
