
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from utils import load_data, split_data, preprocess_data, MatrixMultiplication, decryptor

class DataProtectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Protection App")

        # UI Elements
        self.file_label = tk.Label(root, text="Choose CSV File:")
        self.file_label.pack()

        self.file_button = tk.Button(root, text="Browse", command=self.load_file)
        self.file_button.pack()

        self.checksum_label = tk.Label(root, text="Enter Checksum for Verification:")
        self.checksum_label.pack()

        self.checksum_entry = tk.Entry(root)
        self.checksum_entry.pack()

        self.validate_button = tk.Button(root, text="Validate", command=self.validate_data)
        self.validate_button.pack()

        self.result_text = tk.Text(root, height=20, width=100)
        self.result_text.pack()

    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            self.file_label.config(text=f"Selected File: {self.file_path}")
        else:
            messagebox.showwarning("Warning", "No file selected!")

    def display_seaborn_plots(self, df):
        sns.set_style("darkgrid")
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        sns.histplot(df['age'], kde=True, bins=20, color='blue')
        plt.title('Age Distribution')
        plt.tight_layout()
        plt.show()

    def validate_data(self):
        if not hasattr(self, 'file_path'):
            messagebox.showwarning("Warning", "Please select a file first!")
            return

        checksum = self.checksum_entry.get()
        if not checksum.isdigit():
            messagebox.showwarning("Warning", "Checksum must be an integer!")
            return

        checksum = int(checksum)
        # Load and process data
        df = load_data(self.file_path)
        df = preprocess_data(df)
        features, target = split_data(df, 'insurance_payments')

        # Display seaborn plots
        self.display_seaborn_plots(df)

        # Linear regression on original data
        lin_reg = LinearRegression()
        lin_reg.fit(features, target)
        r2_original = r2_score(target, lin_reg.predict(features))

        # Encrypt data
        attr_adder = MatrixMultiplication(multi=True)
        new_features, matrix = attr_adder.transform(features, matrix=True)

        # Decrypt data
        decrypted_features = decryptor(new_features, matrix, features.columns, checksum)

        # Verify data
        matches = (features == decrypted_features).sum(axis=0)
        result = (matches >= checksum).all()
        # Display results in tkinter
        self.result_text.insert(tk.END, f"R2 on original data: {r2_original}\n")
        self.result_text.insert(tk.END, f"Checksum verification result: {result}\n")
        self.result_text.insert(tk.END, f"Matches:\n{matches.to_string()}\n")
        # Display results in PyCharm and seaborn graphs
        print(df.describe().T)
        df.hist(bins=50, figsize=(20, 15), edgecolor='black', linewidth=2)
        plt.show()
        # Linear regression on original data
        lin_reg = LinearRegression()
        lin_reg.fit(features, target)
        predict = lin_reg.predict(features)
        r_2 = r2_score(target, predict)
        print(f'Coefficient of determination: {r_2}')

        # Linear regression on encrypted data
        lin_reg_rnd = LinearRegression()
        lin_reg_rnd.fit(new_features, target)
        predict_rnd = lin_reg_rnd.predict(new_features)
        r_2_rnd = r2_score(target, predict_rnd)
        print(f'Coefficient of determination (encrypted data): {r_2_rnd}')

        # Linear regression on decrypted data
        lin_reg_decrypted = LinearRegression()
        lin_reg_decrypted.fit(decrypted_features, target)
        predict_decrypted = lin_reg_decrypted.predict(decrypted_features)
        r_2_decrypted = r2_score(target, predict_decrypted)
        print(f'Coefficient of determination (decrypted data): {r_2_decrypted}')

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

        # Linear regression on training and testing sets
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_pred = r2_score(y_test, y_pred)
        print(f'Coefficient of determination (test data): {r2_pred}')

        # Encrypt training and testing sets
        X_train_tr, matrix_1 = attr_adder.transform(X_train, matrix=True)
        X_test_tr = X_test @ matrix_1

        # Linear regression on encrypted training and testing sets
        model_tr = LinearRegression()
        model_tr.fit(X_train_tr, y_train)
        y_pred_tr = model_tr.predict(X_test_tr)
        r2_tr = r2_score(y_test, y_pred_tr)
        print(f'Coefficient of determination (encrypted test data): {r2_tr}')

        # Decrypt testing set
        X_test_decr = decryptor(X_test_tr, matrix_1, features.columns, checksum)

        # Print checksum
        print('Checksum:', checksum)

        print('Checksum for verification:', X_test_decr.shape[0])

        # Number of matches with the original data
        print((X_test == X_test_decr).count().to_frame(f'At least matching quantity {X_test_decr.shape[0]}'))

# Creating the main window
root = tk.Tk()
# Creating an instance of the application class
app = DataProtectionApp(root)
# Starting the main loop
root.mainloop()



