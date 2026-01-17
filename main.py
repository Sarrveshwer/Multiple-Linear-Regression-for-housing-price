import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.interpolate import interpn
import logging
import time
import sys
import datetime
import os
import traceback

"""
FINAL OUTPUT AFTER TESTING
Model Features:
Area Population
Avg. Area House Age
Avg. Area Income
Avg. Area Number of Rooms
mse= 0.08202714761373452
Model Correlation (R): 0.9581
R-Squared: 0.9180
"""
sns.set_style('ticks') 
#--------------Automatic Logger------------------


real_input = input


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        self.is_newline = True  


    def write(self, message):
        self.terminal.write(message)
        
        if not self.log.closed:
            for char in message:
                if self.is_newline and char != '\n':
                    # add timestamp only at the start of a non-empty line
                    timestamp = datetime.datetime.now().strftime("[%H:%M:%S] ")
                    self.log.write(timestamp)
                    self.is_newline = False
                
                self.log.write(char)
                
                if char == '\n':
                    self.is_newline = True


    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()


    def __del__(self):
        if hasattr(self, 'log') and not self.log.closed:
            self.log.close()




filename = os.path.splitext(os.path.basename(__file__))[0]
try:
    a=2 #int(input("Test(1) or Not (2): "))
    if a==1:
        os.mkdir("logs")
    else:
        os.mkdir("logs_test")
    os.mkdir("images")
except FileExistsError:
    pass
except OSError as e:
    print(f"An error occurred: {e}")


safe_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if a==1:
    log_path = os.path.join("logs", f"{filename}@{safe_timestamp}.log")
else:
    log_path = os.path.join("logs_test", f"{filename}_test@{safe_timestamp}.log")

sys.stdout = Logger(log_path)


def input_and_log(prompt=""):
    print(prompt, end="", flush=True)
    answer = real_input()
    if not sys.stdout.log.closed:
        sys.stdout.log.write(answer + "\n")
        sys.stdout.log.flush()
    return answer


input = input_and_log

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    print("\n" + "="*30 + " CRASH DETECTED " + "="*30)
    print(f"Timestamp: {datetime.datetime.now()}")
    print(error_msg)
    print("="*76 + "\n")

sys.excepthook = handle_exception
#--------------------------MODEL---------------------------

class LinearRegressionModel:
    def __init__(self):
        self.df=None
        self.feature_names=None
        self.y=None
        self.loss_history=[]
        self.y_pred=None
        self.theta=None
        self.best_epoch_found = 0
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def dataset(self,filename,column,exclude):
        print(f'\n----- Model for {filename} -----')
        self.df=pd.read_csv(filename).drop_duplicates().dropna().reset_index(drop=True)
        price_col = [col for col in self.df.columns if column.lower() in col.lower()]
        if price_col:
            self.y = self.df[price_col[0]]
        else:
            print("Could not find a price column!")
            sys.exit()
        self.m=len(self.y)
        print(f"This model contains {self.m} datapoints")
        exclude_cols = [price_col[0], 'id', 'date', 'sqft_living15', 'zipcode']+exclude
        self.feature_names = self.df.select_dtypes(include=['number']).columns.difference(exclude_cols).tolist()

    def gradient_descent_engine(self, X_scaled, y_scaled, max_epochs, val_data=None):
        m = len(y_scaled)
        Xb = np.c_[np.ones(m), X_scaled]
        theta = np.ones((Xb.shape[1],1))
        
        initial_alpha = 0.01  
        decay = 0.001 
        prev_mse = float('inf')
        best_val_mse = float('inf')
        best_theta = theta.copy()
        best_epoch = 0
        patience = 15
        counter = 0
        local_loss_history = []

        for i in range(max_epochs):
            alpha = initial_alpha * (1.0 / (1.0 + decay * i))
            y_pred_scaled = (Xb @ theta).reshape(-1) 
            e = y_pred_scaled - y_scaled
            mse = np.mean(e**2)
            local_loss_history.append(mse)
            
            gradJ = (2/m) * (Xb.T @ e.reshape(-1, 1))
            theta = theta - alpha * gradJ  
            
            # Logic for Early Stopping / Best Epoch identification
            if val_data is not None:
                X_val_scaled, y_val_scaled = val_data
                Xb_val = np.c_[np.ones(len(y_val_scaled)), X_val_scaled]
                val_mse = np.mean(((Xb_val @ theta).reshape(-1) - y_val_scaled)**2)
                
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_theta = theta.copy()
                    best_epoch = i
                    counter = 0
                else:
                    counter += 1
                if counter >= patience: break
            else:
                # Standard convergence for final training
                if abs(prev_mse - mse) < 1e-7: 
                    counter += 1
                    if counter >= patience: break
                else:
                    counter = 0
                prev_mse = mse
                best_theta = theta
                best_epoch = i

        return best_theta, local_loss_history, best_epoch

    def Linear_regression(self):
        X = self.df[self.feature_names].values
        y = self.y.values.reshape(-1, 1)

        # 1. SPLIT DATA to find best epoch and validate trends
        print("Step 1: Splitting data to identify optimal trend duration...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_tr_s = self.scaler_x.fit_transform(X_train)
        y_tr_s = self.scaler_y.fit_transform(y_train).flatten()
        X_te_s = self.scaler_x.transform(X_test)
        y_te_s = self.scaler_y.transform(y_test).flatten()

        _, _, self.best_epoch_found = self.gradient_descent_engine(X_tr_s, y_tr_s, len(y_tr_s), val_data=(X_te_s, y_te_s))
        print(f"Optimal trend identified at epoch: {self.best_epoch_found}")

        # 2. FINAL TRAINING on 100% of data
        print(f"Step 2: Training final model on 100% data for {self.best_epoch_found} epochs...")
        X_all_s = self.scaler_x.fit_transform(X)
        y_all_s = self.scaler_y.fit_transform(y).flatten()
        
        self.theta, self.loss_history, _ = self.gradient_descent_engine(X_all_s, y_all_s, self.best_epoch_found)
        
        # Calculate final predictions for analysis
        Xb_all = np.c_[np.ones(len(y_all_s)), X_all_s]
        final_y_pred_scaled = (Xb_all @ self.theta).reshape(-1, 1)
        self.y_pred = self.scaler_y.inverse_transform(final_y_pred_scaled).flatten()

        print("Model Features:", *self.feature_names, sep='\n')
        print("Final Training MSE=", self.loss_history[-1])

    def plot(self):
        sns.set_theme(style="darkgrid")
        safe_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # RESIDUAL DENSITY PLOT
        x_val = self.y_pred
        y_val = self.y - self.y_pred
        data, x_e, y_e = np.histogram2d(x_val, y_val, bins=[100, 100], density=True)
        z_val = interpn((0.5*(x_e[1:]+x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), 
                    data, np.vstack([x_val, y_val]).T, 
                    method="splinef2d", bounds_error=False)
        idx = z_val.argsort()
        x_val, y_val, z_val = x_val[idx], y_val[idx], z_val[idx]

        plt.figure(figsize=(10, 6))
        plt.scatter(x_val, y_val, c=z_val, cmap="mako", s=20, alpha=0.9, edgecolor='none')
        plt.axhline(y=0, color='cyan', linestyle='--', linewidth=2)
        plt.title("Residual Density: Trend Analysis (Full Dataset)", fontsize=15, fontweight='bold')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.savefig(os.path.join("images", f"Residual_plot_{safe_timestamp}.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # CONVERGENCE PLOT
        plt.figure(figsize=(10, 5))
        epochs = range(len(self.loss_history))
        plt.plot(epochs, self.loss_history, color='#FF4500', linewidth=2.5)
        plt.fill_between(epochs, self.loss_history, color='#FF4500', alpha=0.2)
        plt.title(f"Model Optimization Convergence (Epochs: {self.best_epoch_found})", fontsize=14, fontweight='bold')
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error")
        stats_text = (f'Final MSE: {self.loss_history[-1]:.4f}\n'
                      f'Correlation (R): {self.model_correlation:.4f}\n'
                      f'R² Score: {self.r_squared:.4f}')
        plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=1'), color='white')
        plt.savefig(os.path.join("images", f"loss_curve_{safe_timestamp}.png"), dpi=300, bbox_inches='tight')
        plt.show()

        # FEATURE IMPORTANCE
        plt.figure(figsize=(10, 8))
        importance_df = pd.DataFrame({'Feature': self.feature_names, 'Coefficient': self.theta[1:].flatten()})
        importance_df['Abs_Val'] = importance_df['Coefficient'].abs()
        importance_df = importance_df.sort_values(by='Abs_Val', ascending=False)
        sns.barplot(x='Coefficient', y='Feature', hue='Feature', data=importance_df, palette='viridis', legend=False)        
        plt.title("Feature Importance (Standardized Beta Weights)", fontsize=16, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.savefig(os.path.join("images", f"feature_importance_{safe_timestamp}.png"), dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_model(self):
        r_matrix = np.corrcoef(self.y, self.y_pred)
        self.model_correlation = r_matrix[0, 1]
        print(f"Model Correlation (R): {self.model_correlation:.4f}")
        self.r_squared = self.model_correlation**2
        print(f"R-Squared: {self.r_squared:.4f}")

    def show_feature_importance(self):
        print("\n--- Feature Importance (Beta Values) ---")
        for name, weight in zip(self.feature_names, self.theta[1:].flatten()):
            print(f"{name}: {float(weight):.4f}")
    
    def run(self,filename,column,exclude):
        self.dataset(filename,column,exclude)
        self.Linear_regression()
        self.evaluate_model()
        self.show_feature_importance()
        self.plot()

if __name__ == "__main__":
    Model=LinearRegressionModel()
    Model.run('USA_Housing.csv','Price',[])
    Model1=LinearRegressionModel()
    Model1.run('kc_house_data.csv','Price',[])
    Model2=LinearRegressionModel()
    Model2.run('realtor-data.csv','price',[])