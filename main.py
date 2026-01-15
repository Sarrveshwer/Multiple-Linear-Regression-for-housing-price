import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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
        self.y_scaled=None
        self.y_pred_scaled=None

    def dataset(self,filename):
        print(f'\n----- Model for {filename} -----')
        self.df=pd.read_csv(filename)
        price_col = [col for col in self.df.columns if 'price' in col.lower()]

        if price_col:
            self.y = self.df[price_col[0]]
        else:
            print("Could not find a price column!")
            exit()
        self.m=len(self.y)
        print(f"This model contains {self.m} datapoints")
        exclude_cols = [price_col[0], 'id', 'date', 'sqft_living15', 'zipcode']
        self.feature_names = self.df.select_dtypes(include=['number']).columns.difference(exclude_cols).tolist()

    def Linear_regression(self):
        #Making the matrix by adding all the features
        X = self.df[self.feature_names].values
        #scalling
        scaler_x = StandardScaler()
        X_scaled = scaler_x.fit_transform(X) 
        #Creating X_biased
        Xb=np.c_[np.ones(self.m),X_scaled]
        #initialize theta and scale y
        theta = np.ones((Xb.shape[1],1))
        scaler_y = StandardScaler()
        self.y_scaled = scaler_y.fit_transform(self.y.values.reshape(-1, 1)).flatten()
        
        #time for gradient decent
        initial_alpha = 0.01  
        decay = 0.001 
        best_epoch=self.m #this is after testing
        prev_mse = float('inf')
        patience = 5
        counter = 0


        for i in range(best_epoch):
            alpha = initial_alpha * (1.0 / (1.0 + decay * i))
            self.y_pred_scaled = (Xb @ theta).reshape(-1) 
            e = self.y_pred_scaled - self.y_scaled
            mse = np.mean(e**2)
            self.loss_history.append(mse)
            gradJ = (2/self.m) * (Xb.T @ e.reshape(-1, 1))
            theta = theta - alpha * gradJ  
            
            if abs(prev_mse - mse) < 1e-7: 
                counter += 1
                if counter >= patience:
                    break
            else:
                counter = 0
            prev_mse = mse
        self.theta=theta
        print("Model Features:", *self.feature_names,sep='\n')
        print("mse=",mse)


        final_y_pred_scaled_2d = (Xb @ theta).reshape(-1, 1)
        self.y_pred = scaler_y.inverse_transform(final_y_pred_scaled_2d).flatten()
        
    def plot(self):
        sns.set_theme(style="darkgrid")
        safe_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        x_val = self.y_pred
        y_val = self.y - self.y_pred
        data, x_e, y_e = np.histogram2d(x_val, y_val, bins=[100, 100], density=True)
        z_val = interpn((0.5*(x_e[1:]+x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), 
                    data, np.vstack([x_val, y_val]).T, 
                    method="splinef2d", bounds_error=False)
        idx = z_val.argsort()
        x_val, y_val, z_val = x_val[idx], y_val[idx], z_val[idx]

        plt.figure(figsize=(10, 6))
        plt.scatter(x_val, y_val, c=z_val, cmap="Greens_r", s=20, alpha=0.9, edgecolor='none')
        plt.axhline(y=0, color='cyan', linestyle='--', linewidth=2)
        plt.title("Residual Density: Dark (Dense) to Light (Sparse)", fontsize=15, fontweight='bold')
        plt.xlabel("Predicted Price")
        plt.ylabel("Residuals")
        plt.savefig(os.path.join("images", f"Residual_plot_{safe_timestamp}.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(10, 5))
        epochs = range(len(self.loss_history))
        plt.plot(epochs, self.loss_history, color='#FF4500', linewidth=2.5)
        plt.fill_between(epochs, self.loss_history, color='#FF4500', alpha=0.2)
        plt.title("Model Optimization Convergence", fontsize=14, fontweight='bold')
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

        # IMPACT ANALYSIS - FORCED COLOR CHANGE
        plt.figure(figsize=(10, 8))
        importance_df = pd.DataFrame({'Feature': self.feature_names, 'Coefficient': self.theta[1:].flatten()})
        importance_df['Abs_Val'] = importance_df['Coefficient'].abs()
        importance_df = importance_df.sort_values(by='Abs_Val', ascending=False)
        
        # This explicitly overrides any global theme settings for this specific plot
        sns.set_palette("viridis") 
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
    
    def run(self,filename):
        self.dataset(filename)
        self.Linear_regression()
        self.evaluate_model()
        self.show_feature_importance()
        self.plot()
        

if __name__ == "__main__":
    Model1 = LinearRegressionModel()
    Model1.run("USA_Housing.csv")
    Model2 = LinearRegressionModel()
    Model2.run("kc_house_data.csv")
    
