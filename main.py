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
import gc

sns.set_style('ticks') 

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
    a=2 
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
        self.current_filename = ""
        self.current_bin_name = ""
        self.dataset_folder = ""
    def z_scale(self, data, stats=None):
            if stats is None:
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                std = np.where(std == 0, 1.0, std)
                stats = {'mean': mean, 'std': std}
            
            scaled_data = (data - stats['mean']) / stats['std']
            return scaled_data.astype('float32'), stats

    def inverse_scale(self, scaled_data, stats):
        return (scaled_data * stats['std']) + stats['mean']

    def dataset(self, filename, column, exclude):
        print(f'\n----- Model for {filename} -----')
        self.current_filename = filename
        self.dataset_folder = os.path.splitext(filename)[0]
        if not os.path.exists(self.dataset_folder): os.makedirs(self.dataset_folder)
            
        self.df = pd.read_csv(filename).drop_duplicates()
        if len(self.df) > 2000000:
            print("Dataset too large, sampling 2,000,000 rows...")
            self.df = self.df.sample(n=2000000, random_state=42).reset_index(drop=True)

        actual_col = [col for col in self.df.columns if column.lower() in col.lower()][0]
        if self.df[actual_col].dtype == 'object':
            self.df[actual_col] = pd.to_numeric(self.df[actual_col], errors='coerce')
        
        self.df = self.df.dropna(subset=[actual_col]).reset_index(drop=True)
        self.df = self.df[self.df[actual_col] > 0].reset_index(drop=True)
        
        for col in self.df.select_dtypes(include=['number']).columns:
            self.df[col] = self.df[col].astype('float32')

        self.y = self.df[actual_col]
        exclude_cols = [actual_col] + exclude
        self.feature_names = self.df.select_dtypes(include=['number']).columns.difference(exclude_cols).tolist()
        gc.collect()

    def gradient_descent_engine(self, X_scaled, y_scaled, max_epochs, val_data=None):
        m = len(y_scaled)
        if m == 0: return np.ones((X_scaled.shape[1] + 1, 1)), [], 0
        Xb = np.c_[np.ones(m), X_scaled].astype('float32')
        theta = np.ones((Xb.shape[1], 1), dtype='float32')
        
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
            
            if np.isnan(mse) or np.isinf(mse):
                break

            local_loss_history.append(mse)
            gradJ = (2/m) * (Xb.T @ e.reshape(-1, 1))
            theta = theta - np.float32(alpha) * gradJ  
            
            if val_data is not None:
                X_val_scaled, y_val_scaled = val_data
                Xb_val = np.c_[np.ones(len(y_val_scaled)), X_val_scaled].astype('float32')
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
        self.df = self.df.dropna(subset=self.feature_names).reset_index(drop=True)
        if self.df.empty:
            self.loss_history = []
            return

        X = self.df[self.feature_names].values.astype('float32')
        y = self.df[self.y.name].values.reshape(-1, 1).astype('float32')

        X_scaled, self.stats_x = self.z_scale(X)
        y_scaled, self.stats_y = self.z_scale(y)

        print("Step 1: Splitting data for trend analysis...")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled.flatten(), test_size=0.2, random_state=42)
        
        _, _, self.best_epoch_found = self.gradient_descent_engine(X_train, y_train, len(y_train), val_data=(X_test, y_test))
        self.best_epoch_found = max(1, self.best_epoch_found)

        print(f"Step 2: Training on 100% data for {self.best_epoch_found} epochs...")
        self.theta, self.loss_history, _ = self.gradient_descent_engine(X_scaled, y_scaled.flatten(), self.best_epoch_found)
        
        Xb = np.c_[np.ones(len(y_scaled)), X_scaled].astype('float32')
        y_pred_scaled = (Xb @ self.theta).reshape(-1, 1)
        self.y_pred = self.inverse_scale(y_pred_scaled, self.stats_y).flatten()
        self.y = self.df[self.y.name]

        print("Model Features:", *self.feature_names, sep='\n')
        if self.loss_history:
            print(f"Final Training MSE= {self.loss_history[-1]:.6f}")
        gc.collect()

    def evaluate_model(self):
        if not hasattr(self, 'y_pred') or len(self.loss_history) == 0: return
        r_matrix = np.corrcoef(self.y, self.y_pred)
        self.model_correlation = r_matrix[0, 1]
        print(f"Model Correlation (R): {self.model_correlation:.4f}")
        self.r_squared = self.model_correlation**2
        print(f"R-Squared: {self.r_squared:.4f}")
        
        safe_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        metrics_file = os.path.join(self.dataset_folder, f"model_metrics_{safe_timestamp}.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"Dataset: {self.current_filename}\n")
            f.write(f"Bin: {self.current_bin_name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"R-Value: {self.model_correlation:.4f}\n")
            f.write(f"R-Squared: {self.r_squared:.4f}\n")
            f.write(f"Final MSE: {self.loss_history[-1]:.6f}\n")
            f.write(f"Best Epoch: {self.best_epoch_found}\n")
            f.write("\nFeature Importances (Standardized Beta Weights):\n")
            for name, weight in zip(self.feature_names, self.theta[1:].flatten()):
                f.write(f"  {name}: {float(weight):.4f}\n")
    def plot(self):
        if not hasattr(self, 'y_pred') or len(self.loss_history) == 0: return
        sns.set_theme(style="darkgrid")
        safe_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        layout = [
            ["Residuals", "Convergence"],
            ["Importance", "Actual_vs_Pred"]
        ]
        
        fig, axes = plt.subplot_mosaic(layout, figsize=(24, 14))
        fig.suptitle(f"Dashboard: {self.current_filename} {self.current_bin_name} | {safe_timestamp}", fontsize=20, fontweight='bold')

        indices = np.arange(len(self.y_pred))
        if len(self.y_pred) > 50000:
            indices = np.random.choice(len(self.y_pred), 50000, replace=False)
            
        x_pred_raw = self.y_pred[indices]
        y_actual_raw = self.y.values[indices]
        y_res_raw = y_actual_raw - x_pred_raw

        data, x_e, y_e = np.histogram2d(x_pred_raw, y_res_raw, bins=[100, 100], density=True)
        z_val = interpn((0.5*(x_e[1:]+x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), 
                    data, np.vstack([x_pred_raw, y_res_raw]).T, 
                    method="splinef2d", bounds_error=False)
        idx = z_val.argsort()
        axes["Residuals"].scatter(x_pred_raw[idx], y_res_raw[idx], c=z_val[idx], cmap="mako", s=20, alpha=0.9, edgecolor='none')
        axes["Residuals"].axhline(y=0, color='cyan', linestyle='--', linewidth=2)
        axes["Residuals"].set_title("Residual Density", fontsize=14)
        axes["Residuals"].set_xlabel("Predicted Values")
        axes["Residuals"].set_ylabel("Residuals")

        epochs = range(len(self.loss_history))
        axes["Convergence"].plot(epochs, self.loss_history, color='#FF4500', linewidth=2.5)
        axes["Convergence"].fill_between(epochs, self.loss_history, color='#FF4500', alpha=0.2)
        axes["Convergence"].set_title(f"Optimization Convergence", fontsize=14)
        axes["Convergence"].set_xlabel("Epochs")
        axes["Convergence"].set_ylabel("MSE")
        stats_text = (f'Final MSE: {self.loss_history[-1]:.4f}\n'
                      f'Correlation (R): {self.model_correlation:.4f}\n'
                      f'R² Score: {self.r_squared:.4f}')
        axes["Convergence"].text(0.95, 0.95, stats_text, transform=axes["Convergence"].transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=1'), color='white')

        importance_df = pd.DataFrame({'Feature': self.feature_names, 'Coefficient': self.theta[1:].flatten()})
        importance_df['Abs_Val'] = importance_df['Coefficient'].abs()
        importance_df = importance_df.sort_values(by='Abs_Val', ascending=False)
        sns.barplot(x='Coefficient', y='Feature', hue='Feature', data=importance_df, palette='viridis', legend=False, ax=axes["Importance"])        
        axes["Importance"].set_title("Feature Importance", fontsize=14)
        axes["Importance"].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        axes["Actual_vs_Pred"].scatter(y_actual_raw, x_pred_raw, c=z_val[idx], cmap="mako", s=20, alpha=0.9, edgecolor='none')
        lims = [np.min([axes["Actual_vs_Pred"].get_xlim(), axes["Actual_vs_Pred"].get_ylim()]), 
                np.max([axes["Actual_vs_Pred"].get_xlim(), axes["Actual_vs_Pred"].get_ylim()])]
        axes["Actual_vs_Pred"].plot(lims, lims, 'r--', alpha=0.75, zorder=3, linewidth=2)
        axes["Actual_vs_Pred"].set_title("Actual vs Predicted", fontsize=14)
        axes["Actual_vs_Pred"].set_xlabel("Actual Values")
        axes["Actual_vs_Pred"].set_ylabel("Predicted Values")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_name = f"Dashboard_{self.current_bin_name}_{safe_timestamp}.png"
        plt.savefig(os.path.join(self.dataset_folder, save_name), dpi=300, bbox_inches='tight')
        plt.close(fig)
        del fig, axes
        gc.collect()


    def show_feature_importance(self):
        if len(self.loss_history) == 0: return
        print("\n--- Feature Importance (Beta Values) ---")
        for name, weight in zip(self.feature_names, self.theta[1:].flatten()):
            print(f"{name}: {float(weight):.4f}")
    
    def run(self,filename,column,exclude):
        self.current_bin_name = "Full_Data"
        self.dataset(filename,column,exclude)
        target_name = self.y.name
        Q1 = self.df[target_name].quantile(0.25)
        Q3 = self.df[target_name].quantile(0.75)
        IQR = Q3 - Q1
        self.df = self.df[self.df[target_name] <= (Q3 + 1.5 * IQR)].reset_index(drop=True)
        self.y = self.df[target_name]
        self.Linear_regression()
        self.evaluate_model()
        self.show_feature_importance()
        self.plot()

    def segment_data(self, target_col):
        q1 = self.df[target_col].quantile(0.33)
        q2 = self.df[target_col].quantile(0.66)
        print(f"Bin Thresholds: Low < {q1:.2f}, Mid < {q2:.2f}, High >= {q2:.2f}")
        
        self.bins = {
            "Low-Tier": self.df[self.df[target_col] < q1].copy().reset_index(drop=True),
            "Mid-Tier": self.df[(self.df[target_col] >= q1) & (self.df[target_col] < q2)].copy().reset_index(drop=True),
            "High-Tier": self.df[self.df[target_col] >= q2].copy().reset_index(drop=True)
        }

    def run_with_binning(self, filename, column, exclude):
        self.dataset(filename, column, exclude)
        actual_col = [col for col in self.df.columns if column.lower() == col.lower()]
        if not actual_col:
            actual_col = [col for col in self.df.columns if column.lower() in col.lower()]
        target_name = actual_col[0]
        self.segment_data(target_name)
        original_full_df = self.df.copy()
        for name, data_segment in self.bins.items():
            if len(data_segment) < 20: 
                continue
            print(f"\n" + "="*20 + f" ANALYZING BIN: {name} " + "="*20)
            self.current_bin_name = name
            if "High" not in name:
                Q1 = data_segment[target_name].quantile(0.25)
                Q3 = data_segment[target_name].quantile(0.75)
                IQR = Q3 - Q1
                self.df = data_segment[(data_segment[target_name] >= (Q1 - 1.5 * IQR)) & (data_segment[target_name] <= (Q3 + 1.5 * IQR))].copy().reset_index(drop=True)
            else:
                self.df = data_segment.copy()
            self.y = self.df[target_name]
            self.m = len(self.y)
            self.Linear_regression()
            self.evaluate_model()
            self.show_feature_importance()
            self.plot()
        del self.bins
        self.df = original_full_df
        self.run(filename,column,exclude)
        gc.collect()

if __name__ == "__main__":
    Model=LinearRegressionModel()
    Model.run_with_binning('USA_Housing.csv','Price', [])

    Model1=LinearRegressionModel()
    Model1.run_with_binning('kc_house_data.csv','price', ['id', 'date', 'sqft_living15', 'zipcode'])
    
    Model3=LinearRegressionModel()
    Model3.run_with_binning('realtor-data.csv','price', ['status', 'street', 'city', 'state', 'prev_sold_date'])
    
    Model4=LinearRegressionModel()
    Model4.run_with_binning('nyc-rolling-sales.csv','SALE PRICE', ['id', 'date', 'zipcode', 'Unnamed: 0'])