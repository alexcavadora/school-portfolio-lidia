import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout, QPushButton, QComboBox, QWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class ParetoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.data = {}      # Dictionary to hold data for each run
        self.highlight_run = None

    def initUI(self):
        self.setWindowTitle("Pareto Front Visualizer")
        self.setGeometry(100, 100, 800, 600)
        
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        layout = QVBoxLayout(self.centralWidget)
        
        self.loadButton = QPushButton("Load Data", self)
        self.loadButton.clicked.connect(self.load_data)
        layout.addWidget(self.loadButton)
        
        self.runSelector = QComboBox(self)
        layout.addWidget(self.runSelector)
        
        self.plotButton = QPushButton("Plot Data", self)
        self.plotButton.clicked.connect(self.update_plots)
        layout.addWidget(self.plotButton)
        
        # Create a figure with two subplots (left: parallel coordinates; right: Pareto front)
        self.figure, self.axs = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
    
    def load_data(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not folder_path:
            return
        
        self.data.clear()
        self.runSelector.clear()
        
        # Look for files ending with "ps"
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".ps"):
                run_name = file_name.replace(".ps", "")
                var_file = os.path.join(folder_path, file_name)
                fun_file = os.path.join(folder_path, run_name + ".pf")
                
                if os.path.exists(fun_file):
                    # Load decision variables and objective functions
                    variables = np.loadtxt(var_file, delimiter=',')
                    functions = np.loadtxt(fun_file, delimiter=',')
                    self.data[run_name] = (variables, functions)
        
        if self.data:
            self.runSelector.addItems(self.data.keys())
            self.highlight_run = self.runSelector.currentText()
            self.update_plots()  # Automatically plot the first dataset
    
    def is_dominated(self, point, other_points):
        """Check if a point is dominated by any other point in the set"""
        # For minimization problems (lower values are better)
        return np.any(np.all(other_points <= point, axis=1) & np.any(other_points < point, axis=1))
    
    def find_pareto_front(self, functions):
        """Find the non-dominated solutions (Pareto front)"""
        n_points = functions.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            # Check if point i is dominated by any other point
            if is_pareto[i]:
                is_pareto[i] = not self.is_dominated(functions[i], functions[np.arange(n_points) != i])
        
        return is_pareto
    
    def update_plots(self):
        if not self.data:
            return
        
        # Clear previous plots
        self.axs[0].clear()
        self.axs[1].clear()
        
        # Get selected run's data
        self.highlight_run = self.runSelector.currentText()
        variables, functions = self.data[self.highlight_run]
        
        # ==============================
        # Compute Quantiles for Decision Space
        # (Using a simple sum of objectives as a proxy for performance)
        # ==============================
        dominance_sums = np.sum(functions, axis=1)
        sorted_indices = np.argsort(dominance_sums)
        n = len(sorted_indices)
        
        # Handle edge cases with small datasets
        q5_idx = sorted_indices[min(int(0.05 * n), n-1)]
        q50_idx = sorted_indices[min(int(0.50 * n), n-1)]
        q95_idx = sorted_indices[min(int(0.95 * n), n-1)]
        quantile_indices = [q5_idx, q50_idx, q95_idx]
        
        quantile_variables = variables[quantile_indices, :]
        
        # ==============================
        # Plot Parallel Coordinates (Decision Space)
        # ==============================
        # Normalize variables across the entire run (for fair comparison)
        min_vals = np.min(variables, axis=0)
        max_vals = np.max(variables, axis=0)
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Set to 1 to avoid division by zero
        
        norm_quantile = (quantile_variables - min_vals) / range_vals
        num_vars = variables.shape[1]
        x = np.arange(num_vars)
        
        line_styles = ['dashed', 'solid', 'dotted']
        quantile_labels = ['5th Percentile', '50th Percentile', '95th Percentile']
        colors = ['red', 'blue', 'green']
        
        for i in range(3):
            self.axs[0].plot(x, norm_quantile[i], linestyle=line_styles[i], linewidth=2, 
                             color=colors[i], label=quantile_labels[i])
        
        self.axs[0].set_xticks(x)
        self.axs[0].set_xticklabels([f'Var {i+1}' for i in range(num_vars)])
        self.axs[0].set_ylabel('Normalized Value')
        self.axs[0].set_title('Parallel Coordinates (Decision Space Quantiles)')
        self.axs[0].set_ylim(0, 1)  # Set y-axis limits for normalized values
        self.axs[0].legend(fontsize=9)
        
        # ==============================
        # Plot Pareto Front (Objective Space)
        # ==============================
        # Identify the Pareto front
        is_pareto = self.find_pareto_front(functions)
        pareto_functions = functions[is_pareto]
        
        # Sort the pareto front solutions by the first objective for a cleaner plot
        if len(pareto_functions) > 0:
            sorted_indices_pf = np.argsort(pareto_functions[:, 0])
            sorted_pareto = pareto_functions[sorted_indices_pf]
            
            # Plot the full solution set
            self.axs[1].scatter(functions[:, 0], functions[:, 1], color='lightgray', 
                               alpha=0.4, label='All Solutions')
            
            # Plot the Pareto front (scatter + line)
            self.axs[1].scatter(sorted_pareto[:, 0], sorted_pareto[:, 1], 
                               color='blue', alpha=0.1, label='Pareto Front')
            self.axs[1].plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 
                            color='black', alpha=0.6)
            
            # Highlight quantile solutions on the objective space
            q_solutions = functions[quantile_indices]
            marker_styles = ['o', 's', '^']  # circle, square, triangle
            
            for i in range(3):
                self.axs[1].scatter(q_solutions[i, 0], q_solutions[i, 1], 
                                   color=colors[i], s=100, marker=marker_styles[i],
                                   label=quantile_labels[i])
        else:
            # If no Pareto front found, just plot all solutions
            self.axs[1].scatter(functions[:, 0], functions[:, 1], color='gray', 
                               alpha=0.6, label='Solutions')
        
        self.axs[1].set_xlabel('Objective 1')
        self.axs[1].set_ylabel('Objective 2')
        self.axs[1].set_title('Pareto Front with Quantile Markers')
        self.axs[1].legend(fontsize=9)
        
        # Adjust layout
        self.figure.tight_layout()
        
        # Redraw canvas
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ParetoGUI()
    gui.show()
    sys.exit(app.exec_())