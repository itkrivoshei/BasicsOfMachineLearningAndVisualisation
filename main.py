import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.linear_model import LinearRegression  # For performing linear regression
from sklearn.metrics import mean_squared_error, r2_score  # For evaluating the regression model
import matplotlib.pyplot as plt  # For creating plots and visualizations
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # For embedding Matplotlib plots in Tkinter
import tkinter as tk  # For creating graphical user interfaces
from tkinter import filedialog, messagebox  # For file dialog and message boxes in Tkinter
import numpy as np  # For numerical operations (used for sorting indices in this context)


# Function to load the CSV data file
def load_data(file_path):
  # Read the CSV file into a pandas DataFrame
  data = pd.read_csv(file_path)
  # Print the first few rows of the DataFrame for verification
  print(data.head())
  # Print summary statistics for the relevant columns
  print(data[['Experience (Years)', 'Salary']].describe())
  # Return the DataFrame
  return data


# Function to train the linear regression model
def train_model(data):
  # Select the feature (Years of Experience) and the target (Salary)
  X = data[['Experience (Years)']]
  y = data['Salary']
  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=42)
  # Initialize the Linear Regression model
  model = LinearRegression()
  # Train the model using the training data
  model.fit(X_train, y_train)
  # Predict the salaries for the test data
  y_pred = model.predict(X_test)
  # Calculate Mean Squared Error (MSE) and R-squared (R2) metrics
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  # Return the model and the test data along with performance metrics
  return model, X_test, y_test, y_pred, mse, r2


# Function to plot the actual vs predicted salaries
def plot_results(X_test, y_test, y_pred):
  # Create a new figure and axis for the plot
  fig, ax = plt.subplots(figsize=(10, 6))
  # Plot the actual salaries as blue dots
  ax.scatter(X_test, y_test, color='blue', label='Actual Salary')
  # Plot the predicted salaries as red dots
  ax.scatter(X_test, y_pred, color='red', label='Predicted Salary')
  # Sort the values for a cleaner line plot
  sorted_idx = np.argsort(X_test['Experience (Years)'])
  X_test_sorted = X_test.iloc[sorted_idx]
  y_pred_sorted = y_pred[sorted_idx]
  # Plot the regression line
  ax.plot(X_test_sorted,
          y_pred_sorted,
          color='green',
          linewidth=2,
          label='Regression Line')
  # Set the labels and title for the plot
  ax.set_xlabel('Years of Experience')
  ax.set_ylabel('Salary')
  ax.set_title('Years of Experience vs Salary')
  # Add a legend to the plot
  ax.legend()
  # Enable the grid
  ax.grid(True)
  # Return the figure
  return fig


# Function to handle the data loading via GUI
def on_load_data():
  # Open a file dialog to select the CSV file
  file_path = filedialog.askopenfilename()
  if file_path:
    # Load the data from the selected file
    data = load_data(file_path)
    # Show a message box indicating successful data loading
    messagebox.showinfo("Data Loaded", "Data loaded successfully!")
    # Store the loaded data in a global variable
    global loaded_data
    loaded_data = data


# Function to handle the model training and results display via GUI
def on_train_model():
  if loaded_data is not None:
    # Train the model and get the results
    model, X_test, y_test, y_pred, mse, r2 = train_model(loaded_data)
    # Plot the results
    fig = plot_results(X_test, y_test, y_pred)
    # Show a message box with the performance metrics
    messagebox.showinfo(
        "Model Trained",
        f"Model trained successfully!\nMSE: {mse}\nR-squared: {r2}")

    # Display the plot in the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
  else:
    # Show an error message if the data is not loaded
    messagebox.showerror("Error", "Please load the data first!")


# Set up the GUI window
window = tk.Tk()
window.title("Regression Model - Salary Prediction")
window.geometry("800x600")

# Create a button to load the data
btn_load_data = tk.Button(window, text="Load Data", command=on_load_data)
btn_load_data.pack(pady=20)

# Create a button to train the model
btn_train_model = tk.Button(window, text="Train Model", command=on_train_model)
btn_train_model.pack(pady=20)

# Initialize the variable to store the loaded data
loaded_data = None

# Run the tkinter main loop to display the GUI
window.mainloop()
