# Predictive Machine Maintenance

## 📌 Project Background
Industrial equipment failures can result in costly production downtime, with manufacturers facing losses of up to **$50,000-$100,000 per hour**. In response to these high costs, this project focuses on developing a robust predictive maintenance system that leverages machine learning to proactively analyze sensor data. 

By monitoring key performance indicators from various industrial machines, the system identifies early signs of wear and potential failure. This allows maintenance teams to intervene before critical breakdowns occur, thereby minimizing downtime, optimizing repair schedules, and significantly reducing operational costs.

- The dataset can be found [here](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020).
- The Hupyter notebook can be found [here](https://github.com/Zhonghong99/predictive_machine_maintenance/blob/main/Predictive_machine_maintenance.ipynb)

## 📊 Project Workflow
```mermaid
flowchart TB
    data[Data Collection] --> preprocess[Data Preprocessing]
    preprocess --> eda[Exploratory Data Analysis]
    eda --> feature[Feature Engineering]
    feature --> model[Model Selection & Training]
    model --> eval[Model Evaluation]
    eval --> opt[Model Optimization]
    opt --> deploy[Deployment]
    deploy --> monitor[Monitoring]
    monitor --> maintain[Maintenance]
```

## 📂 Dataset
- The dataset used is **ai4i2020.csv**, which contains various sensor readings from industrial equipment.
- Features include:
  - **Air temperature [K]**
  - **Process temperature [K]**
  - **Rotational speed [rpm]**
  - **Torque [Nm]**
  - **Tool wear [min]**
  - **Machine failure labels**

## 🛠️ Technologies Used
- **Python**
- **Pandas, NumPy** (Data Manipulation)
- **Matplotlib, Seaborn** (Visualization)
- **Scikit-learn** (Machine Learning)
- **Flask** (Web Application)

## 🔬 Key Steps
1. **Data Collection & Preprocessing**
   - Load and clean the dataset (handling missing values, removing duplicates).
   - Convert data types and define target variables.

2. **Exploratory Data Analysis (EDA)**
   - Visualize sensor data distributions and correlations.
   - Identify patterns in failure occurrences.

3. **Feature Engineering**
   - Standardization and encoding of categorical variables.
   - Selection of relevant features for modeling.

4. **Model Training & Evaluation**
   - Machine learning models trained using Scikit-learn.
   - Performance evaluation using metrics like **accuracy, precision, recall, and F1-score**.

5. **Deployment & Monitoring (Future Scope)**
   - Deploy the trained model for real-time monitoring of machines.
   - Implement an alert system for early warnings.

## 🚀 Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/Zhonghong99/predictive_machine_maintenance.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Predictive_machine_maintenance.ipynb
   ```
4. Run the application (`app.py`) on your terminal:
   - If using **VS Code**:
     ```bash
     python app.py
     ```
   - If using **PyCharm**:
     - Open `app.py` and click on the **Run** button.
   - If using a standard **terminal**:
     ```bash
     python app.py
     ```

## 📌 Future Improvements
- Integrating deep learning models for better predictions.
- Deploying as a cloud-based API for real-time monitoring.

## 📜 License
This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
