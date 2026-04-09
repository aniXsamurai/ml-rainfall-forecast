## 🌧️ Rainfall Prediction Web Application

This project presents a complete end-to-end Machine Learning solution that predicts rainfall using meteorological features such as humidity, dew point, cloud cover, wind speed, temperature, and atmospheric conditions. It demonstrates not only model building but also how to transform a data science solution into a real-world deployed web application.

The workflow begins with **data loading and exploration**, where the dataset (366 rows × 12 columns) is analyzed for structure, distributions, and class imbalance. Initially, the dataset contains significantly more "rain" instances than "no-rain," which can bias the model. To address this, the minority class was upsampled to achieve a balanced dataset, improving the model’s generalization ability.

During **data preprocessing**, missing values in features like wind direction and wind speed were handled using median imputation, ensuring minimal distortion of data. The target variable (`rainfall`) was encoded into binary format (yes → 1, no → 0), making it suitable for classification algorithms.

A detailed **Exploratory Data Analysis (EDA)** phase was conducted using multiple visualizations:

* Class distribution plots to understand imbalance
* Histograms to observe feature distributions and skewness
* Correlation heatmaps to identify relationships between variables
* Boxplots to compare feature behavior across rainfall outcomes

These visualizations provided key insights into how different features influence rainfall prediction.

For **model development**, a **Random Forest Classifier** was selected due to its high performance, resistance to overfitting, and ability to capture non-linear relationships. Hyperparameter tuning was performed using **GridSearchCV with 5-fold cross-validation**, ensuring the selection of optimal parameters:

* `n_estimators = 100`
* `max_depth = None`
* `min_samples_split = 2`

The trained model achieved strong performance:

* **Accuracy:** ~89%
* **Precision:** ~93.3%
* **Recall:** ~84%
* **F1 Score:** ~88.4%
* **AUC-ROC:** ~0.94

Further evaluation included visual tools such as:

* Confusion Matrix for classification performance
* ROC Curve for threshold analysis
* Feature Importance Graph to identify key predictors
* Metrics comparison charts for overall performance

One of the most important findings from this project is that **humidity, dew point, and cloud cover** are the most influential features in determining rainfall.

The final model was serialized using **Joblib** (`rainfall_rf_model.pkl`) and integrated into a web-based interface. The application allows users to input weather conditions and receive real-time rainfall predictions, making the system practical and user-friendly.

The project is deployed on Render, showcasing the ability to move from a notebook-based ML workflow to a production-ready application.

🔗 **Live Application:** https://ml-rainfall-forecast.onrender.com/

---

## 🚀 Key Features

* End-to-end ML pipeline (EDA → preprocessing → training → evaluation → deployment)
* Handles missing values and class imbalance effectively
* High-performance Random Forest model with tuning
* Rich visualizations for better data understanding
* Interactive web interface for real-time predictions
* Model persistence using `.pkl` file for reuse

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib
* **Model:** Random Forest Classifier
* **Deployment:** Render
* **Interface:** Web-based UI

---

## 📌 Future Enhancements

* Integration with real-time weather APIs
* Deployment using Docker for scalability
* Model improvements using XGBoost or LightGBM
* Mobile-friendly UI/UX improvements
* Continuous model retraining with new data

---

## 👨‍💻 Author

**Aniruddha Garai**

---

⭐ If you found this project useful, consider giving it a star on GitHub!

---
