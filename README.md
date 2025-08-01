# 🧠 Toxic Comment Classification

This project is a machine learning-based multi-label classification system that detects various types of toxicity in user comments such as **insult**, **obscenity**, and **threats**.

---

## 📌 Project Highlights

- 🔍 Preprocessed and balanced imbalanced dataset using SMOTE and Tomek Links
- 🤖 Trained a **Logistic Regression** model for multi-label classification
- 🧪 Evaluated with metrics like Precision, Recall, and F1-score
- 💾 Stored trained model using `pickle`
- 🌐 Deployed a simple web app using **Flask** to input comments and display toxicity labels
- 🎨 Frontend styled with HTML, CSS, and JS

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/PuneethKaleyanda/comment_classification.git
cd comment_classification
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
(Create a requirements.txt file if not present using pip freeze > requirements.txt)

3. Run the Flask App
bash
Copy
Edit
python app.py
App will start on http://127.0.0.1:5000/

🗂️ Project Structure
bash
Copy
Edit
comment_classification/
├── app.py                      # Flask backend
├── comment_classification.ipynb# Model training & evaluation
├── Equal/                      # Balancing methods (SMOTE, Tomek)
├── model/
│   ├── logistic_regression_model.pkl
│   └── logistic_regression_model_balance.pkl
├── static/
│   ├── css/styles.css
│   └── js/script.js
├── templates/
│   └── index.html
├── train.csv                   # Dataset
└── .gitattributes              # Git LFS tracked files
📊 Sample Output
Input: You're such a loser!

Output: ☑️ Insult, ☑️ Obscene, ❌ Threat

🧠 ML Techniques Used
Multi-label Logistic Regression

Label Binarization

Data balancing: SMOTE + Tomek Links

Evaluation: Precision, Recall, F1-score

🛠️ Future Improvements
Add more ML models (e.g., Random Forest, BERT)

Improve UI with result confidence scores

Add user authentication and comment history

👨‍💻 Author
Puneeth Kaleyanda
GitHub