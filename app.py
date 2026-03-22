from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import io
import base64

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
app.secret_key = 'finance_tracker_secret_key'

# Path to the transactions CSV file
DATA_PATH = 'data/transactions.csv'

# Directory for saving models
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

CATEGORY_MODEL_PATH = os.path.join(MODEL_DIR, 'category_model.pkl')
EXPENSE_MODEL_PATH = os.path.join(MODEL_DIR, 'expense_model.pkl')

# Create a sample dataset if needed
def create_sample_dataset():
    if not os.path.exists('data'):
        os.makedirs('data')
        
    if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0:
        sample_data = {
            'date': ['2023-01-01', '2023-01-03', '2023-01-05', '2023-01-10', 
                     '2023-01-15', '2023-01-20', '2023-01-25', '2023-02-01'],
            'description': ['Grocery shopping', 'Restaurant dinner', 'Electricity bill', 'Gas station',
                           'Phone bill', 'Grocery store', 'Movie tickets', 'Internet bill'],
            'amount': [150.50, 85.20, 120.00, 60.75, 45.00, 130.25, 35.00, 55.00],
            'category': ['Groceries', 'Dining', 'Utilities', 'Transportation', 
                        'Utilities', 'Groceries', 'Entertainment', 'Utilities']
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(DATA_PATH, index=False)
        return True
    return False

# Call this function before doing anything else
# created_sample = create_sample_dataset()

# Function to train and save models
# def train_models():
#     # Check if data file exists
#     if not os.path.exists(DATA_PATH):
#         return False, "Data file not found. Please make sure 'data/transactions.csv' exists."
    
#     try:
#         # Load and prepare the data
#         data = pd.read_csv(DATA_PATH)
        
#         # Check if data has required columns
#         required_columns = ['date', 'description', 'amount', 'category']
#         for col in required_columns:
#             if col not in data.columns:
#                 return False, f"Column '{col}' not found in the data. Please make sure your CSV file has these columns: {', '.join(required_columns)}"
        
#         # Train the category model
#         X_cat = data['description'].str.lower().values.reshape(-1, 1)
#         y_cat = data['category']
#         category_model = RandomForestClassifier(n_estimators=100, random_state=42)
#         category_model.fit(X_cat, y_cat)
        
#         # Train the expense prediction model
#         data['date'] = pd.to_datetime(data['date'])
#         data['day_of_month'] = data['date'].dt.day
#         data['month'] = data['date'].dt.month
#         data['year'] = data['date'].dt.year
#         X_exp = data[['day_of_month', 'month', 'year']].values
#         expense_model = LinearRegression()
#         expense_model.fit(X_exp, data['amount'])
        
#         # Save the models
#         with open(CATEGORY_MODEL_PATH, 'wb') as f:
#             pickle.dump(category_model, f)
#         with open(EXPENSE_MODEL_PATH, 'wb') as f:
#             pickle.dump(expense_model, f)
        
#         return True, "Models trained successfully!"
    
#     except Exception as e:
#         return False, f"Error training models: {str(e)}"

# def train_models():
#     # Check if data file exists
#     if not os.path.exists(DATA_PATH):
#         return False, "Data file not found. Please make sure 'data/transactions.csv' exists."
    
#     try:
#         # Load and prepare the data
#         data = pd.read_csv(DATA_PATH)
        
#         # Check if data has required columns
#         required_columns = ['date', 'description', 'amount', 'category']
#         for col in required_columns:
#             if col not in data.columns:
#                 return False, f"Column '{col}' not found in the data. Please make sure your CSV file has these columns: {', '.join(required_columns)}"
        
#         # Train the category model using CountVectorizer for text features
#         from sklearn.feature_extraction.text import CountVectorizer
        
#         # Get description and category data
#         descriptions = data['description'].astype(str).values
#         categories = data['category'].values
        
#         # Create a vectorizer to convert text to numerical features
#         vectorizer = CountVectorizer(analyzer='word', max_features=100)
#         X_cat = vectorizer.fit_transform(descriptions)
        
#         # Train the category model
#         category_model = RandomForestClassifier(n_estimators=100, random_state=42)
#         category_model.fit(X_cat, categories)
        
#         # Save the vectorizer along with the model
#         with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'wb') as f:
#             pickle.dump(vectorizer, f)
        
#         # Train the expense prediction model
#         data['date'] = pd.to_datetime(data['date'])
#         data['day_of_month'] = data['date'].dt.day
#         data['month'] = data['date'].dt.month
#         data['year'] = data['date'].dt.year
#         X_exp = data[['day_of_month', 'month', 'year']].values
#         expense_model = LinearRegression()
#         expense_model.fit(X_exp, data['amount'])
        
#         # Save the models
#         with open(CATEGORY_MODEL_PATH, 'wb') as f:
#             pickle.dump(category_model, f)
#         with open(EXPENSE_MODEL_PATH, 'wb') as f:
#             pickle.dump(expense_model, f)
        
#         return True, "Models trained successfully!"
    
#     except Exception as e:
#         return False, f"Error training models: {str(e)}"

def train_models():
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        return False, "Data file not found. Please make sure 'data/transactions.csv' exists."
    
    try:
        # Load and prepare the data
        data = pd.read_csv(DATA_PATH)
        
        # Check if data has required columns
        required_columns = ['date', 'description', 'amount', 'category']
        for col in required_columns:
            if col not in data.columns:
                return False, f"Column '{col}' not found in the data. Please make sure your CSV file has these columns: {', '.join(required_columns)}"
        
        # Check if there's enough data
        if len(data) < 2:
            # Create a simpler model since we don't have enough data
            category_map = dict(zip(data['description'].astype(str), data['category']))
            
            # Save this as a simple dictionary instead of a model
            with open(CATEGORY_MODEL_PATH, 'wb') as f:
                pickle.dump(category_map, f)
                
            # Set a flag to indicate we're using a simple lookup
            with open(os.path.join(MODEL_DIR, 'using_simple_lookup.txt'), 'w') as f:
                f.write('true')
        else:
            # Make sure descriptions are strings and not empty
            data['description'] = data['description'].astype(str).fillna('unknown')
            
            # Create a vectorizer with minimal preprocessing 
            from sklearn.feature_extraction.text import CountVectorizer
            
            # Configure vectorizer to be more permissive
            vectorizer = CountVectorizer(
                analyzer='word',
                token_pattern=r'\b\w+\b',  # Match any word character
                min_df=1,  # Include terms that appear in at least 1 document
                max_features=100,
                stop_words=None  # Don't remove stop words
            )
            
            # Get features and target
            X_cat = vectorizer.fit_transform(data['description'])
            
            # Train the category model
            category_model = RandomForestClassifier(n_estimators=100, random_state=42)
            category_model.fit(X_cat, data['category'])
            
            # Save the vectorizer and model
            with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'wb') as f:
                pickle.dump(vectorizer, f)
            
            with open(CATEGORY_MODEL_PATH, 'wb') as f:
                pickle.dump(category_model, f)
            
            # Mark that we're using a proper model
            if os.path.exists(os.path.join(MODEL_DIR, 'using_simple_lookup.txt')):
                os.remove(os.path.join(MODEL_DIR, 'using_simple_lookup.txt'))
        
        # Train the expense prediction model
        data['date'] = pd.to_datetime(data['date'])
        data['day_of_month'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        X_exp = data[['day_of_month', 'month', 'year']].values
        expense_model = LinearRegression()
        expense_model.fit(X_exp, data['amount'])
        
        # Save the expense model
        with open(EXPENSE_MODEL_PATH, 'wb') as f:
            pickle.dump(expense_model, f)
        
        return True, "Models trained successfully!"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error training models: {str(e)}"
# Function to load models
def load_models():
    # Check if models exist, if not train them
    if not os.path.exists(CATEGORY_MODEL_PATH) or not os.path.exists(EXPENSE_MODEL_PATH):
        success, message = train_models()
        if not success:
            return None, None, message
    
    try:
        with open(CATEGORY_MODEL_PATH, 'rb') as f:
            category_model = pickle.load(f)
        with open(EXPENSE_MODEL_PATH, 'rb') as f:
            expense_model = pickle.load(f)
        return category_model, expense_model, "Models loaded successfully!"
    except Exception as e:
        return None, None, f"Error loading models: {str(e)}"

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')
    
# Create empty transactions file if it doesn't exist
if not os.path.exists(DATA_PATH):
    df = pd.DataFrame(columns=['date', 'description', 'amount', 'category'])
    df.to_csv(DATA_PATH, index=False)

# Make sure we have sample data
create_sample_dataset()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    # Load models
    # category_model, expense_model, model_message = load_models()
    try:
         category_model, expense_model, model_message = load_models()
         if category_model is None or expense_model is None:
             flash(model_message, 'danger')
    except Exception as e:
         import traceback
         traceback.print_exc()
         flash(f"Error loading models: {str(e)}", 'danger')
         category_model = None
         expense_model = None

    if category_model is None or expense_model is None:
        flash(model_message, 'danger')
        return render_template('dashboard.html', tables=[], total_expense=0, future_expense=0, 
                              category_chart='', expense_trend='', categories=[])
    
    if request.method == 'POST':
        try:
            # Get the new transaction data from the form
            date = request.form['date']
            description = request.form['description']
            amount = float(request.form['amount'])
            # category = request.form['category']
            # category = request.form.get('new_category') or request.form.get('category') or 'Other'
            # category = category.strip()  # Clean up extra spaces
            # ✅ Fix: Get custom or selected category
            new_category = request.form.get('new_category', '').strip()
            selected_category = request.form.get('category', '').strip()

            category = new_category if new_category else selected_category
            if not category:
              category = 'Other'

            print(f"User submitted category: '{category}' (new: '{new_category}', selected: '{selected_category}')")

            # Append the new transaction to the CSV file
            if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0:
                # File exists and is not empty
                new_transaction = pd.DataFrame([[date, description, amount, category]], 
                                              columns=['date', 'description', 'amount', 'category'])
                new_transaction.to_csv(DATA_PATH, mode='a', header=False, index=False)
            else:
                # File doesn't exist or is empty
                new_transaction = pd.DataFrame([[date, description, amount, category]], 
                                              columns=['date', 'description', 'amount', 'category'])
                new_transaction.to_csv(DATA_PATH, index=False)
            
            flash('Transaction added successfully!', 'success')
            # # Retrain models with new data
            # success, message = train_models()
            # if not success:
            #     flash(message, 'warning')
            try:
                success, message = train_models()
                if not success:
                    flash(message, 'warning')
            except Exception as e:
                import traceback
                traceback.print_exc()  # This logs the exact crash in your terminal
                flash(f"Error during model training: {str(e)}", 'danger')

        except Exception as e:
            flash(f'Error adding transaction: {str(e)}', 'danger')
        
        return redirect(url_for('dashboard'))
    
    # Load and process the transactions
    try:
        if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0:
            transactions = pd.read_csv(DATA_PATH)
            # ✅ Clean up category column right here:
            transactions['category'] = transactions['category'].astype(str).str.strip()
            
            # Convert date to datetime
            transactions['date'] = pd.to_datetime(transactions['date'])
            
            # Sort transactions by date
            transactions = transactions.sort_values('date', ascending=False).reset_index(drop=True)
            
            # Format date for display
            transactions['date'] = transactions['date'].dt.strftime('%Y-%m-%d')
            
            # Get unique categories for the form
            categories = transactions['category'].str.strip().unique().tolist()
            if not categories:
                categories = ['Groceries', 'Dining', 'Utilities', 'Entertainment', 'Transportation']

            categories = sorted(set([c.strip() for c in transactions['category'] if isinstance(c, str)]))
            # Get all unique categories dynamically
            categories = sorted(transactions['category'].unique().tolist())

            
            # Calculate total expense
            total_expense = transactions['amount'].sum()
            
            # Predict future expenses for next month
            today = datetime.now()
            next_month = datetime(today.year, today.month % 12 + 1, 1)
            
            # Create prediction features
            features = np.array([[next_month.day, next_month.month, next_month.year]])
            
            # Make prediction
            future_expense = expense_model.predict(features)[0]
            
            # Round to 2 decimal places
            total_expense = round(total_expense, 2)
            future_expense = round(future_expense, 2)
            
            # Create category pie chart
            category_chart = create_category_chart(transactions)
            
            # Create expense trend chart
            expense_trend = create_expense_trend_chart(transactions)
            
            return render_template('dashboard.html', 
                                  tables=[transactions.to_html(classes='table table-striped', index=False)],
                                  total_expense=total_expense, 
                                  future_expense=future_expense,
                                  category_chart=category_chart,
                                  expense_trend=expense_trend,
                                  categories=categories)
        else:
            # No transactions yet
            flash('No transactions found. Add your first transaction below!', 'info')
            return render_template('dashboard.html', tables=[], total_expense=0, future_expense=0, 
                                  category_chart='', expense_trend='', categories=[])
    
    except Exception as e:
        flash(f'Error loading transactions: {str(e)}', 'danger')
        return render_template('dashboard.html', tables=[], total_expense=0, future_expense=0, 
                              category_chart='', expense_trend='', categories=[])

def create_category_chart(transactions):
    # Create category pie chart
    plt.figure(figsize=(8, 6))
    category_data = transactions.groupby('category')['amount'].sum()
    plt.pie(category_data, labels=category_data.index, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Expenses by Category')
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def create_expense_trend_chart(transactions):
    # Create expense trend line chart
    plt.figure(figsize=(10, 6))
    
    # Convert date to datetime and group by month
    transactions['date'] = pd.to_datetime(transactions['date'])
    monthly_data = transactions.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum()
    
    plt.plot(monthly_data.index, monthly_data.values, marker='o')
    plt.title('Monthly Expense Trend')
    plt.xlabel('Month')
    plt.ylabel('Total Expense')
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

# @app.route('/predict_category', methods=['POST'])
# def predict_category():
#     if request.method == 'POST':
#         description = request.form['description']
        
#         # Load category model
#         try:
#             with open(CATEGORY_MODEL_PATH, 'rb') as f:
#                 category_model = pickle.load(f)
            
#             # Predict category
#             prediction = category_model.predict([description.lower()])[0]
            
#             return {'predicted_category': prediction}
#         except:
#             return {'predicted_category': ''}

# @app.route('/predict_category', methods=['POST'])
# def predict_category():
#     if request.method == 'POST':
#         description = request.form['description']
        
#         # Load category model and vectorizer
#         try:
#             with open(CATEGORY_MODEL_PATH, 'rb') as f:
#                 category_model = pickle.load(f)
            
#             with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
#                 vectorizer = pickle.load(f)
            
#             # Transform the description using the vectorizer
#             X = vectorizer.transform([description.lower()])
            
#             # Predict category
#             prediction = category_model.predict(X)[0]
            
#             return {'predicted_category': prediction}
#         except Exception as e:
#             print(f"Error predicting category: {str(e)}")
#             return {'predicted_category': ''}

@app.route('/predict_category', methods=['POST'])
def predict_category():
    if request.method == 'POST':
        description = request.form['description']
        
        try:
            # Check if we're using a simple lookup
            using_simple_lookup = os.path.exists(os.path.join(MODEL_DIR, 'using_simple_lookup.txt'))
            
            if using_simple_lookup:
                # Load the simple category map
                with open(CATEGORY_MODEL_PATH, 'rb') as f:
                    category_map = pickle.load(f)
                
                # Try to find an exact match, or return the first category
                if description in category_map:
                    return {'predicted_category': category_map[description]}
                else:
                    # Return the first category or a default
                    default_category = next(iter(category_map.values())) if category_map else 'Other'
                    return {'predicted_category': default_category}
            else:
                # Using the trained model
                with open(CATEGORY_MODEL_PATH, 'rb') as f:
                    category_model = pickle.load(f)
                
                with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
                    vectorizer = pickle.load(f)
                
                # Transform the description using the vectorizer
                X = vectorizer.transform([description])
                
                # Predict category
                prediction = category_model.predict(X)[0]
                
                return {'predicted_category': prediction}
        except Exception as e:
            print(f"Error predicting category: {str(e)}")
            return {'predicted_category': ''}
        





if __name__ == '__main__':
    app.run(debug=True)