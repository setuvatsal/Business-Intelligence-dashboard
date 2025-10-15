from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import json
import random
from datetime import datetime, timedelta
import sqlite3
import os

app = Flask(__name__)

# Initialize database
def init_db():
    conn = sqlite3.connect('business_data.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            product TEXT,
            category TEXT,
            sales_amount REAL,
            quantity INTEGER,
            region TEXT,
            customer_segment TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            upload_date TEXT,
            data_json TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# Generate sample data
def generate_sample_data():
    conn = sqlite3.connect('business_data.db')
    cursor = conn.cursor()
    
    # Check if data already exists
    cursor.execute('SELECT COUNT(*) FROM sales_data')
    if cursor.fetchone()[0] > 0:
        conn.close()
        return
    
    # Generate sample sales data
    products = ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smartwatch', 'Monitor', 'Keyboard', 'Mouse']
    categories = ['Electronics', 'Accessories', 'Computing']
    regions = ['North', 'South', 'East', 'West', 'Central']
    segments = ['Enterprise', 'Consumer', 'Education', 'Government']
    
    sample_data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(1000):
        date = start_date + timedelta(days=random.randint(0, 365))
        product = random.choice(products)
        category = random.choice(categories)
        sales_amount = random.uniform(100, 5000)
        quantity = random.randint(1, 50)
        region = random.choice(regions)
        segment = random.choice(segments)
        
        sample_data.append((
            date.strftime('%Y-%m-%d'),
            product, category, sales_amount, quantity, region, segment
        ))
    
    cursor.executemany('''
        INSERT INTO sales_data (date, product, category, sales_amount, quantity, region, customer_segment)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', sample_data)
    
    conn.commit()
    conn.close()

# ML Model for predictions
class PredictiveAnalytics:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.is_trained = False
    
    def prepare_data(self, df):
        # Encode categorical variables
        categorical_columns = ['product', 'category', 'region', 'customer_segment']
        df_encoded = df.copy()
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                try:
                    df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                except Exception:
                    df_encoded[col] = 0
            else:
                # Handle unseen categories
                try:
                    df_encoded[col] = self.label_encoders[col].transform(df[col].astype(str))
                except Exception:
                    df_encoded[col] = 0
        # Convert date to numerical features
        df_encoded['date'] = pd.to_datetime(df['date'], errors='coerce')
        df_encoded['month'] = df_encoded['date'].dt.month.fillna(1).astype(int)
        df_encoded['day_of_year'] = df_encoded['date'].dt.dayofyear.fillna(1).astype(int)
        df_encoded['weekday'] = df_encoded['date'].dt.weekday.fillna(0).astype(int)
        return df_encoded
    
    def train_model(self, df):
        df_encoded = self.prepare_data(df)
        features = ['product', 'category', 'region', 'customer_segment', 'month', 'day_of_year', 'weekday', 'quantity']
        X = df_encoded[features]
        y = df_encoded['sales_amount']
        if len(X) < 2:
            raise ValueError('Not enough data to train the model.')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        self.is_trained = True
        return mae
    
    def predict_sales(self, product, category, region, segment, month, quantity):
        if not self.is_trained:
            return None
        # Create input data
        input_data = pd.DataFrame({
            'product': [product],
            'category': [category],
            'region': [region],
            'customer_segment': [segment],
            'month': [month],
            'day_of_year': [180],  # Default to mid-year
            'weekday': [1],  # Default to Tuesday
            'quantity': [quantity]
        })
        # Encode categorical variables
        for col in ['product', 'category', 'region', 'customer_segment']:
            if col in self.label_encoders:
                try:
                    input_data[col] = self.label_encoders[col].transform(input_data[col].astype(str))
                except Exception:
                    input_data[col] = 0
            else:
                input_data[col] = 0
        prediction = self.model.predict(input_data)
        return prediction[0]

# Initialize ML model
ml_model = PredictiveAnalytics()

# AI Insights Generator
class AIInsights:
    @staticmethod
    def generate_insights(df):
        insights = []
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            monthly_sales = df.groupby(df['date'].dt.to_period('M'))['sales_amount'].sum()
            if len(monthly_sales) > 1:
                trend = "increasing" if monthly_sales.iloc[-1] > monthly_sales.iloc[0] else "decreasing"
                insights.append(f"Sales trend is {trend} over the analyzed period.")
            else:
                insights.append("Not enough data to determine sales trend.")
            # Top performing products
            if 'product' in df.columns and not df['product'].isnull().all():
                top_product = df.groupby('product')['sales_amount'].sum().idxmax()
                insights.append(f"'{top_product}' is your highest revenue generating product.")
            # Regional analysis
            if 'region' in df.columns and not df['region'].isnull().all():
                regional_performance = df.groupby('region')['sales_amount'].mean()
                best_region = regional_performance.idxmax()
                insights.append(f"The {best_region} region shows the highest average sales performance.")
            # Seasonal patterns
            seasonal_sales = df.groupby(df['date'].dt.month)['sales_amount'].mean()
            if not seasonal_sales.empty:
                peak_month = seasonal_sales.idxmax()
                month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                              7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
                insights.append(f"Sales typically peak in {month_names.get(peak_month, 'Unknown')}.")
            # Customer segment analysis
            if 'customer_segment' in df.columns and not df['customer_segment'].isnull().all():
                segment_performance = df.groupby('customer_segment')['sales_amount'].sum()
                top_segment = segment_performance.idxmax()
                insights.append(f"The {top_segment} segment contributes the most to total revenue.")
        except Exception as e:
            insights.append(f"Error generating insights: {str(e)}")
        return insights

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard-data')
def get_dashboard_data():
    conn = sqlite3.connect('business_data.db')
    df = pd.read_sql_query('SELECT * FROM sales_data', conn)
    conn.close()
    
    if df.empty:
        return jsonify({'error': 'No data available'})
    
    # Prepare data for frontend
    df['date'] = pd.to_datetime(df['date'])
    
    # Monthly sales trend
    monthly_sales = df.groupby(df['date'].dt.to_period('M')).agg({
        'sales_amount': 'sum',
        'quantity': 'sum'
    }).reset_index()
    monthly_sales['date'] = monthly_sales['date'].astype(str)
    
    # Product performance
    product_sales = df.groupby('product')['sales_amount'].sum().reset_index()
    product_sales = product_sales.sort_values('sales_amount', ascending=False).head(10)
    
    # Regional distribution
    regional_sales = df.groupby('region')['sales_amount'].sum().reset_index()
    
    # Category breakdown
    category_sales = df.groupby('category')['sales_amount'].sum().reset_index()
    
    # Customer segment analysis
    segment_sales = df.groupby('customer_segment')['sales_amount'].sum().reset_index()
    
    # KPIs
    total_revenue = df['sales_amount'].sum()
    total_quantity = df['quantity'].sum()
    avg_order_value = df['sales_amount'].mean()
    total_orders = len(df)
    
    return jsonify({
        'kpis': {
            'total_revenue': round(total_revenue, 2),
            'total_quantity': int(total_quantity),
            'avg_order_value': round(avg_order_value, 2),
            'total_orders': total_orders
        },
        'monthly_sales': monthly_sales.to_dict('records'),
        'product_sales': product_sales.to_dict('records'),
        'regional_sales': regional_sales.to_dict('records'),
        'category_sales': category_sales.to_dict('records'),
        'segment_sales': segment_sales.to_dict('records')
    })

@app.route('/api/train-model', methods=['POST'])
def train_model():
    conn = sqlite3.connect('business_data.db')
    df = pd.read_sql_query('SELECT * FROM sales_data', conn)
    conn.close()
    
    if df.empty:
        return jsonify({'error': 'No data available for training'})
    
    try:
        mae = ml_model.train_model(df)
        return jsonify({'success': True, 'mae': round(mae, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def predict_sales():
    data = request.get_json()
    
    try:
        prediction = ml_model.predict_sales(
            data['product'],
            data['category'],
            data['region'],
            data['customer_segment'],
            data['month'],
            data['quantity']
        )
        if prediction is None:
            return jsonify({'error': 'Model is not trained yet.'})
        return jsonify({'success': True, 'predicted_sales': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/insights', methods=['POST'])
def generate_insights():
    data = request.get_json()
    
    try:
        df = pd.DataFrame(data['data'])
        insights = AIInsights.generate_insights(df)
        return jsonify({'success': True, 'insights': insights})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/insights', methods=['GET'])
def get_insights_from_db():
    conn = sqlite3.connect('business_data.db')
    df = pd.read_sql_query('SELECT * FROM sales_data', conn)
    conn.close()
    if df.empty:
        return jsonify({'success': True, 'insights': ['No data available for analysis']})
    try:
        insights = AIInsights.generate_insights(df)
        return jsonify({'success': True, 'insights': insights})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'})
    try:
        # Save file locally
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        # Read and process the file
        df = pd.read_csv(file_path)
        # Validate required columns
        expected_cols = {'date','product','category','sales_amount','quantity','region','customer_segment'}
        if not expected_cols.issubset(set(df.columns)):
            return jsonify({'error': f'Missing columns. Required: {expected_cols}'})
        # Insert into DB
        conn = sqlite3.connect('business_data.db')
        df.to_sql('sales_data', conn, if_exists='append', index=False)
        conn.close()
        return jsonify({'success': True, 'filename': file.filename})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    import webbrowser
    init_db()
    generate_sample_data()
    # Open the default browser to the Flask app
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True)