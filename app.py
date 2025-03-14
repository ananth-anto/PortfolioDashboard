from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import json
import pdfplumber
import pandas as pd
import numpy as np
import os
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import re

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'

# Data directory
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# Load data from JSON file
def load_data():
    try:
        with open(f'{DATA_DIR}/portfolio_data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {'monthly_data': [], 'uploads': []}
    return data

# Save data to JSON file
def save_data(data):
    with open(f'{DATA_DIR}/portfolio_data.json', 'w') as f:
        json.dump(data, f, indent=4)

# Format currency in Indian format (lakhs, crores)
def format_currency(amount):
    if amount is None or pd.isna(amount):
        return "NA"
    
    amount = float(amount)
    abs_amount = abs(amount)
    
    if abs_amount >= 10000000:  # 1 crore
        formatted = f"₹{abs_amount/10000000:.2f} Cr"
    elif abs_amount >= 100000:  # 1 lakh
        formatted = f"₹{abs_amount/100000:.2f} L"
    else:
        formatted = f"₹{abs_amount:,.2f}"
    
    return formatted if amount >= 0 else f"-{formatted}"

# Format percentage with proper sign
def format_percentage(percentage):
    if percentage is None or pd.isna(percentage):
        return "NA"
    
    return f"+{percentage:.2f}%" if float(percentage) >= 0 else f"{percentage:.2f}%"

# Extract table data from PDF
def extract_table_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Try to find a table with similar structure to what we need
            # Often in 3rd page but let's check all pages
            for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
                tables = page.extract_tables()
                
                for table in tables:
                    # Check if this might be our investment table
                    # Looking for headers like "Month", "Value", "Change"
                    if len(table) > 1:  # At least has header and one row
                        header_row = table[0]
                        header_text = ' '.join([str(h).lower() if h else '' for h in header_row])
                        
                        # Check if it looks like our table
                        if ('month' in header_text and 
                            ('portfolio' in header_text or 'value' in header_text) and
                            'change' in header_text):
                            
                            # This looks like our table, return it
                            return table
        
        # If we didn't find a suitable table
        return None
    
    except Exception as e:
        print(f"Error extracting table: {e}")
        return None

# Process table data and convert to structured format
def process_table_data(table):
    if not table:
        return []
    
    # Skip header row
    data_rows = table[1:]
    processed_data = []
    
    for row in data_rows:
        if len(row) >= 3:  # Expect at least Month, Value, Change
            # Clean and extract month/year
            month_str = str(row[0]).strip() if row[0] else ""
            if not month_str:
                continue
                
            # Try to extract date from the month string
            try:
                # Pattern like "JAN 2025"
                date_match = re.search(r'([A-Z]{3})\s*(\d{4})', month_str, re.IGNORECASE)
                if date_match:
                    month_abbr = date_match.group(1).upper()
                    year = date_match.group(2)
                    
                    # Convert month abbreviation to number
                    month_mapping = {
                        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 
                        'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                        'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                    }
                    month_num = month_mapping.get(month_abbr, 1)
                    
                    # Create date string in ISO format for sorting
                    date_str = f"{year}-{month_num:02d}-01"
                else:
                    # If no match, use a placeholder date
                    date_str = "2025-01-01"
            except:
                date_str = "2025-01-01"
            
            # Extract portfolio value, remove commas and convert to float
            value_str = str(row[1]).strip() if row[1] else "0"
            value_str = re.sub(r'[^\d.]', '', value_str)  # Remove non-numeric chars
            try:
                value = float(value_str) if value_str else 0
            except:
                value = 0
            
            # Extract change amount if available
            change_amount = None
            if len(row) > 2 and row[2] and str(row[2]).strip() != "NA":
                change_str = str(row[2]).strip()
                change_str = re.sub(r'[^\d.-]', '', change_str)  # Remove non-numeric chars
                try:
                    change_amount = float(change_str) if change_str else None
                except:
                    change_amount = None
            
            # Extract change percentage if available
            change_percent = None
            if len(row) > 3 and row[3] and str(row[3]).strip() != "NA":
                percent_str = str(row[3]).strip()
                percent_str = re.sub(r'[^\d.-]', '', percent_str)  # Remove non-numeric chars
                try:
                    change_percent = float(percent_str) if percent_str else None
                except:
                    change_percent = None
            
            # Create record
            processed_data.append({
                'date': date_str,
                'month_display': month_str,
                'value': value,
                'change_amount': change_amount,
                'change_percent': change_percent
            })
    
    # Sort by date
    processed_data.sort(key=lambda x: x['date'])
    return processed_data

# Generate charts
def generate_charts():
    """Generate charts based on data."""
    data = load_data()
    monthly_data = data['monthly_data']
    
    if not monthly_data:
        return None, None, None
    
    # Sort data by date
    monthly_data = sorted(monthly_data, key=lambda x: x['date'])
    
    # Extract months and values
    months = [item['month_display'] for item in monthly_data]
    values = [item['value'] for item in monthly_data]
    
    # Calculate year-over-year data
    years = {}
    for item in monthly_data:
        year = item['month_display'].split(' ')[1]  # Extract year from 'MMM YYYY'
        if year not in years:
            years[year] = []
        years[year].append(item['value'])
    
    yearly_avg = {year: sum(values) / len(values) for year, values in years.items()}
    
    # Generate chart data for JavaScript rendering
    chart_data = {
        'trend': {
            'labels': months,
            'values': values,
            'formatted_values': [format_currency(value) for value in values]
        },
        'yearly': {
            'labels': list(yearly_avg.keys()),
            'values': list(yearly_avg.values()),
            'formatted_values': [format_currency(value) for value in yearly_avg.values()]
        }
    }
    
    # For backward compatibility, still generate the static charts
    # Generate trend chart
    plt.figure(figsize=(12, 6))
    plt.plot(months, values, marker='o', linestyle='-', color='#2980b9')
    plt.title('Portfolio Value Trend')
    plt.xlabel('Month')
    plt.ylabel('Value (₹)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save to buffer
    trend_buffer = BytesIO()
    plt.savefig(trend_buffer, format='png')
    trend_buffer.seek(0)
    trend_chart = base64.b64encode(trend_buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Generate yearly comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(yearly_avg.keys(), yearly_avg.values(), color='#3498db')
    plt.title('Portfolio Value by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Value (₹)')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'₹{height/10000000:.2f} Cr', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    
    # Save to buffer
    yearly_buffer = BytesIO()
    plt.savefig(yearly_buffer, format='png')
    yearly_buffer.seek(0)
    yearly_chart = base64.b64encode(yearly_buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return trend_chart, yearly_chart, chart_data

@app.route('/')
def index():
    # Get data
    data = load_data()
    monthly_data = data['monthly_data']
    uploads = data['uploads']
    
    # Calculate metrics
    latest_value = 0
    absolute_growth = 0
    percentage_growth = 0
    cagr = 0
    best_month = "N/A"
    worst_month = "N/A"
    
    if monthly_data:
        # Sort data by date
        monthly_data = sorted(monthly_data, key=lambda x: x['date'])
        
        # Latest portfolio value
        latest_value = monthly_data[-1]['value']
        
        # Calculate growth
        if len(monthly_data) > 1:
            first_value = monthly_data[0]['value']
            absolute_growth = latest_value - first_value
            percentage_growth = (absolute_growth / first_value) * 100 if first_value > 0 else 0
            
            # Calculate CAGR - fixing date format
            first_date_str = monthly_data[0]['date']
            latest_date_str = monthly_data[-1]['date']
            
            # Handle date strings with proper format (YYYY-MM-DD)
            if '-' in first_date_str:
                parts = first_date_str.split('-')
                if len(parts) == 3:  # YYYY-MM-DD
                    first_date = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                else:  # YYYY-MM
                    first_date = datetime(int(parts[0]), int(parts[1]), 1)
            else:
                # Fallback
                first_date = datetime(2020, 1, 1)
                
            if '-' in latest_date_str:
                parts = latest_date_str.split('-')
                if len(parts) == 3:  # YYYY-MM-DD
                    latest_date = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                else:  # YYYY-MM
                    latest_date = datetime(int(parts[0]), int(parts[1]), 1)
            else:
                # Fallback
                latest_date = datetime(2025, 1, 1)
            
            years = (latest_date - first_date).days / 365.25
            if years > 0 and first_value > 0:
                cagr = ((latest_value / first_value) ** (1 / years) - 1) * 100
        
        # Find best and worst months
        if len(monthly_data) > 1:
            best_month_data = max([x for x in monthly_data if 'change_percent' in x and x['change_percent'] is not None], 
                                key=lambda x: x['change_percent'], default=None)
            worst_month_data = min([x for x in monthly_data if 'change_percent' in x and x['change_percent'] is not None],
                                key=lambda x: x['change_percent'], default=None)
            
            if best_month_data:
                best_month = f"{best_month_data['month_display']} ({format_percentage(best_month_data['change_percent'])})"
            if worst_month_data:
                worst_month = f"{worst_month_data['month_display']} ({format_percentage(worst_month_data['change_percent'])})"
    
    # Generate charts
    trend_chart, yearly_chart, chart_data = generate_charts()
    
    # Create metrics dictionary
    metrics = {
        'latest_value': format_currency(latest_value),
        'total_growth': format_currency(absolute_growth),
        'percentage_growth': format_percentage(percentage_growth),
        'cagr': format_percentage(cagr),
        'best_month': best_month,
        'worst_month': worst_month
    }
    
    return render_template('index.html', 
                          trend_chart=trend_chart,
                          yearly_chart=yearly_chart,
                          chart_data=chart_data,
                          metrics=metrics,
                          data=data,
                          uploads=uploads)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file:
        # Save file temporarily
        os.makedirs('temp', exist_ok=True)
        temp_filepath = os.path.join('temp', file.filename)
        file.save(temp_filepath)
        
        # Extract table from PDF
        table_data = extract_table_from_pdf(temp_filepath)
        
        if table_data:
            # Process the table data
            processed_data = process_table_data(table_data)
            
            if processed_data:
                # Store the extracted data in the session for confirmation
                session['temp_data'] = {
                    'filename': file.filename,
                    'date_uploaded': datetime.now().isoformat(),
                    'records_found': len(processed_data),
                    'processed_data': processed_data
                }
                
                # Clean up
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                
                # Return a JSON response with extracted data for preview
                preview_data = []
                for item in processed_data[:20]:  # Show up to 20 records as preview
                    preview_data.append({
                        'month': item['month_display'],
                        'value': format_currency(item['value']),
                        'change_amount': format_currency(item['change_amount']) if item['change_amount'] else "NA",
                        'change_percent': format_percentage(item['change_percent']) if item['change_percent'] else "NA"
                    })
                
                return jsonify({
                    'success': True,
                    'message': f'Found {len(processed_data)} records in {file.filename}',
                    'preview': preview_data,
                    'total_records': len(processed_data)
                })
            else:
                # Clean up
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                return jsonify({
                    'success': False,
                    'message': 'No valid data could be extracted from the file'
                })
        else:
            # Clean up
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return jsonify({
                'success': False,
                'message': 'Could not find a suitable table in the PDF'
            })
    
    return jsonify({
        'success': False,
        'message': 'Unknown error occurred'
    })

@app.route('/confirm_upload', methods=['POST'])
def confirm_upload():
    if 'temp_data' not in session:
        flash('No data to confirm', 'error')
        return redirect(url_for('index'))
    
    temp_data = session['temp_data']
    
    # Get existing data
    data = load_data()
    
    # Add upload record
    data['uploads'].append({
        'filename': temp_data['filename'],
        'date_uploaded': temp_data['date_uploaded'],
        'records_found': temp_data['records_found']
    })
    
    # Update with new data
    existing_dates = [item['date'] for item in data['monthly_data']]
    
    for item in temp_data['processed_data']:
        if item['date'] not in existing_dates:
            data['monthly_data'].append(item)
        else:
            # Update existing record
            idx = existing_dates.index(item['date'])
            data['monthly_data'][idx] = item
    
    # Save updated data
    save_data(data)
    
    # Clear the temporary data
    session.pop('temp_data', None)
    
    flash(f'Successfully added {temp_data["records_found"]} records from {temp_data["filename"]}', 'success')
    return redirect(url_for('index'))

@app.route('/cancel_upload', methods=['POST'])
def cancel_upload():
    # Clear the temporary data
    session.pop('temp_data', None)
    
    flash('Upload cancelled', 'info')
    return redirect(url_for('index'))

@app.route('/clear_data', methods=['POST'])
def clear_data():
    # Reset data
    save_data({'monthly_data': [], 'uploads': []})
    flash('All data has been cleared', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)