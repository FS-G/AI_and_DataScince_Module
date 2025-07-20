import pandas as pd
import json
from datetime import datetime
import numpy as np

def process_sales_data():
    """Process the sales.csv data and generate insights for the dashboard"""
    
    # Read the CSV file
    df = pd.read_csv('2_excel/practice_data/store sales/sales.csv')
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Add year and month columns for easier analysis
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['MonthYear'] = df['Date'].dt.to_period('M')
    
    # Calculate additional metrics
    df['Profit_Margin_Percent'] = (df['Profit'] / df['Sales'] * 100).fillna(0)
    df['Budget_Variance'] = df['Profit'] - df['Budget Profit']
    df['Sales_Variance'] = df['Sales'] - df['Budget Sales']
    
    # Generate insights
    insights = {
        'summary': generate_summary(df),
        'product_analysis': analyze_products(df),
        'market_analysis': analyze_markets(df),
        'temporal_analysis': analyze_temporal_trends(df),
        'financial_analysis': analyze_financial_metrics(df),
        'performance_insights': generate_performance_insights(df)
    }
    
    return insights

def generate_summary(df):
    """Generate overall summary statistics"""
    return {
        'total_revenue': float(df['Sales'].sum()),
        'total_profit': float(df['Profit'].sum()),
        'total_cogs': float(df['COGS'].sum()),
        'total_expenses': float(df['Total Expenses'].sum()),
        'total_marketing': float(df['Marketing'].sum()),
        'total_inventory': float(df['Inventory'].sum()),
        'total_transactions': len(df),
        'profit_margin': float((df['Profit'].sum() / df['Sales'].sum()) * 100),
        'budget_variance': float(df['Budget_Variance'].sum()),
        'sales_variance': float(df['Sales_Variance'].sum())
    }

def analyze_products(df):
    """Analyze product performance"""
    product_analysis = df.groupby(['Product Type', 'Product']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'COGS': 'sum',
        'Marketing': 'sum',
        'Inventory': 'mean',
        'Profit_Margin_Percent': 'mean'
    }).reset_index()
    
    # Calculate additional metrics
    product_analysis['Profit_Margin'] = (product_analysis['Profit'] / product_analysis['Sales'] * 100).fillna(0)
    product_analysis['ROI'] = (product_analysis['Profit'] / product_analysis['Marketing'] * 100).fillna(0)
    
    # Sort by revenue
    product_analysis = product_analysis.sort_values('Sales', ascending=False)
    
    return product_analysis.to_dict('records')

def analyze_markets(df):
    """Analyze market performance"""
    market_analysis = df.groupby(['Market', 'Market Size']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'COGS': 'sum',
        'Marketing': 'sum',
        'Inventory': 'mean',
        'Profit_Margin_Percent': 'mean'
    }).reset_index()
    
    # Calculate additional metrics
    market_analysis['Profit_Margin'] = (market_analysis['Profit'] / market_analysis['Sales'] * 100).fillna(0)
    market_analysis['Market_Efficiency'] = (market_analysis['Profit'] / market_analysis['Marketing'] * 100).fillna(0)
    
    # Sort by revenue
    market_analysis = market_analysis.sort_values('Sales', ascending=False)
    
    return market_analysis.to_dict('records')

def analyze_temporal_trends(df):
    """Analyze temporal trends"""
    # Monthly trends
    monthly_trends = df.groupby('MonthYear').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Marketing': 'sum',
        'Profit_Margin_Percent': 'mean'
    }).reset_index()
    
    monthly_trends['MonthYear'] = monthly_trends['MonthYear'].astype(str)
    monthly_trends['Profit_Margin'] = (monthly_trends['Profit'] / monthly_trends['Sales'] * 100).fillna(0)
    
    # Yearly trends
    yearly_trends = df.groupby('Year').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Marketing': 'sum',
        'Profit_Margin_Percent': 'mean'
    }).reset_index()
    
    yearly_trends['Profit_Margin'] = (yearly_trends['Profit'] / yearly_trends['Sales'] * 100).fillna(0)
    
    return {
        'monthly': monthly_trends.to_dict('records'),
        'yearly': yearly_trends.to_dict('records')
    }

def analyze_financial_metrics(df):
    """Analyze financial metrics"""
    # Budget vs Actual
    budget_analysis = {
        'total_budget_profit': float(df['Budget Profit'].sum()),
        'total_actual_profit': float(df['Profit'].sum()),
        'total_budget_sales': float(df['Budget Sales'].sum()),
        'total_actual_sales': float(df['Sales'].sum()),
        'profit_variance': float(df['Budget_Variance'].sum()),
        'sales_variance': float(df['Sales_Variance'].sum())
    }
    
    # Efficiency metrics
    efficiency_metrics = {
        'avg_revenue_per_transaction': float(df['Sales'].mean()),
        'avg_profit_per_transaction': float(df['Profit'].mean()),
        'avg_marketing_per_transaction': float(df['Marketing'].mean()),
        'avg_inventory_level': float(df['Inventory'].mean()),
        'profit_margin_std': float(df['Profit_Margin_Percent'].std()),
        'revenue_growth_rate': calculate_growth_rate(df, 'Sales'),
        'profit_growth_rate': calculate_growth_rate(df, 'Profit')
    }
    
    return {
        'budget_analysis': budget_analysis,
        'efficiency_metrics': efficiency_metrics
    }

def calculate_growth_rate(df, column):
    """Calculate growth rate for a column"""
    if len(df) < 2:
        return 0
    
    # Sort by date
    df_sorted = df.sort_values('Date')
    
    # Calculate growth rate
    first_value = df_sorted[column].iloc[0]
    last_value = df_sorted[column].iloc[-1]
    
    if first_value == 0:
        return 0
    
    return ((last_value - first_value) / first_value) * 100

def generate_performance_insights(df):
    """Generate performance insights and recommendations"""
    insights = []
    
    # Top performing products
    top_products = df.groupby(['Product Type', 'Product'])['Sales'].sum().sort_values(ascending=False).head(5)
    insights.append({
        'type': 'top_products',
        'title': 'Top 5 Revenue Generating Products',
        'data': top_products.to_dict()
    })
    
    # Best profit margins
    best_margins = df.groupby(['Product Type', 'Product'])['Profit_Margin_Percent'].mean().sort_values(ascending=False).head(5)
    insights.append({
        'type': 'best_margins',
        'title': 'Top 5 Products by Profit Margin',
        'data': best_margins.to_dict()
    })
    
    # Market performance
    market_performance = df.groupby('Market')['Sales'].sum().sort_values(ascending=False)
    insights.append({
        'type': 'market_performance',
        'title': 'Market Performance by Revenue',
        'data': market_performance.to_dict()
    })
    
    # Budget performance
    budget_performance = df.groupby(['Product Type', 'Product']).agg({
        'Budget_Variance': 'sum',
        'Sales_Variance': 'sum'
    }).sort_values('Budget_Variance', ascending=False).head(5)
    
    insights.append({
        'type': 'budget_performance',
        'title': 'Top 5 Products by Budget Variance',
        'data': budget_performance.to_dict('index')
    })
    
    return insights

def generate_dashboard_data():
    """Generate all data needed for the dashboard"""
    insights = process_sales_data()
    
    # Create dashboard-ready data
    dashboard_data = {
        'kpis': insights['summary'],
        'charts': {
            'revenue_trend': insights['temporal_analysis']['monthly'],
            'product_performance': insights['product_analysis'],
            'market_performance': insights['market_analysis'],
            'budget_variance': insights['financial_analysis']['budget_analysis']
        },
        'insights': insights['performance_insights'],
        'tables': {
            'top_products': insights['product_analysis'][:10],
            'market_rankings': insights['market_analysis']
        }
    }
    
    return dashboard_data

if __name__ == "__main__":
    # Generate dashboard data
    dashboard_data = generate_dashboard_data()
    
    # Save to JSON file for the dashboard to use
    with open('dashboard_data.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    
    print("Dashboard data generated successfully!")
    print(f"Total Revenue: ${dashboard_data['kpis']['total_revenue']:,.2f}")
    print(f"Total Profit: ${dashboard_data['kpis']['total_profit']:,.2f}")
    print(f"Profit Margin: {dashboard_data['kpis']['profit_margin']:.2f}%")
    print(f"Total Transactions: {dashboard_data['kpis']['total_transactions']:,}") 