"""
Visualization utilities for Landslide Detection System.
"""
import folium
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from typing import List, Optional

def get_risk_color(risk_level: str) -> str:
    """
    Return hex color for a given risk level.
    """
    colors = {
        'LOW': '#008000',      # Green
        'MODERATE': '#FFFF00', # Yellow
        'HIGH': '#FFA500',     # Orange
        'VERY_HIGH': '#FF0000',# Red
        'SEVERE': '#8B0000'    # Dark Red
    }
    return colors.get(risk_level.upper(), '#808080')

def create_risk_map(df: pd.DataFrame, center_lat: float = 26.0, center_lon: float = 92.5) -> folium.Map:
    """
    Create a folium map with risk-colored markers.
    df must contain 'lat', 'lon', 'risk_level', and 'risk_score' columns.
    """
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
    
    for _, row in df.iterrows():
        color = get_risk_color(row['risk_level'])
        
        popup_html = f"""
        <b>Location:</b> {row.get('name', 'Unknown')}<br>
        <b>Risk Level:</b> {row['risk_level']}<br>
        <b>Risk Score:</b> {row['risk_score']:.2f}
        """
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=8,
            popup=folium.Popup(popup_html, max_width=300),
            color='black',
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(m)
        
    return m

def plot_feature_importance(importances: List[float], feature_names: List[str], title: str) -> plt.Figure:
    """
    Create a horizontal bar chart of feature importances.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort features by importance
    indices = pd.Series(importances).sort_values().index
    sorted_importances = [importances[i] for i in indices]
    sorted_features = [feature_names[i] for i in indices]
    
    ax.barh(sorted_features, sorted_importances, color='steelblue')
    ax.set_title(title)
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    
    return fig

def plot_risk_distribution(risk_scores: List[float], title: str) -> plt.Figure:
    """
    Create a histogram of risk scores.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(risk_scores, bins=20, color='indianred', edgecolor='black', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', alpha=0.3)
    
    return fig

def plot_rainfall_timeseries(dates: List[str], rainfall_values: List[float], title: str) -> plt.Figure:
    """
    Create a line plot for rainfall time series with threshold lines.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    pd_dates = pd.to_datetime(dates)
    
    ax.plot(pd_dates, rainfall_values, marker='o', linestyle='-', color='royalblue')
    
    # Add arbitrary threshold lines
    ax.axhline(y=50, color='orange', linestyle='--', label='Warning Threshold (50mm)')
    ax.axhline(y=100, color='red', linestyle='--', label='Critical Threshold (100mm)')
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Rainfall (mm)')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def create_state_summary_chart(state_data: pd.DataFrame) -> px.bar:
    """
    Create a plotly bar chart of average risk by state.
    state_data must contain 'state' and 'avg_risk_score' columns.
    """
    fig = px.bar(
        state_data, 
        x='state', 
        y='avg_risk_score',
        title='Average Landslide Risk by State',
        labels={'state': 'State', 'avg_risk_score': 'Average Risk Score'},
        color='avg_risk_score',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig
