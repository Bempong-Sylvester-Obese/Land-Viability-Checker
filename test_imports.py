#!/usr/bin/env python3
"""
Test script to verify all imports are working correctly.
"""

def test_imports():
    """Test all required imports."""
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
        
        import numpy as np
        print("✓ numpy imported successfully")
        
        from sklearn.model_selection import train_test_split
        print("✓ sklearn.model_selection imported successfully")
        
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        print("✓ sklearn.preprocessing imported successfully")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
        
        import seaborn as sns
        print("✓ seaborn imported successfully")
        
        import plotly.express as px
        print("✓ plotly imported successfully")
        
        from PIL import Image
        print("✓ Pillow (PIL) imported successfully")
        
        import cv2
        print("✓ opencv-python imported successfully")
        
        import requests
        print("✓ requests imported successfully")
        
        import jupyter
        print("✓ jupyter imported successfully")
        
        from dotenv import load_dotenv
        print("✓ python-dotenv imported successfully")
        
        from tqdm import tqdm
        print("✓ tqdm imported successfully")
        
        import geopandas as gpd
        print("✓ geopandas imported successfully")
        
        import folium
        print("✓ folium imported successfully")
        
        import scipy
        print("✓ scipy imported successfully")
        
        import statsmodels
        print("✓ statsmodels imported successfully")
        
        print("\n🎉 All imports successful! Your environment is properly configured.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports() 