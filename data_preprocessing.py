import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_crop_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    return df

def clean_crop_data(df):
    print("Cleaning crop yield data...")
    
    # Missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Missing values found:\n{missing_values}")
    
    # Duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Convert Year to datetime
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    
    # Crop yield is numeric
    df['Crop Yield (tons/hectare)'] = pd.to_numeric(df['Crop Yield (tons/hectare)'], errors='coerce')
    
    # Remaining missing values
    if df['Crop Yield (tons/hectare)'].isnull().sum() > 0:
        # Fill missing crop yields with median for each country
        df['Crop Yield (tons/hectare)'] = df.groupby('Country')['Crop Yield (tons/hectare)'].transform(
            lambda x: x.fillna(x.median())
        )
        print(f"Filled {df['Crop Yield (tons/hectare)'].isnull().sum()} missing crop yield values")
    
    # Derived features
    df['Year_Numeric'] = df['Year'].dt.year
    df['Decade'] = (df['Year_Numeric'] // 10) * 10
    
    print(f"Data cleaning completed. Final shape: {df.shape}")
    return df

def encode_categorical_features(df):

    print("Encoding categorical features...")
    
    label_encoders = {}
    
    # Encode Country
    if 'Country' in df.columns:
        le_country = LabelEncoder()
        df['Country_Encoded'] = le_country.fit_transform(df['Country'])
        label_encoders['Country'] = le_country
        print(f"Encoded {len(le_country.classes_) if le_country.classes_ is not None else 0} countries")
    
    return df, label_encoders

def scale_numerical_features(df, numerical_columns):
    print("Scaling numerical features...")
    
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    print(f"Scaled {len(numerical_columns)} numerical features")
    return df, scaler

def create_time_series_features(df):
    print("Creating time series features...")
    
    # Country and year
    df = df.sort_values(['Country', 'Year_Numeric'])
    
    # Lag features for each country
    for lag in [1, 2, 3, 5]:
        df[f'Crop_Yield_Lag_{lag}'] = df.groupby('Country')['Crop Yield (tons/hectare)'].shift(lag)
    
    # Rolling mean features
    for window in [3, 5, 10]:
        df[f'Crop_Yield_Rolling_Mean_{window}'] = df.groupby('Country')['Crop Yield (tons/hectare)'].rolling(
            window=window, min_periods=1
        ).mean().reset_index(0, drop=True)
    
    # Year-over-year change
    df['Crop_Yield_YoY_Change'] = df.groupby('Country')['Crop Yield (tons/hectare)'].pct_change()
    
    print("Time series features created")
    return df

def split_crop_data(df, target_column='Crop Yield (tons/hectare)', test_size=0.2, random_state=42):
    print("Splitting data into training and testing sets...")
    
    # Sort by year to maintain temporal order
    df = df.sort_values('Year_Numeric')
    
    # Use the last 20% of years for testing
    unique_years = sorted(df['Year_Numeric'].unique())
    split_year = unique_years[int(len(unique_years) * (1 - test_size))]
    
    train_df = df[df['Year_Numeric'] < split_year]
    test_df = df[df['Year_Numeric'] >= split_year]
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col not in [
        target_column, 'Year', 'Country', 'Decade'
    ]]
    
    X_train = train_df[feature_columns].dropna()
    y_train = train_df.loc[X_train.index, target_column]
    
    X_test = test_df[feature_columns].dropna()
    y_test = test_df.loc[X_test.index, target_column]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    print(f"Feature columns: {len(feature_columns)}")
    
    return X_train, X_test, y_train, y_test, feature_columns

def analyze_crop_data(df):
    print("\n=== CROP YIELD DATA ANALYSIS ===")
    
    # Basic statistics
    print(f"\nDataset Overview:")
    print(f"Total records: {len(df)}")
    
    # Handle Year_Numeric if it exists, otherwise use Year
    if 'Year_Numeric' in df.columns:
        print(f"Time period: {df['Year_Numeric'].min()} - {df['Year_Numeric'].max()}")
    else:
        print(f"Time period: {df['Year'].min()} - {df['Year'].max()}")
    
    print(f"Countries: {df['Country'].nunique()}")
    print(f"Countries: {', '.join(df['Country'].unique())}")
    
    # Crop yield statistics
    print(f"\nCrop Yield Statistics (tons/hectare):")
    print(df['Crop Yield (tons/hectare)'].describe())
    
    # Country-wise statistics
    print(f"\nCrop Yield by Country (tons/hectare):")
    country_stats = df.groupby('Country')['Crop Yield (tons/hectare)'].agg(['mean', 'std', 'min', 'max'])
    print(country_stats.round(2))
    
    # Year-wise trends (only if Decade column exists)
    if 'Decade' in df.columns:
        print(f"\nCrop Yield Trends by Decade (tons/hectare):")
        decade_stats = df.groupby('Decade')['Crop Yield (tons/hectare)'].agg(['mean', 'std'])
        print(decade_stats.round(2))
    
    return df

def plot_crop_data(df):
    print("Creating data visualizations...")
    
    # Plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Crop yield distribution
    axes[0, 0].hist(df['Crop Yield (tons/hectare)'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Crop Yields')
    axes[0, 0].set_xlabel('Crop Yield (tons/hectare)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Crop yield by country
    country_means = df.groupby('Country')['Crop Yield (tons/hectare)'].mean().sort_values(ascending=True)
    axes[0, 1].barh(range(len(country_means)), country_means.values, color='lightgreen')
    axes[0, 1].set_yticks(range(len(country_means)))
    axes[0, 1].set_yticklabels(country_means.index)
    axes[0, 1].set_title('Average Crop Yield by Country')
    axes[0, 1].set_xlabel('Average Crop Yield (tons/hectare)')
    
    # Time series plot for a few countries
    sample_countries = df['Country'].unique()[:5]  # First 5 countries
    for country in sample_countries:
        country_data = df[df['Country'] == country]
        axes[1, 0].plot(country_data['Year_Numeric'], country_data['Crop Yield (tons/hectare)'], 
                       label=country, marker='o', markersize=3)
    axes[1, 0].set_title('Crop Yield Trends Over Time')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Crop Yield (tons/hectare)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot by country
    df.boxplot(column='Crop Yield (tons/hectare)', by='Country', ax=axes[1, 1], rot=45)
    axes[1, 1].set_title('Crop Yield Distribution by Country')
    axes[1, 1].set_xlabel('Country')
    axes[1, 1].set_ylabel('Crop Yield (tons/hectare)')
    
    plt.tight_layout()
    plt.savefig('Data/crop_yield_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'Data/crop_yield_analysis.png'")

def preprocess_crop_data(file_path='Data/crop_yield_data.csv'):
    print("Starting crop yield data preprocessing...")
    
    # Load data
    df = load_crop_data(file_path)
    
    # Clean data
    df = clean_crop_data(df)
    
    # Analyze data
    df = analyze_crop_data(df)
    
    # Time series features
    df = create_time_series_features(df)
    
    # Categorical features
    df, label_encoders = encode_categorical_features(df)
    
    # Numerical columns for scaling
    numerical_columns = [
        'Year_Numeric', 'Decade', 'Country_Encoded',
        'Crop_Yield_Lag_1', 'Crop_Yield_Lag_2', 'Crop_Yield_Lag_3', 'Crop_Yield_Lag_5',
        'Crop_Yield_Rolling_Mean_3', 'Crop_Yield_Rolling_Mean_5', 'Crop_Yield_Rolling_Mean_10',
        'Crop_Yield_YoY_Change'
    ]
    
    # Remove columns that don't exist
    numerical_columns = [col for col in numerical_columns if col in df.columns]
    
    # Scale numerical features
    df, scaler = scale_numerical_features(df, numerical_columns)
    
    # Split data
    X_train, X_test, y_train, y_test, feature_columns = split_crop_data(df)
    
    # Create visualizations
    plot_crop_data(df)
    
    print("\n=== PREPROCESSING COMPLETED ===")
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Number of features: {len(feature_columns)}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_columns': feature_columns,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'processed_df': df
    }

# Example usage
if __name__ == "__main__":
    # Run the complete preprocessing pipeline
    results = preprocess_crop_data('Data/crop_yield_data.csv')
    
    print("\nData preprocessing completed successfully!")
    print("Results available in 'results' dictionary")

