

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
   
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
        return df
    except Exception as e:
        print(f"✗ Error saat memuat dataset: {str(e)}")
        raise


def handle_missing_values(df):
 
    df_clean = df.copy()
    
    # Mengisi missing values untuk kolom teks dengan 'Unknown'
    df_clean['director'] = df_clean['director'].fillna('Unknown')
    df_clean['cast'] = df_clean['cast'].fillna('Unknown')
    df_clean['country'] = df_clean['country'].fillna('Unknown')
    
    # Drop rows dengan date_added yang missing
    df_clean = df_clean.dropna(subset=['date_added'])
    
    # Mengisi rating dengan modus
    df_clean['rating'] = df_clean['rating'].fillna(df_clean['rating'].mode()[0])
    
    # Mengisi duration jika ada missing
    df_clean['duration'] = df_clean['duration'].fillna('Unknown')
    
    print(f"✓ Missing values ditangani: {len(df_clean)} baris tersisa")
    
    return df_clean


def feature_engineering(df):
   
    df_features = df.copy()
    
    # Extract numeric duration
    df_features['duration_value'] = df_features['duration'].str.extract(r'(\d+)').astype(float)
    
    # Extract duration type
    df_features['duration_type'] = df_features['duration'].str.extract('(min|Season)', expand=False)
    
    # Parse date_added dengan handling whitespace
    df_features['date_added'] = df_features['date_added'].str.strip()
    df_features['date_added'] = pd.to_datetime(df_features['date_added'], format='mixed', errors='coerce')
    
    # Drop rows dengan date_added yang tidak bisa di-parse
    df_features = df_features.dropna(subset=['date_added'])
    
    # Extract year dan month
    df_features['year_added'] = df_features['date_added'].dt.year
    df_features['month_added'] = df_features['date_added'].dt.month
    
    # Hitung content age
    df_features['content_age'] = df_features['year_added'] - df_features['release_year']
    
    # Hitung jumlah genre
    df_features['num_genres'] = df_features['listed_in'].str.count(',') + 1
    
    # Hitung jumlah negara
    df_features['num_countries'] = df_features['country'].apply(
        lambda x: len(x.split(', ')) if x != 'Unknown' else 0
    )
    
    # Binary indicators
    df_features['has_director'] = (df_features['director'] != 'Unknown').astype(int)
    df_features['has_cast'] = (df_features['cast'] != 'Unknown').astype(int)
    
    # Description length
    df_features['description_length'] = df_features['description'].str.len()
    
    print(f"✓ Feature engineering selesai: {len(df_features)} baris, {df_features.shape[1]} kolom")
    
    return df_features


def encode_features(df):
    
    df_encoded = df.copy()
    encoders = {}
    
    # Encode target variable
    df_encoded['type_encoded'] = (df_encoded['type'] == 'Movie').astype(int)
    
    # Label encode rating
    le_rating = LabelEncoder()
    df_encoded['rating_encoded'] = le_rating.fit_transform(df_encoded['rating'])
    encoders['rating'] = le_rating
    
    # Label encode duration_type
    le_duration = LabelEncoder()
    df_encoded['duration_type_encoded'] = le_duration.fit_transform(
        df_encoded['duration_type'].fillna('Unknown')
    )
    encoders['duration_type'] = le_duration
    
    print(f"✓ Encoding selesai")
    
    return df_encoded, encoders


def scale_features(df):
   
    df_scaled = df.copy()
    
    # Fitur numerik yang akan di-scale
    numerical_features = [
        'release_year', 'year_added', 'month_added', 'content_age',
        'num_genres', 'num_countries', 'description_length', 'duration_value'
    ]
    
    scaler = StandardScaler()
    df_scaled[numerical_features] = scaler.fit_transform(df_scaled[numerical_features])
    
    print(f"✓ Feature scaling selesai")
    
    return df_scaled, scaler


def select_final_features(df):
   
    final_columns = [
        'release_year', 'rating_encoded', 'duration_value', 'duration_type_encoded',
        'year_added', 'month_added', 'content_age', 'num_genres', 'num_countries',
        'has_director', 'has_cast', 'description_length', 'type_encoded'
    ]
    
    df_final = df[final_columns].copy()
    
    print(f"✓ Seleksi fitur selesai: {len(final_columns)} kolom")
    
    return df_final


def preprocess_netflix_data(input_path, output_path=None):
    
    print("="*70)
    print("PREPROCESSING DATASET NETFLIX MOVIES AND TV SHOWS")
    print("="*70)
    
    # 1. Load data
    print("\n[1/6] Memuat dataset...")
    df = load_data(input_path)
    
    # 2. Handle missing values
    print("\n[2/6] Menangani missing values...")
    df = handle_missing_values(df)
    
    # 3. Feature engineering
    print("\n[3/6] Melakukan feature engineering...")
    df = feature_engineering(df)
    
    # 4. Encoding
    print("\n[4/6] Melakukan encoding...")
    df, encoders = encode_features(df)
    
    # 5. Scaling
    print("\n[5/6] Melakukan feature scaling...")
    df, scaler = scale_features(df)
    
    # 6. Select final features
    print("\n[6/6] Memilih fitur final...")
    df_final = select_final_features(df)
    
    # Save to file
    if output_path is None:
        output_path = 'netflix_titles_processed.csv'
    
    df_final.to_csv(output_path, index=False)
    print(f"\n✓ Dataset berhasil disimpan ke: {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total samples: {len(df_final)}")
    print(f"Total fitur: {df_final.shape[1] - 1} (+ 1 target)")
    print(f"Target distribution:")
    print(f"  - Movie (1): {(df_final['type_encoded'] == 1).sum()} ({(df_final['type_encoded'] == 1).sum() / len(df_final) * 100:.2f}%)")
    print(f"  - TV Show (0): {(df_final['type_encoded'] == 0).sum()} ({(df_final['type_encoded'] == 0).sum() / len(df_final) * 100:.2f}%)")
    print("="*70)
    
    return df_final


if __name__ == "__main__":
    # Contoh penggunaan
    input_file = "dataset_raw/netflix_titles.csv"
    output_file = "dataset_preprocessing/netflix_titles_processed.csv"

    # Jalankan preprocessing
    df_processed = preprocess_netflix_data(input_file, output_file)

    # Tampilkan sample data
    print("\nSample data (5 baris pertama):")
    print(df_processed.head())
