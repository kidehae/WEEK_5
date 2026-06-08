# src/data_preprocessing.py
import pandas as pd
import numpy as np

def ip_to_int(ip_address):
    """Converts a string IPv4 address or float representation to an integer."""
    if pd.isna(ip_address):
        return np.nan
    try:
        # If it's already a float or integer representing the IP
        if isinstance(ip_address, (int, float)):
            return int(ip_address)
        
        # Parse standard dot-decimal string "192.168.1.1"
        parts = list(map(int, str(ip_address).split('.')))
        if len(parts) == 4:
            return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except Exception:
        return np.nan
    return np.nan

def merge_ip_to_country(fraud_df, ip_country_df):
    """
    Merges E-commerce fraud data with the IP range dataset using an optimized 
    pandas merge_asof technique. Assumes ip_country_df is sorted by lower_bound_ip_address.
    """
    # Ensure IP addresses are clean integers
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    
    # Drop rows where IP conversion failed
    fraud_df = fraud_df.dropna(subset=['ip_int']).copy()
    fraud_df['ip_int'] = fraud_df['ip_int'].astype(int)
    
    # Cast bounds to int for mapping
    ip_country_df['lower_bound_ip_address'] = ip_country_df['lower_bound_ip_address'].astype(int)
    ip_country_df['upper_bound_ip_address'] = ip_country_df['upper_bound_ip_address'].astype(int)
    
    # Sort dataframes to satisfy merge_asof requirements
    fraud_df = fraud_df.sort_values('ip_int')
    ip_country_df = ip_country_df.sort_values('lower_bound_ip_address')
    
    # Perform range-based lookup matching: fraud_df.ip_int >= ip_country_df.lower_bound_ip_address
    merged_df = pd.merge_asof(
        fraud_df,
        ip_country_df,
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    
    # Validate that the integer IP actually falls below or equal to the upper bound
    valid_mask = merged_df['ip_int'] <= merged_df['upper_bound_ip_address']
    merged_df.loc[~valid_mask, 'country'] = 'Unknown'
    
    # Drop extra lookup columns to keep it clean
    merged_df = merged_df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'])
    
    return merged_df