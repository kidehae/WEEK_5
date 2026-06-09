# # src/data_preprocessing.py
# import pandas as pd
# import numpy as np

# def ip_to_int(ip_address):
#     """Converts a string IPv4 address or float representation to an integer."""
#     if pd.isna(ip_address):
#         return np.nan
#     try:
#         # If it's already a float or integer representing the IP
#         if isinstance(ip_address, (int, float)):
#             return int(ip_address)
        
#         # Parse standard dot-decimal string "192.168.1.1"
#         parts = list(map(int, str(ip_address).split('.')))
#         if len(parts) == 4:
#             return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
#     except Exception:
#         return np.nan
#     return np.nan

# def merge_ip_to_country(fraud_df, ip_country_df):
#     """
#     Merges E-commerce fraud data with the IP range dataset using an optimized 
#     pandas merge_asof technique. Assumes ip_country_df is sorted by lower_bound_ip_address.
#     """
#     # Ensure IP addresses are clean integers
#     fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    
#     # Drop rows where IP conversion failed
#     fraud_df = fraud_df.dropna(subset=['ip_int']).copy()
#     fraud_df['ip_int'] = fraud_df['ip_int'].astype(int)
    
#     # Cast bounds to int for mapping
#     ip_country_df['lower_bound_ip_address'] = ip_country_df['lower_bound_ip_address'].astype(int)
#     ip_country_df['upper_bound_ip_address'] = ip_country_df['upper_bound_ip_address'].astype(int)
    
#     # Sort dataframes to satisfy merge_asof requirements
#     fraud_df = fraud_df.sort_values('ip_int')
#     ip_country_df = ip_country_df.sort_values('lower_bound_ip_address')
    
#     # Perform range-based lookup matching: fraud_df.ip_int >= ip_country_df.lower_bound_ip_address
#     merged_df = pd.merge_asof(
#         fraud_df,
#         ip_country_df,
#         left_on='ip_int',
#         right_on='lower_bound_ip_address',
#         direction='backward'
#     )
    
#     # Validate that the integer IP actually falls below or equal to the upper bound
#     valid_mask = merged_df['ip_int'] <= merged_df['upper_bound_ip_address']
#     merged_df.loc[~valid_mask, 'country'] = 'Unknown'
    
#     # Drop extra lookup columns to keep it clean
#     merged_df = merged_df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'])
    
#     return merged_df


# src/data_preprocessing.py
import pandas as pd
import numpy as np
import logging

# Configure basic logging to track pipeline anomalies
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ip_to_int(ip_address: str) -> float:
    """
    Converts a standard dot-decimal IPv4 address string to its 32-bit integer representation.

    Parameters:
    -----------
    ip_address : str
        The IPv4 string (e.g., '192.168.1.1') or a float value representing an IP.

    Returns:
    --------
    float:
        The computed integer value, or np.nan if conversion fails or input is missing.
    """
    if pd.isna(ip_address):
        return np.nan
    try:
        if isinstance(ip_address, (int, float)):
            return float(ip_address)
        
        parts = list(map(int, str(ip_address).split('.')))
        if len(parts) == 4:
            return float((parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3])
    except Exception as e:
        logging.warning(f"Failed to parse IP address parsing constraint: {ip_address}. Error: {e}")
        return np.nan
    return np.nan

def merge_ip_to_country(fraud_df: pd.DataFrame, ip_country_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges e-commerce transactions with country ranges using an optimized binary-search lookup.
    Validates data structural schemas and logs lookup anomalies.

    Parameters:
    -----------
    fraud_df : pd.DataFrame
        Transaction ledger containing an 'ip_address' feature column.
    ip_country_df : pd.DataFrame
        Mapping frame with 'lower_bound_ip_address', 'upper_bound_ip_address', and 'country'.

    Returns:
    --------
    pd.DataFrame:
        Enriched DataFrame containing a validated geographic 'country' assignment.
    """
    # 1. Lightweight defensive validation checks
    required_fraud = ['ip_address']
    required_country = ['lower_bound_ip_address', 'upper_bound_ip_address', 'country']
    
    for col in required_fraud:
        assert col in fraud_df.columns, f"Missing required column in transaction data: {col}"
    for col in required_country:
        assert col in ip_country_df.columns, f"Missing required column in country mapping data: {col}"

    # 2. IP Conversion
    df_copy = fraud_df.copy()
    df_copy['ip_int'] = df_copy['ip_address'].apply(ip_to_int)
    
    initial_rows = len(df_copy)
    df_copy = df_copy.dropna(subset=['ip_int'])
    dropped_rows = initial_rows - len(df_copy)
    if dropped_rows > 0:
        logging.info(f"Dropped {dropped_rows} rows due to unparseable IP addresses.")

    df_copy['ip_int'] = df_copy['ip_int'].astype(int)
    
    # 3. Type Standardization & Sorting for merge_asof
    ip_lookup = ip_country_df.copy()
    ip_lookup['lower_bound_ip_address'] = ip_lookup['lower_bound_ip_address'].astype(int)
    ip_lookup['upper_bound_ip_address'] = ip_lookup['upper_bound_ip_address'].astype(int)
    
    df_copy = df_copy.sort_values('ip_int')
    ip_lookup = ip_lookup.sort_values('lower_bound_ip_address')
    
    # 4. Range-based lookup matching
    merged_df = pd.merge_asof(
        df_copy,
        ip_lookup,
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    
    # 5. Validation Check: Ensure it falls below or equal to the upper bound
    valid_mask = merged_df['ip_int'] <= merged_df['upper_bound_ip_address']
    unmapped_count = (~valid_mask).sum()
    
    if unmapped_count > 0:
        logging.warning(f"Detected {unmapped_count} transactions matching lower bounds but exceeding upper bounds. Defaulting to 'Unknown'.")
        
    merged_df.loc[~valid_mask, 'country'] = 'Unknown'
    
    # Drop intermediate processing artifacts
    merged_df = merged_df.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'])
    
    return merged_df