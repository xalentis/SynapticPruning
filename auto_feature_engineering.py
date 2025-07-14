# Gideon Vos 2025 - James Cook University, Australia
# www.linkedin.com/in/gideonvos
# www.github.com/xalentis
# www.xalentis.com


import re
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from typing import List, Optional, Dict, Any, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose


def _handle_missing_values(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    verbose: bool
) -> pd.DataFrame:
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            if col in metadata['numeric_columns'] or any(col.startswith(f"{target}_lag") or col.startswith(f"{target}_lead") for target in df.columns):
                # for lag/lead features, use forward fill then backward fill
                if any(col.startswith(f"{target}_lag") or col.startswith(f"{target}_lead") for target in df.columns):
                    df[col] = df[col].ffill()
                    df[col] = df[col].bfill()
                    # If still nan, use mean
                    if df[col].isna().sum() > 0:
                        df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mean())
                if verbose:
                    print(f"Filled {missing_count} missing values in {col}")
            elif col in metadata['categorical_columns']:
                # categorical, use mode
                if not df[col].empty:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
                else:
                    df[col] = df[col].fillna(0)
                if verbose:
                    print(f"Filled {missing_count} missing values in {col} with mode")
            else:
                # for other columns, use forward fill then backward fill
                df[col] = df[col].fillna(method='ffill')
                df[col] = df[col].fillna(method='bfill')
                # If nan, use 0
                if df[col].isna().sum() > 0:
                    df[col] = df[col].fillna(0)
                if verbose:
                    print(f"Filled {missing_count} missing values in {col}")
    return df


def detect_lag_lead_periods(
    df: pd.DataFrame,
    target_column: str,
    date_column: str,
    min_lag: int = 1,
    max_lag: int = 30,
    min_lead: int = 1,
    max_lead: int = 15,
    max_periods: int = 5,
    correlation_threshold: float = 0.2,
    freq: Optional[str] = None,
    seasonality_test: bool = True,
    verbose: bool = False
) -> Tuple[List[int], List[int]]:
    
    df_copy = df.copy()
    if target_column not in df_copy.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    if date_column not in df_copy.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")
    
    # convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
        
    # drop rows with NaN in target or date columns
    df_copy = df_copy.dropna(subset=[target_column, date_column])
    
    # sort by date
    df_copy = df_copy.sort_values(by=date_column)
    
    # convert target to numeric if it's not already
    if not pd.api.types.is_numeric_dtype(df_copy[target_column]):
        try:
            df_copy[target_column] = pd.to_numeric(df_copy[target_column], errors='coerce')
            df_copy = df_copy.dropna(subset=[target_column])
        except:
            raise ValueError(f"Target column '{target_column}' cannot be converted to numeric")
    
    # set index to date for time series analysis
    df_ts = df_copy.set_index(date_column)
    series = df_ts[target_column]
    
    # try to infer frequency if not provided
    if freq is None:
        try:
            freq = pd.infer_freq(df_ts.index)
            if freq is None:
                # can't infer exact frequency, so try a reasonable default based on time delta
                time_delta = (df_ts.index[-1] - df_ts.index[0]) / len(df_ts)
                days = time_delta.total_seconds() / (24 * 3600)
                if days < 0.5:  # < 12 hours
                    freq = 'H'  # hourly
                elif days < 2:  # < 2 days
                    freq = 'D'  # daily
                elif days < 8:  # < 8 days
                    freq = 'W'  # wekly
                elif days < 35:  # < 35 days
                    freq = 'M'  # mnthly
                else:
                    freq = 'Q'  # quarterly
                    
            if verbose:
                print(f"Inferred frequency: {freq}")
        except:
            freq = 'D'  # default to daily
            if verbose:
                print(f"Could not infer frequency, defaulting to: {freq}")
    
    # init lag and lead periods
    lag_periods = []
    lead_periods = []
    
    # seasonality detection
    seasonal_periods = []
    if seasonality_test:
        try:
            # try to detect seasonality with decomposition
            decomposition = seasonal_decompose(series, model='additive', period=min(len(series)//2, 365))
            seasonal = decomposition.seasonal
            
            # find peaks in the seasonal component
            seasonal_diff = np.diff(seasonal)
            sign_changes = np.where(np.diff(np.signbit(seasonal_diff)))[0]
            if len(sign_changes) > 1:
                # distances between peaks
                peak_distances = np.diff(sign_changes)
                # most common distances (potential seasonal periods)
                unique_distances, counts = np.unique(peak_distances, return_counts=True)
                # Sort by count (most common first)
                sorted_indices = np.argsort(-counts)
                
                # get top seasonal periods
                for idx in sorted_indices[:3]:  # top 3
                    period = unique_distances[idx]
                    if period >= min_lag and period <= max_lag:
                        seasonal_periods.append(int(period))
                        
            # common business periods if not found
            common_periods = {
                'D': [1, 7, 14, 30, 90],
                'W': [1, 4, 12, 26, 52],
                'M': [1, 3, 6, 12],
                'Q': [1, 2, 4],
                'H': [1, 24, 168]
            }
            
            freq_key = freq[0] if len(freq) > 0 else 'D'
            if freq_key in common_periods:
                for period in common_periods[freq_key]:
                    if period >= min_lag and period <= max_lag and period not in seasonal_periods:
                        seasonal_periods.append(period)
            if verbose and seasonal_periods:
                print(f"Detected seasonal periods: {seasonal_periods}")
        except Exception as e:
            if verbose:
                print(f"Seasonality detection failed: {str(e)}")
    
    # autocorrelationfor lag identification
    try:
        acf_values = acf(series, nlags=max_lag, fft=True)
        
        # get significant lags based on correlation threshold
        significant_lags = [(i, abs(acf_values[i])) for i in range(min_lag, min(len(acf_values), max_lag + 1))]
        significant_lags = [lag for lag, corr in significant_lags if corr > correlation_threshold]
        
        # add seasonal periods to significant lags if not already included
        for period in seasonal_periods:
            if period not in significant_lags:
                significant_lags.append(period)
        
        # sort by correlation strength
        lag_corr = [(lag, abs(acf_values[lag])) for lag in significant_lags if lag < len(acf_values)]
        lag_corr.sort(key=lambda x: x[1], reverse=True)
        
        # select top lags
        lag_periods = [lag for lag, _ in lag_corr[:max_periods]]
        
        if verbose:
            print(f"Detected significant lags: {lag_periods}")
    except Exception as e:
        if verbose:
            print(f"ACF computation failed: {str(e)}")
        lag_periods = seasonal_periods[:max_periods]
    
    # cross-correlation for lead detection
    try:
        # create shifted versions of the target for cross-correlation
        for lead in range(min_lead, max_lead + 1):
            df_copy[f'lead_{lead}'] = df_copy[target_column].shift(-lead)
        
        # calculate correlation between target and leads
        lead_corr = []
        for lead in range(min_lead, max_lead + 1):
            lead_col = f'lead_{lead}'
            if lead_col in df_copy.columns:
                corr = df_copy[[target_column, lead_col]].corr().iloc[0, 1]
                if not pd.isna(corr) and abs(corr) > correlation_threshold:
                    lead_corr.append((lead, abs(corr)))
        
        # srt by correlation and select top leads
        lead_corr.sort(key=lambda x: x[1], reverse=True)
        lead_periods = [lead for lead, _ in lead_corr[:max_periods]]
        if verbose:
            print(f"Detected significant leads: {lead_periods}")
    except Exception as e:
        if verbose:
            print(f"Lead detection failed: {str(e)}")
        lead_periods = list(range(min_lead, min(max_lead, 4)))[:max_periods]
    
    # if no significant lags/leads found, use defaults
    if not lag_periods:
        lag_periods = [1, 7, 14] if max_lag >= 14 else list(range(min_lag, min(max_lag + 1, 4)))
        if verbose:
            print(f"No significant lags found, using defaults: {lag_periods}")
    
    if not lead_periods:
        lead_periods = [1, 2, 3][:max_periods]
        if verbose:
            print(f"No significant leads found, using defaults: {lead_periods}")
    
    # returned periods must be within bounds
    lag_periods = [int(lag) for lag in lag_periods if min_lag <= lag <= max_lag]
    lead_periods = [int(lead) for lead in lead_periods if min_lead <= lead <= max_lead]
    
    # final sort
    lag_periods.sort()
    lead_periods.sort()
    
    return lag_periods, lead_periods


def auto_feature_engineering(
    df: pd.DataFrame,
    date_columns: Optional[List[str]] = None,
    categorical_threshold: int = 10,
    sequential_date_columns: Optional[List[str]] = None,
    lag_periods: List[int] = [1, 3, 7, 14],
    lead_periods: List[int] = [1, 3],
    target_column: Optional[str] = None,
    scale_features: bool = True,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    df_processed = df.copy()
    # store optional metadata
    metadata = {
        'categorical_columns': [],
        'numeric_columns': [],
        'date_columns': [],
        'original_columns': list(df.columns),
        'created_features': [],
        'encoders': {},
        'scalers': {},
        'column_dtypes': {},
    }
    df_processed, metadata = _identify_and_convert_types(
        df_processed, date_columns, categorical_threshold, metadata, verbose
    )
    df_processed, metadata = _process_date_columns(
        df_processed, metadata, verbose
    )
    if sequential_date_columns is not None:
        df_processed, metadata = _create_sequential_features(
            df_processed, sequential_date_columns, lag_periods, 
            lead_periods, target_column, metadata, verbose
        )
    df_processed = _handle_missing_values(df_processed, metadata, verbose)
    if scale_features:
        df_processed, metadata = _scale_features(df_processed, metadata, verbose)
    
    # all columns should be numeric at this point
    for col in df_processed.columns:
        if not is_numeric_dtype(df_processed[col]):
            if verbose:
                print(f"Converting column {col} to numeric as final check")
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())

    return df_processed, metadata


def _identify_and_convert_types(
    df: pd.DataFrame, 
    date_columns: Optional[List[str]], 
    categorical_threshold: int,
    metadata: Dict[str, Any],
    verbose: bool
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    for col in df.columns:
        metadata['column_dtypes'][col] = str(df[col].dtype)
    
    # auto-detect date columns if not explicitly provided
    detected_date_cols = []
    if date_columns is None:
        date_columns = []
        # find date columns by name pattern
        date_pattern = re.compile(r'date|time|day|month|year|dt_', re.IGNORECASE)
        for col in df.columns:
            if date_pattern.search(col):
                detected_date_cols.append(col)
        # parse strings as dates
        for col in df.select_dtypes(include=['object']).columns:
            # skip columns with high cardinality
            if col not in detected_date_cols and df[col].nunique() < 1000:
                try:
                    pd.to_datetime(df[col], errors='raise')
                    detected_date_cols.append(col)
                except:
                    pass
        date_columns.extend(detected_date_cols)
        date_columns = list(set(date_columns))  # duplicates
    
    # date columns
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if verbose:
                    print(f"Converted {col} to datetime")
                metadata['date_columns'].append(col)
            except:
                if verbose:
                    print(f"Failed to convert {col} to datetime")
    
    # categorical columns
    for col in df.columns:
        if col not in metadata['date_columns']:
            n_unique = df[col].nunique()
            if n_unique <= categorical_threshold or df[col].dtype == 'object':
                try:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    metadata['encoders'][col] = encoder
                    metadata['categorical_columns'].append(col)
                    if verbose:
                        print(f"Encoded categorical column {col} with {n_unique} values")
                except Exception as e:
                    if verbose:
                        print(f"Failed to encode {col}: {e}")
            else:
                if not np.issubdtype(df[col].dtype, np.number):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if verbose:
                            print(f"Converted {col} to numeric")
                    except:
                        if verbose:
                            print(f"Failed to convert {col} to numeric")
                metadata['numeric_columns'].append(col)
    return df, metadata


def _process_date_columns(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    verbose: bool
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    for col in metadata['date_columns']:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_week"] = df[col].dt.isocalendar().week
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_quarter"] = df[col].dt.quarter
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_is_weekend"] = (df[col].dt.dayofweek >= 5).astype(int)
            new_features = [
                f"{col}_day", f"{col}_month", f"{col}_week", 
                f"{col}_year", f"{col}_quarter", f"{col}_dayofweek",
                f"{col}_is_weekend"
            ]
            metadata['created_features'].extend(new_features)
            metadata['numeric_columns'].extend(new_features)
            if verbose:
                print(f"Created date features from {col}: {', '.join(new_features)}")
    return df, metadata


def _create_sequential_features(
    df: pd.DataFrame,
    sequential_date_columns: List[str],
    lag_periods: List[int],
    lead_periods: List[int],
    target_column: Optional[str],
    metadata: Dict[str, Any],
    verbose: bool
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    valid_seq_cols = [col for col in sequential_date_columns if col in df.columns]
    if not valid_seq_cols:
        if verbose:
            print("No valid sequential date columns found")
        return df, metadata
    
    for date_col in valid_seq_cols:
        # sort if the column is a datetime
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df = df.sort_values(by=date_col)
            if target_column is not None and target_column in df.columns:
                for lag in lag_periods:
                    lag_col = f"{target_column}_lag_{lag}"
                    df[lag_col] = df[target_column].shift(lag)
                    metadata['created_features'].append(lag_col)
                    metadata['numeric_columns'].append(lag_col)
                    if verbose:
                        print(f"Created lag feature: {lag_col}")
                for lead in lead_periods:
                    lead_col = f"{target_column}_lead_{lead}"
                    df[lead_col] = df[target_column].shift(-lead)
                    metadata['created_features'].append(lead_col)
                    metadata['numeric_columns'].append(lead_col)
                    if verbose:
                        print(f"Created lead feature: {lead_col}")
    
    return df, metadata


def _scale_features(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    verbose: bool
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    # scale only numeric columns
    numeric_cols = [col for col in metadata['numeric_columns'] 
                   if col not in metadata['categorical_columns']]
    
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        metadata['scaler'] = scaler
        if verbose:
            print(f"Scaled {len(numeric_cols)} numeric features")
    return df, metadata


def create_seasonal_indicators(
    df: pd.DataFrame,
    target_column: str,
    date_column: str,
    seasonal_periods: List[int],
    window_size: int = 3
) -> pd.DataFrame:

    df = df.sort_values(by=date_column).copy()
    
    for period in seasonal_periods:
        rolling_mean = df[target_column].rolling(window=window_size, center=True).mean()
        seasonal_rolling_mean = df[target_column].rolling(window=period, center=False).mean()
        seasonal_rolling_std = df[target_column].rolling(window=period, center=False).std()
        
        df[f'above_seasonal_mean_{period}'] = (
            (df[target_column] > seasonal_rolling_mean) & 
            (df[target_column] > rolling_mean)
        ).astype(int)
        df[f'below_seasonal_mean_{period}'] = (
            (df[target_column] < seasonal_rolling_mean) & 
            (df[target_column] < rolling_mean)
        ).astype(int)
        df[f'seasonal_zscore_{period}'] = (
            (df[target_column] - seasonal_rolling_mean) / 
            seasonal_rolling_std.replace(0, 1)
        )
        df[f'seasonal_rise_{period}'] = (
            df[target_column] > df[target_column].shift(period)
        ).astype(int)
        
        df[f'seasonal_fall_{period}'] = (
            df[target_column] < df[target_column].shift(period)
        ).astype(int)
    
    for col in df.columns:
        if col.startswith(('above_seasonal', 'below_seasonal', 'seasonal_rise', 'seasonal_fall')):
            df[col] = df[col].fillna(0)
        elif col.startswith('seasonal_zscore'):
            df[col] = df[col].fillna(0)          
    return df