from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
### Question - 1 
def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    for i in range(0, len(lst), n):
        result.extend(lst[i:i + n][::-1])
    return result

### Question - 2 
def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    result = {}
    for s in lst:
        key = len(s)
        if key not in result:
            result[key] = []
        result[key].append(s)
    return result

## Question- 3

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    def recurse(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(recurse(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    return recurse(nested_dict)

## Question - 4

import itertools
def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    return list(map(list, set(itertools.permutations(nums))))

## Question - 5
import re
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b' # yyyy.mm.dd
    ]
    
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    
    return dates

### Questions- 6
from geopy.distance import geodesic

def decode_polyline(polyline_str):
    # Dummy function to simulate polyline decoding
    # Replace this with an actual polyline decoding library or logic
    return [(lat, lon) for lat, lon in zip(range(10), range(10))] 
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into a list of (latitude, longitude) tuples
    points = decode_polyline(polyline_str)
    
    # Create lists for latitude and longitude
    latitudes = [point[0] for point in points]
    longitudes = [point[1] for point in points]
    
    # Calculate the distance between consecutive points
    distances = [0]  # No distance for the first point
    for i in range(1, len(points)):
        distances.append(geodesic(points[i-1], points[i]).meters)
    
    # Create a DataFrame with latitude, longitude, and distance columns
    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'distance': distances
    })
    
    return df

### Question-7
def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Transpose the matrix and reverse each row to rotate 90 degrees clockwise
    rotated_matrix = [list(row) for row in zip(*matrix[::-1])]
    
    # Multiply each element by the sum of its original row and column index
    result_matrix = []
    for i, row in enumerate(matrix):
        result_row = []
        for j, val in enumerate(row):
            result_row.append(rotated_matrix[i][j] * (i + j))
        result_matrix.append(result_row)
    
    return result_matrix

## Question- 8

def verify_completeness_of_timestamps(df: pd.DataFrame) -> pd.Series:

    def time_to_seconds(time_str):
        t = datetime.strptime(time_str, "%H:%M:%S").time()
        return t.hour * 3600 + t.minute * 60 + t.second
    df.set_index(['id', 'id_2'], inplace=True)
    
    # Convert days of the week into numerical values (assuming Monday=0 and Sunday=6)
    day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                   'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['startDay'] = df['startDay'].map(day_mapping)
    df['endDay'] = df['endDay'].map(day_mapping)
    
    # Convert times to seconds from 00:00:00
    df['startTime'] = df['startTime'].apply(time_to_seconds)
    df['endTime'] = df['endTime'].apply(time_to_seconds)
    
    incomplete_series = pd.Series(False, index=df.index.unique())

    # Iterate through each (id, id_2) group
    for (id_val, id_2_val), group in df.groupby(level=['id', 'id_2']):
        week_coverage = {day: np.zeros(24 * 3600, dtype=bool) for day in range(7)}  # 24 hours, 3600 seconds per hour
        
        for _, row in group.iterrows():
            start_day, end_day = row['startDay'], row['endDay']
            start_time, end_time = row['startTime'], row['endTime']
            
            if start_day == end_day:
                # If the start and end are on the same day, just mark that time
                week_coverage[start_day][start_time:end_time+1] = True
            else:
                # If it spans multiple days, mark from start to end of the first day
                week_coverage[start_day][start_time:] = True
                # Mark full days in between
                for day in range(start_day + 1, end_day):
                    week_coverage[day][:] = True
                # Mark from start to end time on the last day
                week_coverage[end_day][:end_time+1] = True
        
        # Check if all days of the week are fully covered (from 00:00:00 to 23:59:59)
        for day in range(7):
            if not np.all(week_coverage[day]):
                incomplete_series.loc[(id_val, id_2_val)] = True
                break
    
    return incomplete_series

dataset_1_path = 'D:\MapUp Assignment\MapUp-DA-Assessment-2024\datasets\dataset-1.csv'
dataset_1 = pd.read_csv(dataset_1_path)
result = verify_completeness_of_timestamps(dataset_1)
print(result)
