import pandas as pd
import numpy as np
dataset_1_path='D:\MapUp Assignment\MapUp-DA-Assessment-2024\datasets\dataset-1.csv'
dataset_2_path='D:\MapUp Assignment\MapUp-DA-Assessment-2024\datasets\dataset-2.csv'
dataset_1 = pd.read_csv(dataset_1_path)
dataset_2 = pd.read_csv(dataset_2_path)

## Question -9 
def calculate_distance_matrix(dataset_2):
    toll_ids = sorted(set(dataset_2['id_start']).union(set(dataset_2['id_end'])))
    
    #an empty distance matrix with zeros
    distance_matrix = pd.DataFrame(0, index=toll_ids, columns=toll_ids, dtype=float)
   
    # distance matrix with the given distances
    for _, row in dataset_2.iterrows():
        distance_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.loc[row['id_end'], row['id_start']] = row['distance']
    
    # Perform Floyd-Warshall algorithm for cumulative distances
    n = len(toll_ids)
    for k in toll_ids:
        for i in toll_ids:
            for j in toll_ids:
                if distance_matrix.loc[i, j] == 0 and i != j:
                    distance_matrix.loc[i, j] = np.inf  # Assign infinity where no direct route exists
                distance_matrix.loc[i, j] = min(distance_matrix.loc[i, j], distance_matrix.loc[i, k] + distance_matrix.loc[k, j])
    
    # Ensure diagonal elements are 0 (distance from a toll to itself)
    np.fill_diagonal(distance_matrix.values, 0)
    
    return distance_matrix

#cumulative distance matrix
cumulative_distance_matrix = calculate_distance_matrix(dataset_2)
print(cumulative_distance_matrix)

## Question - 10
def unroll_distance_matrix(distance_matrix):
    # Initialize an empty list
    unrolled_data = []

    for i, id_start in enumerate(distance_matrix.index):
        for j, id_end in enumerate(distance_matrix.columns):
            if id_start != id_end:
                distance = distance_matrix.iloc[i, j]  # Get the distance value
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Convert the list into a DataFrame
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df
unrolled_distance_matrix = unroll_distance_matrix(cumulative_distance_matrix)

print(unrolled_distance_matrix)



# ## Question-11
def find_ids_within_ten_percentage_threshold(dataset_2: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): DataFrame containing distance data with columns ['id_start', 'id_end', 'distance'].
        reference_id (int): The reference ID for comparison.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Calculate average distance for the reference ID
    avg_reference_distance = dataset_2[dataset_2['id_start'] == reference_id]['distance'].mean()
    
    # Calculate the percentage threshold
    lower_bound = avg_reference_distance * 0.90  # 10% below
    upper_bound = avg_reference_distance * 1.10  # 10% above

    avg_distances = dataset_2.groupby('id_start')['distance'].mean().reset_index()
    avg_distances = avg_distances[(avg_distances['distance'] >= lower_bound) & 
                                   (avg_distances['distance'] <= upper_bound)]
    
    return avg_distances
# Example
reference_id = 1001400
result = find_ids_within_ten_percentage_threshold(dataset_2, reference_id)
print(f"IDs within 10% distance of ID {reference_id}:")
print(result)

### Question-12
 
def calculate_toll_rates_from_unrolled_df(unrolled_distance_matrix):
    # Calculate toll rates for each vehicle type based on the distance
    unrolled_distance_matrix['moto'] = unrolled_distance_matrix['distance'] * 0.8
    unrolled_distance_matrix['car'] = unrolled_distance_matrix['distance'] * 1.2
    unrolled_distance_matrix['rv'] = unrolled_distance_matrix['distance'] * 1.5
    unrolled_distance_matrix['bus'] = unrolled_distance_matrix['distance'] * 2.2
    unrolled_distance_matrix['truck'] = unrolled_distance_matrix['distance'] * 3.6
    
    return unrolled_distance_matrix
toll_rates_df = calculate_toll_rates_from_unrolled_df(unrolled_distance_matrix)

print(toll_rates_df)

### Question -13 
import datetime

def calculate_time_based_toll_rates(toll_rates_df):
    # Days of the week
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']

    # Time intervals for weekdays
    weekday_intervals = [
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0), 0.8),
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0), 1.2),
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59), 0.8)
    ]
    # Time intervals for weekends (constant factor 0.7)
    weekend_interval = [(datetime.time(0, 0, 0), datetime.time(23, 59, 59), 0.7)]
    # Create a list to store rows before concatenating them into a DataFrame
    result_rows = []
    
    for idx, row in toll_rates_df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']

        base_rates = row[['moto', 'car', 'rv', 'bus', 'truck']]

        for day in weekdays:
            for start_time, end_time, discount in weekday_intervals:
                # Calculate new toll rates based on the discount factor
                modified_rates = base_rates * discount
                new_row = {
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    'moto': modified_rates['moto'],
                    'car': modified_rates['car'],
                    'rv': modified_rates['rv'],
                    'bus': modified_rates['bus'],
                    'truck': modified_rates['truck']
                }
                # Append the new row to the result list
                result_rows.append(new_row)

        # Generate entries for weekends
        for day in weekends:
            for start_time, end_time, discount in weekend_interval:
                # Calculate new toll rates based on the weekend discount factor
                modified_rates = base_rates * discount
                new_row = {
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    'moto': modified_rates['moto'],
                    'car': modified_rates['car'],
                    'rv': modified_rates['rv'],
                    'bus': modified_rates['bus'],
                    'truck': modified_rates['truck']
                }
                # Append the new row to the result list
                result_rows.append(new_row)

    # Convert the result list to a DataFrame using (pd.dataframe)
    result_df = pd.DataFrame(result_rows)
    return result_df
print(calculate_time_based_toll_rates(toll_rates_df))
