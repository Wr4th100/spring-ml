# # import pandas as pd
# # from datetime import datetime

# # # Read the original CSV file
# # df = pd.read_csv('C:\\Users\\Roshan\\Downloads\\madurai.csv')

# # # Define a mapping of month names to month numbers
# # month_mapping = {
# #     'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
# #     'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
# #     'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
# # }

# # # Melt the dataframe to combine columns 'JAN' to 'DEC' into a single 'Month' column
# # melted_df = pd.melt(df, id_vars=['DISTRICT', 'YEAR'], value_vars=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
# #                     var_name='Month', value_name='Rainfall')

# # # Map month names to month numbers
# # melted_df['Month'] = melted_df['Month'].map(month_mapping)

# # # Create a new 'Date' column by combining 'YEAR', 'Month', and '01' (day) columns
# # melted_df['Date'] = melted_df.apply(lambda row: datetime(row['YEAR'], row['Month'], 1), axis=1)

# # # Drop unnecessary columns
# # melted_df = melted_df.drop(['YEAR', 'Month'], axis=1)

# # # Write the new dataframe to a new CSV file
# # melted_df.to_csv('output_file.csv', index=False)

# import pandas as pd
# import numpy as np
# from datetime import datetime

# # Read the original CSV file
# df = pd.read_csv('C:\\Users\\Roshan\\Downloads\\madurai.csv')

# # Define a mapping of month names to month numbers
# month_mapping = {
#     'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
#     'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
#     'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
# }

# # Melt the dataframe to combine columns 'JAN' to 'DEC' into a single 'Month' column
# melted_df = pd.melt(df, id_vars=['DISTRICT', 'YEAR'], value_vars=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
#                     var_name='Month', value_name='Rainfall')

# # Map month names to month numbers
# melted_df['Month'] = melted_df['Month'].map(month_mapping)

# # Create sine and cosine components for the month
# melted_df['Month_sin'] = np.sin(2 * np.pi * melted_df['Month'] / 12)
# melted_df['Month_cos'] = np.cos(2 * np.pi * melted_df['Month'] / 12)

# # Create a new 'Date' column by combining 'YEAR', 'Month', and '01' (day) columns
# melted_df['Date'] = melted_df.apply(lambda row: datetime(row['YEAR'], row['Month'], 1), axis=1)

# # Drop unnecessary columns
# melted_df = melted_df.drop(['YEAR', 'Month'], axis=1)

# # Write the new dataframe to a new CSV file
# melted_df.to_csv('output_file.csv', index=False)
