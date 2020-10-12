import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def find_nearest_N_hotels(hotels_dataframe, N):
    """
    Finds the nearest N hotels for each hotel in the given dataframe.
    Parameters:
    - hotels_dataframe, pandas dataframe with columns hotel_id, longitude,
    latitude
    - N, integer number of nearest hotels required
    Returns:
    Pandas dataframe with columns hotel_id, nearest_hotels (as list)
    """
    # Convert lat and long to radians
    lat_longs_rads = np.radians(
        hotels_dataframe[['latitude', 'longitude']].astype(np.float).values
    )

    # Initialise Ball Tree using haversine metric to calculate distance
    ball = BallTree(lat_longs_rads, metric='haversine')

    # Query includes itself so need to find nearest N + 1
    k = N + 1

    # Returns 2d array of nearest neighbours indices including self
    indices = ball.query(lat_longs_rads, k=k, return_distance=False)
    nearest_hotels_idx = indices[:, 1:]

    # Convert to list of N nearest hotel_ids
    nearest_hotels_ids = pd.Series(nearest_hotels_idx.tolist()).map(
        lambda nearest_idx: list(
            map(
                lambda idx: hotels_dataframe.iloc[idx]['hotel_id'],
                nearest_idx
            )
        )
    )

    return pd.DataFrame(
        np.column_stack(
            (hotels_dataframe['hotel_id'].values, nearest_hotels_ids)
        ),
        columns=('hotel_id', 'nearest_hotel_ids')
    )
