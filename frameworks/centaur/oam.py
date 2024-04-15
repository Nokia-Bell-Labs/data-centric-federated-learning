import argparse
import numpy as np
from typing import Tuple, List

np.random.seed(42)

#the client starts moving from home location and ends the journey at home location
def move_client(location_connectivity_probability: 'float array', temporal_granularity: int, home_loc: int) -> Tuple[List, List]:
    locs = []
    connectivity = []
    locs.append(home_loc)

    connectivity.append(location_connectivity_probability[home_loc])
    for i in range(temporal_granularity-2):
        val = np.random.randint(len(location_connectivity_probability))
        locs.append(val)
        connectivity.append(location_connectivity_probability[val])
    locs.append(home_loc)
    connectivity.append(location_connectivity_probability[home_loc])
    return locs,connectivity


def create_client_profile(num_clients=100, temporal='day', spatial=10) -> None:
    print("Creating client mobility profiles")
    locs_connect_proba = np.random.rand(spatial) #keeping this global across all clients
    
    #this is the home location with maximum connectivity available
    max_connect_loc = np.argmax(locs_connect_proba)
    time = None
    if(temporal == "months"):
        time = 12
    elif(temporal == "weeks"):
        time = 7
    elif(temporal == "day"):
        time = 24
    else:
        raise Exception("Granularity out of reach!")

    for i in range(num_clients):
        out_file = open("client_profiles/"+str(i)+".csv","w+")
        out_file.write("locs,probability\n")
        locs,connectivity = move_client(locs_connect_proba,time,max_connect_loc)
        for l,c in zip(locs,connectivity):
            out_file.write(str(l)+","+str(c)+"\n")
        out_file.close()