import googlemaps
from vrppd_api import vrppd_api_key, vrppd_init_code
import pandas as pd
import numpy as np
from networkx import DiGraph, from_numpy_matrix, relabel_nodes
import networkx as nx
import matplotlib.pyplot as plt
from numpy import array
from vrpy import VehicleRoutingProblem
import streamlit as st

def read_csv_file(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.groupby(['Pickup Address', 'Dropoff Address'])['Units'].sum().reset_index()
        return df
    else:
        st.write("Please upload a CSV file")  

def run_app(df, 
            uhaul_address, 
            keg_dropoff_address, 
            init_code, 
            vrppd_api_key, 
            load_capacity, 
            num_stops,
            correct_init_code):
    if init_code == correct_init_code:
        def read_data(file_path):
            return pd.read_csv(file_path)

        def prepare_location_data(df, uhaul_address):
            locations = [uhaul_address]
            for _, row in df.iterrows():
                locations.extend([row['Pickup Address'], row['Dropoff Address']])
            return locations

        def extract_pickups_deliveries(df):
            pickups_deliveries = {}
            address_dict = {}
            address_dict_reverse = {}
            counter = 0
            for _, row in df.iterrows():
                counter += 1
                pickup_index = counter  
                address_dict_reverse[pickup_index] = row['Pickup Address']
                address_dict = update_address_dict(address_dict, row['Pickup Address'], pickup_index)
                counter += 1
                dropoff_index = counter
                address_dict_reverse[dropoff_index] = row['Dropoff Address']
                address_dict = update_address_dict(address_dict, row['Dropoff Address'], dropoff_index)
                pickups_deliveries[(pickup_index, dropoff_index)] = row['Units']
            return pickups_deliveries, address_dict, address_dict_reverse

        def update_address_dict(address_dict, address, index):
            if address in address_dict:
                address_dict[address].append(index)
            else:
                address_dict[address] = [index]
            return address_dict

        def get_final_route(prob, address_dict_reverse, uhaul_address, keg_dropoff_address):
            addresses = []
            ids = []
            for key in prob.best_routes:
                for item in prob.best_routes[key]:
                    if isinstance(item, int):
                        addresses.append(address_dict_reverse[item])
                        ids.append(item)
            stops_to_remove = remove_stops(ids, address_dict_reverse, address_dict)
            final_route_ids = [i for i in ids if i not in stops_to_remove]
            final_route = [address_dict_reverse[key] for key in final_route_ids]
            final_route.insert(0, keg_dropoff_address)
            final_route.append(keg_dropoff_address)
            final_route.insert(0, uhaul_address)
            final_route.append(uhaul_address)
            return final_route

        def remove_stops(ids, address_dict_reverse, address_dict):
            stops_to_remove = []
            for i in range(len(ids)):
                list_before = ids[:i]
                list_after = ids[i+1:]
                if ids[i] % 2 == 0:
                    if any(element in list_after for element in address_dict[address_dict_reverse[ids[i]]]):
                        stops_to_remove.append(ids[i])            
                else:
                    if any(element in list_before for element in address_dict[address_dict_reverse[ids[i]]]):
                        stops_to_remove.append(ids[i])
            return stops_to_remove

        def generate_gmap_url(final_route):
            base_url = "https://www.google.com/maps/dir/"
            formatted_addresses = [address.replace(' ', '+') for address in final_route]
            locations = '/'.join(formatted_addresses)
            return base_url + locations

        def get_distance_matrix(src_locations, dest_locations, api_key=vrppd_api_key):
            gmaps = googlemaps.Client(key=api_key)  # Replace with your Google Maps API key

            # Request distance matrix
            distance_matrix = gmaps.distance_matrix(src_locations, dest_locations, mode='driving')  # Separate sources and destinations

            # Create matrix from response
            matrix = []
            for row in distance_matrix['rows']:
                matrix_row = []
                for element in row['elements']:
                    matrix_row.append(element['distance']['value'])  # get distance in meters
                matrix.append(matrix_row)
            
            return np.array(matrix)

        def chunked_distance_matrix(locations, chunk_size, key):
            n = len(locations)
            final_matrix = np.zeros((n, n))

            for i in range(0, n, chunk_size):
                for j in range(0, n, chunk_size):
                    chunk_src = locations[i:i+chunk_size]
                    chunk_dest = locations[j:j+chunk_size]
                    chunk_matrix = get_distance_matrix(chunk_src, chunk_dest, key)  # Separate the sources and destinations
                    final_matrix[i:i+len(chunk_src), j:j+len(chunk_dest)] = chunk_matrix

            return final_matrix

        if __name__ == "__main__":
            # replace with your actual file path and
            #uhaul_address = "1616 SE Van Skiver Rd, Port Orchard, WA 98367"
            #file_path = r"C:\Users\trent\OneDrive\Documents\GitHub\sandbox\vrppd_test.csv"

            # read and prepare data
            #df = read_data(file_path)
            locations = prepare_location_data(df, uhaul_address)
            pickups_deliveries, address_dict, address_dict_reverse = extract_pickups_deliveries(df)

            try:
                st.write("Getting distance matrix...")
                # get distance matrix
                distance_matrix = chunked_distance_matrix(locations, 10, vrppd_api_key).tolist()
            except:
                st.write("Error: Distance Matrix API call failed.")
            
            try:
                st.write("Building graph...")
                # manipulate distance_matrix
                for row in distance_matrix:
                    row.append(row.pop(0))
                for i in range(len(distance_matrix)):
                    distance_matrix[i] = [0] + distance_matrix[i]
                distance_matrix.append([0] * len(distance_matrix[0]))

                # create directed graph
                A = array(distance_matrix, dtype=[("cost", int)])
                G_d = from_numpy_matrix(A, create_using=DiGraph())
                G = relabel_nodes(G_d, {0: "Source", len(distance_matrix)-1: "Sink"})

                # add requests to the graph
                for (u, v) in pickups_deliveries:
                    G.nodes[u]["request"] = v
                    G.nodes[u]["demand"] = pickups_deliveries[(u, v)]
                    G.nodes[v]["demand"] = -pickups_deliveries[(u, v)]
            except:
                st.write("Error: Graph creation failed.")

            try:
            # solve the problem
                st.write("Solving for route. This may take a while, but there will be an error message if it fails...")
                prob = VehicleRoutingProblem(G, load_capacity=int(load_capacity), num_stops=int(num_stops), pickup_delivery=True)
                prob.solve(cspy=False)
                
                st.write("Raw Solution:")
                st.write(prob.best_routes)
                st.write(prob.node_load)

                # generate final route
                final_route = get_final_route(prob, address_dict_reverse, uhaul_address, keg_dropoff_address)
                st.write("Final Route:")
                for address in final_route:
                    #print(address)
                    st.write(address)

                # generate Google Maps URL
                final_url = generate_gmap_url(final_route)
                #print(final_url)
                st.write(final_url)
            except:
                st.write("Error: No solution found.")
    else:
        st.write("Error: Code not recognized. Please try again.")

st.title('Distributor VRPPD App')

# Initial condition input
init_code = st.text_input("Code")

# UHaul address input
uhaul_address = st.text_input("Start/Finish Address", "1616 SE Van Skiver Rd, Port Orchard, WA 98367")

# Empty keg dropoff address input
keg_dropoff_address = st.text_input("Empty Keg Pickup/Dropoff Address", "120 Harrison Ave, Port Orchard, WA 98366")

# CSV file upload
csv_file = st.file_uploader("Upload CSV", type=['csv'])

if st.button('Run'):
    df = read_csv_file(csv_file)
    
    # Calculate the sum of the 'Units' column
    units_sum = df['Units'].sum()
    
    # Calculate the total combined count of values in 'Pickup Address' and 'Dropoff Address' columns
    total_count = df['Pickup Address'].count() + df['Dropoff Address'].count()
    
    # Calculate the unique combined count of values in 'Pickup Address' and 'Dropoff Address' columns
    unique_count = pd.concat([df['Pickup Address'], df['Dropoff Address']]).nunique()
    
    st.write(f"A total of {units_sum} kegs are being delivered to {unique_count} unique locations.")
    
    st.write(f"Route optimization is being run on {total_count} total stops.")
    
    run_app(df, 
            uhaul_address, 
            keg_dropoff_address, 
            init_code, 
            vrppd_api_key, 
            units_sum, 
            total_count,
            vrppd_init_code)