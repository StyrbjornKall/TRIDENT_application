import streamlit as st
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Counter file path
counter_file = 'access_count.txt'

# Function to read the counter from the file
def read_counter():
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            count = int(f.read().strip())
    else:
        count = 0
    return count

# Function to update the counter in the file
def update_counter():
    count = read_counter() + 1
    with open(counter_file, 'w') as f:
        f.write(str(count))
    return count

def increment_user_stats():
    # Check if the user has already been counted this session
    if 'has_accessed' not in st.session_state:
        # This is the first access in this session
        st.session_state['has_accessed'] = True
        
        # Update the access counter
        count = update_counter()
        
        # Log the access count to Docker logs
        logger.info(f"App accessed {count} times.")
    else:
        # Do not increment the counter for subsequent interactions
        count = read_counter()