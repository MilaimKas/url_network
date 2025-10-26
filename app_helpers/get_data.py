"""
Plcaeholder for function to get data from database or other sources
"""

from app_helpers import helpers
from scipy.stats import dirichlet, poisson
from datetime import timedelta
import pandas as pd
import numpy as np
from random import randint, seed


def generate_dummy_df(n_sessions=100, min_pages_per_session=2, max_pages_per_session=6, url_sample=None, seed_value=42, choice_prob=None):
    """
    Generate a DataFrame that simulates user sessions, where each row corresponds
    to a session with:
        - a list of URLs visited in order,
        - and time spent per URL (chrono).
    
    Returns:
        pd.DataFrame with columns:
            - session_id
            - url_path: list of full URLs
            - chrono: list of times spent per URL (float)
    """
    seed(seed_value)
    np.random.seed(seed_value)

    # define start and end dates for sessions
    start_date = pd.to_datetime("2023-01-01")
    end_date = pd.to_datetime("2023-03-01")

    # Fake hierarchical URL fragments to construct full URLs
    if url_sample is None:
        url_sample = helpers.url_sample
    
    # Use Dirichlet distribution to generate probabilities of choosing each URL
    if choice_prob is None:
        choice_prob = dirichlet.rvs(poisson.rvs(200, size=len(url_sample)))[0]
    elif isinstance(choice_prob, (list, np.ndarray)):
        if len(choice_prob) != len(url_sample):
            raise ValueError("Length of choice_prob must match length of url_sample")
        if not np.isclose(sum(choice_prob), 1):
            raise ValueError("choice_prob must sum to 1")
    else:
        raise ValueError("choice_prob must be None or a list/array of probabilities")

    sessions = []
    for i in range(n_sessions):
        session_id = f"session_{i:04d}"
        session_length = randint(min_pages_per_session, max_pages_per_session)

        url_path = []
        url_node = []  # To store the last part of the URL as node
        for _ in range(session_length):
            url = np.random.choice(url_sample, p=choice_prob)    
            url_path.append(url)
            url_node.append(url.split("/")[-1])  # Extract the last part of the URL as node

        # Simulate time spent (e.g., exponential time-on-page)
        chrono = np.round(np.random.exponential(scale=20, size=session_length), 2)
        
        # construct random date column
        random_days = randint(0, (end_date - start_date).days)
        random_date = start_date + timedelta(days=random_days)

        sessions.append({
            "session_id": session_id,
            "url_path": url_path,
            "chrono": list(chrono),
            "url_node": url_node,
            "device": "desktop" if randint(0, 1) == 0 else "mobile",
            "date": random_date
        })
    
    return pd.DataFrame(sessions)