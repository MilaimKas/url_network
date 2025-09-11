from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from app_helpers import network_class, network_functions
from flask_caching import Cache
from datetime import datetime

app = Flask(__name__)

cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

# caching logic

def build_cache_key(start_date, end_date, device):
    return f"network_data_{start_date}_{end_date}_{device}" 

@app.route('/available-dates')
def available_dates():

    device = request.args.get('device', 'desktop')  # default to 'desktop'

    # Build WebNetwork object
    W = get_network(device) # dummy date for data fetching

    if 'date' in W.df.columns:
        return jsonify(sorted(W.graphs_by_date.keys()))
    else:
        return jsonify({"Message": "no date column found in data frame"})

@cache.memoize(timeout=None)
def get_network(device, start_date=None, end_date=None, time_granularity='Week'):
    #df = fetch_data(start_date, end_date, device) # placeholder for actual data fetching logic

    # generate random data
    df = network_functions.generate_dummy_df(n_sessions=1000, max_pages_per_session=10)

    # filter
    df_filtered = df[df['device'] == device]

    # Build WebNetwork object
    W = network_class.WebNetwork(df_filtered)

    if 'date' in df.columns:
        W.build_graphs_by_date(time_granularity=time_granularity)
        print("Networks per date period built and cached.")
    
    print("Global Network built and cached")

    return W

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/graph')
def graph():
    
    device = request.args.get('device', 'desktop')  # default to 'desktop'
    date_str = request.args.get('date', None)
    #start_date = request.args.get('start_date', '2023-01-01')
    #end_date = request.args.get('end_date', '2023-12-31')

    # get  network object from cache or build it
    W = get_network(device)

    print(date_str)
    
    if date_str == "All time" or not date_str:
        return jsonify(W.to_cytoscape_json(node_size_range=(5, 20), edge_width_range=(0.1, 5)))
    else:
        try:
            # Convert from ISO format or HTTP date string
            try:
                date_key = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                # Try to parse "Mon, 02 Oct 2023 00:00:00 GMT"
                date_key = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S GMT").date()

        except (ValueError, KeyError):
            return jsonify({"elements": {"nodes": [], "edges": []}, "message": "Invalid or unknown date."})

        # fetch network according to date key
        try:
            G = W.graphs_by_date[date_key]
        except KeyError:
            return jsonify({"elements": {"nodes": [], "edges": []}, "message": f"Key date {date_key} not found"})

        return jsonify(W.to_cytoscape_json(navigation_graph=G, node_size_range=(5, 20), edge_width_range=(0.1, 5)))
    
@app.route("/shortest-path")
def shortest_path():

    #start = request.args.get("start", None)
    #end = request.args.get("end", None)
    device = request.args.get("device")
    source = request.args.get("src")
    target = request.args.get("dst")
    intermediates = request.args.getlist("intermediates[]", None)
    max_paths = int(request.args.get("max_paths", 5))   

    print(f"Shortest path request: src={source}, dst={target}, intermediates={intermediates}")

    W = get_network(device)

    # get max_paths shortest paths as dataframe with sum of weights along the path
    try:
        path_df = W.get_shortest_k_path(source, target, C=intermediates, shift=None, k=max_paths)
    except Exception as e:
        print(f"Error finding path: {e}")
        return jsonify({"paths": [], "message": str(e)})
    
    if path_df.empty:
        return jsonify({"paths": [], "message": "No valid paths found."})

        
    try:
        print("Successfully found paths, formatting response.")
        # Convert DataFrame rows to list of dicts
        paths = path_df.to_dict(orient='records')
        return jsonify({
            "paths": paths,
            "message": "success"
        })

    except Exception as e:
        print(f"Getting error wile formating into json: {e}")
        return jsonify({"paths": [], "message": str(e)})
    

if __name__ == '__main__':
    app.run(debug=True)
