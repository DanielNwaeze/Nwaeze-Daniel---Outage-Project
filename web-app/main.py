from datetime import datetime, timedelta
import json
import os
import requests
import logging

import geopandas as gpd
import joblib
import pandas as pd
import folium
from folium.plugins import HeatMap
from dotenv import load_dotenv

from fastapi import FastAPI, Form, File, UploadFile, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from backend.data.scraping import downloadReport, extract_neso_table, scrape_ikedc_outage_data

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Templates directory
templates = Jinja2Templates(directory="templates")

weather_api_key = os.getenv("WEATHER_API_KEY")

@app.get("/")
async def read_root(request: Request):
    # Render the index.html template
    return templates.TemplateResponse("theme/index.html", {"request": request})

@app.get("/classify-outage", response_class = HTMLResponse)
async def classify_fault(request: Request, message: str = None, error_message: str = None):
    return templates.TemplateResponse(
        "theme/classify-outage.html",
        {"request": request, "message": message, "error_message": error_message}
    )

@app.post("/classify-outage")
async def upload_and_process_file(request: Request, 
                                  file: UploadFile = File(...) ):
    try:
        df = pd.read_csv(file.file, encoding='utf-8')  # For CSV files
        df.dropna(subset=['Fault'], inplace=True)
        X = df['Fault']
        # Load the TF-IDF vectorizer
        tfidf_vectorizer = joblib.load("models/Fault_Classification_tfidf_vectorizer.pkl")

        # Load the trained model
        model = joblib.load("models/Fault_Classification_Model.pkl")
        X_vectorized = tfidf_vectorizer.transform(X)  

        predictions = model.predict(X_vectorized)
        if 'Location' and 'Date' in df.columns:
            output_df = pd.DataFrame({
                'Row Id': X.index,
                'Date': df.Date,
                'Location': df.Location,
                'Fault': X,
                'Outage Type': predictions
            })            
        elif 'Date' in df.columns:      
            output_df = pd.DataFrame({
                'Row Id': X.index,
                'Date': df.Date,
                'Fault': X,
                'Outage Type': predictions
            })
        else:
            output_df = pd.DataFrame({
                'Row Id': X.index,
                'Fault': X,
                'Outage Type': predictions
            })
        output_df = output_df.sort_values(by='Row Id', ascending = True) # Ensure sorted rows

        predictions_json = output_df.to_dict(orient='records')
        return JSONResponse(predictions_json)
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=422, detail="Error Parsing File. Please Upload a Valid CSV File!") 
    except KeyError:
        raise HTTPException(status_code=422, detail="Ensure Outage Column is in your Data!")
    except pd.errors.ParserError:
        raise HTTPException(status_code=422, detail="Error parsing CSV file. Upload a valid CSV file with valid columns")
    except Exception as e:
        #logging.critical(e, exc_info=True) 
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict-outage", response_class=HTMLResponse)
async def read_predict_fault(request: Request, message: str = None, error_message: str = None):
    return templates.TemplateResponse(
        "theme/predict-outage.html", 
        {"request": request, "message": message, "error_message": error_message}
    )

@app.post("/predict-outage")
async def upload_and_process_file(request: Request, 
                                  file: UploadFile = File(...) ):
    try:
        # Use pandas to read the file contents
        df = pd.read_csv(file.file, encoding='utf-8')  # For CSV files
        
        # Select columns to keep
        features = [
            'Peak Generation', 'Daily Energy Generation', 'Lowest Energy Generation', 'Daily Energy Sent', '6:00 Generation',
            'Highest System Frequency', 'Lowest System Frequency', 'Highest Voltage Recorded', 'Lowest Voltage Recorded', 'temp',
            'humidity', 'precip', 'precipprob', 'precipcover', 'windspeed', 'winddir', 'sealevelpressure', 'visibility', 'solarradiation',
            'solarenergy', 'uvindex', 'feelslike'
        ]

        # Select columns from DataFrame
        X = df[features]

        # Drop rows with missing values
        X.dropna(inplace=True)

        X['Outage'] = df['Fault'].apply(lambda x: 0 if pd.isna(x) or x is None or x == 'None' else 1)

        # Load the trained Outage Prediction Model
        model = joblib.load("models/Outage Prediction Model.joblib")

        # Predict outage status
        predictions = model.predict(X)

        # Convert numeric predictions to corresponding texts
        outage_status = ["No Outage" if pred == 0 else "Outage" for pred in predictions]
        # Predict outage probabilities
        outage_probabilities = model.predict_proba(X)[:, 1]  # Probability of outage occurring

        # Represent predictions in the required format
        output_df = pd.DataFrame({
            'Row Id': X.index,
            'Outage Next Day Status': outage_status,
            'Probability of Outage The Next Day (%)': (outage_probabilities*100).round().astype(int)
        })

        # Map dates to the row IDs in the original DataFrame
        output_df['Date'] = output_df['Row Id'].map(df['Date'])
        output_df.insert(1, 'Date', output_df.pop('Date'))
        
        predictions_json = output_df.to_dict(orient='records')
        return JSONResponse(output_df.to_dict(orient='records'))
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=422, detail="Error Parsing File. Please Upload a Valid CSV File!") 
    except KeyError:
        raise HTTPException(status_code=422, detail="Ensure the necessary fields are in your data!")
    except pd.errors.ParserError:
        raise HTTPException(status_code=422, detail="Error parsing CSV file. Upload a valid CSV file with valid columns")
    except Exception as e:
        #logging.critical(e, exc_info=True) 
        raise HTTPException(status_code=500, detail=str(e))

from typing import List, Optional

from fastapi import Query

@app.get("/get-data")
async def download_report(request: Request, 
                           start_date: str = Query(None), 
                           end_date:   str = Query(None), 
                           location_1: str = Query(None),
                           location_2: str = Query(None),
                           location_3: str = Query(None),
                           location_4: str = Query(None),
                           location_5: str = Query(None)):
    data = []
    message = ""

    locations_dict = {
        "abulegba_weather_data": "6.659095,3.309853",
        "akowonjo_weather_data": "6.612,3.315",
        "ikeja_weather_data": "6.605874,3.349149",
        "ikorodu_weather_data": "6.616865,3.508072",
        "oshodi_weather_data": "6.514193,3.308678",
        "shomolu_weather_data": "6.532954,3.36739"
    }
    try:
        if start_date and end_date:
            start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
            end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

            min_date = datetime(2021, 10, 1)
            max_date = datetime(2025, 12, 2)

            if start_datetime < min_date or end_datetime > max_date:
        
                message = "Dates out of range"
            else:
                date_difference = (end_datetime - start_datetime).days

                if date_difference < 5 and start_date <= end_date:

                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")


                    downloadReport(start_date, end_date, timestamp)
                    neso_data = extract_neso_table(start_date, end_date, timestamp)
                    neso_df = pd.DataFrame.from_dict(neso_data)
                    neso_df['Date'] = pd.to_datetime(neso_df['Date'])

                    locations = [location_1, location_2, location_3, location_4, location_5]
                    for i, location in enumerate(locations, start=1):
                        if location:
                            coordinates = locations_dict.get(location)
                            if coordinates:
                                location_start_date = (start_datetime + timedelta(days=i-1)).strftime('%Y-%m-%d')
                                location_end_date = (start_datetime + timedelta(days=i-1)).strftime('%Y-%m-%d')
                                weather_data = fetch_location_data(location_start_date, location_end_date, coordinates)
                                if weather_data:
                                    weather_data['location'] = location
                                    weather_data['coordinates'] = coordinates
                                    data.append(weather_data)
                                else:
                                    message += f"Failed to fetch weather data for location {location}. "
                            else:
                                message += f"Invalid location: {location}. "
                    weather_df = pd.DataFrame(data)
                    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
                    before_epoch = weather_df.columns[:weather_df.columns.get_loc("datetimeEpoch") + 1]
                    after_epoch = weather_df.columns[weather_df.columns.get_loc("datetimeEpoch") + 1:]

                    # Remove 'location' and 'coordinates' from the list of columns after 'datetimeEpoch'
                    after_epoch_filtered = [col for col in after_epoch if col not in ['location', 'coordinates']]

                    # Rearrange the columns
                    new_order = list(before_epoch) + ['location', 'coordinates'] + list(after_epoch_filtered)
                    weather_df = weather_df[new_order]

                    weather_df.rename(columns={'datetime': 'Date'}, inplace=True)
                    # Merge the dataframes on 'Date'
                    merged_df = pd.merge(neso_df, weather_df, on='Date')
                    merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')

                    merged_json = json.loads(merged_df.to_json(orient="records", date_format="iso"))
                    return templates.TemplateResponse("theme/get-data.html",
                            {"request": request, "data": merged_json, "message": message})
                else:
                    if date_difference >= 5:
                        from fastapi import status
                        return templates.TemplateResponse(
                            "theme/get-data.html", 
                            {"request": request, "message": "Maximum date range allowed is 5 days.", "data": None},
                            status_code=status.HTTP_400_BAD_REQUEST
                        )
                    else:
                        message = "Start date cannot be greater than end date"
                    
        else:
            message = "Please provide start and end dates."
    except ValueError:
        message = "Invalid date(s).\n Note: Date format should be YYYY-MM-DD."

    return templates.TemplateResponse(
        "theme/get-data.html", 
        {"request": request, "message": message, "data": data}
    )



def fetch_location_data(start_date, end_date, coordinates):
    # Make API request to fetch data for the specified location and date range
    #weather_api_key = "YOUR_WEATHER_API_KEY"  # Replace with your actual API key
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{coordinates}/{start_date}/{end_date}?key={weather_api_key}&include=days&unitGroup=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        # Parse the response JSON and extract relevant data
        weather_data = response.json()['days'][0]
        return weather_data
    else:
        return None

from folium.plugins import MarkerCluster, TagFilterButton
from folium.plugins import TimestampedGeoJson
from folium.plugins import MousePosition

@app.get("/outage-map")
async def get_outage_map(request: Request, start_date: str = Query(None), end_date: str = Query(None)):
    gdf = scrape_ikedc_outage_data()
    filtered_gdf = gdf
    if start_date and end_date:
        # Convert start_date and end_date to datetime objects for comparison
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Filter data based on date range
        filtered_gdf = gdf[(gdf['Date'] >= start_date) & (gdf['Date'] <= end_date)]

    if not filtered_gdf.empty:
        # Create a map centered at a location
        m = folium.Map(location=[filtered_gdf['y'].mean(), filtered_gdf['x'].mean()], zoom_start=10)

        # Convert the GeoDataFrame to a list of points
        heat_data = [[point.xy[1][0], point.xy[0][0]] for point in filtered_gdf['geometry']]

        # Add the HeatMap layer to the map
        HeatMap(heat_data).add_to(m)
        MousePosition().add_to(m)

        map_html_path = "templates/heatmap.html"
        m.save(map_html_path)
        # Create a lookup table with the most recent outage for each undertaking
        most_recent_outages_lookup = filtered_gdf.loc[filtered_gdf.groupby('Undertaking')['Date'].idxmax()].set_index('Undertaking')

        # Create a dictionary to count total outages for each undertaking
        undertaking_counts = filtered_gdf['Undertaking'].value_counts().to_dict()

        # Create the Folium map
        m = folium.Map(location=[filtered_gdf['y'].mean(), filtered_gdf['x'].mean()], zoom_start=10)

        # Extract unique categories from 'Branch' column for the filter
        categories = filtered_gdf['Branch'].unique().tolist()

        # Create a dictionary to count total outages for each category
        category_counts = filtered_gdf['Branch'].value_counts().to_dict()

        # Create markers for each category using the filtered_gdf data
        for category in categories:
            # Filter GeoDataFrame based on category
            category_data = filtered_gdf[filtered_gdf['Branch'] == category]
            # Convert category data to list of points
            category_heat_data = [[point.xy[1][0], point.xy[0][0]] for point in category_data['geometry']]
            # Add markers with tags for segmentation and Undertaking value
            for point, undertaking in zip(category_heat_data, category_data['Undertaking']):
                # Get the most recent outage details for this undertaking
                most_recent_outage = most_recent_outages_lookup.loc[undertaking]
                # Get the total outages for this undertaking
                total_outages = undertaking_counts[undertaking]
                # Construct tooltip message
                tooltip_msg = f"Branch: {category}<br>Undertaking: {undertaking}<br>Most Recent Date: {most_recent_outage['Date']}<br>Most Recent Outage: {most_recent_outage['Fault']}<br>Total Outages: {total_outages}"
                folium.Marker(location=point, tags=[category, undertaking], tooltip=tooltip_msg).add_to(m)

        # Add TagFilterButton for filtering markers with total number of outages
        TagFilterButton(categories).add_to(m)
        MousePosition().add_to(m)

        # Save map as HTML
        map_html_path = "templates/mapfilter.html"
        m.save(map_html_path)

        # Locations and their coordinates
        locations = {'ABULE-EGBA': [6.6478943, 3.3046224], 'AKOWONJO': [6.6104787, 3.3103079],
                    'IKORODU': [6.6191233, 3.5041271], 'IKEJA': [6.5960605, 3.340787],
                    'OSHODI': [6.5400100000000005, 3.3124146026491035], 'SHOMOLU': [6.533564500000001, 3.38416340600858]}

        # Create a map centered at a location
        m = folium.Map(location=[filtered_gdf['y'].mean(), filtered_gdf['x'].mean()], zoom_start=10)

        # Create a MarkerCluster to group outage incidents by branch
        marker_cluster = MarkerCluster().add_to(m)

        # Iterate through each row in the DataFrame
        for index, row in filtered_gdf.iterrows():
            # Define the color based on the number of outage incidents (you can modify this logic)
            if row['Fault'] == 'High':
                color = 'red'
            elif row['Fault'] == 'Medium':
                color = 'orange'
            else:
                color = 'green'
            
            # Create a marker for each outage incident with a custom popup width
            popup_text = f"Date: {row['Date']}<br>Fault: {row['Fault']}<br>Branch: {row['Branch']}"
            popup = folium.Popup(popup_text, max_width=300)  # Adjust the max_width as needed
            folium.Marker([row['y'], row['x']], popup=popup, icon=folium.Icon(color=color)).add_to(marker_cluster)

        # Save the map
        MousePosition().add_to(m)
        map_html_path = "templates/marker_cluster.html"
        m.save(map_html_path)


        # Create a map centered at the mean location of 'y' and 'x' in filtered_gdf
        m = folium.Map(location=[filtered_gdf['y'].mean(), filtered_gdf['x'].mean()], zoom_start=10)

        # Initialize a dictionary to keep track of outage counts for each undertaking
        undertaking_counts = {}

        # Convert the DataFrame to GeoJSON with timestamp for time-series animation
        features = []

        # Iterate through filtered_gdf sorted by Date
        for idx, row in filtered_gdf.sort_values(by='Date').iterrows():
            # Update or initialize the count for this undertaking
            if row['Undertaking'] in undertaking_counts:
                undertaking_counts[row['Undertaking']] += 1
            else:
                undertaking_counts[row['Undertaking']] = 1
            
            # Create GeoJSON feature for this outage
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [row['x'], row['y']]
                },
                'properties': {
                    'time': str(row['Date']),
                    'popup': f"Undertaking: {row['Undertaking']}<br>Outage Count: {undertaking_counts[row['Undertaking']]}<br>Date: {row['Date']}<br>Fault: {row['Fault']}"
                }
            }
            features.append(feature)

        # Add the TimestampedGeoJson layer for time-series animation
        TimestampedGeoJson(
            {'type': 'FeatureCollection', 'features': features},
            period='P1D',  # Time period
            add_last_point=True,
            auto_play=True
        ).add_to(m)



        # Save the map
        map_html_path = "templates/timestamped_points.html"
        MousePosition().add_to(m)
        m.save(map_html_path)


        heatmap_html = open("templates/heatmap.html", "r").read()
        cluster_html = open("templates/marker_cluster.html", "r").read()
        timestamped_html = open("templates/timestamped_points.html", "r").read()

        return templates.TemplateResponse("theme/outage-map.html", {"request": request})
    else:
        # If filtered_gdf is empty, return an empty map
        m = folium.Map(location=[gdf['y'].mean(), gdf['x'].mean()], zoom_start=10)
        map_html_path = "templates/map.html"
        m.save(map_html_path)
        map_html = open(map_html_path, "r").read()
        error_message = "No outages present in the specified timeframe."
        return templates.TemplateResponse("theme/outage-map.html", {"request": request, "error_message": error_message})

app.mount("/temp", StaticFiles(directory="templates", html=False), name="templates")
