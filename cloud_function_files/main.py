from bs4 import BeautifulSoup
from google.cloud import storage
import lxml
import requests
import time

from config import *


def get_value_from_type(content, child, attribute_name):
    """Finds the value associated with a child node and an attribute.
    Args:
        content:BeautifulSoup object containing the scraped XML content
        child: name of the child node to extract the value from
        attribute_name: name of the child's attribute of interest
    Returns:
        the value as a string
    """
    return content.find(child, {"type": attribute_name}).value.text


def convert_to_celsius(fahrenheit):
    """Converts Fahrenheit to Celsius.
    Args:
        fahrenheit: temperature in F
    Returns:
        temperature in C
    """
    return (fahrenheit - 32) / 1.8


def get_forecast_data(ts):
    """Gets the min and max temperatures, wind speed and probability of precipitation read from a URL.
    Args:
        ts: timestamp
    Returns:
        string containing forecast information (temperatures in Celsius, wind speed in mph and probability of 
        precipitation in percent) and a timestamp.
    """
    xml_data = requests.get(FORECAST_URL).content
    content = BeautifulSoup(xml_data, "lxml")
    tmax = convert_to_celsius(float(get_value_from_type(content, "temperature", "maximum")))
    tmin = convert_to_celsius(float(get_value_from_type(content, "temperature", "minimum")))
    wind = get_value_from_type(content, "wind-speed", "sustained")
    prob_precip = content.find("probability-of-precipitation").value.text
    return f"(tmin: {tmin}, tmax: {tmax}, wind: {wind}, prob_precip: {prob_precip}, timestamp: {ts})"
    

def save_forecast(event, context):
    """Saves forecast data to 2 GCS buckets: (1) overwrites the latest weather forecast in LATEST_FORECAST_BUCKET 
    and (2) adds a file whose name contains a timestamp to HISTORY_FORECAST_BUCKET. 
    Notes:
    - the Cloud Function is triggered by a message on a Cloud Pub/Sub topic
    - this function needs to accept 2 arguments (event and context) but in our case we don't use them.
    """
    # create timestamp
    ts = time.time()

    # get weather forecast as string
    blob_text = get_forecast_data(ts)

    # save to GCS buckets
    storage_client = storage.Client()

    latest_forecast_bucket = storage_client.get_bucket(LATEST_FORECAST_BUCKET)
    latest_forecast_blob = latest_forecast_bucket.blob(DESTINATION_BLOB_NAME)
    latest_forecast_blob.upload_from_string(blob_text)

    history_forecast_bucket = storage_client.get_bucket(HISTORY_FORECAST_BUCKET)
    history_forecast_blob = history_forecast_bucket.blob(f"{DESTINATION_BLOB_NAME}_timestamp_{ts}")
    history_forecast_blob.upload_from_string(blob_text)
    