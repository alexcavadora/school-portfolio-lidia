import pandas as pd
import folium
import os
from google.cloud import bigquery
import json

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "studious-stack-455023-d0-5a97eb750fa4.json"

with open('studious-stack-455023-d0-5a97eb750fa4.json') as json_file:
    project = json.load(json_file)
project_id = project["project_id"]
client = bigquery.Client(project=project_id)
print (f"Project ID = {client.project}")

query = r"SELECT * FROM `studious-stack-455023-d0.mexico_customs_economics.mexico_customs_data` LIMIT 1000"
#print(query)
query_job = client.query(query)
df = query_job.to_dataframe()
print(df)
#print(query_job)