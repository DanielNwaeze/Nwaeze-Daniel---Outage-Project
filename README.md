# Repository for final year project on IKEDC Outages Prediction and GIS Visualisation (Rough Doc)

Student's Name: Nwaeze Daniel Chinedu

Matric Number: -

Department: Electrical/Electronics Engineering

Embarked in the year 2023 in partial fulfillment during my final year undergraduate program in the University of Lagos.

**NOTE:** Read this file with a Markdown Editor, but if you understand regardless, that's okay.

## GENERAL GUIDE

Use Anaconda to install a conda environment (check docs). Use this environment to pip install necessary packages (check the requirements.txt file). The requirements.txt file was used for the website code itself so you might need to install additional packages for the Jupyter notebooks. Just run the Jupyter notebook codes, where you see a dependency/package not found error, pip install the package.



## JUPYTER NOTEBOOKS

You need to install Jupyter Notebook to run this. Use Anaconda to install a conda environment (check docs). Use this environment to pip install necessary packages (check the requirements.txt file). The requirements.txt file was used for the website code itself so you might need to install additional packages. Just run the Jupyter notebook codes, where you see a dependency/package not found error, pip install the package.

The following Jupyter notebooks were used in order:
Please duplicate before making your changes as cells in the notebook aren't orderly arranged.

1. **Extracting Load Demands.ipynb**

This extracts the NESO systems operation data. Pay attention to how it is being used.
It contains Nigeria's systems operation data for that day. Note that if you want the data for instance for 25th December 2024, you specify the query as 26th December, 2024 as the systems operation for a day is updated early next day.

2. **Combine Data.ipynb**

This combines the NESO, weather and Ikeja Electric (it will be your own distribution data in your case) together. The output is a combined_data.csv which is used for machine learning modelling.

3. **Data Exploration.ipynb** to make initial analysis from the data.

4. **Model Development.ipynb** for developing the outage prediction model

5. **Model - Fault Classification.ipynb** for developing the outage/fault classification model.

**Geospatial Visualisation.ipynb** notebook was for testing the visualisation aspect to write into the website code. you can see where i'm playing around with heatmaps, etc. check folium docs for how to use the library. i also used the longitude and latitude of the locations to do this.

You need weather data. I used [visual crossing weather data](https://www.visualcrossing.com/weather/weather-data-services).
Just enter the location name eor its longitude and latitude, specify the time range you want and generate the weather details for the location and timeframe. Generate an API key too, because the website will use it for making requests.



## PROJECT DOCS AND PDF

1. **Nwaeze Daniel - Project Proposal.pdf** was the project proposal I made to my supervisor before the project was accepted.

2. **Nwaeze Daniel - Literature Review Research Papers doc** shows a summary with relevant headings of the 53 literature research papers I reviewed as requested by my project supervisor (Dr. Peter Oluseyi). These formed the foundation for the actual literature review. this was after the project proposal was accepted.

3. **Nwaeze Daniel - Project Report**  is the project report itself.

index.html here wasn't used, but it was the initial thought concept :)

Gantt Chart image shows the project timeline and expectations.

Other documents there are helpful docs e.g for writing a project report.



## WEB APPLICATION

The folder **web_app** is used for anything related the website.

It uses **/backend/data** for the data related operations.
scraping.py scraps data from Ikeja electric disco for all outages and gets the NESO systems operation data for the dates queried in the website. it's extracted from the combine_data jupyter notebook and extract load demands notebook.

**complete_data.csv** has been described.
I observed that the systems operation website doesn't upload data for some days so i use the **NESO base data.csv** extracted from **Extracting Load Demands.ipynb** to generate a model and predict for those days where necessary.

**Requirements.txt** is the full list of requirements to run the webapp. Note that the notebooks might need some extra requirements as it does some additional things.

**models** folder contains the outage prediction, classification and vectoriser (used with the classification model) models.

a **neso** empty folder is created to load the NESO extracted data and delete them after loading (say you have many requests, it will be bad if you store the NESO data for every request). I can't remember if it creates the folder itself when a request comes and puts the data in the folder. Since I added the folder, it means most likely, it wasn't able to put the data without a pre-created folder. You can programmatically just create the folder before the data is added instead.

**main.py** is the backend code itself that does everything backend related for the website. Pay attention to it and study it well.
You'll need to create a ".env" file, remember the dot in dot env. Paste your visual crossing weather API key in the file so the application picks it up and uses it.

**templates** folder contains the front-end code. Really, the "theme" folder within the "templates" folder is what's really the frontend code. 

The frontend was gotten from themefisher and is a template, so not everything there was used.


## RUNNING THE WEBSITE


How to run the app:

1. Activate your conda environment that was installed 
2. Use your terminal and cd to the **web-app** directory i.e cd web-app
3. Then run the fast-api application using: uvicorn main:app 
4. Navigate to whichever. Notice that the endpoints in the fastapi main.py code correspond to the website address endpoints you see in your address bar.
5. That's all!