import os
import re
from bs4 import BeautifulSoup
import requests
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import pandas as pd
import tabula
import re
import os 
from sklearn.ensemble import RandomForestRegressor

zones = {'IJU':[6.6712,3.3515], 'OKE':[6.6263,3.2835], 'AYANGBURIN':[6.619, 3.502],
         'OREGUN':[6.616146, 3.362653], 'EGBEDA':[6.5916, 3.2911], 'FAGBA':[6.657, 3.324],
         'PTC':[6.5920, 3.3435], 'AIT':[6.668, 3.263], 'ABULE': [6.590, 3.285], 'IGANDO':[6.546, 3.248],
         'ADIYAN':[6.69083,3.33972], 'AMUWO':[6.463,3.298], 'LASUNWON':[6.649, 3.514],
         'IJAIYE':[6.629, 3.338], 'IKOTUN':[6.554,3.269], 'OWUTU': [6.645, 3.479], 'IFAKO':[6.638, 3.326],
         'OBA':[6.611, 3.340], 'ODOGUNYAN':[6.673, 3.516], 'OKEIRA':[6.643, 3.347], 'OLAMBE':[6.694, 3.341],
         'AKUTE':[6.682, 3.356],'KETU':[6.598, 3.390],'MENDE':[6.572, 3.372], 'OJODU':[6.634, 3.356],
         'IGBOBI':[6.528, 3.374], 'ILUPEJU':[6.554, 3.356], 'MAGODO':[6.618, 3.400], 'DOPEMU':[6.613, 3.314],
         'OSHODI':[6.554, 3.337], 'ISOLO':[6.538, 3.323], 'ANIFOWOSHE':[6.598, 3.337], 'BARIGA':[6.539, 3.386],
         'OLATEJU':[6.535, 3.359], 'IDIMU':[6.584, 3.246], 'AGO':[6.508, 3.309], 'AYOBO':[6.606, 3.244],
         'OWORO':[6.544, 3.399], 'IKOSI':[6.600, 3.383], 'AJAO':[6.546, 3.326], 'IPAJA':[6.613, 3.266],
         'IJEDE':[6.571, 3.597], 'IGBOGBO':[6.585, 3.525], 'OKOTA':[6.500, 3.310],'GOWON-ESTATE':[6.604, 3.285],
         'OLOWORA':[6.635, 3.384], 'OGBA':[6.625, 3.344], 'IJEGUN': [6.517, 3.262], 'OGUDU':[6.574, 3.394],
         'ABORU':[6.636, 3.283], 'ORILE-AGEGE':[6.629, 3.311], 'EPE':[6.621778, 3.480326]
        }

def scrape_ikedc_outage_data(url='https://www.ikejaelectric.com/cnn/'):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    fault_entries = soup.find_all('div', class_='post')

    post_categories = []
    post_dates = []
    undertakings = []
    faults = []
    areas_affected_s = []

    for entry in fault_entries:
        post_category = entry.find('a', class_='post-category').text.strip()
        post_date = entry.find('span', class_='post-date').text.strip()

        h3_tag = entry.find('h3', class_='post-title')
        h3_text = h3_tag.get_text(separator='\n', strip=True)

        undertaking_match = re.search(r'UNDERTAKING:(.*?)FAULT:', h3_text, re.DOTALL)
        fault_match = re.search(r'FAULT:(.*?)AREAS AFFECTED:', h3_text, re.DOTALL)
        areas_affected_match = re.search(r'AREAS AFFECTED:(.*?)$', h3_text, re.DOTALL)

        undertaking = undertaking_match.group(1).strip() if undertaking_match else "Not found"
        fault = fault_match.group(1).strip() if fault_match else "Not found"
        areas_affected = areas_affected_match.group(1).strip() if areas_affected_match else "Not found"

        post_categories.append(post_category)
        post_dates.append(post_date)
        undertakings.append(undertaking)
        faults.append(fault)
        areas_affected_s.append(areas_affected)

    data = {
        'Branch': post_categories,
        'Date': post_dates,
        'Undertaking': undertakings,
        'Fault': faults,
        'Areas': areas_affected_s
    }

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    for index, row in df.iterrows():
        # Access row data
        undertaking = row['Undertaking']
        y = zones[undertaking][0]
        #y = location['lat']
        #x = location['lon']
        x = zones[undertaking][1]
        df.at[index, 'y'] = y
        df.at[index, 'x'] = x

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))

    return gdf


def downloadReport(start_date,end_date, timestamp):
    #Specify start and end date
    #start_date = '2020-05-01'
    #end_date = '2020-05-24'
    
    # Convert strings to datetime objects
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # Add one day
    start_date_dt += timedelta(days=1)
    end_date_dt += timedelta(days=1)

    # Convert back to strings
    start_date = start_date_dt.strftime('%Y-%m-%d')
    end_date = end_date_dt.strftime('%Y-%m-%d')

    date_to_scrape = pd.date_range(start=start_date,end=end_date).to_list()
    total_dates = len(date_to_scrape)
    i = 0
    while i < total_dates:
        date = str(date_to_scrape[i])[0:10]
        d = datetime.strptime(date, '%Y-%m-%d').strftime('%d%m%Y')
        #get url
        #https://www.nsong.org/DownloadHandler.ashx?FileName=op02122023b.pdf
        url = f'https://nsong.org/DownloadHandler.ashx?FileName=op{d}b.pdf'
        r = requests.get(url, allow_redirects=True)
        #download the content
        file_path = f'neso/DailyOperationalRpt{d}_{timestamp}.pdf'
        open(file_path, 'wb').write(r.content)
        
        file_size_bytes = os.path.getsize(file_path)
        
        # Check if there's missing file (it's less than 20kb). then use a different way to download it
        if file_size_bytes < 100:
            #https://nsong.org/DownloadHandler.ashx?FileName=OP09042024.pdf
            url = f'https://nsong.org/DownloadHandler.ashx?FileName=OP{d}.pdf'
            r = requests.get(url, allow_redirects=True)
            #download the content
            open(file_path, 'wb').write(r.content)

        i +=1

def extract_neso_table(start_date, end_date, timestamp):
    Allfiles = os.listdir("neso")

    Pdffiles = []
    for file in Allfiles:
        if file.endswith(f"{timestamp}.pdf"):
            if os.path.getsize("neso/"+file) > 102 :
                Pdffiles.append(file)
            else:
                os.remove("neso/" + file)

    i =0
    NewDF = pd.DataFrame ( columns = ['Date','Peak Generation', 'Daily Energy Generation',
                                    'Lowest Energy Generation', 'Daily Energy Sent', '6:00 Generation',
                                    'Highest System Frequency', 'Lowest System Frequency',
                                    'Highest Voltage Recorded', 'Lowest Voltage Recorded'])

    #Read all pdfs

    for file in Pdffiles:    
        df = tabula.read_pdf("neso/"+file, pages ='all')
        os.remove("neso/" + file)
        DF = pd.DataFrame (df[0])      
        DF2 = pd.DataFrame.to_numpy (DF)
        #print(DF)
        
        #tabula.convert_into("DailyOperationalRpt02-02-20.pdf", "DailyOperationalRpt02-02-20.csv", output_format="csv")
        
        PeakGeneration = float(DF2[1,1].split('M',1)[0].replace(',',''))
        LowestEnergyGeneration = float(DF2[2,1].split('M',1)[0].replace(',',''))
        DailyEnergyGeneration = float(DF2[3,1].split('M',1)[0].replace(',',''))
        try:
            DailyEnergySentOut = float(DF2[4,1].split('M',1)[0].replace(',',''))
        except:
            DailyEnergySentOut = 0
            #continue
            
        sixhGenerated = float(DF2[5,1].split('M',1)[0].replace(',',''))
        HighestSystemFrequency = float(DF2[6,1].split('H',1)[0].replace(',',''))
        LowestSystemFrequency = float(DF2[7,1].split('H',1)[0].replace(',',''))
        HighestVoltRecorded = float(re.search(r'(\d+(\.\d+)?)\s*[Kk]V', DF2[8,1]).group(1))
        LowestVoltRecorded  = float(re.search(r'(\d+(\.\d+)?)\s*[Kk]V', DF2[9,1]).group(1))
        Date = DF2[1,0].split()[-1]
        
        NewDF.loc[i] = [Date, PeakGeneration, DailyEnergyGeneration, LowestEnergyGeneration, DailyEnergySentOut,
                    sixhGenerated, HighestSystemFrequency, LowestSystemFrequency, HighestVoltRecorded,
                    LowestVoltRecorded]
        i=i+1

    NewDF['Date'] = pd.to_datetime(NewDF['Date'], format='%d/%m/%Y')

    date_range_reference = pd.date_range(start=start_date, end=end_date)
    # Find the dates that are missing from your DataFrame
    missing_dates = date_range_reference[~date_range_reference.isin(NewDF['Date'])]

    if len(missing_dates) > 0:
        base_data = pd.read_csv("backend/data/NESO Base Data.csv", index_col=0)
        X = pd.to_numeric(pd.to_datetime(base_data['Date'])).values.reshape(-1, 1)
        target_variables = ['Peak Generation', 'Daily Energy Generation', 'Lowest Energy Generation', 
                            'Daily Energy Sent', '6:00 Generation', 'Highest System Frequency', 
                            'Lowest System Frequency', 'Highest Voltage Recorded', 'Lowest Voltage Recorded']
        predicted_data = {'Date': missing_dates}

        for target_var in target_variables:
            y = base_data[target_var].values

            model = RandomForestRegressor(max_depth=50, max_features='sqrt', min_samples_split=2,
                            n_estimators=300, random_state=42)
            model.fit(X, y)
            
            missing_dates_numeric = pd.to_numeric(missing_dates).values.reshape(-1, 1)
            predicted_values = model.predict(missing_dates_numeric)

            # Store predicted values in the dictionary
            predicted_data[target_var] = predicted_values
        predicted_df = pd.DataFrame(predicted_data)
        final_df = pd.concat([NewDF, predicted_df], ignore_index=True)
        final_df = final_df.sort_values(by='Date').reset_index(drop=True)
        NewDF = final_df
    return NewDF
        