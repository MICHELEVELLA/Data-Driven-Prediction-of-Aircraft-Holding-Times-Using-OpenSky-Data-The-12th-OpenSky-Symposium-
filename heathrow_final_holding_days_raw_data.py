#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz

def cal_holding_right(df):

    # df is saved as cache and read as a DataFrame
    df.to_csv("cachert")
    df=pd.read_csv("cachert")

    # Convert 'timestamp' column to datetime type
    df['timestamp'] = pd.to_datetime(df['timestamp'])


    # Create an empty DataFrame to store the holding times
    df_holding_times = pd.DataFrame(columns=['Flight Number', 'Holding Time (minutes)', 'Type Code'])

    # Get the unique flight numbers (callsigns)
    callsigns = df['flight_id'].unique()

    print (callsigns)

    # Loop through each flight
    for callsign in callsigns:
        # Filter the flight data for the current flight and sort by timestamp
        df_flight = df[df['flight_id'] == callsign].sort_values(by='timestamp')

        in_holding = False
        start_time = None
        end_time = None
        neverhold= True
        holding_time=0
        right= False
        left= False

        #print(callsign)

        firstrowtime = df_flight['timestamp'].iloc[0]
        firstrowtime=pd.to_datetime(firstrowtime)
        firstrowtime = firstrowtime.tz_localize(None)

        lastrowtime=df_flight['timestamp'].iloc[-1]

        #print(df_flight)

        # Loop through the flight data starting from the second row
        for index, row in df_flight.iloc[1:].iterrows():


            #Calculate right hand turn and left hand turn degrees
            right_turn=((row['track'] - df_flight.loc[index-1, 'track'])+360)%360
            left_turn=((df_flight.loc[index-1, 'track']-row['track'])+360)%360


            if (right_turn<left_turn):
                right= True
                left= False
                turn= right_turn
            else:
                right= False
                left= True
                turn= left_turn

            # Check for a 20 degree deflection to the right
            if (right== True and turn>=20) :
                # Start the timer if not already in a holding pattern
                if not in_holding:
                    in_holding = True
                    neverhold= False
                    start_time = row['timestamp']

            # Check for a deflection to the left
            elif (left==True and turn>=5):
                # Stop the timer if in a holding pattern
                if in_holding:
                    in_holding = False
                    end_time = row['timestamp']
                    holding_time = end_time - start_time

                    # Convert holding_time to minutes
                    holding_time_minutes = holding_time.total_seconds() / 60

                    # Get the type code for the flight
                    type_code = df_flight['typecode'].iloc[0]

                    if holding_time_minutes>=2:
                        # Append flight details to the DataFrame
                        df_holding_times = pd.concat([df_holding_times, pd.DataFrame({'Flight Number': [callsign], 'Holding Time (minutes)': [holding_time_minutes], 'Type Code': [type_code],'Time over fix':[firstrowtime]})], ignore_index=True)
                    else:
                        # Append flight details to the DataFrame as Holding Time is 0
                        df_holding_times = pd.concat([df_holding_times, pd.DataFrame({'Flight Number': [callsign], 'Holding Time (minutes)': 0.0, 'Type Code': [type_code],'Time over fix':[firstrowtime],'Time over fix':[firstrowtime]})], ignore_index=True)
            if row['timestamp']==lastrowtime:
                if in_holding:
                    #Calculate holding time
                    holding_time = lastrowtime - start_time

                    # Convert holding_time to minutes
                    holding_time_minutes = holding_time.total_seconds() / 60

                    # Get the type code for the flight
                    type_code = df_flight['typecode'].iloc[0]

                    if holding_time_minutes>=2:
                        # Append flight details to the DataFrame
                        df_holding_times = pd.concat([df_holding_times, pd.DataFrame({'Flight Number': [callsign], 'Holding Time (minutes)': [holding_time_minutes], 'Type Code': [type_code],'Time over fix':[firstrowtime]})], ignore_index=True)
                    else:
                        # Append flight details to the DataFrame as Holding Time is 0
                        df_holding_times = pd.concat([df_holding_times, pd.DataFrame({'Flight Number': [callsign], 'Holding Time (minutes)': 0.0, 'Type Code': [type_code],'Time over fix':[firstrowtime]})], ignore_index=True)

                # If the flight never been in a holding pattern, add it with a holding time of 0
                elif neverhold:
                    type_code = df_flight['typecode'].iloc[0]
                    # Append flight details to the DataFrame as Holding Time is 0
                    df_holding_times = pd.concat([df_holding_times, pd.DataFrame({'Flight Number': [callsign], 'Holding Time (minutes)': 0.0, 'Type Code': [type_code],'Time over fix':[firstrowtime]})], ignore_index=True)
    # Display the DataFrame
    #print(df_holding_times)
    return df_holding_times


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def cal_holding_left(df):

    # df is saved as cache and read as a DataFrame
    df.to_csv("cachelt")
    df=pd.read_csv("cachelt")

    # Convert 'timestamp' column to datetime type
    df['timestamp'] = pd.to_datetime(df['timestamp'])


    # Create an empty DataFrame to store the holding times
    df_holding_times = pd.DataFrame(columns=['Flight Number', 'Holding Time (minutes)', 'Type Code'])

    # Get the unique flight numbers (callsigns)
    callsigns = df['flight_id'].unique()

    #print (callsigns)

    # Loop through each flight
    for callsign in callsigns:
        # Filter the flight data for the current flight and sort by timestamp
        df_flight = df[df['flight_id'] == callsign].sort_values(by='timestamp')

        in_holding = False
        start_time = None
        end_time = None
        neverhold= True
        holding_time=0
        right= False
        left= False

        firstrowtime = df_flight['timestamp'].iloc[0]
        firstrowtime=pd.to_datetime(firstrowtime)
        firstrowtime = firstrowtime.tz_localize(None)

        lastrowtime=df_flight['timestamp'].iloc[-1]

        # Loop through the flight data starting from the second row
        for index, row in df_flight.iloc[1:].iterrows():


             #Calculate right hand turn and left hand turn degrees
            right_turn=((row['track'] - df_flight.loc[index-1, 'track'])+360)%360
            left_turn=((df_flight.loc[index-1, 'track']-row['track'])+360)%360


            if (right_turn<left_turn):
                right= True
                left= False
                turn= right_turn
            else:
                right= False
                left= True
                turn= left_turn

            # Check for a 20 degree deflection to the left
            if (left== True and turn>=20) :
                # Start the timer if not already in a holding pattern
                if not in_holding:
                    in_holding = True
                    neverhold= False
                    start_time = row['timestamp']

            # Check for a deflection to the right
            elif (right==True and turn>=5):
                # Stop the timer if in a holding pattern
                if in_holding:
                    in_holding = False
                    end_time = row['timestamp']
                    holding_time = end_time - start_time

                    # Convert holding_time to minutes
                    holding_time_minutes = holding_time.total_seconds() / 60

                    # Get the type code for the flight
                    type_code = df_flight['typecode'].iloc[0]

                    if holding_time_minutes>=2:
                        # Append flight details to the DataFrame
                        df_holding_times = pd.concat([df_holding_times, pd.DataFrame({'Flight Number': [callsign], 'Holding Time (minutes)': [holding_time_minutes], 'Type Code': [type_code],'Time over fix':[firstrowtime]})], ignore_index=True)
                    else:
                        df_holding_times = pd.concat([df_holding_times, pd.DataFrame({'Flight Number': [callsign], 'Holding Time (minutes)': 0.0, 'Type Code': [type_code],'Time over fix':[firstrowtime]})], ignore_index=True)
            if row['timestamp']==lastrowtime:
                if in_holding:
                    #Calculate holding time
                    holding_time = lastrowtime - start_time

                    # Convert holding_time to minutes
                    holding_time_minutes = holding_time.total_seconds() / 60

                    # Get the type code for the flight
                    type_code = df_flight['typecode'].iloc[0]

                    if holding_time_minutes>=2:
                        # Append flight details to the DataFrame
                        df_holding_times = pd.concat([df_holding_times, pd.DataFrame({'Flight Number': [callsign], 'Holding Time (minutes)': [holding_time_minutes], 'Type Code': [type_code],'Time over fix':[firstrowtime]})], ignore_index=True)
                    else:
                        df_holding_times = pd.concat([df_holding_times, pd.DataFrame({'Flight Number': [callsign], 'Holding Time (minutes)': 0.0, 'Type Code': [type_code],'Time over fix':[firstrowtime]})], ignore_index=True)

                # If the flight never been in a holding pattern, add it with a holding time of 0
                elif neverhold:
                    type_code = df_flight['typecode'].iloc[0]
                    df_holding_times = pd.concat([df_holding_times, pd.DataFrame({'Flight Number': [callsign], 'Holding Time (minutes)': 0.0, 'Type Code': [type_code],'Time over fix':[firstrowtime]})], ignore_index=True)
    # Display the DataFrame
    #print(df_holding_times)
    return df_holding_times


# In[ ]:


def filter_2rows(df):

    df.to_csv("cache1")
    df=pd.read_csv("cache1")

    # Count the occurrences of the callsign column
    callsign_counts = df['flight_id'].value_counts()

    # Get the callsigns that have exactly 2 occurrences
    callsigns_with_two_rows = callsign_counts[callsign_counts == 2].index

    # Filter the DataFrame to exclude the rows with the selected callsigns
    filtered_df = df[~df['flight_id'].isin(callsigns_with_two_rows)]

    # Return the filtered DataFrame
    return(filtered_df)


# In[ ]:


import requests
import json

doc8643_url = r = "https://www4.icao.int/doc8643/External/AircraftTypes"
r = requests.post(doc8643_url)
doc8643at = pd.json_normalize(r.json())
doc8643at.columns = doc8643at.columns.str.lower()
doc8643at = doc8643at.rename(columns={
    "modelfullname": "model",
    "manufacturercode": "manufacturer_code",
    "aircraftdescription": "aircraft_description",
    "enginecount": "engine_count",
    "enginetype": "engine_type"
    }).drop(columns=['wtg'])

def Icao_doc8643(df):
    for index,row in df.iterrows():
        type_code=None
        wtc=None
        engine_type =None
        type_code=row['Type Code']
        #print(type_code)
        #type(type_code)

        if pd.notna(row['Type Code']):
            icao_row = doc8643at[doc8643at['designator'] == type_code].iloc[0]
            wtc=icao_row["wtc"]
            engine_type=icao_row["engine_type"]

        if wtc=='L':
            df.at[index,'WTC L']=1
            df.at[index,'WTC M']=0
            df.at[index,'WTC H']=0
            df.at[index,'WTC J']=0
        elif wtc=='M':
            df.at[index,'WTC L']=0
            df.at[index,'WTC M']=1
            df.at[index,'WTC H']=0
            df.at[index,'WTC J']=0
        elif wtc=='H':
            df.at[index,'WTC L']=0
            df.at[index,'WTC M']=0
            df.at[index,'WTC H']=1
            df.at[index,'WTC J']=0
        elif wtc=='J':
            df.at[index,'WTC L']=0
            df.at[index,'WTC M']=0
            df.at[index,'WTC H']=0
            df.at[index,'WTC J']=1
        else:
            df.at[index,'WTC L']=0
            df.at[index,'WTC M']=1
            df.at[index,'WTC H']=0
            df.at[index,'WTC J']=0
        

        if engine_type== 'Jet':
            df.at[index,'Engine Jet']= 1
            df.at[index,'Engine Turboprop/shaft']= 0
        elif engine_type== 'Turboprop/Turboshaft':
            df.at[index,'Engine Jet']= 0
            df.at[index,'Engine Turboprop/shaft']= 1
        else:
            df.at[index,'Engine Jet']= 1
            df.at[index,'Engine Turboprop/shaft']= 0
    return df


# In[ ]:


from metar import Metar


def bad_weather_classes(obs: str):
    metar = Metar.Metar(obs, strict=False)
    ceiling = _get_visibility_ceiling_coef(metar)
    wind = _get_wind_coef(metar)
    precip = _get_precipitation_coef(metar)
    freezing = _get_freezing_coef(metar)
    phenomena = _get_dangerous_phenomena_coef(metar)
    wspeed,wdir= _get_wind_speed_dir(metar)

    return (ceiling, wind, precip, freezing, phenomena,wdir,wspeed)


def _get_dangerous_phenomena_coef(metar: Metar.Metar):
    phenomena, showers = __dangerous_weather(metar.weather)
    cb, tcu, ts = __dangerous_clouds(metar.sky)
    if showers is not None and showers > 0:
        if cb == 12:
            ts = 18 if showers == 1 else 24

        if cb == 10 or tcu == 10:
            ts = 12 if showers == 1 else 20

        if cb == 6 or tcu == 8:
            ts = 10 if showers == 1 else 15

        if cb == 4 or tcu == 5:
            ts = 8 if showers == 1 else 12

        if tcu == 3:
            ts = 4 if showers == 1 else 6

    return max(i for i in [phenomena, cb, tcu, ts] if i is not None)


def __dangerous_weather(weather: []):
    phenomena = None
    showers = None
    for intensity, desc, precip, obs, other in weather:
        __phenomena = 0
        __showers = 0
        if other in ["FC", "DS", "SS"] or obs in ["VA", "SA"] or precip in ["GR", "PL"]:
            __phenomena = 24

        if desc == "TS":
            __phenomena = 30 if intensity == "+" else 24

        if precip == "GS":
            __phenomena = 18

        if phenomena is None or __phenomena > phenomena:
            phenomena = __phenomena

        if desc == "SH":
            __showers = 1 if intensity == "-" else 2

        if showers is None or __showers > showers:
            showers = __showers

    return (phenomena, showers)


def __dangerous_clouds(sky: []):
    cb = 0
    tcu = 0
    for cover, height, cloud in sky:
        __cb = 0
        __tcu = 0
        if cover == "OVC":
            if cloud == "TCU":
                __tcu = 10

            if cloud == "CB":
                __cb = 12

        if cover == "BKN":
            if cloud == "TCU":
                __tcu = 8

            if cloud == "CB":
                __cb = 10

        if cover == "SCT":
            if cloud == "TCU":
                __tcu = 5

            if cloud == "CB":
                __cb = 6

        if cover == "FEW":
            if cloud == "TCU":
                __tcu = 3

            if cloud == "CB":
                __cb = 4

        if __cb > cb:
            cb = __cb

        if __tcu > tcu:
            tcu = __tcu

    return (cb, tcu, None)


def _get_wind_coef(metar: Metar.Metar):
    spd = metar.wind_speed.value() if metar.wind_speed is not None else None
    gusts = metar.wind_gust.value() if metar.wind_gust is not None else None
    coef = 0

    if spd is not None:
        if 16 <= spd <= 20:
            coef = 1

        if 21 <= spd <= 30:
            coef = 2

        if spd > 30:
            coef = 4

        if gusts is not None:
            coef += 1

    return coef


def _get_wind_speed_dir(metar: Metar.Metar):
    if metar.wind_speed is  not None:
        wspd = metar.wind_speed.value()
    else:
        wspd=0
    wdir = metar.wind_dir

    if not (wdir==None):
        wdir=wdir.value()
    else:
        wdir=0
    return wspd,wdir


def _get_precipitation_coef(metar: Metar.Metar):
    coef = 0
    for intensity, desc, precip, obs, other in metar.weather:
        __coef = 0
        if desc == "FZ":
            __coef = 3

        if precip == "SN":
            __coef = 2 if intensity == "-" else 3

        if precip == "SG" or (precip == "RA" and intensity == "+"):
            __coef = 2

        if precip in ["RA", "UP", "IC", "DZ"]:
            __coef = 1

        if __coef > coef:
            coef = __coef

    return coef


def _get_freezing_coef(metar: Metar.Metar):
    tt = metar.temp.value() if metar.temp is not None else None
    dp = metar.dewpt.value() if metar.dewpt is not None else None
    moisture = None
    for intensity, desc, precip, obs, other in metar.weather:
        __moisture = 0
        if desc == "FZ":
            __moisture = 5

        if precip == "SN":
            __moisture = 4 if intensity == "-" else 5

        if precip in ["SG", "RASN"] or (precip == "RA" and intensity == "+") or obs == "BR":
            __moisture = 4

        if precip in ["DZ", "IC", "RA", "UP", "GR", "GS", "PL"] or obs == "FG":
            __moisture = 3

        if moisture is None or __moisture > moisture:
            moisture = __moisture

    if tt is not None and tt <= 3 and moisture == 5:
        return 4

    if tt is not None and tt <= -15 and moisture is not None:
        return 4

    if tt is not None and tt <= 3 and moisture == 4:
        return 3

    if tt is not None and tt <= 3 and (moisture == 3 or (tt - dp) < 3):
        return 1

    if tt is not None and tt <= 3 and moisture is None:
        return 0

    if tt is not None and tt > 3 and moisture is not None:
        return 0

    if tt is not None and tt > 3 and (moisture is None or (tt - dp) >= 3):
        return 0

    return 0




def _get_visibility_ceiling_coef(metar: Metar.Metar):
    vis = __get_visibility(metar)
    is_covered, cld_base = __get_ceiling(metar)

    if vis is not None and cld_base is not None:
        if (vis <= 325) or (is_covered and cld_base <= 50):
            return 5

        if (350 <= vis <= 500) or (is_covered and 100 <= cld_base <= 150):
            return 4

        if (550 <= vis <= 750) or (is_covered and 200 <= cld_base <= 250):
            return 2

    return 0



def __get_ceiling(metar: Metar.Metar):
    cld_cover = False
    cld_base = None
    for cover, height, cloud in metar.sky:
        if cover in ["BKN", "OVC"]:
            cld_cover = True

        if height is not None and (cld_base is None or cld_base > height.value()):
            cld_base = height.value()

    return (cld_cover, cld_base)


def __get_visibility(metar: Metar.Metar):
    vis = None
    if metar.vis is not None:
        vis = metar.vis.value()
    rvr = None
    for name, low, high, unit in metar.runway:
        if rvr is None or rvr > low.value():
            rvr = low.value()

    if rvr is not None and rvr < 1500:
        vis = rvr

    return vis


# In[ ]:


"""
Example script that scrapes data from the IEM ASOS download service.
Requires: Python 3
"""
import datetime
import json
import os
import sys
import time
import pandas as pd
from urllib.request import urlopen

# Number of attempts to download data
MAX_ATTEMPTS = 6
# HTTPS here can be problematic for installs that don't have Lets Encrypt CA
SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"

df1= None
startts= None
endts= None
stations=None

def download_data(uri):
    """Fetch the data from the IEM
    The IEM download service has some protections in place to keep the number
    of inbound requests in check.  This function implements an exponential
    backoff to keep individual downloads from erroring.
    Args:
      uri (string): URL to fetch
    Returns:
      string data
    """
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            data = urlopen(uri, timeout=300).read().decode("utf-8")
            if data is not None and not data.startswith("ERROR"):
                return data
        except Exception as exp:
            print(f"download_data({uri}) failed with {exp}")
            time.sleep(5)
        attempt += 1

    print("Exhausted attempts to download, returning empty data")
    return ""


def main_weather(startts,endts,stations):
    """Our main method"""
    #global df1 , endts, startts , stations
    # timestamps in UTC to request data for
    #startts = datetime.datetime(2021, 12, 1)
    #endts = datetime.datetime(2022, 2, 1)

    service = SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=yes&"

    service += startts.strftime("year1=%Y&month1=%m&day1=%d&")
    service += endts.strftime("year2=%Y&month2=%m&day2=%d&")

    # Specify the airport station code
    #stations = ["EGLL"]
    for station in stations:
        uri = f"{service}&station={station}"
        print(f"Downloading: {station}")
        data = download_data(uri)
        outfn = f"{station}_{startts:%Y%m%d%H%M}_{endts:%Y%m%d%H%M}.txt"
        with open(outfn, "w", encoding="ascii") as fh:
            fh.write(data)
        df1 = pd.read_csv(f"C:\\Users\mic__\{station}_{startts:%Y%m%d%H%M}_{endts:%Y%m%d%H%M}.txt", sep = ',', header=5)


    df2=df1[["metar","valid"]]
    df2

    final = pd.DataFrame(columns=['metar', 'valid', 'ceiling', 'wind', 'precip', 'freezing', 'phenomena','wind dir','wind speed'])

    for index, row in df2.iterrows():
        metar = row["metar"]
        valid = row["valid"]
        #print(metar)
        (ceiling, wind, precip, freezing, phenomena,wdir,wspeed) = bad_weather_classes(row["metar"])
        #comment for wind direction and speed

        if ceiling is not None and wind is not None and precip is not None and freezing is not None and phenomena is not None:
            row_df = pd.DataFrame({'metar': metar, 'valid': valid, 'ceiling': ceiling, 'wind': wind, 'precip': precip, 'freezing': freezing, 'phenomena': phenomena,'wind dir':wdir,'wind speed':wspeed}, index=[0])
            final = pd.concat([final, row_df], ignore_index=True)

    final.to_csv(f"{stations}_{startts:%Y%m%d%H%M}_{endts:%Y%m%d%H%M}_final.csv", index=False)

    final['valid'] = pd.to_datetime(final['valid'])

    return final


# In[ ]:


def calulate_heathrow_hold (egll1) :

    biggin=egll1.inside_bbox((-0.03,51.26,1.1,51.39))
    biggin=biggin.aircraft_data()
    biggin=filter_2rows(biggin)
    #print(biggin)
    big_hold=cal_holding_right(biggin)
    big_hold=big_hold.assign(Big=1,Ock=0,Bov=0,Lam=0)


    ock=egll1.inside_bbox((-0.7, 51.0,-0.25, 51.35))
    ock=ock.aircraft_data()
    ock=filter_2rows(ock)
    ock_hold=cal_holding_right(ock)
    ock_hold=ock_hold.assign(Big=0,Ock=1,Bov=0,Lam=0)

    bov=egll1.inside_bbox((-1, 51.4,-0.48, 52.0))
    bov=bov.aircraft_data()
    bov=filter_2rows(bov)
    bov_hold=cal_holding_right(bov)
    bov_hold=bov_hold.assign(Big=0,Ock=0,Bov=1,Lam=0)

    lam=egll1.inside_bbox((0.1, 51.5,0.38, 52))
    lam=lam.aircraft_data()
    lam=filter_2rows(lam)
    lam_hold=cal_holding_left(lam)
    lam_hold=lam_hold.assign(Big=0,Ock=0,Bov=0,Lam=1)

    heathrow_total_hold= pd.concat([big_hold,lam_hold,bov_hold,ock_hold], ignore_index=True)


    egll_final=egll1.aircraft_data()

    egll_final.to_csv("cache_final")
    egll_final=pd.read_csv("cache_final")

    # Loop through the original DataFrame and add missing rows
    for index, row in egll_final.iterrows():
        callsign = row['flight_id']
        if callsign not in heathrow_total_hold['Flight Number'].values:

            filtered_df = egll_final.loc[egll_final['flight_id'] == callsign]
            #filtered_df= filtered_df.sort_values(by='timestamp')
            #first_row= filtered_df.iloc[0]
            #time_over_fix=first_row['timestamp']
            appended_row = {
                'Flight Number': callsign,
                'Holding Time (minutes)':0.0,
                'Type Code': row['typecode'],
                'Big':0,
                'Ock':0,
                'Bov':0,
                'Lam':0,
                'Time over fix':np.nan
            }
            missing_row_df = pd.DataFrame([appended_row])
            heathrow_total_hold = pd.concat([heathrow_total_hold, missing_row_df], ignore_index=True)

    heathrow_total_hold

    # Sort the DataFrame by 'Holding Time (minutes)' column in descending order
    heathrow_total_hold_sorted = heathrow_total_hold.sort_values(by='Holding Time (minutes)', ascending=False)

    # Drop duplicates based on 'Flight Number' and keep the first occurrence (highest 'Holding Time (minutes)' value)
    heathrow_total_hold_no_duplicates= heathrow_total_hold_sorted.drop_duplicates(subset='Flight Number', keep='first')


    #print(heathrow_total_hold_no_duplicates)
    #heathrow_total_hold_no_duplicates.to_csv("without_duplicates")

    return heathrow_total_hold_no_duplicates,egll_final


# In[ ]:


def weather_cal (weather,heathrow_total_hold_no_duplicates,egll_final,arr_times):

    # Get the unique flight numbers (flight_id)
    callsigns = heathrow_total_hold_no_duplicates['Flight Number'].unique()

    # Convert 'valid' column in the weather DataFrame to datetime
    weather['valid'] = pd.to_datetime(weather['valid'])

    #heathrow_total_hold_no_duplicates = heathrow_total_hold_no_duplicates.assign(TimeStamp=None, Ceiling=None, Wind=None,
    #                                                                             Precip=None, Freezing=None, Phenomena=None)

    for time in arr_times:

        # Loop through each flight
        for index,row in heathrow_total_hold_no_duplicates.iterrows():

            callsign=heathrow_total_hold_no_duplicates.at[index, 'Flight Number']
            #Filter the flight data for the current flight and sort by timestamp
            df_flight = egll_final[egll_final['flight_id'] == callsign].sort_values(by='timestamp')

            # Get the first row for the current callsign
            first_row = df_flight.iloc[0] # First row
            first_timestamp=first_row['timestamp']
            first_timestamp=pd.to_datetime(first_timestamp)
            first_timestamp = first_timestamp.tz_localize(None)
            timestamp=first_timestamp-pd.Timedelta(minutes=time)

            heathrow_total_hold_no_duplicates.at[index, 'TimeStamp'] = first_timestamp

            given_timestamp_df=pd.DataFrame({"given_timestamp":[timestamp]})
            
            weather_sorted = weather.sort_values(by='valid')
            #print(given_timestamp_df)
            #print(weather)


            # Merge the METAR timestamps with the given timestamp based on the closest preceding timestamp
            merged_weather = pd.merge_asof(given_timestamp_df, weather_sorted, left_on='given_timestamp', right_on='valid', direction='backward')
            #print(merged_weather)

            ceiling=merged_weather.at[0, 'ceiling']
            heathrow_total_hold_no_duplicates.at[index, f'Ceiling at -{time}'] = ceiling

            wind=merged_weather.at[0, 'wind']
            heathrow_total_hold_no_duplicates.at[index, f'Wind at -{time}'] = wind

            precip=merged_weather.at[0, 'precip']
            heathrow_total_hold_no_duplicates.at[index, f'Precip at -{time}'] = precip

            freezing=merged_weather.at[0, 'freezing']
            heathrow_total_hold_no_duplicates.at[index, f'Freezing at -{time}'] = freezing

            phenomena=merged_weather.at[0, 'phenomena']
            heathrow_total_hold_no_duplicates.at[index, f'Phenomena at -{time}'] = phenomena

            wdir=merged_weather.at[0, 'wind dir']
            heathrow_total_hold_no_duplicates.at[index, f'Wind Direction at -{time}'] = wdir

            wspeed=merged_weather.at[0, 'wind speed']
            heathrow_total_hold_no_duplicates.at[index, f'Wind Speed at -{time}'] = wspeed

    return heathrow_total_hold_no_duplicates


# In[ ]:


import pandas as pd
from datetime import datetime, timedelta

def probabilty_flight_data(df,arrtime):
    # Convert 'Time over fix' to datetime, coerce errors to NaT
    df['Time over fix'] = pd.to_datetime(df['Time over fix'], format='%d/%m/%Y %H:%M', errors='coerce')

    # Initialize the new column with default values
    df['Time leaving fix'] = pd.NaT
    df['Time Range'] = None

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Skip rows where 'Time over fix' is Na
        if pd.isna(row['Time over fix']):
            continue

        # Calculate 'Time leaving fix' by adding 'Holding Time (minutes)' to 'Time over fix'
        time_leaving_fix = row['Time over fix'] + timedelta(minutes=row['Holding Time (minutes)'])

        # Assign the calculated time to the new column
        df.at[index, 'Time leaving fix'] = time_leaving_fix

        # Create a range of times from 'Time over fix' to 'Time leaving fix' with a frequency of 30 seconds
        time_range = pd.date_range(start=row['Time over fix'], end=time_leaving_fix, freq='30s')

       # Assign the time range directly to the 'Time Range' column
        df.at[index, 'Time Range'] = time_range



    for time in arrtime:    
        for index, row in df.iterrows():
                tma_time_stamp=df.at[index, 'TimeStamp']
                flight_time=tma_time_stamp-pd.Timedelta(minutes=time)
                holding_flights = 0
                # Loop through all rows in the DataFrame to find overlaps
                for other_index, other_row in df.iterrows():
                    if other_index == index:
                        continue
                    other_time_range_list = df.at[other_index, 'Time Range']
                    if other_time_range_list is not None:
                        other_time_range = pd.DatetimeIndex(pd.to_datetime(other_time_range_list))

                        # Check if there is an overlap in time ranges
                        overlap = other_time_range.intersection([flight_time])
                        if not overlap.empty: #ovelap not true= false
                            # Check if the other flight is holding
                            if df.at[other_index, 'Holding Time (minutes)'] > 0:
                                holding_flights += 1

                df.at[index, f'Holding flights at -{time}'] = holding_flights

    #drop the 'Time Range' column
    df=df.drop(columns='Time Range')

    return df


# In[ ]:


def calculate_acute_angle(angle1, angle2):

    # Calculate the absolute difference between the angles
    angle_difference = abs(angle1 - angle2)

    # Take the minimum of the absolute difference and its complement
    acute_angle = min(angle_difference, 360 - angle_difference)

    return acute_angle


# In[ ]:


import math

9#filtered_traffic_temp=egll2.between(date_to_filter,timedelta(hours=24),strict=True)
def no_of_landing(df,raw_df,arr_times):

    df['ROL_Range'] = None

    # Loop through each flight
    for index, row in df.iterrows():

        callsign=heathrow_total_hold_no_duplicates.at[index, 'Flight Number']
        # Filter the flight data for the current flight and sort by timestamp
        filter_df = raw_df[raw_df['flight_id'] == callsign].sort_values(by='timestamp')

        lastrowtime=filter_df['timestamp'].iloc[-1]
        lastrowtime=pd.to_datetime(lastrowtime)
        lastrowtime = lastrowtime.tz_localize(None)
        df.at[index, 'Last Timestamp'] = lastrowtime

        lasttrack=filter_df['track'].iloc[-1]
        lasttrack=round(lasttrack,0)
        df.at[index, 'Last Track'] = lasttrack

        lastlat=filter_df['latitude'].iloc[-1]
        lastlat=round(lastlat,3)
        df.at[index, 'Last Latitude'] = lastlat

        for time in arr_times:
            wind_dir = df.loc[df['Flight Number'] == callsign, f'Wind Direction at -{time}'].iloc[0]
            wind_speed = df.loc[df['Flight Number'] == callsign, f'Wind Speed at -{time}'].iloc[0]

            if lasttrack>0 and lasttrack<180 and lastlat>51.472: #Runway 09L
                if time ==0:
                    df.at[index, 'Runway 09L'] = 1
                    df.at[index, 'Runway 09R'] = 0
                    df.at[index, 'Runway 27L'] = 0
                    df.at[index, 'Runway 27R'] = 0

                angular_dif= calculate_acute_angle(wind_dir,90)
                cross_wind = np.round(np.abs(wind_speed * np.sin(np.radians(angular_dif))), 0)
                head_wind = np.round(np.abs(wind_speed * np.cos(np.radians(angular_dif))), 0)
                df.at[index, f'Crosswind Component at -{time}'] = cross_wind
                df.at[index, f'Headwind Component at -{time}'] = head_wind

            elif lasttrack>0 and lasttrack<180 and lastlat<51.472: #Runway 09R
                if time ==0:
                    df.at[index, 'Runway 09L'] = 0
                    df.at[index, 'Runway 09R'] = 1
                    df.at[index, 'Runway 27L'] = 0
                    df.at[index, 'Runway 27R'] = 0
                angular_dif= calculate_acute_angle(wind_dir,90)
                cross_wind = np.round(np.abs(wind_speed * np.sin(np.radians(angular_dif))), 0)
                head_wind = np.round(np.abs(wind_speed * np.cos(np.radians(angular_dif))), 0)
                df.at[index, f'Crosswind Component at -{time}'] = cross_wind
                df.at[index, f'Headwind Component at -{time}'] = head_wind

            elif lasttrack<360 and lasttrack>180 and lastlat>51.472: #Runway 27R
                if time ==0:
                    df.at[index, 'Runway 09L'] = 0
                    df.at[index, 'Runway 09R'] = 0
                    df.at[index, 'Runway 27L'] = 0
                    df.at[index, 'Runway 27R'] = 1
                angular_dif= calculate_acute_angle(wind_dir,270)
                cross_wind = np.round(np.abs(wind_speed * np.sin(np.radians(angular_dif))), 0)
                head_wind = np.round(np.abs(wind_speed * np.cos(np.radians(angular_dif))), 0)
                df.at[index, f'Crosswind Component at -{time}'] = cross_wind
                df.at[index, f'Headwind Component at -{time}'] = head_wind

            elif lasttrack<360 and lasttrack>180 and lastlat<51.472: #Runway 27L
                if time ==0:
                    df.at[index, 'Runway 09L'] = 0
                    df.at[index, 'Runway 09R'] = 0
                    df.at[index, 'Runway 27L'] = 1
                    df.at[index, 'Runway 27R'] = 0
                angular_dif= calculate_acute_angle(wind_dir,270)
                cross_wind = np.round(np.abs(wind_speed * np.sin(np.radians(angular_dif))), 0)
                head_wind = np.round(np.abs(wind_speed * np.cos(np.radians(angular_dif))), 0)
                df.at[index, f'Crosswind Component at -{time}'] = cross_wind
                df.at[index, f'Headwind Component at -{time}'] = head_wind


        # Create a range of times with a frequency of 30 seconds
        time_range = pd.date_range(end=lastrowtime,freq='30s',periods=120)
        #print(callsign)
        #print(time_range)
        df.at[index, 'ROL_Range'] = time_range

    for index, row in df.iterrows():
        current_time_range_list = df.at[index, 'ROL_Range']
        current_time_range = pd.DatetimeIndex(pd.to_datetime(current_time_range_list))
        no_of_landings=0
        for time in arr_times:
            no_of_landings=0
            for other_index, other_row in df.iterrows():
                if other_index == index:
                    continue
                else:
                    other_time= df.at[other_index,'Last Timestamp']
                    other_time = other_time.tz_localize(None)
                    other_time= other_time + pd.Timedelta(minutes=time)
                    
                    # Check if there is an overlap in time range
                    overlap = current_time_range.intersection([other_time])
                    #print(f'overlap is: {overlap}')
                    if not overlap.empty: #ovelap not true= false
                        no_of_landings += 1
            
            #print(no_of_landings)
            df.at[index, f'No of Landings 1HR at -{time}'] = no_of_landings
    #drop the 'Time Range' column
    df=df.drop(columns='ROL_Range')


    return df


# In[ ]:


def track_speed_alt (heathrow_total_df,arr_times):

    for time in arr_times:

        # Loop through each flight
        for index,row in heathrow_total_df.iterrows():

            callsign=heathrow_total_df.at[index, 'Flight Number']
            
            first_timestamp = heathrow_total_df.at[index, 'TimeStamp'] 

            first_timestamp=pd.to_datetime(first_timestamp)
            first_timestamp = first_timestamp.tz_localize(None)
            timestamp=first_timestamp-pd.Timedelta(minutes=time)
            #print(timestamp)
            
            raw_timestamp=egll_full_csv['timestamp']
            raw_timestamp=pd.to_datetime(raw_timestamp)
            raw_timestamp = raw_timestamp.dt.tz_localize(None)
            
            # Conditions
            callsign_condition = egll_full_csv['flight_id'] == callsign
            timestamp_condition = (raw_timestamp) == timestamp
            

            #Applying the conditions
            filtered_egll_full = egll_full_csv[callsign_condition & timestamp_condition]
            #print(filtered_egll_full)

            # Accessing
            if not filtered_egll_full.empty:
                ground_speed = (filtered_egll_full['groundspeed'].iloc[0]).round(decimals=-1)
                heathrow_total_df.at[index, f'Ground Speed at -{time}']=ground_speed
                
                track = (filtered_egll_full['track'].iloc[0]).round(decimals=0)
                heathrow_total_df.at[index, f'Track at -{time}']=track
                
                altitude = (filtered_egll_full['geoaltitude'].iloc[0]).round(decimals=-3)
                heathrow_total_df.at[index, f'Altitude at -{time}']=altitude
        
        heathrow_total_df[[f'Ground Speed at -{time}', f'Track at -{time}', f'Altitude at -{time}']] = heathrow_total_df[[f'Ground Speed at -{time}', f'Track at -{time}', f'Altitude at -{time}']].fillna(0)


    return heathrow_total_df


# In[ ]:


def trans_atlantic(heathrow_days_hold_df):
    #Trans atlantic TMA
    trans_atlantic=egll_full.inside_bbox((-20,48,-8,65))

    trans_atlantic.to_csv("trans_atlantic")
    trans_atlantic_csv=pd.read_csv("trans_atlantic")

    for index, row in heathrow_days_hold_df.iterrows():

        flight_number = heathrow_days_hold_df.at[index, 'Flight Number']
        Ground_speed_at_60 = heathrow_days_hold_df.at[index, 'Ground Speed at -60']

        if Ground_speed_at_60 ==0 and trans_atlantic_csv['flight_id'].str.contains(flight_number).any():
            # Filter the flight data for the current flight and sort by timestamp
            df_flight = trans_atlantic_csv[trans_atlantic_csv['flight_id'] == flight_number].sort_values(by='timestamp')
            #print(df_flight)
            first_ground_speed= (df_flight.iloc[0]['groundspeed']).round(decimals=-1)
            first_track= (df_flight.iloc[0]['track']).round(decimals=0)
            first_attitude= (df_flight.iloc[0]['geoaltitude']).round(decimals=-3)

            heathrow_days_hold_df.at[index, 'Ground Speed at -60']=first_ground_speed
            heathrow_days_hold_df.at[index, 'Track at -60']=first_track
            heathrow_days_hold_df.at[index, 'Altitude at -60']=first_attitude

    return heathrow_days_hold_df


# In[ ]:


def delay_index(heathrow_df,delay_index_df,arr_times):
    heathrow_df = heathrow_df.sort_values(by='TimeStamp')
    delay_index_df = delay_index_df.sort_values(by='to_utc')
    for index, row in heathrow_df.iterrows():
        flight_number = heathrow_df.at[index, 'Flight Number']
        timestamp = heathrow_df.at[index, 'TimeStamp']
        
        for time in arr_times:
            timestamp=timestamp-pd.Timedelta(minutes=time)
            timestamp_df = pd.DataFrame([timestamp], columns=['time_stamp'])
            merged_delay_index = pd.merge_asof(timestamp_df, delay_index_df, left_on='time_stamp', right_on='to_utc', direction='backward')
            #heathrow_df.at[index, f'To Utc at -{time}']=merged_delay_index.at[0, 'to_utc']
            heathrow_df.at[index, f'Departures delayIndex at -{time}']=merged_delay_index.at[0, 'departures_delayIndex']
            heathrow_df.at[index, f'Arrivals numCancelled at -{time}']=merged_delay_index.at[0, 'arrivals_numCancelled']
            heathrow_df.at[index, f'Arrivals delayIndex at -{time}']=merged_delay_index.at[0, 'arrivals_delayIndex']
        
    return heathrow_df


# In[ ]:


#egll_full=opensky.history(start, end, bounds=egll_full_bounds, arrival_airport=station)


# In[1]:


from traffic.core import Traffic
from traffic.data import opensky
from traffic.data import eurofirs
from shapely.geometry import Polygon
import datetime
import time
import pandas as pd
from sqlalchemy import create_engine, text
from pyopensky import trino

import logging
logging.basicConfig(level=logging.INFO)

# Initialize Trino instance
trino_instance = trino.Trino(connect_timeout=120, read_timeout=500)

# Define the start and stop timestamps
start = '2023-08-01 00:00:00+0000'
stop = '2023-08-01 23:30:00+0000'
start_ts = pd.to_datetime(start)
stop_ts = pd.to_datetime(stop)

egll_full_bounds=(-22,42,15,65)

# Define batch size
batch_size = pd.Timedelta(hours=12)

# Convert geographical bounds
lon_1, lon_2 = -22, 15
lat_1, lat_2 = 42, 65
estarrivalairport_1 = 'EGLL'
day_1_ts = int(start_ts.floor("1d").timestamp())
day_2_ts = int(stop_ts.ceil("1d").timestamp())
time_1_ts = int(start_ts.timestamp())
time_2_ts = int(stop_ts.timestamp())


# Create an engine (replace with your database connection string)
engine = trino_instance.engine()

# Initialize an empty DataFrame to store results
egll_full = pd.DataFrame()

# Loop through time intervals and fetch data in batches
current_start = start_ts
while current_start < stop_ts:
    current_stop = min(current_start + batch_size, stop_ts)
    
    # Convert timestamps to Unix time (seconds since epoch)
    time_1_ts = int(current_start.timestamp())
    time_2_ts = int(current_stop.timestamp())
    
    # Convert hour to integer Unix time
    hour_1_ts = int(current_start.replace(minute=0, second=0, microsecond=0).timestamp())
    hour_2_ts = int(current_stop.replace(minute=0, second=0, microsecond=0).timestamp())
    
    # Prepare query using f-strings
    query = f"""
    SELECT 
        v.time, 
        v.icao24, 
        v.lat, 
        v.lon, 
        v.velocity, 
        v.heading, 
        v.vertrate, 
        v.callsign, 
        v.onground, 
        v.alert, 
        v.spi, 
        v.squawk, 
        v.baroaltitude, 
        v.geoaltitude, 
        v.lastposupdate, 
        v.lastcontact, 
        v.serials, 
        v.hour 
    FROM 
        state_vectors_data4 v 
    JOIN 
        (
            SELECT 
                FLOOR(time / 30) AS halfminute, 
                MAX(time) AS recent, 
                icao24 
            FROM 
                state_vectors_data4 
            WHERE 
                time >= {time_1_ts} 
                AND time <= {time_2_ts}
            GROUP BY 
                icao24, FLOOR(time / 30)
        ) AS m 
    ON 
        v.icao24 = m.icao24 
        AND v.time = m.recent 
    JOIN 
        (
            SELECT 
                flights_data4.icao24 AS icao24, 
                flights_data4.callsign AS callsign, 
                flights_data4.firstseen AS firstseen, 
                flights_data4.lastseen AS lastseen, 
                flights_data4.estdepartureairport AS estdepartureairport, 
                flights_data4.estarrivalairport AS estarrivalairport 
            FROM 
                flights_data4 
            WHERE 
                flights_data4.day >= {day_1_ts} 
                AND flights_data4.day <= {time_2_ts} 
                AND flights_data4.estarrivalairport = '{estarrivalairport_1}'
        ) AS anon_1 
    ON 
        anon_1.icao24 = v.icao24 
        AND anon_1.callsign = v.callsign 
    WHERE 
        v.time >= anon_1.firstseen 
        AND v.time <= anon_1.lastseen 
        AND v.lon >= {lon_1} 
        AND v.lon <= {lon_2} 
        AND v.lat >= {lat_1} 
        AND v.lat <= {lat_2} 
        AND v.time >= {time_1_ts} 
        AND v.time <= {time_2_ts} 
        AND v.hour >= {hour_1_ts} 
        AND v.hour <= {hour_2_ts}
        AND v.velocity IS NOT NULL
        AND v.heading IS NOT NULL
        AND v.vertrate IS NOT NULL
        AND v.callsign IS NOT NULL
    """

    # Print the query for debugging purposes
    print(f"Executing query: {query}")
    
    # Retry mechanism
    max_retries = 5
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            query_batch = trino_instance.query(query)
            # Append the batch to the full DataFrame
            egll_full = pd.concat([egll_full, query_batch], ignore_index=True)
            break  # Exit the retry loop if successful
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.error("Max retries reached. Moving to the next batch.")
    
    '''
    # Execute the query
    with engine.connect() as connection:
        batch_df = pd.read_sql_query(query, connection)
        # Append the batch to the full DataFrame
        egll_full = pd.concat([egll_full, batch_df], ignore_index=True)
    
    
    '''
    
    
    # Move to the next batch
    current_start = current_stop

# Print or work with the results
print(egll_full.head())


start_weat=pd.to_datetime(start)
end_weat=pd.to_datetime(stop)
station_weather=[estarrivalairport_1]
weather=main_weather(start_weat,end_weat,station_weather)
weather

delay_index_df=pd.read_csv('Delayindex_of_EGLL_from_2023-03-25_01_00_till_2024-03-23_23_30')
delay_index_df=delay_index_df.fillna(0)
delay_index_df=delay_index_df.drop(columns=["Unnamed: 0",'airportIcao','from_utc','departures_numTotal','departures_numCancelled','departures_medianDelay','arrivals_numTotal','arrivals_medianDelay'])
delay_index_df['to_utc'] = pd.to_datetime(delay_index_df['to_utc'] )
delay_index_df['to_utc']=delay_index_df['to_utc'].dt.tz_localize(None)
delay_index_df


# In[ ]:


egll_full.columns = ['timestamp','icao24','latitude','longitude','groundspeed','track','vertical_rate','callsign','onground','alert','spi','squawk','baroaltitude','geoaltitude','last_position','#1','#2','hour']

egll_full['timestamp'] = pd.to_datetime(egll_full['timestamp'], unit='s')
egll_full['timestamp'] = egll_full['timestamp'].dt.tz_localize('UTC')

egll_full['last_position'] = pd.to_datetime(egll_full['last_position'], unit='s')
egll_full['#1'] = pd.to_datetime(egll_full['#1'], unit='s')
egll_full['hour'] = pd.to_datetime(egll_full['hour'], unit='s')

egll_full['geoaltitude']=(egll_full['geoaltitude']*3.28084).round()
egll_full['baroaltitude']=(egll_full['baroaltitude']*3.28084).round()
egll_full['groundspeed']=(egll_full['groundspeed']*1.944).round()
egll_full['vertical_rate']=egll_full['vertical_rate']*3.28084

traffic_obj=Traffic(egll_full)


# In[ ]:





# In[ ]:


from datetime import datetime

egll_full=traffic_obj.assign_id()
egll_full=egll_full.eval(max_workers=12, desc="preprocessing assign id")

#London TMA
egll_tma=egll_full.inside_bbox((-1,50.5,1.5,52))


egll_full.to_csv("egll_full")
egll_full_csv=pd.read_csv("egll_full")

range_ap=pd.date_range(start_ts,stop_ts)

heathrow_days_hold=[]
heathrow_days_hold_df=[]
for i in range_ap:
    date_to_filter=i.date()

    # Set the desired time to midnight (00:00:00)
    date_to_filter = datetime.combine(date_to_filter, datetime.min.time(),tzinfo=pytz.UTC)

    print(date_to_filter)
    filtered_traffic=egll_tma.between(date_to_filter,timedelta(hours=24),strict=True)
    if filtered_traffic is not None :
        #filtered_traffic.to_csv("filtered_traffic")
        filtered_traffic = filtered_traffic.reset_index(drop=True)


        #Calling the methods to calculate holding
        heathrow_total_hold_no_duplicates,egll_final=calulate_heathrow_hold (filtered_traffic)
        heathrow_total_hold_no_duplicates=Icao_doc8643(heathrow_total_hold_no_duplicates)
        heathrow_total_hold_w_weather= weather_cal(weather,heathrow_total_hold_no_duplicates,egll_final,[0, 30, 60])
        
         # Extract day, month, year, hour, minute, and day of the week
        heathrow_total_hold_w_weather['day'] = heathrow_total_hold_w_weather['TimeStamp'].dt.day
        heathrow_total_hold_w_weather['month'] = heathrow_total_hold_w_weather['TimeStamp'].dt.month
        heathrow_total_hold_w_weather['year'] = heathrow_total_hold_w_weather['TimeStamp'].dt.year
        heathrow_total_hold_w_weather['hour'] = heathrow_total_hold_w_weather['TimeStamp'].dt.hour
        heathrow_total_hold_w_weather['minute'] = heathrow_total_hold_w_weather['TimeStamp'].dt.minute
        heathrow_total_hold_w_weather['day_of_week'] = heathrow_total_hold_w_weather['TimeStamp'].dt.weekday # Monday=0, Sunday=6

        for index, row in heathrow_total_hold_w_weather.iterrows():
            for dow in [0,1,2,3,4,5,6]:
                if heathrow_total_hold_w_weather.at[index,'day_of_week'] == dow:
                    heathrow_total_hold_w_weather.at[index,f'Day_{dow}']=1
                else:
                    heathrow_total_hold_w_weather.at[index,f'Day_{dow}']=0
        
        for time in [0,30,60]:
            # Convert Timestamp to Decimal Hours without Seconds
            current_time=heathrow_total_hold_w_weather['TimeStamp']-pd.Timedelta(minutes=time)
            heathrow_total_hold_w_weather[f'Decimal Hours -{time}'] =(current_time.dt.hour + current_time.dt.minute / 60).round(2)
        
        heathrow_total_hold_w_weather_holdprob=probabilty_flight_data(heathrow_total_hold_w_weather,[0,30,60])
        heathrow_total_hold_w_weather_holdprob_no_of_landing=no_of_landing(heathrow_total_hold_w_weather_holdprob,egll_final,[0, 30, 60])
        heathrow_df=track_speed_alt(heathrow_total_hold_w_weather_holdprob_no_of_landing,[0, 30, 60])
        heathrow_df=trans_atlantic(heathrow_df)
        heathrow_df= delay_index(heathrow_df,delay_index_df,[0,30,60])

        # Append filtered_df to heathrow_days_hold
        heathrow_days_hold.append(heathrow_df)
    else:
        print(f"No data for {date_to_filter}. Skipping to next date.")
        continue

# Concatenate the list of DataFrames into a single DataFrame
heathrow_days_hold_df = pd.concat(heathrow_days_hold, ignore_index=True)
print(heathrow_days_hold_df)


# In[ ]:


#heathrow_total_hold_no_duplicates.to_csv("holding and weather")
start=start.replace(" ", "_").replace(":", "-").replace('-00+0000','')
stop=stop.replace(" ", "_").replace(":", "-").replace('-00+0000','')
heathrow_days_hold_df.to_csv(f"{estarrivalairport_1}_{start}_till_{stop}.csv", index=True)


# In[ ]:


#jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000


# In[ ]:


from traffic.data import opensky
egll_full=opensky.history(start, stop, bounds=egll_full_bounds)


# In[3]:


egll_full.to_csv("egll_full")
egll_full_csv=pd.read_csv("egll_full")


# In[10]:


import folium
from folium.plugins import HeatMap

# Assuming the dataframe 'egll1' has columns 'latitude' and 'longitude'
lats = egll_full_csv['lat'].tolist()
lngs = egll_full_csv['lon'].tolist()

# Define the number of decimal places to round to
decimal_places = 10


# Round the latitude and longitude values
rounded_lats = [round(lat, decimal_places) for lat in lats]
rounded_lngs = [round(lng, decimal_places) for lng in lngs]

# Create a list of rounded [lat, lng] pairs
rounded_locations = list(zip(rounded_lats, rounded_lngs))

# Create a map centered at an appropriate location
m = folium.Map(location=[51.26, 0], zoom_start=12)

# Add the HeatMap to the map
HeatMap(rounded_locations, max_val=0.1, radius=10, min_opacity=1).add_to(m)

# Display the map
m


# In[6]:


egll_full_csv


# In[17]:


m1 = folium.Map(location=[48.8566, 2.3522], zoom_start=6)


# In[34]:


from folium import PolyLine

# Group by the flight ID (assuming icao24 is the unique flight identifier)
grouped = egll_full_csv.groupby('icao24')

# Iterate over each flight
for flight_id, flight_data in grouped:
    # Extract latitude and longitude data as a list of tuples
    coordinates = flight_data[['lat', 'lon']].dropna().values.tolist()
    
    # Add flight path to the map
    if len(coordinates) > 1:
        PolyLine(coordinates, color='blue', weight=0.0001, opacity=0.7).add_to(m1)

# Save the map to an HTML file
#m1.save("flight_paths_map.html")


# In[35]:


m1


# In[ ]:




