import pycaret
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import folium
from geopy.geocoders import Nominatim
from geopy import distance
import imageio
from tqdm import tqdm_notebook
from folium.plugins import MarkerCluster
#import geoplot as gplt
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#import geoplot.crs as gcrs
from sklearn.metrics import accuracy_score
import imageio
import mapclassify as mc
import scipy
from itertools import product
import seaborn as sns
from sklearn.metrics import accuracy_score

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['font.serif'] = 'Ubuntu' 
plt.rcParams['font.monospace'] = 'Ubuntu Mono' 
plt.rcParams['font.size'] = 14 
plt.rcParams['axes.labelsize'] = 12 
plt.rcParams['axes.labelweight'] = 'bold' 
plt.rcParams['axes.titlesize'] = 12 
plt.rcParams['xtick.labelsize'] = 12 
plt.rcParams['ytick.labelsize'] = 12 
plt.rcParams['legend.fontsize'] = 12 
plt.rcParams['figure.titlesize'] = 12 
plt.rcParams['image.cmap'] = 'jet' 
plt.rcParams['image.interpolation'] = 'none' 
plt.rcParams['figure.figsize'] = (12, 10) 
plt.rcParams['axes.grid']=True
plt.rcParams['lines.linewidth'] = 2 
plt.rcParams['lines.markersize'] = 8
colors = ['xkcd:pale orange', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta', 'xkcd:dull purple', 'xkcd:teal', 'xkcd: goldenrod', 'xkcd:cadet blue',
'xkcd:scarlet']

data = pd.read_csv('California_Fire_Incidents.csv')
data = data[['Latitude','Longitude','Started','AcresBurned','Counties']]

city_data = data.drop_duplicates(subset=['Counties'])

LAT = []
LONG = []
for city in city_data.Counties.tolist():
    locator = Nominatim(user_agent="myGeocoder")
    location = locator.geocode(city)
    LAT.append(location.latitude)
    LONG.append(location.longitude)
    
city_data.Latitude = LAT
city_data.Longitude = LONG
city_data = city_data.drop(city_data[city_data.Counties=='Santa Cruz'].index)

world_map= folium.Map()
geolocator = Nominatim(user_agent="Piero")
marker_cluster = MarkerCluster().add_to(world_map)

for i in range(len(city_data)):
        lat = city_data.iloc[i]['Latitude']
        long = city_data.iloc[i]['Longitude']
        radius=5
        folium.CircleMarker(location = [lat, long], radius=radius,fill =True, color='darkred',fill_color='darkred').add_to(marker_cluster)

world_map

data = data[data['Counties']!='Santa Cruz']
plt.figure(figsize=(20,20))
sns.countplot(data.Counties)
plt.xticks(rotation=90)
plt.grid(True)

lat = city_data.Latitude.to_list()
lon = city_data.Longitude.to_list()
dist = []
for i in range(len(lat)):
  lat_i = (lat[i],lon[i])
  I_dist = []
  for j in range(len(lat)):
    lat_j = (lat[j],lon[j])
    I_dist.append(distance.geodesic(lat_i,lat_j).km)
  dist.append(I_dist)
dist = np.array(dist)

dist_data = pd.DataFrame(dist,columns=city_data.Counties)
dist_data.index = city_data.Counties
plt.figure(figsize=(20,20))
sns.heatmap(dist_data)

started = data.Started.to_list()
data['Started']=pd.to_datetime(started)

cities_to_plot = city_data.Counties.to_list()[0:4]
colors = ['navy','k','darkorange','firebrick']
for i in range(len(cities_to_plot)):
  plt.subplot(2,2,i+1)
  sns.histplot(data[data['Counties']==cities_to_plot[i]].Started,color=colors[i],bins=5)
  plt.title('City = %s'%(cities_to_plot[i]))
  plt.xlabel('Time')
  plt.ylabel('Number of Wildfires')
plt.tight_layout()

la = data[data['Counties']=='Los Angeles'].sort_values(by='Started').reset_index().drop(['index','AcresBurned'],axis=1)
data = data.sort_values(by='Started').reset_index().drop(['index'],axis=1)
data = data.drop([0,1]).reset_index().drop('index',axis=1)
non_la = data[data['Counties']!='Los Angeles'].reset_index().drop('index',axis=1)
non_la['AcresBurned'] = non_la['AcresBurned'].fillna(non_la['AcresBurned'].mean())
one_three = []
three_seven = []
seven_ten = []
no = []
for i in range(len(la)):
  diff = non_la['Started']-la['Started'].loc[i]
  diff = diff.reset_index().drop('index',axis=1)
  s = diff.Started.tolist()
  for d in range(len(s)):
    day = s[d].days
    if (day<=-1) == True and (day>=-3)==True:
      one_three.append(d)
    if (day<-3) == True and  (day>=-7) == True:
      three_seven.append(d)
    if (day<-7) == True and  (day>=-10)==True:
      seven_ten.append(d)
    if (day<-10)==True:
      no.append(d)
      
one_three_data = non_la.loc[one_three][0:180]
one_three_data['Class'] = ['1-3 Days']*len(one_three_data)
three_seven_data = non_la.loc[three_seven][0:180]
three_seven_data['Class'] = ['3-7 Days']*len(three_seven_data) 
seven_ten_data = non_la.loc[seven_ten][0:180]
seven_ten_data['Class'] = ['7-10 Days']*len(seven_ten_data)
no_fire_data = non_la.loc[no][0:180]
no_fire_data['Class'] = ['No Wildfire (+10 Days)']*len(no_fire_data)
no_fire_data = no_fire_data.sample(frac=1)

wildfire_data = one_three_data.append(three_seven_data).append(seven_ten_data).append(no_fire_data)
wildfire_data = wildfire_data.reset_index().drop('index',axis=1)
distances_la = []
for i in range(len(wildfire_data)):
  distances_la.append(dist_data[wildfire_data['Counties'].loc[i]].loc['Los Angeles'])
wildfire_data['Distance'] = distances_la
wildfire_data['Class'] = LabelEncoder().fit_transform(wildfire_data.Class)
years = []
months = []
days = []
for i in range(len(wildfire_data)):
  years.append(wildfire_data.Started[i].year)
  months.append(wildfire_data.Started[i].month)
  days.append(wildfire_data.Started[i].day)
wildfire_data['Year'] = years
wildfire_data['Month'] = months
wildfire_data['Days'] = days
wildfire_data = wildfire_data.sample(frac=1)
wildfire_data

from pycaret.classification import *

X = wildfire_data[['Distance','Year','Month','Days','AcresBurned']]
Y = wildfire_data.Class
data_classification = wildfire_data[['Distance','Year','Month','Days','AcresBurned','Class']]
data_classification['Year'] = data_classification['Year'].astype(float)
data_classification['Month'] = data_classification['Month'].astype(float)
data_classification['Day'] = data_classification['Days'].astype(float)
data_classification = data_classification.dropna()
data_classification 

s = setup(data_classification, target = 'Class',imputation_type='iterative',train_size=0.9)

compare_models()

et = create_model('et')

et = tune_model(et)

model = finalize_model(et)

model = finalize_model(et)

X_train,X_test,y_train,y_test = train_test_split(X,Y, train_size=0.9)

results = pd.DataFrame({'Predict':model.predict(X_test),'Target':y_test})

from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(results['Predict'],results['Target']),annot=True,xticklabels=['1-3 Days','3-7 Days','7-10 Days','No Wildfire']
            ,yticklabels=['1-3 Days','3-7 Days','7-10 Days','No Wildfire'],cmap='plasma')
plt.ylabel('Predicted')
plt.xlabel('Target')
acc_score = accuracy_score(y_test.tolist(), model.predict(X_test).tolist())

