<p style="text-align:center">
    <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
    </a>
</p>


# **Launch Sites Locations Analysis with Folium**


Estimated time needed: **40** minutes


The launch success rate may depend on many factors such as payload mass, orbit type, and so on. It may also depend on the location and proximities of a launch site, i.e., the initial position of rocket trajectories. Finding an optimal location for building a launch site certainly involves many factors and hopefully we could discover some of the factors by analyzing the existing launch site locations.


In the previous exploratory data analysis labs, you have visualized the SpaceX launch dataset using `matplotlib` and `seaborn` and discovered some preliminary correlations between the launch site and success rates. In this lab, you will be performing more interactive visual analytics using `Folium`.


## Objectives


This lab contains the following tasks:

*   **TASK 1:** Mark all launch sites on a map
*   **TASK 2:** Mark the success/failed launches for each site on the map
*   **TASK 3:** Calculate the distances between a launch site to its proximities

After completed the above tasks, you should be able to find some geographical patterns about launch sites.


Let's first import required Python packages for this lab:



```python
#import piplite
#await piplite.install(['folium'])
#await piplite.install(['pandas'])
```


```python
import folium
import pandas as pd
```


```python
# Import folium MarkerCluster plugin
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon
```

If you need to refresh your memory about folium, you may download and refer to this previous folium lab:


[Generating Maps with Python](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module\_3/DV0101EN-3-5-1-Generating-Maps-in-Python-py-v2.0.ipynb)


## Task 1: Mark all launch sites on a map


First, let's try to add each site's location on a map using site's latitude and longitude coordinates


The following dataset with the name `spacex_launch_geo.csv` is an augmented dataset with latitude and longitude added for each site.


# Download and read the `spacex_launch_geo.csv`
from js import fetch
import io

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'
resp = await fetch(URL)
spacex_csv_file = io.BytesIO((await resp.arrayBuffer()).to_py())
spacex_df=pd.read_csv(spacex_csv_file)


```python
import requests
import pandas as pd
import io

# Define the URL for the CSV file
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Read the CSV content into a DataFrame
    spacex_df = pd.read_csv(io.StringIO(response.text))
    print(spacex_df.head())  # Display the first few rows of the DataFrame
else:
    print(f"Failed to download the file: {response.status_code}")


```

       Flight Number        Date Time (UTC) Booster Version  Launch Site  \
    0              1  2010-06-04   18:45:00  F9 v1.0  B0003  CCAFS LC-40   
    1              2  2010-12-08   15:43:00  F9 v1.0  B0004  CCAFS LC-40   
    2              3  2012-05-22    7:44:00  F9 v1.0  B0005  CCAFS LC-40   
    3              4  2012-10-08    0:35:00  F9 v1.0  B0006  CCAFS LC-40   
    4              5  2013-03-01   15:10:00  F9 v1.0  B0007  CCAFS LC-40   
    
                                                 Payload  Payload Mass (kg)  \
    0               Dragon Spacecraft Qualification Unit                0.0   
    1  Dragon demo flight C1, two CubeSats,  barrel o...                0.0   
    2                             Dragon demo flight C2+              525.0   
    3                                       SpaceX CRS-1              500.0   
    4                                       SpaceX CRS-2              677.0   
    
           Orbit         Customer        Landing Outcome  class        Lat  \
    0        LEO           SpaceX  Failure   (parachute)      0  28.562302   
    1  LEO (ISS)  NASA (COTS) NRO  Failure   (parachute)      0  28.562302   
    2  LEO (ISS)      NASA (COTS)             No attempt      0  28.562302   
    3  LEO (ISS)       NASA (CRS)             No attempt      0  28.562302   
    4  LEO (ISS)       NASA (CRS)             No attempt      0  28.562302   
    
            Long  
    0 -80.577356  
    1 -80.577356  
    2 -80.577356  
    3 -80.577356  
    4 -80.577356  


Now, you can take a look at what are the coordinates for each site.



```python
# Select relevant sub-columns: `Launch Site`, `Lat(Latitude)`, `Long(Longitude)`, `class`
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
launch_sites_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Launch Site</th>
      <th>Lat</th>
      <th>Long</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CCAFS LC-40</td>
      <td>28.562302</td>
      <td>-80.577356</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CCAFS SLC-40</td>
      <td>28.563197</td>
      <td>-80.576820</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KSC LC-39A</td>
      <td>28.573255</td>
      <td>-80.646895</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VAFB SLC-4E</td>
      <td>34.632834</td>
      <td>-120.610745</td>
    </tr>
  </tbody>
</table>
</div>



Above coordinates are just plain numbers that can not give you any intuitive insights about where are those launch sites. If you are very good at geography, you can interpret those numbers directly in your mind. If not, that's fine too. Let's visualize those locations by pinning them on a map.


We first need to create a folium `Map` object, with an initial center location to be NASA Johnson Space Center at Houston, Texas.



```python
# Start location is NASA Johnson Space Center
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=10)
```

We could use `folium.Circle` to add a highlighted circle area with a text label on a specific coordinate. For example,



```python
# Create a blue circle at NASA Johnson Space Center's coordinate with a popup label showing its name
circle = folium.Circle(nasa_coordinate, radius=1000, color='#0000FF', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
# Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
marker = folium.map.Marker(
    nasa_coordinate,
    # Create an icon as a text label
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#0000FF;"><b>%s</b></div>' % 'NASA JSC',
        )
    )
site_map.add_child(circle)
site_map.add_child(marker)
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-3.7.1.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-glyphicons.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_439680256ef0390fcb0fcc373c6dd7fb {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_439680256ef0390fcb0fcc373c6dd7fb&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_439680256ef0390fcb0fcc373c6dd7fb = L.map(
                &quot;map_439680256ef0390fcb0fcc373c6dd7fb&quot;,
                {
                    center: [29.559684888503615, -95.0830971930759],
                    crs: L.CRS.EPSG3857,
                    zoom: 5,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );





            var tile_layer_95f5cb4d4a6502b56899a85f3edb3edb = L.tileLayer(
                &quot;https://tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;\u0026copy; \u003ca href=\&quot;https://www.openstreetmap.org/copyright\&quot;\u003eOpenStreetMap\u003c/a\u003e contributors&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 19, &quot;maxZoom&quot;: 19, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            );


            tile_layer_95f5cb4d4a6502b56899a85f3edb3edb.addTo(map_439680256ef0390fcb0fcc373c6dd7fb);


            var circle_77f45b7b6f8cc63291cdde464e3a8de8 = L.circle(
                [29.559684888503615, -95.0830971930759],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#d35400&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#d35400&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 1000, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_439680256ef0390fcb0fcc373c6dd7fb);


        var popup_dc91f52a19f14d08fc6e6a3ced67cf47 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_bfec65650d90604d09705a88a4d3cce2 = $(`&lt;div id=&quot;html_bfec65650d90604d09705a88a4d3cce2&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;NASA Johnson Space Center&lt;/div&gt;`)[0];
                popup_dc91f52a19f14d08fc6e6a3ced67cf47.setContent(html_bfec65650d90604d09705a88a4d3cce2);



        circle_77f45b7b6f8cc63291cdde464e3a8de8.bindPopup(popup_dc91f52a19f14d08fc6e6a3ced67cf47)
        ;




            var marker_ca988532e14e8b7c3badb9200cad3063 = L.marker(
                [29.559684888503615, -95.0830971930759],
                {}
            ).addTo(map_439680256ef0390fcb0fcc373c6dd7fb);


            var div_icon_1b4da08616162495254ea267abae2db7 = L.divIcon({&quot;className&quot;: &quot;empty&quot;, &quot;html&quot;: &quot;\u003cdiv style=\&quot;font-size: 12; color:#0000FF;\&quot;\u003e\u003cb\u003eNASA JSC\u003c/b\u003e\u003c/div\u003e&quot;, &quot;iconAnchor&quot;: [0, 0], &quot;iconSize&quot;: [20, 20]});
            marker_ca988532e14e8b7c3badb9200cad3063.setIcon(div_icon_1b4da08616162495254ea267abae2db7);


            tile_layer_95f5cb4d4a6502b56899a85f3edb3edb.addTo(map_439680256ef0390fcb0fcc373c6dd7fb);


            var circle_f0a20959a58be762a1b8f30355d24788 = L.circle(
                [29.559684888503615, -95.0830971930759],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#0000FF&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#0000FF&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 1000, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_439680256ef0390fcb0fcc373c6dd7fb);


        var popup_290ba7af2210f045c72b326073cc9ae5 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_bde20188e3c0bd3dfd8bb52ea50db1d1 = $(`&lt;div id=&quot;html_bde20188e3c0bd3dfd8bb52ea50db1d1&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;NASA Johnson Space Center&lt;/div&gt;`)[0];
                popup_290ba7af2210f045c72b326073cc9ae5.setContent(html_bde20188e3c0bd3dfd8bb52ea50db1d1);



        circle_f0a20959a58be762a1b8f30355d24788.bindPopup(popup_290ba7af2210f045c72b326073cc9ae5)
        ;




            var marker_57aebe069cf3615dc01baf6417c7fef9 = L.marker(
                [29.559684888503615, -95.0830971930759],
                {}
            ).addTo(map_439680256ef0390fcb0fcc373c6dd7fb);


            var div_icon_4f1634d9524955b12d24bead5d040b8a = L.divIcon({&quot;className&quot;: &quot;empty&quot;, &quot;html&quot;: &quot;\u003cdiv style=\&quot;font-size: 12; color:#0000FF;\&quot;\u003e\u003cb\u003eNASA JSC\u003c/b\u003e\u003c/div\u003e&quot;, &quot;iconAnchor&quot;: [0, 0], &quot;iconSize&quot;: [20, 20]});
            marker_57aebe069cf3615dc01baf6417c7fef9.setIcon(div_icon_4f1634d9524955b12d24bead5d040b8a);

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



and you should find a small yellow circle near the city of Houston and you can zoom-in to see a larger circle.


Now, let's add a circle for each launch site in data frame `launch_sites`


*TODO:*  Create and add `folium.Circle` and `folium.Marker` for each launch site on the site map


An example of folium.Circle:


`folium.Circle(coordinate, radius=1000, color='#000000', fill=True).add_child(folium.Popup(...))`


An example of folium.Marker:


`folium.map.Marker(coordinate, icon=DivIcon(icon_size=(20,20),icon_anchor=(0,0), html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'label', ))`



```python
launch_sites_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Launch Site</th>
      <th>Lat</th>
      <th>Long</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CCAFS LC-40</td>
      <td>28.562302</td>
      <td>-80.577356</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CCAFS SLC-40</td>
      <td>28.563197</td>
      <td>-80.576820</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KSC LC-39A</td>
      <td>28.573255</td>
      <td>-80.646895</td>
    </tr>
    <tr>
      <th>3</th>
      <td>VAFB SLC-4E</td>
      <td>34.632834</td>
      <td>-120.610745</td>
    </tr>
  </tbody>
</table>
</div>




```python
import folium
from folium.features import DivIcon

# Assuming nasa_coordinate is defined elsewhere
nasa_coordinate = [28.562302, -80.577356]  # Example coordinates, adjust as needed

# Initialize the map centered around NASA coordinate
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)

# For each launch site, add a Circle object based on its coordinate (Lat, Long) values.
# In addition, add Launch site name as a popup label
for index, row in launch_sites_df.iterrows():
    coor = [row['Lat'], row['Long']]
    site = row['Launch Site']
    
    # Create a Circle and add it to the map
    circle = folium.Circle(
        location=coor, 
        radius=1000, 
        color='#0000FF', 
        fill=True
    ).add_child(folium.Popup(site))
    
    # Create a Marker and add it to the map
    marker = folium.Marker(
        location=coor, 
        icon=DivIcon(
            icon_size=(20,20),
            icon_anchor=(0,0), 
            html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % site
        )
    )
    
    # Add the Circle and Marker to the map
    site_map.add_child(circle)
    site_map.add_child(marker)

# Display the map
site_map


```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-3.7.1.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-glyphicons.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_77cbfd16f7a6c4608b49a1dc747f5d21 {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_77cbfd16f7a6c4608b49a1dc747f5d21&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_77cbfd16f7a6c4608b49a1dc747f5d21 = L.map(
                &quot;map_77cbfd16f7a6c4608b49a1dc747f5d21&quot;,
                {
                    center: [28.562302, -80.577356],
                    crs: L.CRS.EPSG3857,
                    zoom: 5,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );





            var tile_layer_7f005e9a948f436dd245cd9117679c2c = L.tileLayer(
                &quot;https://tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;\u0026copy; \u003ca href=\&quot;https://www.openstreetmap.org/copyright\&quot;\u003eOpenStreetMap\u003c/a\u003e contributors&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 19, &quot;maxZoom&quot;: 19, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            );


            tile_layer_7f005e9a948f436dd245cd9117679c2c.addTo(map_77cbfd16f7a6c4608b49a1dc747f5d21);


            var circle_b43dd8c914c009ac3dfb3ca8e77cc388 = L.circle(
                [28.56230197, -80.57735648],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#0000FF&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#0000FF&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 1000, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_77cbfd16f7a6c4608b49a1dc747f5d21);


        var popup_f41f40ac489375222a52a63c2f923638 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_98903691a532384e234c6309d23d30aa = $(`&lt;div id=&quot;html_98903691a532384e234c6309d23d30aa&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;CCAFS LC-40&lt;/div&gt;`)[0];
                popup_f41f40ac489375222a52a63c2f923638.setContent(html_98903691a532384e234c6309d23d30aa);



        circle_b43dd8c914c009ac3dfb3ca8e77cc388.bindPopup(popup_f41f40ac489375222a52a63c2f923638)
        ;




            var marker_3651512e85d9364c94f4be486260b808 = L.marker(
                [28.56230197, -80.57735648],
                {}
            ).addTo(map_77cbfd16f7a6c4608b49a1dc747f5d21);


            var div_icon_618a056dd291b7afbb3d9ea0a92f05ae = L.divIcon({&quot;className&quot;: &quot;empty&quot;, &quot;html&quot;: &quot;\u003cdiv style=\&quot;font-size: 12; color:#d35400;\&quot;\u003e\u003cb\u003eCCAFS LC-40\u003c/b\u003e\u003c/div\u003e&quot;, &quot;iconAnchor&quot;: [0, 0], &quot;iconSize&quot;: [20, 20]});
            marker_3651512e85d9364c94f4be486260b808.setIcon(div_icon_618a056dd291b7afbb3d9ea0a92f05ae);


            var circle_76f5ab2402f38908b46af5766bd14719 = L.circle(
                [28.56319718, -80.57682003],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#0000FF&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#0000FF&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 1000, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_77cbfd16f7a6c4608b49a1dc747f5d21);


        var popup_14237d71689132e83418103e27de8cb3 = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_20eea9707a752aeb9ee00d67cc20278e = $(`&lt;div id=&quot;html_20eea9707a752aeb9ee00d67cc20278e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;CCAFS SLC-40&lt;/div&gt;`)[0];
                popup_14237d71689132e83418103e27de8cb3.setContent(html_20eea9707a752aeb9ee00d67cc20278e);



        circle_76f5ab2402f38908b46af5766bd14719.bindPopup(popup_14237d71689132e83418103e27de8cb3)
        ;




            var marker_85bb1b6282e31e3e2b2ba9bc89902ee0 = L.marker(
                [28.56319718, -80.57682003],
                {}
            ).addTo(map_77cbfd16f7a6c4608b49a1dc747f5d21);


            var div_icon_285e64a32b2f21486c196e35dc1cf945 = L.divIcon({&quot;className&quot;: &quot;empty&quot;, &quot;html&quot;: &quot;\u003cdiv style=\&quot;font-size: 12; color:#d35400;\&quot;\u003e\u003cb\u003eCCAFS SLC-40\u003c/b\u003e\u003c/div\u003e&quot;, &quot;iconAnchor&quot;: [0, 0], &quot;iconSize&quot;: [20, 20]});
            marker_85bb1b6282e31e3e2b2ba9bc89902ee0.setIcon(div_icon_285e64a32b2f21486c196e35dc1cf945);


            var circle_8d526c67ed1abd5adbb57c2d7888aba3 = L.circle(
                [28.57325457, -80.64689529],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#0000FF&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#0000FF&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 1000, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_77cbfd16f7a6c4608b49a1dc747f5d21);


        var popup_e461c4421175afa01bc00285ed086c1d = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_ac5d840a45f47b23076eecdd1edb0257 = $(`&lt;div id=&quot;html_ac5d840a45f47b23076eecdd1edb0257&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;KSC LC-39A&lt;/div&gt;`)[0];
                popup_e461c4421175afa01bc00285ed086c1d.setContent(html_ac5d840a45f47b23076eecdd1edb0257);



        circle_8d526c67ed1abd5adbb57c2d7888aba3.bindPopup(popup_e461c4421175afa01bc00285ed086c1d)
        ;




            var marker_3cb16416d261a1f6155d9b0062421e8d = L.marker(
                [28.57325457, -80.64689529],
                {}
            ).addTo(map_77cbfd16f7a6c4608b49a1dc747f5d21);


            var div_icon_d6b9e664faf74c8d786665de2f150e15 = L.divIcon({&quot;className&quot;: &quot;empty&quot;, &quot;html&quot;: &quot;\u003cdiv style=\&quot;font-size: 12; color:#d35400;\&quot;\u003e\u003cb\u003eKSC LC-39A\u003c/b\u003e\u003c/div\u003e&quot;, &quot;iconAnchor&quot;: [0, 0], &quot;iconSize&quot;: [20, 20]});
            marker_3cb16416d261a1f6155d9b0062421e8d.setIcon(div_icon_d6b9e664faf74c8d786665de2f150e15);


            var circle_8842c831babef95920c1f82c748ae0c0 = L.circle(
                [34.63283416, -120.6107455],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;#0000FF&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: true, &quot;fillColor&quot;: &quot;#0000FF&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 1000, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_77cbfd16f7a6c4608b49a1dc747f5d21);


        var popup_5464ca0b6f9d9b8435fa8c57f58cfccb = L.popup({&quot;maxWidth&quot;: &quot;100%&quot;});



                var html_00d6a9e17edaf8d87f762530bb515d01 = $(`&lt;div id=&quot;html_00d6a9e17edaf8d87f762530bb515d01&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;VAFB SLC-4E&lt;/div&gt;`)[0];
                popup_5464ca0b6f9d9b8435fa8c57f58cfccb.setContent(html_00d6a9e17edaf8d87f762530bb515d01);



        circle_8842c831babef95920c1f82c748ae0c0.bindPopup(popup_5464ca0b6f9d9b8435fa8c57f58cfccb)
        ;




            var marker_439d4fe8b416de683328ce4532cefe57 = L.marker(
                [34.63283416, -120.6107455],
                {}
            ).addTo(map_77cbfd16f7a6c4608b49a1dc747f5d21);


            var div_icon_3cc02555fddb864fe9e3bdc890139286 = L.divIcon({&quot;className&quot;: &quot;empty&quot;, &quot;html&quot;: &quot;\u003cdiv style=\&quot;font-size: 12; color:#d35400;\&quot;\u003e\u003cb\u003eVAFB SLC-4E\u003c/b\u003e\u003c/div\u003e&quot;, &quot;iconAnchor&quot;: [0, 0], &quot;iconSize&quot;: [20, 20]});
            marker_439d4fe8b416de683328ce4532cefe57.setIcon(div_icon_3cc02555fddb864fe9e3bdc890139286);

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



The generated map with marked launch sites should look similar to the following:


<center>
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_3/images/launch_site_markers.png">
</center>


Now, you can explore the map by zoom-in/out the marked areas
, and try to answer the following questions:

*   Are all launch sites in proximity to the Equator line?
*   Are all launch sites in very close proximity to the coast?

Also please try to explain your findings.



```python
# Task 2: Mark the success/failed launches for each site on the map

```

Next, let's try to enhance the map by adding the launch outcomes for each site, and see which sites have high success rates.
Recall that data frame spacex_df has detailed launch records, and the `class` column indicates if this launch was successful or not



```python
spacex_df.tail(10)
```

Next, let's create markers for all launch records.
If a launch was successful `(class=1)`, then we use a green marker and if a launch was failed, we use a red marker `(class=0)`


Note that a launch only happens in one of the four launch sites, which means many launch records will have the exact same coordinate. Marker clusters can be a good way to simplify a map containing many markers having the same coordinate.


Let's first create a `MarkerCluster` object



```python
marker_cluster = MarkerCluster()

```

*TODO:* Create a new column in `spacex_df` dataframe called `marker_color` to store the marker colors based on the `class` value



```python

# Apply a function to check the value of `class` column
# If class=1, marker_color value will be green
# If class=0, marker_color value will be red
```

*TODO:* For each launch result in `spacex_df` data frame, add a `folium.Marker` to `marker_cluster`



```python
# Add marker_cluster to current site_map
site_map.add_child(marker_cluster)

# for each row in spacex_df data frame
# create a Marker object with its coordinate
# and customize the Marker's icon property to indicate if this launch was successed or failed, 
# e.g., icon=folium.Icon(color='white', icon_color=row['marker_color']
for index, record in spacex_df.iterrows():
    # TODO: Create and add a Marker cluster to the site map
    # marker = folium.Marker(...)
    marker_cluster.add_child(marker)

site_map
```

Your updated map may look like the following screenshots:


<center>
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_3/images/launch_site_marker_cluster.png">
</center>


<center>
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_3/images/launch_site_marker_cluster_zoomed.png">
</center>


From the color-labeled markers in marker clusters, you should be able to easily identify which launch sites have relatively high success rates.



```python
# TASK 3: Calculate the distances between a launch site to its proximities

```

Next, we need to explore and analyze the proximities of launch sites.


Let's first add a `MousePosition` on the map to get coordinate for a mouse over a point on the map. As such, while you are exploring the map, you can easily find the coordinates of any points of interests (such as railway)



```python
# Add Mouse Position to get the coordinate (Lat, Long) for a mouse over on the map
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)

site_map.add_child(mouse_position)
site_map
```

Now zoom in to a launch site and explore its proximity to see if you can easily find any railway, highway, coastline, etc. Move your mouse to these points and mark down their coordinates (shown on the top-left) in order to the distance to the launch site.


Now zoom in to a launch site and explore its proximity to see if you can easily find any railway, highway, coastline, etc. Move your mouse to these points and mark down their coordinates (shown on the top-left) in order to the distance to the launch site.



```python
from math import sin, cos, sqrt, atan2, radians

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance
```

*TODO:* Mark down a point on the closest coastline using MousePosition and calculate the distance between the coastline point and the launch site.



```python
# find coordinate of the closet coastline
# e.g.,: Lat: 28.56367  Lon: -80.57163
# distance_coastline = calculate_distance(launch_site_lat, launch_site_lon, coastline_lat, coastline_lon)
```


```python
# Create and add a folium.Marker on your selected closest coastline point on the map
# Display the distance between coastline point and launch site using the icon property 
# for example
# distance_marker = folium.Marker(
#    coordinate,
#    icon=DivIcon(
#        icon_size=(20,20),
#        icon_anchor=(0,0),
#        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance),
#        )
#    )
```

*TODO:* Draw a `PolyLine` between a launch site to the selected coastline point



```python
# Create a `folium.PolyLine` object using the coastline coordinates and launch site coordinate
# lines=folium.PolyLine(locations=coordinates, weight=1)
site_map.add_child(lines)
```

Your updated map with distance line should look like the following screenshot:


<center>
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_3/images/launch_site_marker_distance.png">
</center>


*TODO:* Similarly, you can draw a line betwee a launch site to its closest city, railway, highway, etc. You need to use `MousePosition` to find the their coordinates on the map first


A railway map symbol may look like this:


<center>
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_3/images/railway.png">
</center>


A highway map symbol may look like this:


<center>
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_3/images/highway.png">
</center>


A city map symbol may look like this:


<center>
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_3/images/city.png">
</center>



```python
# Create a marker with distance to a closest city, railway, highway, etc.
# Draw a line between the marker to the launch site

```


```python

```


```python

```

After you plot distance lines to the proximities, you can answer the following questions easily:

*   Are launch sites in close proximity to railways?
*   Are launch sites in close proximity to highways?
*   Are launch sites in close proximity to coastline?
*   Do launch sites keep certain distance away from cities?

Also please try to explain your findings.


# Next Steps:

Now you have discovered many interesting insights related to the launch sites' location using folium, in a very interactive way. Next, you will need to build a dashboard using Ploty Dash on detailed launch records.


## Authors


[Pratiksha Verma](https://www.linkedin.com/in/pratiksha-verma-6487561b1/)


<!--## Change Log--!>


<!--| Date (YYYY-MM-DD) | Version | Changed By      | Change Description      |
| ----------------- | ------- | -------------   | ----------------------- |
| 2022-11-09        | 1.0     | Pratiksha Verma | Converted initial version to Jupyterlite|--!>


### <h3 align="center"> IBM Corporation 2022. All rights reserved. <h3/>

