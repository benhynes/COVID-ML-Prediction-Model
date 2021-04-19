import React, { useState, useEffect } from 'react';
import {
  GoogleMap,
  useLoadScript,
  HeatmapLayer,
  Data
} from "@react-google-maps/api"
import dataCSV from './data.csv'
import mapStyles from "./mapStyle";
import { csv }from 'd3';

const mapContainerStyle = {
  width: "100vw",
  height: "80vw"
};
const center = {
  lat: 37.782,
  lng: -122.447,
};
const options = {
  styles: mapStyles,
  disableDefaultUI: true,
  zoomControl: true,
};
const heatmapOptions = {
  radius:10,
  dissipating: true,
  opacity:0.5,
  
};
//const Date = '2/4/21';

export default function App() {

  const {isLoaded, loadError} = useLoadScript({
    googleMapsApiKey: process.env.REACT_APP_GOOGLE_MAPS_API_KEY,
    libraries: ["visualization"]
  });

  //Initialize an empty map data
  const [mapData, setData] = useState([]);
  
  //
  useEffect(() => {
    csv(dataCSV).then(dataCSV => {
      setData(dataCSV);
    });
  }, [])

  var weight;
  const filteredMapData = 
    mapData.map(function(d) {
      weight = d["2/22/21"];
      var lat = d.Lat;
      var long = d.Long;
      return {
        location: new window.google.maps.LatLng(d.Lat, d.Long),
        //weight: weight
      }
    });

  console.log(filteredMapData);

  if (loadError) return "Error loading maps";
  if (!isLoaded) return "Loading Maps";
  return <div>

    <GoogleMap
      mapContainerStyle={mapContainerStyle}
      zoom={3}
      center= {center}
      options={options}
    >
    <HeatmapLayer
      options={heatmapOptions}
      data={filteredMapData}
    />
    </GoogleMap>
  </div>
}
