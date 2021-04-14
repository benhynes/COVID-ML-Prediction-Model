
import React from "react"
import {
  GoogleMap,
  useLoadScript,
  HeatmapLayer
} from "@react-google-maps/api"

import mapStyles from "./mapStyle";

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
  radius:20,
  dissipating: true,
  opacity:0.5,
  gradient:[
    "rgba(0, 255, 255, 0)",
    "rgba(0, 255, 255, 1)",
    "rgba(0, 191, 255, 1)",
    "rgba(0, 127, 255, 1)",
    "rgba(0, 63, 255, 1)",
    "rgba(0, 0, 255, 1)",
    "rgba(0, 0, 223, 1)",
    "rgba(0, 0, 191, 1)",
    "rgba(0, 0, 159, 1)",
    "rgba(0, 0, 127, 1)",
    "rgba(63, 0, 91, 1)",
    "rgba(127, 0, 63, 1)",
    "rgba(191, 0, 31, 1)",
    "rgba(255, 0, 0, 1)",
  ]
};
const onLoad = heatmapLayer => {
  console.log('HeatmapLayer onLoad heatmapLayer: ', heatmapLayer)
}

const onUnmount = heatmapLayer => {
  console.log('HeatmapLayer onUnmount heatmapLayer: ', heatmapLayer)
}
export default function App() {
  const {isLoaded, loadError} = useLoadScript({
    googleMapsApiKey: process.env.REACT_APP_GOOGLE_MAPS_API_KEY,
    libraries: ["visualization"]
  });

  if (loadError) return "Error loading maps";
  if (!isLoaded) return "Loading Maps";

  return <div>

    <GoogleMap
      mapContainerStyle={mapContainerStyle}
      zoom={13}
      center= {center}
      options={options}
    >
    <HeatmapLayer
      // optional
      onLoad={onLoad}
      // optional
      onUnmount={onUnmount}
      // required
      options={heatmapOptions}
      data={[
        {location: new window.google.maps.LatLng(37.782, -122.447), weight: 0.5},
        new window.google.maps.LatLng(37.782, -122.445),
        {location: new window.google.maps.LatLng(37.782, -122.443), weight: 2},
        {location: new window.google.maps.LatLng(37.782, -122.441), weight: 3},
        {location: new window.google.maps.LatLng(37.782, -122.439), weight: 2},
        new window.google.maps.LatLng(37.782, -122.437),
        {location: new window.google.maps.LatLng(37.782, -122.435), weight: 0.5},

        {location: new window.google.maps.LatLng(37.785, -122.447), weight: 3},
        {location: new window.google.maps.LatLng(37.785, -122.445), weight: 2},
        new window.google.maps.LatLng(37.785, -122.443),
        {location: new window.google.maps.LatLng(37.785, -122.441), weight: 0.5},
        new window.google.maps.LatLng(37.785, -122.439),
        {location: new window.google.maps.LatLng(37.785, -122.437), weight: 2},
        {location: new window.google.maps.LatLng(37.785, -122.435), weight: 3}
      ]}
      
    />
    </GoogleMap>
  </div>
}