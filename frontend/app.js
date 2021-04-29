var slider = document.getElementById("dateSlider");
var output = document.getElementById("selectedDate");
output.innerHTML = new Date(); // Display the default slider value
var startIndex = slider.value;
var map;
var heatmap;

slider.oninput = function() {
    var d = new Date();
    var selected = new Date((this.value) * (1000*3600*24) + (d.getTime()));
    output.innerHTML = selected; //format to Month Day, Year
    initHeatMap(this.value);
}

/**
 * Reads a csv file, assign values (weights) to respective latitude and longitude,
 * and stores it as LatLng object to the points array.
 * @return Data points from day 0 to day 9
*/
async function getData() {
  var daysArray = [];
  var csv = ['../forecast/f0.csv', '../forecast/f1.csv', '../forecast/f2.csv', '../forecast/f3.csv', '../forecast/f4.csv','../forecast/f5.csv','../forecast/f6.csv',
  '../forecast/f7.csv', '../forecast/f8.csv', '../forecast/f9.csv'];
  for (var i = 0; i < csv.length; i++) {
    var arr = [];
    const response = await fetch(csv[i]);
    const dataFromCsv = await response.text();
    var lat = -90;
    const rows = dataFromCsv.split('\n');
    rows.forEach(elt => {
      var long = -180;
      const columns = elt.split(',');
      columns.forEach(e => {
        if (e != 0) {
          arr.push({location: new google.maps.LatLng( lat, long), weight: e});
        }
        long++;
      });
      lat++;
    });
    daysArray.push(arr);
  }
  return daysArray;
}

/**
 * Initializes the google maps API.
 */
function initMap() {
      map = new google.maps.Map(document.getElementById("map"), {
          zoom: 3,
          center: { lat: 37.774546, lng: -122.433523 },
          styles: [
              { elementType: "geometry", stylers: [{ color: "#242f3e" }] },
              { elementType: "labels.text.stroke", stylers: [{ color: "#242f3e" }] },
              { elementType: "labels.text.fill", stylers: [{ color: "#746855" }] },
              {
                featureType: "administrative.locality",
                elementType: "labels.text.fill",
                stylers: [{ color: "#d59563" }],
              },
              {
                featureType: "poi",
                elementType: "labels.text.fill",
                stylers: [{ color: "#d59563" }],
              },
              {
                featureType: "poi.park",
                elementType: "geometry",
                stylers: [{ color: "#263c3f" }],
              },
              {
                featureType: "poi.park",
                elementType: "labels.text.fill",
                stylers: [{ color: "#6b9a76" }],
              },
              {
                featureType: "road",
                elementType: "geometry",
                stylers: [{ color: "#38414e" }],
              },
              {
                featureType: "road",
                elementType: "geometry.stroke",
                stylers: [{ color: "#212a37" }],
              },
              {
                featureType: "road",
                elementType: "labels.text.fill",
                stylers: [{ color: "#9ca5b3" }],
              },
              {
                featureType: "road.highway",
                elementType: "geometry",
                stylers: [{ color: "#746855" }],
              },
              {
                featureType: "road.highway",
                elementType: "geometry.stroke",
                stylers: [{ color: "#1f2835" }],
              },
              {
                featureType: "road.highway",
                elementType: "labels.text.fill",
                stylers: [{ color: "#f3d19c" }],
              },
              {
                featureType: "transit",
                elementType: "geometry",
                stylers: [{ color: "#2f3948" }],
              },
              {
                featureType: "transit.station",
                elementType: "labels.text.fill",
                stylers: [{ color: "#d59563" }],
              },
              {
                featureType: "water",
                elementType: "geometry",
                stylers: [{ color: "#17263c" }],
              },
              {
                featureType: "water",
                elementType: "labels.text.fill",
                stylers: [{ color: "#515c6d" }],
              },
              {
                featureType: "water",
                elementType: "labels.text.stroke",
                stylers: [{ color: "#17263c" }],
              },
            ],
      });
      initHeatMap(0);
}

/**
 * Creates the heatmap layer
 *  @param day, day 0 - 9; Day used in getting the heatmap points
 */
function initHeatMap(day) {
  (async () => {
    var pointsArray = await getData();

    if (heatmap) {
      heatmap.setMap(null);
      heatmap.setData([]);
    }

    heatmap = new google.maps.visualization.HeatmapLayer({
      data: pointsArray[day], 
      opacity: 0.5,
      map: map,
      radius: 3,
      maxIntensity: 10000,
      dissipating: false,
    });
  })(); 

}