var map, heatmap;
var months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
var date_today = new Date();
const MILLISECONDS_IN_A_DAY = 86400000;

var slider = document.getElementById("dateSlider");
var date_output = document.getElementById("selectedDate");
date_output.innerHTML = months[date_today.getMonth()] + " " + date_today.getDate() + ", " + date_today.getFullYear();

var total_number = document.getElementById("totalNumber");

slider.oninput = function() {
    var date = new Date();
    var selected_day = new Date((this.value) * MILLISECONDS_IN_A_DAY + (date.getTime()));
    date_output.innerHTML = months[selected_day.getMonth()] + " " + selected_day.getDate() + ", " + selected_day.getFullYear();
    initHeatMap(this.value);
}

/**
 * Reads a csv file, assign values (weights) to respective latitude and longitude,
 * and stores it as LatLng object to the points array.
 * @return Data points from day 0 to day 9
*/
async function getData() {
  var days_array = [];
  var csv_files = ['../forecast/f0.csv', '../forecast/f1.csv', '../forecast/f2.csv', '../forecast/f3.csv', '../forecast/f4.csv','../forecast/f5.csv','../forecast/f6.csv',
  '../forecast/f7.csv', '../forecast/f8.csv', '../forecast/f9.csv'];
  for (var i = 0; i < csv_files.length; i++) {
    var points_array = [];
    const response = await fetch(csv_files[i]);
    const data_from_csv = await response.text();
    var lat = -90;
    const rows = data_from_csv.split('\n');
    rows.forEach(row => {
      var long = -180;
      const columns = row.split(',');
      columns.forEach(weight => {
        if (weight != 0) {
          points_array.push({location: new google.maps.LatLng( lat, long), weight: weight});
        }
        long++;
      });
      lat++;
    });
    days_array.push(points_array);
  }
  return days_array;
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

function getTotalNumber(selected_day) {
  var total_number_of_cases = 0;
  for (var i = 0; i < selected_day.length; i++) {
    total_number_of_cases += parseFloat(selected_day[i]['weight']);
  }
  total_number.innerHTML = total_number_of_cases;
}

function addMarker(selected_day) {
  var infowindow = new google.maps.InfoWindow;
  var marker;
  var location;

  for (var i = 0; i < selected_day.length; i++) {

    marker = new google.maps.Marker({
      position: selected_day[i]['location'],
      map,
      icon:  {
        path: google.maps.SymbolPath.CIRCLE,
        scale: 0
      },
    });
    
    google.maps.event.addListener(marker, 'mouseover', (function(marker, i) {
      return function() {
        location = selected_day[i]['location'];
        weight = selected_day[i]['weight'];
          content = 
            'Lat: ' + location.lat() + '  Long: ' + location.lng() + 
            '<h2>' + weight + '</h2>';
          infowindow.setContent(content);
          infowindow.open(map, marker);
      }
    })(marker, i));

    google.maps.event.addListener(marker, 'mouseout', (function(marker, i) {
      return function() {
          infowindow.close();
      }
    })(marker, i));
  }
}

/**
 * Creates the heatmap layer
 *  @param day, day 0 - 9; Day used in getting the heatmap points
 */
function initHeatMap(day) {
  (async () => {
    var days_array = await getData();
    
    var selected_day = days_array[day];
    getTotalNumber(selected_day);
    
    addMarker(selected_day);
    
    if (heatmap) {
      heatmap.setMap(null);
      heatmap.setData([]);
    }
    
    heatmap = new google.maps.visualization.HeatmapLayer({
      data: days_array[day], 
      opacity: 0.5,
      map: map,
      radius: 3,
      maxIntensity: 20000,
      dissipating: false,
    });
  })(); 
}