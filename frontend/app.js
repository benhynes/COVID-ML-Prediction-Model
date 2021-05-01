var map, heatmap;
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const DATE_TODAY = new Date();
const MILLISECONDS_IN_A_DAY = 86400000;
google.charts.load('46', {'packages':['corechart']});
//google.charts.setOnLoadCallback(drawChart);

/**
 * Reads a csv file, assign values (weights) to respective latitude and longitude,
 * and stores it as LatLng object to the points array.
 * @return Array of points from day 0 to day 9
*/
async function getData() {
  var days_array = [];
  var csv_files = ['../past_data/p9.csv', '../past_data/p8.csv', '../past_data/p7.csv', '../past_data/p6.csv', 
                    '../past_data/p5.csv', '../past_data/p4.csv', '../past_data/p3.csv', '../past_data/p2.csv',
                    '../past_data/p1.csv', '../forecast/f0.csv', '../forecast/f1.csv', '../forecast/f2.csv',
                    '../forecast/f3.csv', '../forecast/f4.csv','../forecast/f5.csv','../forecast/f6.csv',
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
 * Initializes the google map
 */
function initMap() {
  map = new google.maps.Map(document.getElementById("map"), {
      zoom: 2.3,
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
      streetViewControl: false,
      mapTypeControl: true,
      gestureHandling: "cooperative",
      minZoom: 2.2,
  });

  /** FOR LEGEND
  const legend = document.getElementById("legend");
  const div = document.createElement("div");
  div.innerHTML = 'legend here'
  // example'<img src="' + icon + '"> ';
  legend.appendChild(div);

  map.controls[google.maps.ControlPosition.LEFT_BOTTOM].push(legend);
   */
  initHeatMap(9);
}

/**
 * Computes the total number of cases in a day
 * @param selected_day, selected day [day 0 - 9]
 * @param total_number
 */
function getTotalNumber(selected_day, total_number) {
  var total_number_of_cases = 0;

  for (var i = 0; i < selected_day.length; i++) {
    total_number_of_cases += parseFloat(selected_day[i]['weight']);
  }
  total_number.innerHTML = total_number_of_cases;
}

function getDataPerLocation(days_array) {

  var location;
  var weight;
  var locations = [];

  //Iterate through day 0 to 9
  for (var i = 0; i < days_array.length; i++) {

    //Iterate through every location
    for (var j = 0; j < days_array[i].length; j++) {
      location = [days_array[i][j]['location'].lat() + " " + days_array[i][j]['location'].lng()];
      weight = days_array[i][j]['weight'];
      var index = -1;
      var k;

      //Check if location is in locations array
      for (k = 0; k < locations.length; k++) {
        if (location == locations[k][0]) {
          index = k;
        }
      }

      if (index === -1) {
        locations.push(location);
        index = k;
      }

      locations[index].push(weight);
    }
  }
  return locations;
}

/**
 * Adds markers and event listeners to locations with cases
 * @param selected_day, selected day [day 0 -9]
 */
function addMarker(days_array, day) {
  var selected_day = days_array[day]

  //Array is arranged such that every index is [location weight1, weight2 ...]
  var location_data = getDataPerLocation(days_array);
  var infoWindow;
  var marker;
  var marker_data;

  //Add markers for every location for the selected day
  for (var i = 0; i < selected_day.length; i++) {
    marker = new google.maps.Marker({
      position: selected_day[i]['location'],
      map,
      icon:  {
        path: google.maps.SymbolPath.CIRCLE,
        scale: 10,
        strokeOpacity: 0,
      },
    });
    
    infoWindow  = new google.maps.InfoWindow();

    var location_string = selected_day[i]['location'].lat() + " " + selected_day[i]['location'].lng();

    //For every location in the location data, check if the marker location is the same
    for (var k = 0; k < location_data.length; k++) {

      //If the marker location string is equal to the location data 
      if (location_string === location_data[k][0]) {
        marker_data = [];

        //Generate marker data
        for (var l = 1; l <= days_array.length; l++) {
          var day = new Date((l - 10) * MILLISECONDS_IN_A_DAY + (DATE_TODAY.getTime()))
          var date = MONTHS[day.getMonth()] + " " + day.getDate() + ", " + day.getFullYear()
          marker_data.push([date, parseInt(location_data[k][l])]);
        }
      }
    }

    marker.data = marker_data;

    google.maps.event.addListener(marker, 'click', function() {
      drawLineChart(this, infoWindow);
    });
  }
}

function drawLineChart(marker, infoWindow) {
  
  // Create the data table.
  var data = new google.visualization.DataTable();
  data.addColumn('string', 'Day');
  data.addColumn('number', 'Number of Cases');

  data.addRows(marker.data);

  var options = {title:'Location: ' + marker.getPosition().toString(),
                  width:700,
                  height:220,
                  legend:'none',
                  colors: ['#b36a5e'],
                  hAxis: {
                    showTextEvery: 2,}
                };
                 
  var node = document.createElement('div'),
            infoWindow,
            chart = new google.visualization.LineChart(node);
      
  infoWindow.setContent(node);
  infoWindow.open(marker.getMap(),marker);
  google.maps.event.addListener(infoWindow, 'domready', function () {
    chart.draw(data, options);
  });
}

/**
 * Creates the heatmap layer
 *  @param day, day 0 - 9; Day used in getting the heatmap points
 */
function initHeatMap(day) {
  (async () => {
    
    //Fetch all the data
    var days_array = await getData();
    
    //Initialize the heatmap layer 
    heatmap = new google.maps.visualization.HeatmapLayer({
      data: days_array[day], 
      opacity: 0.5,
      map: map,
      radius: 3,
      maxIntensity: 20000,
      dissipating: false,
    });

    var slider = document.getElementById("dateSlider");
    var date_output = document.getElementById("selectedDate");
    var total_number = document.getElementById("totalNumber");

    //Display default date
    date_output.innerHTML = MONTHS[DATE_TODAY.getMonth()] + " " + DATE_TODAY.getDate() + ", " + DATE_TODAY.getFullYear();

    //Data for the selected day
    var selected_day = days_array[0];

    //Invoke function to display the total number of cases for default day
    getTotalNumber(selected_day, total_number);
    
    //Invoke function to add markers to the heatmap
    addMarker(days_array, 0);

    //Event when slider input changes
    slider.oninput = function() {

        var selected_date = new Date((this.value - 9) * MILLISECONDS_IN_A_DAY + (DATE_TODAY.getTime()));

        //Update date
        date_output.innerHTML = MONTHS[selected_date.getMonth()] + " " + selected_date.getDate() + ", " + selected_date.getFullYear();

        selected_day = days_array[this.value];
        
        //Update heatmap date
        heatmap.setData(selected_day);

        //Update total number of cases
        getTotalNumber(selected_day, total_number);
        
        //Update markers
        addMarker(days_array, this.value);
    }
  })(); 
}