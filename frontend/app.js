var map, heatmap;
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const DATE_TODAY = new Date();
const MILLISECONDS_IN_A_DAY = 86400000;
google.charts.load('46', {'packages':['corechart']});
var slider_value;
var slider_selected_date;

/**
 * Reads csv files for past, current and future active cases,
 * assign values (weights) to respective latitude and longitude,
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
 * Computes the total number of cases in a day and stores them into an
 * array of days with their respective total number of cases
 * @param days_array, array of number of cases per affected location per day
 */
function getTotalNumberArray(days_array) {
  var total_number_array = [];

  for (var i = 0; i < days_array.length; i++) {
    var total_number_of_cases = 0;

    for (var j = 0; j < days_array[i].length; j++) {
      total_number_of_cases += parseFloat(days_array[i][j]['weight']);
    }

    total_number_array.push(total_number_of_cases);
  }

  return total_number_array;
}

/**
 * Stores number of cases per day into their respective location
 * @param days_array, array of days with information about locations
 *      and number of cases
 * @returns 2d array of locations and their respective number of cases
 *      in the span of 19 days
 */
function getDataPerLocation(days_array) {
  var location;
  var weight;
  var locations = [];

  //Iterate through day 0 to 18
  for (var i = 0; i < days_array.length; i++) {

    //Iterate through every location
    for (var j = 0; j < days_array[i].length; j++) {
      location = [days_array[i][j]['location'].lat() + " " + days_array[i][j]['location'].lng()];
      weight = days_array[i][j]['weight'];
      var index = -1;
      var k;

      //Check if location is in locations array
      for (k = 0; k < locations.length; k++) {

        //Get index of the location
        if (location == locations[k][0]) {
          index = k;
        }
      }

      //If the index is -1, it is not in the locations array
      if (index === -1) {

        //Push the location into the locations array
        locations.push(location);
        index = k;

        //If this is not the first day, add zeroes to number of cases to the preceding days
        if (i != 0) {
          for (var l = 0; l < i; l++) {
            locations[index].push(0);
          }
        } 
      }

      //Push the weight 
      locations[index].push(weight);
    }
  }

  return locations;
}

/**
 * Adds markers and event listeners to locations with cases
 * @param days_array, array of days with information about locations
 *      and number of cases
 * @param day, selected day [day 0 - 18]
 * @param location_data, 2d array of locations and their respective number of cases
 *      in the span of 19 days
 */
function addMarker(days_array, day, location_data) {
  var selected_day = days_array[day]

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
          var date = (day.getMonth() + 1) + "/" + day.getDate();
          marker_data.push([date, parseInt(location_data[k][l])]);
        }
      }
    }

    marker.data = marker_data;

    //Listener if a marker is clicked
    google.maps.event.addListener(marker, 'click', function() {
      drawLineChart(this, infoWindow);
    });
  }
}

/**
 * Draw line chart inside an info window
 * @param marker
 * @param infoWindow 
 */
function drawLineChart(marker, infoWindow) {
  
  // Create the data table.
  var data = new google.visualization.DataTable();
  data.addColumn('string', 'Day');
  data.addColumn('number', 'Number of Cases');

  data.addRows(marker.data);

  var options = {title:'Location: ' + marker.getPosition().toString() + "\n" +
                      slider_selected_date + ": " + marker.data[slider_value][1],
                  width:400,
                  height:200,
                  legend:'none',
                  colors: ['#b36a5e'],
                  pointsVisible: true,
                  hAxis: {
                    title: "Date",
                    showTextEvery: 1,
                    slantedText: true,
                  },
                  vAxis: {
                    title: "Number of Cases",
                  },
                };
                 
  var node = document.createElement('div'),
            infoWindow,
            chart = new google.visualization.LineChart(node);
      
  infoWindow.setContent(node);
  infoWindow.setOptions({maxWidth:400})
  infoWindow.open(marker.getMap(),marker);
  google.maps.event.addListener(infoWindow, 'domready', function () {
    chart.draw(data, options);
  });
}

/**
 * Creates the heatmap layer
 *  @param day, day 0 - 18; Day used in getting the heatmap points
 */
function initHeatMap(day) {
  (async () => {
    
    //Fetch all the data
    var days_array = await getData();
    var location_data = getDataPerLocation(days_array);
    var total_number_array = getTotalNumberArray(days_array);
    
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
    var total_number_of_cases = document.getElementById("totalNumber");

    //Display default date
    slider_value = 9;
    slider_selected_date = MONTHS[DATE_TODAY.getMonth()] + " " + DATE_TODAY.getDate() + ", " + DATE_TODAY.getFullYear()
    date_output.innerHTML = slider_selected_date;

    //Data for the selected day
    var selected_day = days_array[0];

    //Invoke function to display the total number of cases for default day
    total_number_of_cases.innerHTML = total_number_array[9];
    
    //Invoke function to add markers to the heatmap
    addMarker(days_array, 0, location_data);

    //Event when slider input changes
    slider.oninput = function() {
        slider_value = this.value;
        var selected_date = new Date((slider_value - 9) * MILLISECONDS_IN_A_DAY + (DATE_TODAY.getTime()));

        //Update date
        slider_selected_date = MONTHS[selected_date.getMonth()] + " " + selected_date.getDate() + ", " + selected_date.getFullYear()
        date_output.innerHTML = slider_selected_date;

        //Update heatmap date
        selected_day = days_array[slider_value];
        heatmap.setData(selected_day);

        //Update total number of cases
        total_number_of_cases.innerHTML = total_number_array[slider_value];
        
    }
  })(); 
}

/**
 * Initializes the google map
 */
 function initMap() {
  map = new google.maps.Map(document.getElementById("map"), {
      zoom: 2.3,
      center: { lat: 28, lng: 0 },
      streetViewControl: false,
      mapTypeControl: true,
      gestureHandling: "greedy",
      minZoom: 2.2,
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
  
  initHeatMap(9);
}
