var slider = document.getElementById("dateSlider");
var output = document.getElementById("selectedDate");
output.innerHTML = new Date(); // Display the default slider value
var startIndex = slider.value;

slider.oninput = function() {
    var d = new Date();
    var selected = new Date((this.value) * (1000*3600*24) + (d.getTime()));
    output.innerHTML = selected; //format to Month Day, Year
    initHeatMap(this.value);
}

var pointsArray = [];   //Array that stores day0 - day9 arrays

/**
 * Function that reads a csv file, assign values (weights) to respective latitude and longitude,
 * and stores it as LatLng object to the points array.
 */
function getPoints(data, points) {
    var lat = -90;
    for (var i = 0; i < data.length; i++) {
        var long = -180;
        for (var j = 0; j < data[0].length; j++) {
            if (data[i][j] != 0) {
                points.push({location: new google.maps.LatLng( lat, long), weight: data[i][j]});
            }
            long++;
        }
        lat++;
    }
    pointsArray.push(points);
}

/**
 * Initializes the google maps API and the heatmap layer.
 * Only f0 and f1 are being read right now
 */
function initMap() {
    var points0 = [];
    var points1 = [];

    var data0;
	$.ajax({
	  type: "GET",  
	  url: "../forecast/f0.csv",
	  dataType: "text",       
	  success: function(response)  
	  {
		data0 = $.csv.toArrays(response);
		getPoints(data0, points0);
	  }   
	});

    var data1;
        $.ajax({
        type: "GET",  
        url: "../forecast/f1.csv",
        dataType: "text",       
        success: function(response)  
        {
            data1 = $.csv.toArrays(response);
            getPoints(data1, points1);
        }   
        });

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

    heatmap = new google.maps.visualization.HeatmapLayer({
        data: points0, //Just displays f0 array. Ideally, it should call pointsArray[day]
        opacity: 0.5,
        map: map,
        radius: 3,
        maxIntensity: 10000,
        dissipating: false,
    });
}

