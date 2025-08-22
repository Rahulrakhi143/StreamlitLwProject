
let mediaRecorder;
let recordedChunks = [];

navigator.mediaDevices.getUserMedia({ video: true, audio: true })
  .then(stream => {
    document.getElementById('video').srcObject = stream;
    document.getElementById('preview').srcObject = stream;

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      const a = document.getElementById('downloadLink');
      a.href = url;
      a.download = "recorded_video.webm";
      a.style.display = "inline-block";
      a.textContent = "Download Video";
      recordedChunks = [];
    };
  }).catch(err => alert("Webcam error: " + err));

function captureImage() {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const link = document.createElement('a');
  link.download = 'captured_image.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
}

function startRecording() {
  if (mediaRecorder) mediaRecorder.start();
}

function stopRecording() {
  if (mediaRecorder) mediaRecorder.stop();
}

function sendToWhatsApp() {
  const phone = "918079018281";
  const message = "Hi from Firefox-compatible app!";
  window.open(`https://wa.me/${phone}?text=${encodeURIComponent(message)}`);
}

function sendEmail() {
  const email = "example@example.com";
  const subject = encodeURIComponent("Media App Message");
  const body = encodeURIComponent("Hi from my media app!");
  window.location.href = `mailto:${email}?subject=${subject}&body=${body}`;
}

const phoneInput = document.getElementById("phone");
const msgInput = document.getElementById("msg");
const smsLink = document.getElementById("smsLink");

function updateSmsLink() {
  const phone = phoneInput.value.trim();
  const msg = encodeURIComponent(msgInput.value.trim());
  smsLink.href = `sms:${phone}?body=${msg}`;
}
phoneInput.addEventListener("input", updateSmsLink);
msgInput.addEventListener("input", updateSmsLink);
updateSmsLink();

function getTextLocation() {
  const output = document.getElementById("locationOutput");
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      pos => output.innerText = `Latitude: ${pos.coords.latitude.toFixed(6)}\nLongitude: ${pos.coords.longitude.toFixed(6)}`,
      err => output.innerText = "Error: " + err.message
    );
  } else {
    output.innerText = "Geolocation not supported.";
  }
}

let map, marker;
function showMap() {
  if (navigator.geolocation) {
    navigator.geolocation.watchPosition(pos => {
      const lat = pos.coords.latitude;
      const lon = pos.coords.longitude;
      if (!map) {
        map = L.map("map").setView([lat, lon], 15);
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png").addTo(map);
        marker = L.marker([lat, lon]).addTo(map).bindPopup("You are here").openPopup();
      } else {
        marker.setLatLng([lat, lon]);
        map.setView([lat, lon]);
      }
    }, err => alert("Map location error: " + err.message));
  }
}

let leafletRouteMap, leafletRouteControl;
function drawLeafletRoute() {
  const dest = document.getElementById("leaflet-destination").value;
  if (!dest) return alert("Enter a destination");
  fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(dest)}`)
    .then(res => res.json())
    .then(locations => {
      if (!locations.length) return alert("Destination not found.");
      const destCoords = [parseFloat(locations[0].lat), parseFloat(locations[0].lon)];
      navigator.geolocation.getCurrentPosition(pos => {
        const userCoords = [pos.coords.latitude, pos.coords.longitude];
        if (!leafletRouteMap) {
          leafletRouteMap = L.map("leaflet-route-map").setView(userCoords, 13);
          L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png").addTo(leafletRouteMap);
        }
        if (leafletRouteControl) leafletRouteMap.removeControl(leafletRouteControl);
        leafletRouteControl = L.Routing.control({
          waypoints: [L.latLng(...userCoords), L.latLng(...destCoords)],
          routeWhileDragging: false
        }).addTo(leafletRouteMap);
      });
    });
}

let storeMap; // this is used only for showing nearby grocery stores
let storeMarkers = [];
function findNearbyStores() {
  if (!navigator.geolocation) {
    alert("Geolocation not supported in this browser.");
    return;
  }

  navigator.geolocation.getCurrentPosition(
    (position) => {
      const lat = position.coords.latitude;
      const lon = position.coords.longitude;

      if (!storeMap) {
        storeMap = L.map("map").setView([lat, lon], 15);
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          attribution: '&copy; OpenStreetMap contributors'
        }).addTo(storeMap);
      } else {
        storeMap.setView([lat, lon], 15);
      }

      // Clear previous markers
      storeMarkers.forEach(m => storeMap.removeLayer(m));
      storeMarkers = [];

      // Add user marker
      const userMarker = L.marker([lat, lon]).addTo(storeMap).bindPopup("You are here").openPopup();
      storeMarkers.push(userMarker);

      fetchOverpassGroceryStores(lat, lon);
    },
    (error) => {
      alert("Error getting location: " + error.message);
    }
  );
}

function fetchOverpassGroceryStores(lat, lon) {
  const radius = 1000;
  const query = `
    [out:json];
    (
      node["shop"="supermarket"](around:${radius},${lat},${lon});
      node["shop"="grocery"](around:${radius},${lat},${lon});
    );
    out body;
  `;

  const url = `https://overpass-api.de/api/interpreter?data=${encodeURIComponent(query)}`;

  fetch(url)
    .then(res => res.json())
    .then(data => {
      const list = document.getElementById("store-list");
      list.innerHTML = "";

      if (!data.elements.length) {
        list.innerHTML = "<li>No grocery stores found nearby.</li>";
        return;
      }

      data.elements.forEach(store => {
        const name = store.tags.name || "Unnamed Store";
        const latLng = [store.lat, store.lon];

        const marker = L.marker(latLng).addTo(storeMap).bindPopup(name);
        storeMarkers.push(marker);

        const li = document.createElement("li");
        li.textContent = name;
        list.appendChild(li);
      });
    })
    .catch(err => {
      alert("Error fetching stores: " + err.message);
    });
}
 const CLIENT_ID = 'YOUR_CLIENT_ID.apps.googleusercontent.com'; // Replace this
    const SCOPES = 'https://www.googleapis.com/auth/gmail.readonly';

    let tokenClient;

    function authenticate() {
      gapi.load('client:auth2', async () => {
        await gapi.client.init({
          clientId: CLIENT_ID,
          scope: SCOPES
        });

        const GoogleAuth = gapi.auth2.getAuthInstance();
        GoogleAuth.signIn().then(() => {
          console.log("Signed in!");
          loadGmailApi();
        });
      });
    }

    function loadGmailApi() {
      gapi.client.load('gmail', 'v1', () => {
        gapi.client.gmail.users.messages.list({
          userId: 'me',
          maxResults: 1
        }).then(resp => {
          const messageId = resp.result.messages[0].id;

          gapi.client.gmail.users.messages.get({
            userId: 'me',
            id: messageId
          }).then(msg => {
            const headers = msg.result.payload.headers;
            const subject = headers.find(h => h.name === 'Subject')?.value || '(No subject)';
            const from = headers.find(h => h.name === 'From')?.value || '(Unknown sender)';
            const date = headers.find(h => h.name === 'Date')?.value || '(Unknown date)';
            const snippet = msg.result.snippet;

            document.getElementById("output").innerHTML = `
              <p><strong>From:</strong> ${from}</p>
              <p><strong>Subject:</strong> ${subject}</p>
              <p><strong>Date:</strong> ${date}</p>
              <p><strong>Snippet:</strong> ${snippet}</p>
            `;
          });
        });
      });
    }

    const timers = {};
const viewTimes = {};
const viewCounts = {};
const clickCounts = {};

// View tracking with IntersectionObserver
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    const id = entry.target.dataset.id;

    if (entry.isIntersecting) {
      // Count view
      viewCounts[id] = (viewCounts[id] || 0) + 1;
      document.getElementById(`views-${id}`).textContent = `Views: ${viewCounts[id]}`;

      // Start view time timer
      if (!timers[id]) {
        timers[id] = setInterval(() => {
          viewTimes[id] = (viewTimes[id] || 0) + 1;
          document.getElementById(`time-${id}`).textContent = `View time: ${viewTimes[id]}s`;
        }, 1000);
      }
    } else {
      // Stop view time timer
      if (timers[id]) {
        clearInterval(timers[id]);
        timers[id] = null;
      }
    }
  });
}, {
  threshold: 0.6 // At least 60% of the product visible to count as view
});

// Observe all product cards
document.querySelectorAll('.product-card').forEach(card => observer.observe(card));

// Click tracking
function handleClick(productId) {
  clickCounts[productId] = (clickCounts[productId] || 0) + 1;
  document.getElementById(`click-${productId}`).textContent = `Clicks: ${clickCounts[productId]}`;
}
   
  let userData = {};

async function getIPInfo() {
  try {
    const response = await fetch("https://ipapi.co/json/");
    const data = await response.json();
    userData = data; // Save globally for Excel

    document.getElementById("ipInfo").innerHTML = `
      <h2>🛰️ Your current IP address is <strong>${data.ip}</strong></h2>
      <p>📍 Location: ${data.city}, ${data.region}, ${data.country_name}</p>
    `;
  } catch (error) {
    document.getElementById("ipInfo").innerHTML = `<h2>❌ Failed to fetch location info</h2>`;
    console.error("Error fetching IP info:", error);
  }
}

getIPInfo();

    function downloadExcel() {
  const report = [];

  Object.keys(viewCounts).forEach(id => {
    report.push({
      "Product ID": id,
      "Views": viewCounts[id] || 0,
      "View Time (s)": viewTimes[id] || 0,
      "Clicks": clickCounts[id] || 0,
      "Timestamp": new Date().toLocaleString(),
      "IP Address": userData.ip || "N/A",
      "City": userData.city || "N/A",
      "State": userData.region || "N/A",
      "Country": userData.country || "N/A"
    });
  });

  const worksheet = XLSX.utils.json_to_sheet(report);
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, worksheet, "ProductReport");

  XLSX.writeFile(workbook, "product_report.xlsx");
}

