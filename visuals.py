"""Local visual assets and a CDN-free Leaflet map using LandsD public tiles."""
from pathlib import Path
from html import escape
import base64,json
import streamlit as st
import streamlit.components.v1 as components

ASSETS=Path(__file__).resolve().parent/'assets'

@st.cache_data
def uri(name):
    p=ASSETS/name
    if not p.exists():return ''
    kind='image/png' if p.suffix=='.png' else 'image/jpeg'
    return f'data:{kind};base64,'+base64.b64encode(p.read_bytes()).decode()

def picture(category,zh=False,hero=False):
    atlas=uri('hong-kong-atlas.png')
    if not atlas:return ''
    positions={'restaurant':'0% 23%','attraction':'50% 23%','bar':'100% 23%','cafe':'0% 77%','night_club':'50% 77%','shopping_mall':'100% 77%','concert':'50% 77%'}
    return f'<div class="poi-photo {"concert-photo" if hero else ""}" style="background-image:url({atlas});background-size:300% auto;background-position:{positions.get(category,"50% 23%")}"></div>'

def navigation(spot):
    from urllib.parse import urlencode
    return 'https://uri.amap.com/search?'+urlencode({'keyword':str(spot.get('name','')),'city':'香港','view':'map','src':'BeyondEncore'})

def route_map(coords,venue,itinerary,zh=False):
    css=(ASSETS/'leaflet.css').read_text(encoding='utf8')
    js=(ASSETS/'leaflet.js').read_text(encoding='utf8')
    points=[{'lat':coords[0],'lng':coords[1],'name':venue,'number':'♪'}]+[{'lat':float(s['lat']),'lng':float(s['lng']),'name':str(s['name']),'number':str(i+1)} for i,s in enumerate(itinerary)]
    payload=json.dumps(points,ensure_ascii=True).replace('<','\\u003c')
    logo=uri('landsd.png')
    notice='底图暂时无法加载，地点与连线仍可查看。可使用卡片中的高德地图入口。' if zh else 'Basemap unavailable. Stops and connections remain visible; use the Amap links on the cards.'
    html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>'''+css+'''
    body{margin:0;font-family:system-ui}#map{height:460px;border-radius:20px;background:#e8eeee}
    .pin{background:#cda3a5;color:white;border:3px solid white;border-radius:50%;text-align:center;font-weight:700;line-height:30px;box-shadow:0 3px 12px #0002}
    .venue{background:#8faabb}.brand{background:white;padding:5px;border-radius:8px}.brand img{width:40px;height:40px;object-fit:contain}
    #notice{position:absolute;top:14px;left:65px;right:20px;z-index:1000;background:#fff9ed;padding:12px;border-radius:10px;font-size:13px;display:none}
    </style></head><body><div id="map"></div><div id="notice">'''+escape(notice)+'''</div><script>'''+js+'''</script><script>
    const points='''+payload+''';
    const map=L.map('map',{minZoom:10,maxZoom:20}).setView([points[0].lat,points[0].lng],14);
    const tiles=L.tileLayer('https://mapapi.geodata.gov.hk/gs/api/v1.0.0/xyz/basemap/WGS84/{z}/{x}/{y}.png',{minZoom:10,maxZoom:20,attribution:'Map from Lands Department © HKSAR Government | CSDI Portal'}).addTo(map);
    let success=0;tiles.on('tileload',()=>{success++;document.getElementById('notice').style.display='none'});
    tiles.on('tileerror',()=>{if(!success)document.getElementById('notice').style.display='block'});
    setTimeout(()=>{if(!success)document.getElementById('notice').style.display='block'},12000);
    const locations=[];points.forEach((p,i)=>{const loc=[p.lat,p.lng];locations.push(loc);const label=document.createElement('span');label.textContent=p.name;L.marker(loc,{icon:L.divIcon({className:'pin '+(i===0?'venue':''),html:p.number,iconSize:[36,36],iconAnchor:[18,18]})}).bindTooltip(label).addTo(map)});
    L.polyline(locations,{color:'#b98e94',weight:4,dashArray:'8 8'}).addTo(map);
    map.fitBounds(locations,{padding:[48,48],maxZoom:16});
    const brand=L.control({position:'bottomleft'});brand.onAdd=()=>{const div=L.DomUtil.create('div','brand');const img=document.createElement('img');img.src='''+json.dumps(logo)+''';img.alt='Lands Department';div.appendChild(img);return div};brand.addTo(map);
    </script></body></html>'''
    components.html(html,height=470,scrolling=False)
