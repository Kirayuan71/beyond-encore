# Beyond Encore

## Updated interface and map

The interface uses a cream, dusty rose and mist blue palette, glass cards and the supplied logo. Category and itinerary cards include locally bundled illustrative AI images; these are not photographs of the named POIs. Chinese/English switching preserves input and generated plans.

The route map uses Hong Kong Lands Department's public WGS84 XYZ basemap, with the required logo and attribution. Leaflet JavaScript and CSS are bundled locally, avoiding browser CDN dependencies. A street tile was successfully retrieved and visually inspected on the current network. Actual availability varies by visitor network; failed tile loads show a notice while markers remain visible. POI links open Amap search by name in Hong Kong. No Amap developer key is needed for these search links. Route connections and timeline distances are straight-line distances, not walking directions. Browser automation was unavailable, so full browser and mobile visual verification remains outstanding.

Map service documentation: https://p1static.csdi.gov.hk/csdi-webpage/apidoc/TopographicMapAPI

English Streamlit concert trip planner with bundled Hong Kong POI SQLite database.

## Run locally

Use Python 3.13 to match the validated environment. From this directory:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Without an API key, the app offers distance/rating itinerary previews. Music matching requires a new key configured through environment variables.

## Docker deployment

```sh
docker build -t beyond-encore .
docker run -p 8501:8501 --env-file .env beyond-encore
```

Create `.env` locally, excluded from Git and the Docker build:

```dotenv
OPENAI_API_KEY=your-new-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

The original app used another API provider. If using that provider, set its compatible base URL and a new credential; verify it supports `gpt-4o-mini` and `text-embedding-3-small`.

For any Docker host, use the included Dockerfile, container port 8501, and health endpoint `/_stcore/health`. Enable HTTPS through the host's reverse proxy. A server with WebSocket support is required for Streamlit.

## POI storage

`data/poi.sqlite3` includes six indexed tables: restaurant, attraction, bar, cafe, night_club, shopping_mall. The app reads this database in read-only mode. The release includes `data/poi_embeddings.npz`, read without pickle support. CSV sources and the old pickle cache are omitted from the release. No external database account is required for this read-only dataset. Update the database and embedding cache together when changing POIs.

Optional environment variables: `POI_DATA_DIR` and `POI_DATABASE_PATH`.

## Validation and limits

The original files are unchanged. Startup and a no-key itinerary were checked with Streamlit AppTest. Live API calls, browser visual layout, Docker builds and public deployment require separate verification. Venue-to-stop lines indicate straight-line distances, not walking routes. Some categories lack price levels and are excluded when a numeric budget is required. Hours follow supplied weekday records; after-midnight carry-over from the previous weekday is not modelled. Fan tags are thematic suggestions, not verified celebrity visits or organised activities.

Revoke the API key embedded in the original code before publishing. The deployment project contains no copied API credential. Never commit `.env` or Streamlit secrets.
