# Beyond Encore

Beyond Encore helps concertgoers turn a live show into a fuller Hong Kong experience. Enter an artist or concert name, choose a venue, and discover nearby places to eat, relax, shop and explore. With personalised trip themes, budget and opening-hour filters, and optional AI music matching, the app helps you plan activities around your concert and save a suggested itinerary for your visit.

## Features

- Plan around four venues: Hong Kong Coliseum, West Kowloon Cultural District, Hong Kong Cultural Centre and Kai Tak Sports Park.
- Explore 2,992 POIs across restaurants, attractions, bars, cafés, nightclubs and shopping malls.
- Personalise your trip theme, solo or group travel, budget, opening-hour preferences and number of stops.
- With AI enabled, interpret concert inputs and rank POIs using music-related similarity, distance, ratings and review counts. The suggested itinerary is generated separately using trip themes, filters, distance and ratings.
- View your stops on a map and route timeline, look up places in Amap, and download the itinerary as CSV.
- Switch between Chinese and English while retaining your input and generated plan. Place names and addresses remain in the source language.

## How to use

1. Select Chinese or English from the sidebar.
2. Choose your trip theme, travel group, budget and opening-hour filters. Times use Hong Kong local time.
3. Enter an artist or concert name, select a venue, and choose the number of stops.
4. Click **Find my encore** / **开启我的安可之旅**.
5. Review the places, map and timeline. Open a place in Amap or download your itinerary for later.

Without an API key, the app offers itinerary previews based on trip preferences, distance and ratings. Configure a compatible API service to enable concert interpretation and music-related POI matching. A plan may contain fewer stops than requested when the available data does not meet your filters.

## Run locally

Use Python **3.13**. From the project directory:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The project includes its POI database, precomputed vectors and image assets. No separate database service is required.

## Configure AI matching

Set these environment variables, or add them to your application's **Secrets** settings in Streamlit Community Cloud:

```toml
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
```

For another compatible provider, use that provider's key and base URL. The service must support `gpt-4o-mini` and `text-embedding-3-small`. Query vectors must be compatible with the supplied POI vectors.

AI requests use the configured API account's quota, including requests made by other visitors. Do not commit real credentials, `.env` or `.streamlit/secrets.toml` to the repository.

## Data and recommendation notes

- POIs are stored in `data/poi.sqlite3`; precomputed vectors are stored in `data/poi_embeddings.npz`. Update both together when changing POI data.
- Ratings, prices and opening hours reflect the supplied dataset. Check current details before visiting. Places without numeric price levels may be excluded by budget filters.
- Map connections and timeline distances are **straight-line estimates**, not walking routes. Opening-hour filters do not account for carry-over from the previous weekday after midnight.
- Fan-related tags are suggested themes, not confirmation of celebrity visits or organised events. Category images are illustrations rather than verified photographs of named places.
- The basemap is provided by the Hong Kong Lands Department through the [CSDI Map API](https://p1static.csdi.gov.hk/csdi-webpage/apidoc/TopographicMapAPI). Availability depends on the visitor's network.
