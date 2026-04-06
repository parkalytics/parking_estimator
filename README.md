# Parking Stall Estimator

A two-mode parking stall estimation tool:
- **Map Query**: draws a radius on an OSM map, queries the Overpass API for road segments, and applies the parallel parking stall formula
- **Image Analysis**: accepts a satellite photo (Google Earth, Mapbox, etc.) and uses OpenCV Hough line detection to count stall markings

## Setup

```bash
cd parking_estimator
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --port 8000
```

Then open: http://localhost:8000

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | /health | Server health check |
| POST | /estimate/on-street | On-street stall estimate from coordinates |
| POST | /estimate/image | Off-street stall estimate from image upload |

### POST /estimate/on-street

```json
{
  "lat": 43.6532,
  "lon": -79.3832,
  "radius_m": 500,
  "parking_type": "parallel"
}
```

### POST /estimate/image

Multipart form upload with field `file` = image file (PNG/JPG, max 20MB).

## Stall formula (on-street parallel)

```
effective_length = road_length_m - 2 × 8m buffer
stalls_one_side  = floor(effective_length / 6.7)
total_stalls     = stalls_one_side × 2  (both sides, unless one-way)
```

## Next steps

- [ ] Swap Mapbox satellite tiles for image analysis (add MAPBOX_TOKEN to .env)
- [ ] Add angled/perpendicular parking type support
- [ ] Export results to CSV
- [ ] Add YOLO model for better stall detection accuracy
