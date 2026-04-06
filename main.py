import asyncio
import base64
import math
import io
from typing import Optional

import httpx
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Parking Stall Estimator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Models ───────────────────────────────────────────────────────────────────

class EstimateRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    radius_m: int = Field(default=500, ge=50, le=5000)
    parking_type: str = Field(default="parallel")

class StreetSegment(BaseModel):
    name: str
    length_m: float
    stalls: int

class EstimateResponse(BaseModel):
    on_street_stalls: int
    streets_analyzed: int
    total_road_length_m: float
    confidence: str
    segments: list[StreetSegment]

class ImageAnalysisResponse(BaseModel):
    detected_stalls: int
    line_count: int
    lots_detected: int
    confidence: str
    notes: str

# ─── On-Street Estimator ──────────────────────────────────────────────────────

STALL_WIDTH_PARALLEL = 6.7   # metres, standard parallel stall
BUFFER_PER_END = 8.0          # metres reserved at each road end (driveways, intersections)

def haversine_length(geom: list) -> float:
    total = 0.0
    for i in range(len(geom) - 1):
        a, b = geom[i], geom[i + 1]
        R = 6_371_000
        lat1, lat2 = math.radians(a["lat"]), math.radians(b["lat"])
        dlat = lat2 - lat1
        dlon = math.radians(b["lon"] - a["lon"])
        h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        total += 2 * R * math.asin(math.sqrt(h))
    return total

async def estimate_on_street(lat: float, lon: float, radius_m: int) -> dict:
    delta = radius_m / 111_000
    bbox = f"{lat - delta},{lon - delta},{lat + delta},{lon + delta}"

    query = f"""
[out:json][timeout:45];
(
  way[highway~"residential|tertiary|secondary|primary|living_street|unclassified"]({bbox});
);
out geom;
"""

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=50,
        )
        resp.raise_for_status()
        data = resp.json()

    total_stalls = 0
    total_length = 0.0
    segments = []

    for way in data.get("elements", []):
        if way.get("type") != "way":
            continue
        geom = way.get("geometry", [])
        if len(geom) < 2:
            continue

        tags = way.get("tags", {})
        length_m = haversine_length(geom)
        if length_m < 20:
            continue

        # Skip if explicitly no parking
        parking_lane = tags.get("parking:lane", tags.get("parking:lane:both", ""))
        if parking_lane in ("no", "none"):
            continue

        effective_length = max(0, length_m - 2 * BUFFER_PER_END)
        stalls_one_side = int(effective_length / STALL_WIDTH_PARALLEL)

        # Both sides by default; halve if one-way or tagged one side only
        oneway = tags.get("oneway", "no")
        parking_both = tags.get("parking:lane:both", "")
        if oneway == "yes" or parking_lane in ("left", "right"):
            stalls = stalls_one_side
        else:
            stalls = stalls_one_side * 2

        name = tags.get("name", tags.get("ref", f"Unnamed way {way.get('id', '')}"))
        total_stalls += stalls
        total_length += length_m
        segments.append(StreetSegment(name=name, length_m=round(length_m, 1), stalls=stalls))

    segments.sort(key=lambda s: s.stalls, reverse=True)

    confidence = "high" if len(segments) > 10 else "medium" if len(segments) > 3 else "low"
    return {
        "stalls": total_stalls,
        "streets": len(segments),
        "total_length": total_length,
        "confidence": confidence,
        "segments": segments,
    }

# ─── Image / Off-Street Estimator ─────────────────────────────────────────────

def detect_stalls_from_image(img_bytes: bytes) -> dict:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance contrast for satellite images
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    edges = cv2.Canny(gray, threshold1=40, threshold2=120)

    # Dilate to connect nearby edge fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=int(min(h, w) * 0.03),
        maxLineGap=15,
    )

    if lines is None:
        return {"stalls": 0, "lines": 0, "lots": 0, "confidence": "low",
                "notes": "No line structures detected in image."}

    # Cluster lines by angle into orientations
    angle_groups = {"vertical": [], "horizontal": [], "diagonal": []}
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))
        if angle > 70:
            angle_groups["vertical"].append(line)
        elif angle < 20:
            angle_groups["horizontal"].append(line)
        else:
            angle_groups["diagonal"].append(line)

    dominant = max(angle_groups, key=lambda k: len(angle_groups[k]))
    stall_lines = angle_groups[dominant]
    n_lines = len(stall_lines)

    # N parallel lines = N-1 stalls per cluster; estimate lot count from density
    stalls = max(0, n_lines - 1)
    lots = max(1, n_lines // 15) if n_lines > 5 else 0

    if n_lines > 50:
        confidence = "high"
        notes = f"Strong line structures detected ({dominant} orientation dominant)."
    elif n_lines > 15:
        confidence = "medium"
        notes = f"Moderate line structures detected. Consider a higher-res image for better accuracy."
    else:
        confidence = "low"
        notes = "Few line structures detected. Image may be low-res, angled, or the lot uses painted markings only."

    return {"stalls": stalls, "lines": n_lines, "lots": lots, "confidence": confidence, "notes": notes}

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/estimate/on-street", response_model=EstimateResponse)
async def estimate_on_street_endpoint(req: EstimateRequest):
    try:
        result = await estimate_on_street(req.lat, req.lon, req.radius_m)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Overpass API timed out. Try a smaller radius.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return EstimateResponse(
        on_street_stalls=result["stalls"],
        streets_analyzed=result["streets"],
        total_road_length_m=round(result["total_length"], 1),
        confidence=result["confidence"],
        segments=result["segments"][:20],
    )

@app.post("/estimate/image", response_model=ImageAnalysisResponse)
async def estimate_from_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 20MB).")
    try:
        result = detect_stalls_from_image(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return ImageAnalysisResponse(**result)

# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
