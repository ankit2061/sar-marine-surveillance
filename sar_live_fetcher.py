"""
Live Sentinel-1 SAR Data Ingestion Module
=========================================
Automated programmatic streaming of calibrated Sentinel-1 SAR imagery
for marine oil spill detection and vessel surveillance.

Supported Providers:
1. Microsoft Planetary Computer STAC (Sentinel-1 GRD / RTC)
   - Zero credentials required, free public access.
   - HTTP Range streaming directly from Cloud-Optimized GeoTIFFs (COGs).
2. Copernicus Data Space Ecosystem (CDSE)
   - Official ESA OData / OpenSearch API for Sentinel-1 products.
   - Authentication support via CDSE credentials / OAuth2.
3. Google Earth Engine (GEE)
   - Ingestion from COPERNICUS/S1_GRD collection with calibrated sigma0.
"""

from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import numpy as np
import rasterio
from rasterio.windows import Window
import requests
import scipy.ndimage as ndimage

load_dotenv()

# Curated high-interest marine surveillance corridors
PRESET_MARINE_REGIONS = {
    "Mumbai High Offshore (Arabian Sea)": {
        "bbox": [72.20, 19.40, 72.70, 19.85],
        "description": "Major offshore oil & gas production field with active support vessels and platforms.",
        "center_lat": 19.62,
        "center_lon": 72.45,
    },
    "Strait of Malacca (Singapore Approach)": {
        "bbox": [103.40, 1.15, 103.95, 1.45],
        "description": "World's densest maritime corridor; frequent illegal tank cleaning & bilge discharge.",
        "center_lat": 1.30,
        "center_lon": 103.70,
    },
    "Strait of Hormuz (Persian Gulf)": {
        "bbox": [56.10, 26.20, 56.70, 26.70],
        "description": "Strategic crude oil tanker passage with high vessel density.",
        "center_lat": 26.45,
        "center_lon": 56.40,
    },
    "English Channel (Dover Strait)": {
        "bbox": [1.30, 51.00, 1.85, 51.30],
        "description": "Heavy container traffic and strict MARPOL surveillance zone.",
        "center_lat": 51.15,
        "center_lon": 1.55,
    },
}


# ==============================================================================
# PROVIDER 1: Planetary Computer STAC (Zero-Friction Live Streaming)
# ==============================================================================

class PlanetaryComputerSARFetcher:
    """
    Live SAR fetcher utilizing Microsoft Planetary Computer's open STAC API
    for Sentinel-1 Ground Range Detected (GRD) products.
    """

    def __init__(self):
        import pystac_client
        import planetary_computer

        self.stac_endpoint = "https://planetarycomputer.microsoft.com/api/stac/v1"
        self.client = pystac_client.Client.open(
            self.stac_endpoint,
            modifier=planetary_computer.sign_inplace,
        )

    def search_scenes(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        max_items: int = 5,
    ) -> List[Dict]:
        """
        Queries Sentinel-1 GRD scenes intersecting the bounding box within date range.
        """
        search = self.client.search(
            collections=["sentinel-1-grd"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            max_items=max_items,
        )
        items = list(search.items())

        scene_summaries = []
        for item in items:
            dt_str = item.datetime.strftime("%Y-%m-%d %H:%M:%S UTC") if item.datetime else "Unknown"
            polarizations = item.properties.get("sar:polarizations", ["VV", "VH"])
            orbit_direction = item.properties.get("sat:orbit_state", "unknown")
            instrument_mode = item.properties.get("sar:instrument_mode", "IW")

            scene_summaries.append({
                "id": item.id,
                "datetime": dt_str,
                "orbit_state": orbit_direction,
                "instrument_mode": instrument_mode,
                "polarizations": polarizations,
                "item_obj": item,
            })
        return scene_summaries

    def stream_subwindow(
        self,
        item,
        polarization: str = "vv",
        target_shape: Tuple[int, int] = (512, 512),
        crop_center: bool = True,
        offset_ratio: Tuple[float, float] = (0.5, 0.5),
    ) -> Tuple[np.ndarray, Dict]:
        """
        Streams a spatial window (e.g. 512x512) directly from the Cloud-Optimized
        GeoTIFF via HTTP Range requests without downloading the entire 1 GB scene.
        """
        pol_key = polarization.lower()
        if pol_key not in item.assets:
            available_pols = [k for k in ["vv", "vh", "hh", "hv"] if k in item.assets]
            if not available_pols:
                raise ValueError(f"No SAR polarization assets found in item {item.id}")
            pol_key = available_pols[0]

        asset_href = item.assets[pol_key].href

        with rasterio.open(asset_href) as src:
            w, h = src.width, src.height
            tw, th = target_shape

            ox, oy = offset_ratio
            start_x = int(np.clip(ox * (w - tw), 0, w - tw))
            start_y = int(np.clip(oy * (h - th), 0, h - th))

            window = Window(start_x, start_y, tw, th)
            raw_dn = src.read(1, window=window).astype(np.float64)

            # Mask invalid/zero data pixels
            valid_mask = raw_dn > 0
            if not np.any(valid_mask):
                raw_dn = np.ones(target_shape, dtype=np.float64)
            else:
                min_valid = np.percentile(raw_dn[valid_mask], 1.0)
                raw_dn[raw_dn <= 0] = min_valid

            # SAR Intensity Conversion:
            # Digital Number (DN) to relative linear backscatter power: I = DN^2
            # Normalized by median clutter to center ocean baseline around ~1.0
            intensity = raw_dn ** 2
            ocean_baseline = np.median(intensity)
            intensity_norm = intensity / ocean_baseline if ocean_baseline > 1e-6 else intensity

            metadata = {
                "provider": "Planetary Computer STAC (Sentinel-1 GRD)",
                "item_id": item.id,
                "polarization": pol_key.upper(),
                "full_width": w,
                "full_height": h,
                "window_x": start_x,
                "window_y": start_y,
                "crs": str(src.crs),
                "ocean_baseline": float(ocean_baseline),
            }

            return intensity_norm, metadata


# ==============================================================================
# PROVIDER 2: Copernicus Data Space Ecosystem (CDSE OData API)
# ==============================================================================

class CopernicusCDSEFetcher:
    """
    Copernicus Data Space Ecosystem (CDSE) client for searching and accessing
    Sentinel-1 IW products via ESA's official OData catalogue.
    """

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        self.auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        self.username = username or os.environ.get("CDSE_USERNAME")
        self.password = password or os.environ.get("CDSE_PASSWORD")

        # Automatically check Streamlit secrets if running on Streamlit Cloud
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                self.username = self.username or st.secrets.get("CDSE_USERNAME")
                self.password = self.password or st.secrets.get("CDSE_PASSWORD")
        except Exception:
            pass

        self.token = None

    def authenticate(self) -> bool:
        if not self.username or not self.password:
            return False
        try:
            resp = requests.post(
                self.auth_url,
                data={
                    "client_id": "cdse-public",
                    "username": self.username,
                    "password": self.password,
                    "grant_type": "password",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                self.token = resp.json().get("access_token")
                return True
        except Exception:
            pass
        return False

    def search_products(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        max_items: int = 5,
    ) -> List[Dict]:
        """
        Searches Sentinel-1 products in CDSE OData catalogue using WKT intersection.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        center_lon = (min_lon + max_lon) / 2.0
        center_lat = (min_lat + max_lat) / 2.0

        # OData filter query
        filter_str = (
            "Collection/Name eq 'SENTINEL-1' and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({center_lon} {center_lat})') and "
            f"ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T23:59:59.000Z"
        )

        params = {
            "$filter": filter_str,
            "$top": max_items,
            "$orderby": "ContentDate/Start desc",
        }

        try:
            resp = requests.get(self.odata_url, params=params, timeout=12)
            if resp.status_code == 200:
                raw_products = resp.json().get("value", [])
                products = []
                for p in raw_products:
                    products.append({
                        "id": p.get("Id"),
                        "name": p.get("Name"),
                        "datetime": p.get("ContentDate", {}).get("Start", "Unknown"),
                        "size_mb": round(p.get("ContentLength", 0) / (1024 * 1024), 1),
                        "raw_obj": p,
                    })
                return products
        except Exception as e:
            print(f"CDSE Query error: {e}")
        return []


# ==============================================================================
# PROVIDER 3: Google Earth Engine (GEE) Client
# ==============================================================================

class EarthEngineSARFetcher:
    """
    Google Earth Engine (GEE) client for extracting Sentinel-1 calibrated sigma0
    float32 arrays via COPERNICUS/S1_GRD.
    """

    def __init__(self, project_id: Optional[str] = None):
        self.project_id = project_id or os.environ.get("EE_PROJECT_ID")
        self.is_initialized = False

    def initialize(self) -> Tuple[bool, str]:
        """
        Initializes the Earth Engine Python API.
        Checks Streamlit Cloud secrets, local credentials, or project ID.
        """
        import ee

        # 1. Check Streamlit Cloud Secrets (Service Account JSON or Project)
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                if not self.project_id and "EE_PROJECT_ID" in st.secrets:
                    self.project_id = st.secrets["EE_PROJECT_ID"]
                if "EE_SERVICE_ACCOUNT_JSON" in st.secrets:
                    import json
                    key_dict = json.loads(st.secrets["EE_SERVICE_ACCOUNT_JSON"])
                    credentials = ee.ServiceAccountCredentials(
                        key_dict.get("client_email"),
                        key_data=key_dict.get("private_key"),
                    )
                    ee.Initialize(credentials, project=self.project_id)
                    self.is_initialized = True
                    return True, f"Connected via Streamlit Cloud Secrets ({self.project_id})"
        except Exception:
            pass

        # 2. Standard Local Authorization
        try:
            if self.project_id:
                ee.Initialize(project=self.project_id)
            else:
                ee.Initialize()
            self.is_initialized = True
            return True, "Earth Engine initialized successfully!"
        except Exception as e:
            err_msg = (
                f"GEE Initialization failed: {e}.\n"
                "To authenticate locally, run 'earthengine authenticate' in terminal. "
                "On Streamlit Cloud, add EE_SERVICE_ACCOUNT_JSON or EE_PROJECT_ID to Secrets."
            )
            return False, err_msg

    def get_sentinel1_patch(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        polarization: str = "VV",
        patch_size: int = 512,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Pulls a calibrated Sentinel-1 GRD patch from GEE.
        Converts GEE logarithmic dB back to linear intensity power.
        """
        if not self.is_initialized:
            ok, msg = self.initialize()
            if not ok:
                raise RuntimeError(msg)

        import ee

        min_lon, min_lat, max_lon, max_lat = bbox
        geom = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)

        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", polarization))
        )

        image = collection.first().select(polarization)
        # GEE stores calibrated backscatter in dB: sigma0_dB = 10 * log10(I)
        # Convert to linear intensity: I = 10^(sigma0_dB / 10)
        linear_image = ee.Image(10.0).pow(image.divide(10.0))

        # Sample array
        rect = geom.buffer(1000).bounds()
        data = linear_image.sampleRectangle(region=rect, defaultValue=0.0).get(polarization).getInfo()
        arr = np.array(data, dtype=np.float64)

        # Center crop or resize to patch_size
        if arr.shape[0] >= patch_size and arr.shape[1] >= patch_size:
            sy = (arr.shape[0] - patch_size) // 2
            sx = (arr.shape[1] - patch_size) // 2
            arr = arr[sy : sy + patch_size, sx : sx + patch_size]

        meta = {
            "provider": "Google Earth Engine (COPERNICUS/S1_GRD)",
            "polarization": polarization,
            "shape": arr.shape,
        }
        return arr, meta


# ==============================================================================
# 4. UTILITIES: REAL TARGET DETECTION
# ==============================================================================

def find_bright_targets(intensity_image: np.ndarray, threshold_factor: float = 4.0) -> List[Tuple[int, int]]:
    """
    Detects prominent point targets (ships/platforms) in real SAR imagery
    using 2D local spatial peak detection.
    """
    median_val = np.median(intensity_image)
    threshold = median_val * threshold_factor

    footprint = np.ones((7, 7), dtype=bool)
    local_max = ndimage.maximum_filter(intensity_image, footprint=footprint) == intensity_image
    high_pixels = intensity_image > threshold

    peaks = local_max & high_pixels
    peak_coords = list(zip(*np.where(peaks)))

    # Sort by descending intensity
    peak_coords.sort(key=lambda coord: intensity_image[coord], reverse=True)
    return peak_coords[:8]
