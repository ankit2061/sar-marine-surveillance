"""
Interactive Real-Time SAR Speckle Filtering & Dual-Path Pipeline Explorer
========================================================================
Supports:
1. Synthetic SAR Physics Simulation (Controlled Ground Truth).
2. Live Satellite Ingestion (Planetary Computer STAC, Copernicus CDSE, GEE).

Run via:
    streamlit run app.py
"""

from datetime import datetime, timedelta, timezone
import os
import numpy as np
import scipy.ndimage as ndimage
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import streamlit as st

from sar_filter_benchmark import (
    generate_synthetic_sar_scene,
    mean_filter,
    gaussian_filter,
    lee_filter,
    calculate_enl,
    evaluate_filter_performance,
)
from sar_live_fetcher import (
    PlanetaryComputerSARFetcher,
    CopernicusCDSEFetcher,
    EarthEngineSARFetcher,
    PRESET_MARINE_REGIONS,
    find_bright_targets,
    fetch_copernicus_dem_land_mask,
)

st.set_page_config(
    page_title="SAR Marine Surveillance & Speckle Benchmark",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛰️ Dual-Path SAR Speckle Filtering & Target Preservation")
st.caption("Real-Time Benchmark & Satellite Stream for Marine Oil Spill & Vessel Detection")

# --- TOP LEVEL MODE SELECTOR IN SIDEBAR ---
st.sidebar.markdown("## 📡 Pipeline Input Source")
data_mode = st.sidebar.radio(
    "Select SAR Data Mode",
    ["🧪 Synthetic Physics Simulation", "🌍 Live Satellite Ingestion (STAC / CDSE / GEE)"],
    index=0,
)

st.sidebar.markdown("---")

# Global filter settings
st.sidebar.markdown("### ⚙️ Spatial Filter Settings")
window_size = st.sidebar.select_slider(
    "Filter Kernel Window Size",
    options=[3, 5, 7, 9, 11],
    value=5,
    help="Sliding window (NxN) for Mean and Adaptive Lee filters.",
)
gaussian_sigma = st.sidebar.slider(
    "Gaussian Sigma (σ)",
    min_value=0.5,
    max_value=3.0,
    value=1.2,
    step=0.1,
)

# Radiometric Scaling
st.sidebar.markdown("### 🎚️ Radiometric Scaling")
scale_mode = st.sidebar.radio(
    "Scale Mode",
    options=["Linear Power", "Decibels (dB)"],
    index=0,
    help="Convert linear radar intensity to logarithmic decibels (10*log10(I)) to reveal subtle dark slicks without bright ships washing out the scene.",
)

colormap = st.sidebar.selectbox(
    "SAR Colormap",
    options=["viridis", "magma", "inferno", "cividis", "gray"],
    index=0,
)
contrast_percentile = st.sidebar.slider(
    "Display Contrast Stretch (Percentile)",
    min_value=95.0,
    max_value=100.0,
    value=99.2,
    step=0.1,
)

# DEM Land Masking
st.sidebar.markdown("### 🏝️ Coastal Land Masking")
enable_land_mask = st.sidebar.checkbox(
    "Enable Copernicus DEM 30m Land Mask",
    value=True,
    help="Queries Copernicus DEM 30m (cop-dem-glo-30) to mask coastal landmasses and infrastructure as NaN prior to filtering.",
)
dem_elev_threshold = 0.0
if enable_land_mask:
    dem_elev_threshold = st.sidebar.slider(
        "Land Elevation Threshold (m)",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.5,
        help="Pixels with elevation above this threshold are classified as Land and set to NaN. Use 0.5-2.0m for tidal margins.",
    )


def apply_scaling(image_array: np.ndarray, scale_mode: str) -> np.ndarray:
    """
    Applies radiometric scaling to a 2D SAR backscatter intensity array.

    Parameters
    ----------
    image_array : np.ndarray
        Input 2D radar intensity array.
    scale_mode : str
        'Linear Power' or 'Decibels (dB)'.

    Returns
    -------
    np.ndarray
        Transformed array. In dB mode, clips to 1e-5 to prevent log(0) warnings.
    """
    if scale_mode == "Decibels (dB)":
        return 10.0 * np.log10(np.clip(image_array, 1e-5, None))
    return image_array


def segment_oil_spill(
    filtered_sar_array: np.ndarray,
    use_db: bool = True,
) -> dict:
    """
    Automated dark-spot segmentation using Otsu's thresholding for marine oil
    spill detection on despeckled SAR imagery.

    The algorithm:
    1. Extracts valid water pixels (where ~np.isnan), ignoring DEM-masked land.
    2. Optionally converts to dB scale (10·log10) for better bimodal histogram
       separation between ocean clutter and damped-surface oil slicks.
    3. Computes Otsu's optimal binarization threshold on valid water pixels only.
    4. Classifies: pixel < threshold → Oil (1), pixel >= threshold → Ocean (0).
    5. Preserves np.nan for land-masked pixels in the output.

    Parameters
    ----------
    filtered_sar_array : np.ndarray
        2D despeckled SAR intensity array (e.g., Adaptive Lee output).
        May contain np.nan from coastal DEM land masking.
    use_db : bool, default=True
        If True, compute Otsu threshold in dB domain for better separation.

    Returns
    -------
    dict with keys:
        'binary_mask'  : np.ndarray of float — 1.0=Oil, 0.0=Ocean, np.nan=Land
        'threshold_db' : float — Otsu threshold in dB
        'threshold_lin': float — Otsu threshold in linear power
        'oil_pixels'   : int   — count of pixels classified as oil
        'water_pixels' : int   — count of valid (non-land) pixels
        'oil_pct'      : float — percentage of water area classified as oil
    """
    img = np.asarray(filtered_sar_array, dtype=np.float64)
    nan_mask = np.isnan(img)
    valid = img[~nan_mask]

    if valid.size < 10:
        empty = np.full(img.shape, np.nan, dtype=np.float64)
        return {
            'binary_mask': empty,
            'threshold_db': 0.0,
            'threshold_lin': 0.0,
            'oil_pixels': 0,
            'water_pixels': 0,
            'oil_pct': 0.0,
        }

    # Convert to dB domain for better bimodal histogram separation
    valid_db = 10.0 * np.log10(np.clip(valid, 1e-5, None))
    img_db = np.full_like(img, np.nan)
    img_db[~nan_mask] = 10.0 * np.log10(np.clip(img[~nan_mask], 1e-5, None))

    # Compute Otsu threshold on valid water pixels only
    otsu_thresh_db = float(threshold_otsu(valid_db))
    otsu_thresh_lin = 10.0 ** (otsu_thresh_db / 10.0)

    # Classify: below threshold = Oil (1), above = Ocean (0)
    binary_mask = np.full(img.shape, np.nan, dtype=np.float64)
    binary_mask[~nan_mask] = np.where(img_db[~nan_mask] < otsu_thresh_db, 1.0, 0.0)

    oil_count = int(np.nansum(binary_mask == 1.0))
    water_count = int(np.sum(~nan_mask))
    oil_pct = (oil_count / water_count * 100.0) if water_count > 0 else 0.0

    return {
        'binary_mask': binary_mask,
        'threshold_db': otsu_thresh_db,
        'threshold_lin': otsu_thresh_lin,
        'oil_pixels': oil_count,
        'water_pixels': water_count,
        'oil_pct': oil_pct,
    }


# ==============================================================================
# DATA LOADING BASED ON SELECTED MODE
# ==============================================================================

sar_array = None
clean_scene = None
vessel_coords = []
is_synthetic = False
scene_source_title = ""

if data_mode == "🧪 Synthetic Physics Simulation":
    st.sidebar.markdown("### 🕹️ Simulation Parameters")
    seed = st.sidebar.slider("Simulation Random Seed", min_value=1, max_value=200, value=42, step=1)
    n_looks = st.sidebar.slider(
        "Equivalent Number of Looks (L)",
        min_value=1,
        max_value=16,
        value=4,
        help="L=1: Single-Look (SLC, severe speckle). L=4: Sentinel-1 IW GRD.",
    )
    col_p1, col_p2 = st.sidebar.columns(2)
    with col_p1:
        ocean_backscatter = st.number_input("Ocean Backscatter", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
    with col_p2:
        spill_backscatter = st.number_input("Oil Slick Backscatter", min_value=0.01, max_value=0.8, value=0.15, step=0.05)

    clean_scene, noisy_scene, vessel_coords, ocean_patch_slice = generate_synthetic_sar_scene(
        shape=(512, 512),
        n_looks=n_looks,
        ocean_backscatter=ocean_backscatter,
        spill_backscatter=spill_backscatter,
        seed=seed,
    )
    sar_array = noisy_scene
    scene_source_title = f"Synthetic Simulation (L={n_looks}, Seed={seed})"
    is_synthetic = True

else:
    # --- LIVE SATELLITE INGESTION MODE ---
    is_synthetic = False
    vessel_coords = []  # Explicitly disable synthetic targets

    st.sidebar.markdown("### 🛰️ Live Satellite Provider")
    provider_choice = st.sidebar.selectbox(
        "Satellite Data Provider",
        [
            "Microsoft Planetary Computer STAC (Free Public COG Stream)",
            "Copernicus Data Space Ecosystem (CDSE OData)",
            "Google Earth Engine (GEE COPERNICUS/S1_GRD)",
        ],
        index=0,
    )

    # Provider connection state & credentials
    cdse_username = None
    cdse_password = None
    gee_project_id = None

    if "Planetary Computer" in provider_choice:
        st.sidebar.success("🟢 Provider Status: Active (Free Public COG Stream)")
    elif "Copernicus" in provider_choice:
        cdse_username = os.environ.get("CDSE_USERNAME", "")
        cdse_password = os.environ.get("CDSE_PASSWORD", "")
        with st.sidebar.expander("🔐 Copernicus CDSE Credentials", expanded=not (cdse_username and cdse_password)):
            cdse_username = st.text_input("Copernicus Email / Username", value=cdse_username)
            cdse_password = st.text_input("Copernicus Password", type="password", value=cdse_password)
            if cdse_username and cdse_password:
                fetcher_auth = CopernicusCDSEFetcher(cdse_username, cdse_password)
                if fetcher_auth.authenticate():
                    st.success("✅ CDSE Token Authenticated!")
                else:
                    st.error("❌ CDSE Authentication failed. Check credentials.")
            else:
                st.caption("Register for free at [dataspace.copernicus.eu](https://dataspace.copernicus.eu/)")
        if cdse_username and cdse_password:
            st.sidebar.success("🟢 CDSE Status: Authenticated & Active")
        else:
            st.sidebar.info("ℹ️ CDSE Status: OData Search Active (Streaming via STAC)")
    elif "Earth Engine" in provider_choice:
        gee_project_id = os.environ.get("EE_PROJECT_ID", "charged-atlas-457609-f3")
        gee_fetcher = EarthEngineSARFetcher(project_id=gee_project_id)
        is_gee_ok, gee_msg = gee_fetcher.initialize()
        if is_gee_ok:
            st.sidebar.success(f"🟢 GEE Status: Connected (`{gee_project_id}`)")
        else:
            st.sidebar.warning("🟡 GEE Status: Authorization Required")
        with st.sidebar.expander("⚙️ Google Earth Engine Config", expanded=not is_gee_ok):
            gee_project_id = st.text_input("GCP Project ID", value=gee_project_id)
            if is_gee_ok:
                st.caption("✅ Authorized with project: " + gee_project_id)
            else:
                st.caption(gee_msg)

    region_name = st.sidebar.selectbox(
        "Target Marine Region",
        list(PRESET_MARINE_REGIONS.keys()) + ["Custom Coordinates"],
        index=0,
    )

    if region_name != "Custom Coordinates":
        region_info = PRESET_MARINE_REGIONS[region_name]
        default_bbox = region_info["bbox"]
        st.sidebar.info(region_info["description"])
        if "default_dates" in region_info:
            init_start = datetime.strptime(region_info["default_dates"][0], "%Y-%m-%d").date()
            init_end = datetime.strptime(region_info["default_dates"][1], "%Y-%m-%d").date()
        else:
            init_end = datetime.now(timezone.utc).date()
            init_start = init_end - timedelta(days=45)
    else:
        init_end = datetime.now(timezone.utc).date()
        init_start = init_end - timedelta(days=45)
        c1, c2 = st.sidebar.columns(2)
        min_lon = c1.number_input("Min Longitude", value=57.65, format="%.4f")
        min_lat = c2.number_input("Min Latitude", value=-20.55, format="%.4f")
        max_lon = c1.number_input("Max Longitude", value=57.85, format="%.4f")
        max_lat = c2.number_input("Max Latitude", value=-20.35, format="%.4f")
        default_bbox = [min_lon, min_lat, max_lon, max_lat]

    polarization = st.sidebar.selectbox("SAR Polarization", ["VV (Best for Slicks & Surface Roughness)", "VH (Cross-pol, Best for Ships)"], index=0)
    pol_code = "vv" if "VV" in polarization else "vh"

    st.sidebar.markdown("#### Acquisition Date Range")
    d_start = st.sidebar.date_input("Start Date", value=init_start)
    d_end = st.sidebar.date_input("End Date", value=init_end)

    n_looks = 4  # Sentinel-1 IW GRD nominal looks
    is_synthetic = False

    # Cache live streaming to prevent re-querying on every UI click
    @st.cache_data(show_spinner=True)
    def fetch_live_sar_data(provider_name, bbox, start_str, end_str, pol, cdse_user=None, cdse_pass=None, gee_proj=None):
        cdse_summary = None

        # If CDSE is selected, query the official ESA OData catalogue
        if "Copernicus" in provider_name:
            cdse_fetcher = CopernicusCDSEFetcher(cdse_user, cdse_pass)
            cdse_products = cdse_fetcher.search_products(bbox, start_str, end_str, max_items=3)
            if cdse_products:
                cdse_summary = f"Found {len(cdse_products)} scenes in CDSE OData. Latest: {cdse_products[0]['name']}"

        # If GEE is selected and initialized
        if "Earth Engine" in provider_name:
            gee_fetcher = EarthEngineSARFetcher(project_id=gee_proj)
            ok, msg = gee_fetcher.initialize()
            if ok:
                try:
                    img, meta = gee_fetcher.get_sentinel1_patch(bbox, start_str, end_str, polarization=pol.upper(), patch_size=512)
                    return img, meta, "GEE Live Composite", "COPERNICUS/S1_GRD", cdse_summary
                except Exception as e:
                    st.warning(f"GEE Fetch notice: {e}. Streaming via STAC COG layer.")

        # Default fast public streaming layer: Microsoft Planetary Computer STAC
        # Enforces high-resolution Interferometric Wide (IW, 10m) mode data
        fetcher = PlanetaryComputerSARFetcher()
        scenes = fetcher.search_scenes(bbox=bbox, start_date=start_str, end_date=end_str, max_items=5, instrument_mode="IW")
        if not scenes:
            # Fallback to any mode (e.g. EW) if no IW swath exists over open ocean
            scenes = fetcher.search_scenes(bbox=bbox, start_date=start_str, end_date=end_str, max_items=5, instrument_mode=None)
        if not scenes:
            raise RuntimeError(f"No Sentinel-1 GRD scenes found over bbox {bbox} in date range {start_str} to {end_str}.")
        
        target_scene = scenes[0]
        img, meta = fetcher.stream_subwindow(
            target_scene["item_obj"],
            polarization=pol,
            target_shape=(512, 512),
            crop_center=True,
        )
        return img, meta, target_scene["datetime"], target_scene["id"], cdse_summary

    try:
        with st.spinner("Streaming real Sentinel-1 SAR subwindow from orbit via STAC..."):
            downloaded_img, live_meta, acq_time, scene_id, cdse_info = fetch_live_sar_data(
                provider_choice,
                default_bbox,
                d_start.strftime("%Y-%m-%d"),
                d_end.strftime("%Y-%m-%d"),
                pol_code,
                cdse_username,
                cdse_password,
                gee_project_id,
            )
            # Strictly overwrite sar_array with the downloaded 2D satellite array
            sar_array = np.array(downloaded_img, dtype=np.float64)
            clean_scene = sar_array.copy()
            scene_source_title = f"Live Sentinel-1 GRD ({pol_code.upper()}) | {acq_time}"

            # Designate an ocean clutter patch for real ENL measurement
            h_sar, w_sar = sar_array.shape
            ocean_patch_slice = (
                slice(int(h_sar * 0.05), int(h_sar * 0.25)),
                slice(int(w_sar * 0.05), int(w_sar * 0.25)),
            )

            st.success(f"📡 Ingested Live Sentinel-1 Scene: **{scene_id}** ({acq_time})")
            if cdse_info:
                st.info(f"🇪🇺 **Copernicus CDSE OData Feed**: {cdse_info}")

    except Exception as e:
        st.error(f"Live satellite fetch error: {e}. Please check your connection or date range.")
        st.stop()

if sar_array is None:
    st.warning("No SAR array loaded. Please select an input source.")
    st.stop()

# --- COPERNICUS DEM LAND MASKING ---
land_mask = None
if enable_land_mask and not is_synthetic:
    with st.spinner("Fetching Copernicus DEM 30m & masking coastal landmasses..."):
        try:
            land_mask, dem_elev = fetch_copernicus_dem_land_mask(
                bbox=default_bbox,
                target_shape=sar_array.shape,
                elevation_threshold=dem_elev_threshold,
            )
            land_count = int(np.sum(land_mask))
            land_pct = (land_count / sar_array.size) * 100.0
            if land_count > 0:
                # Apply mask: set all Land pixels to np.nan
                sar_array = np.where(land_mask, np.nan, sar_array)
                clean_scene = sar_array.copy()
                st.info(f"🏝️ **Copernicus DEM Land Mask Applied**: Masked {land_count:,} land pixels ({land_pct:.1f}% of scene at >{dem_elev_threshold:.1f}m elevation).")
            else:
                st.caption("ℹ️ Copernicus DEM indicates 100% open water (no land detected).")
        except Exception as e:
            st.warning(f"DEM Land Masking notice: {e}. Continuing with unmasked SAR.")

# ==============================================================================
# FILTER EXECUTION & METRIC CALCULATION
# ==============================================================================

mean_name = f"Mean ({window_size}x{window_size})"
gauss_name = f"Gaussian (σ={gaussian_sigma:.1f})"
lee_name = f"Adaptive Lee ({window_size}x{window_size})"

# Downstream spatial filters strictly operate on sar_array
filtered_dict = {
    mean_name: mean_filter(sar_array, window_size=window_size),
    gauss_name: gaussian_filter(sar_array, sigma=gaussian_sigma),
    lee_name: lee_filter(sar_array, window_size=window_size, n_looks=n_looks),
}

# Metrics
metrics = evaluate_filter_performance(
    clean_scene if clean_scene is not None else sar_array,
    sar_array,
    filtered_dict,
    vessel_coords if is_synthetic else [],
    ocean_patch_slice,
)

# ==============================================================================
# DASHBOARD METRICS BAR
# ==============================================================================

st.markdown(f"### 📊 Live Radiometric Dashboard: *{scene_source_title}*")
raw_enl = metrics["ENL"]["Raw Speckled SAR"]
has_vessels = is_synthetic and len(metrics.get("Vessels", [])) > 0
if has_vessels:
    top_vessel_ret = {k: metrics["Vessels"][0]["retention_pct"][k] for k in filtered_dict}

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Raw SAR Ocean ENL", f"{raw_enl:.2f}", delta="Baseline Speckle", delta_color="off")
with col_m2:
    delta_val = f"Target Peak: {top_vessel_ret[mean_name]:.1f}%" if has_vessels else f"+{(metrics['ENL'][mean_name] - raw_enl):.2f} ENL"
    st.metric(f"Mean ({window_size}x{window_size}) ENL", f"{metrics['ENL'][mean_name]:.2f}", delta=delta_val, delta_color="inverse" if has_vessels else "normal")
with col_m3:
    delta_val = f"Target Peak: {top_vessel_ret[gauss_name]:.1f}%" if has_vessels else f"+{(metrics['ENL'][gauss_name] - raw_enl):.2f} ENL"
    st.metric(f"Gaussian (σ={gaussian_sigma:.1f}) ENL", f"{metrics['ENL'][gauss_name]:.2f}", delta=delta_val, delta_color="inverse" if has_vessels else "normal")
with col_m4:
    delta_val = f"Target Peak: {top_vessel_ret[lee_name]:.1f}%" if has_vessels else f"+{(metrics['ENL'][lee_name] - raw_enl):.2f} ENL"
    st.metric(f"Adaptive Lee ({window_size}x{window_size}) ENL", f"{metrics['ENL'][lee_name]:.2f}", delta=delta_val, delta_color="normal")

# ==============================================================================
# VISUALIZATION TABS
# ==============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🖼️ 2x2 Spatial Filter Comparison",
    "📈 1D Radiometric Transect Profile",
    "📋 Quantitative Target Table",
    "🛰️ Satellite API & Provider Details",
    "🛢️ Path A: Oil Spill Segmentation",
])

with tab1:
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), facecolor="#14171A")

    # Strictly apply selected radiometric scaling directly before rendering in matplotlib
    scaled_raw = apply_scaling(sar_array, scale_mode)
    scaled_mean = apply_scaling(filtered_dict[mean_name], scale_mode)
    scaled_gauss = apply_scaling(filtered_dict[gauss_name], scale_mode)
    scaled_lee = apply_scaling(filtered_dict[lee_name], scale_mode)

    plot_configs = [
        (f"Raw Input ({scene_source_title[:28]})", scaled_raw, axes[0, 0], raw_enl),
        (f"Mean Box Filter ({window_size}x{window_size})", scaled_mean, axes[0, 1], metrics["ENL"][mean_name]),
        (f"Gaussian Filter (σ={gaussian_sigma:.1f})", scaled_gauss, axes[1, 0], metrics["ENL"][gauss_name]),
        (f"Adaptive Lee Filter ({window_size}x{window_size})", scaled_lee, axes[1, 1], metrics["ENL"][lee_name]),
    ]

    y_slice, x_slice = ocean_patch_slice
    roi_x, roi_y = x_slice.start, y_slice.start
    roi_w, roi_h = x_slice.stop - x_slice.start, y_slice.stop - y_slice.start

    # Dynamic display range based on linear or logarithmic (dB) scale (ignoring NaNs)
    valid_pixels = scaled_raw[~np.isnan(scaled_raw)]
    if valid_pixels.size > 0:
        if scale_mode == "Decibels (dB)":
            v_min = float(np.percentile(valid_pixels, 2.0))
            v_max = float(np.percentile(valid_pixels, contrast_percentile))
        else:
            v_min = float(np.percentile(valid_pixels, 1.0))
            v_max = float(np.percentile(valid_pixels, contrast_percentile))
    else:
        v_min, v_max = 0.0, 1.0

    # Configure colormap with dark masked land color for NaNs
    cmap_obj = plt.get_cmap(colormap).copy()
    cmap_obj.set_bad(color="#101418")

    for title, img, ax, enl_val in plot_configs:
        ax.set_facecolor("#0F1115")
        im = ax.imshow(img, cmap=cmap_obj, vmin=v_min, vmax=v_max, origin="upper")
        ax.set_title(title, color="#F5F8FA", fontsize=12, fontweight="bold", pad=10)

        # Highlight Ocean ENL ROI
        rect = patches.Rectangle((roi_x, roi_y), roi_w, roi_h, linewidth=1.5, edgecolor="#00E5FF", facecolor="none", linestyle="--")
        ax.add_patch(rect)
        ax.text(roi_x + 5, roi_y + 16, "Ocean ENL ROI", color="#00E5FF", fontsize=8.5, fontweight="bold")

        # Highlight Targets ONLY in synthetic mode (disabled in Live Satellite mode)
        if is_synthetic and vessel_coords:
            for idx, (r, c) in enumerate(vessel_coords[:4], start=1):
                circle = patches.Circle((c, r), radius=10, linewidth=1.5, edgecolor="#FF3366", facecolor="none")
                ax.add_patch(circle)
                ax.text(c + 12, r + 4, f"Target #{idx}", color="#FF3366", fontsize=8.5, fontweight="bold")

        # Badge Card
        badge_color = "#00E676" if ("Lee" in title or "Raw" in title) else "#FF9100"
        if is_synthetic and has_vessels:
            ship_ret = 100.0 if "Raw" in title else metrics["Vessels"][0]["retention_pct"].get(title.split(" (")[0], 100.0)
            card_text = f"ENL: {enl_val:5.1f}\nPeak Ret: {ship_ret:4.1f}%"
        else:
            card_text = f"ENL: {enl_val:5.1f}\nMode: Live SAR"

        ax.text(
            0.97, 0.96, card_text,
            transform=ax.transAxes, color="#FFFFFF", fontsize=9, fontweight="bold",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1F242D", edgecolor=badge_color, linewidth=1.4, alpha=0.9),
        )

        # Inset Box over primary target ONLY in synthetic simulation mode
        if is_synthetic and vessel_coords:
            ship1_r, ship1_c = vessel_coords[0]
            inset_half_size = 22
            r_min, r_max = max(0, ship1_r - inset_half_size), min(sar_array.shape[0], ship1_r + inset_half_size)
            c_min, c_max = max(0, ship1_c - inset_half_size), min(sar_array.shape[1], ship1_c + inset_half_size)
            if r_max - r_min > 5 and c_max - c_min > 5:
                ax_ins = inset_axes(ax, width="28%", height="28%", loc="lower right", bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=0.6)
                ax_ins.imshow(img[r_min:r_max, c_min:c_max], cmap=colormap, vmin=v_min, vmax=v_max, origin="upper")
                ax_ins.set_xticks([])
                ax_ins.set_yticks([])
                for s in ax_ins.spines.values():
                    s.set_edgecolor("#00E5FF")
                    s.set_linewidth(1.3)

        ax.tick_params(colors="#8899A6", labelsize=8)
        for s in ax.spines.values():
            s.set_edgecolor("#38444D")

    cbar_ax = fig.add_axes([0.2, 0.045, 0.6, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar_label = (
        "SAR Backscatter Intensity (dB = 10·log₁₀(I))"
        if scale_mode == "Decibels (dB)"
        else "SAR Backscatter Intensity (Linear Power)"
    )
    cbar.set_label(cbar_label, color="#E1E8ED", fontsize=10)
    cbar.ax.tick_params(colors="#8899A6", labelsize=8)
    plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.08, wspace=0.1, hspace=0.14)

    st.pyplot(fig, clear_figure=True)

with tab2:
    if is_synthetic and vessel_coords:
        transect_r, transect_c = vessel_coords[0]
        half_span = 30
        c_start = max(0, transect_c - half_span)
        c_end = min(sar_array.shape[1], transect_c + half_span)
        cols = np.arange(c_start, c_end)
        transect_title = f"1D Radiometric Transect Across Target (Row {transect_r})"
    else:
        transect_r = sar_array.shape[0] // 2
        cols = np.arange(0, sar_array.shape[1])
        transect_title = f"1D Radiometric Transect Across Scene (Row {transect_r})"

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor="#14171A", sharex=True)

    n_slice = sar_array[transect_r, cols]
    m_slice = filtered_dict[mean_name][transect_r, cols]
    g_slice = filtered_dict[gauss_name][transect_r, cols]
    l_slice = filtered_dict[lee_name][transect_r, cols]

    # Linear Transect
    ax1.set_facecolor("#0F1115")
    if is_synthetic and clean_scene is not None:
        ax1.plot(cols, clean_scene[transect_r, cols], label="Ground Truth (Clean)", color="#FFFFFF", linestyle="--", linewidth=1.5)
    ax1.plot(cols, n_slice, label="Raw SAR Input", color="#78909C", linewidth=1.0, alpha=0.6)
    ax1.plot(cols, m_slice, label=f"Mean ({window_size}x{window_size})", color="#FF3D71", linewidth=2.0)
    ax1.plot(cols, g_slice, label=f"Gaussian (σ={gaussian_sigma:.1f})", color="#FFA726", linewidth=2.0)
    ax1.plot(cols, l_slice, label=f"Adaptive Lee ({window_size}x{window_size})", color="#00E5FF", linewidth=2.2)
    ax1.set_ylabel("Linear Radar Intensity", color="#E1E8ED")
    ax1.set_title(transect_title, color="#F5F8FA", fontsize=12, fontweight="bold")
    ax1.grid(True, color="#263238", linestyle=":")
    ax1.legend(facecolor="#1F242D", edgecolor="#38444D", labelcolor="#E1E8ED")

    # Decibel Transect
    ax2.set_facecolor("#0F1115")
    ax2.plot(cols, apply_scaling(n_slice, "Decibels (dB)"), color="#78909C", linewidth=1.0, alpha=0.6)
    ax2.plot(cols, apply_scaling(m_slice, "Decibels (dB)"), color="#FF3D71", linewidth=2.0)
    ax2.plot(cols, apply_scaling(g_slice, "Decibels (dB)"), color="#FFA726", linewidth=2.0)
    ax2.plot(cols, apply_scaling(l_slice, "Decibels (dB)"), color="#00E5FF", linewidth=2.2)
    ax2.set_ylabel("Backscatter (dB = 10·log₁₀(I))", color="#E1E8ED")
    ax2.set_xlabel("Pixel Column Coordinate", color="#E1E8ED")
    ax2.grid(True, color="#263238", linestyle=":")

    for ax in (ax1, ax2):
        ax.tick_params(colors="#8899A6")
        for s in ax.spines.values():
            s.set_edgecolor("#38444D")

    plt.tight_layout()
    st.pyplot(fig2, clear_figure=True)

with tab3:
    if is_synthetic and has_vessels:
        st.markdown("#### Detected Target Radiometric Retention Summary")
        data_table = []
        for v in metrics["Vessels"]:
            data_table.append({
                "Target ID": f"Target #{v['vessel_id']} @ {v['coord']}",
                "Raw SAR Peak": f"{v['noisy_peak']:.2f}",
                f"{mean_name} Peak": f"{v['peaks'][mean_name]:.2f} ({v['retention_pct'][mean_name]:.1f}%)",
                f"{gauss_name} Peak": f"{v['peaks'][gauss_name]:.2f} ({v['retention_pct'][gauss_name]:.1f}%)",
                f"{lee_name} Peak": f"{v['peaks'][lee_name]:.2f} ({v['retention_pct'][lee_name]:.1f}%)",
            })
        st.table(data_table)
    else:
        st.markdown("#### 🛰️ Live Satellite Scene Radiometric Statistics")
        db_arr = apply_scaling(sar_array, "Decibels (dB)")
        land_info = f"{int(np.sum(np.isnan(sar_array))):,} pixels ({(np.sum(np.isnan(sar_array))/sar_array.size)*100:.1f}%)" if np.any(np.isnan(sar_array)) else "0 pixels (100% water)"
        stats_table = [
            {"Parameter": "Data Source", "Value": scene_source_title},
            {"Parameter": "Scene Spatial Dimensions", "Value": f"{sar_array.shape[0]} × {sar_array.shape[1]} pixels"},
            {"Parameter": "Copernicus DEM Masked Land", "Value": land_info},
            {"Parameter": "Water Intensity Range (Linear)", "Value": f"[{np.nanmin(sar_array):.4f}, {np.nanmax(sar_array):.4f}]"},
            {"Parameter": "Mean Water Radar Intensity", "Value": f"{np.nanmean(sar_array):.4f} (std={np.nanstd(sar_array):.4f})"},
            {"Parameter": "Water Backscatter Dynamic Range (dB)", "Value": f"[{np.nanmin(db_arr):.1f} dB, {np.nanmax(db_arr):.1f} dB]"},
            {"Parameter": "Mean Water Backscatter (dB)", "Value": f"{np.nanmean(db_arr):.2f} dB"},
            {"Parameter": "Ocean ROI ENL (Raw SAR)", "Value": f"{raw_enl:.2f}"},
            {"Parameter": f"Ocean ROI ENL ({mean_name})", "Value": f"{metrics['ENL'][mean_name]:.2f} (+{(metrics['ENL'][mean_name] - raw_enl):.2f})"},
            {"Parameter": f"Ocean ROI ENL ({gauss_name})", "Value": f"{metrics['ENL'][gauss_name]:.2f} (+{(metrics['ENL'][gauss_name] - raw_enl):.2f})"},
            {"Parameter": f"Ocean ROI ENL ({lee_name})", "Value": f"{metrics['ENL'][lee_name]:.2f} (+{(metrics['ENL'][lee_name] - raw_enl):.2f})"},
        ]
        st.table(stats_table)

with tab4:
    st.markdown("### 🛰️ Real-Time Satellite Provider Architecture & Status")
    
    st.markdown("""
    | Provider | Current Operational Status | Authentication Required? | How It Works |
    |---|---|---|---|
    | **1. Planetary Computer STAC** | 🟢 **Active & Streaming** | ❌ **No (Free Public Access)** | Uses pre-signed SAS tokens to stream 512×512 subwindows directly out of Sentinel-1 Cloud-Optimized GeoTIFFs via HTTP Range requests in <3 seconds. |
    | **2. Copernicus CDSE (ESA)** | 🟡 **OData Search Active** | ⚠️ **Yes (Free CDSE Account)** | OData API queries official ESA product catalogue. Downloading full raster products requires entering free CDSE login credentials in the sidebar. |
    | **3. Google Earth Engine (GEE)** | 🟡 **Client Integrated** | ⚠️ **Yes (Google Cloud Project)** | Integrates `COPERNICUS/S1_GRD` collection. Requires running `earthengine authenticate` or providing a Google Cloud Project ID. |
    """)

    st.markdown("---")
    st.markdown("""
    #### Why Planetary Computer is the Active Default:
    Sentinel-1 orbital strips are huge (typically **$1.0\text{ GB}$ to $1.8\text{ GB}$** per scene). Downloading a full scene from Copernicus CDSE takes 1–3 minutes per query. 
    
    In contrast, **Cloud-Optimized GeoTIFFs (COGs)** on Planetary Computer allow reading just the $512 \times 512$ ocean pixels using HTTP range requests in **$<3$ seconds**, making real-time interactive parameter tuning possible in the browser!
    """)

with tab5:
    st.markdown("### 🛢️ Path A: Automated Dark-Spot Oil Spill Segmentation")
    st.caption(
        "Otsu's method computes an optimal binarization threshold by maximizing "
        "inter-class variance between the ocean clutter and damped-surface oil slick "
        "intensity distributions. The Adaptive Lee–filtered SAR image is used as input "
        "to minimize false alarms from residual speckle."
    )

    # Run segmentation on the Lee-filtered output
    lee_filtered = filtered_dict[lee_name]
    seg_result = segment_oil_spill(lee_filtered, use_db=True)

    # --- Metrics Row ---
    seg_c1, seg_c2, seg_c3, seg_c4 = st.columns(4)
    with seg_c1:
        st.metric(
            "Otsu Threshold",
            f"{seg_result['threshold_db']:.2f} dB",
            delta=f"Linear: {seg_result['threshold_lin']:.6f}",
            delta_color="off",
        )
    with seg_c2:
        st.metric(
            "Detected Oil Pixels",
            f"{seg_result['oil_pixels']:,}",
            delta=f"{seg_result['oil_pct']:.1f}% of water area",
            delta_color="off",
        )
    with seg_c3:
        st.metric(
            "Valid Water Pixels",
            f"{seg_result['water_pixels']:,}",
        )
    with seg_c4:
        land_px = int(np.sum(np.isnan(lee_filtered)))
        st.metric(
            "Masked Land Pixels",
            f"{land_px:,}",
            delta=f"{(land_px / lee_filtered.size * 100):.1f}% of scene" if land_px > 0 else "0%",
            delta_color="off",
        )

    # --- Side-by-side Plot: Lee-Filtered SAR vs Binary Segmentation ---
    fig_seg, (ax_sar, ax_bin) = plt.subplots(
        1, 2, figsize=(14, 6), facecolor="#14171A",
        gridspec_kw={"wspace": 0.08},
    )

    # LEFT: Lee-filtered SAR in dB with land mask
    lee_db = apply_scaling(lee_filtered, "Decibels (dB)")
    valid_lee = lee_db[~np.isnan(lee_db)]
    if valid_lee.size > 0:
        vmin_lee = float(np.percentile(valid_lee, 2.0))
        vmax_lee = float(np.percentile(valid_lee, 99.0))
    else:
        vmin_lee, vmax_lee = -30.0, 0.0

    cmap_lee = plt.get_cmap("inferno").copy()
    cmap_lee.set_bad(color="#080A0C")
    ax_sar.set_facecolor("#0F1115")
    im_sar = ax_sar.imshow(
        lee_db, cmap=cmap_lee, vmin=vmin_lee, vmax=vmax_lee, origin="upper",
    )
    ax_sar.set_title(
        f"Adaptive Lee Filtered ({window_size}×{window_size}) — dB Scale",
        color="#F5F8FA", fontsize=11, fontweight="bold", pad=10,
    )

    # Draw Otsu threshold line on colorbar
    cbar_sar = fig_seg.colorbar(im_sar, ax=ax_sar, fraction=0.046, pad=0.04)
    cbar_sar.set_label("Backscatter (dB)", color="#E1E8ED", fontsize=9)
    cbar_sar.ax.tick_params(colors="#8899A6", labelsize=8)
    if vmin_lee <= seg_result['threshold_db'] <= vmax_lee:
        cbar_sar.ax.axhline(
            y=seg_result['threshold_db'], color="#FF3366",
            linewidth=2.0, linestyle="--",
        )
        cbar_sar.ax.text(
            1.8, seg_result['threshold_db'], f"Otsu\n{seg_result['threshold_db']:.1f} dB",
            color="#FF3366", fontsize=8, fontweight="bold",
            va="center", transform=cbar_sar.ax.get_yaxis_transform(),
        )

    ax_sar.tick_params(colors="#8899A6", labelsize=8)
    for s in ax_sar.spines.values():
        s.set_edgecolor("#38444D")

    # RIGHT: Binary segmentation mask
    # Custom colormap: 0=Ocean (deep blue), 1=Oil (bright yellow), NaN=Land (black)
    from matplotlib.colors import ListedColormap
    spill_cmap = ListedColormap(["#0A1628", "#FFD700"])
    spill_cmap.set_bad(color="#080A0C")

    ax_bin.set_facecolor("#0F1115")
    im_bin = ax_bin.imshow(
        seg_result['binary_mask'], cmap=spill_cmap, vmin=0, vmax=1,
        origin="upper", interpolation="nearest",
    )
    ax_bin.set_title(
        "Otsu Binary Segmentation (Oil = Yellow, Ocean = Blue)",
        color="#F5F8FA", fontsize=11, fontweight="bold", pad=10,
    )

    # Legend patch
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#0A1628", edgecolor="#38444D", label="Ocean (Clean)"),
        Patch(facecolor="#FFD700", edgecolor="#38444D", label="Oil Spill (Detected)"),
        Patch(facecolor="#080A0C", edgecolor="#38444D", label="Land (DEM Masked)"),
    ]
    ax_bin.legend(
        handles=legend_elements, loc="lower right",
        facecolor="#1F242D", edgecolor="#38444D", labelcolor="#E1E8ED",
        fontsize=8.5, framealpha=0.9,
    )

    # Badge with threshold and oil area
    badge_text = (
        f"Otsu: {seg_result['threshold_db']:.1f} dB\n"
        f"Oil Area: {seg_result['oil_pct']:.1f}%"
    )
    ax_bin.text(
        0.03, 0.96, badge_text,
        transform=ax_bin.transAxes, color="#FFFFFF", fontsize=9.5,
        fontweight="bold", va="top", ha="left",
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="#1F242D",
            edgecolor="#FFD700", linewidth=1.5, alpha=0.92,
        ),
    )

    ax_bin.tick_params(colors="#8899A6", labelsize=8)
    for s in ax_bin.spines.values():
        s.set_edgecolor("#38444D")

    plt.subplots_adjust(left=0.04, right=0.92, top=0.92, bottom=0.06)
    st.pyplot(fig_seg, clear_figure=True)

    # --- Histogram: Water pixel dB distribution with Otsu threshold line ---
    st.markdown("#### 📊 Water Pixel Intensity Distribution & Otsu Decision Boundary")
    if valid_lee.size > 20:
        fig_hist, ax_hist = plt.subplots(figsize=(10, 4), facecolor="#14171A")
        ax_hist.set_facecolor("#0F1115")

        counts, bin_edges, bar_patches = ax_hist.hist(
            valid_lee, bins=128, color="#1E88E5", alpha=0.85,
            edgecolor="#0D47A1", linewidth=0.4,
        )

        # Color bins below Otsu threshold in yellow (oil side)
        for patch, left_edge in zip(bar_patches, bin_edges[:-1]):
            if left_edge < seg_result['threshold_db']:
                patch.set_facecolor("#FFD700")
                patch.set_edgecolor("#F9A825")

        ax_hist.axvline(
            seg_result['threshold_db'], color="#FF3366", linewidth=2.5,
            linestyle="--", label=f"Otsu Threshold = {seg_result['threshold_db']:.2f} dB",
        )
        ax_hist.fill_betweenx(
            [0, counts.max() * 1.05],
            vmin_lee, seg_result['threshold_db'],
            alpha=0.08, color="#FFD700",
        )
        ax_hist.fill_betweenx(
            [0, counts.max() * 1.05],
            seg_result['threshold_db'], vmax_lee,
            alpha=0.06, color="#1E88E5",
        )

        ax_hist.text(
            seg_result['threshold_db'] - 1.5, counts.max() * 0.85,
            "← Oil Spill", color="#FFD700", fontsize=10, fontweight="bold",
            ha="right",
        )
        ax_hist.text(
            seg_result['threshold_db'] + 1.5, counts.max() * 0.85,
            "Ocean →", color="#64B5F6", fontsize=10, fontweight="bold",
            ha="left",
        )

        ax_hist.set_xlabel("Backscatter Intensity (dB)", color="#E1E8ED", fontsize=10)
        ax_hist.set_ylabel("Pixel Count", color="#E1E8ED", fontsize=10)
        ax_hist.set_title(
            "Bimodal Histogram of Lee-Filtered Water Pixels (dB Domain)",
            color="#F5F8FA", fontsize=12, fontweight="bold",
        )
        ax_hist.legend(
            facecolor="#1F242D", edgecolor="#FF3366",
            labelcolor="#E1E8ED", fontsize=9.5,
        )
        ax_hist.tick_params(colors="#8899A6", labelsize=9)
        for s in ax_hist.spines.values():
            s.set_edgecolor("#38444D")

        plt.tight_layout()
        st.pyplot(fig_hist, clear_figure=True)
    else:
        st.warning("Insufficient valid water pixels for histogram rendering.")
