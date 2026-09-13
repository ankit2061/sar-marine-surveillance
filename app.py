"""
Interactive Real-Time SAR Speckle Filtering & Dual-Path Pipeline Explorer
========================================================================
Supports:
1. Synthetic SAR Physics Simulation (Controlled Ground Truth).
2. Live Satellite Ingestion (Planetary Computer STAC, Copernicus CDSE, GEE).

Run via:
    streamlit run app.py
"""

from datetime import datetime, timedelta
import os
import numpy as np
import scipy.ndimage as ndimage
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

# ==============================================================================
# DATA LOADING BASED ON SELECTED MODE
# ==============================================================================

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
    scene_source_title = f"Synthetic Simulation (L={n_looks}, Seed={seed})"
    is_synthetic = True

else:
    # --- LIVE SATELLITE INGESTION MODE ---
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
        st.sidebar.success("🟢 Provider Status: Active (No Login or Keys Required)")
    elif "Copernicus" in provider_choice:
        st.sidebar.warning("🟡 CDSE OData Catalogue: Live | Pixel Download: Free Login Required")
        with st.sidebar.expander("🔐 Optional CDSE Credentials"):
            cdse_username = st.text_input("Copernicus Email / Username", value=os.environ.get("CDSE_USERNAME", ""))
            cdse_password = st.text_input("Copernicus Password", type="password", value=os.environ.get("CDSE_PASSWORD", ""))
            if cdse_username and cdse_password:
                fetcher_auth = CopernicusCDSEFetcher(cdse_username, cdse_password)
                if fetcher_auth.authenticate():
                    st.success("✅ CDSE Token Authenticated!")
                else:
                    st.error("❌ CDSE Authentication failed. Check credentials.")
    elif "Earth Engine" in provider_choice:
        st.sidebar.warning("🟡 GEE Status: Requires Local Google Cloud Auth")
        with st.sidebar.expander("🔐 Google Earth Engine Config"):
            gee_project_id = st.text_input("GCP Project ID", value=os.environ.get("EE_PROJECT_ID", ""))
            st.caption("Run `earthengine authenticate` in your local terminal to authorize access.")

    region_name = st.sidebar.selectbox(
        "Target Marine Region",
        list(PRESET_MARINE_REGIONS.keys()) + ["Custom Coordinates"],
        index=0,
    )

    if region_name != "Custom Coordinates":
        region_info = PRESET_MARINE_REGIONS[region_name]
        default_bbox = region_info["bbox"]
        st.sidebar.info(region_info["description"])
    else:
        c1, c2 = st.sidebar.columns(2)
        min_lon = c1.number_input("Min Longitude", value=72.2)
        min_lat = c2.number_input("Min Latitude", value=19.4)
        max_lon = c1.number_input("Max Longitude", value=72.7)
        max_lat = c2.number_input("Max Latitude", value=19.8)
        default_bbox = [min_lon, min_lat, max_lon, max_lat]

    polarization = st.sidebar.selectbox("SAR Polarization", ["VV (Best for Slicks & Surface Roughness)", "VH (Cross-pol, Best for Ships)"], index=0)
    pol_code = "vv" if "VV" in polarization else "vh"

    st.sidebar.markdown("#### Acquisition Date Range")
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=45)
    d_start = st.sidebar.date_input("Start Date", value=start_dt.date())
    d_end = st.sidebar.date_input("End Date", value=end_dt.date())

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
        fetcher = PlanetaryComputerSARFetcher()
        scenes = fetcher.search_scenes(bbox=bbox, start_date=start_str, end_date=end_str, max_items=5)
        if not scenes:
            scenes = fetcher.search_scenes(bbox=bbox, start_date="2024-01-01", end_date="2024-03-31", max_items=5)
        if not scenes:
            raise RuntimeError(f"No Sentinel-1 GRD scenes found over bbox {bbox} in date range.")
        
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
            noisy_scene, live_meta, acq_time, scene_id, cdse_info = fetch_live_sar_data(
                provider_choice,
                default_bbox,
                d_start.strftime("%Y-%m-%d"),
                d_end.strftime("%Y-%m-%d"),
                pol_code,
                cdse_username,
                cdse_password,
                gee_project_id,
            )
            clean_scene = noisy_scene.copy()  # Ground truth is unknown in real satellite data
            scene_source_title = f"Live Sentinel-1 GRD ({pol_code.upper()}) | {acq_time}"

            # Auto-detect real targets (ships/rigs) in the satellite scene
            vessel_coords = find_bright_targets(noisy_scene, threshold_factor=4.5)
            if not vessel_coords:
                vessel_coords = [(256, 256)]
            # Designate an ocean clutter patch for real ENL measurement
            ocean_patch_slice = (slice(20, 140), slice(20, 140))

            st.success(f"📡 Ingested Live Sentinel-1 Scene: **{scene_id}** ({acq_time})")
            if cdse_info:
                st.info(f"🇪🇺 **Copernicus CDSE OData Feed**: {cdse_info}")

    except Exception as e:
        st.error(f"Live fetch error: {e}. Falling back to high-fidelity simulation.")
        clean_scene, noisy_scene, vessel_coords, ocean_patch_slice = generate_synthetic_sar_scene(
            shape=(512, 512), n_looks=4, seed=42
        )
        scene_source_title = "Synthetic Fallback Scene"
        is_synthetic = True

# ==============================================================================
# FILTER EXECUTION & METRIC CALCULATION
# ==============================================================================

mean_name = f"Mean ({window_size}x{window_size})"
gauss_name = f"Gaussian (σ={gaussian_sigma:.1f})"
lee_name = f"Adaptive Lee ({window_size}x{window_size})"

filtered_dict = {
    mean_name: mean_filter(noisy_scene, window_size=window_size),
    gauss_name: gaussian_filter(noisy_scene, sigma=gaussian_sigma),
    lee_name: lee_filter(noisy_scene, window_size=window_size, n_looks=n_looks),
}

# Metrics
metrics = evaluate_filter_performance(
    clean_scene,
    noisy_scene,
    filtered_dict,
    vessel_coords,
    ocean_patch_slice,
)

# ==============================================================================
# DASHBOARD METRICS BAR
# ==============================================================================

st.markdown(f"### 📊 Live Radiometric Dashboard: *{scene_source_title}*")
raw_enl = metrics["ENL"]["Raw Speckled SAR"]
top_vessel_ret = {k: metrics["Vessels"][0]["retention_pct"][k] for k in filtered_dict}

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Raw SAR Ocean ENL", f"{raw_enl:.2f}", delta="Baseline Speckle", delta_color="off")
with col_m2:
    st.metric(f"Mean ({window_size}x{window_size}) ENL", f"{metrics['ENL'][mean_name]:.2f}", delta=f"Target Peak: {top_vessel_ret[mean_name]:.1f}%", delta_color="inverse")
with col_m3:
    st.metric(f"Gaussian (σ={gaussian_sigma:.1f}) ENL", f"{metrics['ENL'][gauss_name]:.2f}", delta=f"Target Peak: {top_vessel_ret[gauss_name]:.1f}%", delta_color="inverse")
with col_m4:
    st.metric(f"Adaptive Lee ({window_size}x{window_size}) ENL", f"{metrics['ENL'][lee_name]:.2f}", delta=f"Target Peak: {top_vessel_ret[lee_name]:.1f}%", delta_color="normal")

# ==============================================================================
# VISUALIZATION TABS
# ==============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ 2x2 Spatial Filter Comparison",
    "📈 1D Radiometric Transect Profile",
    "📋 Quantitative Target Table",
    "🛰️ Satellite API & Provider Details",
])

with tab1:
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), facecolor="#14171A")

    plot_configs = [
        (f"Raw Input ({scene_source_title[:28]})", noisy_scene, axes[0, 0], raw_enl, 100.0),
        (f"Mean Box Filter ({window_size}x{window_size})", filtered_dict[mean_name], axes[0, 1], metrics["ENL"][mean_name], top_vessel_ret[mean_name]),
        (f"Gaussian Filter (σ={gaussian_sigma:.1f})", filtered_dict[gauss_name], axes[1, 0], metrics["ENL"][gauss_name], top_vessel_ret[gauss_name]),
        (f"Adaptive Lee Filter ({window_size}x{window_size})", filtered_dict[lee_name], axes[1, 1], metrics["ENL"][lee_name], top_vessel_ret[lee_name]),
    ]

    y_slice, x_slice = ocean_patch_slice
    roi_x, roi_y = x_slice.start, y_slice.start
    roi_w, roi_h = x_slice.stop - x_slice.start, y_slice.stop - y_slice.start

    v_min, v_max = 0.0, float(np.percentile(noisy_scene, contrast_percentile))
    ship1_r, ship1_c = vessel_coords[0]
    inset_half_size = 22
    r_min, r_max = max(0, ship1_r - inset_half_size), min(512, ship1_r + inset_half_size)
    c_min, c_max = max(0, ship1_c - inset_half_size), min(512, ship1_c + inset_half_size)

    for title, img, ax, enl_val, ship_ret in plot_configs:
        ax.set_facecolor("#0F1115")
        im = ax.imshow(img, cmap=colormap, vmin=v_min, vmax=v_max, origin="upper")
        ax.set_title(title, color="#F5F8FA", fontsize=12, fontweight="bold", pad=10)

        # Highlight Ocean Patch
        rect = patches.Rectangle((roi_x, roi_y), roi_w, roi_h, linewidth=1.5, edgecolor="#00E5FF", facecolor="none", linestyle="--")
        ax.add_patch(rect)
        ax.text(roi_x + 5, roi_y + 16, "Ocean ENL ROI", color="#00E5FF", fontsize=8.5, fontweight="bold")

        # Highlight Targets
        for idx, (r, c) in enumerate(vessel_coords[:4], start=1):
            circle = patches.Circle((c, r), radius=10, linewidth=1.5, edgecolor="#FF3366", facecolor="none")
            ax.add_patch(circle)
            ax.text(c + 12, r + 4, f"Target #{idx}", color="#FF3366", fontsize=8.5, fontweight="bold")

        # Badge Card
        badge_color = "#00E676" if ("Lee" in title or "Raw" in title) else "#FF9100"
        ax.text(
            0.97, 0.96, f"ENL: {enl_val:5.1f}\nPeak Ret: {ship_ret:4.1f}%",
            transform=ax.transAxes, color="#FFFFFF", fontsize=9, fontweight="bold",
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1F242D", edgecolor=badge_color, linewidth=1.4, alpha=0.9),
        )

        # Inset Box over primary target
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
    cbar.set_label("SAR Backscatter Intensity (Linear Power)", color="#E1E8ED", fontsize=10)
    cbar.ax.tick_params(colors="#8899A6", labelsize=8)
    plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.08, wspace=0.1, hspace=0.14)

    st.pyplot(fig, clear_figure=True)

with tab2:
    ship_r, ship_c = vessel_coords[0]
    half_span = 30
    c_start = max(0, ship_c - half_span)
    c_end = min(512, ship_c + half_span)
    cols = np.arange(c_start, c_end)

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), facecolor="#14171A", sharex=True)

    n_slice = noisy_scene[ship_r, cols]
    m_slice = filtered_dict[mean_name][ship_r, cols]
    g_slice = filtered_dict[gauss_name][ship_r, cols]
    l_slice = filtered_dict[lee_name][ship_r, cols]

    # Linear Transect
    ax1.set_facecolor("#0F1115")
    if is_synthetic:
        ax1.plot(cols, clean_scene[ship_r, cols], label="Ground Truth (Clean)", color="#FFFFFF", linestyle="--", linewidth=1.5)
    ax1.plot(cols, n_slice, label="Raw SAR Input", color="#78909C", linewidth=1.0, alpha=0.6)
    ax1.plot(cols, m_slice, label=f"Mean ({window_size}x{window_size})", color="#FF3D71", linewidth=2.0)
    ax1.plot(cols, g_slice, label=f"Gaussian (σ={gaussian_sigma:.1f})", color="#FFA726", linewidth=2.0)
    ax1.plot(cols, l_slice, label=f"Adaptive Lee ({window_size}x{window_size})", color="#00E5FF", linewidth=2.2)
    ax1.set_ylabel("Linear Radar Intensity", color="#E1E8ED")
    ax1.set_title(f"1D Radiometric Transect Across Target (Row {ship_r})", color="#F5F8FA", fontsize=12, fontweight="bold")
    ax1.grid(True, color="#263238", linestyle=":")
    ax1.legend(facecolor="#1F242D", edgecolor="#38444D", labelcolor="#E1E8ED")

    # Decibel Transect
    def to_db(arr):
        return 10.0 * np.log10(np.maximum(1e-4, arr))

    ax2.set_facecolor("#0F1115")
    ax2.plot(cols, to_db(n_slice), color="#78909C", linewidth=1.0, alpha=0.6)
    ax2.plot(cols, to_db(m_slice), color="#FF3D71", linewidth=2.0)
    ax2.plot(cols, to_db(g_slice), color="#FFA726", linewidth=2.0)
    ax2.plot(cols, to_db(l_slice), color="#00E5FF", linewidth=2.2)
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
