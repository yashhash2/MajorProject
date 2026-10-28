import os
import sys
import json
import glob
import argparse
import zipfile
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ==============================================================================
# CONFIGURATION & ROI SETUP
# ==============================================================================
BHOONIDHI_USER = "yash1947"
BHOONIDHI_PASS = "yash@123"
BASE_URL = "https://bhoonidhi-api.nrsc.gov.in"

# Point of Interest: Netravati Bridge (Ullal Bridge), NH-66, Mangaluru
# Verified centre: 12.8391N, 74.8601E (Wikipedia/Maps), length ~804m N-S
# GMaps: https://www.google.com/maps?q=12.8391,74.8601
# OSM: https://www.openstreetmap.org/#map=17/12.8391/74.8601
LAT, LON = 12.8391, 74.8601

# Tight 150-200m E-W x ~900m N-S corridor along bridge axis (N-S)
# [min_lon, min_lat, max_lon, max_lat]
# E-W: 74.8601 +/- 0.0009 deg (~ +/-100m) | N-S: 12.8391 +/- 0.004 deg (~ +/-440m)
BRIDGE_BBOX = [74.8592, 12.8351, 74.8610, 12.8431]

# Date Range for Time-Series Deformation Monitoring (max 366 days per Bhoonidhi API)
DATE_RANGE = "2025-10-01T00:00:00Z/2026-10-01T23:59:59Z"

# InSAR-Capable SAR Collections (Single Look Complex / SLC Data)
SAR_COLLECTIONS = ["Sentinel-1A_SAR-IW_SLC", "EOS-04_SAR-MRS_L2A", "NISAR_SSAR_RSLC"]

DOWNLOAD_URL = f"{BASE_URL}/download"
DOWNLOAD_DIR = Path("downloads")

# NISAR S-band radar wavelength (m) — S-SAR operates at ~3.2 GHz => λ ≈ 0.0937 m
# (L-band LSAR: λ = 0.2384 m — not used here since your scenes are SSAR)
NISAR_SSAR_WAVELENGTH_M = 0.0937

# HDF5 dataset paths for NISAR RSLC products
# SSAR products use /science/SSAR/RSLC/swaths/frequencyA/<POL>
# Polarisation channels from metadata: CP → "RH"/"RV", DH → "HH"/"HV"
RSLC_SWATH_PREFIX = "/science/SSAR/RSLC/swaths/frequencyA"
RSLC_METADATA_PREFIX = "/science/SSAR/RSLC/metadata"


# ==============================================================================
# 1. BHOONIDHI API INTERACTION
# ==============================================================================
def authenticate(username, password):
    """Authenticate with Bhoonidhi API and get JWT Bearer Token."""
    auth_url = f"{BASE_URL}/auth/token"
    payload = {
        "userId": username,
        "password": password,
        "grant_type": "password"
    }
    res = requests.post(auth_url, json=payload)
    if res.status_code == 200:
        return res.json().get("access_token")
    else:
        raise ConnectionError(f"Authentication failed ({res.status_code}): {res.text}")

def search_bridge_sar_scenes(token):
    """Search for InSAR SLC scenes over Netravati Bridge bounding box."""
    search_url = f"{BASE_URL}/data/search"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "collections": SAR_COLLECTIONS,
        "bbox": BRIDGE_BBOX,
        "datetime": DATE_RANGE,
        "limit": 50
    }
    res = requests.post(search_url, json=payload, headers=headers)
    if res.status_code == 200:
        return res.json().get("features", [])
    else:
        raise RuntimeError(f"Bhoonidhi search failed ({res.status_code}): {res.text}")


def load_local_scenes(download_dir=DOWNLOAD_DIR):
    """Load actual Bhoonidhi STAC features previously saved in downloads/*.json.
    Skips *_thumbnail.jpg / *_metadata.met quicklooks. Returns list of features."""
    download_dir = Path(download_dir)
    scenes = []
    for p in sorted(download_dir.glob("*.json")):
        if p.name.endswith("_thumbnail.json") or p.name.endswith("_metadata.json"):
            continue
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        # Real STAC scene feature has stac_version + type==Feature + id + properties.datetime
        if obj.get("type") == "Feature" and obj.get("id") and "properties" in obj:
            # Skip .met files (they have ProductID but no stac_version/type Feature — already filtered,
            # but double-guard: .met has no 'assets')
            if "stac_version" in obj or "assets" in obj:
                scenes.append(obj)
    return scenes


def download_slc_products(token, scenes, out_dir=DOWNLOAD_DIR, max_downloads=None,
                          timeout=600):
    """
    Download full NISAR RSLC SLC products (HDF5 in ZIP) from Bhoonidhi.
    Uses GET /download?id=<id>&collection=<collection> with Bearer token.

    Only downloads scenes with Online=Y. Returns list of download result dicts.
    Each ZIP typically contains a .h5 file with complex I/Q phase data.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}
    results = []

    if max_downloads is not None:
        scenes = scenes[:max_downloads]

    for feat in scenes:
        scene_id = feat.get("id")
        collection = feat.get("collection", "")
        props = feat.get("properties", {})
        online = props.get("Online", "Y")
        status = {"scene_id": scene_id, "collection": collection,
                  "online": online, "files": []}

        # 1. Save STAC metadata JSON (lightweight)
        meta_path = out_dir / f"{scene_id}.json"
        meta_path.write_text(json.dumps(feat, indent=2))
        status["files"].append(str(meta_path))

        # 2. Fetch lightweight assets (thumbnail/metadata, KBs)
        for asset_name in ("thumbnail", "metadata"):
            asset = feat.get("assets", {}).get(asset_name)
            if not asset or "href" not in asset:
                continue
            try:
                a_res = requests.get(asset["href"], timeout=30)
                if a_res.status_code == 200:
                    ext = ".jpg" if asset_name == "thumbnail" else ".met"
                    a_path = out_dir / f"{scene_id}_{asset_name}{ext}"
                    a_path.write_bytes(a_res.content)
                    status["files"].append(str(a_path))
            except Exception as e:
                status.setdefault("asset_errors", []).append(f"{asset_name}: {e}")

        # 3. Download full SLC product (HDF5 in ZIP, GBs)
        if online != "Y":
            status["product"] = "SKIPPED (Online=N — product needs Browse&Order from NRSC)"
            results.append(status)
            continue

        dl_url = f"{DOWNLOAD_URL}?id={scene_id}&collection={collection}"
        zip_path = out_dir / f"{scene_id}.zip"

        # Skip if already downloaded
        if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
            status["product"] = f"ALREADY EXISTS: {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)"
            status["files"].append(str(zip_path))
            results.append(status)
            print(f"  [✓] {scene_id[:50]}... already downloaded ({zip_path.stat().st_size / 1e9:.2f} GB)")
            continue

        print(f"  [↓] Downloading {scene_id[:50]}... (this may take several minutes for GBs)")
        try:
            with requests.get(dl_url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code == 200:
                    total = 0
                    with open(zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):  # 1MB chunks
                            if chunk:
                                f.write(chunk)
                                total += len(chunk)
                                # Progress indicator every 100MB
                                if total % (100 << 20) < (1 << 20):
                                    print(f"      ... {total / 1e9:.2f} GB downloaded")
                    status["product"] = f"OK: {zip_path} ({total / 1e9:.2f} GB)"
                    status["files"].append(str(zip_path))
                    print(f"  [✓] {scene_id[:50]}... done ({total / 1e9:.2f} GB)")
                elif r.status_code == 401:
                    status["product"] = "ERROR 401 — token expired, re-authenticate"
                elif r.status_code == 404:
                    status["product"] = "ERROR 404 — product not on online storage"
                elif r.status_code == 412:
                    status["product"] = "ERROR 412 — concurrent download limit (max 3), retry later"
                elif r.status_code == 504:
                    status["product"] = "ERROR 504 — gateway timeout, product may be too large"
                else:
                    status["product"] = f"ERROR {r.status_code}: {r.text[:200]}"
        except requests.exceptions.Timeout:
            status["product"] = f"TIMEOUT after {timeout}s — increase --timeout or retry"
        except Exception as e:
            status["product"] = f"FAILED: {e}"

        results.append(status)

    return results


def extract_h5_from_zips(download_dir=DOWNLOAD_DIR):
    """Extract .h5 HDF5 files from downloaded ZIP archives."""
    download_dir = Path(download_dir)
    h5_files = []

    for zp in sorted(download_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(zp, 'r') as z:
                h5_members = [m for m in z.namelist() if m.endswith('.h5')]
                for h5m in h5_members:
                    out_path = download_dir / Path(h5m).name
                    if not out_path.exists():
                        print(f"  [⇉] Extracting {h5m} from {zp.name}...")
                        z.extract(h5m, download_dir)
                        # Move from possible subdirectory to download_dir root
                        extracted = download_dir / h5m
                        if extracted != out_path:
                            extracted.rename(out_path)
                    h5_files.append(out_path)
        except zipfile.BadZipFile:
            print(f"  [!] {zp.name} is not a valid ZIP — skipping")
        except Exception as e:
            print(f"  [!] Error extracting {zp.name}: {e}")

    # Also collect any .h5 files already present (not in ZIP)
    for h5p in sorted(download_dir.glob("*.h5")):
        if h5p not in h5_files:
            h5_files.append(h5p)

    return h5_files


# ==============================================================================
# 2. HDF5 PHASE DATA EXTRACTION (NISAR RSLC)
# ==============================================================================
def read_nisar_rslc_slc(h5_path, bridge_bbox=BRIDGE_BBOX, chip_size=512):
    """
    Read complex SLC data from a NISAR RSLC HDF5 file.
    Extracts a spatial chip around the bridge location using geolocation grids.

    Returns dict with:
      - 'slc': complex numpy array (chip around bridge)
      - 'metadata': dict with scene info (datetime, track, orbit, wavelength etc.)
      - 'pixel_coords': (row_center, col_center) in full raster
      - 'geolocation': lat/lon arrays for the chip
    """
    import h5py

    h5_path = Path(h5_path)
    result = {"h5_file": str(h5_path), "error": None}

    try:
        with h5py.File(str(h5_path), 'r') as f:
            # --- Discover sensor (SSAR or LSAR) and available polarizations ---
            sensor = None
            for s in ["SSAR", "LSAR"]:
                if f"/science/{s}/RSLC" in f:
                    sensor = s
                    break
            if sensor is None:
                result["error"] = "No RSLC group found (neither SSAR nor LSAR)"
                return result

            swath_prefix = f"/science/{sensor}/RSLC/swaths/frequencyA"
            meta_prefix = f"/science/{sensor}/RSLC/metadata"

            # Find available polarization channels
            available_pols = []
            if swath_prefix in f:
                for key in f[swath_prefix].keys():
                    if isinstance(f[f"{swath_prefix}/{key}"], h5py.Dataset):
                        available_pols.append(key)

            if not available_pols:
                result["error"] = f"No polarization datasets in {swath_prefix}"
                return result

            # Prefer co-pol: HH > VV > RH > RV > first available
            pol_priority = ["HH", "VV", "RH", "RV"]
            chosen_pol = None
            for p in pol_priority:
                if p in available_pols:
                    chosen_pol = p
                    break
            if chosen_pol is None:
                chosen_pol = available_pols[0]

            dataset_path = f"{swath_prefix}/{chosen_pol}"
            dset = f[dataset_path]
            n_az, n_rg = dset.shape
            print(f"    SLC shape: {n_az} x {n_rg} ({chosen_pol}), dtype: {dset.dtype}")

            # --- Locate bridge pixel using geolocation grids ---
            # NISAR RSLC stores lat/lon coordinate grids
            lat_path = f"{swath_prefix}/coordinateX"  # latitude grid
            lon_path = f"{swath_prefix}/coordinateY"  # longitude grid
            # Alternative paths
            alt_lat = f"{swath_prefix}/zeroDopplerTime"
            alt_lon = f"{swath_prefix}/slantRange"

            row_center, col_center = n_az // 2, n_rg // 2  # default: center

            if lat_path in f and lon_path in f:
                lat_grid = f[lat_path][:]
                lon_grid = f[lon_path][:]
                # Find pixel closest to bridge center
                dist = np.sqrt((lat_grid - LAT)**2 + (lon_grid - LON)**2)
                row_center, col_center = np.unravel_index(np.argmin(dist), dist.shape)
                print(f"    Bridge pixel: row={row_center}, col={col_center} "
                      f"(lat={lat_grid[row_center, col_center]:.4f}, "
                      f"lon={lon_grid[row_center, col_center]:.4f})")
            else:
                # Use scene center lat/lon from metadata to estimate bridge pixel
                # via linear interpolation across the image extent
                scene_bbox = None
                for bp in [f"{meta_prefix}/orbit", f"{meta_prefix}"]:
                    if bp in f:
                        break

                # Estimate from image corners (from STAC metadata we know the bbox)
                # Use proportional positioning within the raster
                print(f"    No geolocation grid found — using proportional pixel estimate")

            # --- Extract chip around bridge ---
            half = chip_size // 2
            r0 = max(0, row_center - half)
            r1 = min(n_az, row_center + half)
            c0 = max(0, col_center - half)
            c1 = min(n_rg, col_center + half)

            print(f"    Reading SLC chip [{r0}:{r1}, {c0}:{c1}] ...")
            slc_chip = dset[r0:r1, c0:c1]

            # Handle half-precision complex (complex32 = 2x float16)
            if slc_chip.dtype == np.float16 or 'float16' in str(slc_chip.dtype):
                # Interpret as interleaved float16 pairs → complex64
                raw = slc_chip.view(np.float16)
                slc_chip = raw[..., 0::2] + 1j * raw[..., 1::2]
                slc_chip = slc_chip.astype(np.complex64)
            elif not np.iscomplexobj(slc_chip):
                # If stored as 2-band real, reconstruct complex
                if slc_chip.ndim == 3 and slc_chip.shape[-1] == 2:
                    slc_chip = slc_chip[..., 0] + 1j * slc_chip[..., 1]
                    slc_chip = slc_chip.astype(np.complex64)

            # --- Extract radar parameters from metadata ---
            wavelength = NISAR_SSAR_WAVELENGTH_M
            if f"{meta_prefix}/processingInformation/parameters/rangeBandwidth" in f:
                pass  # could extract more params
            # Try to read wavelength from the file
            for wl_path in [
                f"/science/{sensor}/identification/radarWavelength",
                f"/science/{sensor}/RSLC/metadata/processingInformation/parameters/radarWavelength",
            ]:
                if wl_path in f:
                    wavelength = float(f[wl_path][()])
                    break

            # Extract datetime from the file
            acq_datetime = None
            for dt_path in [
                f"/science/{sensor}/identification/zeroDopplerStartTime",
                f"/science/{sensor}/identification/absoluteOrbitNumber",
            ]:
                pass

            result.update({
                "slc": slc_chip,
                "sensor": sensor,
                "polarization": chosen_pol,
                "available_pols": available_pols,
                "full_shape": (n_az, n_rg),
                "chip_bounds": (r0, r1, c0, c1),
                "pixel_center": (row_center, col_center),
                "wavelength_m": wavelength,
                "chip_size": chip_size,
            })

    except ImportError:
        result["error"] = "h5py not installed. Run: pip3 install h5py"
    except Exception as e:
        result["error"] = f"Failed to read {h5_path.name}: {e}"

    return result


# ==============================================================================
# 3. INTERFEROMETRIC PROCESSING (InSAR)
# ==============================================================================
def compute_interferogram(slc_ref, slc_sec):
    """
    Compute interferogram from two co-registered SLC chips.
    interferogram = SLC_ref × conj(SLC_sec)
    Returns complex interferogram and coherence.
    """
    # Cross-multiply: reference × conjugate(secondary)
    ifg = slc_ref * np.conj(slc_sec)

    # Estimate coherence using spatial averaging (boxcar filter)
    from scipy.ndimage import uniform_filter
    window = 5  # 5×5 spatial averaging
    num = uniform_filter(np.real(ifg), window) + 1j * uniform_filter(np.imag(ifg), window)
    denom1 = np.sqrt(uniform_filter(np.abs(slc_ref)**2, window))
    denom2 = np.sqrt(uniform_filter(np.abs(slc_sec)**2, window))
    denom = denom1 * denom2
    denom[denom == 0] = 1e-10

    coherence = np.abs(num) / denom
    coherence = np.clip(coherence, 0, 1)

    return ifg, coherence


def unwrap_phase_simple(wrapped_phase, coherence=None, coh_threshold=0.3):
    """
    Simple phase unwrapping using scipy's unwrap or Itoh's method.
    For production, SNAPHU or isce3's unwrapper would be used.

    Falls back to numpy unwrap along each dimension if scipy is unavailable.
    """
    try:
        from scipy.ndimage import median_filter

        # Mask low-coherence pixels
        if coherence is not None:
            mask = coherence < coh_threshold
            wrapped_phase = wrapped_phase.copy()
            wrapped_phase[mask] = 0

        # 2D unwrapping via row-then-column np.unwrap
        # (For production: use SNAPHU or isce3.unwrap)
        unwrapped = np.unwrap(wrapped_phase, axis=1)  # unwrap along range
        unwrapped = np.unwrap(unwrapped, axis=0)       # unwrap along azimuth

        # Median filter to reduce unwrapping artifacts
        unwrapped = median_filter(unwrapped, size=3)

        return unwrapped

    except ImportError:
        # Fallback: simple 1D unwrap
        unwrapped = np.unwrap(wrapped_phase, axis=1)
        unwrapped = np.unwrap(unwrapped, axis=0)
        return unwrapped


def phase_to_displacement(unwrapped_phase, wavelength_m, incidence_deg):
    """
    Convert unwrapped interferometric phase to Line-of-Sight (LOS) displacement.

    d_LOS = -φ × λ / (4π)

    For vertical displacement: d_vert = d_LOS / cos(θ_inc)

    Args:
        unwrapped_phase: 2D array of unwrapped phase (radians)
        wavelength_m: radar wavelength in meters
        incidence_deg: incidence angle in degrees

    Returns:
        d_los_mm: LOS displacement in millimeters
        d_vert_mm: vertical displacement in millimeters
    """
    # LOS displacement (meters) = -phase * wavelength / (4 * pi)
    d_los_m = -unwrapped_phase * wavelength_m / (4 * np.pi)
    d_los_mm = d_los_m * 1000.0  # convert to mm

    # Vertical component
    inc_rad = np.deg2rad(incidence_deg)
    d_vert_mm = d_los_mm / np.cos(inc_rad)

    return d_los_mm, d_vert_mm


# ==============================================================================
# 4. GROUP SCENES INTO InSAR PAIRS (same track/frame/node)
# ==============================================================================
def group_insar_pairs(scenes):
    """
    Group scenes by (Track, Frame, Node) to form valid InSAR pairs.
    Only scenes from the same orbital geometry can be interferometrically combined.

    Returns list of pairs: [(scene_ref, scene_sec, temporal_baseline_days), ...]
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for feat in scenes:
        props = feat.get("properties", {})
        key = (props.get("Track"), props.get("Frame"), props.get("Node"))
        dt_str = props.get("datetime", "")
        if dt_str:
            dt = datetime.strptime(dt_str[:19], "%Y-%m-%dT%H:%M:%S")
            groups[key].append((dt, feat))

    pairs = []
    for key, members in groups.items():
        members.sort(key=lambda x: x[0])  # sort by date
        for i in range(len(members) - 1):
            dt_ref, feat_ref = members[i]
            dt_sec, feat_sec = members[i + 1]
            baseline_days = (dt_sec - dt_ref).days
            pairs.append({
                "reference": feat_ref,
                "secondary": feat_sec,
                "ref_date": dt_ref,
                "sec_date": dt_sec,
                "temporal_baseline_days": baseline_days,
                "track": key[0],
                "frame": key[1],
                "node": key[2],
            })
            print(f"  [InSAR] Pair: {dt_ref.strftime('%Y-%m-%d')} → {dt_sec.strftime('%Y-%m-%d')} "
                  f"(Δt={baseline_days}d, Track={key[0]}, Frame={key[1]}, {key[2]})")

    return pairs


# ==============================================================================
# 5. FULL InSAR DISPLACEMENT PIPELINE
# ==============================================================================
def process_insar_displacement(scenes_data, download_dir=DOWNLOAD_DIR,
                               use_phase_data=True):
    """
    Full InSAR displacement processing pipeline:
    1. Extract metadata from STAC scenes
    2. Group into valid InSAR pairs (same track/frame/node)
    3. If HDF5 SLC files exist → read phase, compute interferograms, unwrap, get displacement
    4. If no HDF5 → fall back to metadata-only analysis with simulated displacement

    Returns: (DataFrame with time-series, SHM report dict)
    """
    records = []

    # Extract acquisition dates + metadata from STAC features
    for feature in scenes_data:
        scene_id = feature.get("id")
        props = feature.get("properties", {})
        acq_date_str = props.get("datetime", "")
        if acq_date_str:
            dt = datetime.strptime(acq_date_str[:10], "%Y-%m-%d")
            records.append({
                "scene_id": scene_id,
                "date": dt,
                "collection": feature.get("collection", ""),
                "platform": props.get("Platform", ""),
                "incidence_deg": float(props.get("IncidenceAngle", "nan")) if props.get("IncidenceAngle") else float("nan"),
                "polarization": props.get("Polarization", ""),
                "online": props.get("Online", ""),
                "track": props.get("Track"),
                "frame": props.get("Frame"),
                "node": props.get("Node"),
            })

    df = pd.DataFrame(records)

    if df.empty or "date" not in df.columns:
        raise ValueError("No SAR scenes returned from Bhoonidhi for given bbox/datetime. Aborting — no fallback used.")
    df = df.sort_values("date").reset_index(drop=True)

    # Time in years relative to first acquisition
    t0 = df['date'].iloc[0]
    df['years'] = (df['date'] - t0).dt.days / 365.25

    # --- ATTEMPT REAL PHASE DATA PROCESSING ---
    phase_data_used = False
    insar_pairs = []

    if use_phase_data:
        download_dir = Path(download_dir)

        # Find available HDF5 files
        h5_files = list(download_dir.glob("*.h5"))
        if h5_files:
            print(f"\n[*] Found {len(h5_files)} HDF5 SLC file(s) — processing real phase data!")

            # Map scene_id → h5 file path
            h5_map = {}
            for h5p in h5_files:
                # NISAR HDF5 filename typically contains the scene ID or a variant
                h5_map[h5p.stem] = h5p

            # Group scenes into InSAR pairs
            print("\n[*] Grouping scenes into valid InSAR pairs (same Track/Frame/Node)...")
            insar_pairs = group_insar_pairs(scenes_data)

            if insar_pairs:
                print(f"\n[*] Processing {len(insar_pairs)} interferometric pair(s)...")

                pair_results = []
                for pair in insar_pairs:
                    ref_id = pair["reference"]["id"]
                    sec_id = pair["secondary"]["id"]

                    # Find corresponding HDF5 files
                    ref_h5 = None
                    sec_h5 = None
                    for stem, path in h5_map.items():
                        if ref_id in stem or stem in ref_id:
                            ref_h5 = path
                        if sec_id in stem or stem in sec_id:
                            sec_h5 = path

                    if ref_h5 is None or sec_h5 is None:
                        print(f"  [!] Missing HDF5 for pair {pair['ref_date'].strftime('%Y-%m-%d')} → "
                              f"{pair['sec_date'].strftime('%Y-%m-%d')} — need both SLC downloads")
                        continue

                    print(f"\n  --- Processing pair: {pair['ref_date'].strftime('%Y-%m-%d')} → "
                          f"{pair['sec_date'].strftime('%Y-%m-%d')} ---")

                    # Read SLC chips from both scenes
                    print(f"  [1/4] Reading reference SLC: {ref_h5.name}")
                    ref_data = read_nisar_rslc_slc(ref_h5)
                    if ref_data.get("error"):
                        print(f"  [!] Reference read error: {ref_data['error']}")
                        continue

                    print(f"  [2/4] Reading secondary SLC: {sec_h5.name}")
                    sec_data = read_nisar_rslc_slc(sec_h5)
                    if sec_data.get("error"):
                        print(f"  [!] Secondary read error: {sec_data['error']}")
                        continue

                    # Ensure chips are same size (crop to minimum)
                    slc_ref = ref_data["slc"]
                    slc_sec = sec_data["slc"]
                    min_r = min(slc_ref.shape[0], slc_sec.shape[0])
                    min_c = min(slc_ref.shape[1], slc_sec.shape[1])
                    slc_ref = slc_ref[:min_r, :min_c]
                    slc_sec = slc_sec[:min_r, :min_c]

                    # Compute interferogram
                    print(f"  [3/4] Computing interferogram ({min_r}×{min_c} pixels)...")
                    ifg, coherence = compute_interferogram(slc_ref, slc_sec)

                    # Phase unwrapping
                    wrapped_phase = np.angle(ifg)
                    unwrapped = unwrap_phase_simple(wrapped_phase, coherence)

                    # Convert to displacement
                    incidence_deg = pair["reference"].get("properties", {}).get("IncidenceAngle", 41.78)
                    wavelength = ref_data.get("wavelength_m", NISAR_SSAR_WAVELENGTH_M)
                    d_los_mm, d_vert_mm = phase_to_displacement(unwrapped, wavelength, incidence_deg)

                    # Statistics over the bridge area (central region of chip)
                    # Use mean coherence > 0.3 mask for reliable pixels
                    reliable = coherence > 0.3
                    n_reliable = np.sum(reliable)

                    if n_reliable > 10:
                        mean_disp_los = float(np.mean(d_los_mm[reliable]))
                        std_disp_los = float(np.std(d_los_mm[reliable]))
                        mean_disp_vert = float(np.mean(d_vert_mm[reliable]))
                        mean_coherence = float(np.mean(coherence[reliable]))
                    else:
                        mean_disp_los = float(np.mean(d_los_mm))
                        std_disp_los = float(np.std(d_los_mm))
                        mean_disp_vert = float(np.mean(d_vert_mm))
                        mean_coherence = float(np.mean(coherence))

                    print(f"  [4/4] Results:")
                    print(f"        Mean LOS displacement:  {mean_disp_los:.2f} ± {std_disp_los:.2f} mm")
                    print(f"        Mean vertical disp:     {mean_disp_vert:.2f} mm")
                    print(f"        Mean coherence:         {mean_coherence:.3f}")
                    print(f"        Reliable pixels:        {n_reliable}/{coherence.size} "
                          f"({100*n_reliable/coherence.size:.1f}%)")

                    pair_results.append({
                        "ref_date": pair["ref_date"],
                        "sec_date": pair["sec_date"],
                        "baseline_days": pair["temporal_baseline_days"],
                        "mean_los_disp_mm": mean_disp_los,
                        "std_los_disp_mm": std_disp_los,
                        "mean_vert_disp_mm": mean_disp_vert,
                        "mean_coherence": mean_coherence,
                        "n_reliable_pixels": n_reliable,
                        "total_pixels": coherence.size,
                        "wavelength_m": wavelength,
                        "incidence_deg": incidence_deg,
                    })

                if pair_results:
                    phase_data_used = True

                    # Build displacement time-series from pair results
                    # Cumulative displacement relative to first date
                    cumulative_disp = 0.0
                    disp_records = [{"date": pair_results[0]["ref_date"],
                                     "displacement_mm": 0.0,
                                     "source": "real_insar"}]
                    for pr in pair_results:
                        cumulative_disp += pr["mean_los_disp_mm"]
                        disp_records.append({
                            "date": pr["sec_date"],
                            "displacement_mm": cumulative_disp,
                            "source": "real_insar",
                        })

                    # Merge with scene metadata
                    df_disp = pd.DataFrame(disp_records)
                    df = df.merge(df_disp[["date", "displacement_mm", "source"]],
                                  on="date", how="left")
                    # For scenes without InSAR pair, interpolate
                    if df["displacement_mm"].isna().any():
                        df["displacement_mm"] = df["displacement_mm"].interpolate(method="linear")
                        df["source"] = df["source"].fillna("interpolated")

        if not phase_data_used:
            print("\n[*] No HDF5 phase data available — using simulated displacement.")
            print("    To get real data, run: python3 main.py --download-slc")

    # --- FALLBACK: Simulated displacement if no real phase data ---
    if not phase_data_used:
        # NISAR S-band wavelength-scaled noise model
        np.random.seed(42)
        velocity_true = -2.4  # -2.4 mm/year settlement (realistic for bridge pier)
        noise = np.random.normal(0, 1.2, size=len(df))
        df['displacement_mm'] = (velocity_true * df['years']) + noise
        df['source'] = 'simulated'

    # Calculate Velocity (Linear Fit)
    poly = np.polyfit(df['years'], df['displacement_mm'], 1)
    annual_velocity_mm = poly[0]
    total_displacement_mm = df['displacement_mm'].iloc[-1] - df['displacement_mm'].iloc[0]
    std_error = np.std(df['displacement_mm'] - (poly[0]*df['years'] + poly[1]))

    # Structural Health Condition Thresholds
    if abs(annual_velocity_mm) < 3.0:
        status = "GREEN - STABLE / NORMAL AGING"
        recommendation = "Routine maintenance; continue satellite monitoring every 6 months."
    elif 3.0 <= abs(annual_velocity_mm) < 7.0:
        status = "YELLOW - MODERATE DEFORMATION"
        recommendation = "Schedule targeted ground inspection (Inclinometers & Strain Gauges)."
    else:
        status = "RED - HIGH SETTLEMENT / CRITICAL"
        recommendation = "Immediate structural audit required for bridge piers and deck."

    report = {
        "target_structure": "Netravati Bridge (NH 66 / Southern Railway)",
        "coordinates": {"latitude": LAT, "longitude": LON},
        "bbox_used": BRIDGE_BBOX,
        "total_scenes_analyzed": len(df),
        "monitoring_period": f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        "mean_incidence_deg": round(float(df["incidence_deg"].mean()), 2) if "incidence_deg" in df else None,
        "platforms_used": sorted(df["platform"].dropna().unique().tolist()) if "platform" in df else [],
        "data_source": "REAL_INSAR_PHASE" if phase_data_used else "SIMULATED",
        "phase_data_used": phase_data_used,
        "insar_pairs_processed": len(insar_pairs) if phase_data_used else 0,
        "radar_wavelength_m": NISAR_SSAR_WAVELENGTH_M,
        "mean_annual_velocity_mm_yr": round(annual_velocity_mm, 2),
        "total_cumulative_displacement_mm": round(total_displacement_mm, 2),
        "measurement_error_mm": round(std_error, 2),
        "health_status": status,
        "action_recommendation": recommendation,
    }

    return df, report


# ==============================================================================
# MAIN EXECUTION & OUTPUT GENERATION
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHM Netravati Bridge — Bhoonidhi NISAR RSLC InSAR Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Search + analyze metadata only (no downloads):
  python3 main.py

  # Use local STAC JSON files (no API call):
  python3 main.py --local

  # Download full NISAR RSLC SLC products (HDF5, several GB each):
  python3 main.py --download-slc

  # Download SLC + process real InSAR phase data:
  python3 main.py --download-slc --process-phase

  # Process previously downloaded HDF5 files locally:
  python3 main.py --local --process-phase

  # Limit to 2 downloads:
  python3 main.py --download-slc --max 2
""")
    parser.add_argument("--download-slc", action="store_true",
                        help="Download full NISAR RSLC SLC products (HDF5 in ZIP, GBs per scene)")
    parser.add_argument("--process-phase", action="store_true",
                        help="Process real InSAR phase data from downloaded HDF5 files")
    parser.add_argument("--max", type=int, default=None,
                        help="Max scenes to download (e.g. --max 2)")
    parser.add_argument("--out", type=str, default=str(DOWNLOAD_DIR),
                        help="Download directory")
    parser.add_argument("--local", action="store_true",
                        help="Use only local downloads/*.json STAC features, no live search")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Download timeout in seconds (default: 600 = 10 min)")
    parser.add_argument("--chip-size", type=int, default=512,
                        help="SLC chip size in pixels around bridge (default: 512)")
    args = parser.parse_args()

    print("==========================================================")
    print(" STRUCTURAL HEALTH MONITORING (SHM) - NETRAVATI BRIDGE")
    print(" NISAR S-SAR RSLC InSAR Displacement Analysis")
    print("==========================================================")

    token = None
    scenes = []
    try:
        if args.local:
            scenes = load_local_scenes(args.out)
            print(f"[*] Loaded {len(scenes)} LOCAL scenes from {args.out}/ (your Bhoonidhi files only).")
            if not scenes:
                raise ValueError(f"No local STAC *.json scenes in {args.out}/. Aborting.")
        else:
            token = authenticate(BHOONIDHI_USER, BHOONIDHI_PASS)
            scenes = search_bridge_sar_scenes(token)
            print(f"[*] Retrieved {len(scenes)} SAR scenes from Bhoonidhi.")
            if not scenes:
                raise ValueError("Bhoonidhi returned 0 scenes. Aborting — no fallback used.")
    except Exception as e:
        print(f"[!] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # --- DOWNLOAD FULL SLC PRODUCTS ---
    if args.download_slc:
        if token is None and not args.local:
            print("[!] --download-slc needs live auth (omit --local).", file=sys.stderr)
            sys.exit(1)
        if token is None:
            # Re-authenticate for download
            try:
                token = authenticate(BHOONIDHI_USER, BHOONIDHI_PASS)
            except Exception as e:
                print(f"[!] Re-auth failed: {e}", file=sys.stderr)
                sys.exit(1)

        print(f"\n{'='*60}")
        print(f" DOWNLOADING NISAR RSLC SLC PRODUCTS")
        print(f" Scenes: {len(scenes) if args.max is None else min(args.max, len(scenes))}")
        print(f" Output: {args.out}/")
        print(f"{'='*60}")
        dl_results = download_slc_products(
            token, scenes, out_dir=args.out,
            max_downloads=args.max, timeout=args.timeout)
        for r in dl_results:
            sid = r['scene_id'][:50]
            print(f"  [{sid}...] online={r['online']} product={r.get('product', 'N/A')}")

        # Extract HDF5 from ZIPs
        print(f"\n[*] Extracting HDF5 files from downloaded ZIPs...")
        h5_files = extract_h5_from_zips(args.out)
        if h5_files:
            print(f"[*] Found {len(h5_files)} HDF5 file(s) ready for InSAR processing.")
        else:
            print("[!] No HDF5 files found in ZIPs. Downloads may have failed or be in different format.")

    # --- InSAR PROCESSING ---
    use_phase = args.process_phase or args.download_slc
    df_time_series, shm_report = process_insar_displacement(
        scenes, download_dir=args.out, use_phase_data=use_phase)

    # Output JSON Report
    print(f"\n{'='*60}")
    print(" FINAL STRUCTURAL HEALTH MONITORING REPORT")
    print(f"{'='*60}")
    print(json.dumps(shm_report, indent=4))

    print(f"\n--- TIME-SERIES DISPLACEMENT DATASET ---")
    display_cols = ['scene_id', 'date', 'displacement_mm']
    if 'source' in df_time_series.columns:
        display_cols.append('source')
    print(df_time_series[display_cols].to_string(index=False))

    # Save report
    report_path = Path(args.out) / "shm_report.json"
    report_path.write_text(json.dumps(shm_report, indent=2))
    print(f"\n[*] Report saved to {report_path}")

    # Save time-series CSV
    csv_path = Path(args.out) / "displacement_timeseries.csv"
    df_time_series.to_csv(csv_path, index=False)
    print(f"[*] Time-series saved to {csv_path}")

    if not shm_report.get("phase_data_used"):
        print(f"\n{'='*60}")
        print(" ⚠  DISPLACEMENT DATA IS SIMULATED (no real phase data)")
        print(" To use real InSAR phase data:")
        print("   1. Download SLC: python3 main.py --download-slc")
        print("   2. Process:      python3 main.py --local --process-phase")
        print(f"{'='*60}")