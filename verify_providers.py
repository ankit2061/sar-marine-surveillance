#!/usr/bin/env python3
"""
Diagnostic & Verification Script for SAR Satellite Providers
============================================================
Checks connectivity and authorization status for:
1. Microsoft Planetary Computer STAC
2. Copernicus Data Space Ecosystem (CDSE)
3. Google Earth Engine (GEE)
"""

import os
from dotenv import load_dotenv

load_dotenv()


def check_planetary_computer():
    print("\n[1/3] Testing Microsoft Planetary Computer STAC...")
    try:
        from sar_live_fetcher import PlanetaryComputerSARFetcher
        fetcher = PlanetaryComputerSARFetcher()
        scenes = fetcher.search_scenes(
            bbox=[72.2, 19.4, 72.7, 19.8],
            start_date="2024-03-01",
            end_date="2024-03-10",
            max_items=1,
        )
        if scenes:
            print("  ✅ SUCCESS: Planetary Computer STAC is active and accessible (No login required).")
            print(f"     Sample scene found: {scenes[0]['id']}")
            return True
        else:
            print("  ⚠️ Notice: Query returned 0 scenes in date range.")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
    return False


def check_cdse():
    print("\n[2/3] Testing Copernicus Data Space Ecosystem (CDSE)...")
    username = os.environ.get("CDSE_USERNAME")
    password = os.environ.get("CDSE_PASSWORD")

    from sar_live_fetcher import CopernicusCDSEFetcher
    fetcher = CopernicusCDSEFetcher(username=username, password=password)

    # 1. Test public OData catalogue search
    products = fetcher.search_products(
        bbox=[72.2, 19.4, 72.7, 19.8],
        start_date="2024-03-01",
        end_date="2024-03-10",
        max_items=1,
    )
    if products:
        print(f"  ✅ OData Catalogue Search: Connected! (Found {len(products)} products).")
    else:
        print("  ⚠️ OData Catalogue Search: No products returned.")

    # 2. Test user authentication
    if username and password:
        if fetcher.authenticate():
            print("  ✅ User Authentication: CDSE OAuth2 Access Token successfully issued!")
            return True
        else:
            print("  ❌ User Authentication: CDSE login failed. Check CDSE_USERNAME and CDSE_PASSWORD.")
    else:
        print("  ℹ️ User Authentication: Credentials not set in .env or environment.")
        print("     To set up, register for free at https://dataspace.copernicus.eu/ and add:")
        print("     CDSE_USERNAME=your_email\n     CDSE_PASSWORD=your_password to .env")
    return False


def check_gee():
    print("\n[3/3] Testing Google Earth Engine (GEE)...")
    project_id = os.environ.get("EE_PROJECT_ID", "multi-domain-chatbot")
    try:
        import ee
        try:
            ee.Initialize(project=project_id)
            print(f"  ✅ SUCCESS: Earth Engine successfully initialized with project '{project_id}'!")
            return True
        except Exception as e:
            print(f"  ⚠️ GEE Authorization needed: {e}")
            print("     To authorize, run in your terminal:")
            print("     earthengine authenticate")
    except ImportError:
        print("  ❌ ERROR: 'earthengine-api' package not found.")
    return False


def main():
    print("=" * 65)
    print("      SAR SATELLITE PROVIDER AUTHENTICATION STATUS CHECK")
    print("=" * 65)

    pc_ok = check_planetary_computer()
    cdse_ok = check_cdse()
    gee_ok = check_gee()

    print("\n" + "=" * 65)
    print("SUMMARY:")
    print(f"  * Planetary Computer STAC : {'🟢 READY' if pc_ok else '🔴 FAILED'}")
    print(f"  * Copernicus CDSE        : {'🟢 READY' if cdse_ok else '🟡 CATALOGUE OK (LOGIN NEEDED)'}")
    print(f"  * Google Earth Engine    : {'🟢 READY' if gee_ok else '🟡 AUTH NEEDED'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
