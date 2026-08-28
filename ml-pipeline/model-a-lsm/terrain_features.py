"""
terrain_features.py
Extracts slope, aspect, and plan curvature from a DEM GeoTIFF.

Usage:
    python terrain_features.py --dem data/dem_east_khasi_hills.tif --outdir data/features
"""
import argparse
import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def reproject_to_utm(src_path, dst_path):
    with rasterio.open(src_path) as src:
        # SRTM tiles from OpenTopography come in EPSG:4326 (lat/lon degrees).
        # Slope/aspect math needs pixel size in METERS, not degrees, or the
        # gradient calculation silently produces wrong angles. East Khasi
        # Hills sits in UTM zone 46N (EPSG:32646) - reprojecting here once
        # avoids every downstream script having to reason about CRS units.
        dst_crs = "EPSG:32646"
        if src.crs.to_string() == dst_crs:
            with rasterio.open(dst_path, "w", **src.meta) as dst:
                dst.write(src.read())
            return dst_path

        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update(crs=dst_crs, transform=transform, width=width, height=height)

        with rasterio.open(dst_path, "w", **meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
    return dst_path


def compute_slope_aspect_curvature(dem_path):
    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype("float64")
        transform = src.transform
        profile = src.profile
        nodata = src.nodata if src.nodata is not None else -9999.0
        elevation[elevation == nodata] = np.nan

        # transform.a = pixel width in meters (x), transform.e = pixel height
        # in meters (y, negative because raster rows go north->south).
        px_x = transform.a
        px_y = -transform.e

        dz_dy, dz_dx = np.gradient(elevation, px_y, px_x)

        # Slope in degrees via the standard terrain-analysis formula.
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)

        # Aspect: compass direction the slope faces (0=N, 90=E, 180=S, 270=W).
        aspect_rad = np.arctan2(dz_dy, -dz_dx)
        aspect_deg = np.degrees(aspect_rad)
        aspect_deg = np.where(aspect_deg < 0, 90.0 - aspect_deg, 90.0 - aspect_deg)
        aspect_deg = np.mod(aspect_deg, 360.0)

        # Plan curvature (2nd derivative) - positive = convex (ridges, more
        # prone to failure), negative = concave (valleys, water-collecting).
        d2z_dx2 = np.gradient(dz_dx, px_x, axis=1)
        d2z_dy2 = np.gradient(dz_dy, px_y, axis=0)
        curvature = -2.0 * (d2z_dx2 + d2z_dy2)

    return elevation, slope_deg, aspect_deg, curvature, profile


def write_raster(path, array, profile):
    profile = profile.copy()
    profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    utm_dem_path = os.path.join(args.outdir, "dem_utm46n.tif")
    reproject_to_utm(args.dem, utm_dem_path)

    elevation, slope, aspect, curvature, profile = compute_slope_aspect_curvature(utm_dem_path)

    write_raster(os.path.join(args.outdir, "elevation.tif"), elevation, profile)
    write_raster(os.path.join(args.outdir, "slope.tif"), slope, profile)
    write_raster(os.path.join(args.outdir, "aspect.tif"), aspect, profile)
    write_raster(os.path.join(args.outdir, "curvature.tif"), curvature, profile)

    print(f"Wrote elevation.tif, slope.tif, aspect.tif, curvature.tif to {args.outdir}")
    print(f"Slope range: {np.nanmin(slope):.1f} to {np.nanmax(slope):.1f} degrees")


if __name__ == "__main__":
    main()
