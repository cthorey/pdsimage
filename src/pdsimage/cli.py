"""Command-line interface: render a region or named structure to a PNG."""

from __future__ import annotations

import argparse
import sys

from . import plotting
from .area import Area
from .structures import Crater, Dome


def _build_area(args) -> Area:
    if args.crater:
        return Crater("name", args.crater, root=args.cache)
    if args.dome:
        return Dome("name", args.dome, root=args.cache)
    if args.lon is None or args.lat is None or args.size is None:
        raise SystemExit("Provide --crater/--dome, or --lon --lat --size.")
    return Area(args.lon, args.lat, args.size, root=args.cache)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pdsimage", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="Render a region to a PNG image.")
    render.add_argument("--crater", help="Crater name (e.g. Copernicus).")
    render.add_argument("--dome", help="Dome name (e.g. M13).")
    render.add_argument("--lon", type=float, help="Center longitude (0-360).")
    render.add_argument("--lat", type=float, help="Center latitude.")
    render.add_argument("--size", type=float, help="Window radius (km).")
    render.add_argument(
        "--type", choices=["lola", "wac", "overlay"], default="lola", help="Image type."
    )
    render.add_argument("--cache", default=None, help="PDS cache directory.")
    render.add_argument("-o", "--output", default="pdsimage.png", help="Output PNG path.")

    args = parser.parse_args(argv)

    if args.command == "render":
        area = _build_area(args)
        renderer = {"lola": plotting.lola_image, "wac": plotting.wac_image, "overlay": plotting.overlay}[
            args.type
        ]
        fig = renderer(area)
        with open(args.output, "wb") as fh:
            fh.write(plotting.fig_to_png(fig))
        print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
