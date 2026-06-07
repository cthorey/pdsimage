import { useEffect, useState } from "react";
import {
  CatalogRow,
  ImgType,
  Window,
  fetchCatalog,
  fetchProfile,
  renderImage,
} from "./api";
import ProfileChart from "./ProfileChart";

const PPD_OPTIONS: Record<ImgType, number[]> = {
  lola: [4, 16, 64, 128, 512],
  wac: [4, 8, 16, 32, 64, 128, 256],
  overlay: [16, 64, 128],
};

type Mode = "catalog" | "manual";
type CatalogKind = "craters" | "domes";

interface PickedPoint {
  fx: number; // normalized x within the image
  fy: number;
  lon: number;
  lat: number;
}

export default function App() {
  const [mode, setMode] = useState<Mode>("catalog");
  const [catalogKind, setCatalogKind] = useState<CatalogKind>("craters");
  const [query, setQuery] = useState("");
  const [rows, setRows] = useState<CatalogRow[]>([]);

  const [lon0, setLon0] = useState(339.87);
  const [lat0, setLat0] = useState(9.62);
  const [size, setSize] = useState(100);
  const [imgType, setImgType] = useState<ImgType>("lola");
  const [ppd, setPpd] = useState(16);

  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [window_, setWindow] = useState<Window | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [picks, setPicks] = useState<PickedPoint[]>([]);
  const [profile, setProfile] = useState<number[] | null>(null);
  const [profileBusy, setProfileBusy] = useState(false);

  // Load catalog when kind/query changes (catalog mode only).
  useEffect(() => {
    if (mode !== "catalog") return;
    let cancelled = false;
    fetchCatalog(catalogKind, query)
      .then((r) => !cancelled && setRows(r))
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [mode, catalogKind, query]);

  // Keep ppd valid for the chosen image type.
  useEffect(() => {
    if (!PPD_OPTIONS[imgType].includes(ppd)) setPpd(PPD_OPTIONS[imgType][0]);
  }, [imgType]); // eslint-disable-line react-hooks/exhaustive-deps

  function selectStructure(row: CatalogRow) {
    setLon0(row.lon0);
    setLat0(row.lat0);
    setSize(Math.round(0.8 * row.diameter));
  }

  async function doRender() {
    setLoading(true);
    setError(null);
    setProfile(null);
    setPicks([]);
    try {
      const res = await renderImage({ lon0, lat0, size_km: size, img_type: imgType, ppd });
      setImageUrl(res.url);
      setWindow(res.window);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function onImageClick(e: React.MouseEvent<HTMLImageElement>) {
    if (!window_) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    const [lonLL, lonTR, latLL, latTR] = window_;
    const lon = lonLL + fx * (lonTR - lonLL);
    const lat = latTR - fy * (latTR - latLL);
    const next = [...picks, { fx, fy, lon, lat }].slice(-2);
    setPicks(next);

    if (next.length === 2) {
      setProfileBusy(true);
      try {
        const res = await fetchProfile({
          lon0,
          lat0,
          size_km: size,
          p1: { lon: next[0].lon, lat: next[0].lat },
          p2: { lon: next[1].lon, lat: next[1].lat },
          num_points: 300,
          img_type: "lola",
          ppd,
        });
        setProfile(res.z);
      } catch (err) {
        setError(String(err));
      } finally {
        setProfileBusy(false);
      }
    } else {
      setProfile(null);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>pdsimage</h1>
        <span className="subtitle">NASA LRO lunar data explorer · LOLA / WAC</span>
      </header>

      <div className="layout">
        <aside className="panel">
          <div className="tabs">
            <button className={mode === "catalog" ? "active" : ""} onClick={() => setMode("catalog")}>
              Catalog
            </button>
            <button className={mode === "manual" ? "active" : ""} onClick={() => setMode("manual")}>
              Manual
            </button>
          </div>

          {mode === "catalog" && (
            <div className="group">
              <div className="seg">
                <button
                  className={catalogKind === "craters" ? "active" : ""}
                  onClick={() => setCatalogKind("craters")}
                >
                  Craters
                </button>
                <button
                  className={catalogKind === "domes" ? "active" : ""}
                  onClick={() => setCatalogKind("domes")}
                >
                  Domes
                </button>
              </div>
              <input
                type="text"
                placeholder="Search by name…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <select
                size={8}
                className="catalog-list"
                onChange={(e) => {
                  const row = rows[Number(e.target.value)];
                  if (row) selectStructure(row);
                }}
              >
                {rows.map((r, i) => (
                  <option key={`${r.name}-${i}`} value={i}>
                    {String(r.name)} — ⌀{r.diameter} km
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="group">
            <label>
              Longitude (0–360)
              <input type="number" value={lon0} step={0.01} onChange={(e) => setLon0(+e.target.value)} />
            </label>
            <label>
              Latitude (−90–90)
              <input type="number" value={lat0} step={0.01} onChange={(e) => setLat0(+e.target.value)} />
            </label>
            <label>
              Window radius (km)
              <input type="number" value={size} step={1} onChange={(e) => setSize(+e.target.value)} />
            </label>
          </div>

          <div className="group">
            <label>Image type</label>
            <div className="seg">
              {(["lola", "wac", "overlay"] as ImgType[]).map((t) => (
                <button key={t} className={imgType === t ? "active" : ""} onClick={() => setImgType(t)}>
                  {t.toUpperCase()}
                </button>
              ))}
            </div>
            <label>
              Resolution (ppd)
              <select value={ppd} onChange={(e) => setPpd(+e.target.value)}>
                {PPD_OPTIONS[imgType].map((p) => (
                  <option key={p} value={p}>
                    {p} ppd
                  </option>
                ))}
              </select>
            </label>
          </div>

          <button className="render" disabled={loading} onClick={doRender}>
            {loading ? "Rendering…" : "Render"}
          </button>
          {imgType !== "lola" && (
            <p className="hint">Note: WAC tiles are large and the source server may be slow.</p>
          )}
          {error && <p className="error">{error}</p>}
        </aside>

        <main className="viewer">
          {imageUrl ? (
            <>
              <div className="image-wrap">
                <img src={imageUrl} alt="rendered region" onClick={onImageClick} />
                {picks.map((p, i) => (
                  <span
                    key={i}
                    className="pick"
                    style={{ left: `${p.fx * 100}%`, top: `${p.fy * 100}%` }}
                  />
                ))}
              </div>
              <p className="hint">
                {window_
                  ? "Click two points on the image to extract an elevation profile."
                  : "Profile tool needs window metadata (re-render)."}
              </p>
              {profileBusy && <p className="hint">Computing profile…</p>}
              {profile && <ProfileChart z={profile} />}
            </>
          ) : (
            <div className="placeholder">Pick a region and hit Render.</div>
          )}
        </main>
      </div>
    </div>
  );
}
