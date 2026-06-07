// Thin client for the pdsimage FastAPI backend.
// All requests go through the Vite dev proxy at /api -> backend.

const BASE = "/api";

export interface CatalogRow {
  name: string;
  lat0: number;
  lon0: number;
  diameter: number;
  index: number;
  [k: string]: unknown;
}

export type ImgType = "lola" | "wac" | "overlay";

export interface RenderParams {
  lon0: number;
  lat0: number;
  size_km: number;
  img_type: ImgType;
  ppd?: number;
}

export type Window = [number, number, number, number]; // lonLL, lonTR, latLL, latTR

export interface RenderResult {
  url: string;
  window: Window | null;
}

export async function fetchCatalog(
  kind: "craters" | "domes",
  q: string,
): Promise<CatalogRow[]> {
  const res = await fetch(`${BASE}/catalog/${kind}?q=${encodeURIComponent(q)}&limit=100`);
  if (!res.ok) throw new Error(`Catalog request failed: ${res.status}`);
  return res.json();
}

export async function renderImage(params: RenderParams): Promise<RenderResult> {
  const res = await fetch(`${BASE}/render`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail || `Render failed: ${res.status}`);
  }
  const header = res.headers.get("X-Window");
  const window = header ? (JSON.parse(header) as Window) : null;
  const blob = await res.blob();
  return { url: URL.createObjectURL(blob), window };
}

export interface ProfileParams {
  lon0: number;
  lat0: number;
  size_km: number;
  p1: { lon: number; lat: number };
  p2: { lon: number; lat: number };
  num_points: number;
  img_type: ImgType;
  ppd?: number;
}

export interface ProfileResult {
  distance: number[];
  z: number[];
}

export async function fetchProfile(params: ProfileParams): Promise<ProfileResult> {
  const res = await fetch(`${BASE}/profile`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail || `Profile failed: ${res.status}`);
  }
  return res.json();
}
