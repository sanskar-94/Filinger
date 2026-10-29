// Server side helper: every call to the Python API goes through here so the key stays on the server.
const BASE = process.env.FLAB_API_URL ?? "http://localhost:8080";

export async function backend(path: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers);
  headers.set("content-type", "application/json");
  if (process.env.FLAB_API_KEY) headers.set("x-api-key", process.env.FLAB_API_KEY);
  return fetch(`${BASE}${path}`, { ...init, headers, cache: "no-store" });
}

export async function backendJson<T>(path: string): Promise<T | null> {
  try {
    const r = await backend(path);
    return r.ok ? ((await r.json()) as T) : null;
  } catch {
    return null;
  }
}
