// The experiment unit: one stable anonymous id per browser, used for sticky A/B assignment and as the
// reviewer id on the gold review page.

export const UID_COOKIE = "flab_uid";
// set by the middleware on every request it forwards, overwriting anything the client sent
export const UID_HEADER = "x-flab-uid";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function isValidUid(value: string | null | undefined): value is string {
  return !!value && UUID.test(value);
}

// Behind the load balancer TLS ends before Next.js, so the original scheme is in x-forwarded-proto.
export function isSecureRequest(headers: Headers, protocol: string): boolean {
  const forwarded = headers.get("x-forwarded-proto")?.split(",")[0]?.trim().toLowerCase();
  return (forwarded ?? protocol.replace(/:$/, "").toLowerCase()) === "https";
}

// Route handlers read the id the middleware attached, so even the very first request of a new browser (before
// its cookie exists) belongs to that browser rather than to a shared "anonymous" bucket.
export function userIdFrom(headers: Headers): string {
  const uid = headers.get(UID_HEADER);
  return isValidUid(uid) ? uid : "anonymous";
}
