import { NextResponse, type NextRequest } from "next/server";
import { UID_COOKIE, UID_HEADER, isSecureRequest, isValidUid } from "./lib/uid";

// Gives every browser a stable anonymous id (the unit of randomisation for the A/B test) and forwards it to
// the route handlers in a request header, including on the request that creates the cookie.
export function middleware(req: NextRequest) {
  const existing = req.cookies.get(UID_COOKIE)?.value;
  const uid = isValidUid(existing) ? existing : crypto.randomUUID();
  const headers = new Headers(req.headers);
  headers.set(UID_HEADER, uid); // never trust a client supplied value
  const res = NextResponse.next({ request: { headers } });
  if (uid !== existing) {
    res.cookies.set(UID_COOKIE, uid, {
      httpOnly: true,
      sameSite: "lax",
      secure: isSecureRequest(req.headers, req.nextUrl.protocol),
      maxAge: 60 * 60 * 24 * 365,
      path: "/",
    });
  }
  return res;
}

export const config = { matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"] };
