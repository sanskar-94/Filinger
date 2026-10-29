import { NextResponse, type NextRequest } from "next/server";

// A stable anonymous id per browser: the unit of randomisation for the A/B test (sticky assignment).
export function middleware(req: NextRequest) {
  const res = NextResponse.next();
  if (!req.cookies.get("flab_uid")) {
    res.cookies.set("flab_uid", crypto.randomUUID(), {
      httpOnly: true,
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
      maxAge: 60 * 60 * 24 * 365,
      path: "/",
    });
  }
  return res;
}

export const config = { matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"] };
