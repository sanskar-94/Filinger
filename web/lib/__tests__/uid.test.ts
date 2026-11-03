import { describe, expect, it } from "vitest";
import { NextRequest } from "next/server";
import { middleware } from "../../middleware";
import { UID_HEADER, isSecureRequest, isValidUid, userIdFrom } from "../uid";

const forwardedUid = (res: Response) => res.headers.get(`x-middleware-request-${UID_HEADER}`);

describe("experiment unit id", () => {
  it("accepts only uuids", () => {
    expect(isValidUid("3f2b8c1e-9a4d-4c6b-8e2f-1a2b3c4d5e6f")).toBe(true);
    expect(isValidUid("anonymous")).toBe(false);
    expect(isValidUid("x".repeat(500))).toBe(false);
    expect(isValidUid(undefined)).toBe(false);
  });

  it("decides Secure from the original scheme, not from NODE_ENV", () => {
    expect(isSecureRequest(new Headers(), "http:")).toBe(false);
    expect(isSecureRequest(new Headers(), "https:")).toBe(true);
    expect(isSecureRequest(new Headers({ "x-forwarded-proto": "https" }), "http:")).toBe(true);
    expect(isSecureRequest(new Headers({ "x-forwarded-proto": "http, https" }), "https:")).toBe(false);
  });

  it("falls back to anonymous only when the middleware did not run", () => {
    expect(userIdFrom(new Headers())).toBe("anonymous");
    expect(userIdFrom(new Headers({ [UID_HEADER]: "not-a-uuid" }))).toBe("anonymous");
  });

  it("gives a new browser an id on its very first request and forwards that same id", () => {
    const res = middleware(new NextRequest("http://lab.example/api/ask", { method: "POST" }));
    const cookie = res.cookies.get("flab_uid");
    expect(isValidUid(cookie?.value)).toBe(true);
    expect(forwardedUid(res)).toBe(cookie?.value);
    expect(cookie?.secure).toBe(false);
  });

  it("keeps an existing id and ignores a spoofed header", () => {
    const uid = "3f2b8c1e-9a4d-4c6b-8e2f-1a2b3c4d5e6f";
    const req = new NextRequest("https://lab.example/", {
      headers: { cookie: `flab_uid=${uid}`, [UID_HEADER]: "11111111-1111-1111-1111-111111111111" },
    });
    const res = middleware(req);
    expect(forwardedUid(res)).toBe(uid);
    expect(res.cookies.get("flab_uid")).toBeUndefined();
  });

  it("replaces a malformed cookie", () => {
    const res = middleware(new NextRequest("https://lab.example/", { headers: { cookie: "flab_uid=../../etc" } }));
    expect(isValidUid(res.cookies.get("flab_uid")?.value)).toBe(true);
    expect(res.cookies.get("flab_uid")?.secure).toBe(true);
  });
});
