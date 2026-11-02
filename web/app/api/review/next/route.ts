import { NextResponse } from "next/server";
import { backend } from "@/lib/backend";

export async function GET() {
  const r = await backend("/v1/review/next");
  return NextResponse.json(await r.json(), { status: r.status });
}
