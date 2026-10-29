import { NextResponse, type NextRequest } from "next/server";
import { backend } from "@/lib/backend";

export async function POST(req: NextRequest) {
  const body = await req.text();
  const r = await backend("/v1/feedback", { method: "POST", body });
  return NextResponse.json(await r.json(), { status: r.status });
}
