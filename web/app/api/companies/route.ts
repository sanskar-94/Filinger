import { NextResponse } from "next/server";
import { backend } from "@/lib/backend";

export async function GET() {
  const r = await backend("/v1/companies");
  return NextResponse.json(await r.json(), { status: r.status });
}
