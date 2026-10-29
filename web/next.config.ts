import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The container build sets NEXT_OUTPUT=standalone; local `next start` uses the default output.
  output: process.env.NEXT_OUTPUT === "standalone" ? "standalone" : undefined,
  poweredByHeader: false,
  reactStrictMode: true,
};

export default nextConfig;
