import type { Metadata } from "next";
import { DM_Serif_Display, DM_Sans } from "next/font/google";
import "./globals.css";
import { ThemeInitializer } from "@/components/ui/ThemeInitializer";
import { SmoothScroll } from "@/components/ui/SmoothScroll";
import { Particles } from "@/components/ui/Particles";
import { CustomCursor } from "@/components/ui/CustomCursor";
import { ThemeSwitcher } from "@/components/ui/ThemeSwitcher";

const dmSerifDisplay = DM_Serif_Display({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-serif",
  display: "swap",
});

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Study Helper",
  description: "AI-powered study platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${dmSerifDisplay.variable} ${dmSans.variable} antialiased`}
      >
        <ThemeInitializer />
        <SmoothScroll>
          <Particles />
          <CustomCursor />
          {children}
          <ThemeSwitcher />
        </SmoothScroll>
      </body>
    </html>
  );
}
