"use client";

import {
  Settings,
  Palette,
  Layers,
  Info,
  Lightbulb,
  ChevronRight,
} from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type SidebarProps = {
  isOpen?: boolean;
  onClose?: () => void;
  className?: string;
};

const processingTips = [
  "Use high-resolution images for best reflection detection",
  "NEF / RAW files are fully supported — camera WB is applied automatically",
  "Enhance-preserve keeps floor, walls & corner intact",
  "Batch up to 50 images at once",
  "~10–15 seconds per image in enhance-preserve mode",
  "Reflection removal works on hood, roof, doors, bumper & glass",
];

export function Sidebar({ isOpen = true, className }: SidebarProps) {
  return (
    <div className={cn("w-full space-y-6", !isOpen && "hidden", className)}>
        {/* Processing options */}
        <Card className="border-slate-200/80 dark:border-slate-700/80">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Settings className="h-4 w-4 text-emerald-600" />
              <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                Processing Options
              </h3>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="mb-1.5 flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-400">
                <Layers className="h-3.5 w-3.5" />
                Output Format
              </label>
              <select
                className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
                defaultValue="png"
              >
                <option value="png">PNG (with transparency)</option>
                <option value="jpg">JPEG (white background)</option>
              </select>
            </div>
            <div>
              <label className="mb-1.5 flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-400">
                <Palette className="h-3.5 w-3.5" />
                Background
              </label>
              <select
                className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
                defaultValue="white"
              >
                <option value="white">White (studio)</option>
                <option value="transparent">Transparent</option>
                <option value="light-gray">Light gray</option>
              </select>
            </div>
          </CardContent>
        </Card>

        {/* Quick tips */}
        <Card className="border-slate-200/80 dark:border-slate-700/80">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Lightbulb className="h-4 w-4 text-amber-500" />
              <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                Quick Tips
              </h3>
            </div>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {processingTips.map((tip, i) => (
                <li
                  key={i}
                  className="flex gap-2 text-xs text-slate-600 dark:text-slate-400"
                >
                  <ChevronRight className="mt-0.5 h-3 w-3 shrink-0 text-emerald-500" />
                  {tip}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        {/* Model info */}
        <Card className="border-slate-200/80 dark:border-slate-700/80">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Info className="h-4 w-4 text-slate-500" />
              <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                Models Used
              </h3>
            </div>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2.5">
              <li>
                <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400">FLUX.1-Fill-dev</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">Black Forest Labs · Primary reflection removal (Replicate)</p>
              </li>
              <li>
                <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400">RMBG-1.4</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">BRIA AI · Car mask &amp; background removal</p>
              </li>
              <li>
                <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400">SegFormer-B0</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">NVIDIA · Sky &amp; ceiling segmentation (ADE20K)</p>
              </li>
              <li>
                <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400">rawpy (LibRaw)</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">NEF / RAW image decoding with camera WB</p>
              </li>
              <li>
                <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400">OpenCV TELEA</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">Local fallback · Reflection &amp; dust removal</p>
              </li>
            </ul>
          </CardContent>
        </Card>
    </div>
  );
}
