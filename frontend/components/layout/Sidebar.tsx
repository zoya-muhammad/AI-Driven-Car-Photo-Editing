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
  "Gemini 3.1 Flash Image powers all AI editing",
  "NEF / RAW files supported — camera WB applied automatically",
  "Enhance mode: reflections, floor, glass, tires — keeps walls intact",
  "Batch up to 50 images at once",
  "~20–60 seconds per image (Gemini API)",
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
                AI Model
              </h3>
            </div>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2.5">
              <li>
                <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400">Gemini 3.1 Flash Image</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">Google · Image editing, reflection removal, background replacement</p>
              </li>
              <li>
                <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400">rawpy (LibRaw)</p>
                <p className="text-xs text-slate-500 dark:text-slate-400">NEF / RAW decoding with camera WB</p>
              </li>
            </ul>
          </CardContent>
        </Card>
    </div>
  );
}
