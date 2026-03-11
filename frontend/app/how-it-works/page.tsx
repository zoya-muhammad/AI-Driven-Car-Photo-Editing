import { Card, CardContent } from "@/components/ui/card";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Upload, Sparkles, Eye, Download } from "lucide-react";

const steps = [
  {
    step: 1,
    icon: Upload,
    title: "Upload",
    description: "Drag & drop your car photos or click to browse. Supports JPEG, PNG, WebP, and Nikon NEF (RAW). Single image or batch up to 50 at once.",
    tags: ["JPEG", "PNG", "WebP", "NEF / RAW"],
  },
  {
    step: 2,
    icon: Sparkles,
    title: "AI Processing Pipeline",
    description: "The enhance-preserve pipeline runs six stages automatically:",
    pipeline: [
      { label: "Sky & ceiling removal", detail: "SegFormer-B0 (ADE20K) segments sky/ceiling pixels. OpenCV TELEA inpainting fills the removed area." },
      { label: "Car mask extraction", detail: "RMBG-1.4 isolates the car body — used for all subsequent steps." },
      { label: "Reflection removal", detail: "Tier 1: HSV correction reduces specular highlights while keeping paint texture. Tier 2: OpenCV TELEA reconstructs intensely overexposed spots." },
      { label: "Car enhancement", detail: "Unsharp mask + contrast boost applied only to the car region. Background stays untouched." },
      { label: "Tire cleaning", detail: "Local contrast analysis finds dust and bright spots on tires. Inpainting removes them; gentle darkening deepens tire blacks." },
      { label: "Lighting adjustment", detail: "Global brightness boost (configurable) applied as a final pass." },
    ],
  },
  {
    step: 3,
    icon: Eye,
    title: "Preview",
    description: "Use the before/after slider to compare the original and processed image side by side. Verify quality before downloading.",
    tags: [],
  },
  {
    step: 4,
    icon: Download,
    title: "Download",
    description: "Download individual processed images or the full batch as a ZIP. PNG, JPEG, and WebP are supported. Floor, walls, and corner are always intact.",
    tags: ["PNG", "JPEG", "WebP", "ZIP batch"],
  },
];

export default function HowItWorksPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-12 sm:px-6 lg:px-8">
      <div className="mb-12 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-slate-800 dark:text-slate-100 sm:text-4xl">
          How it works
        </h1>
        <p className="mt-4 text-lg text-slate-600 dark:text-slate-400">
          From upload to download — four steps powered by a six-stage AI pipeline.
        </p>
      </div>

      <div className="space-y-6">
        {steps.map((s) => (
          <Card key={s.step} className="border-slate-200/80 dark:border-slate-700/80">
            <CardContent className="flex flex-col gap-4 p-6 sm:flex-row sm:items-start sm:gap-6">
              <div className="flex shrink-0 items-center gap-4 sm:flex-col sm:items-center sm:gap-2">
                <span className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-100 text-lg font-bold text-emerald-600 dark:bg-emerald-900/50 dark:text-emerald-400">
                  {s.step}
                </span>
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 dark:bg-slate-800">
                  <s.icon className="h-5 w-5 text-slate-600 dark:text-slate-400" />
                </div>
              </div>
              <div className="flex-1">
                <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-100">
                  {s.title}
                </h2>
                <p className="mt-1 text-slate-600 dark:text-slate-400">{s.description}</p>

                {s.pipeline && (
                  <ol className="mt-3 space-y-2">
                    {s.pipeline.map((p, i) => (
                      <li key={i} className="flex gap-3 text-sm">
                        <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-xs font-bold text-emerald-600 dark:bg-emerald-900/50 dark:text-emerald-400">
                          {i + 1}
                        </span>
                        <span>
                          <span className="font-medium text-slate-700 dark:text-slate-200">{p.label}: </span>
                          <span className="text-slate-500 dark:text-slate-400">{p.detail}</span>
                        </span>
                      </li>
                    ))}
                  </ol>
                )}

                {s.tags && s.tags.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {s.tags.map((tag) => (
                      <span key={tag} className="rounded-md bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="mt-16 flex justify-center">
        <Button size="lg" asChild>
          <Link href="/">Get started</Link>
        </Button>
      </div>
    </div>
  );
}
