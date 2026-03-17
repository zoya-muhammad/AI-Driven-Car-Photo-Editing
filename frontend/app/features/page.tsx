import {
  Zap,
  Layers,
  ImageIcon,
  Clock,
  Download,
  CheckCircle2,
  SunMedium,
  Sparkles,
  Camera,
  CircleDot,
} from "lucide-react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import Link from "next/link";
import { Button } from "@/components/ui/button";

const features = [
  {
    icon: Layers,
    title: "Keeps Floor & Walls",
    description: "Enhance mode preserves floor, walls, and corner. Background removal replaces with white or transparent.",
  },
  {
    icon: SunMedium,
    title: "Reflection Removal",
    description: "Gemini removes bright light reflections from car body and glass. Maintains natural metallic color.",
  },
  {
    icon: Sparkles,
    title: "Car Enhancement",
    description: "Single AI pass: reflections, floor cleanup, tire detail, glass clarity. Powered by Gemini 3.1 Flash Image.",
  },
  {
    icon: CircleDot,
    title: "Tire & Floor Cleanup",
    description: "Deep black tires, removes dust and grime. Cleans floor tiles with consistent dark texture.",
  },
  {
    icon: Camera,
    title: "NEF / RAW Support",
    description: "Upload Nikon NEF (and other RAW formats) directly. rawpy decodes with camera white balance.",
  },
  {
    icon: Zap,
    title: "Batch Processing",
    description: "Process up to 50 car images in one batch. Upload once, process all with one click.",
  },
  {
    icon: ImageIcon,
    title: "Before/After Preview",
    description: "Interactive slider to compare original and processed images side by side before downloading.",
  },
  {
    icon: Download,
    title: "Flexible Output",
    description: "Export as PNG, JPEG, or WebP. White or transparent background.",
  },
  {
    icon: Clock,
    title: "AI Processing",
    description: "Keep-floor-walls is instant. Enhance and background removal use Gemini API (~20–60s per image).",
  },
  {
    icon: CheckCircle2,
    title: "Error Tracking",
    description: "Failed images are flagged with the error reason. Every job is logged for debugging.",
  },
];

export default function FeaturesPage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6 lg:px-8">
      <div className="mb-12 text-center">
        <h1 className="text-3xl font-bold tracking-tight text-slate-800 dark:text-slate-100 sm:text-4xl">
          Features
        </h1>
        <p className="mt-4 text-lg text-slate-600 dark:text-slate-400">
          AI-powered car photo editing — reflection removal, sky cleanup, car enhancement, and RAW support.
        </p>
      </div>

      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {features.map((feature) => (
          <Card
            key={feature.title}
            className="border-slate-200/80 transition-shadow hover:shadow-md dark:border-slate-700/80"
          >
            <CardHeader>
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-100 dark:bg-emerald-900/50">
                <feature.icon className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
              </div>
              <h2 className="mt-4 text-lg font-semibold text-slate-800 dark:text-slate-100">
                {feature.title}
              </h2>
            </CardHeader>
            <CardContent>
              <p className="text-slate-600 dark:text-slate-400">{feature.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="mt-16 flex justify-center">
        <Button size="lg" asChild>
          <Link href="/">
            Open Editor
          </Link>
        </Button>
      </div>
    </div>
  );
}
