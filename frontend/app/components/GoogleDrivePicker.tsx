"use client";

import { useCallback, useEffect, useState } from "react";
import { Cloud } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type Props = {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
  className?: string;
};

const SCOPE = "https://www.googleapis.com/auth/drive.readonly";

export function GoogleDrivePicker({
  onFilesSelected,
  disabled,
  className,
}: Props) {
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;
  const apiKey = process.env.NEXT_PUBLIC_GOOGLE_API_KEY;
  const appId = process.env.NEXT_PUBLIC_GOOGLE_APP_ID;

  useEffect(() => {
    if (!clientId) return;
    const ids = ["gapi-script", "gsi-script"];
    if (ids.every((id) => document.getElementById(id))) {
      setReady(true);
      return;
    }
    const loads: Promise<void>[] = [];
    if (!document.getElementById("gsi-script")) {
      loads.push(
        new Promise((res, rej) => {
          const s = document.createElement("script");
          s.id = "gsi-script";
          s.src = "https://accounts.google.com/gsi/client";
          s.async = true;
          s.onload = () => res();
          s.onerror = () => rej(new Error("GSI load failed"));
          document.head.appendChild(s);
        })
      );
    }
    if (!document.getElementById("gapi-script")) {
      loads.push(
        new Promise((res, rej) => {
          const s = document.createElement("script");
          s.id = "gapi-script";
          s.src = "https://apis.google.com/js/api.js";
          s.async = true;
          s.onload = () => res();
          s.onerror = () => rej(new Error("GAPI load failed"));
          document.head.appendChild(s);
        })
      );
    }
    Promise.all(loads).then(() => setReady(true)).catch(() => setError("Failed to load Google"));
  }, [clientId]);

  const handlePick = useCallback(async () => {
    if (!clientId || !apiKey || !appId || !ready || disabled) {
      if (!apiKey || !appId) setError("Missing API Key or App ID");
      return;
    }
    setLoading(true);
    setError(null);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const gw = (window as any).google;
    const gapi = (window as any).gapi;

    try {
      const accessToken = await new Promise<string>((resolve, reject) => {
        const tc = gw?.accounts?.oauth2?.initTokenClient?.({
          client_id: clientId,
          scope: SCOPE,
          callback: (r: { error?: unknown; access_token?: string }) => {
            if (r.error) reject(r.error);
            else if (r.access_token) resolve(r.access_token);
            else reject(new Error("No token"));
          },
        });
        if (!tc) return reject(new Error("OAuth not ready"));
        tc.requestAccessToken();
      });

      await new Promise<void>((res) => {
        if (gapi) gapi.load("picker", { callback: res });
        else res();
      });

      const view = new gw.picker.View(gw.picker.ViewId.DOCS ?? "DOCS");
      view.setMimeTypes("image/png,image/jpeg,image/jpg,image/webp,image/nef");

      const builder = new gw.picker.PickerBuilder();

      const picker = builder
        .addView(view)
        .setOAuthToken(accessToken)
        .setDeveloperKey(apiKey)
        .setAppId(appId)
        .setCallback(async (data: { action: string; docs?: Array<{ id: string; name: string; mimeType: string }> }) => {
          if (data.action !== "picked" || !data.docs?.length) {
            setLoading(false);
            return;
          }
          try {
            const files: File[] = [];
            for (const doc of data.docs) {
              const res = await fetch(`https://www.googleapis.com/drive/v3/files/${doc.id}?alt=media`, {
                headers: { Authorization: `Bearer ${accessToken}` },
              });
              const blob = await res.blob();
              const ext = (doc.name.split(".").pop() || "jpg").toLowerCase();
              const mime = doc.mimeType || (ext === "nef" ? "image/nef" : "image/jpeg");
              files.push(new File([blob], doc.name, { type: mime }));
            }
            onFilesSelected(files);
          } catch {
            setError("Failed to download files");
          }
          setLoading(false);
        })
        .build();
      picker.setVisible(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Google Drive error");
      setLoading(false);
    }
  }, [clientId, apiKey, appId, ready, disabled, onFilesSelected]);

  if (!clientId) {
    return (
      <Button variant="outline" size="sm" disabled className={cn("opacity-60", className)} title="Set NEXT_PUBLIC_GOOGLE_CLIENT_ID">
        <Cloud className="mr-2 h-4 w-4" />
        Google Drive (not configured)
      </Button>
    );
  }

  return (
    <div className="flex flex-col items-start gap-1">
      <Button variant="outline" size="sm" onClick={handlePick} disabled={disabled || loading || !ready} className={className}>
        <Cloud className="mr-2 h-4 w-4" />
        {loading ? "Loading…" : "Import from Google Drive"}
      </Button>
      {error && <span className="text-xs text-red-600 dark:text-red-400">{error}</span>}
    </div>
  );
}
