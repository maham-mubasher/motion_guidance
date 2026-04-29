"use client";

import { useEffect, useMemo, useState } from "react";

type JobResponse = {
  status?: string;
  progress?: number;
  message?: string;
  code?: number;
};

type RunResponse = JobResponse & {
  job_id?: string;
  results_url_prefix?: string;
  source_image_url?: string;
};

type FlowMode = "file" | "primitive";
type PrimitiveKind = "translate" | "scale" | "rotate" | "stretch";

type ExamplePreset = {
  label: string;
  prompt: string;
  inputDir: string;
  mask: string;
  flow: string;
  targetFlowMode: FlowMode;
  primitiveKind: PrimitiveKind;
  primitiveDx: number;
  primitiveDy: number;
  primitiveScaleX: number;
  primitiveScaleY: number;
  primitiveAngleDeg: number;
  useHardWarpInit: boolean;
  useSelectiveRefinement: boolean;
  selectiveOuterWeight: number;
  preserveUneditedOutput: boolean;
  useExampleReferenceOutput: boolean;
  guidanceWeight: number;
  clipGrad: number;
  hint: string;
};

const EXAMPLE_PRESETS: Record<string, ExamplePreset> = {
  apple: {
    label: "Apple - move right",
    prompt: "an apple on a wooden table",
    inputDir: "./data/apple",
    mask: "right.mask.pth",
    flow: "right.pth",
    targetFlowMode: "primitive",
    primitiveKind: "translate",
    primitiveDx: 150,
    primitiveDy: 0,
    primitiveScaleX: 1.1,
    primitiveScaleY: 1.1,
    primitiveAngleDeg: 10,
    useHardWarpInit: true,
    useSelectiveRefinement: true,
    selectiveOuterWeight: 0.15,
    preserveUneditedOutput: false,
    useExampleReferenceOutput: false,
    guidanceWeight: 30,
    clipGrad: 60,
    hint: "Apple setup: start with `X=150`, `Y=0` using `right.pth` as the reference support flow.",
  },
  teapot: {
    label: "Teapot - move down",
    prompt: "a teapot floating in water",
    inputDir: "./data/teapot",
    mask: "down150.mask.pth",
    flow: "down150.pth",
    targetFlowMode: "primitive",
    primitiveKind: "translate",
    primitiveDx: 0,
    primitiveDy: 150,
    primitiveScaleX: 1.1,
    primitiveScaleY: 1.1,
    primitiveAngleDeg: 10,
    useHardWarpInit: true,
    useSelectiveRefinement: true,
    selectiveOuterWeight: 0.15,
    preserveUneditedOutput: false,
    useExampleReferenceOutput: false,
    guidanceWeight: 30,
    clipGrad: 60,
    hint: "Teapot setup: start with `X=0`, `Y=150` using `down150.pth` as the reference support flow.",
  },
  lion: {
    label: "Lion - move left",
    prompt: "a photo of a lion",
    inputDir: "./data/lion",
    mask: "left.mask.pth",
    flow: "left.pth",
    targetFlowMode: "primitive",
    primitiveKind: "translate",
    primitiveDx: -100,
    primitiveDy: 0,
    primitiveScaleX: 1.1,
    primitiveScaleY: 1.1,
    primitiveAngleDeg: 10,
    useHardWarpInit: true,
    useSelectiveRefinement: true,
    selectiveOuterWeight: 0.15,
    preserveUneditedOutput: false,
    useExampleReferenceOutput: false,
    guidanceWeight: 30,
    clipGrad: 60,
    hint: "Lion setup: start with `X=-100`, `Y=0` using `left.pth` as the reference support flow.",
  },
  woman: {
    label: "Woman - grow hair",
    prompt: "a portrait photo of a woman",
    inputDir: "./data/woman",
    mask: "growHair.mask.pth",
    flow: "growHair.pth",
    targetFlowMode: "file",
    primitiveKind: "translate",
    primitiveDx: 0,
    primitiveDy: 0,
    primitiveScaleX: 1.1,
    primitiveScaleY: 1.1,
    primitiveAngleDeg: 10,
    useHardWarpInit: false,
    useSelectiveRefinement: true,
    selectiveOuterWeight: 0.25,
    preserveUneditedOutput: true,
    useExampleReferenceOutput: true,
    guidanceWeight: 30,
    clipGrad: 60,
    hint: "Woman setup: preserves the original portrait outside the hair-flow region.",
  },
  tree: {
    label: "Tree - squeeze",
    prompt: "a painting of a lone tree",
    inputDir: "./data/tree",
    mask: "squeeze.mask.pth",
    flow: "squeeze.pth",
    targetFlowMode: "primitive",
    primitiveKind: "stretch",
    primitiveDx: 0,
    primitiveDy: 0,
    primitiveScaleX: 0.8,
    primitiveScaleY: 1.1,
    primitiveAngleDeg: 10,
    useHardWarpInit: true,
    useSelectiveRefinement: true,
    selectiveOuterWeight: 0.15,
    preserveUneditedOutput: false,
    useExampleReferenceOutput: false,
    guidanceWeight: 30,
    clipGrad: 60,
    hint: "Tree setup: use primitive `Stretch` with `X=0.8`, `Y=1.1`; `squeeze.pth` is only used to localize the tree support.",
  },
  topiary: {
    label: "Topiary - precomputed flow",
    prompt: "a photo of topiary",
    inputDir: "./data/topiary",
    mask: "mask.pth",
    flow: "flow.pth",
    targetFlowMode: "file",
    primitiveKind: "translate",
    primitiveDx: 0,
    primitiveDy: 0,
    primitiveScaleX: 1.1,
    primitiveScaleY: 1.1,
    primitiveAngleDeg: 10,
    useHardWarpInit: false,
    useSelectiveRefinement: true,
    selectiveOuterWeight: 0.25,
    preserveUneditedOutput: false,
    useExampleReferenceOutput: true,
    guidanceWeight: 30,
    clipGrad: 60,
    hint: "Topiary setup: uses the shipped motion-edited reference output for this example.",
  },
};

export default function Home() {
  const API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
  const [examplePreset, setExamplePreset] = useState("lion");
  const activePreset = EXAMPLE_PRESETS[examplePreset];

  const [prompt, setPrompt] = useState("a photo of a lion");
  const [inputDir, setInputDir] = useState("./data/lion");
  const [mask, setMask] = useState("left.mask.pth");
  const [flow, setFlow] = useState("left.pth");
  const [targetFlowMode, setTargetFlowMode] = useState<FlowMode>("primitive");
  const [primitiveKind, setPrimitiveKind] = useState<PrimitiveKind>("translate");
  const [primitiveDx, setPrimitiveDx] = useState(-100);
  const [primitiveDy, setPrimitiveDy] = useState(0);
  const [primitiveScaleX, setPrimitiveScaleX] = useState(1.1);
  const [primitiveScaleY, setPrimitiveScaleY] = useState(1.1);
  const [primitiveAngleDeg, setPrimitiveAngleDeg] = useState(10);
  const [useHardWarpInit, setUseHardWarpInit] = useState(true);
  const [useSelectiveRefinement, setUseSelectiveRefinement] = useState(true);
  const [selectiveInnerWeight, setSelectiveInnerWeight] = useState(1.0);
  const [selectiveOuterWeight, setSelectiveOuterWeight] = useState(0.15);
  const [preserveUneditedOutput, setPreserveUneditedOutput] = useState(false);
  const [useExampleReferenceOutput, setUseExampleReferenceOutput] = useState(false);
  const [fastPreview, setFastPreview] = useState(true);
  const [ddimSteps, setDdimSteps] = useState(20);
  const [guidanceWeight, setGuidanceWeight] = useState(30);
  const [clipGrad, setClipGrad] = useState(60);
  const [raftIters, setRaftIters] = useState(1);
  const [recursiveSteps, setRecursiveSteps] = useState(1);

  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [message, setMessage] = useState<string>("ready");
  const [progress, setProgress] = useState<number>(0);
  const [resultsPrefix, setResultsPrefix] = useState<string>("");
  const [submittedSourceImageUrl, setSubmittedSourceImageUrl] = useState<string>("");
  const [isCancelling, setIsCancelling] = useState(false);

  const isRunning = status === "queued" || status === "running";
  const progressSafe = Math.max(0, Math.min(100, progress));

  const previewUrl = useMemo(() => {
    if (!resultsPrefix) return "";
    return `${resultsPrefix}sample_000/pred.png`;
  }, [resultsPrefix]);
  const inputSourceImageUrl = useMemo(() => {
    const dataPrefix = inputDir.startsWith("./data/") ? "./data/" : "data/";
    if (!inputDir.startsWith(dataPrefix)) return "";

    const inputPath = inputDir
      .slice(dataPrefix.length)
      .split("/")
      .filter(Boolean)
      .map(encodeURIComponent)
      .join("/");

    return inputPath ? `${API}/inputs/${inputPath}/pred.png` : "";
  }, [API, inputDir]);

  const applyPreset = (presetKey: string) => {
    const preset = EXAMPLE_PRESETS[presetKey];
    setExamplePreset(presetKey);
    setPrompt(preset.prompt);
    setInputDir(preset.inputDir);
    setMask(preset.mask);
    setFlow(preset.flow);
    setTargetFlowMode(preset.targetFlowMode);
    setPrimitiveKind(preset.primitiveKind);
    setPrimitiveDx(preset.primitiveDx);
    setPrimitiveDy(preset.primitiveDy);
    setPrimitiveScaleX(preset.primitiveScaleX);
    setPrimitiveScaleY(preset.primitiveScaleY);
    setPrimitiveAngleDeg(preset.primitiveAngleDeg);
    setUseHardWarpInit(preset.useHardWarpInit);
    setUseSelectiveRefinement(preset.useSelectiveRefinement);
    setSelectiveInnerWeight(1.0);
    setSelectiveOuterWeight(preset.selectiveOuterWeight);
    setPreserveUneditedOutput(preset.preserveUneditedOutput);
    setUseExampleReferenceOutput(preset.useExampleReferenceOutput);
    setGuidanceWeight(preset.guidanceWeight);
    setClipGrad(preset.clipGrad);
  };

  useEffect(() => {
    if (!jobId || !isRunning) return;

    const intervalId = window.setInterval(async () => {
      try {
        const res = await fetch(`${API}/jobs/${jobId}`);
        if (!res.ok) return;

        const data = (await res.json()) as JobResponse;
        if (typeof data.status === "string") setStatus(data.status);
        if (typeof data.message === "string") setMessage(data.message);
        if (typeof data.progress === "number") setProgress(data.progress);

        if (data.status === "done") {
          setMessage("completed");
          setProgress(100);
          window.clearInterval(intervalId);
        }
        if (data.status === "failed") {
          setMessage(data.code ? `failed (code ${data.code})` : "failed");
          setIsCancelling(false);
          window.clearInterval(intervalId);
        }
        if (data.status === "cancelled") {
          setMessage(data.message || "cancelled by user");
          setIsCancelling(false);
          window.clearInterval(intervalId);
        }
        if (data.status === "not_found") {
          setStatus("failed");
          setProgress(0);
          setMessage(data.message || "job state lost (backend restarted), please rerun");
          setIsCancelling(false);
          window.clearInterval(intervalId);
        }
      } catch {
        setMessage("lost connection to backend");
      }
    }, 1200);

    return () => window.clearInterval(intervalId);
  }, [API, jobId, isRunning]);

  const runJob = async () => {
    setStatus("queued");
    setMessage("submitting");
    setProgress(0);
    setJobId(null);
    setResultsPrefix("");
    setSubmittedSourceImageUrl(inputSourceImageUrl);
    setIsCancelling(false);

    try {
      const res = await fetch(`${API}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          input_dir: inputDir,
          edit_mask_path: mask,
          target_flow_name: flow,
          target_flow_mode: targetFlowMode,
          primitive_kind: primitiveKind,
          primitive_dx: primitiveDx,
          primitive_dy: primitiveDy,
          primitive_scale_x: primitiveScaleX,
          primitive_scale_y: primitiveScaleY,
          primitive_angle_deg: primitiveAngleDeg,
          use_hard_warp_init: useHardWarpInit,
          use_selective_refinement: useSelectiveRefinement,
          selective_inner_weight: selectiveInnerWeight,
          selective_outer_weight: selectiveOuterWeight,
          preserve_unedited_output: preserveUneditedOutput,
          use_example_reference_output: useExampleReferenceOutput,
          use_cached_latents: true,
          log_freq: fastPreview ? 0 : 5,
          ddim_steps: fastPreview ? Math.max(ddimSteps, 20) : Math.max(ddimSteps, 25),
          guidance_weight: guidanceWeight,
          clip_grad: clipGrad,
          raft_iters: raftIters,
          scale: 6.5,
          num_recursive_steps: recursiveSteps,
        }),
      });

      if (!res.ok) {
        let detail = "";
        try {
          const errorBody = await res.json();
          if (typeof errorBody.detail === "string") detail = `: ${errorBody.detail}`;
        } catch {
          detail = "";
        }
        setStatus("failed");
        setMessage(`failed (${res.status})${detail}`);
        return;
      }

      const data = (await res.json()) as RunResponse;
      setJobId(data.job_id || null);
      setStatus(data.status || "running");
      setMessage("starting");
      setProgress(typeof data.progress === "number" ? data.progress : 0);
      setResultsPrefix(`${API}${data.results_url_prefix || ""}`);
      setSubmittedSourceImageUrl(
        data.source_image_url ? `${API}${data.source_image_url}` : inputSourceImageUrl,
      );
    } catch {
      setStatus("failed");
      setMessage("backend unreachable (start API on http://localhost:8000)");
    }
  };

  const stopJob = async () => {
    if (!jobId || !isRunning || isCancelling) return;

    setIsCancelling(true);
    setMessage("stopping job");

    try {
      const res = await fetch(`${API}/jobs/${jobId}/cancel`, {
        method: "POST",
      });

      if (!res.ok) {
        setMessage(`stop failed (${res.status})`);
        setIsCancelling(false);
        return;
      }

      const data = (await res.json()) as JobResponse;
      setStatus(data.status || "cancelled");
      setMessage(data.message || "cancelled by user");
      setIsCancelling(false);
    } catch {
      setMessage("failed to contact backend to stop job");
      setIsCancelling(false);
    }
  };

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_20%_15%,#1f8f88_0%,transparent_35%),radial-gradient(circle_at_85%_0%,#d38b30_0%,transparent_32%),linear-gradient(160deg,#0a0f15_0%,#101925_60%,#0d141e_100%)] px-4 py-8 text-slate-100 md:px-8">
      <div className="mx-auto max-w-5xl rounded-3xl border border-white/15 bg-white/8 p-6 shadow-[0_20px_80px_rgba(0,0,0,0.45)] backdrop-blur md:p-8">
        <div className="mb-6 flex items-center justify-between gap-3">
          <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">Motion Guidance Console</h1>
          <div className="rounded-full border border-emerald-300/30 bg-emerald-300/10 px-3 py-1 text-xs font-semibold text-emerald-200">
            {status.toUpperCase()}
          </div>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          <section className="space-y-3 rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Example Preset</div>
              <select
                className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm"
                value={examplePreset}
                onChange={(e) => applyPreset(e.target.value)}
              >
                {Object.entries(EXAMPLE_PRESETS).map(([key, preset]) => (
                  <option key={key} value={key}>{preset.label}</option>
                ))}
                <option disabled>Cat - create ./data/cat first</option>
              </select>
              <div className="text-xs text-slate-400">{activePreset.hint}</div>
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Prompt</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Input Dir</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" value={inputDir} onChange={(e) => setInputDir(e.target.value)} />
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Target Flow Source</div>
              <select
                className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm"
                value={targetFlowMode}
                onChange={(e) => setTargetFlowMode(e.target.value as FlowMode)}
              >
                <option value="file">Precomputed Flow File</option>
                <option value="primitive">Primitive Motion</option>
              </select>
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Mask File</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" value={mask} onChange={(e) => setMask(e.target.value)} />
            </label>
            {targetFlowMode === "file" ? (
              <div className="space-y-3">
                <label className="block space-y-1">
                  <div className="text-xs uppercase tracking-wide text-slate-300">Target Flow</div>
                  <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" value={flow} onChange={(e) => setFlow(e.target.value)} />
                </label>
                <div className="rounded-lg border border-white/10 bg-slate-950/40 p-3">
                  <label className="flex items-center gap-2 text-sm text-slate-200">
                    <input type="checkbox" checked={useHardWarpInit} onChange={(e) => setUseHardWarpInit(e.target.checked)} />
                    Use hard warp initialization
                  </label>
                  <div className="mt-2 text-xs text-slate-400">Starts diffusion from a warped version of the source image.</div>
                </div>
                <div className="rounded-lg border border-white/10 bg-slate-950/40 p-3">
                  <label className="flex items-center gap-2 text-sm text-slate-200">
                    <input type="checkbox" checked={useSelectiveRefinement} onChange={(e) => setUseSelectiveRefinement(e.target.checked)} />
                    Use selective refinement
                  </label>
                  {useSelectiveRefinement && (
                    <div className="mt-3 space-y-3">
                      <label className="block space-y-1">
                        <div className="text-xs uppercase tracking-wide text-slate-300">Inner Weight</div>
                        <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" step="0.05" value={selectiveInnerWeight} onChange={(e) => setSelectiveInnerWeight(Number(e.target.value))} />
                      </label>
                      <label className="block space-y-1">
                        <div className="text-xs uppercase tracking-wide text-slate-300">Outer Weight</div>
                        <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" step="0.05" value={selectiveOuterWeight} onChange={(e) => setSelectiveOuterWeight(Number(e.target.value))} />
                      </label>
                    </div>
                  )}
                </div>
                <div className="rounded-lg border border-white/10 bg-slate-950/40 p-3">
                  <label className="flex items-center gap-2 text-sm text-slate-200">
                    <input type="checkbox" checked={preserveUneditedOutput} onChange={(e) => setPreserveUneditedOutput(e.target.checked)} />
                    Preserve unedited output
                  </label>
                  <div className="mt-2 text-xs text-slate-400">Keeps the original image outside the edit region after decoding, useful for face and hair edits.</div>
                </div>
                <div className="rounded-lg border border-white/10 bg-slate-950/40 p-3">
                  <label className="flex items-center gap-2 text-sm text-slate-200">
                    <input type="checkbox" checked={useExampleReferenceOutput} onChange={(e) => setUseExampleReferenceOutput(e.target.checked)} />
                    Use example reference output
                  </label>
                  <div className="mt-2 text-xs text-slate-400">Uses the shipped motion-edited result when this dataset includes one.</div>
                </div>
              </div>
            ) : (
              <div className="space-y-3 rounded-xl border border-white/10 bg-slate-900/40 p-3">
                <label className="block space-y-1">
                  <div className="text-xs uppercase tracking-wide text-slate-300">Primitive Type</div>
                  <select
                    className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm"
                    value={primitiveKind}
                    onChange={(e) => setPrimitiveKind(e.target.value as PrimitiveKind)}
                  >
                    <option value="translate">Translate</option>
                    <option value="scale">Scale</option>
                    <option value="rotate">Rotate</option>
                    <option value="stretch">Stretch</option>
                  </select>
                </label>
                {primitiveKind === "translate" && (
                  <>
                    <label className="block space-y-1">
                      <div className="text-xs uppercase tracking-wide text-slate-300">Translate X</div>
                      <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" value={primitiveDx} onChange={(e) => setPrimitiveDx(Number(e.target.value))} />
                    </label>
                    <label className="block space-y-1">
                      <div className="text-xs uppercase tracking-wide text-slate-300">Translate Y</div>
                      <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" value={primitiveDy} onChange={(e) => setPrimitiveDy(Number(e.target.value))} />
                    </label>
                    <div className="text-xs text-slate-400">{activePreset.hint}</div>
                  </>
                )}
                {primitiveKind === "scale" && (
                  <>
                    <label className="block space-y-1">
                      <div className="text-xs uppercase tracking-wide text-slate-300">Scale X</div>
                      <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" step="0.05" value={primitiveScaleX} onChange={(e) => setPrimitiveScaleX(Number(e.target.value))} />
                    </label>
                    <label className="block space-y-1">
                      <div className="text-xs uppercase tracking-wide text-slate-300">Scale Y</div>
                      <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" step="0.05" value={primitiveScaleY} onChange={(e) => setPrimitiveScaleY(Number(e.target.value))} />
                    </label>
                    <div className="text-xs text-slate-400">Values above `1.0` expand the masked object, and values below `1.0` shrink it.</div>
                  </>
                )}
                {primitiveKind === "stretch" && (
                  <>
                    <label className="block space-y-1">
                      <div className="text-xs uppercase tracking-wide text-slate-300">Stretch X</div>
                      <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" step="0.05" value={primitiveScaleX} onChange={(e) => setPrimitiveScaleX(Number(e.target.value))} />
                    </label>
                    <label className="block space-y-1">
                      <div className="text-xs uppercase tracking-wide text-slate-300">Stretch Y</div>
                      <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" step="0.05" value={primitiveScaleY} onChange={(e) => setPrimitiveScaleY(Number(e.target.value))} />
                    </label>
                    <div className="text-xs text-slate-400">Stretch lets you scale width and height independently for a more directional deformation.</div>
                  </>
                )}
                {primitiveKind === "rotate" && (
                  <>
                    <label className="block space-y-1">
                      <div className="text-xs uppercase tracking-wide text-slate-300">Angle Deg</div>
                      <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" value={primitiveAngleDeg} onChange={(e) => setPrimitiveAngleDeg(Number(e.target.value))} />
                    </label>
                    <div className="text-xs text-slate-400">Use small angles first, such as `5` to `15`, so the masked object does not deform too aggressively.</div>
                  </>
                )}
                <div className="rounded-lg border border-white/10 bg-slate-950/40 p-3">
                  <label className="flex items-center gap-2 text-sm text-slate-200">
                    <input type="checkbox" checked={useHardWarpInit} onChange={(e) => setUseHardWarpInit(e.target.checked)} />
                    Use hard warp initialization
                  </label>
                  <div className="mt-2 text-xs text-slate-400">Starts diffusion from a warped version of the source image instead of relying only on soft motion guidance.</div>
                </div>
                <div className="rounded-lg border border-white/10 bg-slate-950/40 p-3">
                  <label className="flex items-center gap-2 text-sm text-slate-200">
                    <input type="checkbox" checked={useSelectiveRefinement} onChange={(e) => setUseSelectiveRefinement(e.target.checked)} />
                    Use selective refinement
                  </label>
                  {useSelectiveRefinement && (
                    <div className="mt-3 space-y-3">
                      <label className="block space-y-1">
                        <div className="text-xs uppercase tracking-wide text-slate-300">Inner Weight</div>
                        <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" step="0.05" value={selectiveInnerWeight} onChange={(e) => setSelectiveInnerWeight(Number(e.target.value))} />
                      </label>
                      <label className="block space-y-1">
                        <div className="text-xs uppercase tracking-wide text-slate-300">Outer Weight</div>
                        <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" step="0.05" value={selectiveOuterWeight} onChange={(e) => setSelectiveOuterWeight(Number(e.target.value))} />
                      </label>
                      <div className="text-xs text-slate-400">Lower outer weight keeps more of the original image unchanged outside the edited region.</div>
                    </div>
                  )}
                </div>
                <div className="rounded-lg border border-white/10 bg-slate-950/40 p-3">
                  <label className="flex items-center gap-2 text-sm text-slate-200">
                    <input type="checkbox" checked={preserveUneditedOutput} onChange={(e) => setPreserveUneditedOutput(e.target.checked)} />
                    Preserve unedited output
                  </label>
                  <div className="mt-2 text-xs text-slate-400">Keeps the original image outside the edit region after decoding.</div>
                </div>
                <label className="block space-y-1">
                  <div className="text-xs uppercase tracking-wide text-slate-300">Reference Support Flow</div>
                  <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" value={flow} onChange={(e) => setFlow(e.target.value)} />
                  <div className="text-xs text-slate-400">Used to localize primitive motion to the same object region as the original example flow.</div>
                </label>
              </div>
            )}
          </section>

          <section className="space-y-3 rounded-2xl border border-white/10 bg-slate-950/45 p-4">
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Mode</div>
              <select className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" value={fastPreview ? "fast" : "quality"} onChange={(e) => setFastPreview(e.target.value === "fast")}>
                <option value="fast">Fast Preview</option>
                <option value="quality">Better Quality</option>
              </select>
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">DDIM Steps</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" min={10} max={250} value={ddimSteps} onChange={(e) => setDdimSteps(Number(e.target.value))} />
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Guidance Weight</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" min={1} max={400} value={guidanceWeight} onChange={(e) => setGuidanceWeight(Number(e.target.value))} />
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Clip Grad</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" min={0} max={400} value={clipGrad} onChange={(e) => setClipGrad(Number(e.target.value))} />
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">RAFT Iters</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" min={1} max={5} value={raftIters} onChange={(e) => setRaftIters(Number(e.target.value))} />
            </label>
            <label className="block space-y-1">
              <div className="text-xs uppercase tracking-wide text-slate-300">Recursive Steps</div>
              <input className="w-full rounded-lg border border-white/20 bg-slate-900/80 px-3 py-2 text-sm" type="number" min={1} max={2} value={recursiveSteps} onChange={(e) => setRecursiveSteps(Number(e.target.value))} />
            </label>

            <button
              onClick={runJob}
              disabled={isRunning}
              className="mt-2 w-full rounded-lg bg-[linear-gradient(120deg,#0ea5a4_0%,#14b8a6_45%,#f59e0b_100%)] px-4 py-2 font-semibold text-slate-950 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isRunning ? "Running..." : "Run Generation"}
            </button>
            <button
              onClick={stopJob}
              disabled={!isRunning || isCancelling}
              className="w-full rounded-lg border border-rose-400/40 bg-rose-500/15 px-4 py-2 font-semibold text-rose-100 transition hover:bg-rose-500/25 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isCancelling ? "Stopping..." : "Stop Generation"}
            </button>
          </section>
        </div>

        <section className="mt-6 rounded-2xl border border-white/10 bg-slate-950/45 p-4">
          <div className="mb-2 flex items-center justify-between text-sm">
            <span className="text-slate-200">{message}</span>
            <span className="font-semibold">{progressSafe}%</span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-[linear-gradient(90deg,#22d3ee_0%,#34d399_50%,#f59e0b_100%)] transition-all duration-500"
              style={{ width: `${progressSafe}%` }}
            />
          </div>
          <div className="mt-3 text-xs text-slate-300">Job ID: {jobId || "-"}</div>
          {previewUrl && (
            <div className="mt-3">
              <a className="text-sm font-medium text-cyan-300 underline" href={previewUrl} target="_blank" rel="noreferrer">
                Open latest output image
              </a>
            </div>
          )}
          {status === "done" && (submittedSourceImageUrl || inputSourceImageUrl) && previewUrl && (
            <div className="mt-5">
              <div className="mb-3 text-sm font-semibold text-slate-100">Before and after</div>
              <div className="grid gap-4 md:grid-cols-2">
                <figure className="overflow-hidden rounded-lg border border-white/10 bg-slate-900/55">
                  <div className="border-b border-white/10 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-slate-300">
                    Before
                  </div>
                  {/* Backend input assets are served dynamically from the API. */}
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    className="aspect-square w-full bg-slate-950 object-contain"
                    src={submittedSourceImageUrl || inputSourceImageUrl}
                    alt="Before generation"
                  />
                </figure>
                <figure className="overflow-hidden rounded-lg border border-white/10 bg-slate-900/55">
                  <div className="border-b border-white/10 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-slate-300">
                    After
                  </div>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    className="aspect-square w-full bg-slate-950 object-contain"
                    src={previewUrl}
                    alt="After generation"
                  />
                </figure>
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
