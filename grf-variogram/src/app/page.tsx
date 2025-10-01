"use client";  // <-- add this line, at line 1

// Version 1.0 — arbitrary-size FFT (no power-of-two snapping)
// =====================================================
// 2D Gaussian Random Field — Variogram Visualizer
// =====================================================

import React, { useEffect, useRef, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { RefreshCw, Shuffle, ZoomIn, ZoomOut, Bug } from "lucide-react";
import { motion } from "framer-motion";
import FFT from "fft.js";

// =====================================================
// 2D GAUSSIAN RANDOM FIELD — VARIOGRAM-DRIVEN SAMPLER
// Circulant embedding + FFT-based spectral synthesis
// Models: Spherical, Exponential, Matérn (ν = 0.5, 1.5, 2.5)
// Anisotropy via range (major), subrange (minor), and azimuth (deg)
// =====================================================

// ------------------ Utility ------------------
function nextPow2(n: number) {
  return Math.pow(2, Math.ceil(Math.log2(Math.max(2, n))));
}

// ------------------ RNG ------------------
function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randn(rng: () => number) {
  const u = Math.max(1e-12, rng());
  const v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ------------------ Variogram -> Covariance (sill=1) ------------------
function sphericalCov(r: number, a: number) {
  if (r >= a) return 0;
  const x = r / a;
  const gamma = 1.5 * x - 0.5 * x * x * x; // γ(r)
  return 1 - gamma; // C = 1 - γ
}
function exponentialCov(r: number, a: number) {
  return Math.exp(-r / a);
}
function maternCov(r: number, a: number, nu: 0.5 | 1.5 | 2.5) {
  const x = r / a;
  if (nu === 0.5) return Math.exp(-x);
  if (nu === 1.5) return (1 + x) * Math.exp(-x);
  return (1 + x + (x * x) / 3) * Math.exp(-x);
}
function deg2rad(d: number) {
  return (d * Math.PI) / 180;
}

function buildCovarianceKernel({
  nx,
  ny,
  model,
  rangeMajor,
  rangeMinor,
  azimuthDeg,
  maternNu,
}: {
  nx: number;
  ny: number;
  model: "spherical" | "exponential" | "matern";
  rangeMajor: number;
  rangeMinor: number;
  azimuthDeg: number;
  maternNu: 0.5 | 1.5 | 2.5;
}) {
  const a = Math.max(1e-6, rangeMajor);
  const b = Math.max(1e-6, rangeMinor);
  const theta = deg2rad(azimuthDeg);
  const ct = Math.cos(theta);
  const st = Math.sin(theta);

  const cov = new Float64Array(nx * ny);
  const chooseCov = (r: number) => {
    if (model === "spherical") return sphericalCov(r, 1);
    if (model === "exponential") return exponentialCov(r, 1);
    return maternCov(r, 1, maternNu);
  };

  // Toroidal distances ensure a circulant embedding (periodic boundary)
  for (let j = 0; j < ny; j++) {
    const dyWrap = j <= ny / 2 ? j : j - ny;
    for (let i = 0; i < nx; i++) {
      const dxWrap = i <= nx / 2 ? i : i - nx;
      const xr = dxWrap * ct + dyWrap * st; // rotate by -theta
      const yr = -dxWrap * st + dyWrap * ct;
      const rScaled = Math.sqrt((xr / a) * (xr / a) + (yr / b) * (yr / b));
      let c = chooseCov(rScaled);
      if (!Number.isFinite(c)) c = 0;
      cov[j * nx + i] = c;
    }
  }
  return cov;
}

// ------------------ Arbitrary-size 1D FFT via Bluestein ------------------
// We implement Bluestein (chirp-z) to support any N using fft.js for the convolution step.
// Reference: Bluestein 1968. For forward transform: Y[k] = sum_n x[n] e^{-2πi n k / N}.

function fftConvolve(aRe: Float64Array, aIm: Float64Array, bRe: Float64Array, bIm: Float64Array) {
  const M = aRe.length; // must equal b length
  const f = new FFT(M);
  const A = f.createComplexArray();
  const B = f.createComplexArray();
  const FA = f.createComplexArray();
  const FB = f.createComplexArray();
  for (let i = 0; i < M; i++) { A[2*i] = aRe[i]; A[2*i+1] = aIm[i]; B[2*i] = bRe[i]; B[2*i+1] = bIm[i]; }
  f.transform(FA, A);
  f.transform(FB, B);
  // pointwise multiply
  for (let i = 0; i < M; i++) {
    const ar = FA[2*i], ai = FA[2*i+1], br = FB[2*i], bi = FB[2*i+1];
    const rr = ar * br - ai * bi;
    const ii = ar * bi + ai * br;
    FA[2*i] = rr; FA[2*i+1] = ii;
  }
  // inverse
  const out = f.createComplexArray();
  f.inverseTransform(out, FA);
  const outRe = new Float64Array(M);
  const outIm = new Float64Array(M);
  const scale = 1 / M; // fft.js inverse is unnormalized
  for (let i = 0; i < M; i++) { outRe[i] = out[2*i] * scale; outIm[i] = out[2*i+1] * scale; }
  return { re: outRe, im: outIm };
}

function fft1dAny(reIn: Float64Array, imIn: Float64Array, N: number, inverse = false) {
  if (N === 1) return { re: Float64Array.from(reIn), im: Float64Array.from(imIn) };
  // Bluestein setup
  const M = nextPow2(2 * N - 1);
  const aRe = new Float64Array(M);
  const aIm = new Float64Array(M);
  const bRe = new Float64Array(M);
  const bIm = new Float64Array(M);

  const sign = inverse ? 1 : -1;
  // Precompute chirp c[n] = exp(i*pi*sign * n^2 / N)
  for (let n = 0; n < N; n++) {
    const ang = Math.PI * sign * (n * n) / N;
    const cos = Math.cos(ang), sin = Math.sin(ang);
    // a[n] = x[n] * conj(c[n]) for forward; for inverse, x[n] * conj(c[n]) with sign flip matches below
    const xr = reIn[n] || 0, xi = imIn[n] || 0;
    // conj(c[n]) = cos - i sin
    const ar = xr * cos + xi * sin;
    const ai = -xr * sin + xi * cos;
    aRe[n] = ar; aIm[n] = ai;
  }
  // b[0] = 1; for k=1..N-1, b[k] = c[k], and mirror b[M-k] = c[k]
  bRe[0] = 1; bIm[0] = 0;
  for (let k = 1; k < N; k++) {
    const ang = Math.PI * sign * (k * k) / N;
    const cos = Math.cos(ang), sin = Math.sin(ang);
    bRe[k] = cos; bIm[k] = sin;
    bRe[M - k] = cos; bIm[M - k] = sin;
  }

  const { re: convRe, im: convIm } = fftConvolve(aRe, aIm, bRe, bIm);

  // y[k] = conv[k] * conj(c[k])
  const yRe = new Float64Array(N);
  const yIm = new Float64Array(N);
  const norm = inverse ? 1 / N : 1; // normalize inverse
  for (let k = 0; k < N; k++) {
    const ang = Math.PI * sign * (k * k) / N;
    const cos = Math.cos(ang), sin = Math.sin(ang);
    const vr = convRe[k];
    const vi = convIm[k];
    // multiply by conj(c[k]) = cos - i sin
    const rr = vr * cos + vi * sin;
    const ii = -vr * sin + vi * cos;
    yRe[k] = rr * norm;
    yIm[k] = ii * norm;
  }
  return { re: yRe, im: yIm };
}

// ------------------ 2D FFT using arbitrary-size 1D ------------------
function fft2dRealForward(dataRe: Float64Array, nx: number, ny: number) {
  const re = new Float64Array(dataRe);
  const im = new Float64Array(nx * ny);
  // rows
  for (let j = 0; j < ny; j++) {
    const ofs = j * nx;
    const { re: R, im: I } = fft1dAny(re.subarray(ofs, ofs + nx), im.subarray(ofs, ofs + nx), nx, false);
    for (let i = 0; i < nx; i++) { re[ofs + i] = R[i]; im[ofs + i] = I[i]; }
  }
  // cols
  const colRe = new Float64Array(ny);
  const colIm = new Float64Array(ny);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) { colRe[j] = re[j * nx + i]; colIm[j] = im[j * nx + i]; }
    const { re: R, im: I } = fft1dAny(colRe, colIm, ny, false);
    for (let j = 0; j < ny; j++) { re[j * nx + i] = R[j]; im[j * nx + i] = I[j]; }
  }
  return { re, im };
}

function ifft2dToReal(reIn: Float64Array, imIn: Float64Array, nx: number, ny: number) {
  let re = new Float64Array(reIn);
  let im = new Float64Array(imIn);
  // inverse rows
  for (let j = 0; j < ny; j++) {
    const ofs = j * nx;
    const { re: R, im: I } = fft1dAny(re.subarray(ofs, ofs + nx), im.subarray(ofs, ofs + nx), nx, true);
    for (let i = 0; i < nx; i++) { re[ofs + i] = R[i]; im[ofs + i] = I[i]; }
  }
  // inverse cols
  const colRe = new Float64Array(ny);
  const colIm = new Float64Array(ny);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) { colRe[j] = re[j * nx + i]; colIm[j] = im[j * nx + i]; }
    const { re: R, im: I } = fft1dAny(colRe, colIm, ny, true);
    for (let j = 0; j < ny; j++) { re[j * nx + i] = R[j]; im[j * nx + i] = I[j]; }
  }
  // Output real part
  const out = new Float64Array(nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) out[j * nx + i] = re[j * nx + i];
  return out;
}

function sqrtClamp(arr: Float64Array, eps = 1e-12) {
  const re = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) re[i] = Math.sqrt(Math.max(eps, arr[i]));
  return re;
}

function generateGRF({
  nx,
  ny,
  model,
  rangeMajor,
  rangeMinor,
  azimuthDeg,
  maternNu,
  seed,
  padFFT = true,
  padFactor = 2,
}: {
  nx: number;
  ny: number;
  model: "spherical" | "exponential" | "matern";
  rangeMajor: number;
  rangeMinor: number;
  azimuthDeg: number;
  maternNu: 0.5 | 1.5 | 2.5;
  seed: number;
  padFFT?: boolean;
  padFactor?: number;
}) {
  const makeSample = (NX: number, NY: number) => {
    const cov = buildCovarianceKernel({ nx: NX, ny: NY, model, rangeMajor, rangeMinor, azimuthDeg, maternNu });
    const { re: lamRe } = fft2dRealForward(cov, NX, NY);
    const lam = new Float64Array(lamRe.length);
    for (let k = 0; k < lam.length; k++) lam[k] = lamRe[k];
    const sqrtLam = sqrtClamp(lam);

    const rng = mulberry32(seed >>> 0);
    const white = new Float64Array(NX * NY);
    for (let i = 0; i < white.length; i++) white[i] = randn(rng);

    const { re: wRe, im: wIm } = fft2dRealForward(white, NX, NY);

    const sRe = new Float64Array(wRe.length);
    const sIm = new Float64Array(wIm.length);
    for (let k = 0; k < wRe.length; k++) {
      sRe[k] = wRe[k] * sqrtLam[k];
      sIm[k] = wIm[k] * sqrtLam[k];
    }

    return ifft2dToReal(sRe, sIm, NX, NY);
  };

  if (padFFT) {
    const NX = Math.max(2, Math.floor(nx * padFactor));
    const NY = Math.max(2, Math.floor(ny * padFactor));
    const big = makeSample(NX, NY);

    // center-crop to nx×ny
    const out = new Float64Array(nx * ny);
    const x0 = Math.floor((NX - nx) / 2);
    const y0 = Math.floor((NY - ny) / 2);
    for (let y = 0; y < ny; y++) {
      const srcOfs = (y0 + y) * NX + x0;
      const dstOfs = y * nx;
      for (let x = 0; x < nx; x++) out[dstOfs + x] = big[srcOfs + x];
    }

    // normalize
    let mean = 0; for (let v of out) mean += v; mean /= out.length;
    let varsum = 0; for (let v of out) varsum += (v - mean) * (v - mean);
    const std = Math.sqrt(varsum / out.length) || 1;
    for (let i = 0; i < out.length; i++) out[i] = (out[i] - mean) / std;
    return out;
  } else {
    const field = makeSample(nx, ny);
    let mean = 0; for (let v of field) mean += v; mean /= field.length;
    let varsum = 0; for (let v of field) varsum += (v - mean) * (v - mean);
    const std = Math.sqrt(varsum / field.length) || 1;
    for (let i = 0; i < field.length; i++) field[i] = (field[i] - mean) / std;
    return field;
  }
}

function drawFieldToCanvas(
  canvas: HTMLCanvasElement,
  field: Float64Array,
  nx: number,
  ny: number,
  zoom = 1
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = nx * zoom;
  const h = ny * zoom;
  canvas.width = w;
  canvas.height = h;

  const img = ctx.createImageData(w, h);

  let fmin = Infinity,
    fmax = -Infinity;
  for (let v of field) {
    if (v < fmin) fmin = v;
    if (v > fmax) fmax = v;
  }
  const span = fmax - fmin || 1;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const fx = Math.floor(x / zoom);
      const fy = Math.floor(y / zoom);
      const v = (field[fy * nx + fx] - fmin) / span; // 0..1
      const c = Math.pow(v, 0.9);
      const r = Math.max(0, Math.min(255, Math.floor(255 * c)));
      const g = Math.max(0, Math.min(255, Math.floor(255 * (0.9 * c + 0.1 * (1 - c)))));
      const b = Math.max(0, Math.min(255, Math.floor(255 * (0.8 * (1 - c) + 0.2 * c))));
      const idx = (y * w + x) * 4;
      img.data[idx + 0] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

export default function GaussianRandomFieldApp() {
  // Inject tiny style (in-browser only) for slow spin
  useEffect(() => {
    const style = document.createElement("style");
    style.innerHTML = `.animate-spin-slow { animation: spin 2.2s linear infinite; }`;
    document.head.appendChild(style);
    return () => { style.remove(); };
  }, []);

  // Requested grid (now used directly, no snapping)
  const [nx, setNx] = useState(150);
  const [ny, setNy] = useState(110);

  const [model, setModel] = useState<"spherical" | "exponential" | "matern">("spherical");
  const [nu, setNu] = useState<0.5 | 1.5 | 2.5>(1.5);
  const [rangeMajor, setRangeMajor] = useState(30);
  const [rangeMinor, setRangeMinor] = useState(15);
  const [azimuth, setAzimuth] = useState(30);
  const [seed, setSeed] = useState(1234);
  const [zoom, setZoom] = useState(3);
  const [autoRegenerate, setAutoRegenerate] = useState(true);
  const [padFFT, setPadFFT] = useState(true);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [field, setField] = useState<Float64Array | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [testReport, setTestReport] = useState<string>("");

  const regenerate = React.useCallback(() => {
    setIsBusy(true);
    setTimeout(() => {
      const fld = generateGRF({
        nx,
        ny,
        model,
        rangeMajor,
        rangeMinor,
        azimuthDeg: azimuth,
        maternNu: nu,
        seed,
        padFFT,
        padFactor: 2,
      });
      setField(fld);
      setIsBusy(false);
    }, 10);
  }, [nx, ny, model, rangeMajor, rangeMinor, azimuth, nu, seed, padFFT]);

  useEffect(() => {
    if (autoRegenerate) regenerate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nx, ny, model, rangeMajor, rangeMinor, azimuth, nu, seed, autoRegenerate, padFFT]);

  useEffect(() => {
    if (!canvasRef.current || !field) return;
    drawFieldToCanvas(canvasRef.current, field, nx, ny, zoom);
  }, [field, nx, ny, zoom]);

  // --- Tiny test harness ---
  function runTests() {
    const lines: string[] = [];
    let pass = 0;
    let fail = 0;
    function t(name: string, fn: () => void) {
      try {
        fn();
        lines.push(`✔ ${name}`);
        pass++;
      } catch (e: any) {
        lines.push(`✘ ${name} -> ${e?.message ?? e}`);
        fail++;
      }
    }
    function expect(cond: boolean, msg: string) {
      if (!cond) throw new Error(msg);
    }

    // New tests for arbitrary sizes
    t("1D Bluestein matches delta property (N=7)", () => {
      const N = 7;
      const re = new Float64Array(N); const im = new Float64Array(N);
      re[1] = 1; // x[n] = δ[n-1]
      const { re: R, im: I } = fft1dAny(re, im, N, false);
      for (let k = 0; k < N; k++) {
        // |X[k]| should be 1 for delta input
        const mag = Math.hypot(R[k], I[k]);
        if (Math.abs(mag - 1) > 1e-9) throw new Error(`mag != 1 at k=${k}: ${mag}`);
      }
    });

    t("2D forward+inverse returns original (N=150x110)", () => {
      const nx = 150, ny = 110;
      const re = new Float64Array(nx * ny);
      for (let i = 0; i < re.length; i++) re[i] = Math.sin(0.01 * i);
      const { re: R, im: I } = fft2dRealForward(re, nx, ny);
      const back = ifft2dToReal(R, I, nx, ny);
      let err = 0;
      for (let i = 0; i < re.length; i++) err += Math.abs(back[i] - re[i]);
      err /= re.length;
      if (err > 1e-6) throw new Error(`ifft error too large: ${err}`);
    });

    // New: covariance models differ and Matérn ν ordering
    t("Covariance models differ (r=0.5)", () => {
      const r = 0.5, a = 1;
      const cS = sphericalCov(r, a);
      const cE = exponentialCov(r, a);
      if (Math.abs(cS - cE) < 1e-6) throw new Error("spherical and exponential too similar at r=0.5");
    });
    t("Matérn ν=2.5 gives larger covariance than ν=0.5 at r=0.8", () => {
      const r = 0.8, a = 1;
      const c05 = maternCov(r, a, 0.5);
      const c25 = maternCov(r, a, 2.5);
      if (!(c25 > c05)) throw new Error(`expected c25>c05, got ${c25} vs ${c05}`);
    });

    t("GRF generation on prime-ish sizes (129x97)", () => {
      const fld = generateGRF({ nx: 129, ny: 97, model: "exponential", rangeMajor: 20, rangeMinor: 10, azimuthDeg: 25, maternNu: 1.5, seed: 12, padFFT: false });
      if (fld.length !== 129 * 97) throw new Error("length mismatch");
      for (let v of fld) if (!Number.isFinite(v)) throw new Error("non-finite");
    });

    t("Padded generation crops to requested size (150x110 from 2× embed)", () => {
      const fld = generateGRF({ nx: 150, ny: 110, model: "matern", rangeMajor: 24, rangeMinor: 16, azimuthDeg: 40, maternNu: 2.5, seed: 3, padFFT: true, padFactor: 2 });
      if (fld.length !== 150 * 110) throw new Error("padded length mismatch");
      for (let v of fld) if (!Number.isFinite(v)) throw new Error("non-finite after pad");
    });

    // Padding should change the realization with the same seed (different embedding), but keep size
    t("Padding toggles change sample but preserve length", () => {
      const a = generateGRF({ nx: 100, ny: 80, model: "spherical", rangeMajor: 20, rangeMinor: 12, azimuthDeg: 10, maternNu: 1.5, seed: 7, padFFT: false });
      const b = generateGRF({ nx: 100, ny: 80, model: "spherical", rangeMajor: 20, rangeMinor: 12, azimuthDeg: 10, maternNu: 1.5, seed: 7, padFFT: true, padFactor: 2 });
      expect(a.length === b.length && a.length === 100 * 80, "lengths differ");
      let diff = 0; for (let i = 0; i < a.length; i++) diff += Math.abs(a[i] - b[i]);
      expect(diff > 1e-6, "padding produced identical sample (unexpected)");
    });

    // Extra: 1D arbitrary-size forward+inverse sanity (N=9)
    t("1D arbitrary-size FFT inverse sanity (N=9)", () => {
      const N = 9;
      const re = new Float64Array(N);
      const im = new Float64Array(N);
      for (let n = 0; n < N; n++) re[n] = Math.sin(0.3 * n) + 0.1 * n;
      const { re: R, im: I } = fft1dAny(re, im, N, false);
      const { re: r2, im: i2 } = fft1dAny(R, I, N, true);
      let err = 0;
      for (let n = 0; n < N; n++) err += Math.abs(r2[n] - re[n]);
      err /= N;
      if (err > 1e-7) throw new Error(`1D roundtrip error too large: ${err}`);
    });

    setTestReport(`${pass} passed, ${fail} failed\n` + lines.join("\n"));
  }

  const maxRange = Math.max(4, Math.floor(0.45 * Math.min(nx, ny)));

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-slate-50 to-slate-100 p-6">
      <div className="mx-auto max-w-6xl">
        <motion.h1
          className="text-3xl md:text-4xl font-bold tracking-tight text-slate-900 mb-2"
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          2D Gaussian Random Field — Variogram Visualizer <span className="text-xs align-super text-slate-500">v1.0</span>
        </motion.h1>
        <p className="text-slate-600 mb-4 max-w-3xl">
          Explore anisotropic Gaussian random fields generated from geostatistical variogram models (Spherical, Exponential, Matérn). Adjust
          <span className="font-semibold"> range</span>, <span className="font-semibold">subrange</span>, and <span className="font-semibold">azimuth</span>.
        </p>

        <Tabs defaultValue="viewer" className="w-full">
          <TabsList>
            <TabsTrigger value="viewer">Viewer</TabsTrigger>
            <TabsTrigger value="tests"><Bug className="h-4 w-4 mr-1" />Tests</TabsTrigger>
          </TabsList>

          <TabsContent value="viewer">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card className="col-span-2 shadow-sm">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-4 text-slate-700">
                      <div className="flex items-center gap-2">
                        <Switch id="auto" checked={autoRegenerate} onCheckedChange={setAutoRegenerate} />
                        <Label htmlFor="auto">Auto-regenerate on change</Label>
                      </div>
                      <div className="flex items-center gap-2">
                        <Switch id="pad" checked={padFFT} onCheckedChange={setPadFFT} />
                        <Label htmlFor="pad">Pad FFT (2×) to reduce artifacts</Label>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button variant="secondary" size="sm" onClick={() => setSeed((s) => (s * 1664525 + 1013904223) >>> 0)}>
                        <Shuffle className="h-4 w-4 mr-1" /> New seed
                      </Button>
                      <Button onClick={regenerate} disabled={isBusy}>
                        <RefreshCw className="h-4 w-4 mr-1 animate-spin-slow" /> Generate
                      </Button>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-xs sm:text-sm text-slate-600 mb-2">
                    <div>
                      Model: <b>{model.toUpperCase()}</b> {model === "matern" ? ` (ν = ${nu})` : ""} · Grid <b>{nx}×{ny}</b>
                    </div>
                    <div>Seed {seed}</div>
                  </div>

                  <div className="relative rounded-2xl overflow-hidden bg-white border">
                    <div className="flex items-center justify-between p-2 border-b bg-slate-50">
                      <div className="text-sm text-slate-600">Zoom {zoom}×</div>
                      <div className="flex gap-2">
                        <Button variant="outline" size="sm" onClick={() => setZoom((z) => Math.max(1, z - 1))}>
                          <ZoomOut className="h-4 w-4" />
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => setZoom((z) => Math.min(8, z + 1))}>
                          <ZoomIn className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                    <div className="w-full overflow-auto">
                      <canvas ref={canvasRef} className="block" />
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="shadow-sm">
                <CardContent className="p-4 space-y-5">
                  {/* Variogram model choice */}
                  <div>
                    <Label className="mb-1 block">Variogram model</Label>
                    <Select value={model} onValueChange={(v) => setModel(v as any)}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select model" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="spherical">Spherical</SelectItem>
                        <SelectItem value="exponential">Exponential</SelectItem>
                        <SelectItem value="matern">Matérn</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Matérn nu when applicable */}
                  {model === "matern" && (
                    <div>
                      <Label className="mb-1 block">Matérn smoothness ν</Label>
                      <Select value={String(nu)} onValueChange={(v) => setNu(Number(v) as any)}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0.5">ν = 0.5 (≈ Exponential)</SelectItem>
                          <SelectItem value="1.5">ν = 1.5</SelectItem>
                          <SelectItem value="2.5">ν = 2.5</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  )}

                  {/* Range controls */}
                  <div>
                    <Label className="mb-1 block">Range (major axis)</Label>
                    <Slider value={[rangeMajor]} min={2} max={maxRange} step={1} onValueChange={([v]) => setRangeMajor(v)} />
                    <div className="text-sm text-slate-600 mt-1">{rangeMajor} px</div>
                  </div>

                  <div>
                    <Label className="mb-1 block">Subrange (minor axis)</Label>
                    <Slider value={[rangeMinor]} min={1} max={maxRange} step={1} onValueChange={([v]) => setRangeMinor(v)} />
                    <div className="text-sm text-slate-600 mt-1">{rangeMinor} px</div>
                  </div>

                  <div>
                    <Label className="mb-1 block">Azimuth (°)</Label>
                    <Slider value={[azimuth]} min={0} max={180} step={1} onValueChange={([v]) => setAzimuth(v)} />
                    <div className="text-sm text-slate-600 mt-1">{azimuth}°</div>
                  </div>

                  {/* Seed */}
                  <div>
                    <Label className="mb-1 block">Random seed</Label>
                    <Input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} />
                  </div>

                  <div className="text-xs text-slate-500 leading-relaxed">
                    <p className="mb-1 font-medium">Notes</p>
                    <ul className="list-disc pl-5 space-y-1">
                      <li>FFT supports any grid size via Bluestein’s algorithm (no power-of-two requirement).</li>
                      <li><b>Optional padding</b> (2×) embeds the covariance on a larger torus and crops the center; this often reduces wrap-around artifacts.</li>
                      <li>Fields are generated via FFT-based circulant embedding (periodic boundary conditions).</li>
                      <li>Matérn ν options (0.5, 1.5, 2.5) use closed-form covariances.</li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="tests">
            <Card className="shadow-sm">
              <CardContent className="p-4 space-y-3">
                <p className="text-sm text-slate-700">Runtime tests for arbitrary-size FFT & generator.</p>
                <div className="flex gap-2">
                  <Button variant="outline" onClick={runTests}><Bug className="h-4 w-4 mr-1" />Run tests</Button>
                  <Button onClick={regenerate}><RefreshCw className="h-4 w-4 mr-1" />Regenerate sample</Button>
                </div>
                <pre className="text-xs bg-slate-50 border rounded p-3 whitespace-pre-wrap">{testReport || "(no test report yet)"}</pre>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
