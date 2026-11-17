#pragma once
// The moving-filter engine (flanger): a sound plus a copy of itself whose delay moves.
//   find()      measures the copy on one sound: its delay over time (per channel), its gain and its feedback, or nothing
//   apply()     s = -1: the found copy out of the sound it was found on (the exact inverse moving filter);
//               s = +1: onto another sound (the same moving filter at the same times); in between and up to +2: scaled
//   check()     the engine's own test of a result: is the line gone from the picture, did the level fall by what the copy added
//   where()     the stretches of the sound that have the copy (for the caption and the line under the waveform)
// How: in the cepstrum of a short stretch of sound a copy at delay d is a peak at lag d, as high as the copy's gain. Frame
// by frame the peaks draw a line (the picture: frames x lags; short lags from short windows, long lags from long ones, so
// a quick sweep is still seen). The smooth line that collects the most is tracked by dynamic programming, once through
// the whole picture and once through its moving part (the picture minus its running median over time: a held note's own
// cepstrum lines stand still and drop out). Where the line fits a sine or triangle sweep that curve takes its place. Gain
// and feedback are corrected until taking the copy out leaves nothing along the line. The rule (accept) then asks: strong
// enough, reaching at least 1.3 ms, moving like a sweep (hardly any jumps, not standing still like held notes), gone
// after removal, and the level above 2 * SR / move fell by about what a copy of that gain adds.
// A line that passes is then asked where along the sound the copy really is (presence): the copy is taken out there only,
// and a line with no such stretch is refused. Sounds of 24 s and more are measured in pieces (see "a whole sound").
// The measuring of one piece is a port of the Python prototype Research/sweep/sweep.py (same numbers on sounds that have
// the copy all the way: Tools/sweep_parity.py). Only here, not in the prototype: presence, the pieces of a long sound, and
// the short cuts that change no number (picture() with a line, the sliding median of staticPart, the two channels of a
// piece measured at the same time). 48 kHz; plain C++17, no JUCE.

#include <algorithm>
#include <cmath>
#include <functional>
#include <future>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "Room.h"      // dr::Vec, dr::Stereo (Dsp.h) and room::Transform

namespace dr
{
namespace sweep
{
using Stop = std::function<bool()>;                    // asked now and then while measuring: true = give up (the answer is then not used)
constexpr int HOP = 256;                                // a frame of the picture every 5.3 ms
struct Scale { int n, a, b; };                          // window, lags from .. to (samples)
constexpr Scale SCALES[4] = { { 512, 8, 96 }, { 1024, 96, 192 }, { 2048, 192, 384 }, { 4096, 384, 768 } };
constexpr int QLO = 8, QHI = 768, NQ = QHI - QLO;       // delays 0.17 .. 16 ms
constexpr double F_LO = 150.0, F_HI = 16000.0;          // the part of the spectrum the picture is made from
constexpr double FLOOR = 1e-5;                          // each frame's spectrum is held at this much of its top
constexpr int KMAX = 10;                                // the line may move this many samples of delay per frame (39 ms/s)
constexpr double LAM = 0.004;                           // what a move of one sample costs, squared
constexpr int MED = 31;                                 // frames (165 ms): what stays put for longer than half of this is the sound's own
constexpr double FPS = SR / HOP;
constexpr double CAP = 0.8;

// ---------------------------------------------------------------- the picture
struct Picture
{
    int T = 0; Vec E; std::vector<char> act;            // E[t * QHI + q]; act: frames that count (within 50 dB of the loud frames)
    double at (int t, int q) const { return E[(size_t) t * QHI + (size_t) q]; }
    double& at (int t, int q) { return E[(size_t) t * QHI + (size_t) q]; }
};
struct Need { double lo, hi; };                         // a range of lags that will be read

// The cepstrum picture of one channel. need: if given, windows that serve none of these lag ranges are left out (their
// lags stay 0 and act stays empty). line: if given too, the picture will only be read along this line and along twice
// it (along), and a window is left out at every frame where neither reads from its lags.
inline Picture picture (const Vec& x, const std::vector<Need>* need = nullptr, const Vec* line = nullptr)
{
    Picture P; const long len = (long) x.size(); P.T = (int) (len / HOP) + 1; P.E.assign ((size_t) P.T * QHI, 0.0); Vec en;
    for (const Scale& sc : SCALES)
    {
        if (need != nullptr) { bool any = false; for (auto& r : *need) any = any || (r.lo < sc.b && r.hi >= sc.a); if (! any) continue; }
        const int n = sc.n, nb = n / 2 + 1; const room::Transform tr (n); const Vec win = hannPeriodic ((size_t) n);
        Vec w ((size_t) nb, 0.0); { std::vector<int> m; for (int k = 0; k < nb; ++k) { const double f = (double) k * SR / n; if (f >= F_LO && f <= F_HI) m.push_back (k); } const Vec h = hanningSym (m.size()); for (size_t i = 0; i < m.size(); ++i) w[(size_t) m[i]] = h[i]; }
        double sw = 0; for (double v : w) sw += v;
        Vec re ((size_t) n), im ((size_t) n), La ((size_t) nb), Lb ((size_t) nb);
        if (n == 1024) en.assign ((size_t) P.T, 0.0);
        for (int t = 0; t < P.T; t += 2)                 // two frames ride through one transform: one as the real, one as the imaginary part
        {
            if (line != nullptr)
            {
                bool read = false;
                for (int u = t; u < std::min (t + 2, P.T) && u < (int) line->size() && ! read; ++u)
                    for (double k : { 1.0, 2.0 }) { const long i = (long) std::floor (k * (*line)[(size_t) u]); read = read || (i + 1 >= sc.a && i < sc.b); }
                if (! read) continue;
            }
            const bool two = t + 1 < P.T; double ea = 0, eb = 0;
            for (int i = 0; i < n; ++i)
            {
                const long pa = (long) t * HOP - n / 2 + i, pb = pa + HOP;
                const double a = (pa >= 0 && pa < len) ? x[(size_t) pa] * win[(size_t) i] : 0.0, b = (two && pb >= 0 && pb < len) ? x[(size_t) pb] * win[(size_t) i] : 0.0;
                re[(size_t) i] = a; im[(size_t) i] = b; ea += a * a; eb += b * b;
            }
            if (n == 1024) { en[(size_t) t] = ea; if (two) en[(size_t) t + 1] = eb; }
            tr.run (re, im);
            double ta = 0, tb = 0;
            for (int k = 0; k < nb; ++k)
            {
                const size_t u = (size_t) k, m = (size_t) ((n - k) & (n - 1));
                const double ar = 0.5 * (re[u] + re[m]), ai = 0.5 * (im[u] - im[m]), br = 0.5 * (im[u] + im[m]), bi = 0.5 * (re[m] - re[u]);
                La[u] = std::sqrt (ar * ar + ai * ai); Lb[u] = std::sqrt (br * br + bi * bi); ta = std::max (ta, La[u]); tb = std::max (tb, Lb[u]);
            }
            double ma = 0, mb = 0;
            for (int k = 0; k < nb; ++k)
            {
                La[(size_t) k] = std::log (std::max (La[(size_t) k], FLOOR * ta + 1e-30)); Lb[(size_t) k] = std::log (std::max (Lb[(size_t) k], FLOOR * tb + 1e-30));
                ma += La[(size_t) k] * w[(size_t) k]; mb += Lb[(size_t) k] * w[(size_t) k];
            }
            ma /= sw; mb /= sw;
            for (int k = 0; k < nb; ++k)                 // the weighted ln spectra, mirrored: their transform back is the cepstrum (real)
            {
                const double a = (La[(size_t) k] - ma) * w[(size_t) k], b = (Lb[(size_t) k] - mb) * w[(size_t) k];
                re[(size_t) k] = a; im[(size_t) k] = b; if (k > 0 && k < n / 2) { re[(size_t) (n - k)] = a; im[(size_t) (n - k)] = b; }
            }
            tr.back (re, im); const double sc2 = (double) n / sw;
            for (int q = sc.a; q < sc.b; ++q) { P.at (t, q) = re[(size_t) q] * sc2; if (two) P.at (t + 1, q) = im[(size_t) q] * sc2; }
        }
    }
    if (need != nullptr || en.empty()) return P;
    const double top = percentile (en, 95.0); P.act.resize ((size_t) P.T); for (int t = 0; t < P.T; ++t) P.act[(size_t) t] = en[(size_t) t] > top * 1e-5;
    return P;
}

// The picture's running median over MED frames (edges: the first and last frame repeated): what stands still.
inline Vec staticPart (const Picture& P)
{
    Vec M (P.E.size(), 0.0); const int h = MED / 2, T = P.T; if (T <= 0) return M;
    Vec col ((size_t) T), s ((size_t) MED);              // one lag over time; the window's values in rising order
    for (int q = QLO; q < QHI; ++q)
    {
        for (int t = 0; t < T; ++t) col[(size_t) t] = P.at (t, q);
        for (int k = 0; k < MED; ++k) s[(size_t) k] = col[(size_t) std::clamp (k - h, 0, T - 1)];
        std::sort (s.begin(), s.end());
        for (int t = 0;; ++t)
        {
            M[(size_t) t * QHI + (size_t) q] = s[(size_t) h];
            if (t + 1 >= T) break;
            const double gone = col[(size_t) std::clamp (t - h, 0, T - 1)], come = col[(size_t) std::clamp (t + 1 + h, 0, T - 1)];
            if (gone == come) continue;
            size_t i = (size_t) (std::lower_bound (s.begin(), s.end(), gone) - s.begin());       // the value that leaves the window; the one that comes takes its place in the order
            if (come > gone) while (i + 1 < (size_t) MED && s[i + 1] < come) { s[i] = s[i + 1]; ++i; }
            else while (i > 0 && s[i - 1] > come) { s[i] = s[i - 1]; --i; }
            s[i] = come;
        }
    }
    return M;
}

// ---------------------------------------------------------------- the line
// The path q(t) through E (T x QHI, lags QLO .. QHI-1) that maximises the sum of clip (sign * E, 0, CAP) on the frames that
// count minus LAM * step^2, steps of at most KMAX. Returns the path (lags) and the total.
inline std::pair<std::vector<int>, double> track (const Vec& E, const std::vector<char>& act, int T, int sign)
{
    Vec S ((size_t) T * NQ); std::vector<int> B ((size_t) T * NQ, 0);
    auto ev = [&] (int t, int q) { return act[(size_t) t] ? std::clamp ((double) sign * E[(size_t) t * QHI + (size_t) (q + QLO)], 0.0, CAP) : 0.0; };
    for (int q = 0; q < NQ; ++q) S[(size_t) q] = ev (0, q);
    for (int t = 1; t < T; ++t)
    {
        const double* prev = &S[(size_t) (t - 1) * NQ]; double* cur = &S[(size_t) t * NQ]; int* bk = &B[(size_t) t * NQ];
        for (int q = 0; q < NQ; ++q)
        {
            double best = -1e30; int bp = 0;
            for (int k = -KMAX; k <= KMAX; ++k) { const int p = q + k; if (p < 0 || p >= NQ) continue; const double v = prev[p] - LAM * (double) k * (double) k; if (v > best) { best = v; bp = p; } }
            cur[q] = best + ev (t, q); bk[q] = bp;
        }
    }
    std::vector<int> path ((size_t) T); int q = 0; double best = -1e30;
    for (int j = 0; j < NQ; ++j) if (S[(size_t) (T - 1) * NQ + (size_t) j] > best) { best = S[(size_t) (T - 1) * NQ + (size_t) j]; q = j; }
    for (int t = T - 1; t >= 0; --t) { path[(size_t) t] = q + QLO; q = B[(size_t) t * NQ + (size_t) q]; }
    return { path, best };
}

// scipy.ndimage.gaussian_filter1d (mode 'nearest').
inline Vec gauss (const Vec& x, double sig)
{
    const int r = (int) (4.0 * sig + 0.5), n = (int) x.size(); Vec k ((size_t) (2 * r + 1)); double s = 0;
    for (int i = -r; i <= r; ++i) { k[(size_t) (i + r)] = std::exp (-0.5 * (double) i * (double) i / (sig * sig)); s += k[(size_t) (i + r)]; }
    for (auto& v : k) v /= s;
    Vec o ((size_t) n, 0.0);
    for (int i = 0; i < n; ++i) { double a = 0; for (int j = -r; j <= r; ++j) a += k[(size_t) (j + r)] * x[(size_t) std::clamp (i + j, 0, n - 1)]; o[(size_t) i] = a; }
    return o;
}

// The picture read along k times the line (linear between lags); outside the picture: 0.
inline Vec along (const Vec& E, const Vec& d, int T, double k = 1.0)
{
    Vec v ((size_t) T, 0.0);
    for (int t = 0; t < T; ++t)
    {
        const double q = k * d[(size_t) t]; const double fl = std::floor (q); const long i = (long) fl; const double f = q - fl;
        if (i >= 1 && i + 1 < QHI) v[(size_t) t] = E[(size_t) t * QHI + (size_t) i] * (1 - f) + E[(size_t) t * QHI + (size_t) i + 1] * f;
    }
    return v;
}
// What the picture holds along a line: the sum over the frames that count of its value there (0 .. CAP).
inline double collected (const Vec& E, const std::vector<char>& act, int T, int sign, const Vec& d)
{
    const Vec v = along (E, d, T); double s = 0; for (int t = 0; t < T; ++t) if (act[(size_t) t]) s += std::clamp ((double) sign * v[(size_t) t], 0.0, CAP);
    return s;
}

// The line to a fraction of a sample: per frame the top of a parabola through the picture around the path, then a weighted
// smoothing over time. d: delay per frame (samples); hgt: how far the picture stands out there.
struct Line { Vec d, hgt; };
inline Line refine (const Vec& E, const std::vector<char>& act, int T, const std::vector<int>& path, int sign, int half = 4, double sig = 4.0)
{
    Line L; L.d.resize ((size_t) T); L.hgt.assign ((size_t) T, 0.0); for (int t = 0; t < T; ++t) L.d[(size_t) t] = (double) path[(size_t) t];
    for (int t = 0; t < T; ++t)
    {
        if (! act[(size_t) t]) continue;
        const int a = std::max (QLO, path[(size_t) t] - half), b = std::min (QHI - 1, path[(size_t) t] + half); int j = 0; double v = -1e300;
        for (int q = a; q <= b; ++q) { const double s = (double) sign * E[(size_t) t * QHI + (size_t) q]; if (s > v) { v = s; j = q - a; } }
        if (v <= 0) continue;
        double off = 0.0;
        if (j > 0 && j < b - a)
        {
            const double y0 = (double) sign * E[(size_t) t * QHI + (size_t) (a + j - 1)], y1 = v, y2 = (double) sign * E[(size_t) t * QHI + (size_t) (a + j + 1)], den = y0 - 2 * y1 + y2;
            if (den < 0) off = std::clamp (0.5 * (y0 - y2) / den, -0.5, 0.5);
        }
        L.d[(size_t) t] = (double) (a + j) + off; L.hgt[(size_t) t] = v;
    }
    Vec w ((size_t) T), dw ((size_t) T); double sw = 0;
    for (int t = 0; t < T; ++t) { const double m = std::min (L.hgt[(size_t) t], 0.8); w[(size_t) t] = m * m; dw[(size_t) t] = L.d[(size_t) t] * w[(size_t) t]; sw += w[(size_t) t]; }
    if (sw <= 0) return L;
    const Vec num = gauss (dw, sig), den = gauss (w, sig); std::vector<int> ok; Vec val;
    for (int t = 0; t < T; ++t) if (den[(size_t) t] > 1e-6) { ok.push_back (t); val.push_back (num[(size_t) t] / std::max (den[(size_t) t], 1e-12)); }
    if (ok.empty()) return L;
    size_t j = 0;                                       // stretches with nothing to see: straight lines between their ends
    for (int t = 0; t < T; ++t)
    {
        if (t <= ok.front()) L.d[(size_t) t] = val.front();
        else if (t >= ok.back()) L.d[(size_t) t] = val.back();
        else { while (ok[j + 1] < t) ++j; const int a = ok[j], b = ok[j + 1]; L.d[(size_t) t] = val[j] + (val[j + 1] - val[j]) * (double) (t - a) / (double) (b - a); }
    }
    return L;
}

// ---------------------------------------------------------------- the moving filter
// The moving comb on one channel. forward: y = x + g w, w(t) = u(t - d(t)), u = x + fb w. inverse: x = y - g w with the
// same w. d: delay in samples for every sample (at least 1), read with a cubic through four points (linear under 2).
inline Vec run (const Vec& x, const Vec& d, double g, double fb, bool inverse)
{
    const long n = (long) x.size(); Vec u ((size_t) n, 0.0), out ((size_t) n, 0.0);
    for (long t = 0; t < n; ++t)
    {
        const double dd = std::max (d[(size_t) t], 1.0), p = (double) t - dd, fl = std::floor (p), f = p - fl; const long i = (long) fl; double w = 0.0;
        if (i >= 1 && dd >= 2.0)
        {
            const double a0 = u[(size_t) (i - 1)], a1 = u[(size_t) i], a2 = u[(size_t) (i + 1)], a3 = i + 2 < t ? u[(size_t) (i + 2)] : u[(size_t) (i + 1)];
            w = a1 + 0.5 * f * (a2 - a0 + f * (2 * a0 - 5 * a1 + 4 * a2 - a3 + f * (3 * (a1 - a2) + a3 - a0)));
        }
        else if (i >= 0) w = u[(size_t) i] * (1 - f) + u[(size_t) (i + 1)] * f;
        if (inverse) { const double v = x[(size_t) t] - g * w; out[(size_t) t] = v; u[(size_t) t] = v + fb * w; }
        else { out[(size_t) t] = x[(size_t) t] + g * w; u[(size_t) t] = x[(size_t) t] + fb * w; }
    }
    return out;
}
// The same with a gain and a feedback for every sample (a long sound: each piece has its own, see find).
inline Vec runVar (const Vec& x, const Vec& d, const Vec& g, const Vec& fb, bool inverse)
{
    const long n = (long) x.size(); Vec u ((size_t) n, 0.0), out ((size_t) n, 0.0);
    for (long t = 0; t < n; ++t)
    {
        const double dd = std::max (d[(size_t) t], 1.0), p = (double) t - dd, fl = std::floor (p), f = p - fl; const long i = (long) fl; double w = 0.0;
        if (i >= 1 && dd >= 2.0)
        {
            const double a0 = u[(size_t) (i - 1)], a1 = u[(size_t) i], a2 = u[(size_t) (i + 1)], a3 = i + 2 < t ? u[(size_t) (i + 2)] : u[(size_t) (i + 1)];
            w = a1 + 0.5 * f * (a2 - a0 + f * (2 * a0 - 5 * a1 + 4 * a2 - a3 + f * (3 * (a1 - a2) + a3 - a0)));
        }
        else if (i >= 0) w = u[(size_t) i] * (1 - f) + u[(size_t) (i + 1)] * f;
        if (inverse) { const double v = x[(size_t) t] - g[(size_t) t] * w; out[(size_t) t] = v; u[(size_t) t] = v + fb[(size_t) t] * w; }
        else { out[(size_t) t] = x[(size_t) t] + g[(size_t) t] * w; u[(size_t) t] = x[(size_t) t] + fb[(size_t) t] * w; }
    }
    return out;
}
// The line (one value per frame) for every sample of a sound of n samples.
inline Vec perSample (const Vec& d, size_t n)
{
    Vec o (n); const long T = (long) d.size();
    for (size_t s = 0; s < n; ++s)
    {
        const double pos = (double) s / HOP; const long i = (long) std::floor (pos);
        o[s] = i + 1 < T ? d[(size_t) i] + (d[(size_t) i + 1] - d[(size_t) i]) * (pos - (double) i) : d[(size_t) (T - 1)];
    }
    return o;
}
// Gain and feedback of the copy from the picture's height at the delay (c1) and at twice the delay (c2): the filter
// (1 + r z^-d) / (1 - fb z^-d) has c1 = r + fb and c2 = (fb^2 - r^2) / 2; the copy's gain is g = r + fb.
struct GainFb { double g = 0, fb = 0; };
inline GainFb fromPeaks (double c1, double c2)
{
    if (std::abs (c1) < 1e-6) return {};
    const double dif = std::clamp (2 * c2 / c1, -1.6, 1.6), fb = std::clamp (0.5 * (c1 + dif), -0.9, 0.9), r = std::clamp (0.5 * (c1 - dif), -0.97, 0.97);
    return { r + fb, fb };
}

// ---------------------------------------------------------------- the movement as a periodic curve
// delay(t) = lo + (hi - lo) * shape (rate * t + phase): kind 0 = sine, 1 = triangle, both 0 .. 1.
struct Curve { int kind = 0; double rate = 0, phase = 0, lo = 0, hi = 0; bool valid = false; };
inline double lfoShape (double ph, int kind) { return kind == 0 ? 0.5 * (1 + std::sin (2 * PI * ph)) : 1 - std::abs (2 * (ph - std::floor (ph)) - 1); }
inline Vec lfoCurve (int T, const Curve& p) { Vec d ((size_t) T); for (int t = 0; t < T; ++t) d[(size_t) t] = p.lo + (p.hi - p.lo) * lfoShape (p.rate * ((double) t / FPS) + p.phase, p.kind); return d; }
// The periodic curve nearest the measured line d (weights w): least squares over a grid of rates and phases (a frame's
// error counts up to 8 samples), the best four fitted once more on the frames they agree with, then moved step by step
// to where the picture holds the most. J: what the picture holds along the curve.
inline Curve lfoFit (const Vec& E, const std::vector<char>& act, int T, int sign, const Vec& d, const Vec& w, double& J)
{
    Curve none; double sw = 0, md = 0; for (int t = 0; t < T; ++t) { sw += w[(size_t) t]; md += d[(size_t) t] * w[(size_t) t]; }
    if (sw <= 0) return none;
    md /= sw;
    struct Cand { double e; int kind; double r, ph, lo, hi; }; std::vector<Cand> found; Vec s ((size_t) T);
    for (int kind = 0; kind < 2; ++kind)
        for (int ri = 0; ri < 160; ++ri)
        {
            const double l0 = std::log10 (0.06), stp = (std::log10 (4.0) - l0) / 159.0;        // numpy.geomspace (0.06, 4.0, 160)
            const double r = ri == 0 ? 0.06 : ri == 159 ? 4.0 : std::pow (10.0, (double) ri * stp + l0); Cand best { 0, kind, r, 0, 0, 0 }; bool have = false;
            for (int pj = 0; pj < 32; ++pj)
            {
                const double ph = (double) pj / 32.0; double ms = 0;
                for (int t = 0; t < T; ++t) { s[(size_t) t] = lfoShape (r * ((double) t / FPS) + ph, kind); ms += s[(size_t) t] * w[(size_t) t]; }
                ms /= sw; double v = 0, c = 0;
                for (int t = 0; t < T; ++t) { const double u = s[(size_t) t] - ms; v += u * u * w[(size_t) t]; c += u * (d[(size_t) t] - md) * w[(size_t) t]; }
                const double b = c / std::max (v, 1e-12), a = md - b * ms; double e = 0;
                for (int t = 0; t < T; ++t) { const double q = d[(size_t) t] - a - b * s[(size_t) t]; e += std::min (q * q, 64.0) * w[(size_t) t]; }
                e /= sw; if (v < 1e-6 * sw || b <= 0) e = 1e9;
                if (! have || e < best.e) { have = true; best = { e, kind, r, ph, a, a + b }; }
            }
            found.push_back (best);
        }
    std::stable_sort (found.begin(), found.end(), [] (const Cand& p, const Cand& q) { return p.e < q.e; });
    Curve bestP; double bestJ = 0; bool got = false;
    for (size_t i = 0; i < std::min<size_t> (4, found.size()); ++i)
    {
        Curve p; p.kind = found[i].kind; p.rate = found[i].r; p.phase = found[i].ph; p.lo = found[i].lo; p.hi = found[i].hi; p.valid = true;
        {
            double swi = 0, ms = 0, mdi = 0; int cnt = 0;
            for (int t = 0; t < T; ++t)
            {
                s[(size_t) t] = lfoShape (p.rate * ((double) t / FPS) + p.phase, p.kind);
                const bool ok = std::abs (d[(size_t) t] - p.lo - (p.hi - p.lo) * s[(size_t) t]) < 6.0 && w[(size_t) t] > 0; if (! ok) continue;
                ++cnt; swi += w[(size_t) t]; ms += s[(size_t) t] * w[(size_t) t]; mdi += d[(size_t) t] * w[(size_t) t];
            }
            if (cnt >= 8)
            {
                ms /= swi; mdi /= swi; double v = 0, c = 0; const double lo0 = p.lo, hi0 = p.hi;
                for (int t = 0; t < T; ++t)
                {
                    const bool ok = std::abs (d[(size_t) t] - lo0 - (hi0 - lo0) * s[(size_t) t]) < 6.0 && w[(size_t) t] > 0; if (! ok) continue;
                    const double u = s[(size_t) t] - ms; v += u * u * w[(size_t) t]; c += u * (d[(size_t) t] - mdi) * w[(size_t) t];
                }
                if (v > 1e-9) { const double b = c / v; p.lo = mdi - b * ms; p.hi = mdi - b * ms + b; }
            }
        }
        double Jp = collected (E, act, T, sign, lfoCurve (T, p)); double step[4] = { 0.02 * p.rate + 0.002, 0.01, 1.0, 1.0 };
        for (int it = 0; it < 40; ++it)
        {
            bool moved = false;
            for (int k = 0; k < 4; ++k)
                for (int sg = 0; sg < 2; ++sg)
                {
                    Curve q = p; const double dv = (sg == 0 ? 1.0 : -1.0) * step[k];
                    if (k == 0) q.rate += dv; else if (k == 1) q.phase += dv; else if (k == 2) q.lo += dv; else q.hi += dv;
                    if (q.lo < QLO || q.hi > QHI - 2 || q.hi - q.lo < 4 || q.rate < 0.03) continue;
                    const double Jq = collected (E, act, T, sign, lfoCurve (T, q));
                    if (Jq > Jp + 1e-9) { p = q; Jp = Jq; moved = true; }
                }
            if (! moved) { for (double& v : step) v *= 0.5; if (step[2] < 0.05) break; }
        }
        if (! got || Jp > bestJ) { got = true; bestJ = Jp; bestP = p; }
    }
    J = bestJ; return bestP;
}

// ---------------------------------------------------------------- measuring
// What was measured on one channel, with the numbers the decision is made from.
struct Found
{
    bool any = false;           // a line was followed up (false: nothing to see)
    bool followed = false;      // gain and feedback were worked out (false: the first numbers already ruled it out)
    Vec d;                      // the copy's delay per frame (samples)
    double g = 0, fb = 0;       // the copy's gain against the sound (negative: flipped) and its feedback
    int sign = 1; Curve curve;  // the periodic curve, if the movement is one
    double c1First = 0, move = 0, seen = 0, dmin = 0, dmax = 0, res = 1, res2 = 1, level = 0, expect = 0;
    double levelB = 0;          // the level change above 2 * SR / move Hz (see levelAbove)
    double still = 0;           // how much of what the picture holds along the line stands still there (0 .. 1)
    double jumps = 0;           // how much of the tracked line's way is done in jumps of 4 samples a frame or more (0 .. 1)
    bool moving = false;        // found in the moving part of the picture
    Vec weight;                 // per frame: how clearly the picture shows the line (0 .. 0.8)
    std::vector<char> act;
    Vec there;                  // per frame: 1 where the copy is, 0 where the sound has none (see presence); empty: everywhere
    bool nowhere = false;       // the line passes the rule, but no stretch of the sound shows the copy
};
// dB by which the level changed from x to y over the frames that count (frames of 1024 samples).
inline double levelChange (const Vec& x, const Vec& y, const std::vector<char>& act)
{
    const long n = (long) x.size(); double a = 0, b = 0;
    for (size_t t = 0; t < act.size(); ++t)
    {
        if (! act[t]) continue;
        for (long i = (long) t * HOP - 512; i < (long) t * HOP + 512; ++i) if (i >= 0 && i < n) { a += x[(size_t) i] * x[(size_t) i]; b += y[(size_t) i] * y[(size_t) i]; }
    }
    return 10.0 * std::log10 (std::max (b, 1e-30) / std::max (a, 1e-30));
}
// Two one-pole high-passes in a row at fc Hz.
inline Vec highPass (const Vec& x, double fc)
{
    const double a = std::exp (-2.0 * PI * fc / SR); Vec y (x.size()); double p1 = 0, q1 = 0, x1 = 0, z1 = 0;
    for (size_t i = 0; i < x.size(); ++i) { const double s = a * (p1 + x[i] - x1); x1 = x[i]; p1 = s; const double t = a * (q1 + s - z1); z1 = s; q1 = t; y[i] = t; }
    return y;
}
// The level change in the part of the spectrum where the sweep means something for the level: above 2 * SR / move Hz
// (move: how far the delay travels, in samples) every partial of the sound is carried across at least two whole teeth of
// the comb, so a copy of gain g adds 10 log10 (1 + g^2 / (1 - fb^2)) dB there whatever the sound is.
inline double levelAbove (const Vec& x, const Vec& y, const std::vector<char>& act, double move)
{
    const double fc = 2.0 * SR / std::max (move, 1.0); return levelChange (highPass (x, fc), highPass (y, fc), act);
}
constexpr double REACH = 64.0;      // samples (1.3 ms): the line must reach at least this delay; below it the picture shows the sound's own moving colour (formants)
constexpr double JUMPS = 0.30;      // at most this much of the tracked line's way may be jumps
constexpr double STILL_FREE = 0.6, STILL_PERIODIC = 0.8;        // at most this much of the line may stand still
inline bool quickTest (const Found& f) { return std::abs (f.c1First) >= 0.15 && std::abs (f.c1First) <= 1.0 && f.move >= 12 && f.seen >= 0.15 && f.dmax >= REACH; }
// The engine's test of a result: with the copy taken out the line is gone from the picture (res: what is left of it), and
// the level (above 2 * SR / move) fell by about what a copy of this gain and feedback adds (expect, dB): not less than a
// third of it (a comb pressed onto a sound that has none makes it louder), not more than 2.2 times (a note's own
// harmonics lose much more).
inline bool passes (double res, double level, double expect)
{
    if (std::abs (res) > 0.3 || expect > -0.15) return false;
    const double q = level / expect; return q >= 0.35 && q <= 2.2;
}
// The rule for one channel: is this line a moving copy of the sound?
inline bool accept (const Found& f)
{
    if (! f.any || ! f.followed || f.nowhere) return false;
    if (std::abs (f.g) < 0.2 || f.move < 12 || f.dmax < REACH || f.seen < 0.15) return false;      // too weak; standing still (the tone engine's case); too short; hardly seen
    if (std::abs (f.fb) > 0.75 || std::abs (f.c1First) > 1.0) return false;                         // rings like a pitch
    if (f.jumps > JUMPS || f.still > (f.curve.valid ? STILL_PERIODIC : STILL_FREE)) return false;   // moves like a melody, not like a sweep
    return passes (f.res, f.levelB, f.expect);
}

// ---------------------------------------------------------------- where the copy is
// A line through a parabola fitted to the W frames on either side of every frame: a copy's line comes out as it is, a
// line that wanders from one thing in the picture to the next leaves them.
inline Vec smoothLine (const Vec& d, int W = 24)
{
    const int T = (int) d.size(); Vec o ((size_t) T);
    for (int t = 0; t < T; ++t)
    {
        const int a = std::max (0, t - W), b = std::min (T - 1, t + W); double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, y0 = 0, y1 = 0, y2 = 0;
        for (int k = a; k <= b; ++k) { const double u = (double) (k - t) / W, y = d[(size_t) k], uu = u * u; s0 += 1; s1 += u; s2 += uu; s3 += uu * u; s4 += uu * uu; y0 += y; y1 += u * y; y2 += uu * y; }
        double A[3][4] = { { s0, s1, s2, y0 }, { s1, s2, s3, y1 }, { s2, s3, s4, y2 } }; bool ok = true;
        for (int i = 0; i < 3 && ok; ++i)
        {
            if (std::abs (A[i][i]) < 1e-12) { ok = false; break; }
            for (int j = i + 1; j < 3; ++j) { const double r = A[j][i] / A[i][i]; for (int q = i; q < 4; ++q) A[j][q] -= r * A[i][q]; }
        }
        if (! ok) { o[(size_t) t] = d[(size_t) t]; continue; }
        double co[3]; for (int i = 2; i >= 0; --i) { double s = A[i][3]; for (int j = i + 1; j < 3; ++j) s -= A[i][j] * co[j]; co[i] = s / A[i][i]; }
        o[(size_t) t] = co[0];
    }
    return o;
}
// Where along the sound the copy is: per frame 1 or 0 (empty: everywhere). The line is followed through the whole sound,
// also through a stretch that has no copy (a flanger switched on halfway): there it holds on to whatever the picture
// has, and the inverse filter would press a comb onto a sound that has none. So every half second is asked what the
// rule asks of the whole sound:
//  - is there a moving line? What the picture holds along the smoothed line (smoothLine; a periodic curve is smooth
//    already), less what stands still there, against the line's height on this sound (c1First): 40 % of it and more is
//    a yes (+1), 20 % cannot be told (0), nothing is a no (-1). A line that stands still is the copy at a turn of its
//    sweep or a held note: cannot be told (0).
//  - does taking the copy out take away what a copy adds (the level above 2 * SR / move, hx to ho: 0.2 to 2.8 times
//    what is expected; half a second is judged more loosely than a whole sound)? If not: a no (-0.5 or lower).
// The answers that gather the most are chosen, a change between "there" and "not there" costing HOLD: a copy that is not
// seen for a second and a half is still there, a stretch of several seconds without one is left alone.
constexpr int STRETCH = 94;         // frames: half a second
constexpr double HOLD = 3.0;
inline Vec presence (const Vec& hx, const Vec& ho, const Picture& pic, const Vec& S, const Found& f)
{
    const int T = pic.T, K = std::max (1, (T + STRETCH / 2) / STRETCH); if (K < 2 || (int) f.d.size() < T) return {};
    const Vec ds = f.curve.valid ? f.d : smoothLine (f.d); const Vec vE = along (pic.E, ds, T), vS = along (S, ds, T);
    const double ref = std::max (std::abs (f.c1First), 1e-6); const long n = (long) std::min (hx.size(), ho.size()); Vec u ((size_t) K, 0.0);
    for (int k = 0; k < K; ++k)
    {
        const int t0 = k * STRETCH, t1 = k + 1 == K ? T : t0 + STRETCH; double e = 0, s = 0; int na = 0;
        for (int t = t0; t < t1; ++t)
            if (pic.act[(size_t) t]) { ++na; e += std::clamp ((double) f.sign * vE[(size_t) t], 0.0, 0.8); s += std::clamp ((double) f.sign * vS[(size_t) t], 0.0, 0.8); }
        if (na < 12) continue;                                              // (nearly) silent: cannot be told
        e /= na; s /= na; double v = std::clamp (((e - s) / ref - 0.2) / 0.2, -1.0, 1.0); if (v < 0 && e >= 0.4 * ref) v = 0.0;
        double pa = 0, pb = 0; for (long i = (long) t0 * HOP; i < std::min ((long) t1 * HOP, n); ++i) { pa += hx[(size_t) i] * hx[(size_t) i]; pb += ho[(size_t) i] * ho[(size_t) i]; }
        if (pa > 0 && f.expect < -0.15) { const double q = 10.0 * std::log10 (std::max (pb, 1e-30) / pa) / f.expect; if (q < 0.2 || q > 2.8) v = std::min (v, -0.5); }
        u[(size_t) k] = v;
    }
    Vec B ((size_t) K * 2); std::vector<char> from ((size_t) K * 2, 0); B[0] = -u[0]; B[1] = u[0];
    for (int k = 1; k < K; ++k)
        for (int s = 0; s < 2; ++s)
        {
            const double stay = B[(size_t) (k - 1) * 2 + (size_t) s], turn = B[(size_t) (k - 1) * 2 + (size_t) (1 - s)] - HOLD;
            from[(size_t) k * 2 + (size_t) s] = (char) (turn > stay ? 1 - s : s); B[(size_t) k * 2 + (size_t) s] = std::max (stay, turn) + (s ? u[(size_t) k] : -u[(size_t) k]);
        }
    std::vector<char> on ((size_t) K); int s = B[(size_t) (K - 1) * 2 + 1] >= B[(size_t) (K - 1) * 2] ? 1 : 0, any = 0, all = 1;
    for (int k = K - 1; k >= 0; --k) { on[(size_t) k] = (char) s; any |= s; all &= s; s = from[(size_t) k * 2 + (size_t) s]; }
    if (all) return {};
    Vec th ((size_t) f.d.size(), 0.0); if (! any) return th;
    for (size_t t = 0; t < th.size(); ++t) th[t] = on[(size_t) std::min ((int) t / STRETCH, K - 1)] ? 1.0 : 0.0;
    return th;
}
// The same for every sample of the sound (n samples): 1 where the copy is, 0 where it is not, over 20 ms from one to the
// other.
inline Vec gateOf (const Found& f, size_t n)
{
    Vec w (n, 1.0); if (f.there.empty() || n == 0) return w;
    const long X = 480, T = (long) f.there.size(); Vec c (n + 1, 0.0);
    for (size_t i = 0; i < n; ++i) c[i + 1] = c[i] + f.there[(size_t) std::min ((long) (i / HOP), T - 1)];
    for (long i = 0; i < (long) n; ++i) { const long a = std::max (0L, i - X), b = std::min ((long) n, i + X + 1); w[(size_t) i] = (c[(size_t) b] - c[(size_t) a]) / (double) (b - a); }
    return w;
}
// A stretch of a found line that has the copy: frames a .. b - 1 of the line (first: the line's first frame in the
// whole sound).
struct Seg { const Found* f = nullptr; long first = 0, a = 0, b = 0; };
inline void segsOf (const Found& f, long first, std::vector<Seg>& out)
{
    const long n = (long) f.d.size(); if (n < 2) return;
    if (f.there.empty()) { out.push_back ({ &f, first, 0, n }); return; }
    for (long t = 0; t < n;)
    {
        if (f.there[(size_t) t] <= 0) { ++t; continue; }
        long e = t; while (e < n && f.there[(size_t) e] > 0) ++e;
        if (e - t >= 2) out.push_back ({ &f, first, t, e });
        t = e;
    }
}
// A stretch's line at any frame t of the whole sound: inside the stretch as measured, outside it carried on as its
// periodic curve (if it has one; the line's small departure from the curve at its end fades over 16 frames) or run
// back and forth.
inline double lineAt (const Seg& s, double t)
{
    const Found& f = *s.f; const long n = s.b - s.a; const double q = t - (double) (s.first + s.a); if (n < 2) return n ? f.d[(size_t) s.a] : 0.0;
    auto in = [&] (double r) { const long i = s.a + std::clamp ((long) std::floor (r), 0L, n - 2); return f.d[(size_t) i] + (f.d[(size_t) i + 1] - f.d[(size_t) i]) * (r - (double) (i - s.a)); };
    if (q >= 0 && q <= (double) (n - 1)) return in (q);
    if (f.curve.valid)
    {
        auto cv = [&] (double r) { return f.curve.lo + (f.curve.hi - f.curve.lo) * lfoShape (f.curve.rate * ((r + (double) s.a) / FPS) + f.curve.phase, f.curve.kind); };
        const double end = q < 0 ? 0.0 : (double) (n - 1), far = std::abs (q - end);
        return cv (q) + (f.d[(size_t) (s.a + (long) end)] - cv (end)) * std::max (0.0, 1.0 - far / 16.0);
    }
    const double m = 2.0 * (double) (n - 1); double r = std::fmod (q, m); if (r < 0) r += m; if (r > (double) (n - 1)) r = m - r;
    return in (r);
}
// The stretches' lines as one line of T frames: each stretch's own inside it; from 4 frames before the end of one
// stretch to 4 frames into the next, the two (carried on) are blended. So the line runs on through a stretch that has
// no copy, and through a piece that shows nothing.
inline Vec joinedLine (const std::vector<Seg>& F, int T)
{
    Vec d ((size_t) T, 0.0); if (F.empty()) return d;
    auto first = [] (const Seg& s) { return s.first + s.a; }; auto last = [] (const Seg& s) { return s.first + s.b - 1; };
    size_t j = 0; const long m = 4;
    for (int t = 0; t < T; ++t)
    {
        while (j + 1 < F.size() && t > first (F[j + 1]) + m) ++j;
        double v = lineAt (F[j], (double) t);
        if (j + 1 < F.size() && t >= last (F[j]) - m)
        {
            const double z0 = (double) (last (F[j]) - m), z1 = (double) (first (F[j + 1]) + m), w = std::clamp (((double) t - z0) / std::max (z1 - z0, 1.0), 0.0, 1.0);
            v = (1.0 - w) * v + w * lineAt (F[j + 1], (double) t);
        }
        d[(size_t) t] = std::clamp (v, 1.0, (double) QHI - 1.0);
    }
    return d;
}

// One line through the picture P (the whole picture E, or its moving part), followed up.
// S: the picture's static part (staticPart).
inline Found candidate (const Vec& x, const Picture& pic, const Vec& P, const Vec& S, int rounds = 3, int paths = 2, const Stop& stop = {})
{
    Found f; const int T = pic.T; const std::vector<char>& act = pic.act; const Vec& E = pic.E;
    bool have = false; double col = 0; int sign = 1; Line L; std::vector<int> path;
    for (int sg : { 1, -1 })
    {
        Vec Pk = P;
        for (int k = 0; k < paths; ++k)
        {
            if (stop && stop()) return f;
            const auto tr = track (Pk, act, T, sg); Line l = refine (P, act, T, tr.first, sg); const double c = collected (P, act, T, sg, l.d);
            if (! have || c > col) { have = true; col = c; sign = sg; L = std::move (l); path = tr.first; }
            if (k + 1 < paths) for (int t = 0; t < T; ++t) for (int q = std::max (0, tr.first[(size_t) t] - 6); q < std::min (QHI, tr.first[(size_t) t] + 7); ++q) Pk[(size_t) t * QHI + (size_t) q] = 0.0;
        }
    }
    Vec d = L.d, hgt = L.hgt, w0 ((size_t) T); int n0 = 0;
    for (int t = 0; t < T; ++t) { w0[(size_t) t] = act[(size_t) t] ? std::clamp (hgt[(size_t) t], 0.0, 0.8) : 0.0; n0 += w0[(size_t) t] > 0.05; }
    if (n0 >= 12)
    {
        Vec w2 ((size_t) T); for (int t = 0; t < T; ++t) w2[(size_t) t] = w0[(size_t) t] * w0[(size_t) t];
        double J = 0; const Curve c = lfoFit (P, act, T, sign, d, w2, J);
        if (c.valid && J >= 0.8 * collected (P, act, T, sign, d))
        {
            const Vec dc = lfoCurve (T, c); Vec a ((size_t) T), b ((size_t) T);
            for (int t = 0; t < T; ++t) { const double dev = d[(size_t) t] - dc[(size_t) t], ww = std::abs (dev) < 6.0 ? w2[(size_t) t] : 0.0; a[(size_t) t] = ww * dev; b[(size_t) t] = ww; }
            const Vec ga = gauss (a, 4.0), gb = gauss (b, 4.0); f.curve = c;
            for (int t = 0; t < T; ++t) d[(size_t) t] = dc[(size_t) t] + std::clamp (ga[(size_t) t] / (gb[(size_t) t] + 0.02), -4.0, 4.0);
            const Vec v = along (P, d, T); for (int t = 0; t < T; ++t) hgt[(size_t) t] = std::max ((double) sign * v[(size_t) t], 0.0);
        }
    }
    Vec w ((size_t) T); double sw = 0; int ns = 0, na = 0;
    for (int t = 0; t < T; ++t) { w[(size_t) t] = act[(size_t) t] ? std::clamp (hgt[(size_t) t], 0.0, 0.8) : 0.0; sw += w[(size_t) t]; ns += w[(size_t) t] > 0.05; na += act[(size_t) t] ? 1 : 0; }
    if (sw <= 1e-9 || ns < 12) return f;
    const Vec v1 = along (E, d, T, 1.0), v2 = along (E, d, T, 2.0); Vec sv; double s1 = 0, s2 = 0, lo = 1e300, hi = -1e300;
    for (int t = 0; t < T; ++t)
    {
        s1 += w[(size_t) t] * v1[(size_t) t]; s2 += w[(size_t) t] * v2[(size_t) t];
        if (w[(size_t) t] > 0.05) { sv.push_back ((double) sign * v1[(size_t) t]); lo = std::min (lo, d[(size_t) t]); hi = std::max (hi, d[(size_t) t]); }
    }
    double c1 = (double) sign * percentile (sv, 75.0), c2 = std::abs (s1) > 1e-9 ? s2 / s1 * c1 : 0.0;
    f.any = true; f.d = d; f.sign = sign; f.weight = w; f.act = act; f.c1First = c1; f.move = hi - lo; f.seen = (double) ns / (double) std::max (na, 1); f.dmin = lo; f.dmax = hi;
    if (! quickTest (f)) return f;
    // gain and feedback are corrected until taking the copy out leaves nothing along the line and along twice the line
    const Vec ds = perSample (d, x.size()); GainFb gf = fromPeaks (c1, c2); Vec out; double dlo = 1e300, dhi = -1e300; for (double v : d) { dlo = std::min (dlo, v); dhi = std::max (dhi, v); }
    const std::vector<Need> need { { std::floor (dlo) - 1, std::ceil (dhi) + 2 }, { std::floor (2 * dlo) - 1, std::ceil (2 * dhi) + 2 } };
    for (int k = 0; k <= rounds; ++k)
    {
        if (stop && stop()) { f.any = false; return f; }
        out = run (x, ds, gf.g, gf.fb, true); const Picture P2 = picture (out, &need, &d); const Vec r1 = along (P2.E, d, T, 1.0), r2 = along (P2.E, d, T, 2.0);
        double a = 0, b = 0; for (int t = 0; t < T; ++t) { a += w[(size_t) t] * r1[(size_t) t]; b += w[(size_t) t] * r2[(size_t) t]; }
        f.res = a / s1; f.res2 = b / s1;
        if (k == rounds) break;
        c1 = f.res < 0.9 ? c1 / (1 - f.res) : c1 * 1.5; c1 = std::clamp (c1, -1.5, 1.5); c2 = c2 + f.res2 * c1; gf = fromPeaks (c1, c2);
    }
    f.followed = true; f.g = gf.g; f.fb = gf.fb; f.level = levelChange (x, out, act); f.expect = -10.0 * std::log10 (1 + gf.g * gf.g / std::max (1 - gf.fb * gf.fb, 1e-3));
    const double fc = 2.0 * SR / std::max (f.move, 1.0); const Vec hx = highPass (x, fc), ho = highPass (out, fc); f.levelB = levelChange (hx, ho, act);      // = levelAbove (x, out, act, f.move)
    // what tells a moving copy from a melody's own line: how much of the line stands still, how much of its way is jumps
    {
        const Vec vs = along (S, d, T); double tot = 0, st = 0, way = 0, big = 0;
        for (int t = 0; t < T; ++t) { tot += w[(size_t) t] * std::max ((double) sign * v1[(size_t) t], 0.0); st += w[(size_t) t] * std::max ((double) sign * vs[(size_t) t], 0.0); }
        for (int t = 0; t + 1 < T; ++t) if (act[(size_t) t] && act[(size_t) t + 1]) { const int k = std::abs (path[(size_t) t + 1] - path[(size_t) t]); way += k; if (k >= 4) big += k; }
        f.still = tot > 1e-9 ? st / tot : 0.0; f.jumps = way > 0 ? big / way : 0.0;
    }
    if (accept (f))                 // where along the sound the copy is
    {
        f.there = presence (hx, ho, pic, S, f); bool any = f.there.empty(); for (double v : f.there) any = any || v > 0;
        f.nowhere = ! any;
    }
    return f;
}
// One channel: the candidates (a line through the whole picture, a line through its moving part), each followed up.
inline std::vector<Found> measureChannel (const Vec& x, const Stop& stop = {})
{
    std::vector<Found> out; const Picture pic = picture (x); int na = 0; for (char a : pic.act) na += a ? 1 : 0;
    if (na < 16 || (stop && stop())) return out;
    const Vec S = staticPart (pic);
    { Found f = candidate (x, pic, pic.E, S, 3, 2, stop); if (f.any) { f.moving = false; out.push_back (std::move (f)); } }
    if (stop && stop()) return out;
    { Vec M (S.size()); for (size_t i = 0; i < M.size(); ++i) M[i] = pic.E[i] - S[i]; Found f = candidate (x, pic, M, S, 3, 2, stop); if (f.any) { f.moving = true; out.push_back (std::move (f)); } }
    return out;
}

// What was found on a sound: per channel the copy (or nothing on that channel: a silent one). found: false = no report.
// n: the length of the sound it was measured on (samples). stopped: the measuring was given up (nothing in here counts).
// parts: a long sound is measured in pieces (see find); then ch[] is what the pieces' lines make together, and each
// piece keeps its own line, gain and feedback for taking the copy out. shown: how many pieces show the copy, of `sounding`.
struct Part { size_t a = 0, b = 0; bool found = false, has[2] = { false, false }; Found ch[2]; };
struct Result { bool found = false, stopped = false; size_t n = 0; Found ch[2]; bool has[2] = { false, false }; std::vector<Found> all[2]; std::vector<Part> parts; int shown = 0, sounding = 0; };
// One sound measured as a whole.
inline Result findPiece (const Stereo& x, const Stop& stop = {})
{
    Result R; R.n = x.size(); const bool same = x.l == x.r; double e[2] = { 0, 0 };
    for (size_t i = 0; i < x.size(); ++i) { e[0] += x.l[i] * x.l[i]; e[1] += x.r[i] * x.r[i]; }
    const double top = std::max (e[0], e[1]); const bool quiet[2] = { e[0] < 1e-4 * top, e[1] < 1e-4 * top };
    std::future<std::vector<Found>> right;              // left and right are measured at the same time (stop is asked from both threads)
    if (! same && ! quiet[0] && ! quiet[1]) right = std::async (std::launch::async, [&x, &stop] { return measureChannel (x.r, stop); });
    for (int c = 0; c < 2; ++c)
    {
        if (c == 1 && same) { R.ch[1] = R.ch[0]; R.has[1] = R.has[0]; continue; }
        if (quiet[c]) continue;
        R.all[c] = c == 1 && right.valid() ? right.get() : measureChannel (c == 0 ? x.l : x.r, stop); double best = -1;
        if (stop && stop()) { if (right.valid()) right.wait(); R.stopped = true; return R; }
        for (auto& f : R.all[c]) if (accept (f)) { const double k = std::abs (f.g) * f.seen; if (k > best) { best = k; R.ch[c] = f; R.has[c] = true; } }
    }
    for (int c = 0; c < 2; ++c) if (! R.has[c] && ! quiet[c]) return R;      // one side shows nothing: no report
    if (! R.has[0] && ! R.has[1]) return R;
    if (R.has[0] && R.has[1] && R.ch[0].g * R.ch[1].g < 0) return R;
    R.found = true; R.shown = R.sounding = 1; return R;
}

// ---------------------------------------------------------------- a whole sound, however long
// The picture of a long sound takes a great deal of memory (a megabyte a second, several times over), and the rule was
// set on sounds of 6 s. So a sound of 24 s or more is measured in pieces of 12 to 24 s, each like a sound of its own.
// Taken out, every piece that shows the copy loses it with its own line, gain and feedback: one moving filter through
// the whole sound, whose gain is 0 where a piece shows nothing (and where a piece has no copy: presence) and whose
// settings run into each other over 20 ms from piece to piece. Put onto another sound, the lines of the stretches that
// have the copy run on as one line (through a piece that shows nothing, and through a stretch without a copy, its
// neighbours' lines carry on, as their periodic curve or back and forth) with the pieces' middle gain and feedback, over
// the whole of that sound: a piece that shows nothing may still have the copy (it is often simply not found), so
// "nothing found here" is no reason to leave the other sound dry there.
constexpr double PIECE = 12.0;
// For every sample the gain and feedback of the piece it lies in (0 where the piece shows nothing, and where the piece
// has no copy: gateOf), run into the next piece's over 20 ms across the border. cover: how much of a stretch that has
// the copy each sample belongs to (0 .. 1).
inline void settings (const std::vector<Part>& parts, int c, size_t n, Vec& g, Vec& fb, Vec& cover)
{
    g.assign (n, 0.0); fb.assign (n, 0.0); cover.assign (n, 0.0); const long X = 480;
    for (size_t j = 0; j < parts.size(); ++j)
    {
        const Part& p = parts[j]; if (! p.found || ! p.has[c] || p.a >= n) continue;
        const long lo = p.a == 0 ? 0 : (long) p.a - X, hi = j + 1 == parts.size() ? (long) n : std::min ((long) n, (long) p.b + X), len = (long) (p.b - p.a);
        const Vec gate = p.ch[c].there.empty() ? Vec() : gateOf (p.ch[c], (size_t) len);
        for (long i = lo; i < hi; ++i)
        {
            double w = 1.0;
            if (p.a > 0 && i < (long) p.a + X) w = (double) (i - ((long) p.a - X)) / (2.0 * X);
            if (j + 1 < parts.size() && i >= (long) p.b - X) w = std::min (w, (double) ((long) p.b + X - i) / (2.0 * X));
            if (! gate.empty()) w *= gate[(size_t) std::clamp (i - (long) p.a, 0L, len - 1)];
            g[(size_t) i] += w * p.ch[c].g; fb[(size_t) i] += w * p.ch[c].fb; cover[(size_t) i] += w;
        }
    }
}
// What the pieces make together (R.parts filled in): the report, and the line, gain and feedback that are shown and put
// onto another sound. Periodic only if every piece is, at one speed (within 5 %).
inline void join (Result& R)
{
    const int T = (int) (R.n / HOP) + 1; R.shown = 0; for (const Part& p : R.parts) R.shown += p.found ? 1 : 0;
    for (int c = 0; c < 2; ++c)
    {
        std::vector<const Part*> F; for (const Part& p : R.parts) if (p.found && p.has[c]) F.push_back (&p);
        R.has[c] = ! F.empty(); if (F.empty()) continue;
        Vec gs, fbs, rates; bool periodic = true; const Part* best = nullptr; double bk = -1;
        Found f; f.any = f.followed = true; { std::vector<Seg> segs; for (const Part* p : F) segsOf (p->ch[c], (long) (p->a / HOP), segs); f.d = joinedLine (segs, T); } f.weight.assign ((size_t) T, 0.0); f.act.assign ((size_t) T, 0); f.there.assign ((size_t) T, 0.0);
        for (const Part* p : F)
        {
            const Found& q = p->ch[c]; const size_t first = p->a / HOP;
            for (size_t t = 0; t < q.weight.size() && first + t < (size_t) T; ++t) { f.weight[first + t] = q.weight[t]; f.act[first + t] = q.act[t]; f.there[first + t] = q.there.empty() ? 1.0 : q.there[t]; }
            gs.push_back (q.g); fbs.push_back (q.fb); periodic = periodic && q.curve.valid; if (q.curve.valid) rates.push_back (q.curve.rate);
            const double k = std::abs (q.g) * q.seen; if (k > bk) { bk = k; best = p; }
        }
        const Found& b = best->ch[c]; f.g = percentile (gs, 50.0); f.fb = percentile (fbs, 50.0); f.sign = f.g < 0 ? -1 : 1; f.moving = b.moving;
        f.c1First = b.c1First; f.res = b.res; f.res2 = b.res2; f.level = b.level; f.expect = b.expect; f.levelB = b.levelB; f.still = b.still; f.jumps = b.jumps;
        double lo = 1e300, hi = -1e300; int ns = 0, na = 0;
        for (int t = 0; t < T; ++t) { na += f.act[(size_t) t] ? 1 : 0; if (f.weight[(size_t) t] > 0.05 && f.there[(size_t) t] > 0) { ++ns; lo = std::min (lo, f.d[(size_t) t]); hi = std::max (hi, f.d[(size_t) t]); } }
        if (ns == 0) { lo = b.dmin; hi = b.dmax; }
        { bool all = true; for (double v : f.there) all = all && v > 0; if (all) f.there.clear(); }
        f.dmin = lo; f.dmax = hi; f.move = hi - lo; f.seen = (double) ns / (double) std::max (na, 1);
        if (periodic) { const double r = percentile (rates, 50.0); for (double v : rates) periodic = periodic && std::abs (v / r - 1.0) <= 0.05; }
        if (periodic) { f.curve = b.curve; f.curve.phase -= f.curve.rate * (double) (best->a / HOP) / FPS; }
        R.ch[c] = std::move (f);
    }
    R.found = R.shown > 0 && (R.has[0] || R.has[1]);
}
inline Result find (const Stereo& x, const Stop& stop = {})
{
    const size_t n = x.size();
    if ((double) n < 2.0 * PIECE * SR)
    {
        Result R = findPiece (x, stop);
        for (int c = 0; c < 2 && R.found; ++c)          // the copy on a part of the sound: through the rest the line is carried on from where it is (for showing, and for another sound)
        {
            if (! R.has[c] || R.ch[c].there.empty()) continue;
            const Found as = R.ch[c]; std::vector<Seg> segs; segsOf (as, 0, segs);
            if (! segs.empty()) R.ch[c].d = joinedLine (segs, (int) as.d.size());
        }
        return R;
    }
    Result R; R.n = n; const size_t np = (size_t) ((double) n / (PIECE * SR)); Vec pw (np, 0.0); double top = 0;
    for (size_t j = 0; j < np; ++j)
    {
        Part p; p.a = (j * n / np) / HOP * HOP; p.b = j + 1 == np ? n : ((j + 1) * n / np) / HOP * HOP;
        for (size_t i = p.a; i < p.b; ++i) pw[j] += x.l[i] * x.l[i] + x.r[i] * x.r[i];
        pw[j] /= (double) std::max<size_t> (1, p.b - p.a); top = std::max (top, pw[j]); R.parts.push_back (std::move (p));
    }
    for (size_t j = 0; j < np; ++j)
    {
        Part& p = R.parts[j]; if (! (top > 0 && pw[j] > top * 1e-4)) continue;       // a piece 40 dB under the loudest one says nothing
        ++R.sounding; Result r = findPiece (x.slice (p.a, p.b - p.a), stop);
        if (r.stopped || (stop && stop())) { R.stopped = true; return R; }
        p.found = r.found; if (r.found) for (int c = 0; c < 2; ++c) { p.has[c] = r.has[c]; if (r.has[c]) p.ch[c] = std::move (r.ch[c]); }
    }
    join (R); return R;
}
// Where along the sound the copy is, for showing: stretches from .. to (samples), on either channel. One stretch over
// the whole sound: everywhere.
inline std::vector<std::pair<size_t, size_t>> where (const Result& R)
{
    std::vector<std::pair<size_t, size_t>> out; if (! R.found) return out;
    const size_t T = std::max (R.has[0] ? R.ch[0].d.size() : (size_t) 0, R.has[1] ? R.ch[1].d.size() : (size_t) 0);
    auto on = [&] (size_t t) { for (int c = 0; c < 2; ++c) if (R.has[c] && (R.ch[c].there.empty() || (t < R.ch[c].there.size() && R.ch[c].there[t] > 0))) return true; return false; };
    for (size_t t = 0; t < T;)
    {
        if (! on (t)) { ++t; continue; }
        size_t e = t; while (e < T && on (e)) ++e;
        out.push_back ({ std::min (t * (size_t) HOP, R.n), e >= T ? R.n : std::min (e * (size_t) HOP, R.n) }); t = e;
    }
    return out;
}

// The found line for T frames, as the group's faders want it. speed: the line is read that many times as fast; where it
// ends (a longer sound, or speed above 1) a periodic movement carries on as its curve, a free one runs back and forth.
// sweepK: the line's excursion around its middle is scaled (0: a comb that stands still at the middle, 1: as found).
inline Vec movement (const Found& f, int T, double speed = 1.0, double sweepK = 1.0)
{
    const int n = (int) f.d.size();
    if (speed == 1.0 && sweepK == 1.0 && T <= n) return Vec (f.d.begin(), f.d.begin() + T);
    auto at = [n] (const Vec& v, double q)              // v read between its frames (q within 0 .. n - 1)
    {
        const double fl = std::floor (q); const long i = (long) fl;
        return i >= n - 1 ? v[(size_t) n - 1] : i < 0 ? v[0] : v[(size_t) i] + (v[(size_t) i + 1] - v[(size_t) i]) * (q - fl);
    };
    Vec out ((size_t) T); double mid = 0;
    if (f.curve.valid)                                  // the curve at any time, plus the line's own small departures while it lasts
    {
        const Vec dc = lfoCurve (n, f.curve); Vec corr ((size_t) n); for (int t = 0; t < n; ++t) corr[(size_t) t] = f.d[(size_t) t] - dc[(size_t) t];
        for (int t = 0; t < T; ++t)
        {
            const double pos = (double) t * speed;
            out[(size_t) t] = f.curve.lo + (f.curve.hi - f.curve.lo) * lfoShape (f.curve.rate * pos / FPS + f.curve.phase, f.curve.kind) + (pos <= n - 1 ? at (corr, pos) : 0.0);
        }
        mid = 0.5 * (f.curve.lo + f.curve.hi);
    }
    else
    {
        const double m = 2.0 * (n - 1);
        for (int t = 0; t < T; ++t) { double q = std::fmod ((double) t * speed, m); if (q > n - 1) q = m - q; out[(size_t) t] = at (f.d, q); }
        Vec s; for (int t = 0; t < n; ++t) if (f.weight[(size_t) t] > 0.05 && (f.there.empty() || f.there[(size_t) t] > 0)) s.push_back (f.d[(size_t) t]);
        mid = percentile (s.empty() ? f.d : s, 50.0);
    }
    for (double& v : out) v = std::clamp (mid + sweepK * (v - mid), 1.0, (double) QHI - 1.0);
    return out;
}

// x through what was found. remove: the found filter's inverse first (x is the sound it was found on); then the filter
// itself at `strength` times the copy's gain (0: nothing), its movement and feedback as the faders want them.
inline Stereo render (const Stereo& x, const Result& R, bool remove, double strength, double speed = 1.0, double sweepK = 1.0, double feedback = 0.0)
{
    Stereo y = x;
    for (int c = 0; c < 2; ++c)
    {
        if (! R.has[c]) continue;
        const Found& f = R.ch[c]; Vec& v = c == 0 ? y.l : y.r;
        if (! remove || (R.parts.empty() && f.there.empty()))        // onto another sound: all of it. Its own sound, the copy everywhere: one filter
        {
            if (remove) v = run (v, perSample (f.d, v.size()), f.g, f.fb, true);
            if (strength > 0) v = run (v, perSample (movement (f, (int) (v.size() / HOP) + 1, speed, sweepK), v.size()), std::clamp (strength * f.g, -0.98, 0.98), std::clamp (f.fb + feedback, -0.9, 0.9), false);
            continue;
        }
        // the copy on a part of the sound, or a long sound: out where it is (a long sound: of every piece that shows the
        // copy with that piece's own settings), and back in only there
        const size_t n = v.size(); Vec g, fb, cover;
        if (R.parts.empty()) { cover = gateOf (f, n); g.resize (n); fb.resize (n); for (size_t i = 0; i < n; ++i) { g[i] = f.g * cover[i]; fb[i] = f.fb * cover[i]; } }
        else settings (R.parts, c, n, g, fb, cover);
        v = runVar (v, perSample (f.d, n), g, fb, true);
        if (strength > 0)
        {
            for (size_t i = 0; i < n; ++i) { g[i] = std::clamp (strength * g[i], -0.98, 0.98); fb[i] = std::clamp (fb[i] + feedback * cover[i], -0.9, 0.9); }
            v = runVar (v, perSample (movement (f, (int) (n / HOP) + 1, speed, sweepK), n), g, fb, false);
        }
    }
    return y;
}
// s = -1: the found effect out of x (the sound it was found on); -1 < s < 0: partly out (strength 1 + s is left).
// s = +1: onto x (another sound) as found; 0 < s <= 2: scaled. s = 0: x itself (on its own sound, with the faders moved:
// the copy taken out and put back as they say, render (x, R, true, 1, ...)).
// The faders (see movement): speed, sweepK, and feedback (added to the found feedback) shape the copy that is put on.
inline Stereo apply (const Stereo& x, const Result& R, double s, double speed = 1.0, double sweepK = 1.0, double feedback = 0.0)
{
    if (! R.found || s == 0) return x;
    return s < 0 ? render (x, R, true, 1.0 + s, speed, sweepK, feedback) : render (x, R, false, std::min (s, 2.0), speed, sweepK, feedback);
}

// The engine's own test of a result, on the sound it was found on: with the copy taken out, (1) how much of the line is
// left in the picture (res: 0 = gone, 1 = as before), (2) by how much the level fell above 2 * SR / move (level, dB)
// against what a copy of this gain and feedback adds (expect, dB). ok: both as they must be (passes) on every channel
// that has a copy. find() reports only what passes; the app can call this again on whatever it is about to play.
struct Check { bool ok = false; double res[2] = { 0, 0 }, level[2] = { 0, 0 }, expect[2] = { 0, 0 }; };
inline Check check (const Stereo& x, const Result& R)
{
    Check C; if (! R.found) return C;
    if (! R.parts.empty())                              // a long sound: every piece that shows the copy by itself; the numbers of the worst one
    {
        C.ok = true;
        for (const Part& p : R.parts)
        {
            if (! p.found || p.b > x.size()) continue;
            Result r; r.found = true; r.n = p.b - p.a; for (int c = 0; c < 2; ++c) { r.has[c] = p.has[c]; r.ch[c] = p.ch[c]; }
            const Check q = check (x.slice (p.a, p.b - p.a), r); C.ok = C.ok && q.ok;
            for (int c = 0; c < 2; ++c) if (p.has[c] && std::abs (q.res[c]) >= std::abs (C.res[c])) { C.res[c] = q.res[c]; C.level[c] = q.level[c]; C.expect[c] = q.expect[c]; }
        }
        return C;
    }
    C.ok = true;
    for (int c = 0; c < 2; ++c)
    {
        if (! R.has[c]) continue;
        const Found& f = R.ch[c]; const Vec& v = c == 0 ? x.l : x.r; const int T = (int) f.d.size(); const bool part = ! f.there.empty();
        const Picture P0 = picture (v); Vec out; std::vector<char> act = P0.act;
        if (! part) out = run (v, perSample (f.d, v.size()), f.g, f.fb, true);
        else            // the copy on a part of the sound: taken out there, and judged there
        {
            const Vec gate = gateOf (f, v.size()); Vec g (v.size()), fb (v.size()); for (size_t i = 0; i < v.size(); ++i) { g[i] = f.g * gate[i]; fb[i] = f.fb * gate[i]; }
            out = runVar (v, perSample (f.d, v.size()), g, fb, true); for (int t = 0; t < T && t < P0.T; ++t) if (f.there[(size_t) t] <= 0) act[(size_t) t] = 0;
        }
        const Picture P1 = picture (out); const Vec a0 = along (P0.E, f.d, T), a1 = along (P1.E, f.d, T); double s0 = 0, s1 = 0;
        for (int t = 0; t < T && t < P0.T; ++t) { if (part && f.there[(size_t) t] <= 0) continue; s0 += f.weight[(size_t) t] * a0[(size_t) t]; s1 += f.weight[(size_t) t] * a1[(size_t) t]; }
        C.res[c] = std::abs (s0) > 1e-9 ? s1 / s0 : 1.0; C.level[c] = levelAbove (v, out, act, f.move); C.expect[c] = -10.0 * std::log10 (1 + f.g * f.g / std::max (1 - f.fb * f.fb, 1e-3));
        C.ok = C.ok && passes (C.res[c], C.level[c], C.expect[c]);
    }
    return C;
}

} // namespace sweep
} // namespace dr
