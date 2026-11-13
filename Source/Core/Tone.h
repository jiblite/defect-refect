#pragma once
// The tone engine: what the long-term spectrum of a sound says, and what can be done with it.
//   analyse()     a sound's long-term spectra and the short reflections found on it (a comb filter: delay and gain)
//   spectra()     the long-term spectra alone (what Tone Match compares)
//   matchCurve()  Tone Match: the curve that gives one sound the long-term spectrum of another, smoothed by Detail
//   combCurve()   the curve of the found reflections
//   gainFilter()  a curve as a minimum-phase filter on a sound, with soft caps;  toIr(): the curve as an impulse response
// A curve is in dB on the analysis bins: 4097 points, 5.86 Hz apart, 0 .. 24 kHz. Evenly spaced bins, not a
// log-frequency grid: a comb's teeth and narrow resonances are evenly spaced in Hz and too dense for a log grid
// at the top (a 1/24-octave grid loses a fifth of a comb).
// A port of Research/tone/tone.py (its default path and Tone Match; the optional model for a blind guess at an EQ is
// not in the app: on one sound alone a plain EQ cannot be told from the sound's own colour, see Research/tone).
// Tools/tone_parity.py compares the two. 48 kHz; plain C++17, no JUCE.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "Room.h"

namespace dr
{
namespace tone
{
constexpr int N = 8192, HOP = 2048, NB = N / 2 + 1;     // frame, step, bins
constexpr double binHz = SR / N;
constexpr double maxBoostDb = 18.0, maxCutDb = 30.0;    // caps on what a curve may do, bent in softly from 3/4 of the cap
constexpr int irTaps = 16384;                           // the filter: minimum phase, this many taps
constexpr int maxFrames = 3000;                         // a long sound: every k-th of the frames that count is kept, so that no more than these are
using Stop = room::Stop;

struct Echo { double delay = 0, gain = 0; };            // a reflection: samples after the sound, level against the sound (negative: flipped)

// ---------------------------------------------------------------- frames
// numpy.percentile (linear) of v, which is reordered.
inline double percentileOf (std::vector<float>& v, double q)
{
    const size_t n = v.size(); if (n == 0) return 0.0;
    const double pos = q / 100.0 * (double) (n - 1); const size_t i = (size_t) std::floor (pos);
    std::nth_element (v.begin(), v.begin() + (std::ptrdiff_t) i, v.end()); const double a = v[i];
    if (i + 1 >= n || pos == (double) i) return a;
    const double b = *std::min_element (v.begin() + (std::ptrdiff_t) i + 1, v.end());
    return a + (b - a) * (pos - (double) i);
}

// The power spectra of a sound's frames (Hann, half a window of silence in front and a window behind). Only the
// frames that count are looked at: total power within 45 dB of the loud frames (95th percentile), so silence
// and faded tails are left out.
struct Frames
{
    int T = 0;                          // frames kept
    std::vector<float> P;               // their power per bin, left and right together: [frame][NB] (only when asked for)
    Vec meanL, meanR;                   // mean power per bin over the frames kept
    const float* row (int t) const { return P.data() + (size_t) t * (size_t) NB; }
};
inline Frames frames (const Stereo& x, bool keepRows, const Stop& stop = {})
{
    Frames F; const long n = (long) x.size(), T = 1 + ((n + N / 2 + N) - N) / HOP;
    static const Vec win = hannPeriodic ((size_t) N);
    Vec energy ((size_t) T, 0.0); F.meanL.assign ((size_t) NB, 0.0); F.meanR.assign ((size_t) NB, 0.0);
    // every frame's total power, without its spectrum: the sum over the bins 0 .. N/2 of a real frame is
    // (N * sum of squares + (plain sum)^2 + (alternating sum)^2) / 2
    for (long t = 0; t < T; ++t)
    {
        const long a = t * HOP - N / 2; double s2 = 0, sl = 0, al = 0, sr = 0, ar = 0;
        for (long i = std::max (0L, -a); i < N && a + i < n; ++i)
        {
            const double w = win[(size_t) i], l = x.l[(size_t) (a + i)] * w, r = x.r[(size_t) (a + i)] * w;
            s2 += l * l + r * r; sl += l; sr += r; if (i & 1) { al -= l; ar -= r; } else { al += l; ar += r; }
        }
        energy[(size_t) t] = 0.5 * ((double) N * s2 + sl * sl + al * al + sr * sr + ar * ar);
    }
    std::vector<long> act; { const double thr = percentile (energy, 95.0) * std::pow (10.0, -4.5); for (long t = 0; t < T; ++t) if (energy[(size_t) t] > thr) act.push_back (t); }
    if (act.empty()) return F;
    const size_t every = (act.size() + (size_t) maxFrames - 1) / (size_t) maxFrames; std::vector<long> keep;
    for (size_t i = 0; i < act.size(); i += every) keep.push_back (act[i]);
    const int K = (int) keep.size(); if (keepRows) F.P.assign ((size_t) K * (size_t) NB, 0.0f);
    const room::Transform tr (N); Vec re ((size_t) N), im ((size_t) N);
    for (int q = 0; q < K; ++q)
    {
        if (stop && (q & 15) == 0 && stop()) { F.P.clear(); return F; }       // given up: T stays 0
        const long a = keep[(size_t) q] * HOP - N / 2;
        for (long i = 0; i < N; ++i) { const long j = a + i; const bool in = j >= 0 && j < n; const double w = win[(size_t) i]; re[(size_t) i] = in ? x.l[(size_t) j] * w : 0.0; im[(size_t) i] = in ? x.r[(size_t) j] * w : 0.0; }
        tr.run (re, im);        // left as the real part and right as the imaginary part: one transform for both
        float* row = keepRows ? F.P.data() + (size_t) q * (size_t) NB : nullptr;
        for (int k = 0; k < NB; ++k)
        {
            const size_t u = (size_t) k, m = (size_t) ((N - k) & (N - 1));
            const double lr = 0.5 * (re[u] + re[m]), li = 0.5 * (im[u] - im[m]), rr = 0.5 * (im[u] + im[m]), ri = 0.5 * (re[m] - re[u]);
            const double pl = lr * lr + li * li, pr = rr * rr + ri * ri;
            F.meanL[u] += pl; F.meanR[u] += pr; if (row != nullptr) row[k] = (float) (pl + pr);
        }
    }
    for (auto& v : F.meanL) v /= (double) K; for (auto& v : F.meanR) v /= (double) K;
    F.T = K; return F;
}

// A sound's long-term spectra (dB per bin): both channels together, left, right.
struct Spectra { Vec both, left, right; bool silent = true; };
inline Spectra spectraOf (const Frames& F)
{
    Spectra S; S.silent = F.T < 1; if (S.silent) return S;
    S.both.resize ((size_t) NB); S.left.resize ((size_t) NB); S.right.resize ((size_t) NB);
    for (size_t k = 0; k < (size_t) NB; ++k)
    {
        S.left[k] = 10.0 * std::log10 (std::max (F.meanL[k], 1e-30)); S.right[k] = 10.0 * std::log10 (std::max (F.meanR[k], 1e-30));
        S.both[k] = 10.0 * std::log10 (std::max (F.meanL[k] + F.meanR[k], 1e-30));
    }
    return S;
}
inline Spectra spectra (const Stereo& x, const Stop& stop = {}) { return spectraOf (frames (x, false, stop)); }

// How far one sound is a filtered copy of the other, 0 .. 1: the coherence of their spectra over the frames (Hann
// frames side by side, left with left and right with right, weighted by power over the bins). On the scoreboard:
// two versions of one recording with a tone effect between them 0.94 .. 1.00 (128 pairs), with a reverb between
// them 0.2 .. 1.0 (median 0.85), two different sounds at most 0.12 (282 pairs, also where both are the same kind
// of sound with the same rise and fall of level). Sounds under 0.7 s: 0.
// Two versions of the same recording (over sameCoherence): Tone Match can take every detail of the difference,
// which is then the effect's own curve, and left and right each get their own curve.
// offA, offB: where to start in each sound (see pairCoherence).
inline double coherence (const Stereo& a, const Stereo& b, const Stop& stop = {}, long offA = 0, long offB = 0)
{
    offA = std::clamp (offA, 0L, (long) a.size()); offB = std::clamp (offB, 0L, (long) b.size());
    const long n = std::min ((long) a.size() - offA, (long) b.size() - offB), T = n >= N ? (n - N) / N + 1 : 0; if (T < 4) return 0.0;
    const long every = (T + 399) / 400; static const Vec win = hannPeriodic ((size_t) N);
    const room::Transform tr (N); Vec re ((size_t) N), im ((size_t) N), sr ((size_t) NB), si ((size_t) NB), saa ((size_t) NB), sbb ((size_t) NB); double tot = 0; int chans = 0;
    for (int c = 0; c < 2; ++c)
    {
        const Vec& x = c == 0 ? a.l : a.r; const Vec& y = c == 0 ? b.l : b.r;
        std::fill (sr.begin(), sr.end(), 0.0); std::fill (si.begin(), si.end(), 0.0); std::fill (saa.begin(), saa.end(), 0.0); std::fill (sbb.begin(), sbb.end(), 0.0);
        for (long t = 0; t < T; t += every)
        {
            if (stop && stop()) return 0.0;
            const size_t p = (size_t) (t * N), pa = p + (size_t) offA, pb = p + (size_t) offB;
            for (size_t i = 0; i < (size_t) N; ++i) { re[i] = x[pa + i] * win[i]; im[i] = y[pb + i] * win[i]; }
            tr.run (re, im);        // the one sound as the real part and the other as the imaginary part
            for (int k = 0; k < NB; ++k)
            {
                const size_t u = (size_t) k, m = (size_t) ((N - k) & (N - 1));
                const double ar = 0.5 * (re[u] + re[m]), ai = 0.5 * (im[u] - im[m]), br = 0.5 * (im[u] + im[m]), bi = 0.5 * (re[m] - re[u]);
                sr[u] += ar * br + ai * bi; si[u] += ai * br - ar * bi; saa[u] += ar * ar + ai * ai; sbb[u] += br * br + bi * bi;
            }
        }
        double num = 0, den = 0, ea = 0, eb = 0;
        for (size_t k = 0; k < (size_t) NB; ++k) { const double d = std::sqrt (saa[k] * sbb[k]); if (d > 0) num += (sr[k] * sr[k] + si[k] * si[k]) / d; den += d; ea += saa[k]; eb += sbb[k]; }
        if (den > 0 && ea > 1e-20 * eb && eb > 1e-20 * ea) { tot += num / den; ++chans; }      // a silent channel on one side does not count (what is left of it is rounding)
    }
    return chans > 0 ? tot / chans : 0.0;
}
constexpr double sameCoherence = 0.5;

// The time (samples) by which sound b is late against sound a (negative: early), up to maxLag either way: the
// highest peak of their cross-correlation with the spectrum evened out, so that a filter between the two does not
// smear the peak. Looks at the loudest 5.5 s of a. Two files of one recording often do not start together (an
// effect's own delay, a different cut); without this shift their coherence falls to nothing from a few
// milliseconds on. For two unrelated sounds the answer means nothing and does no harm: their coherence is low at
// any shift (scoreboard: at most 0.18 with the shift, 0.12 without).
inline long lagOf (const Stereo& a, const Stereo& b, long maxLag = 48000)
{
    const long n = (long) std::min (a.size(), b.size()); if (n < N) return 0;
    constexpr long L = 1 << 18; constexpr int M = 1 << 19; long s0 = 0; const long len = std::min (n, L);
    if (n > L)      // the loudest stretch of a, in steps of a tenth of a second
    {
        const long blk = 4800, nb = (n + blk - 1) / blk, k = L / blk; Vec cs ((size_t) nb + 1, 0.0);
        for (long i = 0; i < nb; ++i) { double e = 0; for (long j = i * blk; j < std::min (n, (i + 1) * blk); ++j) { const double v = a.l[(size_t) j] + a.r[(size_t) j]; e += v * v; } cs[(size_t) i + 1] = cs[(size_t) i] + e; }
        double best = -1.0; for (long i = 0; i + k <= nb; ++i) { const double e = cs[(size_t) (i + k)] - cs[(size_t) i]; if (e > best) { best = e; s0 = i * blk; } }
        s0 = std::min (s0, n - L);
    }
    Vec re ((size_t) M, 0.0), im ((size_t) M, 0.0);
    for (long i = 0; i < len; ++i) { const size_t j = (size_t) (s0 + i); re[(size_t) i] = a.l[j] + a.r[j]; im[(size_t) i] = b.l[j] + b.r[j]; }
    const room::Transform tr (M); tr.run (re, im);      // a as the real part, b as the imaginary part
    Vec cr ((size_t) M), ci ((size_t) M); double top = 0;
    for (int k = 0; k < M; ++k)
    {
        const size_t u = (size_t) k, m = (size_t) ((M - k) & (M - 1));
        const double ar = 0.5 * (re[u] + re[m]), ai = 0.5 * (im[u] - im[m]), br = 0.5 * (im[u] + im[m]), bi = 0.5 * (re[m] - re[u]);
        cr[u] = ar * br + ai * bi; ci[u] = ar * bi - ai * br; top = std::max (top, std::hypot (cr[u], ci[u]));      // conj (A) B
    }
    if (top <= 0) return 0;
    for (size_t k = 0; k < (size_t) M; ++k) { const double w = 1.0 / std::max (std::hypot (cr[k], ci[k]), 1e-3 * top); re[k] = cr[k] * w; im[k] = -ci[k] * w; }
    tr.run (re, im);                                    // the inverse transform of a spectrum of a real signal: its real part
    const long ml = std::min<long> (maxLag, M / 2 - 1); long at = 0; double pk = -1.0;
    for (long t = 0; t <= ml; ++t) if (std::abs (re[(size_t) t]) > pk) { pk = std::abs (re[(size_t) t]); at = t; }
    for (long t = M - ml; t < M; ++t) if (std::abs (re[(size_t) t]) > pk) { pk = std::abs (re[(size_t) t]); at = t - M; }
    return at;
}
// coherence() of two sounds that may not start together: b is moved by lagOf first. lag: if given, receives that shift.
inline double pairCoherence (const Stereo& a, const Stereo& b, long* lag = nullptr)
{
    const long d = lagOf (a, b); if (lag != nullptr) *lag = d;
    return d >= 0 ? coherence (a, b, {}, 0, d) : coherence (a, b, {}, -d, 0);
}

// ---------------------------------------------------------------- curves
// Gains beyond the caps are bent in softly: untouched up to 3/4 of the cap, then levelling out at the cap.
inline double softCap (double g)
{
    auto one = [] (double v, double cap) { const double k = 0.75 * cap; return v <= k ? v : k + (cap - k) * std::tanh ((v - k) / std::max (cap - k, 1e-9)); };
    return g >= 0 ? one (g, maxBoostDb) : -one (-g, maxCutDb);
}
inline Vec softCap (Vec g) { for (auto& v : g) v = softCap (v); return g; }

// The curve of one reflection: 20 log10 |1 + gain e^(-j 2 pi f delay)|.
inline Vec combDb (double delay, double gain)
{
    Vec c ((size_t) NB);
    for (size_t k = 0; k < (size_t) NB; ++k) c[k] = 10.0 * std::log10 (std::max (1.0 + gain * gain + 2.0 * gain * std::cos (2.0 * PI * ((double) k * binHz) * delay / SR), 1e-12));
    return c;
}
// dB to add to a curve so that taking it out (caps included) leaves the long-term power of a sound with the
// long-term spectrum `spec` as it is.
inline double levelOffset (const Vec& curve, const Vec& spec)
{
    const double mx = *std::max_element (spec.begin(), spec.end()); double num = 0, den = 0;
    for (size_t k = 0; k < (size_t) NB; ++k) { const double p = std::pow (10.0, (spec[k] - mx) / 10.0); num += p * std::pow (10.0, softCap (-curve[k]) / 10.0); den += p; }
    return 10.0 * std::log10 (std::max (num / std::max (den, 1e-300), 1e-12));
}
// The found reflections as one curve (delays stretched by delayScale), level set by levelOffset. No reflections: all zero.
inline Vec combCurve (const std::vector<Echo>& echoes, const Vec& spec, double delayScale = 1.0)
{
    Vec c ((size_t) NB, 0.0); if (echoes.empty() || spec.size() != (size_t) NB) return c;
    for (auto& e : echoes) { const Vec d = combDb (e.delay * delayScale, e.gain); for (size_t k = 0; k < (size_t) NB; ++k) c[k] += d[k]; }
    const double off = levelOffset (c, spec); for (auto& v : c) v += off;
    return c;
}

// The Detail control, 0 .. 1: the curve smoothed along log frequency by a Gaussian that is 1 octave wide at 0, 1/8 at
// 0.5, 1/12 at 0.6, 1/23 at 0.75; 1 leaves the curve as it is. Three passes of a running mean whose half width
// grows with the bin number.
inline double detailOctaves (double detail) { return detail >= 1.0 ? 0.0 : std::pow (2.0, -6.0 * std::max (detail, 0.0)); }
inline Vec smooth (const Vec& curve, double detail)
{
    const double s = detailOctaves (detail); if (s == 0.0 || curve.empty()) return curve;
    const int n = (int) curve.size(); Vec v = curve, cs ((size_t) n + 1), o ((size_t) n);
    auto at = [&] (double q) { const double u = q + 0.5; const int i = std::clamp ((int) std::floor (u), 0, n - 1); return cs[(size_t) i] + (cs[(size_t) i + 1] - cs[(size_t) i]) * (u - (double) i); };
    for (int pass = 0; pass < 3; ++pass)
    {
        cs[0] = 0.0; for (int i = 0; i < n; ++i) cs[(size_t) i + 1] = cs[(size_t) i] + v[(size_t) i];
        for (int k = 0; k < n; ++k)
        {
            const double sig = s * std::log (2.0) * (double) k, h = 0.5 * (std::sqrt (4.0 * sig * sig + 1.0) - 1.0);
            const double a = std::clamp ((double) k - h - 0.5, -0.5, (double) n - 0.5), b = std::clamp ((double) k + h + 0.5, -0.5, (double) n - 0.5);
            o[(size_t) k] = (at (b) - at (a)) / std::max (b - a, 1e-9);
        }
        v = o;
    }
    return v;
}

// Tone Match: the curve that gives a sound with the long-term spectrum Sb that of Sa. The difference of the two,
// taken only where both sounds have something (within 60 dB of their loudest region, from 20 Hz; outside, the
// nearest value is held), smoothed by Detail, cut at +-maxDb, and its level set so that the sound keeps its
// long-term power when the curve is put on in full (caps included).
inline Vec matchCurve (const Vec& Sa, const Vec& Sb, double detail, double maxDb = 30.0)
{
    Vec c ((size_t) NB, 0.0); if (Sa.size() != (size_t) NB || Sb.size() != (size_t) NB) return c;
    if (*std::max_element (Sa.begin(), Sa.end()) < -250.0 || *std::max_element (Sb.begin(), Sb.end()) < -250.0) return c;      // a silent channel: nothing to match
    auto box9 = [] (const Vec& S) { Vec o ((size_t) NB); for (int k = 0; k < NB; ++k) { double s = 0; for (int i = std::max (0, k - 4); i <= std::min (NB - 1, k + 4); ++i) s += S[(size_t) i]; o[(size_t) k] = s / 9.0; } return o; };
    const Vec va = box9 (Sa), vb = box9 (Sb); const double ma = *std::max_element (va.begin(), va.end()), mb = *std::max_element (vb.begin(), vb.end());
    std::vector<int> ok; for (int k = 0; k < NB; ++k) if (va[(size_t) k] > ma - 60.0 && vb[(size_t) k] > mb - 60.0 && (double) k * binHz >= 20.0) ok.push_back (k);
    if (ok.empty()) return c;
    auto d = [&] (int k) { return Sa[(size_t) k] - Sb[(size_t) k]; };
    size_t j = 0;
    for (int k = 0; k < NB; ++k)
    {
        if (k <= ok.front()) c[(size_t) k] = d (ok.front());
        else if (k >= ok.back()) c[(size_t) k] = d (ok.back());
        else { while (ok[j + 1] < k) ++j; const int a = ok[j], b = ok[j + 1]; c[(size_t) k] = d (a) + (d (b) - d (a)) / (double) (b - a) * (double) (k - a); }
    }
    c = smooth (c, detail);
    Vec vals; vals.reserve (ok.size()); for (int k : ok) vals.push_back (c[(size_t) k]);
    const double med = median (vals); for (auto& v : c) v = std::clamp (v - med, -maxDb, maxDb);
    const double mx = *std::max_element (Sb.begin(), Sb.end()); Vec p ((size_t) NB); double den = 0;
    for (size_t k = 0; k < (size_t) NB; ++k) { p[k] = std::pow (10.0, (Sb[k] - mx) / 10.0); den += p[k]; }
    for (int it = 0; it < 3; ++it)
    {
        double num = 0; for (size_t k = 0; k < (size_t) NB; ++k) num += p[k] * std::pow (10.0, softCap (c[k]) / 10.0);
        const double off = 10.0 * std::log10 (std::max (num / std::max (den, 1e-300), 1e-12)); for (auto& v : c) v -= off;
    }
    return c;
}

// ---------------------------------------------------------------- the echo finder
// A short reflection (a comb filter) adds a ripple to the log spectrum that is evenly spaced in Hz: one peak in the
// cepstrum of the long-term log spectrum, at the reflection's delay and as high as its gain. A peak between 0.5
// and 15 ms is taken for a reflection when
//   1. its gain is at least echoMin (echoMinShort for delays under 1 ms, where a sound's own formants ripple too);
//   2. its second peak, at twice the delay, is negative: at most -0.1 g^2 (a reflection has -0.5 g^2, a steady pitch a positive one);
//   3. every sixth of the sound (in time, each with its own long-term spectrum) shows it with the same sign and at
//      least 0.4 of its size (sixths with less than half of the whole sound's frequency range are left out; three must remain);
//   4. it stands alone: with delay and gain refined and its comb taken out, nothing of half its size is left in the
//      cepstrum within 10 samples of its delay (a cluster of peaks is a ripple of the sound itself).
// The comb is then taken out of the spectra and one more reflection is looked for. On the scoreboard this finds all
// 8 comb cases (delay within 0.06 samples on 7 of them, gain within 0.035) and nothing on the 184 other sounds; on
// 1057 clean clips it reports 2 reflections that are not there (Research/tone/REPORT.md).
constexpr double echoMin = 0.25, echoMinShort = 0.40;
constexpr int QM = 1441, QLO = 24, QHI = 720;       // delays up to 30 ms are worked out, 0.5 .. 15 ms are searched

// ln-magnitude long-term spectrum of the frames a .. b-1 (mean over the frames of the log power, each bin held at
// 40 dB under its own loud frames) and the loud frames' level (dB). A fixed filter adds its own ln-magnitude to the
// first, whatever the sound does.
struct LongTerm { Vec lm, lev; };
inline LongTerm longTerm (const Frames& F, int a, int b)
{
    LongTerm o; o.lm.assign ((size_t) NB, 0.0); o.lev.assign ((size_t) NB, 0.0); const int T = b - a; if (T < 1) return o;
    std::vector<float> col ((size_t) T);
    for (int k = 0; k < NB; ++k)
    {
        for (int t = 0; t < T; ++t) col[(size_t) t] = F.row (a + t)[k];
        const double top = percentileOf (col, 95.0), fl = std::max (top * 1e-4, 1e-30); double s = 0;
        for (float v : col) s += std::log (std::max ((double) v, fl));
        o.lm[(size_t) k] = 0.5 * s / (double) T; o.lev[(size_t) k] = 10.0 * std::log10 (std::max (top, 1e-30));
    }
    return o;
}
// Bins that count for the cepstrum: 100 Hz .. 16 kHz with a Hann taper, and only where the sound has something. The
// level is smoothed over 1.5 kHz first: otherwise a comb's own notches print themselves into the weights and
// split its peak into side peaks.
inline Vec echoWeight (const Vec& lev)
{
    constexpr int lo = 18, hi = 2730, M = hi - lo + 1;          // 100 Hz .. 16 kHz
    double mb = -1e300; for (int k = lo; k <= hi; ++k) mb = std::max (mb, lev[(size_t) k]);
    Vec v ((size_t) NB); for (size_t k = 0; k < (size_t) NB; ++k) v[k] = std::max (lev[k], mb - 80.0);
    const Vec sm = uniformFilter (v, 257, 1); double ms = -1e300; for (int k = lo; k <= hi; ++k) ms = std::max (ms, sm[(size_t) k]);
    Vec w ((size_t) NB, 0.0);
    for (int k = lo; k <= hi; ++k) w[(size_t) k] = (0.5 - 0.5 * std::cos (2.0 * PI * (double) (k - lo) / (double) (M - 1))) * std::clamp ((sm[(size_t) k] - (ms - 60.0)) / 10.0, 0.0, 1.0);
    return w;
}
struct Weighted { Vec x; double sw = 1e-9; };      // (the spectrum minus its weighted mean) times the weights, and the sum of the weights
inline Weighted weighted (const Vec& lm, const Vec& w)
{
    Weighted o; o.x.resize ((size_t) NB); double sw = 0, s = 0; for (size_t k = 0; k < (size_t) NB; ++k) { sw += w[k]; s += lm[k] * w[k]; }
    o.sw = std::max (sw, 1e-9); const double m = s / o.sw; for (size_t k = 0; k < (size_t) NB; ++k) o.x[k] = (lm[k] - m) * w[k];
    return o;
}
// Cepstrum, scaled so that a reflection of gain g at a delay of d samples gives g at index d.
inline Vec cepstrum (const Weighted& W, const room::Transform& tr)
{
    Vec re ((size_t) N, 0.0), im ((size_t) N, 0.0);
    for (int k = 0; k < NB; ++k) { re[(size_t) k] = W.x[(size_t) k]; if (k > 0 && k < N / 2) re[(size_t) (N - k)] = W.x[(size_t) k]; }
    tr.back (re, im); re.resize ((size_t) QM); for (auto& v : re) v *= (double) N / W.sw;
    return re;
}
// Gain of the ripple cos(2 pi f d) in the weighted spectrum, for a (fractional) delay of d samples.
inline double project (const Weighted& W, double d)
{
    const double th = 2.0 * PI * d / (double) N; double s = 0; for (int k = 0; k < NB; ++k) s += std::cos (th * (double) k) * W.x[(size_t) k];
    return 2.0 / W.sw * s;
}
// Delay and gain by a least-squares fit of the whole comb curve ln|1 + g e^(-j theta)| = sum over k of
// (-1)^(k+1) g^k / k cos(k theta), first four terms, over delays d +- 1 and gains 0.7 g .. 1.3 g.
inline Echo refine (const Weighted& W, double d, double g)
{
    constexpr int K = 4, nd = 101; double P[K][nd];
    for (int k = 1; k <= K; ++k) for (int j = 0; j < nd; ++j) P[k - 1][j] = project (W, (double) k * (d + (-1.0 + 0.02 * (double) j)));
    Echo best { d, g }; double bestSc = 0; bool have = false;
    for (int i = 0; i <= 12; ++i)
    {
        const double gg = g * (0.7 + 0.05 * (double) i); double co[K], c2 = 0;
        for (int k = 1; k <= K; ++k) { co[k - 1] = ((k & 1) ? 1.0 : -1.0) * std::pow (gg, (double) k) / (double) k; c2 += co[k - 1] * co[k - 1]; }
        int bj = 0; double bs = 1e300;
        for (int j = 0; j < nd; ++j) { double s = 0; for (int k = 0; k < K; ++k) s += co[k] * P[k][j]; s = -2.0 * s + c2; if (s < bs) { bs = s; bj = j; } }
        if (! have || bs < bestSc) { have = true; bestSc = bs; best = { d + (-1.0 + 0.02 * (double) bj), gg }; }
    }
    return best;
}
inline std::vector<Echo> findEchoes (const Frames& F, const Stop& stop = {}, double sens = 1.0, int maxEchoes = 2, int ncand = 40)
{
    std::vector<Echo> out; const int T = F.T, nseg = std::min (6, T / 8); if (nseg < 3 || F.P.size() != (size_t) T * (size_t) NB) return out;
    std::vector<Vec> LM, WW;
    for (int i = -1; i < nseg; ++i)
    {
        if (stop && stop()) return {};
        const LongTerm lt = i < 0 ? longTerm (F, 0, T) : longTerm (F, (int) ((long) i * T / nseg), (int) ((long) (i + 1) * T / nseg));
        LM.push_back (lt.lm); WW.push_back (echoWeight (lt.lev));
    }
    auto sum = [] (const Vec& w) { double s = 0; for (double v : w) s += v; return s; };
    const double s0 = sum (WW[0]); if (s0 < 50.0) return out;
    std::vector<size_t> use; for (int i = 1; i <= nseg; ++i) if (sum (WW[(size_t) i]) >= 0.5 * s0) use.push_back ((size_t) i);
    if (use.size() < 3) return out;
    auto thr = [sens] (double d) { return (d < 48.0 ? echoMinShort : echoMin) / sens; };
    auto sign = [] (double v) { return (v > 0) - (v < 0); };
    const room::Transform tr (N); const double ln = std::log (10.0) / 20.0;
    for (int round = 0; round < maxEchoes; ++round)
    {
        if (stop && stop()) return {};
        const Weighted W0 = weighted (LM[0], WW[0]); std::vector<Weighted> WS; for (size_t i : use) WS.push_back (weighted (LM[i], WW[i]));
        Vec a = cepstrum (W0, tr); for (int q = 0; q < QM; ++q) a[(size_t) q] = (q < QLO || q >= QHI) ? 0.0 : std::abs (a[(size_t) q]) / thr ((double) q);
        struct Cand { double strength, d, g; }; std::vector<Cand> cands;
        for (int c = 0; c < ncand; ++c)
        {
            const int k = (int) (std::max_element (a.begin(), a.end()) - a.begin()); if (a[(size_t) k] < 0.8) break;
            for (int q = std::max (0, k - 3); q < std::min (QM, k + 4); ++q) a[(size_t) q] = 0.0;
            double d = 0, g = 0, top = -1;
            for (int i = 0; i <= 60; ++i) { const double dd = (double) k + (-1.5 + 0.05 * (double) i), p = project (W0, dd); if (std::abs (p) > top) { top = std::abs (p); d = dd; g = p; } }
            if (std::abs (g) < thr (d)) continue;                                                   // 1
            if (project (W0, 2.0 * d) > -0.1 * g * g) continue;                                     // 2
            Vec gs { g }; bool all = true;
            for (auto& w : WS) { const double v = project (w, d); gs.push_back (v); all = all && sign (v) == sign (g) && std::abs (v) >= 0.4 * std::abs (g); }
            if (! all) continue;                                                                    // 3
            cands.push_back ({ std::abs (g) / thr (d), d, median (gs) });
        }
        std::sort (cands.begin(), cands.end(), [] (const Cand& p, const Cand& q) { return p.strength != q.strength ? p.strength > q.strength : p.d != q.d ? p.d > q.d : p.g > q.g; });
        bool got = false; Echo e; Vec cl;
        for (auto& c : cands)
        {
            e = refine (W0, c.d, c.g); e.gain = std::clamp (e.gain, -0.95, 0.95); cl = combDb (e.delay, e.gain); for (auto& v : cl) v *= ln;
            Vec lm = LM[0]; for (size_t k = 0; k < (size_t) NB; ++k) lm[k] -= cl[k];
            const Vec cp = cepstrum (weighted (lm, WW[0]), tr); const int k = (int) std::nearbyint (e.delay); double mx = 0;
            for (int q = std::max (QLO, k - 10); q < std::min (QM, k + 11); ++q) mx = std::max (mx, std::abs (cp[(size_t) q]));
            if (mx <= 0.5 * std::abs (e.gain)) { got = true; break; }                               // 4
        }
        if (! got) break;
        out.push_back (e); for (auto& L : LM) for (size_t k = 0; k < (size_t) NB; ++k) L[k] -= cl[k];
    }
    return out;
}

// What the tone engine keeps of a sound: its long-term spectra and the reflections found on it.
struct Found { Spectra spec; std::vector<Echo> echoes; bool stopped = false; };
inline Found analyse (const Stereo& x, const Stop& stop = {})
{
    Found f; const Frames F = frames (x, true, stop);
    if (stop && stop()) { f.stopped = true; return f; }
    f.spec = spectraOf (F); if (F.T >= 2) f.echoes = findEchoes (F, stop);
    if (stop && stop()) { f.stopped = true; f.echoes.clear(); }
    return f;
}

// ---------------------------------------------------------------- curve -> filter
// The curve (dB per bin, taken as given: no caps here) as a minimum-phase impulse response of n taps: straight
// lines on a log-frequency axis between the bins, then the cepstrum of the log magnitude folded. Minimum phase
// because the effects are (nearly) minimum phase themselves, so the inverse curve also undoes their phase; it
// starts at tap 0 and has no latency.
inline Vec toIr (const Vec& curveDb, int n = irTaps)
{
    const int nfft = (int) nextPow2 ((size_t) 4 * (size_t) n); const room::Transform tr (nfft); Vec re ((size_t) nfft, 0.0), im ((size_t) nfft, 0.0);
    const double l0 = std::log (1e-3), ln = std::log (10.0) / 20.0;
    auto lf = [l0] (int k) { return k == 0 ? l0 : std::log ((double) k * binHz); };
    for (int j = 0; j <= nfft / 2; ++j)
    {
        double v = curveDb[0];
        if (j > 0)
        {
            const int k = (int) ((int64_t) j * N / nfft);
            if (k >= NB - 1) v = curveDb[(size_t) NB - 1];
            else { const double a = lf (k), b = lf (k + 1), t = (std::log ((double) j * SR / (double) nfft) - a) / (b - a); v = curveDb[(size_t) k] + (curveDb[(size_t) k + 1] - curveDb[(size_t) k]) * t; }
        }
        re[(size_t) j] = v * ln; if (j > 0 && j < nfft / 2) re[(size_t) (nfft - j)] = re[(size_t) j];
    }
    tr.back (re, im);                                                       // the cepstrum (real, even)
    for (int j = 1; j < nfft / 2; ++j) re[(size_t) j] *= 2.0;
    for (int j = nfft / 2 + 1; j < nfft; ++j) re[(size_t) j] = 0.0;
    std::fill (im.begin(), im.end(), 0.0); tr.run (re, im);
    for (int j = 0; j <= nfft / 2; ++j)
    {
        const double m = std::exp (re[(size_t) j]), ph = im[(size_t) j]; const bool edge = j == 0 || j == nfft / 2;
        const double er = m * std::cos (ph), ei = edge ? 0.0 : m * std::sin (ph);
        re[(size_t) j] = er; im[(size_t) j] = ei; if (! edge) { re[(size_t) (nfft - j)] = er; im[(size_t) (nfft - j)] = -ei; }
    }
    tr.back (re, im);
    Vec h (re.begin(), re.begin() + n); const int kf = std::max (8, n / 8);
    for (int i = 0; i < kf && i < n; ++i) h[(size_t) (n - kf + i)] *= 0.5 * (1.0 + std::cos (PI * ((double) i + 0.5) / (double) kf));
    return h;
}

// x through the impulse responses hl (left) and hr (right); same length out. Empty when given up.
inline Stereo filter (const Stereo& x, const Vec& hl, const Vec& hr, const Stop& stop = {})
{
    const size_t n = x.size(), m = std::max (hl.size(), hr.size()); Stereo y; y.l.assign (n, 0.0); y.r.assign (n, 0.0); if (n == 0 || m == 0) return y;
    const size_t NF = nextPow2 (2 * m), L = NF - m + 1; const room::Transform tr ((int) NF); const bool same = hl == hr;
    auto spectrum = [&] (const Vec& h, Vec& re, Vec& im) { re.assign (NF, 0.0); im.assign (NF, 0.0); std::copy (h.begin(), h.end(), re.begin()); tr.run (re, im); };
    Vec Hlr, Hli, Hrr, Hri; spectrum (hl, Hlr, Hli); if (! same) spectrum (hr, Hrr, Hri);
    Vec re (NF), im (NF);
    auto times = [&] (const Vec& Hr, const Vec& Hi) { for (size_t k = 0; k < NF; ++k) { const double a = re[k], b = im[k]; re[k] = a * Hr[k] - b * Hi[k]; im[k] = a * Hi[k] + b * Hr[k]; } };
    for (size_t s = 0; s < n; s += L)
    {
        if (stop && stop()) return {};
        const size_t len = std::min (L, n - s), room = std::min (NF, n - s);
        if (same)       // left as the real part and right as the imaginary part: one pass for both
        {
            std::fill (re.begin(), re.end(), 0.0); std::fill (im.begin(), im.end(), 0.0);
            for (size_t i = 0; i < len; ++i) { re[i] = x.l[s + i]; im[i] = x.r[s + i]; }
            tr.run (re, im); times (Hlr, Hli); tr.back (re, im);
            for (size_t i = 0; i < room; ++i) { y.l[s + i] += re[i]; y.r[s + i] += im[i]; }
        }
        else
            for (int c = 0; c < 2; ++c)
            {
                const Vec& in = c == 0 ? x.l : x.r; Vec& o = c == 0 ? y.l : y.r;
                std::fill (re.begin(), re.end(), 0.0); std::fill (im.begin(), im.end(), 0.0);
                for (size_t i = 0; i < len; ++i) re[i] = in[s + i];
                tr.run (re, im); if (c == 0) times (Hlr, Hli); else times (Hrr, Hri); tr.back (re, im);
                for (size_t i = 0; i < room; ++i) o[s + i] += re[i];
            }
    }
    return y;
}

// A gain curve (dB per bin; one for the left and one for the right channel) on a sound: capped softly, then the
// minimum-phase filter. A curve of nothing but zeros gives the sound itself, exactly. Empty when given up.
inline bool nothing (const Vec& g) { for (double v : g) if (v != 0.0) return false; return true; }
inline Stereo gainFilter (const Stereo& x, const Vec& gainL, const Vec& gainR, const Stop& stop = {})
{
    if (gainL.size() != (size_t) NB || gainR.size() != (size_t) NB || (nothing (gainL) && nothing (gainR))) return x;
    const Vec hl = toIr (softCap (gainL));
    return gainL == gainR ? filter (x, hl, hl, stop) : filter (x, hl, toIr (softCap (gainR)), stop);
}

} // namespace tone
} // namespace dr
