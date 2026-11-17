#pragma once
// The verification step: before an effect is shown, its case is put to a second set of models that look at the
// whole stretch of sound (Research/detect has the work behind it and the Python side).
//
// Why. The detectors look at 3 s windows and the highest window score counted, so one odd window was enough and
// longer sounds collected more of them: on 438 clean sounds 13 % came out clean, with 1.6 effects per sound that
// were not there. No choice of thresholds repairs that (cuts that bring 86 % of the clean sounds out clean leave
// 28 % of the effects found, from 50 %).
//
// What. One small set of gradient-boosted trees per effect (the kind Models.h already reads; Resources/verify.bin,
// written by Research/detect/export_verify.py). Each reads
//   - what the two detectors computed, gathered over the windows: the mean, highest and lowest of the CNN outputs,
//     of the 15 hand numbers and of the deciders' scores; the scores as they were shown up to 0.4.1; the echo check;
//   - evidence(): 55 physical measurements on the stretch itself. Is one side a late copy of the other, does the
//     level or the balance between left and right move periodically and in all bands alike, are sample values held
//     or cut off, is there a steady line in the envelope at audio rate, do stretches repeat exactly, where are the
//     long-term spectrum's edges and its top end, is there a ceiling on the loud parts.
// The trees were trained on 5503 sounds from 313 sources (2150 of them clean: dry, reverb only, tone only, mixes,
// other speeds, lossy codecs, padded with silence) with the app's own effects and effects written apart from them.
//
// The score. Each effect's raw tree output is mapped so that the cut fixed on the training sounds (0.07 effects
// that are not there per clean sound in all, split over the effects so that most real ones are found) sits at 65 %,
// the Sureness slider's default; the slope is the one a logistic fit of right against wrong has on sounds the trees
// had not seen. So one rule serves every effect: shown when the score is above the slider. See README.md in
// Research/detect for what the slider gives at 50, 65 and 80 %.
//
// Long sounds. The trees judge stretches of the length they were trained on (3 to 12 s, most of them 6 to 9 s).
// A longer sound is cut into blocks of 6 s, each judged as a sound of its own from the windows that lie wholly
// inside it; an end piece shorter than 4.5 s stays with the block before it, so a block is 4.5 to 10.5 s long and a
// sound shorter than 10.5 s is one block. An effect's score is the highest score that at least half of the blocks
// reach (with two blocks: the higher one), so a single odd block no longer shows an effect on a long sound. Blocks
// without one window loud enough for the detectors are left out. A sound shorter than 3 s is one block with the
// one zero-padded window the detectors use.

#include "Inspector.h"
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace dr
{
namespace verif
{
using Cx = std::complex<double>;
using Numbers = std::vector<std::pair<std::string, double>>;

// numpy.fft.rfftfreq (n, 1 / fs)[k], computed the way numpy does it: some bins sit exactly on a band edge
// (6 kHz is bin 128 of 1024) and must fall on the same side here.
inline double binHz (size_t k, size_t n, double fs = SR) { const double d = 1.0 / fs; return (double) k * (1.0 / ((double) n * d)); }
// The bins for which `in` holds, as first and one past the last (they lie together).
template <typename In> inline std::pair<size_t, size_t> binRange (size_t bins, In in)
{
    size_t a = bins, b = 0; for (size_t k = 0; k < bins; ++k) if (in (k)) { a = std::min (a, k); b = k + 1; }
    return a < b ? std::make_pair (a, b) : std::make_pair ((size_t) 0, (size_t) 0);
}

// The transforms of two real sequences from one complex FFT: `re` holds the first, `im` the second.
struct Pair
{
    explicit Pair (size_t n) : re (n, 0.0), im (n, 0.0) {}
    Vec re, im;
    void run() { fft (re, im); }
    Cx a (size_t k) const { const size_t j = k == 0 ? 0 : re.size() - k; return { 0.5 * (re[k] + re[j]), 0.5 * (im[k] - im[j]) }; }
    Cx b (size_t k) const { const size_t j = k == 0 ? 0 : re.size() - k; return { 0.5 * (im[k] + im[j]), 0.5 * (re[j] - re[k]) }; }
};
// |FFT|^2 (bins 0 .. N/2) of `count` real frames of N values, two frames per transform. fill (j, buffer) writes
// frame j, use (j, power) takes its spectrum.
template <typename Fill, typename Use> inline void powerSpectra (size_t N, size_t count, Fill fill, Use use)
{
    Pair z (N); Vec pa (N / 2 + 1), pb (N / 2 + 1);
    for (size_t j = 0; j < count; j += 2)
    {
        const bool two = j + 1 < count;
        fill (j, z.re); if (two) fill (j + 1, z.im); else std::fill (z.im.begin(), z.im.end(), 0.0);
        z.run();
        for (size_t k = 0; k <= N / 2; ++k) { pa[k] = std::norm (z.a (k)); if (two) pb[k] = std::norm (z.b (k)); }
        use (j, pa); if (two) use (j + 1, pb);
    }
}
// scipy.signal.welch (x, SR, nperseg=N, noverlap=N/2): the mean periodogram, Hann window, each segment's mean
// taken off, power per Hz.
inline Vec welch (const Vec& x, size_t N)
{
    Vec P (N / 2 + 1, 0.0); if (x.size() < N) return P;
    const Vec w = hannPeriodic (N); double w2 = 0; for (double v : w) w2 += v * v;
    const size_t segs = (x.size() - N) / (N / 2) + 1;
    powerSpectra (N, segs,
                  [&] (size_t j, Vec& buf) { const size_t s = j * (N / 2); double m = 0; for (size_t i = 0; i < N; ++i) m += x[s + i]; m /= (double) N; for (size_t i = 0; i < N; ++i) buf[i] = (x[s + i] - m) * w[i]; },
                  [&] (size_t, const Vec& p) { for (size_t k = 0; k < P.size(); ++k) P[k] += p[k]; });
    const double scale = 1.0 / (SR * w2 * (double) segs);
    for (size_t k = 0; k < P.size(); ++k) P[k] *= (k == 0 || k == N / 2) ? scale : 2.0 * scale;
    return P;
}

// The strongest line of a track between lo and hi Hz (evidence.py _line): its rate, the share of the band's power
// that sits in it, how well a sine of that rate fits (r2) and the sine's amplitude. F is the track's spectrum
// (Hann window, padded eight times), hz the width of its bins.
struct Line
{
    double rate = 0, share = 0, r2 = 0, amp = 0, hz = 0; std::vector<Cx> F;
    double at (double f) const       // |F| at the bin nearest to f
    {
        if (F.empty()) return 0.0;
        size_t bk = 0; double bd = 1e300; for (size_t k = 0; k < F.size(); ++k) { const double d = std::abs ((double) k * hz - f); if (d < bd) { bd = d; bk = k; } }
        return std::abs (F[bk]);
    }
};
inline Line line (const Vec& track, double lo, double hi, double fs)
{
    Line o; const size_t n = track.size();
    if (n < 16 || stdev (track) < 1e-9) return o;
    const double m = mean (track); Vec v (n); for (size_t i = 0; i < n; ++i) v[i] = track[i] - m;
    const size_t N = nextPow2 (n * 8); const Vec w = hanningSym (n); Vec vw (n); for (size_t i = 0; i < n; ++i) vw[i] = v[i] * w[i];
    o.F = rfft (vw, N); o.hz = binHz (1, N, fs);
    const double flo = std::max (lo, 1.5 * fs / (double) n), half = 2.0 * fs / (double) n;
    double best = 0, all = 0; size_t bk = 0; bool any = false;
    for (size_t k = 0; k < o.F.size(); ++k)
    {
        const double f = (double) k * o.hz; if (! (f >= flo && f <= hi)) continue;
        const double p = std::norm (o.F[k]); any = true; all += p; if (p > best) { best = p; bk = k; }
    }
    if (! any) return o;
    o.rate = (double) bk * o.hz; double in = 0;
    for (size_t k = 0; k < o.F.size(); ++k) { const double f = (double) k * o.hz; if (f >= flo && f <= hi && std::abs (f - o.rate) <= half) in += std::norm (o.F[k]); }
    o.share = in / (all + 1e-30);
    const SineFit s = fitSine (v, o.rate, fs); o.r2 = std::max (0.0, s.r2); o.amp = std::hypot (s.a, s.b);
    return o;
}
// The complex amplitude of a track at rate r (Hann window), and how much several tracks move together at it:
// 1 = all in step and equally strong.
inline Cx ampAt (const Vec& track, const Vec& window, double r, double fs)
{
    Cx s = 0; for (size_t t = 0; t < track.size(); ++t) { const double ph = -2.0 * PI * r * ((double) t / fs); s += track[t] * window[t] * Cx (std::cos (ph), std::sin (ph)); }
    return s;
}
inline double together (const std::vector<Cx>& A) { Cx s = 0; double e = 0; for (auto& a : A) { s += a; e += std::norm (a); } return std::norm (s) / ((double) A.size() * e + 1e-30); }
inline double correlation (const Vec& a, const Vec& b)       // numpy.corrcoef (a, b)[0, 1]
{
    const double ma = mean (a), mb = mean (b); double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < a.size(); ++i) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) * (a[i] - ma); sbb += (b[i] - mb) * (b[i] - mb); }
    return std::clamp (sab / std::sqrt (saa * sbb), -1.0, 1.0);
}
} // namespace verif

// The physical measurements on a stretch of sound, by name (Research/detect/evidence.py is the same thing in
// Python; Tests/verify.cpp prints these so that the two can be compared number by number).
inline verif::Numbers evidence (const Stereo& y)
{
    using namespace verif;
    Numbers out; auto put = [&out] (const char* name, double v) { out.emplace_back (name, std::isnan (v) ? 0.0 : std::isinf (v) ? (v > 0 ? 1e6 : -1e6) : v); };
    const size_t n = y.size(); const Vec& L = y.l; const Vec& R = y.r; const Vec M = y.mid();

    // ---------------------------------------------------------------- stereo: is one side a late copy of the other, is it one source through a fixed path
    double eM = 0, eS = 0; for (size_t i = 0; i < n; ++i) { const double s = 0.5 * (L[i] - R[i]); eM += M[i] * M[i]; eS += s * s; }
    eM = eM / (double) std::max<size_t> (1, n) + 1e-20; eS = eS / (double) std::max<size_t> (1, n) + 1e-20;
    const double sideDb = std::clamp (10.0 * std::log10 (eS / eM), -80.0, 20.0), sl = stdev (L), sr = stdev (R);
    const bool wide = sl > 1e-7 && sr > 1e-7 && sideDb > -60.0;
    double corr0 = 1, xcPeak = 1, xcLagMs = 0, xcGain = 0, xcNeg = 0, msc = 1, mscHi = 1, balDb = 0;
    if (wide)
    {
        balDb = 20.0 * std::log10 (sl / sr);
        // the cross-correlation of left and right (the first 12 s, means taken off) for lags up to 40 ms either way,
        // summed block by block: xr[K + lag] = sum of a[i + lag] b[i]
        const size_t m = std::min (n, (size_t) (12 * SR)); const long K = (long) (0.04 * SR);
        double ma = 0, mb = 0; for (size_t i = 0; i < m; ++i) { ma += L[i]; mb += R[i]; } ma /= (double) m; mb /= (double) m;
        double ea = 0, eb = 0; for (size_t i = 0; i < m; ++i) { const double a = L[i] - ma, b = R[i] - mb; ea += a * a; eb += b * b; }
        const size_t N = 8192, B = 4096; Vec xr (N, 0.0), xi (N, 0.0); Pair z (N);
        for (size_t s = 0; s < m; s += B)
        {
            for (size_t i = 0; i < N; ++i)
            {
                const long ia = (long) s - K + (long) i;
                z.re[i] = ia >= 0 && ia < (long) m ? L[(size_t) ia] - ma : 0.0;
                z.im[i] = i < B && s + i < m ? R[s + i] - mb : 0.0;
            }
            z.run();
            for (size_t k = 0; k < N; ++k) { const Cx c = z.a (k) * std::conj (z.b (k)); xr[k] += c.real(); xi[k] += c.imag(); }
        }
        fft (xr, xi, true);
        const double den = std::sqrt (ea * eb) + 1e-30; auto v = [&] (long lag) { return xr[(size_t) (K + lag)] / den; };
        const long farFrom = (long) (0.002 * SR), nearTo = (long) (0.0005 * SR);       // a late copy is at least 2 ms away (closer is how microphones stand)
        double best = -1e300, low = 2.0, nearMax = 0; long bl = 0;
        auto look = [&] (long lag)
        {
            const bool far = std::abs (lag) >= farFrom; const double c = v (lag);
            if ((far ? c : -2.0) > best) { best = far ? c : -2.0; bl = lag; }
            if (far) low = std::min (low, c);
            if (std::abs (lag) < nearTo) nearMax = std::max (nearMax, std::abs (c));
        };
        for (long lag = 0; lag <= K; ++lag) look (lag);
        for (long lag = -K; lag < 0; ++lag) look (lag);
        corr0 = v (0); xcPeak = v (bl); xcLagMs = (double) std::abs (bl) / SR * 1e3; xcGain = v (bl) - nearMax; xcNeg = -low;
        // how far left and right are one signal, frequency by frequency (scipy.signal.csd and welch, 2048 / 1024)
        const size_t Nc = 2048;
        if (n >= Nc)
        {
            const Vec w = hannPeriodic (Nc); double w2 = 0; for (double a : w) w2 += a * a;
            Vec pxx (Nc / 2 + 1, 0.0), pyy (Nc / 2 + 1, 0.0), cr (Nc / 2 + 1, 0.0), ci (Nc / 2 + 1, 0.0); Pair q (Nc); size_t segs = 0;
            for (size_t s = 0; s + Nc <= n; s += Nc / 2, ++segs)
            {
                double ml = 0, mr = 0; for (size_t i = 0; i < Nc; ++i) { ml += L[s + i]; mr += R[s + i]; } ml /= (double) Nc; mr /= (double) Nc;
                for (size_t i = 0; i < Nc; ++i) { q.re[i] = (L[s + i] - ml) * w[i]; q.im[i] = (R[s + i] - mr) * w[i]; }
                q.run();
                for (size_t k = 0; k <= Nc / 2; ++k) { const Cx a = q.a (k), b = q.b (k), c = std::conj (a) * b; pxx[k] += std::norm (a); pyy[k] += std::norm (b); cr[k] += c.real(); ci[k] += c.imag(); }
            }
            const double scale = 2.0 / (SR * w2 * (double) segs); double num = 0, sum = 0, numHi = 0, sumHi = 0;
            for (size_t k = 0; k <= Nc / 2; ++k)
            {
                const double f = binHz (k, Nc); if (! (f > 80.0 && f < 16000.0)) continue;
                const double xx = pxx[k] * scale, yy = pyy[k] * scale, xy2 = (cr[k] * cr[k] + ci[k] * ci[k]) * scale * scale, c = xy2 / (xx * yy + 1e-30), wt = std::sqrt (xx * yy);
                num += c * wt; sum += wt; if (f > 1500.0) { numHi += c * wt; sumHi += wt; }
            }
            msc = num / (sum + 1e-30); mscHi = numHi / (sumHi + 1e-30);
        }
    }
    put ("st_side_db", sideDb); put ("st_corr0", corr0); put ("st_xc_peak", xcPeak); put ("st_xc_lag_ms", xcLagMs); put ("st_xc_gain", xcGain);
    put ("st_xc_neg", xcNeg); put ("st_msc", msc); put ("st_msc_hi", mscHi); put ("st_bal_db", balDb);

    // ---------------------------------------------------------------- band levels over time: 100 frames a second (scipy.signal.stft, Hann 1024, hop 480), 7 octave bands 100 Hz - 12.8 kHz
    const size_t NF = 1024, HOP = 480, KB = NF / 2 + 1, T = n >= NF ? (n - NF) / HOP + 1 : 0;
    Vec PM (KB * T), PL (KB * T), PR (KB * T);        // power of the mean of left and right, of left, of right: [frame * KB + bin]
    {
        const Vec w = hannPeriodic (NF); double ws = 0; for (double a : w) ws += a; const double sc = 1.0 / (ws * ws); Pair z (NF);
        for (size_t t = 0; t < T; ++t)
        {
            const size_t s = t * HOP; for (size_t i = 0; i < NF; ++i) { z.re[i] = L[s + i] * w[i]; z.im[i] = R[s + i] * w[i]; }
            z.run();
            for (size_t k = 0; k < KB; ++k) { const Cx a = z.a (k), b = z.b (k); PL[t * KB + k] = std::norm (a) * sc; PR[t * KB + k] = std::norm (b) * sc; PM[t * KB + k] = std::norm (0.5 * (a + b)) * sc; }
        }
    }
    auto hz = [NF] (size_t k) { return binHz (k, NF); };
    auto bandDb = [KB] (const Vec& P, size_t t, std::pair<size_t, size_t> b, double floor) { double s = 0; for (size_t k = b.first; k < b.second; ++k) s += P[t * KB + k]; return 10.0 * std::log10 (s + floor); };
    const int NB = 7; std::pair<size_t, size_t> oct[NB];
    for (int i = 0; i < NB; ++i) { const double lo = 100.0 * (double) (1 << i), hi = 100.0 * (double) (2 << i); oct[i] = binRange (KB, [&] (size_t k) { return hz (k) >= lo && hz (k) < hi; }); }
    const auto full = binRange (KB, [&] (size_t k) { return hz (k) >= 100.0 && hz (k) < 12800.0; });
    std::vector<Vec> EM (NB, Vec (T)), EL (NB, Vec (T)), ER (NB, Vec (T)); Vec tot (T), totL (T), totR (T);
    for (size_t t = 0; t < T; ++t)
    {
        for (int b = 0; b < NB; ++b) { EM[(size_t) b][t] = bandDb (PM, t, oct[b], 1e-14); EL[(size_t) b][t] = bandDb (PL, t, oct[b], 1e-14); ER[(size_t) b][t] = bandDb (PR, t, oct[b], 1e-14); }
        tot[t] = bandDb (PM, t, full, 1e-14); totL[t] = bandDb (PL, t, full, 1e-14); totR[t] = bandDb (PR, t, full, 1e-14);
    }
    const double top = T ? *std::max_element (tot.begin(), tot.end()) : 0.0;
    std::vector<char> act (T, 0); size_t nact = 0, t0 = 0, t1 = T; { bool first = true; for (size_t t = 0; t < T; ++t) if (tot[t] > top - 40.0) { act[t] = 1; ++nact; if (first) { t0 = t; first = false; } t1 = t + 1; } }
    double emMax = -1e300; Vec bandMax ((size_t) NB, -1e300); for (int b = 0; b < NB; ++b) for (size_t t = 0; t < T; ++t) { bandMax[(size_t) b] = std::max (bandMax[(size_t) b], EM[(size_t) b][t]); emMax = std::max (emMax, EM[(size_t) b][t]); }
    std::vector<int> okb; for (int b = 0; b < NB; ++b) if (T && bandMax[(size_t) b] > emMax - 40.0) okb.push_back (b);
    const bool enough = T >= 300 && okb.size() >= 2 && nact > 100;       // 3 s of frames, two live bands, a second of sound
    auto onActive = [&] (const Vec& v) { Vec o; o.reserve (nact); for (size_t t = 0; t < T; ++t) if (act[t]) o.push_back (v[t]); return o; };
    auto meanTrack = [T] (const std::vector<Vec>& rows) { Vec c (T, 0.0); for (auto& r : rows) for (size_t t = 0; t < T; ++t) c[t] += r[t]; for (auto& v : c) v /= (double) rows.size(); return c; };

    // a periodic level pattern, and whether all bands follow it in step (tremolo, gate)
    double lvRate = 0, lvShare = 0, lvR2 = 0, lvDepth = 0, lvPar = 0, lvAmpcv = 1, lvH2 = 0, lvH3 = 0, lvDip = 0, lvEdge = 0, lvHalves = 1, lvSd = 0;
    if (enough)
    {
        std::vector<Vec> D;
        for (int b : okb)
        {
            Vec E = EM[(size_t) b]; const double fl = bandMax[(size_t) b] - 60.0; for (auto& e : E) e = std::max (e, fl);
            const Vec smooth = uniformFilter (E, 100, 1); for (size_t t = 0; t < T; ++t) E[t] = (E[t] - smooth[t]) * (double) act[t];
            D.push_back (std::move (E));
        }
        const Vec c = meanTrack (D); const Line ln = line (c, 1.0, 16.0, 100.0);
        lvRate = ln.rate; lvShare = ln.share; lvR2 = ln.r2; lvDepth = 2.0 * ln.amp; lvSd = stdev (onActive (c));
        if (ln.rate > 0)
        {
            const Vec w = hanningSym (T); std::vector<Cx> A; Vec mag; for (auto& d : D) { A.push_back (ampAt (d, w, ln.rate, 100.0)); mag.push_back (std::abs (A.back())); }
            lvPar = together (A); lvAmpcv = stdev (mag) / (mean (mag) + 1e-30);
            const double a1 = ln.at (ln.rate) + 1e-30; lvH2 = ln.at (2.0 * ln.rate) / a1; lvH3 = ln.at (3.0 * ln.rate) / a1;
            const size_t h = T / 2; const double ra = line (Vec (c.begin(), c.begin() + (long) h), 1.0, 16.0, 100.0).rate, rb = line (Vec (c.begin() + (long) h, c.end()), 1.0, 16.0, 100.0).rate;
            lvHalves = std::abs (std::log2 ((ra + 1e-3) / (rb + 1e-3)));        // the same speed in both halves?
        }
        const Vec seg (tot.begin() + (long) t0, tot.begin() + (long) t1); const long S = (long) seg.size(); size_t dips = 0;
        for (long i = 0; i < S; ++i) { double mx = -1e300; for (long j = std::max (0L, i - 50); j <= std::min (S - 1, i + 49); ++j) mx = std::max (mx, seg[(size_t) j]); dips += seg[(size_t) i] < mx - 15.0; }
        lvDip = (double) dips / (double) std::max (1L, S);
        if (S > 2) { Vec d ((size_t) S - 1); for (long i = 0; i + 1 < S; ++i) d[(size_t) i] = std::abs (seg[(size_t) i + 1] - seg[(size_t) i]); lvEdge = percentile (d, 90.0); }
    }
    put ("lv_rate", lvRate); put ("lv_share", lvShare); put ("lv_r2", lvR2); put ("lv_depth", lvDepth); put ("lv_par", lvPar); put ("lv_ampcv", lvAmpcv);
    put ("lv_h2", lvH2); put ("lv_h3", lvH3); put ("lv_dip", lvDip); put ("lv_edge", lvEdge); put ("lv_halves", lvHalves); put ("lv_sd", lvSd);

    // left against right over time (auto pan)
    double pnRate = 0, pnShare = 0, pnR2 = 0, pnDepth = 0, pnPar = 0, pnSd = 0, pnEnvcorr = 1;
    if (wide && enough)
    {
        const double ref = emMax - 70.0; std::vector<Vec> I;
        for (int b : okb)
        {
            Vec v (T); double s = 0; for (size_t t = 0; t < T; ++t) { v[t] = (std::max (EL[(size_t) b][t], ref) - std::max (ER[(size_t) b][t], ref)) * (double) act[t]; s += v[t]; }
            const double mu = s / (double) nact; for (size_t t = 0; t < T; ++t) v[t] -= mu * (double) act[t];
            I.push_back (std::move (v));
        }
        const Vec d = meanTrack (I); const Line ln = line (medianFilter (d, 5), 0.1, 6.0, 100.0);
        pnRate = ln.rate; pnShare = ln.share; pnR2 = ln.r2; pnDepth = 2.0 * ln.amp; pnSd = stdev (onActive (d));
        if (ln.rate > 0) { const Vec w = hanningSym (T); std::vector<Cx> A; for (auto& v : I) A.push_back (ampAt (v, w, ln.rate, 100.0)); pnPar = together (A); }
        const Vec sl3 = uniformFilter (totL, 300, 1), sr3 = uniformFilter (totR, 300, 1); Vec a, b;
        for (size_t t = 0; t < T; ++t) if (act[t]) { a.push_back (totL[t] - sl3[t]); b.push_back (totR[t] - sr3[t]); }
        pnEnvcorr = stdev (a) > 1e-6 && stdev (b) > 1e-6 ? correlation (a, b) : 1.0;
    }
    put ("pn_rate", pnRate); put ("pn_share", pnShare); put ("pn_r2", pnR2); put ("pn_depth", pnDepth); put ("pn_par", pnPar); put ("pn_sd", pnSd); put ("pn_envcorr", pnEnvcorr);

    // ---------------------------------------------------------------- sample values (as 32-bit floats): few distinct values, held samples (bit crush), flat tops (clipping)
    double svTopk = 0, svHold = 0, svFlat = 0, svTop = 0; const double pk = y.peak() + 1e-12;
    for (const Vec* ch : { &L, &R })
    {
        std::vector<float> x (n), live; live.reserve (n); std::vector<char> nz (n, 0);
        for (size_t i = 0; i < n; ++i) { x[i] = (float) (*ch)[i]; if ((double) std::abs (x[i]) > 1e-6 * pk) { nz[i] = 1; live.push_back (x[i]); } }
        if (live.size() < 1000) continue;
        const double count = (double) live.size();
        std::sort (live.begin(), live.end()); std::vector<size_t> runs;
        for (size_t i = 0; i < live.size();) { size_t j = i + 1; while (j < live.size() && live[j] == live[i]) ++j; runs.push_back (j - i); i = j; }
        if (runs.size() > 1024) { std::nth_element (runs.begin(), runs.begin() + 1024, runs.end(), std::greater<size_t>()); runs.resize (1024); }
        size_t common = 0; for (size_t r : runs) common += r;
        svTopk = std::max (svTopk, (double) common / count);       // the share of the samples that carry one of the 1024 most frequent values
        size_t same = 0, flat = 0, high = 0; float p1 = 0; for (float v : x) p1 = std::max (p1, std::abs (v));
        const float hot = 0.9f * p1, upper = 0.8f * p1, step = 1e-4f * p1;
        for (size_t i = 0; i < n; ++i)
        {
            high += std::abs (x[i]) > upper;
            if (i > 0) { same += x[i] == x[i - 1] && nz[i]; flat += std::abs (x[i]) > hot && std::abs (x[i] - x[i - 1]) < step; }
        }
        svHold = std::max (svHold, (double) same / count); svFlat = std::max (svFlat, (double) flat / (double) n); svTop = std::max (svTop, (double) high / (double) n);
    }
    put ("sv_topk", svTopk); put ("sv_hold", svHold); put ("sv_flat", svFlat); put ("sv_top", svTop);

    // ---------------------------------------------------------------- a carrier: a steady line in the envelope at audio rate (ring modulator)
    double crStab = 0, crZ = 0, crF = 0;
    {
        const size_t seg = 8192, hop = 4096; Vec e2 (n); double e2max = 0; for (size_t i = 0; i < n; ++i) { e2[i] = M[i] * M[i]; e2max = std::max (e2max, e2[i]); }
        std::vector<size_t> starts; Vec means; std::vector<char> flat;       // flat: every value of the stretch is the same (a square wave, one bit): no line, and none made up from rounding
        for (size_t s = 0; s + seg < n; s += hop)
        {
            double m = 0, lo = e2[s], hi = e2[s]; for (size_t i = 0; i < seg; ++i) { m += e2[s + i]; lo = std::min (lo, e2[s + i]); hi = std::max (hi, e2[s + i]); }
            m /= (double) seg; if (m < 1e-4 * e2max) continue;
            starts.push_back (s); means.push_back (m); flat.push_back (lo == hi);
        }
        const auto sel = binRange (seg / 2 + 1, [&] (size_t k) { return binHz (k, seg) >= 40.0 && binHz (k, seg) <= 4500.0; });
        const Vec w = hanningSym (seg); Vec fs (starts.size()), zs (starts.size());
        powerSpectra (seg, starts.size(),
                      [&] (size_t j, Vec& buf) { for (size_t i = 0; i < seg; ++i) buf[i] = flat[j] ? 0.0 : (e2[starts[j] + i] - means[j]) * w[i]; },
                      [&] (size_t j, const Vec& p)
                      {
                          if (flat[j]) { fs[j] = 0.0; zs[j] = 10.0 * std::log10 (1e-30); return; }       // as an all-zero spectrum comes out below
                          size_t bk = 0; double best = 0; for (size_t k = sel.first; k < sel.second; ++k) if (p[k] > best) { best = p[k]; bk = k; }
                          fs[j] = binHz (bk, seg); zs[j] = 10.0 * std::log10 (p[bk] / (median (Vec (p.begin() + (long) sel.first, p.begin() + (long) sel.second)) + 1e-30) + 1e-30);
                      });
        if (fs.size() >= 6)
        {
            const double fm = median (fs), tol = std::max (1.5 * SR / (double) seg, 0.015 * fm); size_t in = 0; for (double f : fs) in += std::abs (f - fm) <= tol;
            crStab = (double) in / (double) fs.size(); crZ = median (zs); crF = std::log2 (fm + 1.0);
        }
    }
    put ("cr_stab", crStab); put ("cr_z", crZ); put ("cr_f", crF);

    // ---------------------------------------------------------------- stretches that repeat exactly (stutter)
    double rpFrac = 0, rpContrast = 0, rpLag = 0;
    if (n >= 16)
    {
        const size_t m = std::min (n, (size_t) (12 * SR)); Vec c = hand::ceps (m == n ? M : Vec (M.begin(), M.begin() + (long) m)); for (auto& v : c) v = std::abs (v);
        const long lo = (long) (0.02 * SR), hi = std::min ((long) (0.75 * SR), (long) c.size() / 2 - 1);
        if (hi > lo + 100)
        {
            const Vec seg (c.begin() + lo, c.begin() + hi); const double med = median (seg); Vec dev (seg.size()); for (size_t i = 0; i < seg.size(); ++i) dev[i] = std::abs (seg[i] - med);
            const double mad = median (dev) + 1e-12;
            // the share of 10 ms pieces that come again `lag` samples later, all but exactly
            auto rep = [&] (long lag) -> double
            {
                const size_t w = 480; if (lag < 1 || (long) m - lag < 0) return 0.0;
                const size_t k = (m - (size_t) lag) / w; if (k < 20) return 0.0;
                Vec ea (k, 0.0), eb (k, 0.0), ab (k, 0.0); double eaMax = 0;
                for (size_t j = 0; j < k; ++j)
                {
                    for (size_t i = 0; i < w; ++i) { const double a = M[j * w + i], b = M[(size_t) lag + j * w + i]; ea[j] += a * a; eb[j] += b * b; ab[j] += a * b; }
                    eaMax = std::max (eaMax, ea[j]);
                }
                size_t alive = 0, close = 0;
                for (size_t j = 0; j < k; ++j) if (ea[j] > 1e-4 * eaMax && eb[j] > 1e-4 * eaMax) { ++alive; close += ab[j] / std::sqrt (ea[j] * eb[j]) > 0.98; }
                return alive < 10 ? 0.0 : (double) close / (double) alive;
            };
            std::vector<size_t> idx (seg.size()); std::iota (idx.begin(), idx.end(), 0); const size_t three = std::min<size_t> (3, idx.size());
            std::partial_sort (idx.begin(), idx.begin() + (long) three, idx.end(), [&] (size_t a, size_t b) { return seg[a] > seg[b]; });
            long bestLag = 0;
            for (size_t q = 0; q < three; ++q)
            {
                if ((seg[idx[q]] - med) / mad < 15.0) break;
                const long lag = (long) idx[q] + lo; const double fr = rep (lag), con = fr - std::max (rep ((long) std::nearbyint (0.618 * (double) lag)), rep ((long) std::nearbyint (1.382 * (double) lag)));
                if ((con > rpContrast || bestLag == 0) && con >= rpContrast) { rpFrac = fr; rpContrast = con; bestLag = lag; }
            }
            rpLag = (double) bestLag / SR * 1e3;
        }
    }
    put ("rp_frac", rpFrac); put ("rp_contrast", rpContrast); put ("rp_lag", rpLag);

    // ---------------------------------------------------------------- the long-term spectrum: an edge (a fixed filter), the top end (exciter)
    double lpSlope = 0, lpF = 0, hpSlope = 0, hpF = 0, peak = 0, hfDb = -80, lfDb = -80, cent = 0, topP95 = 0, topSpread = 0, hfCorr = 0, hfSlope = 0;
    if (T >= 100 && nact > 50)
    {
        const int N6 = 52; Vec fe6 ((size_t) N6 + 1); for (int i = 0; i <= N6; ++i) fe6[(size_t) i] = 50.0 * std::pow (2.0, (double) i / 6.0);      // sixth-octave bands from 50 Hz
        // with 6 Hz lines (welch, 8192), so that the low sixth-octaves are filled
        const size_t NW = 8192; const Vec Pw = welch (M, NW); Vec lt ((size_t) N6); double ltMax = -1e300;
        for (int i = 0; i < N6; ++i)
        {
            const auto b = binRange (NW / 2 + 1, [&] (size_t k) { return binHz (k, NW) >= fe6[(size_t) i] && binHz (k, NW) < fe6[(size_t) i + 1]; });
            double s = 0; for (size_t k = b.first; k < b.second; ++k) s += Pw[k];
            lt[(size_t) i] = 10.0 * std::log10 (s / (double) (b.second - b.first) + 1e-20); ltMax = std::max (ltMax, lt[(size_t) i]);
        }
        for (auto& v : lt) v = std::max (v, ltMax - 80.0);
        double fall = 0, rise = 0; size_t kf = 0, kr = 0; peak = -1e300;      // the change over an octave: the steepest fall that starts in a live band, the steepest rise that ends in one
        for (size_t j = 0; j + 6 < (size_t) N6; ++j)
        {
            const double d = lt[j + 6] - lt[j], down = lt[j] > ltMax - 30.0 ? d : 0.0, up = lt[j + 6] > ltMax - 30.0 ? d : 0.0;
            if (j == 0 || down < fall) { fall = down; kf = j; }
            if (j == 0 || up > rise) { rise = up; kr = j; }
            peak = std::max (peak, lt[j + 3] > ltMax - 30.0 ? lt[j + 3] - 0.5 * (lt[j] + lt[j + 6]) : 0.0);       // a peak standing out of its half octave either side
        }
        lpSlope = -(lt[kf + 6] - lt[kf]); lpF = std::log2 (fe6[kf + 3]); hpSlope = lt[kr + 6] - lt[kr]; hpF = std::log2 (fe6[kr + 3]);
        Vec mp (KB, 0.0); for (size_t t = 0; t < T; ++t) if (act[t]) for (size_t k = 0; k < KB; ++k) mp[k] += PM[t * KB + k];
        double tp = 0, hf = 0, lf = 0, fsum = 0;
        for (size_t k = 0; k < KB; ++k) { mp[k] /= (double) nact; tp += mp[k]; fsum += mp[k] * hz (k); if (hz (k) >= 6000.0) hf += mp[k]; if (hz (k) < 150.0) lf += mp[k]; }
        tp += 1e-30; hfDb = std::max (-80.0, 10.0 * std::log10 (hf / tp + 1e-30)); lfDb = std::max (-80.0, 10.0 * std::log10 (lf / tp + 1e-30)); cent = std::log2 (fsum / tp + 1.0);
        // each frame's highest sixth-octave band (from 400 Hz up) within 30 dB of its strongest; mid band against top band over time
        std::vector<std::pair<size_t, size_t>> b6; for (int i = 18; i < N6; ++i) b6.push_back (binRange (KB, [&] (size_t k) { return hz (k) >= fe6[(size_t) i] && hz (k) < fe6[(size_t) i + 1]; }));
        const auto midB = binRange (KB, [&] (size_t k) { return hz (k) >= 500.0 && hz (k) < 3000.0; }), hfB = binRange (KB, [&] (size_t k) { return hz (k) >= 5000.0 && hz (k) < 12000.0; });
        Vec tops, mid, high; Vec lf6 (b6.size());
        for (size_t t = 0; t < T; ++t)
        {
            if (! act[t]) continue;
            double mx = -1e300;
            for (size_t b = 0; b < b6.size(); ++b)
            {
                double s = 0; for (size_t k = b6[b].first; k < b6[b].second; ++k) s += PM[t * KB + k];
                lf6[b] = 10.0 * std::log10 (s / (double) std::max<size_t> (1, b6[b].second - b6[b].first) + 1e-16); mx = std::max (mx, lf6[b]);
            }
            size_t topBand = 18; for (size_t b = b6.size(); b-- > 0;) if (lf6[b] > mx - 30.0) { topBand = 18 + b; break; }
            tops.push_back (std::log2 (fe6[topBand + 1])); mid.push_back (bandDb (PM, t, midB, 1e-14)); high.push_back (bandDb (PM, t, hfB, 1e-14));
        }
        topP95 = percentile (tops, 95.0); topSpread = topP95 - percentile (tops, 50.0);
        if (stdev (mid) > 1e-6 && stdev (high) > 1e-6)
        {
            hfCorr = correlation (mid, high);
            const double mm = mean (mid), mh = mean (high); double sxy = 0, sxx = 0; for (size_t i = 0; i < mid.size(); ++i) { sxy += (mid[i] - mm) * (high[i] - mh); sxx += (mid[i] - mm) * (mid[i] - mm); }
            hfSlope = sxy / sxx;
        }
    }
    put ("ft_lp_slope", lpSlope); put ("ft_lp_f", lpF); put ("ft_hp_slope", hpSlope); put ("ft_hp_f", hpF); put ("ft_peak", peak); put ("ft_hf_db", hfDb); put ("ft_lf_db", lfDb);
    put ("ft_cent", cent); put ("ft_top_p95", topP95); put ("ft_top_spread", topSpread); put ("hf_corr", hfCorr); put ("hf_slope", hfSlope);

    // ---------------------------------------------------------------- a ceiling: do the loud stretches have flatter waves than the quiet ones (saturation, distortion)
    double satCeil = 0, satCrestTop = 0, satCrestDiff = 0, satLvlSpread = 0, satPkSpread = 0;
    if (n >= 1024 + 49 * 512)
    {
        const Vec& x = sl >= sr ? L : R; const size_t nf = (n - 1024) / 512 + 1; Vec rm (nf), pkf (nf); double rmMax = 0, pkMax = 0;
        for (size_t j = 0; j < nf; ++j)
        {
            double e = 0, p = 0; for (size_t i = 0; i < 1024; ++i) { const double v = x[j * 512 + i]; e += v * v; p = std::max (p, std::abs (v)); }
            rm[j] = std::sqrt (e / 1024.0) + 1e-12; pkf[j] = p + 1e-12; rmMax = std::max (rmMax, rm[j]); pkMax = std::max (pkMax, pkf[j]);
        }
        Vec cr, lv, pd; size_t ceil = 0; const double gate = rmMax * std::pow (10.0, -30.0 / 20.0);
        for (size_t j = 0; j < nf; ++j) if (rm[j] > gate) { cr.push_back (20.0 * std::log10 (pkf[j] / rm[j])); lv.push_back (20.0 * std::log10 (rm[j])); pd.push_back (20.0 * std::log10 (pkf[j])); ceil += pkf[j] > 0.891 * pkMax; }
        if (lv.size() >= 30)
        {
            std::vector<size_t> q (lv.size()); std::iota (q.begin(), q.end(), 0); std::stable_sort (q.begin(), q.end(), [&] (size_t a, size_t b) { return lv[a] < lv[b]; });
            const size_t k3 = std::max<size_t> (5, q.size() / 3); Vec loud, quiet; for (size_t i = 0; i < k3; ++i) { quiet.push_back (cr[q[i]]); loud.push_back (cr[q[q.size() - 1 - i]]); }
            satCeil = (double) ceil / (double) lv.size(); satCrestTop = median (loud); satCrestDiff = satCrestTop - median (quiet);
            satLvlSpread = percentile (lv, 95.0) - percentile (lv, 50.0); satPkSpread = percentile (pd, 95.0) - percentile (pd, 50.0);
        }
    }
    put ("sat_ceil", satCeil); put ("sat_crest_top", satCrestTop); put ("sat_crest_diff", satCrestDiff); put ("sat_lvl_spread", satLvlSpread); put ("sat_pk_spread", satPkSpread);
    return out;
}

// ---------------------------------------------------------------- the models (Resources/verify.bin)
// Little-endian: 'DRVF', version u32 (2), the inputs' names (count u32, then length u32 + bytes each), the effects
// (count u32), then per effect: its name, the raw tree output that stands for 65 % (f32), the slope of the score
// (f32), and the trees as Models.h Boosted reads them.
struct Verifier
{
    struct Family { std::string name; double at65 = 0, slope = 1; Boosted trees; };
    std::vector<std::string> names; std::vector<Family> families;
    void load (const void* data, size_t n)
    {
        Reader r (data, n); char magic[4]; r.take (magic, 4);
        if (std::memcmp (magic, "DRVF", 4) != 0 || r.u32() != 2) throw std::runtime_error ("verify.bin is not a verifier file of this version");
        names.resize (r.u32()); for (auto& s : names) s = r.str();
        families.resize (r.u32());
        for (auto& f : families) { f.name = r.str(); f.at65 = r.f32(); f.slope = r.f32(); f.trees.load (r); if (f.trees.base.size() != 1) throw std::runtime_error ("verify.bin: one output per effect expected"); }
    }
    bool ready() const { return ! families.empty(); }
    // The score shown for an effect whose trees give z: 65 % where the fixed cut is, rising with z, never quite 100 %.
    static double shown (const Family& f, double z) { return std::min (0.999, 1.0 / (1.0 + std::exp (-(f.slope * (z - f.at65) + std::log (0.65 / 0.35))))); }
};

namespace verif
{
// The blocks a sound is judged in, as (first sample, length): 6 s each, an end piece shorter than 4.5 s stays with
// the block before it. A sound shorter than 10.5 s is one block.
inline std::vector<std::pair<size_t, size_t>> blocks (size_t n)
{
    const size_t B = (size_t) (6 * SR), join = (size_t) (4.5 * SR), full = n / B, rest = n - full * B;
    std::vector<std::pair<size_t, size_t>> out; if (full == 0) { out.push_back ({ 0, n }); return out; }
    for (size_t b = 0; b < full; ++b) out.push_back ({ b * B, B });
    if (rest < join) out.back().second += rest; else out.push_back ({ full * B, rest });
    return out;
}
// One block's inputs by name, as the trees were trained on them (Research/detect/verifier.py feat_names): the
// block is taken as a sound of its own. rows / rows27: the block's windows in I.win / I.win27. echo: the echo
// check on the block (score, ms).
inline std::unordered_map<std::string, float> inputs (const Inspection& I, const std::vector<size_t>& rows, const std::vector<size_t>& rows27, std::pair<double, double> echo, const Numbers& ev)
{
    static const char* const handNames[15] = { "echo_sc", "echo_ms", "side_ratio", "corr", "haas_z", "haas_ms", "pan_r2", "pan_span", "clip_frac", "log_unique", "hold", "crest", "kurt", "trem_r2", "trem_depth" };
    static const char* const six[6] = { "saturation", "exciter", "ringmod", "filter", "stutter", "gatedrhythm" };
    std::unordered_map<std::string, float> x;
    // mean, highest and lowest over the windows (in single precision, summed in order, as the training code has it)
    auto gather = [&x] (const std::string& key, size_t count, const std::vector<size_t>& rw, auto name, auto value)
    {
        for (size_t k = 0; k < count; ++k)
        {
            float sum = 0, hi = 0, lo = 0; bool first = true;
            for (size_t r : rw) { const float v = value (r, k); sum += v; hi = first ? v : std::max (hi, v); lo = first ? v : std::min (lo, v); first = false; }
            const std::string base = key + "_" + name (k) + "_";
            x[base + "mean"] = rw.empty() ? 0.0f : sum / (float) rw.size(); x[base + "max"] = hi; x[base + "min"] = lo;
        }
    };
    gather ("lg12", I.fx.size(), rows, [&] (size_t k) { return I.fx[k]; }, [&] (size_t r, size_t k) { return I.win[r].logits[k]; });
    gather ("lg27", I.fx27.size(), rows27, [&] (size_t k) { return I.fx27[k]; }, [&] (size_t r, size_t k) { return I.win27[r].logits[k]; });
    gather ("hd", 15, rows, [&] (size_t k) { return std::string (handNames[k]); }, [&] (size_t r, size_t k) { return I.win[r].hand[k]; });
    gather ("p12", I.fx.size(), rows, [&] (size_t k) { return I.fx[k]; }, [&] (size_t r, size_t k) { return I.win[r].p[k]; });
    gather ("p27", I.fx27.size(), rows27, [&] (size_t k) { return I.fx27[k]; }, [&] (size_t r, size_t k) { return I.win27[r].p[k]; });
    // the scores as they were shown up to 0.4.1 (Inspector.h inspect and Estimate.h mergeNew with its 80 % for the six later effects)
    std::map<std::string, double> s12, s27;
    for (size_t k = 0; k < I.fx.size(); ++k) s12[I.fx[k]] = (double) x[std::string ("p12_") + I.fx[k] + "_max"];
    for (size_t k = 0; k < I.fx27.size(); ++k) s27[I.fx27[k]] = (double) x[std::string ("p27_") + I.fx27[k] + "_max"];
    for (auto* s : { &s12, &s27 }) if (echo.first > 60 && (*s)["bitcrush"] < 0.5) (*s)["delay"] = std::max ((*s)["delay"], std::min (0.99, 0.6 + echo.first / 1000.0));
    std::map<std::string, double> fin = s12; bool anyNew = false;
    for (const char* f : six) { fin[f] = s27[f]; anyNew = anyNew || fin[f] > 0.8; }
    if (anyNew) for (auto& f : I.fx) { auto q = s27.find (f); if (q != s27.end() && q->second < 0.15) fin[f] = std::min (fin[f], q->second); }
    for (auto& [f, v] : fin) x["fin_" + f] = (float) v;
    x["wf_sc"] = (float) echo.first; x["wf_ms"] = (float) echo.second;
    for (auto& [k, v] : ev) x[k] = (float) v;
    return x;
}
} // namespace verif

// Puts the verifier's score for every effect it knows into I.prob (in place of the detectors' own), except for the
// effects named in `leave`: those belong to an engine of their own, whose verdict is not touched whether it is
// already in I.prob or comes later. Reverb (the room engine) and the comb filter (the tone engine) are not among
// the verifier's effects at all. Needs the inspection of both detectors (inspect, then mergeNew).
// stop (optional) is asked before every block; when it says yes, nothing is written and false comes back.
// blockScores (optional): every judged block's score per effect, for the tests.
inline bool verify (Inspection& I, const Stereo& y, const Verifier& V, const std::set<std::string>& leave = {}, const std::function<bool()>& stop = {},
                    std::map<std::string, Vec>* blockScores = nullptr)
{
    using namespace verif;
    if (! V.ready()) return true;
    if (I.fx.empty() || I.fx27.empty() || I.win.size() != I.win27.size()) throw std::runtime_error ("verify: needs the inspection of both detectors");
    const size_t W = (size_t) (3 * SR); const auto bl = blocks (y.size());
    std::map<std::string, Vec> scores; std::vector<float> x (V.names.size());
    for (const auto& [b0, len] : bl)
    {
        if (stop && stop()) return false;
        // the block's windows: those that lie wholly inside it (a sound of 3 s or less has its one padded window)
        std::vector<size_t> rows;
        for (size_t r = 0; r < I.win.size(); ++r) { const size_t s = I.win[r].start; if (len <= W ? s == b0 : (s >= b0 && s + W <= b0 + len)) rows.push_back (r); }
        if (rows.empty()) continue;         // nothing loud enough for the detectors in it: no verdict from this block
        for (size_t r : rows) if (I.win27[r].start != I.win[r].start) throw std::runtime_error ("verify: the two inspections do not match");
        const bool whole = bl.size() == 1; const Stereo part = whole ? Stereo() : y.slice (b0, len); const Stereo& yb = whole ? y : part;
        const auto in = inputs (I, rows, rows, whole ? std::make_pair (I.echoScore, I.echoMs) : hand::echoDetect (yb), evidence (yb));
        for (size_t i = 0; i < V.names.size(); ++i)
        {
            auto p = in.find (V.names[i]); if (p == in.end()) throw std::runtime_error ("verify.bin asks for a number this build does not have: " + V.names[i]);
            x[i] = p->second;
        }
        for (auto& f : V.families) if (! leave.count (f.name)) scores[f.name].push_back (Verifier::shown (f, f.trees.raw (x)[0]));
    }
    for (auto& f : V.families)
    {
        if (leave.count (f.name)) continue;
        Vec s = scores[f.name]; std::sort (s.begin(), s.end(), std::greater<double>());
        I.prob[f.name] = s.empty() ? 0.0 : s[(s.size() + 1) / 2 - 1];      // the highest score that at least half of the blocks reach
    }
    if (blockScores != nullptr) *blockScores = scores;
    return true;
}

} // namespace dr
