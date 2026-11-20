#pragma once
// The bend engine (DISTORTION, SATURATION): a fixed curve that a sound's values were pushed through.
//   features()  what the meter sees of a sound, one half-wave at a time (the values above zero, or those under):
//               frames of 2048 samples sorted into ten classes by their peak against the half-wave's own; per class
//               how the frame's values lie under its peak (eleven places in their order), how steep the wave is at six
//               heights against the frame's own steepness, and how many frames the class has; and how much of the
//               half-wave sits within 0.1, 0.5, 1 and 2 dB of its peak
//   Meter       a few small nets taught with sounds through curves of many shapes (Research/bend): per half-wave, by
//               how many dB the curve squeezed each of eleven levels of the sound as it was (30 dB under its peak up
//               to the peak) against its plain gain for small values; and whether a bend is there to be heard. The
//               nets are taught alike from different starts and their answers averaged: on a real bend they agree, on
//               a chance find they do not
//   find()      the meter's answer for one sound
//   own()       the sound itself with the group's Amount at s: 0 = the bend walked back (three quarters of what was
//               read, less 1 dB: careful()), 1 = untouched (bit for bit), 2 = pushed as far again
//   onto()      another sound through the found curve: 0 = none, 1 = its peak pushed as far into the curve as the
//               first sound's was, 2 = as far again
// The curve is not one of a few textbook shapes: it is the eleven numbers per half-wave, joined smoothly, so a tube's
// uneven halves, a tape's soft shoulder and a clipper's corner are each what they were read as.
// The way back spreads values apart again. Where the curve pressed many levels into one (flat tops), nothing in the
// sound says which was which: there the way back spreads only as far as spreadMost allows, and the tops stay flat.
// What it does not do: a box whose bend is hidden behind a strong filter or a room (the wave no longer shows it); a
// limiter or a compressor (a level held down by a moving gain leaves the wave's shape alone: the level engine's work).
#include "Tone.h"
#include <cstring>
#include <stdexcept>

namespace dr
{
namespace bend
{
using Stop = room::Stop;
constexpr int FR = 2048, HOPF = 1024, NAL = 12, NLEV = 10, NVAL = 5, NG = 11;
constexpr int NF = (NAL - 1) * NLEV + NLEV + (NVAL + 1) * NLEV + 4;         // 184 numbers per half-wave
constexpr int PLACE[NAL] = { 1535, 1638, 1740, 1842, 1904, 1945, 1975, 2006, 2027, 2037, 2045, 2047 };     // places in a frame's values in rising order (75 % .. the largest)
constexpr double LEV[NLEV] = { -48.0, -36.0, -27.0, -20.0, -14.0, -9.0, -5.0, -2.5, -1.0, -0.3 };            // the frame classes: a frame's peak against the half-wave's (dB), from
constexpr double VAL[NVAL] = { 0.2, 0.4, 0.6, 0.8, 0.9 };                                                    // the heights: a value against its frame's peak
constexpr double GRID[NG] = { -30.0, -26.0, -22.0, -18.0, -15.0, -12.0, -9.0, -6.0, -4.0, -2.0, 0.0 };        // where the answer is given: levels of the sound as it was against its peak (dB)
constexpr double SMALL = -34.0;                 // values this far under the peak and further pass with the curve's plain gain
constexpr double maxSqueeze = 40.0;             // no answer is used beyond this many dB
constexpr double spreadMost = 6.0;              // the way back spreads values at most this many times as far apart as they lie in the sound (in dB)
constexpr size_t minLen = 24000;                // half a second: under that nothing is read
constexpr double backPart = 0.75, backLess = 1.0;        // the way back takes this part of the read squeeze, less this many dB (see careful())
constexpr double sureAt = -2.0, sureTop = 5.0;  // the meter's verdict (a logit) and the squeeze it reads at the peak (dB) from which a bend is shown with the Sureness slider where it stands at first: both must be there
constexpr double lowAt = -6.0, topAt = 3.0;     // ... and from which it is kept at all: such a bend shows when the slider is turned down
constexpr double hardAt = 10.0;                 // squeezed by this much at the peak the group is called DISTORTION, under it SATURATION
constexpr double shownSure = 0.65;              // what the Sureness slider cuts at by default: a bend just found shows this

// A sound's peak without its few largest values (1 in 10000). pol 0: of both half-waves; +1 or -1: of that half-wave
// alone (the other half counts as zeros).
inline double peakOf (const Stereo& y, double pol)
{
    const size_t n = y.size(); if (n == 0) return 0.0;
    std::vector<double> a; a.reserve (2 * n);
    for (const Vec* v : { &y.l, &y.r }) for (size_t i = 0; i < n; ++i) a.push_back (pol == 0.0 ? std::abs ((*v)[i]) : std::max (pol * (*v)[i], 0.0));
    const size_t k = std::max<size_t> (1, (size_t) std::floor ((double) a.size() * 1e-4 + 0.5)), at = a.size() - std::min (k, a.size());
    std::nth_element (a.begin(), a.begin() + (std::ptrdiff_t) at, a.end()); return a[at];
}

// What the meter sees of one half-wave: NF numbers, and the half-wave's peak. False when given up.
inline bool features (const Stereo& y, double pol, float* out, double& peak, const Stop& stop = {})
{
    const double vref = std::max (peakOf (y, pol), 1e-9); peak = vref;
    double qAcc[NLEV][NAL - 1] = {}, qN[NLEV] = {}, sAcc[NLEV][NVAL + 1] = {}, sN[NLEV][NVAL + 1] = {}, near[4] = {}; double npos = 0;
    const double nearAt[4] = { vref * std::pow (10.0, -0.1 / 20.0), vref * std::pow (10.0, -0.5 / 20.0), vref * std::pow (10.0, -1.0 / 20.0), vref * std::pow (10.0, -2.0 / 20.0) };
    const long n = (long) y.size(); std::vector<double> z ((size_t) std::max (n, 0L)), d2 (z.size()), srt ((size_t) FR);
    for (const Vec* ch : { &y.l, &y.r })
    {
        for (long i = 0; i < n; ++i) z[(size_t) i] = std::max (pol * (*ch)[(size_t) i], 0.0);
        for (long i = 1; i + 1 < n; ++i) { const double d = 0.5 * pol * ((*ch)[(size_t) (i + 1)] - (*ch)[(size_t) (i - 1)]); d2[(size_t) i] = d * d; }       // how steep the wave is at each value
        if (n > 0) d2[0] = d2[(size_t) (n - 1)] = 0.0;
        const long frames = n >= FR ? (n - FR) / HOPF + 1 : 0;
        for (long t = 0; t < frames; ++t)
        {
            if (stop && (t & 63) == 0 && stop()) return false;
            const double* f = z.data() + (size_t) (t * HOPF); const double* fd = d2.data() + (size_t) (t * HOPF);
            std::copy (f, f + FR, srt.begin()); std::nth_element (srt.begin(), srt.begin() + PLACE[0], srt.end()); std::sort (srt.begin() + PLACE[0], srt.end());
            const double p = srt[(size_t) PLACE[NAL - 1]], lev = 20.0 * std::log10 (std::max (p, 1e-12) / vref); if (! (lev > LEV[0])) continue;
            int b = 0; while (b + 1 < NLEV && lev >= LEV[b + 1]) ++b;
            for (int a = 0; a + 1 < NAL; ++a) qAcc[b][a] += std::max (20.0 * std::log10 (std::max (srt[(size_t) PLACE[a]], 1e-12) / std::max (p, 1e-12)), -60.0);
            qN[b] += 1.0;
            double sum[NVAL + 1] = {}, tot = 0; long cnt[NVAL + 1] = {}, pos = 0; const double pp = std::max (p, 1e-12);
            for (int i = 0; i < FR; ++i)
            {
                if (! (f[i] > 0.0)) continue;
                const double r = f[i] / pp; int k = 0; while (k < NVAL && r >= VAL[k]) ++k;
                sum[k] += fd[i]; ++cnt[k]; tot += fd[i]; ++pos;
            }
            tot /= (double) std::max (pos, 1L);
            if (tot > 0.0) for (int k = 0; k <= NVAL; ++k) if (cnt[k] >= 8)
            {
                sAcc[b][k] += std::clamp (10.0 * std::log10 (std::max (sum[k] / (double) cnt[k], 1e-30) / std::max (tot, 1e-30)), -40.0, 20.0); sN[b][k] += 1.0;
            }
        }
        for (long i = 0; i < n; ++i) { const double v = z[(size_t) i]; if (v > 0.0) { npos += 1.0; for (int k = 0; k < 4; ++k) if (v > nearAt[k]) near[k] += 1.0; } }
    }
    double all = 0; for (int b = 0; b < NLEV; ++b) all += qN[b];
    int o = 0;
    for (int b = 0; b < NLEV; ++b) for (int a = 0; a + 1 < NAL; ++a) out[o++] = (float) (qAcc[b][a] / std::max (qN[b], 1.0) / 20.0);
    for (int b = 0; b < NLEV; ++b) out[o++] = (float) (qN[b] / std::max (all, 1.0));
    for (int b = 0; b < NLEV; ++b) for (int k = 0; k <= NVAL; ++k) out[o++] = (float) (sAcc[b][k] / std::max (sN[b][k], 1.0) / 10.0);
    for (int k = 0; k < 4; ++k) out[o++] = (float) (std::log10 (near[k] / std::max (npos, 1.0) + 1e-6) / 3.0);
    return true;
}

// ---------------------------------------------------------------- the meter
struct Meter
{
    struct Lin { int in = 0, out = 0; std::vector<float> w, b; };       // w[out][in]
    struct Net { std::vector<float> mean, sd; std::vector<Lin> lin; };
    std::vector<Net> nets;              // several nets, taught alike from different starts: the meter's answer is the mean of theirs
    bool ready() const { return ! nets.empty(); }
    [[noreturn]] static void fail (const char* why) { throw std::runtime_error (std::string ("bend meter model: ") + why); }

    // Reads the file written by Tools/export_bend_model.py from memory. Throws std::runtime_error, and stays as it
    // was, when the block is not a complete model of the sizes its own header gives.
    void load (const void* data, size_t size)
    {
        const uint8_t* p = (const uint8_t*) data; size_t left = data != nullptr ? size : 0;
        auto take = [&] (void* dst, size_t bytes) { if (bytes > left) fail ("the file is cut short"); std::memcpy (dst, p, bytes); p += bytes; left -= bytes; };
        auto u32 = [&] { uint32_t v = 0; take (&v, 4); return v; };
        auto floats = [&] (std::vector<float>& v, size_t count)
        {
            if ((size_t) u32() != count) fail ("a layer does not have the size the header gives");
            if (count > left / 4) fail ("the file is cut short");
            v.resize (count); take (v.data(), count * 4);
            for (float f : v) if (! std::isfinite (f)) fail ("a number is not finite");
        };
        char magic[4]; take (magic, 4); if (std::memcmp (magic, "DRBM", 4) != 0) fail ("not a bend meter model file");
        if (u32() != 2) fail ("unknown format version");
        if (u32() != (uint32_t) NF) fail ("made for a different number of inputs");
        if (u32() != (uint32_t) NG) fail ("made for a different number of levels");
        const uint32_t count = u32(); if (count < 1 || count > 16) fail ("bad number of nets");
        Meter m;
        for (uint32_t k = 0; k < count; ++k)
        {
            const uint32_t layers = u32(); if (layers < 1 || layers > 16) fail ("bad number of layers");
            Net net; floats (net.mean, (size_t) NF); floats (net.sd, (size_t) NF);
            for (float s : net.sd) if (! (s > 0.0f)) fail ("a spread is not above zero");
            int in = 2 * NF + 1;
            for (uint32_t i = 0; i < layers; ++i)
            {
                const uint32_t out = u32(); if (out < 1 || out > 4096) fail ("bad layer width");
                Lin l; l.in = in; l.out = (int) out; floats (l.w, (size_t) out * (size_t) in); floats (l.b, (size_t) out); net.lin.push_back (std::move (l)); in = (int) out;
            }
            if (in != NG + 1) fail ("the last layer does not give the levels and the verdict");
            m.nets.push_back (std::move (net));
        }
        if (left != 0) fail ("more data than the header gives");
        *this = std::move (m);
    }

    // X: both half-waves' numbers ([2][NF], the upper half first); ratioDb: the upper half's peak against the lower
    // half's. Out: the squeeze per half-wave and level (dB), and the verdict (a logit: above 0 = more likely than not).
    void run (const float* X, double ratioDb, double S[2][NG], double& logit) const
    {
        logit = -99.0; for (int h = 0; h < 2; ++h) for (int g = 0; g < NG; ++g) S[h][g] = 0.0;
        if (! ready()) return;
        std::vector<float> Z ((size_t) 2 * (size_t) NF); double sum = 0;
        for (const Net& net : nets)
        {
            for (int h = 0; h < 2; ++h) for (int i = 0; i < NF; ++i) Z[(size_t) (h * NF + i)] = (X[h * NF + i] - net.mean[(size_t) i]) / net.sd[(size_t) i];
            for (int h = 0; h < 2; ++h)        // each half-wave in turn: its own numbers first, the other half's after, and how their peaks compare
            {
                std::vector<float> a ((size_t) (2 * NF + 1)), b;
                for (int i = 0; i < NF; ++i) { a[(size_t) i] = Z[(size_t) (h * NF + i)]; a[(size_t) (NF + i)] = Z[(size_t) ((1 - h) * NF + i)]; }
                a[(size_t) (2 * NF)] = (float) ((h == 0 ? ratioDb : -ratioDb) / 20.0);
                for (size_t li = 0; li < net.lin.size(); ++li)
                {
                    const Lin& L = net.lin[li]; b.assign ((size_t) L.out, 0.0f);
                    for (int o = 0; o < L.out; ++o)
                    {
                        const float* w = L.w.data() + (size_t) o * (size_t) L.in; double s = L.b[(size_t) o];
                        for (int i = 0; i < L.in; ++i) s += (double) w[i] * (double) a[(size_t) i];
                        b[(size_t) o] = li + 1 < net.lin.size() ? (float) std::max (s, 0.0) : (float) s;
                    }
                    a.swap (b);
                }
                for (int g = 0; g < NG; ++g) S[h][g] += 10.0 * (double) a[(size_t) g];
                sum += (double) a[(size_t) NG];
            }
        }
        const double k = 1.0 / (double) nets.size();
        for (int h = 0; h < 2; ++h) for (int g = 0; g < NG; ++g) S[h][g] *= k;
        logit = sum * k;
    }
};

// ---------------------------------------------------------------- the curve
// An answer made fit for use: no squeeze under zero, never falling towards the peak, none growing faster than the
// level itself allows (1 - 1/K dB per dB: the way back then spreads a value at most K times as far apart), held at
// maxSqueeze.
inline void tidy (const double* S, double K, double* r)
{
    const double most = 1.0 - 1.0 / K; double run = 0.0;
    for (int i = 0; i < NG; ++i) { run = std::max (run, std::min (std::max (S[i], 0.0), maxSqueeze)); r[i] = run; }
    r[0] = std::min (r[0], most * (GRID[0] - (SMALL - 3.0)));
    for (int i = 1; i < NG; ++i) r[i] = std::min (r[i], r[i - 1] + most * (GRID[i] - GRID[i - 1]));
}
// A smooth curve through rising points that never turns back between them (cubic pieces; the steepness at a point
// is the harmonic mean of the two straight pieces that meet there, at the two ends that of the one piece).
struct Smooth
{
    std::vector<double> x, y, m;
    Smooth (std::vector<double> xs, std::vector<double> ys) : x (std::move (xs)), y (std::move (ys)), m (x.size(), 0.0)
    {
        const size_t n = x.size(); std::vector<double> h (n - 1), d (n - 1);
        for (size_t i = 0; i + 1 < n; ++i) { h[i] = x[i + 1] - x[i]; d[i] = (y[i + 1] - y[i]) / h[i]; }
        m[0] = d[0]; m[n - 1] = d[n - 2];
        for (size_t i = 1; i + 1 < n; ++i) if (d[i - 1] * d[i] > 0.0) { const double w1 = 2.0 * h[i] + h[i - 1], w2 = h[i] + 2.0 * h[i - 1]; m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]); }
    }
    double at (double v) const
    {
        size_t i = (size_t) (std::upper_bound (x.begin(), x.end(), v) - x.begin()); i = i == 0 ? 0 : i - 1; i = std::min (i, x.size() - 2);
        const double h = x[i + 1] - x[i], t = (v - x[i]) / h, t2 = t * t, t3 = t2 * t;
        return (2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * h * m[i + 1];
    }
};
// One half-wave's curve both ways, on a table: inn (a value before the bend, in the scale where small values pass
// as they are) <-> out (the bent value, as in the sound). 16 steps per dB from 96 dB under the unbent peak to 12 over.
struct Curve
{
    static constexpr int N = 1729;
    double vp = 0, xp = 0, top = 0; std::vector<double> inn, out;
    Curve (const double* S, double peak, double K = spreadMost) : vp (peak), inn ((size_t) N), out ((size_t) N)
    {
        double r[NG]; tidy (S, K, r); top = r[NG - 1]; xp = vp * std::pow (10.0, top / 20.0);
        std::vector<double> lev { -60.0, SMALL - 3.0 }, sq { 0.0, 0.0 };
        for (int i = 0; i < NG; ++i) { lev.push_back (GRID[i]); sq.push_back (r[i]); }
        lev.push_back (12.0); sq.push_back (r[NG - 1] + (1.0 - 1.0 / K) * 12.0);         // above the peak: as flat as the cap allows
        const Smooth sm (lev, sq);
        for (int i = 0; i < N; ++i)
        {
            const double l = -96.0 + (double) i / 16.0; inn[(size_t) i] = xp * std::pow (10.0, l / 20.0); out[(size_t) i] = inn[(size_t) i] * std::pow (10.0, -sm.at (l) / 20.0);
            if (i > 0) out[(size_t) i] = std::max (out[(size_t) i], out[(size_t) (i - 1)] * (1.0 + 1e-9));       // the bent value never falls as the value before the bend rises
        }
    }
    static double look (const std::vector<double>& from, const std::vector<double>& to, double v)
    {
        if (v < from.front()) return v;         // under the table everything is a line
        if (v >= from.back()) return to.back();
        const size_t i = (size_t) (std::upper_bound (from.begin(), from.end(), v) - from.begin()) - 1;
        return to[i] + (to[i + 1] - to[i]) * (v - from[i]) / (from[i + 1] - from[i]);
    }
    double back (double v) const { return look (out, inn, v); }     // a bent value (>= 0) -> the value before the bend
    double fwd (double x) const { return look (inn, out, x); }      // a value before the bend (>= 0) -> bent
};

// ---------------------------------------------------------------- one sound's bend
struct Found
{
    bool found = false, looked = false, stopped = false;
    size_t n = 0;                       // the sound it belongs to: its length
    double logit = -99.0, sure = 0.0;   // the meter's verdict, and the sureness 0 .. 1 made of it and of the squeeze at the peak
    double S[2][NG] = {};               // the squeeze per half-wave (the upper first) and level, as the meter read it (dB)
    double vp[2] = { 0, 0 };            // each half-wave's peak in the sound
    double top = 0;                     // how far the loudest values were squeezed, the stronger half-wave's (dB, made fit for use)
    bool hard() const { return top >= hardAt; }
};
// How sure a bend is: the weaker of two witnesses, the meter's verdict and the squeeze it reads at the peak. The
// verdict alone is cautious (it was taught to say no where a bend cannot be heard, and says no to many a soft one
// that can); the squeeze alone takes any flat-topped sound. Together, on 96 sounds through outside boxes and 198
// clean ones: 45 of 48 distorted and 21 of 25 saturated sounds (6 dB and more at the peak) are shown where the slider
// stands at first, and of the clean ones only the four that have flat tops themselves (Research/bend).
inline double sureness (double logit, double top)
{
    const double e = std::min (logit - sureAt, top - sureTop);
    return 1.0 / (1.0 + std::exp (-e * 0.5 - std::log (shownSure / (1.0 - shownSure))));
}

inline Found find (const Stereo& x, const Meter& m, const Stop& stop = {})
{
    Found F; F.n = x.size(); if (! m.ready() || x.size() < minLen) return F;
    std::vector<float> X ((size_t) 2 * (size_t) NF);
    if (! features (x, 1.0, X.data(), F.vp[0], stop) || ! features (x, -1.0, X.data() + NF, F.vp[1], stop)) { F.stopped = true; return F; }
    F.looked = true;
    m.run (X.data(), 20.0 * std::log10 (F.vp[0] / F.vp[1]), F.S, F.logit);
    double r[NG]; for (int h = 0; h < 2; ++h) { tidy (F.S[h], spreadMost, r); F.top = std::max (F.top, r[NG - 1]); }
    F.sure = sureness (F.logit, F.top); F.found = F.logit >= lowAt && F.top >= topAt;
    return F;
}

// A file at another rate than the app's: the bend is read off the file's own samples, where every value still lies on
// the curve (bringing a sound to 48 kHz turns flat tops into ripples). From about twice the app's rate on, every
// second (third ...) sample is looked at: which samples are looked at does not matter to a curve, and the frames keep
// their length in time. n: the length of the sound at the app's rate, which the answer then belongs to.
inline Found findAt (const Stereo& raw, double rate, size_t n, const Meter& m, const Stop& stop = {})
{
    const size_t step = (size_t) std::max (1.0, std::floor (rate / SR + 0.25)); Found F;
    if (step <= 1) F = find (raw, m, stop);
    else
    {
        Stereo d; d.l.reserve (raw.size() / step + 1); d.r.reserve (raw.size() / step + 1);
        for (size_t i = 0; i < raw.size(); i += step) { d.l.push_back (raw.l[i]); d.r.push_back (raw.r[i]); }
        F = find (d, m, stop);
    }
    F.n = n; return F;
}

// The Amount as a drive: 0 = none of the found drive, 1 = as found, 2 = as far again in dB as the loudest values were squeezed.
inline double drive (double s, double top) { return s <= 1.0 ? s * s : std::pow (10.0, (s - 1.0) * top / 20.0); }

// The read squeeze as the way back uses it: three quarters of it, less 1 dB. The meter is off by a dB or two, and the
// two ways of being off are not alike: read too low, some of the bend stays in; read too high, values are spread apart
// that were never squeezed, which is a new fault and a worse one than the bend. On 48 outside cases this way back
// scores more than twice what the plain reading does and harms fewer of them (Research/bend).
inline void careful (const double* S, double* r) { for (int i = 0; i < NG; ++i) r[i] = std::max (backPart * std::min (std::max (S[i], 0.0), maxSqueeze) - backLess, 0.0); }

// The sound itself. s: 0 = the bend walked back (as far as careful() goes), 1 = untouched, 2 = pushed as far again.
// The sound's peak stays where it was.
inline Stereo own (const Stereo& y, const Found& F, double s, const Stop& stop = {})
{
    if (! F.found || s == 1.0 || y.size() == 0) return y;
    double Sp[NG], Sn[NG];
    if (s < 1.0) { careful (F.S[0], Sp); careful (F.S[1], Sn); } else { std::copy (F.S[0], F.S[0] + NG, Sp); std::copy (F.S[1], F.S[1] + NG, Sn); }
    const Curve cp (Sp, F.vp[0]), cn (Sn, F.vp[1]); const double g = drive (std::clamp (s, 0.0, 2.0), std::max (cp.top, cn.top)), pk = peakOf (y, 0.0);
    Stereo o = y;
    for (Vec* ch : { &o.l, &o.r })
    {
        if (stop && stop()) return y;
        for (double& v : *ch)
        {
            const double x = v >= 0.0 ? cp.back (v) : -cn.back (-v), d = g * x;
            v = g <= 0.0 ? x : d >= 0.0 ? cp.fwd (d) : -cn.fwd (-d);
        }
    }
    const double k = pk / std::max (peakOf (o, 0.0), 1e-30); for (Vec* ch : { &o.l, &o.r }) for (double& v : *ch) v *= k;
    return o;
}
// Another sound through the found curve. s: 0 = none, 1 = its peak pushed as far into the curve as the first sound's
// was, 2 = as far again. Its peak stays where it was.
inline Stereo onto (const Stereo& b, const Found& F, double s, const Stop& stop = {})
{
    if (! F.found || s <= 0.0 || b.size() == 0) return b;
    const double pb = peakOf (b, 0.0); if (! (pb > 1e-12)) return b;
    const Curve cp (F.S[0], F.vp[0]), cn (F.S[1], F.vp[1]); const double k = drive (std::min (s, 2.0), std::max (cp.top, cn.top)) * std::max (cp.xp, cn.xp) / pb;
    Stereo o = b;
    for (Vec* ch : { &o.l, &o.r })
    {
        if (stop && stop()) return b;
        for (double& v : *ch) { const double d = k * v; v = d >= 0.0 ? cp.fwd (d) : -cn.fwd (-d); }
    }
    const double c = pb / std::max (peakOf (o, 0.0), 1e-30); for (Vec* ch : { &o.l, &o.r }) for (double& v : *ch) v *= c;
    return o;
}

} // namespace bend
} // namespace dr
