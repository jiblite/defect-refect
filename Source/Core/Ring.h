#pragma once
// The ring engine (RING MOD): a sound multiplied by a steady tone, the same on both sides:
//     wet = dry * B (rho + sin (w t + p)),   B = 1 / sqrt (rho^2 + 1/2)   (so the gain's mean power is 1)
//   find()    reads the tone off one sound: its pitch to a few hundred-thousandths of a hertz, its phase, and how much
//             of the plain sound is left beside it (rho = 0: none, the gain goes through zero twice a cycle; rho > 1:
//             the gain never reaches zero)
//   own()     the sound itself with the group's Amount at s: 0 = the tone out, 1 = untouched, 2 = the tone part twice
//             as strong; carrier: the tone's pitch against the found one
//   onto()    the found gain on another sound: 0 = none, 1 = as found, 2 = the tone part twice as strong
// How (three steps, each with less doubt and on more data than the one before):
//  1. Candidates. The sound's power over time (left^2 + right^2), divided by its own local mean, carries a line at w
//     that is rho / (rho^2 + 1/2) of its mean and one at 2w that is 1/4 / (rho^2 + 1/2) of it, whatever the sound:
//     for every rho up to 2 one of the two is 0.44 or more. The highest lines of its spectrum that reach 0.2 of the
//     mean are tried, each as the tone's own line and as the line at twice the tone (spectrumPeaks).
//  2. The first look. The power, divided by its own local mean (so that a loud moment counts no more than a quiet
//     one), is sorted by where in the tone's cycle it falls (64 steps) and where in the sound (8 stretches). The gain
//     that explains that table best is looked for: phase, a small correction of the pitch, rho. "Explains": the sound
//     is taken as noise whose level is the gain (a likelihood). The candidate that wins the most over "no gain at
//     all" goes on; if none wins enough, there is no ring modulation.
//  3. The same likelihood on the samples themselves, by Newton's method, with the floor under the gain (what the
//     sound holds where the gain is zero) fitted along with it: first held, coming down a step at a time, then free
//     (settle). Near the gain's zeros that pins the phase to a thousandth of a sample. A gain that never reaches zero
//     has no floor to measure: a fixed small one is used.
// The check: the power sorted by the tone's cycle must have the gain's own shape (to 1 dB), the gain must dip by 10 dB
// or more (a shallower one is a tremolo), the sound divided by the gain must not carry a steady offset (a plain note at
// the tone's pitch is a "ring modulated" constant: not what is looked for), and the win must be a real one.
// Taken out: the sound divided by the gain. Next to the gain's zeros a sample over the gain is not simply the answer:
// there each sample is worked out from the sample and from the sound around it, each counted by what it is worth
// (takeOut).
// What it does not do: a tone that is on a part of the sound only, or whose pitch moves.
// A port of the Python prototype Research/ring/ring.py (Tools/ring_parity.py). 48 kHz; plain C++17, no JUCE.

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <utility>
#include <vector>

#include "Room.h"      // dr::Vec, dr::Stereo (Dsp.h) and room::Transform

namespace dr
{
namespace ring
{
using Stop = std::function<bool()>;                     // asked now and then while measuring: true = give up (the answer is then not used)
constexpr double FLO = 20.0, FHI = 5000.0;              // tones looked for, Hz
constexpr int NB = 64;                                  // steps of the tone's cycle
constexpr int MOST_HI = 6, MOST_LO = 4;                 // lines of the power's spectrum that are tried: over SPLIT Hz, under it
constexpr double THR = 0.2;                             // .. and how high a line must be for that, against the power's mean
constexpr double APART = 5.0;                           // Hz: a peak nearer than this to a higher one belongs to it
constexpr double SPLIT = 80.0;                          // Hz: lines over this are looked for in the power levelled over W_HI samples, lines under it over W_LO
constexpr long W_HI = 1024, W_LO = 8192;
constexpr double GATE = 1e-8;                           // a sample counts if the sound around it is within 80 dB of its loudest moment: the tone sits in a quiet tail as it does in a loud stroke
constexpr double GATE_LINES = 1e-3;                     // .. for the candidate lines: within 30 dB
constexpr int MS = 8;                                   // first look: stretches along the part of the sound that is looked at
constexpr long SPEC = 1L << 19;                         // the spectrum is taken over 10.9 s at most (the loudest)
constexpr long LONG = 20L * 48000L;                     // a longer sound: 20 stretches of one second, evenly spread, are measured
constexpr double HOLE = 0.01;                           // the samples where the gain is under this are not divided but filled
constexpr double EPS0 = 0.05;                           // first look: the floor under the gain
constexpr double EPS_NC = 1e-3;                         // the floor used when the gain never reaches zero
constexpr double TWO_PI = 2.0 * PI;
constexpr double RHOS[] = { 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.5, 1.8 };       // first look: rho tried
constexpr double VS[] = { -0.4, -0.2, 0.0, 0.2, 0.4 };  // first look: corrections of the pitch tried (radians at the two ends of the stretch looked at)

// The rule: what counts as found.
struct Rule { double match = 1.0, dip = 10.0, offset = -20.0, win = 0.3, firstWin = 0.2, holes = 0.1; };

struct Found
{
    bool found = false, looked = false, measured = false;   // looked: a candidate had a first look; measured: it went on to the exact fit
    size_t n = 0;                                           // the sound's length
    double firstWin = 0.0, firstF = 0.0, firstRho = 0.0, firstV = 0.0;
    double f = 0.0, w = 0.0, phase = 0.0, rho = 0.0, eps = 0.0;      // the tone: Hz, radians a sample, phase at sample 0; rho; the floor
    double win = 0.0, ideal = 0.0, match = 0.0, dip = 0.0, offset = 0.0, holes = 0.0; int passes = 0;
    std::vector<float> xl, xr;                              // found: the sound without the gain
};

// ---------------------------------------------------------------- small pieces
inline Vec power (const Stereo& y) { Vec p (y.size()); for (size_t i = 0; i < p.size(); ++i) p[i] = y.l[i] * y.l[i] + y.r[i] * y.r[i]; return p; }
// The stretch of `most` samples with the most power (looked for in steps of half a second): start, length.
inline std::pair<long, long> loudest (const Vec& p, long most, long hop = 24000)
{
    const long n = (long) p.size(); if (n <= most) return { 0, n };
    const long nb = n / hop, k = std::max (1L, most / hop); Vec c ((size_t) nb + 1, 0.0);
    for (long b = 0; b < nb; ++b) { double e = 0.0; for (long i = b * hop; i < (b + 1) * hop; ++i) e += p[(size_t) i]; c[(size_t) b + 1] = c[(size_t) b] + e; }
    long best = 0; double top = -1.0;
    for (long i = 0; i + k <= nb; ++i) { const double e = c[(size_t) (i + k)] - c[(size_t) i]; if (e > top) { top = e; best = i; } }
    return { std::min (best * hop, n - most), most };
}
// The samples that are measured: all of them, or in a long sound 20 stretches of one second, evenly spread (start,
// length): the tone's pitch is pinned by how far apart the stretches are, not by how many samples there are.
using Ranges = std::vector<std::pair<long, long>>;
inline Ranges samples (long n)
{
    if (n <= LONG) return { { 0L, n } };
    Ranges r; const long nc = n / 48000; for (long j = 0; j < 20; ++j) r.push_back ({ (j * nc) / 20 * 48000, 48000L });
    return r;
}
// The length over which the sound's own level is taken for a tone at f Hz: eight of its cycles, 21 ms at least (a power
// of two). For a slow tone (under 94 Hz) that would be 170 or 340 ms, longer than a drum stroke: there the length is a
// whole number of the tone's cycles instead (each of the two smoothings then leaves the tone's own cycle exactly as it
// is), 85 ms at least.
inline long window (double f)
{
    const double need = std::max (8.0 * SR / f, 960.0), P = SR / f; long W = 1; while ((double) W < need) W *= 2;
    return std::min (W, 2 * (long) std::floor (std::ceil (2048.0 / P) * P + 0.5));
}
// The power over its own local mean (a window of W samples, smoothed twice): what is left is the gain's cycle and chance,
// and a loud moment counts no more than a quiet one. ok: which samples count (the ones whose local mean is over gate
// times the highest one).
struct Level { Vec p; std::vector<char> ok; };
inline Level local (const Vec& pw, long W, double gate = GATE)
{
    const long n = (long) pw.size(), h = std::max (W / 2, 1L); Vec e = pw, c ((size_t) n + 1, 0.0);
    for (int pass = 0; pass < 2; ++pass)
    {
        for (long i = 0; i < n; ++i) c[(size_t) i + 1] = c[(size_t) i] + e[(size_t) i];
        for (long i = 0; i < n; ++i) { const long a = std::max (i - h / 2, 0L), b = std::min (i - h / 2 + h, n); e[(size_t) i] = (c[(size_t) b] - c[(size_t) a]) / (double) (b - a); }
    }
    double top = 0.0; for (double v : e) top = std::max (top, v);
    Level L; L.p.assign ((size_t) n, 0.0); L.ok.assign ((size_t) n, 0);
    for (long i = 0; i < n; ++i) if (e[(size_t) i] > gate * top) { L.ok[(size_t) i] = 1; L.p[(size_t) i] = pw[(size_t) i] / std::max (e[(size_t) i], 1e-300); }
    return L;
}
// The same, kept for the samples a .. a + len only.
inline Level localPart (const Vec& pw, long W, long a, long len, double gate = GATE)
{
    const Level all = local (pw, W, gate); Level L;
    L.p.assign (all.p.begin() + a, all.p.begin() + a + len); L.ok.assign (all.ok.begin() + a, all.ok.begin() + a + len); return L;
}
// Candidate lines between flo and fhi Hz: the highest peaks in the spectrum of the power levelled over W samples (the
// loudest SPEC samples of it, and of those the ones within 30 dB of the loudest moment), measured against that power's
// mean. Their frequencies (Hz) are added to out, the highest first.
inline void lines (const Vec& pw, long W, double flo, double fhi, int most, std::vector<double>& out)
{
    const auto part = loudest (pw, SPEC); const long a0 = part.first, n = part.second; if (n < 4096) return;
    const Level L = localPart (pw, W, a0, n, GATE_LINES);
    double m = 0.0; long cnt = 0; for (long i = 0; i < n; ++i) if (L.ok[(size_t) i]) { m += L.p[(size_t) i]; ++cnt; }
    if (cnt == 0) return;
    m /= (double) cnt; long nfft = 1; while (nfft < 2 * n) nfft *= 2;
    Vec re ((size_t) nfft, 0.0), im ((size_t) nfft, 0.0); double top = 0.0;
    for (long i = 0; i < n; ++i) if (L.ok[(size_t) i]) { const double w = 0.5 - 0.5 * std::cos (TWO_PI * (double) i / (double) n); re[(size_t) i] = (L.p[(size_t) i] - m) * w; top += w; }
    { const room::Transform T ((int) nfft); T.run (re, im); }
    const long H = nfft / 2 + 1; Vec P ((size_t) H); for (long k = 0; k < H; ++k) P[(size_t) k] = std::hypot (re[(size_t) k], im[(size_t) k]);
    re = Vec(); im = Vec();
    const long lo = std::max ((long) (flo / SR * (double) nfft), 2L), hi = std::min ((long) (fhi / SR * (double) nfft), H - 2);
    if (hi - lo < 3) return;
    const double floor = THR * m * top, apart = APART / SR * (double) nfft;
    struct Cand { long i; double high; }; std::vector<Cand> cand;        // i: counted from lo
    for (long j = 1; j + 1 < hi - lo; ++j)
    {
        const long i = lo + j;
        if (P[(size_t) i] > P[(size_t) i - 1] && P[(size_t) i] >= P[(size_t) i + 1] && P[(size_t) i] >= floor) cand.push_back ({ j, P[(size_t) i] });
    }
    std::stable_sort (cand.begin(), cand.end(), [] (const Cand& a, const Cand& b) { return a.high > b.high; });
    std::vector<long> idx;
    for (const Cand& q : cand)
    {
        bool far = true; for (long u : idx) far = far && (double) std::abs (q.i - u) > apart;
        if (far) idx.push_back (q.i);
        if ((int) idx.size() >= most) break;
    }
    for (long i0 : idx)
    {
        const double y0 = P[(size_t) (lo + i0 - 1)], y1 = P[(size_t) (lo + i0)], y2 = P[(size_t) (lo + i0 + 1)], den = y0 - 2.0 * y1 + y2, d = den < 0.0 ? 0.5 * (y0 - y2) / den : 0.0;
        out.push_back (((double) (lo + i0) + d) * SR / (double) nfft);
    }
}
// Candidate lines in the power's spectrum. A gain B (rho + sin) puts a line at w that is rho / (rho^2 + 1/2) of the
// power's mean and one at 2w that is 1/4 / (rho^2 + 1/2) of it, whatever the sound: for every rho up to 2 one of the two
// is 0.44 or more. So a line under THR of the mean is not tried, and of the others the highest are.
// The power is first divided by its own local mean. What the sound's loudness does in time (its beat, the shape of each
// stroke) sits around zero in the power's spectrum and, as a copy, around each of the tone's lines: on a drum loop or a
// row of short strokes that hides the lines among their neighbours and fills the low end with higher peaks than
// theirs. Divided by the local mean (21 ms) the loudness is gone and the lines stand alone. Lines under SPLIT Hz would
// go with it: they are looked for apart, with a local mean of 171 ms.
inline std::vector<double> spectrumPeaks (const Vec& pw)
{
    std::vector<double> out;
    lines (pw, W_HI, SPLIT, std::min (2.0 * FHI, 0.45 * SR), MOST_HI, out); lines (pw, W_LO, FLO, SPLIT, MOST_LO, out);
    return out;
}
// The samples of the ranges that count: their levelled power, where they are, and the cost of "no gain at all".
// from: the sample that L's first entry belongs to.
struct Taken { Vec p, t; double flat = 0.0; };
inline Taken taken (const Level& L, const Ranges& rs, long from = 0)
{
    Taken T; double sum = 0.0; size_t most = 0; for (const auto& r : rs) most += (size_t) r.second;
    T.p.reserve (most); T.t.reserve (most);
    for (const auto& r : rs) for (long i = r.first; i < r.first + r.second; ++i) if (L.ok[(size_t) (i - from)]) { const double v = L.p[(size_t) (i - from)]; T.p.push_back (v); T.t.push_back ((double) i); sum += v; }
    T.flat = T.p.empty() ? 0.0 : std::log (sum / (double) T.p.size()); return T;
}
inline double turn (double x) { double th = std::fmod (x, TWO_PI); if (th < 0.0) th += TWO_PI; return th; }      // into 0 .. 2 pi

// ---------------------------------------------------------------- the first look: the power sorted by the tone's cycle
// The table: per stretch (M of them along the samples a0 .. a0 + n0) and step of the tone's cycle (NB) the power's sum
// and the number of samples. The cycle is counted from the middle of a0 .. a0 + n0.
struct Table { int M = MS; Vec P, C; };
inline Table fold (const Taken& T, long a0, long n0, double w, int M = MS)
{
    Table F; F.M = M; F.P.assign ((size_t) (M * NB), 0.0); F.C.assign ((size_t) (M * NB), 0.0); const double mid = (double) a0 + 0.5 * (double) n0;
    for (size_t i = 0; i < T.p.size(); ++i)
    {
        const int k = std::min ((int) (turn (w * (T.t[i] - mid)) * (NB / TWO_PI)), NB - 1); const long at = ((long) T.t[i] - a0) * M / n0 * NB + k;
        F.P[(size_t) at] += T.p[i]; F.C[(size_t) at] += 1.0;
    }
    return F;
}
// The gain's power (with the floor) averaged over each cell of the table, for the gain B (rho + sin (theta + phi + v tau)),
// tau = -1 .. 1 along the sound.
inline void cellGain (double phi, double v, double rho, double eps, int M, Vec& G)
{
    G.resize ((size_t) (M * NB)); const double d = TWO_PI / NB, r2 = rho * rho + 0.5;
    for (int m = 0; m < M; ++m)
    {
        const double tm = (2.0 * m + 1.0) / M - 1.0;
        for (int k = 0; k < NB; ++k)
        {
            const double ta = (double) k * d + phi + v * tm, tb = ta + d;
            G[(size_t) (m * NB + k)] = (r2 + (2.0 * rho * (std::cos (ta) - std::cos (tb)) - 0.25 * (std::sin (2.0 * tb) - std::sin (2.0 * ta))) / d) / r2 + eps * eps;
        }
    }
}
// Minus the likelihood of the table as noise at the level of the gain (per sample, up to a constant).
inline double cellCost (const Table& F, const Vec& G)
{
    double N = 0.0, a = 0.0, b = 0.0;
    for (size_t i = 0; i < G.size(); ++i) { N += F.C[i]; a += F.C[i] * std::log (G[i]); b += F.P[i] / G[i]; }
    return a / N + std::log (b / N);
}
template <typename Fn> double golden (Fn f, double a, double b, int it = 20)
{
    const double gr = 0.6180339887498949; double c = b - gr * (b - a), d = a + gr * (b - a), fc = f (c), fd = f (d);
    for (int i = 0; i < it; ++i)
    {
        if (fc < fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = f (c); }
        else { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = f (d); }
    }
    return 0.5 * (a + b);
}
// The gain that explains the table best: phase (at the middle), v, rho and the cost. A grid over all three, then each
// made finer in turn, three times.
struct Look { double phi = 0.0, v = 0.0, rho = 0.0, J = 0.0; };
inline Look firstLook (const Table& F)
{
    const int M = F.M; const size_t cells = (size_t) (M * NB); double N = 0.0; for (double c : F.C) N += c;
    Vec G, L (cells), R (cells); bool any = false; double bestJ = 0.0; Look B;
    for (double v : VS) for (double rho : RHOS)
    {
        cellGain (0.0, v, rho, EPS0, M, G); for (size_t i = 0; i < cells; ++i) { L[i] = std::log (G[i]); R[i] = 1.0 / G[i]; }
        for (int j = 0; j < NB; ++j)                    // the phase moved by j steps: the same table of gains, read j steps further on
        {
            double a = 0.0, b = 0.0;
            for (int m = 0; m < M; ++m) for (int k = 0; k < NB; ++k) { const size_t at = (size_t) (m * NB + k), rl = (size_t) (m * NB + (k + j) % NB); a += F.C[at] * L[rl]; b += F.P[at] * R[rl]; }
            const double J = a / N + std::log (b / N);
            if (! any || J < bestJ) { any = true; bestJ = J; B.phi = (double) j * TWO_PI / NB; B.v = v; B.rho = rho; }
        }
    }
    const double d = TWO_PI / NB; auto cost = [&] (double phi, double v, double rho) { cellGain (phi, v, rho, EPS0, M, G); return cellCost (F, G); };
    for (int it = 0; it < 3; ++it)                      // rho may go under zero here: that is the same gain half a cycle on
    {
        B.phi = golden ([&] (double x) { return cost (x, B.v, B.rho); }, B.phi - d, B.phi + d);
        B.v = golden ([&] (double x) { return cost (B.phi, x, B.rho); }, B.v - 0.2, B.v + 0.2);
        B.rho = golden ([&] (double x) { return cost (B.phi, B.v, x); }, B.rho - 0.15, B.rho + 0.15);
    }
    B.J = cost (B.phi, B.v, B.rho);
    if (B.rho < 0.0) { B.rho = -B.rho; B.phi += PI; }
    B.phi = turn (B.phi); return B;
}

// ---------------------------------------------------------------- the likelihood on the samples
// One pass: the cost J at u = (phase at the middle, v, rho, log of the floor), its slope and its curvature.
// tau = t - mid, tp = tau / half.
// F: the curvature the cost would have on average if the sound were exactly what the model takes it for (how the
// slopes of log (g^2 + eps^2) vary from sample to sample); never negative in any direction.
struct Pass { double J = 0.0, g[4] = { 0, 0, 0, 0 }, H[4][4] = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } }, F[4][4] = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } }; };
inline Pass passes (const Taken& T, double mid, double half, double w, const double* u)
{
    const double phi = u[0], v = u[1], rho = u[2], eps2 = std::exp (2.0 * u[3]), B2 = 1.0 / (rho * rho + 0.5), B = std::sqrt (B2);
    double sLog = 0.0, sR = 0.0, sU[4] = { 0, 0, 0, 0 }, sRU[4] = { 0, 0, 0, 0 }, sQ[4][4] = {}, sUU[4][4] = {}, sRUU[4][4] = {}, sRQ[4][4] = {};
    const size_t N = T.p.size();
    for (size_t i = 0; i < N; ++i)
    {
        const double tau = T.t[i] - mid, tp = tau / half, th = w * tau + phi + v * tp, s = std::sin (th), c = std::cos (th);
        const double g = B * (rho + s), q = g * g + eps2, iq = 1.0 / q, r = T.p[i] * iq, Bc = B * c, Bs = B * s;
        const double g1[3] = { Bc, Bc * tp, B - rho * B2 * g };
        const double uu[4] = { 2.0 * g * g1[0] * iq, 2.0 * g * g1[1] * iq, 2.0 * g * g1[2] * iq, 2.0 * eps2 * iq };
        double g2[3][3];                               // the gain's second derivatives
        g2[0][0] = -Bs; g2[0][1] = -Bs * tp; g2[1][1] = -Bs * tp * tp; g2[0][2] = -rho * B2 * Bc; g2[1][2] = -rho * B2 * Bc * tp; g2[2][2] = -2.0 * rho * B2 * B - B2 * g + 3.0 * rho * rho * B2 * B2 * g;
        sLog += std::log (q); sR += r;
        for (int a = 0; a < 4; ++a)
        {
            sU[a] += uu[a]; sRU[a] += r * uu[a];
            for (int b = a; b < 4; ++b)
            {
                const double pr = uu[a] * uu[b]; sUU[a][b] += pr; sRUU[a][b] += r * pr;
                if (b < 3) { const double qab = (2.0 * g1[a] * g1[b] + 2.0 * g * g2[a][b]) * iq; sQ[a][b] += qab; sRQ[a][b] += r * qab; }
                else if (a == 3) { const double qab = 4.0 * eps2 * iq; sQ[a][b] += qab; sRQ[a][b] += r * qab; }
            }
        }
    }
    Pass P; const double n = (double) N, S = sR / n; P.J = sLog / n + std::log (S); double Si[4];
    for (int a = 0; a < 4; ++a) { Si[a] = -sRU[a] / n; P.g[a] = sU[a] / n + Si[a] / S; }
    for (int a = 0; a < 4; ++a) for (int b = a; b < 4; ++b)
    {
        const double Aab = sQ[a][b] / n - sUU[a][b] / n, Sab = 2.0 * sRUU[a][b] / n - sRQ[a][b] / n;
        P.H[a][b] = P.H[b][a] = Aab + Sab / S - Si[a] * Si[b] / (S * S);
        P.F[a][b] = P.F[b][a] = sUU[a][b] / n - (sU[a] / n) * (sU[b] / n);
    }
    return P;
}
// A x = b for a symmetric positive definite 4 x 4 matrix (Cholesky). False: it is not positive definite.
inline bool solve4 (const double A[4][4], const double* b, double* x)
{
    double L[4][4] = {};
    for (int i = 0; i < 4; ++i) for (int j = 0; j <= i; ++j)
    {
        double s = A[i][j]; for (int k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
        if (i == j) { if (! (s > 0.0) || ! std::isfinite (s)) return false; L[i][i] = std::sqrt (s); } else L[i][j] = s / L[j][j];
    }
    double z[4];
    for (int i = 0; i < 4; ++i) { double s = b[i]; for (int k = 0; k < i; ++k) s -= L[i][k] * z[k]; z[i] = s / L[i][i]; }
    for (int i = 3; i >= 0; --i) { double s = z[i]; for (int k = i + 1; k < 4; ++k) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
    return true;
}
// The cost made smallest from u on: Newton's steps, damped (a step is taken only if the cost comes down). Where the
// cost does not curve upwards in every direction (beside a zero of the gain that a sample sits on it curves downwards)
// the step is made with the average curvature F instead, which always points downhill.
// floorFree false: the floor stays where it is. stopped: given up on the way.
// slope: the cost's slope in the (log) floor at the answer, also when the floor was held.
struct Fit { double u[4] = { 0, 0, 0, 0 }, J = 0.0, slope = 0.0; int passes = 0; bool stopped = false; };
inline Fit newton (const Taken& T, double mid, double half, double w, const double* u0, bool floorFree, const Stop& stop, int most = 30)
{
    const double E_LO = std::log (1e-5), E_HI = std::log (0.3), clip[4] = { 0.3, 0.5, 0.3, 1.5 }, tiny[4] = { 1e-7, 1e-7, 1e-7, 1e-4 };
    Fit F; for (int i = 0; i < 4; ++i) F.u[i] = u0[i];
    double raw = 0.0;                                    // the slope in the floor of the pass last made
    auto at = [&] (const double* u)
    {
        Pass P = passes (T, mid, half, w, u); raw = P.g[3];
        if (! floorFree) { P.g[3] = 0.0; for (int i = 0; i < 4; ++i) { P.H[3][i] = 0.0; P.H[i][3] = 0.0; P.F[3][i] = 0.0; P.F[i][3] = 0.0; } P.H[3][3] = 1.0; P.F[3][3] = 1.0; }
        const double none[4] = { 0, 0, 0, 0 }; double x[4];
        if (! solve4 (P.H, none, x)) for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) P.H[i][j] = P.F[i][j];       // not a pit here: the average curvature
        return P;
    };
    double lam = 1e-2; Pass P = at (F.u); F.passes = 1; F.J = P.J; F.slope = raw;
    for (int it = 0; it < most; ++it)
    {
        if (stop && stop()) { F.stopped = true; return F; }
        double D[4], step[4]; bool have = false; for (int i = 0; i < 4; ++i) D[i] = std::max (std::abs (P.H[i][i]), 1e-12);
        for (int tries = 0; tries < 12 && ! have; ++tries)
        {
            double A[4][4], mg[4]; for (int i = 0; i < 4; ++i) { mg[i] = -P.g[i]; for (int j = 0; j < 4; ++j) A[i][j] = P.H[i][j] + (i == j ? lam * D[i] : 0.0); }
            if (solve4 (A, mg, step)) have = true; else lam *= 10.0;
        }
        if (! have) break;
        double un[4]; for (int i = 0; i < 4; ++i) un[i] = F.u[i] + std::clamp (step[i], -clip[i], clip[i]);
        un[3] = std::min (std::max (un[3], E_LO), E_HI);
        bool still = true; for (int i = 0; i < 4; ++i) still = still && std::abs (un[i] - F.u[i]) < tiny[i];
        if (still) break;                               // a step too small to matter: done
        const Pass Pn = at (un); ++F.passes;
        if (Pn.J < F.J)
        {
            const bool small = F.J - Pn.J < 1e-12;
            for (int i = 0; i < 4; ++i) F.u[i] = un[i];
            F.J = Pn.J; F.slope = raw; P = Pn; lam = std::max (lam * 0.3, 1e-9);
            if (small) break;
        }
        else { lam *= 10.0; if (lam > 1e12) break; }
    }
    if (F.u[2] < 0.0) { F.u[2] = -F.u[2]; F.u[0] += PI; }      // the same gain half a cycle on
    return F;
}
// The cost made smallest from u on, the floor coming down a step at a time and only then set free. Under a high floor
// the cost is smooth and its lowest point lies near the true one. Under a low floor every sample that sits next to a
// zero of the gain digs a narrow pit of its own, and where the tone's cycle meets the samples at the same places again
// and again (a round pitch) the pits add up to walls: a fit that lets the floor fall at once can end on the wrong side
// of one, a few thousandths of a radian from the tone. A step is left out when the sound wants no lower floor than the
// one before, or when the gain never reaches zero.
constexpr int LADDER = 8;                               // floors held on the way down: 0.05, 0.016, 0.005 .. 1.6e-5
inline Fit settle (const Taken& T, double mid, double half, double w, const double* u0, const Stop& stop)
{
    Fit F; for (int i = 0; i < 4; ++i) F.u[i] = u0[i];
    int count = 0;
    for (int k = 0; k < LADDER; ++k)
    {
        F.u[3] = std::log (0.05 * std::pow (10.0, -0.5 * (double) k)); F = newton (T, mid, half, w, F.u, false, stop); count += F.passes;
        if (F.stopped) { F.passes = count; return F; }
        if (F.u[2] > 1.0 || F.slope <= 0.0) break;
    }
    F = newton (T, mid, half, w, F.u, true, stop); F.passes += count; return F;
}
// The check: the power sorted by the tone's cycle (NB steps) against the gain's own shape. match: how far the two are
// apart (dB, root mean square over the steps, each step counted by how much of the gain's power it holds); dip: how deep
// the gain's dip is (dB).
inline void profile (const Taken& T, double w, double ph, double rho, double eps, double& match, double& dip)
{
    double cnt[NB] = {}, P[NB] = {}, G[NB] = {}; const double b = 1.0 / std::sqrt (rho * rho + 0.5);
    for (size_t i = 0; i < T.p.size(); ++i)
    {
        const double x = w * T.t[i] + ph, g = b * (rho + std::sin (x)); const int k = std::min ((int) (turn (x) * (NB / TWO_PI)), NB - 1);
        cnt[k] += 1.0; P[k] += T.p[i]; G[k] += g * g + eps * eps;
    }
    double gm = 0.0, pm = 0.0; for (int k = 0; k < NB; ++k) { const double c = std::max (cnt[k], 1.0); P[k] /= c; G[k] /= c; gm += G[k]; pm += P[k]; }
    gm /= NB; pm = std::max (pm / NB, 1e-30); double gs = 0.0, top = 0.0, low = 1e300;
    for (int k = 0; k < NB; ++k) { G[k] /= gm; P[k] /= pm; gs += G[k]; top = std::max (top, G[k]); low = std::min (low, G[k]); }
    double d[NB], md = 0.0; for (int k = 0; k < NB; ++k) { d[k] = 10.0 * std::log10 (std::max (P[k], 1e-9) / std::max (G[k], 1e-9)); md += G[k] / gs * d[k]; }
    double ss = 0.0; for (int k = 0; k < NB; ++k) { const double e = d[k] - md; ss += G[k] / gs * e * e; }
    match = std::sqrt (ss); dip = 10.0 * std::log10 (top / std::max (low, 1e-9));
}
// What a sound multiplied by exactly this gain wins over "no gain at all" (per sample), with no floor.
inline double idealWin (double rho) { return std::log (4.0 * rho * rho + 2.0) - (rho > 1.0 ? 2.0 * std::log (rho + std::sqrt (rho * rho - 1.0)) : 0.0); }
// The gain at sample i.
inline double gainOf (double w, double ph, double rho, double i) { const double b = 1.0 / std::sqrt (rho * rho + 0.5); return b * (rho + std::sin (w * i + ph)); }

// ---------------------------------------------------------------- taking it out
// A x = b for a symmetric positive definite matrix of half-width bw, given as its lower band (row i holds A[i][i - bw]
// .. A[i][i] at band[i * (bw + 1) ..]). Cholesky, in place; b comes back as x. False: it is not positive definite.
inline bool solveBand (Vec& band, Vec& b, size_t m, size_t bw)
{
    const size_t W = bw + 1; auto L = [&] (size_t i, size_t j) -> double& { return band[i * W + (j + bw - i)]; };
    for (size_t i = 0; i < m; ++i)
    {
        const size_t j0 = i > bw ? i - bw : 0;
        for (size_t j = j0; j <= i; ++j)
        {
            double sum = L (i, j); const size_t k0 = std::max (j0, j > bw ? j - bw : 0);
            for (size_t k = k0; k < j; ++k) sum -= L (i, k) * L (j, k);
            if (i == j) { if (! (sum > 0.0) || ! std::isfinite (sum)) return false; L (i, i) = std::sqrt (sum); } else L (i, j) = sum / L (j, j);
        }
    }
    for (size_t i = 0; i < m; ++i) { double sum = b[i]; for (size_t k = i > bw ? i - bw : 0; k < i; ++k) sum -= L (i, k) * b[k]; b[i] = sum / L (i, i); }
    for (size_t i = m; i-- > 0;) { double sum = b[i]; for (size_t k = i + 1; k < std::min (m, i + bw + 1); ++k) sum -= L (k, i) * b[k]; b[i] = sum / L (i, i); }
    return true;
}
// One channel of the sound without the gain, at the doubtful samples (idx, in rising order; doubt: the same as flags):
// the samples whose gain is so near zero that the sample over the gain is not simply the answer. There a sample says
// little (it is the sound times next to nothing, and the gain is known to the floor e only), but the sound's own course
// says where it has to go. Each doubtful sample gets the value that fits best with both: the sample over the gain,
// counted by (gain / floor)^2, and the linear prediction of the sound around it (order 24, from 256 samples on either
// side), counted by how well that sound can be predicted at all (its prediction error against its power, 1 % at
// least). Exactly at a zero that is the fill that makes the prediction error smallest; a few floors away it is the
// division. A sample that does not fit after that (the gain is off there by more than the floor) counts less in a
// second and a third round. Doubtful samples close together are worked out as one group (64 at most); every other
// doubtful sample in the stretch is unknown too, but gets its value from its own group, where it sits in the middle.
// out[k]: the value for the sample idx[k]. False: given up.
constexpr double EPS_MIN = 1e-4;                        // taking out: the gain is not taken to be known better than this
constexpr double DOUBT = 100.0, GMAX = 0.1;             // doubtful: the gain is under this many floors, and under GMAX
constexpr double KAPPA_MIN = 0.01;                      // the prediction error counts as 1 % of the sound's power at least
constexpr int ROUNDS = 3;                               // the weights are set anew twice from what the first answer leaves unexplained
constexpr double CZ = 3.0;                              // .. a sample that is off by this many floors counts half
inline bool takeOut (const Vec& y, const Vec& g, double e, const std::vector<char>& doubt, const std::vector<long>& idx, Vec& out, const Stop& stop, int order = 24, long around = 256)
{
    const long n = (long) y.size(); const double e2 = e * e; out.assign (idx.size(), 0.0); size_t i = 0, groups = 0;
    Vec v, r ((size_t) order + 1), c ((size_t) order + 1), old ((size_t) order + 1), cc ((size_t) (2 * order + 1)), base, band, rhs0, rhs, w0, wt; std::vector<long> mi;
    while (i < idx.size())
    {
        if ((++groups & 255) == 0 && stop && stop()) return false;
        size_t j = i; while (j + 1 < idx.size() && idx[j + 1] - idx[j] <= order && j - i < 63) ++j;       // nearer than the filter's length: one group (64 at most)
        const long a = std::max (idx[i] - around, 0L), b = std::min (idx[j] + around + 1, n), len = b - a;
        v.assign ((size_t) len, 0.0); mi.clear();
        for (long k = 0; k < len; ++k) { const double gg = g[(size_t) (a + k)]; v[(size_t) k] = y[(size_t) (a + k)] * (gg / (gg * gg + e2)); if (doubt[(size_t) (a + k)]) mi.push_back (k); }
        for (size_t k = i; k <= j; ++k) out[k] = v[(size_t) (idx[k] - a)];        // a silent stretch, or no answer: the divided values stay
        // the prediction filter from the divided sound as it is (under the floor it is next to nothing by itself)
        for (int k = 0; k <= order; ++k) { double sum = 0.0; for (long t = 0; t + k < len; ++t) sum += v[(size_t) t] * v[(size_t) (t + k)]; r[(size_t) k] = sum; }
        if (r[0] > 0.0)
        {
            const double P = r[0] / (double) len; r[0] *= 1.0 + 1e-6; std::fill (c.begin(), c.end(), 0.0); c[0] = 1.0; double err = r[0];
            for (int q = 1; q <= order; ++q)
            {
                double acc = r[(size_t) q]; for (int t = 1; t < q; ++t) acc += c[(size_t) t] * r[(size_t) (q - t)];
                const double kk = -acc / err; std::copy (c.begin(), c.begin() + q, old.begin());
                for (int t = 1; t < q; ++t) c[(size_t) t] = old[(size_t) t] + kk * old[(size_t) (q - t)];
                c[(size_t) q] = kk; err *= 1.0 - kk * kk; if (! (err > 0.0)) break;
            }
            const double kappa = err > 0.0 ? std::min (std::max (err / r[0], KAPPA_MIN), 1.0) : KAPPA_MIN;
            // |c * v|^2 + the sum of kappa (g / e)^2 (v - y / g)^2 made smallest over the doubtful samples
            for (int d = 0; d <= order; ++d) { double sum = 0.0; for (int t = 0; t + d <= order; ++t) sum += c[(size_t) t] * c[(size_t) (t + d)]; cc[(size_t) (order + d)] = sum; cc[(size_t) (order - d)] = sum; }
            const size_t m = mi.size(), bw = std::min ((size_t) order, m - 1), W = bw + 1; base.assign (m * W, 0.0); rhs0.assign (m, 0.0); w0.assign (m, 0.0); wt.assign (m, 0.0);
            for (size_t p = 0; p < m; ++p)
            {
                for (size_t q = p > bw ? p - bw : 0; q <= p; ++q) { const long dl = mi[p] - mi[q]; if (dl <= order) base[p * W + (q + bw - p)] = cc[(size_t) (order + dl)]; }
                double sum = 0.0; for (int d = -order; d <= order; ++d) { const long t = mi[p] - d; if (t >= 0 && t < len && ! doubt[(size_t) (a + t)]) sum += cc[(size_t) (order + d)] * v[(size_t) t]; }
                rhs0[p] = -sum; const double gg = g[(size_t) (a + mi[p])]; w0[p] = (gg / e) * (gg / e); wt[p] = kappa * w0[p];
            }
            const double ridge = 1e-9 * cc[(size_t) order]; bool solved = false;
            for (int it = 0; it < ROUNDS; ++it)
            {
                band = base; rhs.assign (m, 0.0);
                for (size_t p = 0; p < m; ++p)
                {
                    const double gg = g[(size_t) (a + mi[p])], yy = y[(size_t) (a + mi[p])];
                    band[p * W + bw] += wt[p] + ridge; rhs[p] = rhs0[p] + wt[p] * yy * gg / std::max (gg * gg, 1e-300);
                }
                solved = solveBand (band, rhs, m, bw); if (! solved) break;
                if (it + 1 < ROUNDS) for (size_t p = 0; p < m; ++p)
                {
                    const double gg = g[(size_t) (a + mi[p])], res = y[(size_t) (a + mi[p])] - gg * rhs[p];
                    wt[p] = kappa * w0[p] / (1.0 + res * res / (e2 * (P + rhs[p] * rhs[p])) / (CZ * CZ));
                }
            }
            if (solved) for (size_t k = i, p = 0; k <= j; ++k) { while (p < m && mi[p] < idx[k] - a) ++p; if (p < m && mi[p] == idx[k] - a) out[k] = rhs[p]; }
        }
        i = j + 1;
    }
    return true;
}

// ---------------------------------------------------------------- the verdict
inline Found find (const Stereo& y, const Stop& stop = {}, const Rule& R = {})
{
    Found F; F.n = y.size(); const long n = (long) y.size(); const Vec pw = power (y);
    double total = 0.0; for (double v : pw) total += v;
    if (n < 12000 || ! (total > 0.0)) return F;
    const auto part = loudest (pw, SPEC); const long a0 = part.first, n0 = part.second; const double mid = (double) a0 + 0.5 * (double) n0; const Ranges first { { a0, n0 } };
    std::map<long, Level> made;                         // the levelled power of the part looked at, per window
    bool any = false; double bestWin = 0.0, w = 0.0; Look best; long W = 0;
    for (double f0 : spectrumPeaks (pw)) for (int k = 1; k <= 2; ++k)           // the peak is the line at w, or the one at 2w
    {
        if (stop && stop()) return Found();
        const double f = f0 / k; if (! (f >= FLO && f <= FHI)) continue;
        const long Wf = window (f); auto it = made.find (Wf); if (it == made.end()) it = made.emplace (Wf, localPart (pw, Wf, a0, n0)).first;
        const Taken T = taken (it->second, first, a0); if ((long) T.p.size() < 6000) continue;
        const double wf = TWO_PI * f / SR; const Look L = firstLook (fold (T, a0, n0, wf));
        if (! any || T.flat - L.J > bestWin) { any = true; bestWin = T.flat - L.J; w = wf; best = L; W = Wf; }
    }
    made.clear();
    if (! any) return F;
    F.looked = true; F.firstWin = bestWin; F.firstF = w * SR / TWO_PI; F.firstRho = best.rho; F.firstV = best.v;
    if (bestWin < R.firstWin) return F;                 // the first look already says no
    const Level LV = local (pw, W); Taken T; double u[4] = { best.phi, best.v, best.rho, std::log (EPS0) }, win = 0.0; int count = 0; bool stopped = false;
    auto exact = [&] (const Ranges& rs, double half)      // the likelihood on these samples, from u on; the pitch correction is taken into w
    {
        T = taken (LV, rs); if (T.p.empty()) { stopped = true; return; }
        Fit f = settle (T, mid, half, w, u, stop); count += f.passes;
        if (! f.stopped && f.u[2] > 1.0)                // the gain never reaches zero: there is no floor to measure (it would only stand in for rho), so none is fitted
        {
            f.u[3] = std::log (EPS_NC); f = newton (T, mid, half, w, f.u, false, stop); count += f.passes;
        }
        if (f.stopped) { stopped = true; return; }
        w += f.u[1] / half; f.u[1] = 0.0; for (int i = 0; i < 4; ++i) u[i] = f.u[i];
        win = T.flat - f.J;
    };
    exact (first, 0.5 * (double) n0);
    if (! stopped && n > n0) exact (samples (n), 0.5 * (double) n);          // a longer sound: now all of it, from what its loudest part gave
    if (stopped) return Found();
    F.measured = true; F.w = w; F.f = w * SR / TWO_PI; F.phase = turn (u[0] - w * mid); F.rho = u[2]; F.eps = std::exp (u[3]); F.win = win; F.ideal = idealWin (F.rho); F.passes = count;
    profile (T, F.w, F.phase, F.rho, F.eps, F.match, F.dip);
    // the sound divided by the gain: its steady offset, and how many samples cannot be divided
    const double floor2 = F.eps * F.eps, edge = std::max (2.0 * F.eps, HOLE); Vec gain ((size_t) n); long holes = 0;
    double sl = 0.0, sr = 0.0, sll = 0.0, srr = 0.0;
    for (long i = 0; i < n; ++i)
    {
        const double g = gainOf (F.w, F.phase, F.rho, (double) i), q = g / (g * g + floor2), xl = y.l[(size_t) i] * q, xr = y.r[(size_t) i] * q;
        gain[(size_t) i] = g; sl += xl; sr += xr; sll += xl * xl; srr += xr * xr;
        if (std::abs (g) < edge) ++holes;
    }
    const double dn = (double) n, dc = ((sl / dn) * (sl / dn) + (sr / dn) * (sr / dn)) / std::max (sll / dn + srr / dn, 1e-30);
    F.offset = 10.0 * std::log10 (std::max (dc, 1e-12)); F.holes = (double) holes / dn;
    F.found = F.match <= R.match && F.dip >= R.dip && F.offset <= R.offset && F.win >= R.win && F.holes <= R.holes;
    if (! F.found) return F;
    // the sound without the gain: divided where the gain is well over the floor, worked out with the sound around it where it is not
    const double e = std::max (F.eps, EPS_MIN), e2 = e * e, lim = std::min (DOUBT * e, GMAX); std::vector<char> doubt ((size_t) n, 0); std::vector<long> idx;
    F.xl.resize ((size_t) n); F.xr.resize ((size_t) n);
    for (long i = 0; i < n; ++i)
    {
        const double g = gain[(size_t) i], q = g / (g * g + e2); F.xl[(size_t) i] = (float) (y.l[(size_t) i] * q); F.xr[(size_t) i] = (float) (y.r[(size_t) i] * q);
        if (std::abs (g) < lim) { doubt[(size_t) i] = 1; idx.push_back (i); }
    }
    if (! idx.empty())
    {
        Vec v;
        if (! takeOut (y.l, gain, e, doubt, idx, v, stop)) return Found();
        for (size_t k = 0; k < idx.size(); ++k) F.xl[(size_t) idx[k]] = (float) v[k];
        if (! takeOut (y.r, gain, e, doubt, idx, v, stop)) return Found();
        for (size_t k = 0; k < idx.size(); ++k) F.xr[(size_t) idx[k]] = (float) v[k];
    }
    return F;
}

// ---------------------------------------------------------------- the faders
// The gain with Amount at s: left of the middle a straight line from no gain (1) to the found one, right of it the
// tone part grows. carrier: the tone's pitch against the found one (1 = as found).
struct GainAt
{
    double base = 1.0, swing = 0.0, w = 0.0, ph = 0.0;
    GainAt (const Found& F, double s, double carrier = 1.0)
    {
        const double b = 1.0 / std::sqrt (F.rho * F.rho + 0.5), a = F.rho * b;
        base = s <= 1.0 ? (1.0 - s) + s * a : a; swing = s * b; w = carrier * F.w; ph = F.phase;
    }
    double operator() (double i) const { return base + swing * std::sin (w * i + ph); }
};
// The sound itself with the group's Amount at s: 0 = the tone out, 1 = untouched, 2 = the tone part twice as strong.
inline Stereo own (const Stereo& y, const Found& F, double s, double carrier = 1.0)
{
    if (! F.found || F.n != y.size() || F.xl.size() != y.size() || (std::abs (s - 1.0) < 1e-12 && std::abs (carrier - 1.0) < 1e-12)) return y;
    const GainAt now (F, s, carrier); Stereo out = y;
    for (size_t i = 0; i < y.size(); ++i)
    {
        const double d = now ((double) i) - gainOf (F.w, F.phase, F.rho, (double) i);
        out.l[i] += (double) F.xl[i] * d; out.r[i] += (double) F.xr[i] * d;
    }
    return out;
}
// The found gain on another sound: 0 = none, 1 = as found, 2 = the tone part twice as strong.
inline Stereo onto (const Stereo& z, const Found& F, double s, double carrier = 1.0)
{
    if (! F.found || s < 1e-12) return z;
    const GainAt now (F, s, carrier); Stereo out = z;
    for (size_t i = 0; i < z.size(); ++i) { const double g = now ((double) i); out.l[i] *= g; out.r[i] *= g; }
    return out;
}

} // namespace ring
} // namespace dr
