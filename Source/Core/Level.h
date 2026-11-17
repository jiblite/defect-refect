#pragma once
// The level-movement engine: tremolo, auto pan and gated rhythm are a gain per channel that moves over time and
// repeats. It is measured on ONE sound (the period, the shape over one cycle, the depth, how left and right differ)
// and used both ways:
//   find()      what was measured on a sound, and how sure it is
//   gainDb()    the found gain in dB over time, per channel
//   apply()     s = -1 takes it out of the sound it was found on, s = +1 puts it onto another sound (one that starts
//               where the analysed one started); in between and up to +2 scales it in dB
//   analyse()   a sound of any length: find() piece by piece, the clock that runs through the pieces, and where along
//               the sound the movement is (an effect on a part of the sound); ownDb() and ontoDb() are its gains for
//               the sound it was found on and for another sound
// How a gain put on the sound is told from the sound's own rhythm: a gain moves ALL bands by the same amount in dB at
// the same moment and repeats exactly; notes and hits do not. The picture is the power in 15 bands, 5.3 ms windows
// every millisecond.
//   together    both channels and all bands move as one (tremolo, gate). Every period on a fine grid is asked how much
//               of what the bands do together repeats with it (epoch folding of the level change over about a third of
//               the period). At the periods that stand out the shape over one cycle is solved from level differences
//               between phases (the median over bands and cycles: a note that starts or ends upsets only the few pairs
//               that straddle it), and it is measured how well every band and every cycle carries that shape.
//   against     left and right move against each other (auto pan): measured on the difference of their levels, where
//               the sound itself cancels. A pan seen less than twice is taken as the course that was measured.
// A small fitted rule (weights below, from Research/level) says how sure a movement is. A shape that jumps up and sinks
// back slowly is what struck and plucked notes do in every bar of music: never reported.
// A shape with steep steps (a gate, a sawtooth) is looked at again in steps of 0.125 ms: the period that makes the steps
// sharpest when all cycles are laid over each other, and each step's place and course.
// Taking it out lifts the sound by what the gain took. Safeguards: never more than 40 dB nor more than the bands agree
// the movement goes deep; a gate that closes fully (55 dB down) is left closed, what is under it is as likely noise.
// find() and what it uses are a port of the Python prototype Research/level/level.py (same numbers:
// Tools/level_parity.py); analyse() and below exist here only (tested through Tests/level.cpp by Research/level/long.py).
// 48 kHz; plain C++17.

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Dsp.h"

namespace dr
{
namespace level
{
using Stop = std::function<bool()>;
constexpr int W = 256, HOP = 48, NB = 15, NC = 2 * NB;                 // 5.3 ms windows, one every millisecond; 15 bands per channel
constexpr int EDG[NB + 1] = { 0, 2, 3, 4, 5, 7, 9, 12, 16, 21, 28, 37, 49, 64, 85, 108 };      // band edges in bins of 187.5 Hz (up to 20 kHz)
constexpr double FLOOR = 1e-10;                                         // a band counts as silent 100 dB under its loudest moment
constexpr double CLAMP = 30.0;                                          // for the agreement measures a level more than this under the top counts as this far under it
constexpr double PMIN = 60.0;                                           // movements faster than 16.7 Hz are not looked for: up there a sound's own roughness and beating take over
constexpr double CAP_DB = 40.0, CLOSED_HI = -40.0, CLOSED_LO = -55.0;   // the most that is lifted; where a gate counts as closing, and as closed
constexpr int GUARD = 48, DEC = 6;                                      // 1 ms either side of every moment; the fine look at an edge: power in steps of 6 samples
constexpr double P_DEFAULT = 1.0;                                       // the pan law when the sound does not say: the two gains add up to a constant
constexpr double SHARE_MIN = 0.12, CYC_SPREAD = 9.0, STRUCK = 0.55;
constexpr int TAUS[8] = { 4, 8, 16, 32, 64, 128, 256, 512 };
constexpr double PAN_ADJ_MIN = 0.2;                                      // a pan's repeating shape explains at least this much of the lane (over chance)
constexpr double SEEN_MIN = 1.25;                                       // a pan seen less often than this ...
constexpr double ONCE_CAP = -0.62;                                      // ... scores no higher than this (a sureness of one half)
constexpr double SCORE_MIN = -2.0;                                      // under this a movement is not offered at all (the default Sureness shows it from 0 up)
constexpr double sureNone = 0.03;                                       // the sureness of a movement that was not found
inline double nan() { return std::numeric_limits<double>::quiet_NaN(); }
inline bool fin (double v) { return std::isfinite (v); }
inline double sq (double v) { return v * v; }
inline int pmod (long a, int k) { return (int) (((a % k) + k) % k); }

// ---------------------------------------------------------------- numpy's ways of counting
// numpy.median (the values get reordered); nothing: NaN
inline double med (Vec& v)
{
    const size_t n = v.size(); if (n == 0) return nan();
    const size_t h = n / 2; std::nth_element (v.begin(), v.begin() + (long) h, v.end()); const double hi = v[h];
    if (n % 2) return hi;
    return 0.5 * (*std::max_element (v.begin(), v.begin() + (long) h) + hi);
}
inline double medc (Vec v) { return med (v); }
// numpy.percentile (linear); nothing: 0
inline double pct (Vec v, double q)
{
    const size_t n = v.size(); if (n == 0) return 0.0;
    std::sort (v.begin(), v.end());
    const double pos = (double) (n - 1) * (q / 100.0), fl = std::floor (pos), t = pos - fl; const size_t i = (size_t) fl, j = std::min (i + 1, n - 1);
    const double a = v[i], b = v[j], d = b - a; return t >= 0.5 ? b - d * (1.0 - t) : a + d * t;
}
inline double meanOf (const Vec& v) { double s = 0; for (double a : v) s += a; return v.empty() ? 0.0 : s / (double) v.size(); }
// numpy.linspace
inline Vec linspace (double a, double b, int n)
{
    Vec y ((size_t) n); if (n == 1) { y[0] = a; return y; }
    const double step = (b - a) / (double) (n - 1); for (int i = 0; i < n; ++i) y[(size_t) i] = (double) i * step + a;
    y[(size_t) n - 1] = b; return y;
}
// numpy.exp (numpy.arange (log lo, log hi, step))
inline Vec expGrid (double lo, double hi, double step)
{
    Vec out; if (! (lo > 0) || ! (hi > 0)) return out;
    const double a = std::log (lo), b = std::log (hi), len = std::ceil ((b - a) / step); if (! (len > 0)) return out;
    const double a1 = a + step, delta = a1 - a; out.resize ((size_t) len);
    for (size_t i = 0; i < out.size(); ++i) out[i] = std::exp (i == 0 ? a : (i == 1 ? a1 : a + (double) i * delta));
    return out;
}
// numpy.interp (xp rising)
inline double interp (double x, const Vec& xp, const Vec& fp)
{
    const size_t n = xp.size(); if (n == 0) return 0.0;
    if (x <= xp[0]) return fp[0];
    if (x >= xp[n - 1]) return fp[n - 1];
    const size_t j = (size_t) (std::upper_bound (xp.begin(), xp.end(), x) - xp.begin()) - 1;
    return (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]) * (x - xp[j]) + fp[j];
}
// The least squares solution of smallest size of A x = b for a symmetric A (numpy.linalg.lstsq with rcond = None):
// eigenvalues by Jacobi rotations; those not above n eps times the largest count as nothing.
inline Vec solveSym (std::vector<Vec> A, const Vec& b)
{
    const int n = (int) b.size(); std::vector<Vec> V ((size_t) n, Vec ((size_t) n, 0.0)); for (int i = 0; i < n; ++i) V[(size_t) i][(size_t) i] = 1.0;
    for (int sweep = 0; sweep < 80; ++sweep)
    {
        double off = 0, dia = 0;
        for (int p = 0; p < n; ++p) { dia += sq (A[(size_t) p][(size_t) p]); for (int q = p + 1; q < n; ++q) off += sq (A[(size_t) p][(size_t) q]); }
        if (off <= 1e-32 * dia || off == 0.0) break;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q)
            {
                const double apq = A[(size_t) p][(size_t) q]; if (std::abs (apq) < 1e-300) continue;
                const double th = (A[(size_t) q][(size_t) q] - A[(size_t) p][(size_t) p]) / (2.0 * apq), t = (th >= 0 ? 1.0 : -1.0) / (std::abs (th) + std::sqrt (th * th + 1.0));
                const double c = 1.0 / std::sqrt (t * t + 1.0), s = t * c;
                for (int k = 0; k < n; ++k) { const double x = A[(size_t) k][(size_t) p], y = A[(size_t) k][(size_t) q]; A[(size_t) k][(size_t) p] = c * x - s * y; A[(size_t) k][(size_t) q] = s * x + c * y; }
                for (int k = 0; k < n; ++k) { const double x = A[(size_t) p][(size_t) k], y = A[(size_t) q][(size_t) k]; A[(size_t) p][(size_t) k] = c * x - s * y; A[(size_t) q][(size_t) k] = s * x + c * y; }
                for (int k = 0; k < n; ++k) { const double x = V[(size_t) k][(size_t) p], y = V[(size_t) k][(size_t) q]; V[(size_t) k][(size_t) p] = c * x - s * y; V[(size_t) k][(size_t) q] = s * x + c * y; }
            }
    }
    double mx = 0; for (int i = 0; i < n; ++i) mx = std::max (mx, std::abs (A[(size_t) i][(size_t) i]));
    const double cut = mx * (double) n * std::numeric_limits<double>::epsilon(); Vec x ((size_t) n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        const double lam = A[(size_t) i][(size_t) i]; if (! (std::abs (lam) > cut)) continue;
        double d = 0; for (int k = 0; k < n; ++k) d += V[(size_t) k][(size_t) i] * b[(size_t) k];
        d /= lam; for (int k = 0; k < n; ++k) x[(size_t) k] += d * V[(size_t) k][(size_t) i];
    }
    return x;
}
// Least squares by Householder reflections: the x that makes |A x - b| smallest (A: rows x cols, row after row); rss = what is left, squared.
inline Vec lsq (Vec A, int rows, int cols, Vec b, double& rss)
{
    Vec x ((size_t) cols, 0.0); auto at = [&] (int i, int j) -> double& { return A[(size_t) i * (size_t) cols + (size_t) j]; };
    for (int j = 0; j < cols && j < rows; ++j)
    {
        double nrm = 0; for (int i = j; i < rows; ++i) nrm += sq (at (i, j));
        nrm = std::sqrt (nrm); if (nrm == 0.0) continue;
        const double alpha = at (j, j) > 0 ? -nrm : nrm, vj = at (j, j) - alpha; double vn = sq (vj); for (int i = j + 1; i < rows; ++i) vn += sq (at (i, j));
        if (vn > 0)
        {
            for (int c = j + 1; c < cols; ++c)
            {
                double dot = vj * at (j, c); for (int i = j + 1; i < rows; ++i) dot += at (i, j) * at (i, c);
                dot *= 2.0 / vn; at (j, c) -= dot * vj; for (int i = j + 1; i < rows; ++i) at (i, c) -= dot * at (i, j);
            }
            double dot = vj * b[(size_t) j]; for (int i = j + 1; i < rows; ++i) dot += at (i, j) * b[(size_t) i];
            dot *= 2.0 / vn; b[(size_t) j] -= dot * vj; for (int i = j + 1; i < rows; ++i) b[(size_t) i] -= dot * at (i, j);
        }
        at (j, j) = alpha;
    }
    for (int j = std::min (cols, rows) - 1; j >= 0; --j) { double s = b[(size_t) j]; for (int c = j + 1; c < cols; ++c) s -= at (j, c) * x[(size_t) c]; x[(size_t) j] = at (j, j) != 0.0 ? s / at (j, j) : 0.0; }
    rss = 0; for (int i = cols; i < rows; ++i) rss += sq (b[(size_t) i]);
    return x;
}

// ---------------------------------------------------------------- what is looked at
struct Fft256
{
    int rev[W]; double cs[W / 2], sn[W / 2];
    Fft256()
    {
        for (int i = 0; i < W; ++i) { int r = 0; for (int b = 0; b < 8; ++b) if (i & (1 << b)) r |= 1 << (7 - b); rev[i] = r; }
        for (int k = 0; k < W / 2; ++k) { cs[k] = std::cos (2.0 * PI * k / W); sn[k] = -std::sin (2.0 * PI * k / W); }
    }
    void run (double* re, double* im) const        // in place; the input has to stand in bit-reversed order
    {
        for (int len = 2; len <= W; len <<= 1)
        {
            const int half = len / 2, step = W / len;
            for (int i = 0; i < W; i += len)
                for (int k = 0; k < half; ++k)
                {
                    const double wr = cs[k * step], wi = sn[k * step]; const int a = i + k, b = a + half;
                    const double tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
                }
        }
    }
};
struct Levels { int T = 0; Vec e[2]; };        // power per band and frame: e[channel][frame * NB + band]
// Power per band in frames of W samples centred on every HOP-th sample.
inline void bandPower (const Vec& x, Vec& out, int& T)
{
    static const Fft256 F; static const std::array<double, W> win = [] { std::array<double, W> w {}; for (int i = 0; i < W; ++i) w[(size_t) i] = 0.5 - 0.5 * std::cos (2.0 * PI * i / W); return w; }();
    const long n = (long) x.size(); T = (int) (n / HOP) + 1; out.assign ((size_t) T * NB, 0.0);
    double re[W], im[W], cs[EDG[NB] + 1];
    for (int t = 0; t < T; ++t)
    {
        const long a = (long) t * HOP - W / 2;
        for (int i = 0; i < W; ++i) { const long p = a + i; re[F.rev[i]] = (p >= 0 && p < n) ? x[(size_t) p] * win[(size_t) i] : 0.0; im[F.rev[i]] = 0.0; }
        F.run (re, im);
        cs[0] = 0.0; for (int k = 0; k < EDG[NB]; ++k) cs[k + 1] = cs[k] + (re[k] * re[k] + im[k] * im[k]);
        for (int b = 0; b < NB; ++b) out[(size_t) t * NB + (size_t) b] = cs[EDG[b + 1]] - cs[EDG[b]];
    }
}
inline Levels levels (const Stereo& x) { Levels E; bandPower (x.l, E.e[0], E.T); bandPower (x.r, E.e[1], E.T); return E; }
// Mean over m frames centred on each frame (nothing outside), column by column.
inline Vec box (const Vec& v, int T, int cols, int m)
{
    if (m <= 1) return v;
    Vec out ((size_t) T * (size_t) cols), cs ((size_t) T + 1);
    for (int c = 0; c < cols; ++c)
    {
        cs[0] = 0.0; for (int t = 0; t < T; ++t) cs[(size_t) t + 1] = cs[(size_t) t] + v[(size_t) t * (size_t) cols + (size_t) c];
        for (int t = 0; t < T; ++t) { const int a = std::clamp (t - m / 2, 0, T), b = std::clamp (t - m / 2 + m, 0, T); out[(size_t) t * (size_t) cols + (size_t) c] = (cs[(size_t) b] - cs[(size_t) a]) / m; }
    }
    return out;
}
// What all bands do together between t and t + tau frames: the median over bands (both channels) of the level change
// in dB, one value per frame (NaN where fewer than three bands sound at either moment).
inline Vec diffLane (const Levels& E, int tau, int m)
{
    const int T = E.T; Vec out; if (T <= tau) return out;
    const Vec S0 = box (E.e[0], T, NB, m), S1 = box (E.e[1], T, NB, m);
    auto S = [&] (int t, int c) { return c < NB ? S0[(size_t) t * NB + (size_t) c] : S1[(size_t) t * NB + (size_t) (c - NB)]; };
    double top[NC], mx = -1e300; for (int c = 0; c < NC; ++c) { top[c] = -1e300; for (int t = 0; t < T; ++t) top[c] = std::max (top[c], S (t, c)); mx = std::max (mx, top[c]); }
    std::vector<int> cols; for (int c = 0; c < NC; ++c) if (top[c] > mx * 1e-7) cols.push_back (c);
    const int nc = (int) cols.size(); Vec L ((size_t) T * (size_t) nc); std::vector<char> on ((size_t) T * (size_t) nc);
    for (int i = 0; i < nc; ++i)
    {
        const double tp = top[cols[(size_t) i]], fl = tp * FLOOR, low = tp * FLOOR * 0.01 + 1e-300;
        for (int t = 0; t < T; ++t) { const double s = S (t, cols[(size_t) i]); on[(size_t) t * (size_t) nc + (size_t) i] = s > fl; L[(size_t) t * (size_t) nc + (size_t) i] = 10.0 * std::log10 (std::max (s, low)); }
    }
    out.assign ((size_t) (T - tau), nan()); Vec buf;
    for (int t = 0; t + tau < T; ++t)
    {
        buf.clear(); const size_t a = (size_t) t * (size_t) nc, b = (size_t) (t + tau) * (size_t) nc;
        for (int i = 0; i < nc; ++i) if (on[b + (size_t) i] || on[a + (size_t) i]) buf.push_back (L[b + (size_t) i] - L[a + (size_t) i]);
        if (buf.size() >= 3) out[(size_t) t] = med (buf);
    }
    return out;
}
// A lane made safe for sums: its median taken off, everything beyond k times its usual size pulled in, unknown set to
// 0. What is left says little more than 'up' or 'down': a steady small movement then counts as much as a loud hit.
struct Lane { Vec z; double pw = 0; };
inline Lane clipLane (const Vec& v, double k = 1.5)
{
    Lane o; o.z.assign (v.size(), 0.0); Vec f; for (double a : v) if (fin (a)) f.push_back (a);
    if (f.size() < 32) return o;
    Vec t = f; const double m = med (t); for (size_t i = 0; i < f.size(); ++i) t[i] = std::abs (f[i] - m);
    const double c = med (t) * k + 1e-9; double s = 0;
    for (size_t i = 0; i < v.size(); ++i) if (fin (v[i])) { o.z[i] = std::clamp (v[i] - m, -c, c); s += o.z[i]; }
    const double mu = s / (double) f.size();
    for (size_t i = 0; i < v.size(); ++i) if (fin (v[i])) { o.z[i] -= mu; o.pw += o.z[i] * o.z[i]; }
    return o;
}
// Epoch folding: the share of a lane's power that repeats with period P (frames), seen in K phase bins.
inline double foldStat (const Vec& z, double P, int K)
{
    double s[64] = {}, c[64] = {}; const size_t n = z.size(); const double pm1 = P - 1.0; double r = 0.0;       // r = i mod P, exactly
    for (size_t i = 0; i < n; ++i)
    {
        int k = (int) std::floor (r / P * K); if (k >= K) k = K - 1;
        s[k] += z[i]; c[k] += 1.0;
        if (r >= pm1) r -= pm1; else r += 1.0;
    }
    double out = 0; for (int k = 0; k < K; ++k) out += s[k] * s[k] / std::max (c[k], 1.0);
    return out;
}
inline int tauFor (double P) { const double lim = std::max (P / 2.5, (double) TAUS[0]); int t = TAUS[0]; for (int v : TAUS) if (v <= lim) t = v; return t; }
// For every period on a fine grid (frames): how much of what all bands do together repeats with it.
struct Scan { Vec Ps, S; std::map<int, Lane> lane; };
inline Scan scanTogether (const Levels& E, double pmin, const Stop& stop)
{
    Scan sc; sc.Ps = expGrid (pmin, E.T / 3.0, 0.002); sc.S.assign (sc.Ps.size(), 0.0);
    for (size_t i = 0; i < sc.Ps.size(); ++i)
    {
        if ((i & 63) == 0 && stop && stop()) return sc;
        const int tau = tauFor (sc.Ps[i]); auto it = sc.lane.find (tau);
        if (it == sc.lane.end()) it = sc.lane.emplace (tau, clipLane (diffLane (E, tau, std::max (2, tau / 4)))).first;
        if (it->second.pw > 0) sc.S[i] = foldStat (it->second.z, sc.Ps[i], 16) / it->second.pw;
    }
    return sc;
}
// Periods where the scan stands out of its surroundings (within 0.2 either side in log period), largest excess first.
struct Peak { double P, share, excess; };
inline std::vector<Peak> peaksOf (const Vec& Ps, const Vec& S, int most)
{
    std::vector<Peak> out; const long n = (long) S.size(); if (n < 2) return out;
    const long w = 101, a = 15;                                         // 0.2 and 0.03 in log period on the grid of 0.002
    Vec base ((size_t) n), win ((size_t) (2 * w + 1)), ex ((size_t) n);
    for (long i = 0; i < n; ++i)
    {
        for (long k = -w; k <= w; ++k) win[(size_t) (k + w)] = S[(size_t) std::clamp (i + k, 0L, n - 1)];
        std::nth_element (win.begin(), win.begin() + w, win.end()); base[(size_t) i] = win[(size_t) w]; ex[(size_t) i] = S[(size_t) i] - base[(size_t) i];
    }
    Vec e = ex;
    for (int r = 0; r < most; ++r)
    {
        long k = 0; for (long i = 1; i < n; ++i) if (e[(size_t) i] > e[(size_t) k]) k = i;
        if (e[(size_t) k] <= 0.01) break;
        out.push_back ({ Ps[(size_t) k], S[(size_t) k], ex[(size_t) k] });
        for (long j = std::max (0L, k - a); j <= std::min (n - 1, k + a); ++j) e[(size_t) j] = 0.0;
    }
    return out;
}
// The period near P (frames) that makes a lane repeat best (a fine scan, then the top of a parabola through the three best points).
inline double refinePeriod (const Vec& z, double P, int K, double span, int steps)
{
    const Vec y = linspace (-span, span, steps); Vec Ps ((size_t) steps), s ((size_t) steps); int k = 0;
    for (int i = 0; i < steps; ++i) { Ps[(size_t) i] = P * (1.0 + y[(size_t) i]); s[(size_t) i] = foldStat (z, Ps[(size_t) i], K); if (s[(size_t) i] > s[(size_t) k]) k = i; }
    if (k > 0 && k < steps - 1)
    {
        const double den = s[(size_t) k - 1] - 2.0 * s[(size_t) k] + s[(size_t) k + 1], off = den < 0 ? 0.5 * (s[(size_t) k - 1] - s[(size_t) k + 1]) / den : 0.0;
        return Ps[(size_t) k] + off * (Ps[1] - Ps[0]);
    }
    return Ps[(size_t) k];
}

// A table with holes (NaN): rows x cols.
struct Grid { int rows = 0, cols = 0; Vec v; double at (int r, int c) const { return v[(size_t) r * (size_t) cols + (size_t) c]; } double& at (int r, int c) { return v[(size_t) r * (size_t) cols + (size_t) c]; } };
// Band levels gathered into cells of a K-th of the period Pf (frames): dB of the mean power per cell, for every band
// of both channels: (cells, 2 NB), cells = whole cycles * K. Below the band's floor: NaN.
inline Grid cellLevels (const Vec& e0, const Vec& e1, int T, double Pf, int K)
{
    Grid L; L.cols = NC; if (T <= 0) return L;
    auto cell = [&] (int t) { return (long) std::floor ((double) t / Pf * K); };
    const long n = (cell (T - 1) + 1) / K * K; if (n < 2 * K) return L;
    L.rows = (int) n; L.v.assign ((size_t) n * NC, 0.0); Vec cnt ((size_t) n, 0.0);
    for (int t = 0; t < T; ++t)
    {
        const long i = cell (t); if (i >= n) break;
        cnt[(size_t) i] += 1.0; double* row = &L.v[(size_t) i * NC]; const double* a = &e0[(size_t) t * NB]; const double* b = &e1[(size_t) t * NB];
        for (int k = 0; k < NB; ++k) { row[k] += a[k]; row[NB + k] += b[k]; }
    }
    double top[NC], mx = -1e300; for (int c = 0; c < NC; ++c) top[c] = -1e300;
    for (long i = 0; i < n; ++i) { const double d = std::max (cnt[(size_t) i], 1.0); for (int c = 0; c < NC; ++c) { double& v = L.v[(size_t) i * NC + (size_t) c]; v /= d; top[c] = std::max (top[c], v); } }
    for (int c = 0; c < NC; ++c) mx = std::max (mx, top[c]);
    for (long i = 0; i < n; ++i) for (int c = 0; c < NC; ++c) { double& v = L.v[(size_t) i * NC + (size_t) c]; v = (v <= top[c] * FLOOR || top[c] <= mx * 1e-7) ? nan() : 10.0 * std::log10 (std::max (v, 1e-300)); }
    return L;
}
// Each column's lowest level (0 where it never sounds).
inline Vec lowOf (const Grid& L)
{
    Vec low ((size_t) L.cols, std::numeric_limits<double>::infinity());
    for (int i = 0; i < L.rows; ++i) for (int c = 0; c < L.cols; ++c) { const double v = L.at (i, c); if (! std::isnan (v)) low[(size_t) c] = std::min (low[(size_t) c], v); }
    for (auto& v : low) if (! fin (v)) v = 0.0;
    return low;
}
// The shape over one cycle that all bands share, from level differences: for several lags j, the median over bands and
// cycles of level(k + j) - level(k) says how the gain differs between two phases; the shape is the least squares
// solution of all those medians. r2: the share of the differences' spread it explains.
struct Fold { Vec g; double r2 = 0; long npairs = 0; };
inline Fold pairsFold (const Grid& L, int K, const std::vector<char>* use = nullptr)
{
    Fold F; F.g.assign ((size_t) K, 0.0); const int n = L.rows, nc = L.cols; if (n < 2 * K) return F;
    const Vec low = lowOf (L); auto Lf = [&] (int i, int c) { const double v = L.at (i, c); return std::isnan (v) ? low[(size_t) c] - 20.0 : v; };
    std::vector<Vec> A ((size_t) K, Vec ((size_t) K, 0.0)); Vec rhs ((size_t) K, 0.0), buf, medv ((size_t) K), madv ((size_t) K), cnt ((size_t) K);
    struct Kept { int j = 0, m = 0; Vec d; }; std::vector<Kept> keep;
    for (int j : { 1, 2, 3, 4, 6, 8, 12, 16, 24, 32 })
    {
        if (j > K / 2) continue;
        const int m = (n - j) / K * K; if (m < K) continue;
        Kept kp; kp.j = j; kp.m = m; kp.d.assign ((size_t) m * (size_t) nc, nan());
        for (int i = 0; i < m; ++i)
        {
            if (use != nullptr && ! ((*use)[(size_t) (i + j)] && (*use)[(size_t) i])) continue;
            for (int c = 0; c < nc; ++c) { if (std::isnan (L.at (i + j, c)) && std::isnan (L.at (i, c))) continue; kp.d[(size_t) i * (size_t) nc + (size_t) c] = Lf (i + j, c) - Lf (i, c); }
        }
        bool any = false, all = true;
        for (int k = 0; k < K; ++k)
        {
            buf.clear(); for (int i = k; i < m; i += K) for (int c = 0; c < nc; ++c) { const double v = kp.d[(size_t) i * (size_t) nc + (size_t) c]; if (fin (v)) buf.push_back (v); }
            cnt[(size_t) k] = (double) buf.size(); medv[(size_t) k] = 0.0; madv[(size_t) k] = 1.0;
            if (buf.size() >= 3) { any = true; const double md = med (buf); for (auto& v : buf) v = std::abs (v - md); medv[(size_t) k] = md; madv[(size_t) k] = med (buf) * 1.4826; } else all = false;
        }
        if (! any) continue;
        if (all) { const double mu = meanOf (medv); for (auto& v : medv) v -= mu; }      // a shape that repeats has differences that add up to nothing round the cycle: what is left over is the sound's own drift
        for (int k = 0; k < K; ++k)
        {
            const double w = cnt[(size_t) k] >= 3 ? cnt[(size_t) k] / sq (std::max (madv[(size_t) k], 0.3)) : 0.0; const int b = (k + j) % K;
            A[(size_t) k][(size_t) k] += w; A[(size_t) b][(size_t) b] += w; A[(size_t) k][(size_t) b] -= w; A[(size_t) b][(size_t) k] -= w; rhs[(size_t) b] += w * medv[(size_t) k]; rhs[(size_t) k] -= w * medv[(size_t) k];
        }
        keep.push_back (std::move (kp));
    }
    if (keep.empty()) return F;
    double tr = 0; for (int k = 0; k < K; ++k) tr += A[(size_t) k][(size_t) k];
    const double reg = 1e-9 * tr / K; for (auto& row : A) for (auto& v : row) v += reg;
    Vec g = solveSym (A, rhs); const double mu = meanOf (g); for (auto& v : g) v -= mu;
    double num = 0, den = 0; long np = 0;
    for (auto& kp : keep)                               // how much of the differences' spread does the shape explain (absolute values: robust)
    {
        buf.clear(); for (double v : kp.d) if (fin (v)) buf.push_back (v);
        const double c = med (buf);
        for (int i = 0; i < kp.m; ++i)
        {
            const int k = i % K, b = (k + kp.j) % K; const double pred = g[(size_t) b] - g[(size_t) k];
            for (int cc = 0; cc < nc; ++cc) { const double v = kp.d[(size_t) i * (size_t) nc + (size_t) cc]; if (! fin (v)) continue; num += std::abs (std::clamp (v - pred, -40.0, 40.0)); den += std::abs (std::clamp (v - c, -40.0, 40.0)); ++np; }
        }
    }
    F.g = g; F.r2 = 1.0 - num / std::max (den, 1e-9); F.npairs = np; return F;
}

// ---------------------------------------------------------------- a cycle shape in numbers
inline double topOf (const Vec& g) { Vec s = g; std::sort (s.begin(), s.end()); const size_t k = std::max<size_t> (1, g.size() / 8); double t = 0; for (size_t i = s.size() - k; i < s.size(); ++i) t += s[i]; return t / (double) k; }
inline double lowEnd (const Vec& g, size_t k) { Vec s = g; std::sort (s.begin(), s.end()); double t = 0; for (size_t i = 0; i < k; ++i) t += s[i]; return t / (double) k; }
inline double depthOf (const Vec& g)
{
    Vec s = g; std::sort (s.begin(), s.end()); const size_t k = std::max<size_t> (1, g.size() / 16); double a = 0, b = 0;
    for (size_t i = 0; i < k; ++i) { a += s[i]; b += s[s.size() - k + i]; }
    return b / (double) k - a / (double) k;
}
// up = share of the cycle spent above the middle between top and bottom, steep = the largest step between neighbouring
// bins as a share of the depth, fund = share of the shape's variance in its first harmonic, rise = share of the cycle
// it takes from the bottom to the top.
struct Shape { double up = 0, steep = 0, fund = 0, rise = 0; };
inline Shape shapeNumbers (const Vec& g)
{
    Shape s; const int K = (int) g.size(); if (K == 0) return s;
    int imin = 0, imax = 0; for (int i = 1; i < K; ++i) { if (g[(size_t) i] < g[(size_t) imin]) imin = i; if (g[(size_t) i] > g[(size_t) imax]) imax = i; }
    const double lo = g[(size_t) imin], hi = g[(size_t) imax], dep = std::max (hi - lo, 1e-9), mid = 0.5 * (lo + hi), mu = meanOf (g);
    double up = 0, st = 0; for (int i = 0; i < K; ++i) { if (g[(size_t) i] > mid) up += 1; st = std::max (st, std::abs (g[(size_t) ((i + 1) % K)] - g[(size_t) i])); }
    s.up = up / K; s.steep = st / dep; double f1 = 0, all = 0;
    for (int h = 1; h <= K / 2; ++h)
    {
        double re = 0, im = 0; for (int i = 0; i < K; ++i) { const double a = 2.0 * PI * (double) ((long) h * i % K) / K; re += (g[(size_t) i] - mu) * std::cos (a); im -= (g[(size_t) i] - mu) * std::sin (a); }
        const double p = re * re + im * im; if (h == 1) f1 = p; all += p;
    }
    s.fund = f1 / std::max (all, 1e-12); s.rise = (double) pmod (imax - imin, K) / K; return s;
}
// How a shape gets from one level to the other: the share of the cycle spent falling through the middle (between a
// fifth of the depth under the top and a fifth over the bottom), the same rising, and the share within 3 dB of the top.
struct Struck { double fall = 0, climb = 0, flat = 0; };
inline Struck struckNumbers (const Vec& g)
{
    Struck s; const int K = (int) g.size(); if (K == 0) return s;
    const size_t k = std::max<size_t> (1, (size_t) K / 16); Vec srt = g; std::sort (srt.begin(), srt.end()); double lo = 0, hi = 0;
    for (size_t i = 0; i < k; ++i) { lo += srt[i]; hi += srt[srt.size() - k + i]; }
    lo /= (double) k; hi /= (double) k; const double dep = std::max (hi - lo, 1e-9);
    for (int i = 0; i < K; ++i)
    {
        const double v = g[(size_t) i], sl = g[(size_t) ((i + 1) % K)] - g[(size_t) pmod (i - 1, K)]; const bool mid = v > lo + 0.2 * dep && v < hi - 0.2 * dep;
        if (mid && sl < 0) s.fall += 1;
        if (mid && sl > 0) s.climb += 1;
        if (v >= hi - 3.0) s.flat += 1;
    }
    s.fall /= K; s.climb /= K; s.flat /= K; return s;
}

// ---------------------------------------------------------------- do all bands, and all cycles, carry the shape?
// Levels are taken against each band's top in the cycle and clamped CLAMP dB under it (so a gate that closes fully can
// be compared too).
//   slopes, corrs   per band (of both channels) its own shape over the cycle (median over cycles) set against g: size and likeness
//   cyc             per cycle what the bands share (median over bands) set against g: size (1 = as deep as g)
//   halves          likeness of the shared shape of the even and of the odd cycles
//   r2c             share of the shared level's movement that g explains
//   silent          share of cells in which nothing sounds
//   dis             how far the bands' own shapes lie apart, phase by phase, against the depth
struct Agree
{
    bool ok = false; int bands = 0, cyc_n = 0;
    double slope_med = 0, slope_lo = 0, slope_hi = 0, slope_min = 0, slope_max = 0, corr_med = 0, corr_lo = 0, corr_min = 0, cyc_med = 0, cyc_lo = 0, cyc_hi = 0, halves = 0, r2c = 0, silent = 0, dis_med = 1, dis_hi = 1;
};
inline Agree agreement (const Grid& L, int K, const Vec& g, int start, const std::vector<int>& skip)
{
    Agree a; const int nc = L.cols, n = (L.rows - start) / K; if (n < 2) return a;
    std::vector<int> keep; for (int i = 0; i < n; ++i) if (std::find (skip.begin(), skip.end(), i) == skip.end()) keep.push_back (i);
    const int m = (int) keep.size(); if (m < 2) return a;
    Vec dev ((size_t) m * (size_t) K * (size_t) nc, nan()), buf; std::vector<char> snd ((size_t) m * (size_t) nc, 0); long silent = 0;
    auto D = [&] (int ci, int k, int c) -> double& { return dev[((size_t) ci * (size_t) K + (size_t) k) * (size_t) nc + (size_t) c]; };
    for (int ci = 0; ci < m; ++ci)
    {
        const int r0 = start + keep[(size_t) ci] * K;
        for (int c = 0; c < nc; ++c)
        {
            buf.clear(); for (int k = 0; k < K; ++k) { const double v = L.at (r0 + k, c); if (! std::isnan (v)) buf.push_back (v); }
            if (buf.empty()) continue;                                  // the band does not sound in this cycle
            snd[(size_t) ci * (size_t) nc + (size_t) c] = 1; const double top = pct (buf, 90.0);
            for (int k = 0; k < K; ++k) { const double v = L.at (r0 + k, c); D (ci, k, c) = std::isnan (v) ? -CLAMP : std::max (v - top, -CLAMP); }
        }
        for (int k = 0; k < K; ++k) { bool all = true; for (int c = 0; c < nc && all; ++c) if (! std::isnan (L.at (r0 + k, c))) all = false; if (all) ++silent; }
    }
    Vec own ((size_t) K * (size_t) nc, nan()), com ((size_t) m * (size_t) K, nan());
    for (int k = 0; k < K; ++k) for (int c = 0; c < nc; ++c) { buf.clear(); for (int ci = 0; ci < m; ++ci) { const double v = D (ci, k, c); if (! std::isnan (v)) buf.push_back (v); } if (! buf.empty()) own[(size_t) k * (size_t) nc + (size_t) c] = med (buf); }
    for (int ci = 0; ci < m; ++ci) for (int k = 0; k < K; ++k) { buf.clear(); for (int c = 0; c < nc; ++c) { const double v = D (ci, k, c); if (! std::isnan (v)) buf.push_back (v); } if (! buf.empty()) com[(size_t) ci * (size_t) K + (size_t) k] = med (buf); }
    Vec gl ((size_t) K); for (int i = 0; i < K; ++i) gl[(size_t) i] = g[(size_t) ((i + start) % K)];
    const double tp = topOf (gl); Vec gc ((size_t) K), gg ((size_t) K); for (int i = 0; i < K; ++i) gc[(size_t) i] = std::max (gl[(size_t) i] - tp, -CLAMP);
    const double gm = meanOf (gc); double den = 1e-12, gmin = gc[0]; for (int i = 0; i < K; ++i) { gg[(size_t) i] = gc[(size_t) i] - gm; gmin = std::min (gmin, gc[(size_t) i]); }
    { double s = 0; for (double v : gg) s += v * v; den = s + 1e-12; }
    Vec sl, co; std::vector<int> ub;
    for (int c = 0; c < nc; ++c)
    {
        bool okb = true; double mu = 0, sn = 0; for (int k = 0; k < K; ++k) { const double v = own[(size_t) k * (size_t) nc + (size_t) c]; if (! fin (v)) { okb = false; break; } mu += v; }
        for (int ci = 0; ci < m; ++ci) sn += snd[(size_t) ci * (size_t) nc + (size_t) c];
        if (! okb || sn / m < 0.3) continue;
        mu /= K; double dot = 0, vv = 0; for (int k = 0; k < K; ++k) { const double v = own[(size_t) k * (size_t) nc + (size_t) c] - mu; dot += v * gg[(size_t) k]; vv += v * v; }
        sl.push_back (dot / den); co.push_back (dot / std::sqrt (den * (vv + 1e-12))); ub.push_back (c);
    }
    if (ub.size() >= 3)
    {
        Vec dis ((size_t) K);
        for (int k = 0; k < K; ++k)
        {
            buf.clear(); for (int c : ub) buf.push_back (own[(size_t) k * (size_t) nc + (size_t) c]);
            Vec t = buf; const double md = med (t); for (auto& v : buf) v = std::abs (v - md); dis[(size_t) k] = med (buf);
        }
        const double depc = std::max (-gmin, 1e-6); a.dis_med = medc (dis) / depc; a.dis_hi = pct (dis, 90.0) / depc;
    }
    Vec sz;
    for (int ci = 0; ci < m; ++ci)
    {
        bool okc = true; double mu = 0; for (int k = 0; k < K; ++k) { const double v = com[(size_t) ci * (size_t) K + (size_t) k]; if (! fin (v)) { okc = false; break; } mu += v; }
        if (! okc) continue;
        mu /= K; double dot = 0; for (int k = 0; k < K; ++k) dot += (com[(size_t) ci * (size_t) K + (size_t) k] - mu) * gg[(size_t) k];
        sz.push_back (dot / den);
    }
    Vec e2, o2;
    for (int k = 0; k < K; ++k)
    {
        Vec be, bo; for (int ci = 0; ci < m; ++ci) { const double v = com[(size_t) ci * (size_t) K + (size_t) k]; if (! std::isnan (v)) (ci % 2 ? bo : be).push_back (v); }
        if (! be.empty() && ! bo.empty()) { e2.push_back (med (be)); o2.push_back (med (bo)); }
    }
    if ((int) e2.size() > K / 2)
    {
        const double me = meanOf (e2), mo = meanOf (o2); double see = 0, soo = 0, seo = 0;
        for (size_t i = 0; i < e2.size(); ++i) { see += sq (e2[i] - me); soo += sq (o2[i] - mo); seo += (e2[i] - me) * (o2[i] - mo); }
        if (std::sqrt (see / (double) e2.size()) > 1e-9 && std::sqrt (soo / (double) e2.size()) > 1e-9) a.halves = std::clamp (seo / std::sqrt (see * soo), -1.0, 1.0);
    }
    double num = 0, den2 = 0;
    for (int ci = 0; ci < m; ++ci)
    {
        double mu = 0; int cn = 0; for (int k = 0; k < K; ++k) { const double v = com[(size_t) ci * (size_t) K + (size_t) k]; if (fin (v)) { mu += v; ++cn; } }
        if (cn == 0) continue;
        mu /= cn; for (int k = 0; k < K; ++k) { const double v = com[(size_t) ci * (size_t) K + (size_t) k]; if (! fin (v)) continue; num += std::abs (v - mu - gg[(size_t) k]); den2 += std::abs (v - mu); }
    }
    a.ok = true; a.bands = (int) sl.size(); a.cyc_n = (int) sz.size();
    a.slope_med = pct (sl, 50); a.slope_lo = pct (sl, 25); a.slope_hi = pct (sl, 75); a.slope_min = pct (sl, 10); a.slope_max = pct (sl, 90);
    a.corr_med = pct (co, 50); a.corr_lo = pct (co, 25); a.corr_min = pct (co, 10); a.cyc_med = pct (sz, 50); a.cyc_lo = pct (sz, 25); a.cyc_hi = pct (sz, 75);
    a.r2c = 1.0 - num / std::max (den2, 1e-9); a.silent = (double) silent / ((double) m * K);
    return a;
}

// ---------------------------------------------------------------- a gate that leaves steps out
// A two-level shape: the first bin of the cycle laid out so that it starts in the middle of the low stretch, the bins of
// the high stretch, of the low stretch before it and of the low stretch after it (as places in that laid-out cycle).
struct Plateau { bool ok = false; int start = 0; std::vector<int> hi, loA, loB; };
inline Plateau plateaus (const Vec& g)
{
    Plateau p; const int K = (int) g.size(); if (K < 4) return p;
    const double lo = *std::min_element (g.begin(), g.end()), hi = *std::max_element (g.begin(), g.end()), dep = hi - lo; if (dep < 6.0) return p;
    std::vector<char> high ((size_t) K), low ((size_t) K); int nh = 0, nl = 0, either = 0;
    for (int i = 0; i < K; ++i) { high[(size_t) i] = g[(size_t) i] > hi - 0.25 * dep; low[(size_t) i] = g[(size_t) i] < lo + 0.25 * dep; nh += high[(size_t) i]; nl += low[(size_t) i]; either += (high[(size_t) i] || low[(size_t) i]); }
    if (nh < 2 || nl < 2 || (double) either / K < 0.7) return p;
    int kr = 0, kf = 0; Vec d ((size_t) K); for (int i = 0; i < K; ++i) d[(size_t) i] = g[(size_t) ((i + 1) % K)] - g[(size_t) i];
    for (int i = 1; i < K; ++i) { if (d[(size_t) i] > d[(size_t) kr]) kr = i; if (d[(size_t) i] < d[(size_t) kf]) kf = i; }      // the step up is between kr and kr + 1, down between kf and kf + 1
    const int nlow = pmod (kr - kf, K), start = (kf + 1 + nlow / 2) % K, upAt = pmod (kr + 1 - start, K), downAt = pmod (kf + 1 - start, K);
    if (! (0 < upAt && upAt < downAt && downAt < K)) return p;
    auto bin = [&] (int r) { return (start + r) % K; };
    for (int r = upAt; r < downAt; ++r) if (high[(size_t) bin (r)]) p.hi.push_back (r);
    for (int r = 0; r < upAt; ++r) if (low[(size_t) bin (r)]) p.loA.push_back (r);
    for (int r = downAt; r < K; ++r) if (low[(size_t) bin (r)]) p.loB.push_back (r);
    if (p.hi.empty() || p.loA.size() + p.loB.size() < 1) return p;
    p.ok = true; p.start = start; return p;
}
// Steps a gate leaves out: cycles in which the level neither steps up nor steps down where the shape says it should
// (looked at right at the two edges, two bins either side, median over the bands that sound there).
inline bool leftOut (const Grid& L, int K, const Vec& g, double share, int& start, std::vector<int>& out)
{
    const Plateau pl = plateaus (g); if (! pl.ok) return false;
    start = pl.start; out.clear(); const int n = (L.rows - start) / K, nc = L.cols; if (n < 3) return false;
    const Vec low = lowOf (L); auto Lf = [&] (int i, int c) { const double v = L.at (i, c); return std::isnan (v) ? low[(size_t) c] - 20.0 : v; };
    const double dep = *std::max_element (g.begin(), g.end()) - *std::min_element (g.begin(), g.end());
    auto edge = [&] (const std::vector<int>& hi, const std::vector<int>& lo)
    {
        Vec r ((size_t) n, nan()), buf; if (hi.empty() || lo.empty()) return r;
        for (int i = 0; i < n; ++i)
        {
            buf.clear(); const int r0 = start + i * K;
            for (int c = 0; c < nc; ++c)
            {
                double sh = 0, sl = 0; bool any = false;
                for (int k : hi) { sh += Lf (r0 + k, c); any = any || ! std::isnan (L.at (r0 + k, c)); }
                for (int k : lo) { sl += Lf (r0 + k, c); any = any || ! std::isnan (L.at (r0 + k, c)); }
                if (any) buf.push_back (sh / (double) hi.size() - sl / (double) lo.size());
            }
            if (buf.size() >= 3) r[(size_t) i] = med (buf);
        }
        return r;
    };
    auto first2 = [] (const std::vector<int>& v) { return std::vector<int> (v.begin(), v.begin() + (long) std::min<size_t> (2, v.size())); };
    auto last2 = [] (const std::vector<int>& v) { return std::vector<int> (v.end() - (long) std::min<size_t> (2, v.size()), v.end()); };
    const Vec up = edge (first2 (pl.hi), last2 (pl.loA)), down = edge (last2 (pl.hi), first2 (pl.loB));
    for (int i = 0; i < n; ++i)
    {
        const bool a = fin (up[(size_t) i]), b = fin (down[(size_t) i]); if (! a && ! b) continue;
        const double mx = a && b ? std::max (up[(size_t) i], down[(size_t) i]) : (a ? up[(size_t) i] : down[(size_t) i]);
        if (mx < share * dep) out.push_back (i);
    }
    return true;
}
// A gate's shape measured again from the steps it does not leave out.
inline void gateSteps (const Grid& L, int K, Vec& g, int& start, std::vector<int>& skip)
{
    start = 0; skip.clear();
    for (int rnd = 0; rnd < 2; ++rnd)
    {
        int st = 0; std::vector<int> sk;
        if (! leftOut (L, K, g, rnd == 0 ? 0.3 : 0.4, st, sk)) { start = 0; skip.clear(); return; }
        start = st; skip = sk;
        if (skip.empty() || (double) skip.size() > 0.7 * (double) ((L.rows - start) / K)) { skip.clear(); return; }
        std::vector<char> use ((size_t) L.rows, 1);
        for (int i : skip) for (int r = start + i * K; r < start + (i + 1) * K && r < L.rows; ++r) use[(size_t) r] = 0;
        const Fold f = pairsFold (L, K, &use); if (f.npairs == 0) { skip.clear(); return; }
        g = f.g;
    }
}

// ---------------------------------------------------------------- together: everything measured about one period
struct Tog
{
    double P = 0; int K = 0; Vec g; int start = 0; std::vector<int> skip; double r2 = 0, depth = 0; int cycles = 0;
    double skipfrac = 0, up = 0, steep = 0, fund = 0, rise = 0, fall = 0, climb = 0, flat = 0, share = 0, excess = 0;
    Agree a; double score = -std::numeric_limits<double>::infinity();
};
inline bool measureTogether (const Levels& E, double P, const Lane* lane, Tog& m)
{
    if (lane != nullptr && lane->pw > 0) P = refinePeriod (lane->z, P, 32, 0.004, 41);
    const int K = (int) std::clamp (std::nearbyint (P / 2.0), 12.0, 48.0); const Grid L = cellLevels (E.e[0], E.e[1], E.T, P, K);
    const Fold f = pairsFold (L, K); if (f.npairs == 0) return false;
    Vec g = f.g; int start = 0; std::vector<int> skip; Shape s = shapeNumbers (g); const int n = L.rows / K;
    if (s.steep >= 0.3 && depthOf (g) >= 6.0) { gateSteps (L, K, g, start, skip); s = shapeNumbers (g); }      // a gate's shape: are steps left out? then the shape from the others
    const Agree a = agreement (L, K, g, start, skip); if (! a.ok) return false;
    const Struck st = struckNumbers (g);
    m.P = P; m.K = K; m.g = g; m.start = start; m.skip = skip; m.r2 = f.r2; m.depth = depthOf (g); m.cycles = n; m.skipfrac = (double) skip.size() / std::max (n, 1);
    m.up = s.up; m.steep = s.steep; m.fund = s.fund; m.rise = s.rise; m.fall = st.fall; m.climb = st.climb; m.flat = st.flat; m.a = a; return true;
}

// ---------------------------------------------------------------- left against right (auto pan)
// What each side does (dB) when left is d dB above right and the two follow the law gL^p + gR^p = 2 (p = 1: the gains
// add up to a constant, p = 2: the powers do, p -> 0: as many dB up on one side as down on the other).
inline void lawGains (double d, double p, double& gl, double& gr)
{
    if (p < 0.05) { gl = d / 2.0; gr = -d / 2.0; return; }
    const double r = std::pow (10.0, std::clamp (d, -80.0, 80.0) / 20.0), c = (20.0 / p) * (std::log10 (2.0) - std::log10 (1.0 + std::pow (r, p)));
    gl = c + d; gr = c;
}
// Left minus right in dB over time (median over the bands that sound on both sides, each band against its own usual
// place in the picture), and per band.
inline void panLane (const Levels& E, Vec& d, Grid& D)
{
    const int T = E.T; const Vec S0 = box (E.e[0], T, NB, 16), S1 = box (E.e[1], T, NB, 16);
    D.rows = T; D.cols = NB; D.v.assign ((size_t) T * NB, nan()); d.assign ((size_t) T, nan());
    double top[NB], mx = -1e300; for (int b = 0; b < NB; ++b) { top[b] = -1e300; for (int t = 0; t < T; ++t) top[b] = std::max (top[b], S0[(size_t) t * NB + (size_t) b] + S1[(size_t) t * NB + (size_t) b]); mx = std::max (mx, top[b]); }
    bool live[NB], anyLive = false; for (int b = 0; b < NB; ++b) { live[b] = top[b] > mx * 1e-7; anyLive = anyLive || live[b]; }
    std::vector<int> cnt ((size_t) T, 0);
    for (int t = 0; t < T; ++t) for (int b = 0; b < NB; ++b)
    {
        const double a = S0[(size_t) t * NB + (size_t) b], c = S1[(size_t) t * NB + (size_t) b];
        if (live[b] && a + c > top[b] * 1e-5) { D.at (t, b) = std::clamp (10.0 * std::log10 (std::max (a, 1e-300) / std::max (c, 1e-300)), -60.0, 60.0); ++cnt[(size_t) t]; }
    }
    double own[NB]; Vec buf;
    for (int b = 0; b < NB; ++b) { buf.clear(); for (int t = 0; t < T; ++t) { const double v = D.at (t, b); if (! std::isnan (v)) buf.push_back (v); } own[b] = buf.empty() ? 0.0 : med (buf); }
    double add = 0; if (anyLive) { buf.clear(); for (int b = 0; b < NB; ++b) if (live[b]) buf.push_back (own[b]); add = med (buf); }
    for (int t = 0; t < T; ++t)
    {
        if (cnt[(size_t) t] < 3) continue;
        buf.clear(); for (int b = 0; b < NB; ++b) { const double v = D.at (t, b); if (! std::isnan (v)) buf.push_back (v - own[b]); }
        d[(size_t) t] = anyLive ? med (buf) + add : med (buf);
    }
}
// One cycle of a lane (NaN = unknown) at period P frames in K bins (medians); bins nothing fell into are filled in from their neighbours.
inline Vec foldProfile (const Vec& v, double P, int K)
{
    std::vector<Vec> bins ((size_t) K); const size_t n = v.size();
    for (size_t i = 0; i < n; ++i) { if (! fin (v[i])) continue; int k = (int) std::floor (std::fmod ((double) i, P) / P * K); if (k >= K) k = K - 1; bins[(size_t) k].push_back (v[i]); }
    Vec prof ((size_t) K, nan()), xi, fi; int have = 0;
    for (int k = 0; k < K; ++k) if (! bins[(size_t) k].empty()) { prof[(size_t) k] = med (bins[(size_t) k]); ++have; }
    if (have == 0) return Vec ((size_t) K, 0.0);
    if (have < K)
    {
        for (int rep = -1; rep <= 1; ++rep) for (int k = 0; k < K; ++k) if (fin (prof[(size_t) k])) { xi.push_back ((double) (k + rep * K)); fi.push_back (prof[(size_t) k]); }
        Vec out ((size_t) K); for (int k = 0; k < K; ++k) out[(size_t) k] = interp ((double) k, xi, fi);
        return out;
    }
    return prof;
}
// A cycle shape read at a position (in bins, any real number), straight lines between bin centres, round the cycle.
inline double atPhase (const Vec& prof, double pos)
{
    const int K = (int) prof.size(); const double q = pos - 0.5, fl = std::floor (q), f = q - fl; const long i = (long) fl;
    return prof[(size_t) pmod (i, K)] * (1.0 - f) + prof[(size_t) pmod (i + 1, K)] * f;
}
// A movement slower than half the sound (seen less than twice): the best fit of a sine with its second and third
// harmonic, for periods from half the sound's length up to four times its length; the shortest period that fits
// (about) as well as the best.
struct Slow { bool ok = false; double r2 = 0, P = 0; Vec c; };
inline Slow slowFit (const Vec& d, int T)
{
    Slow best; std::vector<int> t; Vec v; for (int i = 0; i < T; ++i) if (fin (d[(size_t) i])) { t.push_back (i); v.push_back (d[(size_t) i]); }
    const int n = (int) v.size(); if (n < 8) return best;
    const double mu = meanOf (v); double var = 1e-12; { double s = 0; for (double a : v) s += sq (a - mu); var = s + 1e-12; }
    const Vec lp = linspace (std::log (T / 2.0), std::log (4.0 * T), 60); std::vector<Slow> fits; Vec A ((size_t) n * 7); double top = -1e300;
    for (double l : lp)
    {
        const double P = std::exp (l), w = 2.0 * PI / P;
        for (int i = 0; i < n; ++i) { double* r = &A[(size_t) i * 7]; r[0] = 1.0; for (int h = 1; h <= 3; ++h) { const double a = (double) h * w * (double) t[(size_t) i]; r[2 * h - 1] = std::cos (a); r[2 * h] = std::sin (a); } }
        Slow f; double rss = 0; f.c = lsq (A, n, 7, v, rss); f.r2 = 1.0 - rss / var; f.P = P; f.ok = true; top = std::max (top, f.r2); fits.push_back (std::move (f));
    }
    if (top <= 0) return best;
    for (auto& f : fits) if (f.r2 >= top - 0.02) return f;
    return best;
}
struct Pan
{
    bool ok = false, slow = false; double P = 0; int K = 0; Vec prof, lane; double centre = 0, swing = 0, share = 0, adj = 0, resid = 0, cycles = 0, slope_med = 0, slope_lo = 0, slope_hi = 0, known = 0, fund = 0, steep = 0;
    int bands = 0; double score = -std::numeric_limits<double>::infinity();
};
// The left-right lane of a sound, made ready: is there enough of it to say anything, and how far does it swing.
struct PanLook { Vec d; Grid D; Lane ln; size_t known = 0; double swing = 0; bool ok = false; };
inline PanLook panLook (const Levels& E)
{
    PanLook L; panLane (E, L.d, L.D); Vec f; for (double v : L.d) if (fin (v)) f.push_back (v);
    L.known = f.size(); if (f.size() < 200) return L;
    L.swing = pct (f, 95.0) - pct (f, 5.0); L.ln = clipLane (L.d); L.ok = L.ln.pw > 0 && L.swing >= 0.5; return L;
}
// Everything measured about a left-right movement of period P (frames). sf: seen less than twice, the slow fit stands for its shape.
inline void panMeasure (const PanLook& L, int T, double P, double share, double adj, const Slow* sf, Pan& R)
{
    const Vec& d = L.d; const Grid& D = L.D; const bool slow = sf != nullptr; int K; Vec prof; double centre;
    if (slow)
    {
        P = sf->P; K = 64; prof.resize (64); const double w = 2.0 * PI / P; double lo = 1e300, hi = -1e300;
        for (int i = 0; i < K; ++i)
        {
            const double tt = ((double) i + 0.5) / K * P; double s = 0; for (int h = 1; h <= 3; ++h) s += sf->c[(size_t) (2 * h - 1)] * std::cos ((double) h * w * tt) + sf->c[(size_t) (2 * h)] * std::sin ((double) h * w * tt);
            prof[(size_t) i] = sf->c[0] + s; lo = std::min (lo, prof[(size_t) i]); hi = std::max (hi, prof[(size_t) i]);
        }
        share = adj = sf->r2; centre = 0.5 * (hi + lo);
    }
    else { K = (int) std::clamp (std::nearbyint (P / 8.0), 16.0, 64.0); prof = foldProfile (d, P, K); centre = meanOf (prof); }
    Vec res; for (int t = 0; t < T; ++t) if (fin (d[(size_t) t])) res.push_back (std::abs (d[(size_t) t] - atPhase (prof, std::fmod ((double) t, P) / P * K)));
    R.resid = med (res) * 1.4826;
    Vec sl, bv, bu;                                                     // does every band swing by the same amount?
    for (int b = 0; b < NB; ++b)
    {
        bv.clear(); bu.clear(); for (int t = 0; t < T; ++t) { const double v = D.at (t, b); if (fin (v) && fin (d[(size_t) t])) { bv.push_back (v); bu.push_back (d[(size_t) t]); } }
        if ((double) bv.size() < 0.3 * (double) L.known) continue;
        const double mv = medc (bv), mu = medc (bu); double vu = 0, uu = 0; for (size_t i = 0; i < bv.size(); ++i) { vu += (bv[i] - mv) * (bu[i] - mu); uu += sq (bu[i] - mu); }
        sl.push_back (vu / (uu + 1e-12));
    }
    R.bands = (int) sl.size(); if (sl.empty()) sl.push_back (0.0);
    const Shape sh = shapeNumbers (prof);
    R.ok = true; R.slow = slow; R.P = P; R.K = K; R.centre = centre; R.prof = prof; for (auto& v : R.prof) v -= centre; R.lane = d; for (auto& v : R.lane) v -= centre;
    R.share = share; R.adj = adj; R.cycles = T / P; R.slope_med = medc (sl); R.slope_lo = pct (sl, 25.0); R.slope_hi = pct (sl, 75.0); R.known = (double) L.known / T; R.fund = sh.fund; R.steep = sh.steep;
}
inline Pan findPan (const Levels& E, const Stop& stop)
{
    Pan R; const int T = E.T; const PanLook L = panLook (E); R.swing = L.swing; if (! L.ok) return R;
    const Lane& ln = L.ln;
    const Vec Ps = expGrid (60.0, T / 2.0, 0.004); const long np = (long) Ps.size(); if (np < 1) return R;
    Vec S ((size_t) np); long k = 0;
    for (long i = 0; i < np; ++i) { if ((i & 63) == 0 && stop && stop()) return R; S[(size_t) i] = foldStat (ln.z, Ps[(size_t) i], 24) / ln.pw; if (S[(size_t) i] > S[(size_t) k]) k = i; }
    double P = Ps[(size_t) k], share = S[(size_t) k]; const double P0 = P, share0 = share;
    for (int mlt = 2; mlt <= 6; ++mlt)                                  // a shorter period that explains (nearly) as much is the period
    {
        const double want = P0 / mlt; if (! (Ps[0] <= want)) continue;
        long j = 0; for (long i = 1; i < np; ++i) if (std::abs (Ps[(size_t) i] - want) < std::abs (Ps[(size_t) j] - want)) j = i;
        const long a = std::max (0L, j - 3), b = std::min (np, j + 4); long jj = a; for (long i = a + 1; i < b; ++i) if (S[(size_t) i] > S[(size_t) jj]) jj = i;
        if (S[(size_t) jj] >= 0.9 * share0 && (Ps[(size_t) jj] < P || (Ps[(size_t) jj] == P && S[(size_t) jj] < share))) { P = Ps[(size_t) jj]; share = S[(size_t) jj]; }
    }
    P = refinePeriod (ln.z, P, 32, 0.01, 81); const double cyc = T / P; const double adj = (share - 1.0 / cyc) / (1.0 - 1.0 / cyc);      // few cycles explain much of anything
    const Slow sf = slowFit (L.d, T); const bool slow = sf.ok && sf.r2 > 0.7 && (adj < 0.6 || (cyc < 2.5 && sf.r2 > adj + 0.05));    // nothing that is seen twice explains the lane, a slower movement does
    panMeasure (L, T, P, share, adj, slow ? &sf : nullptr, R); return R;
}
// The same measures at a period given from outside (frames; the lane's own best period within 1 % of it).
inline Pan panAt (const Levels& E, double P0)
{
    Pan R; const int T = E.T; const PanLook L = panLook (E); R.swing = L.swing; if (! L.ok || ! (P0 >= 60.0 && P0 <= T / 2.0)) return R;
    const double P = refinePeriod (L.ln.z, P0, 32, 0.01, 81), share = foldStat (L.ln.z, P, 24) / L.ln.pw, cyc = T / P;
    panMeasure (L, T, P, share, (share - 1.0 / cyc) / (1.0 - 1.0 / cyc), nullptr, R); return R;
}
// Which law do the two sides follow? The level of both sides together over the cycle (measured like a tremolo's shape)
// is set against what each law predicts from the measured left-right difference.
inline double fitLaw (const Levels& E, double P, int K, const Vec& dprof, double centre, double& lawfit)
{
    lawfit = 0.0; Vec Et (E.e[0].size()); for (size_t i = 0; i < Et.size(); ++i) Et[i] = E.e[0][i] + E.e[1][i];
    const Fold f = pairsFold (cellLevels (Et, Et, E.T, P, K), K); if (f.npairs == 0) return P_DEFAULT;
    const double a = std::pow (10.0, centre / 10.0), wl = a / (1.0 + a), wr = 1.0 / (1.0 + a), cm = meanOf (f.g); double best = 1e30, bp = P_DEFAULT, worst = -1e300; Vec h ((size_t) K);
    for (double l : linspace (std::log (0.25), std::log (6.0), 33))
    {
        const double p = std::exp (l); for (int k = 0; k < K; ++k) { double gl, gr; lawGains (dprof[(size_t) k], p, gl, gr); h[(size_t) k] = 10.0 * std::log10 (wl * std::pow (10.0, gl / 10.0) + wr * std::pow (10.0, gr / 10.0)); }
        const double hm = meanOf (h); double e = 0; for (int k = 0; k < K; ++k) e += sq ((f.g[(size_t) k] - cm) - (h[(size_t) k] - hm));
        e /= K; worst = std::max (worst, e); if (e < best) { best = e; bp = p; }
    }
    lawfit = 1.0 - best / std::max (worst, 1e-12); return bp;
}

// ---------------------------------------------------------------- the rule (Research/level: fit4.py, fit5.py -> rule.json)
inline double near1 (double v, double e = 0.02) { return -std::log (std::max (1.0 - v, 0.0) + e); }      // stretches the last stretch before 1: 0.9, 0.99, 0.999 lie evenly apart
constexpr int NTOG = 33, NPAN = 9;
constexpr double TOG_W[NTOG] = { -0.1440008586166994, 0.22669922237842646, 0.43638006666901, 0.3408344858030646, 0.604241229638823, -0.05664043189888368, -0.4260443591788236, -0.2997225966916462,
    0.39930764643689687, -0.015750306011053202, -0.28242711480245275, 0.7022535852707659, 1.5249927793894935, 1.5918539297525789, 2.9474540621889096, 0.40088748967935167, 3.5989665530875734,
    6.733980544742677, -0.1011529596170954, 0.7212018165768357, 0.054476804377302575, 0.39310123883445125, -4.973047471492535, 1.1030315071618688, -1.4112030027898759, -0.10249770812268408,
    -2.8188107531576425, 4.09188237077445, -2.4412772618869156, 3.859407816417894, 3.5765512823705357, -14.660032967935306, 6.351606564458543 };
constexpr double TOG_B = -7.063581789987328, TOG_CUT = 0.17151598267319404;
constexpr double PAN_W[NPAN] = { 3.528493258938674, 0.7216256610490664, -0.16233146608561377, 0.6944722506704705, -1.3467822564493916, -0.36905567905127434, -7.036053589238173, 1.0915702087703323, -0.07749520529590165 };
constexpr double PAN_B = -10.482197404015293, PAN_CUT = -2.6657134569988425;
// The measures as the rule sees them (the last nine: how far outside the range true movements keep to).
inline std::array<double, NTOG> togFeatures (const Tog& m)
{
    const Agree& a = m.a; const double sp = a.slope_hi - a.slope_lo;
    return { std::log (std::min (std::max (m.depth, 0.3), 60.0)), near1 (a.corr_lo), near1 (a.corr_med), near1 (a.corr_min), std::log (std::min (sp, 3.0) + 0.03), std::log (std::min (std::abs (a.slope_med - 1.0), 2.0) + 0.03),
             std::log (std::min (std::max (a.slope_max - a.slope_min, 0.0), 6.0) + 0.05), std::log (std::min (std::abs (a.cyc_med - 1.0), 2.0) + 0.03), std::log (std::min (std::max (a.cyc_hi - a.cyc_lo, 0.0), 4.0) + 0.03),
             near1 (a.halves), near1 (std::max (a.r2c, -1.0), 0.05), near1 (m.share, 0.05), near1 (m.excess, 0.05), std::log ((double) std::max (a.cyc_n, 1)), m.steep, near1 (m.fund, 0.05), a.silent, m.skipfrac,
             near1 (std::max (m.r2, -1.0), 0.05), std::log (std::min (a.dis_med, 2.0) + 0.01), std::log (std::min (a.dis_hi, 2.0) + 0.01), m.flat, m.fall, m.climb,
             std::max (0.0, 0.7 - a.cyc_lo), std::max (0.0, a.cyc_hi - 1.3), std::max (0.0, 12.0 - a.cyc_n) / 8.0, std::min (a.cyc_n, 60) / 60.0, std::max (0.0, a.slope_hi - 1.2), std::max (0.0, 0.8 - a.slope_lo),
             std::max (0.0, 0.9 - a.halves), std::max (0.0, 0.3 - m.share), std::max (0.0, 0.9 - a.corr_lo) };
}
// Coarse: what does not pass this is never reported. Deep enough, seen often enough, even and odd cycles alike, every
// band moving by about the same amount (what one gain on the whole sound does), explaining a fair share of what the
// level does, and about as far in every cycle.
inline bool firstFilter (const Tog& m)
{
    const Agree& a = m.a;
    return m.P >= PMIN && m.depth >= 3.0 && a.corr_lo >= 0.6 && a.cyc_n >= 4 && a.halves >= 0.8 && a.slope_hi - a.slope_lo <= 0.5 && std::abs (a.slope_med - 1.0) <= 0.5 && m.share >= SHARE_MIN && a.cyc_hi - a.cyc_lo <= CYC_SPREAD;
}
// A shape that jumps up and sinks back slowly is what struck and plucked notes do in every bar of music: never reported.
inline bool struck (const Tog& m) { return m.fall >= 0.15 && m.climb <= STRUCK * m.fall && m.fall * m.P >= 20.0; }
// How sure a shared movement is (0 and up: reported at the default Sureness); -inf: never reported.
inline double scoreTogether (const Tog& m)
{
    if (! firstFilter (m) || struck (m)) return -std::numeric_limits<double>::infinity();
    const auto f = togFeatures (m); double s = 0; for (int i = 0; i < NTOG; ++i) s += TOG_W[i] * f[(size_t) i];
    return s + TOG_B - TOG_CUT;
}
inline std::array<double, NPAN> panFeatures (const Pan& m)
{
    return { std::log (std::max (m.swing, 0.1)), std::clamp (m.adj, -1.0, 1.0), std::log (m.resid / std::max (m.swing, 0.1) + 0.003), std::log (std::min (std::abs (m.slope_med - 1.0), 2.0) + 0.01),
             std::log (std::min (std::max (m.slope_hi - m.slope_lo, 0.0), 3.0) + 0.01), m.known, m.steep, near1 (m.fund, 0.05), std::log (std::max (m.cycles, 0.25)) };
}
// Never a pan: a swing of less than 3 dB; bands that do not swing alike; a repeating shape that explains less of the
// lane than 0.2 over what chance explains (echoes that fall on alternating sides swing the lane by tens of dB and
// repeat nothing: every true pan of the fit data explains 0.3 and more). adjMin: that least share; the look in long
// windows asks for it afterwards (findSlowPan).
inline double scorePanRaw (const Pan& m, double adjMin = PAN_ADJ_MIN)
{
    if (! m.ok || m.swing < 3.0 || m.known < 0.3 || m.slope_hi - m.slope_lo > 0.2 || m.adj < adjMin) return -std::numeric_limits<double>::infinity();     // a pan swings every band alike
    const auto f = panFeatures (m); double s = 0; for (int i = 0; i < NPAN; ++i) s += PAN_W[i] * f[(size_t) i];
    return s + PAN_B - PAN_CUT;
}
// A movement from side to side that the sound shows once only (less than SEEN_MIN cycles: no part of it is seen coming
// back) may as well be the sound's own: something that passes by. Where the bands happen to swing alike nothing tells
// the two apart, so it is offered at a sureness of one half at most, which the default Sureness does not show.
inline double seenCap (const Pan& m, double score) { return m.cycles < SEEN_MIN ? std::min (score, ONCE_CAP) : score; }
inline double scorePan (const Pan& m) { return seenCap (m, scorePanRaw (m)); }
// What the Sureness slider compares: a score of 0 (the rule's own cut) sits at the slider's default, 65 %.
inline double sureness (double score) { return fin (score) ? 1.0 / (1.0 + std::exp (-(score + 0.6190392084062235))) : 0.0; }

inline std::vector<Tog> candidatesTogether (const Levels& E, const Stop& stop, int most = 12)
{
    std::vector<Tog> out; const Scan sc = scanTogether (E, PMIN, stop); if (stop && stop()) return out;
    for (const Peak& p : peaksOf (sc.Ps, sc.S, most))
    {
        if (stop && stop()) return out;
        auto it = sc.lane.find (tauFor (p.P)); Tog m; if (! measureTogether (E, p.P, it != sc.lane.end() ? &it->second : nullptr, m)) continue;
        m.share = p.share; m.excess = p.excess; m.score = scoreTogether (m); out.push_back (std::move (m));
    }
    return out;
}
// The same for a period given from outside (frames): the top of the scan within 3 % of it, measured and scored as a
// peak of the sound's own scan would be.
inline bool togetherAt (const Levels& E, double P0, Tog& m)
{
    const double lo = PMIN, hi = E.T / 3.0; if (! (P0 >= lo && P0 <= hi)) return false;
    const long w = 101, a = 15, n = 2 * (w + a) + 1; Vec Ps ((size_t) n), S ((size_t) n, 0.0); std::map<int, Lane> lane;
    for (long i = 0; i < n; ++i)
    {
        Ps[(size_t) i] = std::clamp (P0 * std::exp (0.002 * (double) (i - w - a)), lo, hi);
        const int tau = tauFor (Ps[(size_t) i]); auto it = lane.find (tau);
        if (it == lane.end()) it = lane.emplace (tau, clipLane (diffLane (E, tau, std::max (2, tau / 4)))).first;
        if (it->second.pw > 0) S[(size_t) i] = foldStat (it->second.z, Ps[(size_t) i], 16) / it->second.pw;
    }
    long k = w + a; for (long i = w; i <= w + 2 * a; ++i) if (S[(size_t) i] > S[(size_t) k]) k = i;
    Vec win ((size_t) (2 * w + 1)); for (long j = -w; j <= w; ++j) win[(size_t) (j + w)] = S[(size_t) std::clamp (k + j, 0L, n - 1)];
    std::nth_element (win.begin(), win.begin() + w, win.end());
    auto it = lane.find (tauFor (Ps[(size_t) k])); if (! measureTogether (E, Ps[(size_t) k], it != lane.end() ? &it->second : nullptr, m)) return false;
    m.share = S[(size_t) k]; m.excess = S[(size_t) k] - win[(size_t) w]; m.score = scoreTogether (m); return true;
}
// Of the candidates at or above `least`: the surest; the same movement seen at a shorter period (a whole fraction) wins if it passes too.
inline int chooseTogether (const std::vector<Tog>& c, double least = 0.0)
{
    int best = -1; for (int i = 0; i < (int) c.size(); ++i) if (c[(size_t) i].score >= least && (best < 0 || c[(size_t) i].score > c[(size_t) best].score)) best = i;
    if (best < 0) return -1;
    std::vector<int> ord; for (int i = 0; i < (int) c.size(); ++i) if (c[(size_t) i].score >= least) ord.push_back (i);
    std::stable_sort (ord.begin(), ord.end(), [&] (int a, int b) { return c[(size_t) a].P < c[(size_t) b].P; });
    for (int i : ord)
    {
        const double r = c[(size_t) best].P / c[(size_t) i].P, rr = std::nearbyint (r);
        if (c[(size_t) i].P < c[(size_t) best].P * 0.75 && std::abs (r - rr) < 0.03 * rr && c[(size_t) i].share >= 0.6 * c[(size_t) best].share) return i;
    }
    return best;
}

// ---------------------------------------------------------------- sharp edges (a gate's steps, a sawtooth's drop)
struct Edge { double off = 0; int half = 0, k = 0; Vec u; };       // its place in the cycle (samples), half the stretch it replaces (samples), the bin it follows, its course from 0 (before) to 1 (after) in steps of DEC samples
// Power of both channels together in steps of DEC samples, each step set against the mean power of the second around it.
inline Vec finePower (const Stereo& x)
{
    const size_t n = x.size() / DEC; Vec q (n); double mx = 0;
    for (size_t i = 0; i < n; ++i) { double s = 0; for (int k = 0; k < DEC; ++k) { const size_t j = i * DEC + (size_t) k; s += x.l[j] * x.l[j] + x.r[j] * x.r[j]; } q[i] = s / DEC; mx = std::max (mx, q[i]); }
    const Vec ref = box (q, (int) n, 1, (int) SR / DEC); for (size_t i = 0; i < n; ++i) q[i] = q[i] / (ref[i] + 1e-30 + 1e-9 * mx);
    return q;
}
// Steep steps of a cycle shape: (k, size in dB), the step lies between bin k and bin k + 1; the largest rise and the largest fall, if steep.
inline std::vector<std::pair<int, double>> edgesOf (const Vec& g, double depth)
{
    std::vector<std::pair<int, double>> out; const int K = (int) g.size(); if (K < 3) return out;
    Vec d ((size_t) K); for (int i = 0; i < K; ++i) d[(size_t) i] = g[(size_t) ((i + 1) % K)] - g[(size_t) i];
    for (double sgn : { 1.0, -1.0 })
    {
        int k = 0; for (int i = 1; i < K; ++i) if (sgn * d[(size_t) i] > sgn * d[(size_t) k]) k = i;
        const double a = d[(size_t) pmod (k - 1, K)], b = d[(size_t) ((k + 1) % K)], tot = d[(size_t) k] + (sgn * a > 0 ? a : 0.0) + (sgn * b > 0 ? b : 0.0);
        if (sgn * tot >= std::max (4.0, 0.35 * depth) && sgn * d[(size_t) k] >= 0.5 * sgn * tot) out.push_back ({ k, tot });
    }
    return out;
}
// Power around one edge, laid over all cycles.
inline Vec edgeFold (const Vec& q, double Ps, double off, int w, const Vec& cyc)
{
    Vec s ((size_t) (2 * w + 1), 0.0), c ((size_t) (2 * w + 1), 0.0); const long n = (long) q.size();
    for (double cy : cyc)
    {
        const long base = (long) std::nearbyint ((cy * Ps + off) / DEC);
        for (int t = -w; t <= w; ++t) { const long i = base + t; if (i >= 0 && i < n) { s[(size_t) (t + w)] += q[(size_t) i]; c[(size_t) (t + w)] += 1.0; } }
    }
    for (size_t i = 0; i < s.size(); ++i) s[i] /= std::max (c[i], 1.0);
    return s;
}
// Where along F does the level step most (in the direction sgn), comparing the m points after with the m points before: size in dB, place.
inline std::pair<double, int> contrast (const Vec& F, int m, double sgn)
{
    const int n = (int) F.size(); Vec cs ((size_t) n + 1, 0.0); for (int i = 0; i < n; ++i) cs[(size_t) i + 1] = cs[(size_t) i] + F[(size_t) i];
    double best = -1e300; int at = m;
    for (int t = m; t < n - m; ++t)
    {
        const double a = (cs[(size_t) (t + m)] - cs[(size_t) t]) / m, b = (cs[(size_t) t] - cs[(size_t) (t - m)]) / m, c = sgn * 10.0 * std::log10 (std::max (a, 1e-30) / std::max (b, 1e-30));
        if (c > best) { best = c; at = t; }
    }
    return { best, at };
}
// A shape with steep steps, looked at finely: the period that makes the steps sharpest when all cycles are laid over
// each other, and each step's place and course in steps of 0.125 ms. Returns the period in frames; no edges: P as it was.
inline double sharpen (const Vec& q, double P, int K, const Vec& g, int start, const std::vector<int>& skip, int ncyc, std::vector<Edge>& out)
{
    out.clear(); const double depth = depthOf (g); const auto ed = edgesOf (g, depth); if (ed.empty()) return P;
    const double Ps0 = P * HOP, binw = Ps0 / K; const int w = (int) std::ceil ((1.5 * binw + 0.004 * SR) / DEC), m = 8;
    if (2 * w + 1 <= 2 * m) return P;
    Vec cyc; for (int c = -1; c <= ncyc; ++c) cyc.push_back ((double) c);
    auto use = [&] (int k)                                              // cycles to lay over each other for the edge after bin k (left-out steps do not have it)
    {
        if (skip.empty()) return cyc;
        Vec u; for (double c : cyc) { const double lc = std::floor ((c * K + k + 1 - start) / K); if (std::find (skip.begin(), skip.end(), (int) lc) == skip.end()) u.push_back (c); }
        return u;
    };
    double bestTot = -1.0, Ps = Ps0;
    for (double dlt : linspace (-4e-4, 4e-4, 41))
    {
        const double p = Ps0 * (1.0 + dlt); double tot = 0;
        for (auto& e : ed) tot += contrast (edgeFold (q, p, (double) (e.first + 1) / K * p - HOP / 2.0, w, use (e.first)), m, e.second > 0 ? 1.0 : -1.0).first;
        if (tot > bestTot) { bestTot = tot; Ps = p; }
    }
    for (auto& e : ed)
    {
        const int k = e.first; const double size = e.second, sgn = size > 0 ? 1.0 : -1.0, off0 = (double) (k + 1) / K * Ps - HOP / 2.0; const Vec cy = use (k);
        const auto c = contrast (edgeFold (q, Ps, off0, w, cy), m, sgn); if (c.first < 0.5 * std::abs (size)) continue;      // the fine look does not show the step: leave the coarse shape
        const double off = off0 + (double) (c.second - w) * DEC; const int half = (int) std::min ((double) w, std::ceil ((1.0 * binw + 0.003 * SR) / DEC));
        const Vec F = edgeFold (q, Ps, off, half, cy); const int n = (int) F.size(), en = std::max (2, half / 4); Vec L ((size_t) n); for (int i = 0; i < n; ++i) L[(size_t) i] = 10.0 * std::log10 (std::max (F[(size_t) i], 1e-30));
        double lo = 0, hi = 0; for (int i = 0; i < en; ++i) { lo += L[(size_t) i]; hi += L[(size_t) (n - en + i)]; }
        lo /= en; hi /= en; if (std::abs (hi - lo) < 3.0) continue;
        Edge E; E.u.resize ((size_t) n); double run = 0; for (int i = 0; i < n; ++i) { const double v = std::clamp ((L[(size_t) i] - lo) / (hi - lo), 0.0, 1.0); run = i == 0 ? v : std::max (run, v); E.u[(size_t) i] = run; }
        for (int i = 0; i < en / 2; ++i) { E.u[(size_t) i] = 0.0; E.u[(size_t) (n - 1 - i)] = 1.0; }
        E.off = off; E.half = half * DEC; E.k = k; out.push_back (std::move (E));
    }
    return Ps / HOP;
}
// A gate that switches hard clicks at its edges: bands the sound leaves empty then show a short peak there, and the
// shared shape gets peaks (up to three bins wide) over its open level. Peaks narrower than five bins are cut off when
// that lowers the top by more than 3 dB and leaves a movement standing: most of the depth, or 6 dB of it (clicks can
// stand higher over the open level than the gate goes down; a gate that is open for a short moment only leaves
// nothing when its peak is cut, and is kept as it is).
inline Vec despike (const Vec& g)
{
    const int K = (int) g.size(); Vec er ((size_t) K), o ((size_t) K);
    for (int i = 0; i < K; ++i) { double v = g[(size_t) i]; for (int j = -2; j <= 2; ++j) v = std::min (v, g[(size_t) pmod (i + j, K)]); er[(size_t) i] = v; }
    for (int i = 0; i < K; ++i) { double v = er[(size_t) i]; for (int j = -2; j <= 2; ++j) v = std::max (v, er[(size_t) pmod (i + j, K)]); o[(size_t) i] = v; }
    const double left = depthOf (o);
    return (topOf (g) - topOf (o) > 3.0 && (left >= 0.5 * depthOf (g) || left >= 6.0)) ? o : g;
}
// Per band (of both channels): how far its level is down in the low quarter of the shape against the high quarter
// (median over cycles); nothing sounding counts as 20 dB under the band's quietest moment.
inline Vec bandDepths (const Grid& L, int K, const Vec& g, int start, const std::vector<int>& skip)
{
    Vec out; const int n = (L.rows - start) / K, nc = L.cols; std::vector<int> keep; for (int i = 0; i < n; ++i) if (std::find (skip.begin(), skip.end(), i) == skip.end()) keep.push_back (i);
    if (keep.size() < 2) return out;
    const Vec low = lowOf (L); auto Lf = [&] (int i, int c) { const double v = L.at (i, c); return std::isnan (v) ? low[(size_t) c] - 20.0 : v; };
    std::vector<int> o ((size_t) K); for (int i = 0; i < K; ++i) o[(size_t) i] = i;
    std::stable_sort (o.begin(), o.end(), [&] (int a, int b) { return g[(size_t) ((a + start) % K)] < g[(size_t) ((b + start) % K)]; });
    const int q = std::max (1, K / 4); Vec buf;
    for (int c = 0; c < nc; ++c)
    {
        buf.clear();
        for (int ci : keep)
        {
            const int r0 = start + ci * K; double hi = 0, lo = 0; bool any = false;
            for (int i = 0; i < q; ++i) { const int kh = o[(size_t) (K - q + i)], kl = o[(size_t) i]; hi += Lf (r0 + kh, c); lo += Lf (r0 + kl, c); any = any || ! std::isnan (L.at (r0 + kh, c)); }
            if (any) buf.push_back (hi / q - lo / q);
        }
        if (! buf.empty()) out.push_back (med (buf));
    }
    return out;
}

// Of the steps a gate leaves out (cycles without its steps up and down), those in which there is no gate at all (a
// sound that has the effect on a part of it only): the sound is as loud there as in the open stretch of the cycles
// around them. The level of all bands together over the bins where the shape is high, against the middle of that
// level in the four nearest cycles that are not left out; a step is as loud when it is less than half the shape's
// depth under them. A sound's own level goes up and down from step to step, so one step alone is not asked: a gate
// that follows a pattern leaves out a step or a few in a row, and those stay closed steps. Eight or more in a row are
// open when half of them are as loud; fewer are asked one by one only where they reach the sound's start or end
// (where an effect may begin or stop).
inline std::vector<int> openSteps (const Grid& L, int K, const Vec& g, int start, const std::vector<int>& skip)
{
    std::vector<int> out; const int n = (L.rows - start) / K, nc = L.cols; if (skip.empty() || n < 2) return out;
    const double lo = *std::min_element (g.begin(), g.end()), hi = *std::max_element (g.begin(), g.end()), dep = hi - lo; if (dep < 6.0) return out;
    std::vector<int> high; for (int j = 0; j < K; ++j) if (g[(size_t) ((start + j) % K)] > hi - 0.25 * dep) high.push_back (j);
    if (high.empty()) return out;
    std::vector<char> left ((size_t) n, 0); for (int i : skip) if (i >= 0 && i < n) left[(size_t) i] = 1;
    Vec lev ((size_t) n);
    for (int i = 0; i < n; ++i)
    {
        double pw = 0; for (int j : high) for (int c = 0; c < nc; ++c) { const double v = L.at (start + i * K + j, c); if (! std::isnan (v)) pw += std::pow (10.0, v / 10.0); }
        lev[(size_t) i] = 10.0 * std::log10 (pw / (double) high.size() + 1e-300);
    }
    Vec near; std::vector<char> loud ((size_t) n, 0);
    for (int i = 0; i < n; ++i)
    {
        if (! left[(size_t) i]) continue;
        near.clear(); for (int d = 1; d < n && near.size() < 4; ++d) for (int k : { i - d, i + d }) if (k >= 0 && k < n && ! left[(size_t) k] && near.size() < 4) near.push_back (lev[(size_t) k]);
        loud[(size_t) i] = ! near.empty() && lev[(size_t) i] > med (near) - 0.5 * dep;
    }
    for (int a = 0; a < n;)
    {
        if (! left[(size_t) a]) { ++a; continue; }
        int b = a; while (b + 1 < n && left[(size_t) (b + 1)]) ++b;
        const int len = b - a + 1; int cnt = 0; for (int i = a; i <= b; ++i) cnt += loud[(size_t) i];
        if (len >= 8) { if (2 * cnt >= len) for (int i = a; i <= b; ++i) out.push_back (i); }
        else if (a == 0 || b >= n - 1) for (int i = a; i <= b; ++i) if (loud[(size_t) i]) out.push_back (i);
        a = b + 1;
    }
    return out;
}

// ---------------------------------------------------------------- what was found
struct Found
{
    bool found = false, against = false, stopped = false;      // against: left and right against each other (auto pan); otherwise together (tremolo, gate)
    double period = 0, rate = 0;                                // seconds, Hz
    int K = 0; Vec prof[2];                                     // the gain over one cycle in dB per channel (K bins), its top at 0
    Vec lane[2];                                                // a pan seen less than twice: the course as it was measured, per frame (the repeating shape after it)
    size_t n = 0; double depth = 0, law = P_DEFAULT, lawfit = 0;
    int start = 0; std::vector<int> skip; std::vector<Edge> edges; double trust = CAP_DB;      // a gate: steps it leaves out, its steep steps; how deep the bands agree the movement goes
    std::vector<int> open;                                      // of the steps left out: those as loud as the open ones (no gate there at all: nothing is done to them)
    double score = -std::numeric_limits<double>::infinity(), sure = 0;                          // the rule's answer (0 and up: reported by the prototype), and what the Sureness slider compares
    bool steps() const { return ! against && ! edges.empty(); }                                // a shape with steep steps: a gate
    Tog tog; Pan pan; std::vector<Tog> cands;                   // everything measured (for the parity test and the picture)
};
// The found shared movement made ready for use: for a shape with steep steps the period and the steps are looked at finely first.
inline void togetherResult (const Stereo& x, const Levels& E, const Tog& best, Found& F)
{
    double P = best.P; const int K = best.K; Vec g = despike (best.g); int start = best.start; std::vector<int> skip = best.skip; std::vector<Edge> edges;
    if (best.steep >= 0.25 && best.depth >= 4.0)
    {
        const int ncyc = (int) (E.T / P) + 1; const Vec q = finePower (x); std::vector<Edge> ed; const double P2 = sharpen (q, P, K, g, start, skip, ncyc, ed);
        if (! ed.empty())
        {
            const Grid L = cellLevels (E.e[0], E.e[1], E.T, P2, K); const Fold f = pairsFold (L, K);
            if (f.npairs > 0)
            {
                Vec g2 = despike (f.g); int st2 = 0; std::vector<int> sk2;
                if (shapeNumbers (g2).steep >= 0.3 && depthOf (g2) >= 6.0) gateSteps (L, K, g2, st2, sk2);
                std::vector<Edge> ed3; const double P3 = sharpen (q, P2, K, g2, st2, sk2, ncyc, ed3);
                if (! ed3.empty()) { P = P3; g = g2; start = st2; skip = sk2; edges = ed3; }
            }
        }
    }
    const Grid L = cellLevels (E.e[0], E.e[1], E.T, P, K); const Vec bd = bandDepths (L, K, g, start, skip);
    F.trust = bd.size() >= 4 ? pct (bd, 25.0) + 6.0 : CAP_DB; F.open = openSteps (L, K, g, start, skip);
    const double tp = topOf (g); for (auto& v : g) v -= tp;
    F.found = true; F.against = false; F.period = P * HOP / SR; F.rate = SR / HOP / P; F.K = K; F.prof[0] = F.prof[1] = g; F.lane[0].clear(); F.lane[1].clear(); F.n = x.size();
    F.depth = depthOf (g); F.start = start; F.skip = skip; F.edges = edges; F.tog = best; F.score = best.score; F.sure = sureness (best.score);
}
inline void againstResult (const Stereo& x, const Levels& E, const Pan& pan, Found& F)
{
    const double P = pan.P; const int K = pan.K, T = E.T; double p = P_DEFAULT, lawfit = 0.0; Vec pl ((size_t) K), pr ((size_t) K);
    F.lane[0].clear(); F.lane[1].clear();
    if (pan.slow)                                                       // less than two cycles seen: the course as it was measured, holes bridged
    {
        Vec xi, fi; for (int t = 0; t < T; ++t) if (fin (pan.lane[(size_t) t])) { xi.push_back ((double) t); fi.push_back (pan.lane[(size_t) t]); }
        Vec pad ((size_t) T + 64), cs ((size_t) T + 65, 0.0);
        for (int i = 0; i < T + 64; ++i) pad[(size_t) i] = interp ((double) std::clamp (i - 32, 0, T - 1), xi, fi);
        for (int i = 0; i < T + 64; ++i) cs[(size_t) i + 1] = cs[(size_t) i] + pad[(size_t) i];
        F.lane[0].resize ((size_t) T); F.lane[1].resize ((size_t) T); double top = -1e9;
        for (int t = 0; t < T; ++t) { const double d = (cs[(size_t) (t + 32 + 16)] - cs[(size_t) (t + 32 - 16)]) / 32.0; lawGains (d, p, F.lane[0][(size_t) t], F.lane[1][(size_t) t]); top = std::max ({ top, F.lane[0][(size_t) t], F.lane[1][(size_t) t] }); }
        for (int k = 0; k < K; ++k) { lawGains (pan.prof[(size_t) k], p, pl[(size_t) k], pr[(size_t) k]); pl[(size_t) k] -= top; pr[(size_t) k] -= top; }
        for (int t = 0; t < T; ++t) { F.lane[0][(size_t) t] -= top; F.lane[1][(size_t) t] -= top; }
    }
    else
    {
        p = fitLaw (E, P, K, pan.prof, pan.centre, lawfit); if (lawfit < 0.8) p = P_DEFAULT;      // the level of both sides together does not say clearly: the usual law
        for (int k = 0; k < K; ++k) lawGains (pan.prof[(size_t) k], p, pl[(size_t) k], pr[(size_t) k]);
        const double top = std::max (topOf (pl), topOf (pr)); for (int k = 0; k < K; ++k) { pl[(size_t) k] -= top; pr[(size_t) k] -= top; }
    }
    F.found = true; F.against = true; F.period = P * HOP / SR; F.rate = SR / HOP / P; F.K = K; F.prof[0] = pl; F.prof[1] = pr; F.n = x.size(); F.law = p; F.lawfit = lawfit; F.depth = pan.swing;
    F.start = 0; F.skip.clear(); F.edges.clear(); F.trust = CAP_DB; F.pan = pan; F.score = pan.score; F.sure = sureness (pan.score);
}
// x at 48 kHz. `least`: the lowest score that is still offered (the prototype reports from 0 up; the app shows a
// movement when its sureness is above the Sureness slider, so it asks for everything down to SCORE_MIN).
// slowToo = false: a pan seen less than twice in x is not taken (x is a piece of a longer sound, which is looked at as a whole for that).
inline Found find (const Stereo& x, const Stop& stop = {}, double least = SCORE_MIN, bool slowToo = true)
{
    Found F; F.n = x.size(); if (x.size() < (size_t) (4 * HOP)) return F;
    const Levels E = levels (x); if (stop && stop()) { F.stopped = true; return F; }
    Pan pan = findPan (E, stop); pan.score = scorePan (pan); F.pan = pan; if (stop && stop()) { F.stopped = true; return F; }
    if (! slowToo && pan.slow) pan.ok = false;
    if (pan.ok && pan.score >= 0.0) { againstResult (x, E, pan, F); return F; }
    F.cands = candidatesTogether (E, stop); if (stop && stop()) { F.stopped = true; return F; }
    int best = chooseTogether (F.cands, 0.0);
    if (best < 0 && least < 0.0)                                        // nothing at the default Sureness: the likeliest of what is left, for a lower one
    {
        best = chooseTogether (F.cands, least); const double st = best >= 0 ? F.cands[(size_t) best].score : -std::numeric_limits<double>::infinity();
        if (pan.ok && pan.score >= least && pan.score >= st) { againstResult (x, E, pan, F); return F; }
    }
    if (best >= 0) { const Tog t = F.cands[(size_t) best]; togetherResult (x, E, t, F); }
    return F;
}
// x measured at a movement found elsewhere in the same sound (its kind and period given): the same measures and the
// same rule as for a movement x shows by itself.
inline Found findAt (const Stereo& x, const Levels& E, bool against, double period)
{
    Found F; F.n = x.size(); const double P = period * SR / HOP;
    if (against) { Pan pan = panAt (E, P); pan.score = scorePan (pan); F.pan = pan; if (pan.ok && fin (pan.score)) againstResult (x, E, pan, F); }
    else { Tog m; if (togetherAt (E, P, m) && fin (m.score)) togetherResult (x, E, m, F); }
    return F;
}
// Only a pan slower than `slowest` seconds a cycle (what a piece of PIECE seconds cannot see often enough).
// Over a long window the sound's own place between left and right wanders (one part of it sits further left than the
// next), and that pulls the found period to a whole multiple of the movement's: three cycles laid over each other hold
// the wander as well, and the fit for a movement seen less than twice holds a faster one as its third harmonic. So the
// lane is asked again at a half and a third of the found period. A fraction is the period when the rule finds it surer
// than a movement that is already sure, or when it explains clearly more of the lane for its count of cycles (then
// the less sure of the two scores stands): neither can show a movement that was not shown before. A fraction that
// explains less than 0.4 of what the found period explains is not asked at all: a movement folded at a wrong fraction
// of its period all but disappears (0 to 0.15 is left), and the rule's score means nothing for a period that does not
// explain the lane (it was fitted on periods that do).
// The least share a pan must explain (PAN_ADJ_MIN) is not asked here: a pan that is on a part of the window only
// explains less of the whole window (0.13 for one on the first half of a minute). What this look finds is a hint: it
// stands only when it explains that share after all, or when the stretch that has it, measured by itself, passes
// the whole rule (narrow); otherwise it is no find (lookPieces).
inline Found findSlowPan (const Stereo& x, const Stop& stop, double slowest)
{
    Found F; F.n = x.size(); if (x.size() < (size_t) (4 * HOP)) return F;
    const Levels E = levels (x); if (stop && stop()) { F.stopped = true; return F; }
    const double any = -std::numeric_limits<double>::infinity();        // no least share here: see above
    Pan pan = findPan (E, stop); pan.score = scorePanRaw (pan, any); F.pan = pan; if (stop && stop()) { F.stopped = true; return F; }
    if (pan.ok && pan.score >= SCORE_MIN)
    {
        Pan surer, fuller;
        for (int m = 2; m <= 3; ++m)
        {
            Pan q = panAt (E, pan.P / m); q.score = scorePanRaw (q, any); if (! q.ok || ! fin (q.score) || q.adj < std::max (0.1, 0.4 * pan.adj)) continue;
            if (pan.score >= 0.0 && q.score > pan.score && (! surer.ok || q.score > surer.score)) surer = q;
            if (! pan.slow && q.adj >= 0.1 && q.adj >= pan.adj + 0.1 && (! fuller.ok || q.adj > fuller.adj)) fuller = q;
        }
        if (surer.ok) pan = surer;
        else if (fuller.ok) { fuller.score = std::min (fuller.score, pan.score); pan = fuller; }
    }
    pan.score = seenCap (pan, pan.score); F.pan = pan;                  // seen once only: offered at a lower Sureness (see scorePan)
    if (pan.ok && pan.score >= SCORE_MIN && pan.P * HOP / SR > slowest) againstResult (x, E, pan, F);
    return F;
}

// ---------------------------------------------------------------- the gain, and its use
// The found gain in dB at any place of a sound that starts where the analysed sound started.
struct GainAt
{
    const Found& F; double Pf = 1, Ps = 1, low[2] = { 0, 0 }; std::vector<std::array<double, 4>> ab;
    explicit GainAt (const Found& f) : F (f)
    {
        if (! F.found || F.K <= 0) return;
        Pf = F.period * SR / HOP; Ps = Pf * HOP; const int K = F.K;
        for (const Edge& e : F.edges)
        {
            std::array<double, 4> v {}; const int k = e.k;
            for (int ch = 0; ch < 2; ++ch) { const Vec& p = F.prof[ch]; v[(size_t) ch * 2] = 0.5 * (p[(size_t) pmod (k - 2, K)] + p[(size_t) pmod (k - 1, K)]); v[(size_t) ch * 2 + 1] = 0.5 * (p[(size_t) pmod (k + 2, K)] + p[(size_t) pmod (k + 3, K)]); }
            ab.push_back (v);
        }
        if (! F.skip.empty()) { const size_t q = std::max<size_t> (1, (size_t) K / 8); low[0] = lowEnd (F.prof[0], q); low[1] = lowEnd (F.prof[1], q); }
    }
    // ts: the place in samples (any real number; before 0 the movement is thought on backwards)
    void operator() (double ts, double& gl, double& gr) const
    {
        gl = gr = 0.0; if (! F.found || F.K <= 0) return;
        const int K = F.K; const double t = ts / HOP, pos = std::fmod (t, Pf) / Pf * K; gl = atPhase (F.prof[0], pos); gr = atPhase (F.prof[1], pos);
        for (size_t j = 0; j < F.edges.size(); ++j)                     // steep steps: their place and course as seen in the fine look
        {
            const Edge& e = F.edges[j]; const double half = (double) e.half; const int nu = (int) e.u.size(); if (nu < 2 || half <= 0) continue;
            double rel = std::fmod (ts - e.off + half, Ps); if (rel < 0) rel += Ps;
            if (! (rel < 2.0 * half)) continue;
            const double pu = rel / DEC; double u; if (pu >= nu - 1) u = e.u[(size_t) nu - 1]; else { const int i = (int) pu; u = (e.u[(size_t) i + 1] - e.u[(size_t) i]) * (pu - i) + e.u[(size_t) i]; }
            const double taper = std::clamp (std::min (rel, 2.0 * half - rel) / (0.25 * half), 0.0, 1.0);
            gl += taper * (ab[j][0] + u * (ab[j][1] - ab[j][0]) - gl); gr += taper * (ab[j][2] + u * (ab[j][3] - ab[j][2]) - gr);
        }
        if (! F.skip.empty())                                           // steps the gate left out: closed all through
        {
            const double cyc = std::floor ((t / Pf * K - F.start) / K);
            if (std::find (F.skip.begin(), F.skip.end(), (int) cyc) != F.skip.end())
            {
                if (std::find (F.open.begin(), F.open.end(), (int) cyc) != F.open.end()) gl = gr = 0.0; else { gl = low[0]; gr = low[1]; }
            }
        }
        if (! F.lane[0].empty() && ts >= 0)                             // the measured course while it lasts, the repeating shape after it
        {
            const size_t len = F.lane[0].size();
            if (t < (double) (len - 1)) { const size_t i = (size_t) t; const double f = t - (double) i; gl = (F.lane[0][i + 1] - F.lane[0][i]) * f + F.lane[0][i]; gr = (F.lane[1][i + 1] - F.lane[1][i]) * f + F.lane[1][i]; }
        }
    }
};
// The found gain in dB for n samples; sample i stands at the place first + i * speed of the analysed sound (speed 2:
// the movement twice as fast).
inline void gainDb (const Found& F, size_t n, Vec& gl, Vec& gr, double speed = 1.0, double first = 0.0)
{
    gl.assign (n, 0.0); gr.assign (n, 0.0); if (! F.found || F.K <= 0) return;
    const GainAt at (F); for (size_t i = 0; i < n; ++i) at (first + (double) i * speed, gl[i], gr[i]);
}
// How far to lift (dB) to take the gain g (dB) out again. Safeguards: never more than CAP_DB, nor more than `trust`;
// around every moment the gain is taken as the highest within 1 ms (an edge that is a fraction of a millisecond off
// then leaves a short dip, not a click); and where the gain is under CLOSED_HI the gate counts as closing: the lift
// fades and is nothing from CLOSED_LO down. For a shape with steep steps `low` (its low level) decides that for the
// whole movement, the steps included.
inline Vec liftDb (const Vec& g, double trust, bool steps, double low)
{
    const long n = (long) g.size(); Vec out ((size_t) n); const double cap = std::min (CAP_DB, trust), fadeAll = std::clamp ((low - CLOSED_LO) / (CLOSED_HI - CLOSED_LO), 0.0, 1.0);
    std::vector<long> dq ((size_t) n); long head = 0, tail = 0, next = 0;                         // the highest within GUARD samples either side
    for (long i = 0; i < n; ++i)
    {
        const long hi = std::min (n - 1, i + GUARD);
        for (; next <= hi; ++next) { while (tail > head && g[(size_t) dq[(size_t) tail - 1]] <= g[(size_t) next]) --tail; dq[(size_t) tail++] = next; }
        while (dq[(size_t) head] < i - GUARD) ++head;
        const double gg = g[(size_t) dq[(size_t) head]], lift = std::min (-gg, cap);
        out[(size_t) i] = lift * (steps ? fadeAll : std::clamp ((gg - CLOSED_LO) / (CLOSED_HI - CLOSED_LO), 0.0, 1.0));
    }
    return out;
}
// What strength s makes of the found gain (dB, in place): s >= 0: s times it; s < 0: the lift that takes it out, -s times.
inline void shapeDb (const Found& F, double s, Vec& gl, Vec& gr)
{
    if (s >= 0) { for (size_t i = 0; i < gl.size(); ++i) { gl[i] *= s; gr[i] *= s; } return; }
    const bool steps = ! F.edges.empty(); const double low = steps ? lowEnd (F.prof[0], std::max<size_t> (1, (size_t) F.K / 8)) : 0.0;
    gl = liftDb (gl, F.trust, steps, low); gr = liftDb (gr, F.trust, steps, low);
    for (size_t i = 0; i < gl.size(); ++i) { gl[i] *= -s; gr[i] *= -s; }
}
// The gain in dB that the movement means at strength s: s >= 0 puts it on (s times in dB, at `speed`), s < 0 takes it
// out of the sound it was found on (-1: all of it).
inline void strengthDb (const Found& F, size_t n, double s, Vec& gl, Vec& gr, double speed = 1.0, double first = 0.0)
{
    gainDb (F, n, gl, gr, s >= 0 ? speed : 1.0, first); if (F.found) shapeDb (F, s, gl, gr);
}
// On the sound the movement was found on, as its faders ask. amount: 0 = taken out, 1 = untouched, 2 = twice as deep.
// speed: 1 = as found; anything else: the found movement is taken out and put back at that speed, amount times.
inline void faderDb (const Found& F, size_t n, double amount, double speed, Vec& gl, Vec& gr, double first = 0.0)
{
    if (std::abs (speed - 1.0) < 1e-9) { strengthDb (F, n, amount - 1.0, gl, gr, 1.0, first); return; }
    strengthDb (F, n, -1.0, gl, gr, 1.0, first); Vec al, ar; gainDb (F, n, al, ar, speed, first);
    for (size_t i = 0; i < n; ++i) { gl[i] += amount * al[i]; gr[i] += amount * ar[i]; }
}
inline Stereo withDb (const Stereo& x, const Vec& gl, const Vec& gr)
{
    Stereo y = x; const double k = std::log (10.0) / 20.0;
    for (size_t i = 0; i < x.size() && i < gl.size(); ++i) { y.l[i] = x.l[i] * std::exp (k * gl[i]); y.r[i] = x.r[i] * std::exp (k * gr[i]); }
    return y;
}
// s = -1: the found movement out of x (the sound it was found on). s = +1: onto x. In between and up to +2: scaled in dB.
inline Stereo apply (const Stereo& x, const Found& F, double s = -1.0)
{
    if (! F.found) return x;
    Vec gl, gr; strengthDb (F, x.size(), s, gl, gr); return withDb (x, gl, gr);
}

// ---------------------------------------------------------------- a whole sound, however long
// The rule was fitted and tested on sounds of 6 to 9 s. A longer sound is measured in pieces of 6 to 9 s: each piece
// first on its own, as tested. Then every piece is measured again at the movements other pieces found (the same
// measures and the same rule, the period given): the pieces that show the movement at its period carry it. The
// movement's score is the middle one of the carrying pieces' scores (not the best one: with more pieces the best score
// of a clean sound only goes up), less ALONE when just one of several pieces carries it. A pan too slow for such
// pieces (seen less than four times in one) is looked for in windows of 36 to 72 s instead; what such a window finds is
// asked again at a half and a third of its period (findSlowPan: a pan three times as fast looks like a slow one with a
// strong third harmonic, and three cycles laid over each other hold the sound's own wander from side to side).
// The clock: where in the surest piece's cycle every place of the sound stands, from the pieces that carry the
// movement (one steady speed if that fits them all, else from piece to piece).
// Where the movement is: every stretch of half a second or more is asked whether the sound does there what would be
// taken out (askPiece, spansOf). Only those stretches are touched: an effect that is on a part of the sound is taken
// out of that part, and the rest is left exactly as it was. If one window found the movement and it turns out to be on
// a part of it only, that part is measured again by itself (narrow): the movement is surer and truer there.
// Taken out, a piece that carries the movement by the rule loses its own measured movement, any other stretch that
// has it the surest piece's movement on the clock. Put onto another sound, the surest piece's shape runs on the first
// sound's clock, at the times at which the first sound has the movement (an effect that comes in halfway comes in
// halfway on the second sound too; past the first sound's end as at its end).
constexpr double PIECE = 6.0;                                           // seconds: the shortest piece
constexpr double LONG = 36.0;                                           // seconds: the shortest window of the look for slow pans
constexpr double ALONE = 1.0;                                           // what the score loses when only one of several pieces shows the movement
constexpr double CARRY = -2.0;                                          // a piece measured at the found period carries the movement from this score up
constexpr double SAME = 0.03;                                           // periods within 3 % of each other are one movement
constexpr double WIN = 0.5;                                             // seconds: the shortest stretch that is asked whether the movement is there
constexpr double SWITCH = 4.0;                                          // what a change between 'there' and 'not there' costs: that many half seconds of plain evidence
constexpr double LEAN = 0.10;                                           // a piece that carries the movement by the rule leans towards 'there' by this much evidence, any other piece away from it
constexpr double EDGE_GUARD = 0.050, EDGE_RAMP = 0.030;                 // seconds: what is left alone just inside each end of a stretch with the movement, and the fade after it
// The first sound's clock: for every place in the sound (samples) the count of the surest piece's cycles (rising
// without a break). Straight between its knots and beyond them.
struct Clock
{
    Vec t, ph; bool steady = false; mutable size_t i = 1;
    double at (double x) const
    {
        const size_t n = t.size(); if (n < 2) return n ? ph[0] : 0.0;
        while (i < n - 1 && x >= t[i]) ++i;
        while (i > 1 && x < t[i - 1]) --i;
        return ph[i - 1] + (ph[i] - ph[i - 1]) * (x - t[i - 1]) / (t[i] - t[i - 1]);
    }
};
// A stretch of the sound that is asked whether the movement is there. u: +1 = it does just what would be taken out,
// -1 = none of it, 0 = cannot be told; w: how much that says; own: the stretch lies in a piece that carries the
// movement by the rule; on: the answer. e: the same question frame by frame (first: the first frame's place in samples).
struct Window { size_t a = 0, b = 0; double u = 0, w = 0; bool own = false, on = false; std::vector<float> e; size_t first = 0; };
struct Span { size_t a = 0, b = 0; };
// own: the score of what the piece showed by itself. guided: f was measured at the period other pieces found. clocked:
// the piece shows nothing by the rule, but at least half of it follows the surest piece's movement on the clock.
// there: the share of the piece that has the movement.
struct Piece { size_t a = 0, b = 0; bool sounds = false, use = false, guided = false, clocked = false; double own = -std::numeric_limits<double>::infinity(), there = 0; Found f; };
struct Movement
{
    bool found = false, against = false, steps = false, stopped = false, wide = false;      // wide: from the look in long windows (a slow pan)
    double score = -std::numeric_limits<double>::infinity(), sure = 0, rate = 0, depth = 0; int lead = -1, sounding = 0, carrying = 0; size_t n = 0;
    std::vector<Piece> pieces; Clock clock; std::vector<Window> windows; std::vector<Span> spans;      // spans: where the movement is (what is taken out)
    const Found* best() const { return lead >= 0 ? &pieces[(size_t) lead].f : nullptr; }
};
inline bool sameMovement (const Found& a, const Found& b)
{
    if (! a.found || ! b.found || a.against != b.against) return false;
    if (a.against && ! a.lane[0].empty() && ! b.lane[0].empty()) return true;         // both seen less than twice
    return std::abs (b.period / a.period - 1.0) <= SAME;
}
// How many cycles the shape of f lies ahead of the lead's: f at a place in its cycle is the lead at that place + this.
inline double shiftOf (const Found& lead, const Found& f)
{
    const int N = 240; Vec a ((size_t) 2 * N), b ((size_t) 2 * N);
    for (int ch = 0; ch < 2; ++ch)
    {
        double ma = 0, mb = 0;
        for (int i = 0; i < N; ++i)
        {
            const double u = ((double) i + 0.5) / N; a[(size_t) (ch * N + i)] = atPhase (lead.prof[ch], u * lead.K); b[(size_t) (ch * N + i)] = atPhase (f.prof[ch], u * f.K);
            ma += a[(size_t) (ch * N + i)]; mb += b[(size_t) (ch * N + i)];
        }
        for (int i = 0; i < N; ++i) { a[(size_t) (ch * N + i)] -= ma / N; b[(size_t) (ch * N + i)] -= mb / N; }
    }
    Vec c ((size_t) N); int k = 0;
    for (int sft = 0; sft < N; ++sft)
    {
        double v = 0; for (int ch = 0; ch < 2; ++ch) for (int i = 0; i < N; ++i) v += b[(size_t) (ch * N + i)] * a[(size_t) (ch * N + (i + sft) % N)];
        c[(size_t) sft] = v; if (v > c[(size_t) k]) k = sft;
    }
    const double l = c[(size_t) ((k + N - 1) % N)], m = c[(size_t) k], r = c[(size_t) ((k + 1) % N)], den = l - 2.0 * m + r;
    return ((double) k + (den < 0 ? 0.5 * (l - r) / den : 0.0)) / N;
}
// The clock from the pieces that carry the movement by the rule. Each says where in the surest piece's cycle its own
// middle stands (its shape set against the surest piece's shape). If one steady speed puts all of them within 0.03
// cycles the clock is that straight line (an effect's own oscillator: the usual case); otherwise it runs from middle
// to middle (a speed that wanders). One piece alone: its own period.
inline Clock clockOf (const Movement& M)
{
    Clock C; const Found* F = M.best(); if (! M.found || F == nullptr) return C;
    struct Knot { double c, ph, P; }; std::vector<Knot> kn; int lead = -1; const bool course = ! F->lane[0].empty();
    for (size_t j = 0; j < M.pieces.size(); ++j)
    {
        const Piece& p = M.pieces[j]; const bool isLead = (int) j == M.lead; if (! p.use || p.clocked || (course && ! isLead)) continue;
        const double P = p.f.period * SR, c = 0.5 * ((double) p.a + (double) p.b); if (isLead) lead = (int) kn.size();
        kn.push_back ({ c, (isLead ? 0.0 : shiftOf (*F, p.f)) + (c - (double) p.a) / P, P });
    }
    if (lead < 0) return C;
    const int n = (int) kn.size(); const Knot L = kn[(size_t) lead];
    auto line = [&] (double c0, double ph0, double P) { C.t = { c0, c0 + P }; C.ph = { ph0, ph0 + 1.0 }; };
    if (n == 1) { line (L.c, L.ph, L.P); return C; }
    // one steady speed? the period near the pieces' own that brings their places in the cycle together best
    Vec per; for (const Knot& k : kn) per.push_back (k.P); const double Pm = medc (per);
    Vec dev; for (const Knot& k : kn) dev.push_back (std::abs (k.P / Pm - 1.0)); const double spread = std::clamp (2.0 * medc (dev), 0.001, 0.004), reach = 3.0 * spread;
    double far = 0; for (const Knot& k : kn) far = std::max (far, std::abs (k.c - L.c));
    const int steps = std::clamp ((int) std::ceil (2.0 * reach * far / Pm / 0.02), 8, 4000); double bestP = Pm, bestR = -1;
    for (int i = 0; i <= steps; ++i)
    {
        const double P = Pm * (1.0 - reach + 2.0 * reach * i / steps); double re = 0, im = 0;
        for (const Knot& k : kn) { const double r = 2.0 * PI * (k.ph - L.ph - (k.c - L.c) / P); re += std::cos (r); im += std::sin (r); }
        const double R = std::sqrt (re * re + im * im) / n * std::exp (-0.5 * sq ((P / Pm - 1.0) / spread));
        if (R > bestR) { bestR = R; bestP = P; }
    }
    Vec x ((size_t) n), y ((size_t) n); double re = 0, im = 0;
    for (int j = 0; j < n; ++j) { const double r = 2.0 * PI * (kn[(size_t) j].ph - L.ph - (kn[(size_t) j].c - L.c) / bestP); re += std::cos (r); im += std::sin (r); }
    const double mid = std::atan2 (im, re) / (2.0 * PI);
    for (int j = 0; j < n; ++j) { const double r = kn[(size_t) j].ph - L.ph - (kn[(size_t) j].c - L.c) / bestP - mid; x[(size_t) j] = kn[(size_t) j].c - L.c; y[(size_t) j] = r - std::nearbyint (r) + mid; }
    const double mx = meanOf (x), my = meanOf (y); double sxy = 0, sxx = 0; for (int j = 0; j < n; ++j) { sxy += (x[(size_t) j] - mx) * (y[(size_t) j] - my); sxx += sq (x[(size_t) j] - mx); }
    const double slope = sxx > 0 ? sxy / sxx : 0.0, off = my - slope * mx; double rss = 0; for (int j = 0; j < n; ++j) rss += sq (y[(size_t) j] - off - slope * x[(size_t) j]);
    if (std::sqrt (rss / n) <= 0.03) { const double P = 1.0 / (1.0 / bestP + slope); line (L.c, L.ph + off, P); C.steady = true; return C; }
    // a wandering speed: from middle to middle, whole cycles added so that the count runs on, outwards from the surest piece
    Vec ph ((size_t) n); ph[(size_t) lead] = L.ph;
    for (int j = lead + 1; j < n; ++j) { const Knot& q = kn[(size_t) j - 1]; const Knot& k = kn[(size_t) j]; const double pred = ph[(size_t) j - 1] + (k.c - q.c) * 0.5 * (1.0 / q.P + 1.0 / k.P); ph[(size_t) j] = k.ph + std::nearbyint (pred - k.ph); }
    for (int j = lead - 1; j >= 0; --j) { const Knot& q = kn[(size_t) j + 1]; const Knot& k = kn[(size_t) j]; const double pred = ph[(size_t) j + 1] - (q.c - k.c) * 0.5 * (1.0 / q.P + 1.0 / k.P); ph[(size_t) j] = k.ph + std::nearbyint (pred - k.ph); }
    for (int j = 0; j < n; ++j) { C.t.push_back (kn[(size_t) j].c); C.ph.push_back (ph[(size_t) j]); }
    return C;
}
// Where in the sound is the movement? A movement found in some pieces need not be on all of the sound (an effect
// switched on for a part of it), and a piece that carries it need not carry it from end to end. So before anything is
// taken out, every stretch of half a second or more (a whole cycle; of a pan half a cycle) is asked whether the sound
// does there what would be taken out. What the stretch does (together: the level change of all bands over a short
// step; against: left minus right) is set against what the movement would do there, z against p, and against no
// movement at all: u = (sum |z - m0| - sum |z - p - m1|) / sum |p - mp|. That is +1 when the sound does just what the
// movement says, -1 when it does none of it, and near 0 when it cannot be told (silence, or the sound's own changes far
// larger than the movement's). Together, m0, m1 and mp are the stretch's own middles of z, z - p and p (a sound that
// dies away does not count for or against). Against, the sound's own place between left and right is one number m for
// both answers (mp = 0), taken from the stretches within a cycle either side: the middle of z or of z - p, whichever
// leaves less when every frame may follow either answer; so a pan that holds one side for a while still says
// something there.
// A piece that carries the movement by the rule is asked twice: with its own measured movement, and with the surest
// piece's movement on the clock. The one that the piece follows better is what is taken out of it (`lead` says which):
// a piece that has the movement on a part of it only has measured a blurred shape of its own. A gate's own measurement
// can also be wrong in a way the answers per stretch do not show (steps it takes for left out predict nothing and so
// cost nothing): there the surest piece's gate is taken as well when it takes clearly more of the piece's level steps
// away (by 8 % and more: the sum of |z - m0| - |z - p - m1| over the piece).
inline void askPiece (const Stereo& x, const Movement& M, const Piece& p, std::vector<Window>& out, bool& lead)
{
    const Found& L = *M.best(); const bool rule = p.use && ! p.clocked, isLead = &p == &M.pieces[(size_t) M.lead]; lead = ! rule;
    const double wl = std::max (WIN * SR, (L.against ? 0.5 : 1.0) * L.period * SR); const size_t len = p.b - p.a, nw = std::max<size_t> (1, (size_t) ((double) len / wl));
    auto blank = [&] { std::vector<Window> W (nw); for (size_t k = 0; k < nw; ++k) { W[k].a = p.a + k * len / nw; W[k].b = p.a + (k + 1) * len / nw; W[k].own = rule; } return W; };
    if (! p.sounds) { for (auto& w : blank()) out.push_back (std::move (w)); return; }
    const int tauL = L.against ? 0 : tauFor (L.period * SR / HOP), tauO = rule && ! p.f.against ? tauFor (p.f.period * SR / HOP) : tauL;
    const size_t mg = (size_t) (std::max (tauL, tauO) + 64) * HOP, a0 = (p.a > mg ? p.a - mg : 0) / HOP * HOP, b0 = std::min (x.size(), p.b + mg);
    const Levels E = levels (x.slice (a0, b0 - a0)); const int T = E.T; Vec zp; Grid D; if (L.against) panLane (E, zp, D);
    std::map<int, Vec> zt;                                              // together: the lane for a step of tau frames
    auto ask = [&] (bool own, double& total, double& taken)
    {
        const Found& F = own ? p.f : L; const double Ps = F.period * SR; const GainAt at (F); std::vector<Window> W = blank(); Vec pr; total = 0; taken = 0; std::vector<char> gone; double usual = 0;
        auto place = [&] (int t) { const double i = (double) a0 + (double) t * HOP; return own ? i - (double) p.a : M.clock.at (i) * Ps; };
        const Vec* zq = &zp;
        if (F.against) { pr.resize ((size_t) T); for (int t = 0; t < T; ++t) { double gl, gr; at (place (t), gl, gr); pr[(size_t) t] = gl - gr; } }
        else
        {
            const int tau = own ? tauO : tauL; auto it = zt.find (tau); if (it == zt.end()) it = zt.emplace (tau, diffLane (E, tau, std::max (2, tau / 4))).first;
            zq = &it->second; Vec g ((size_t) T); for (int t = 0; t < T; ++t) { double gl, gr; at (place (t), gl, gr); g[(size_t) t] = 0.5 * (gl + gr); }
            pr.resize (zq->size()); for (size_t t = 0; t < pr.size(); ++t) pr[t] = g[t + (size_t) tau] - g[t];
            if (! F.open.empty())                                       // a left-out step as loud as the open ones: the measured movement itself says there is no gate there
            {
                gone.assign (pr.size(), 0); const double Pf = Ps / HOP; double sum = 0; size_t cnt = 0;
                for (size_t t = 0; t < pr.size(); ++t)
                {
                    const double cyc = std::floor ((place ((int) t) / HOP / Pf * F.K - F.start) / F.K);
                    gone[t] = std::find (F.open.begin(), F.open.end(), (int) cyc) != F.open.end(); if (! gone[t]) { sum += std::abs (pr[t]); ++cnt; }
                }
                usual = cnt ? sum / (double) cnt : 0.0;
            }
        }
        const Vec& z = *zq; std::vector<size_t> t0 (nw), t1 (nw);
        for (size_t k = 0; k < nw; ++k) { t0[k] = std::min (z.size(), (W[k].a - a0 + HOP - 1) / HOP); t1[k] = std::min (z.size(), (W[k].b - a0 + HOP - 1) / HOP); }
        Vec zv, pv, dv, tmp;
        for (size_t k = 0; k < nw; ++k)
        {
            Window& w = W[k]; if (t1[k] <= t0[k]) continue;
            w.first = a0 + t0[k] * HOP; w.e.assign (t1[k] - t0[k], 0.0f); double m0 = 0, m1 = 0, mp = 0;
            if (F.against)                                              // the sound's own place between left and right, from the stretches within a cycle either side
            {
                const size_t ka = k >= 2 ? k - 2 : 0, kb = std::min (nw, k + 3); zv.clear(); dv.clear(); pv.clear();
                for (size_t t = t0[ka]; t < t1[kb - 1]; ++t) if (fin (z[t])) { zv.push_back (z[t]); pv.push_back (pr[t]); dv.push_back (z[t] - pr[t]); }
                if (zv.size() < 24) continue;
                tmp = zv; const double ma = med (tmp); tmp = dv; const double mb = med (tmp); double ca = 0, cb = 0;
                for (size_t i = 0; i < zv.size(); ++i) { ca += std::min (std::abs (zv[i] - ma), std::abs (dv[i] - ma)); cb += std::min (std::abs (zv[i] - mb), std::abs (dv[i] - mb)); }
                m0 = m1 = ca <= cb ? ma : mb;
            }
            zv.clear(); pv.clear(); dv.clear(); size_t no = 0;
            for (size_t t = t0[k]; t < t1[k]; ++t) if (fin (z[t])) { if (! gone.empty() && gone[t]) { ++no; continue; } zv.push_back (z[t]); pv.push_back (pr[t]); dv.push_back (z[t] - pr[t]); }
            const size_t cnt = zv.size() + no; if (cnt < 24) continue;
            if (! F.against && zv.size() >= 8) { tmp = zv; m0 = med (tmp); tmp = dv; m1 = med (tmp); tmp = pv; mp = med (tmp); }
            double num = -usual * (double) no, den = usual * (double) no; for (size_t i = 0; i < zv.size(); ++i) { num += std::abs (zv[i] - m0) - std::abs (dv[i] - m1); den += std::abs (pv[i] - mp); }
            taken += num + usual * (double) no;                         // how much of what the stretch does in level steps this answer takes away
            if (den < 0.05 * (double) cnt) continue;                    // the movement hardly moves here: nothing to tell by
            w.u = std::clamp (num / den, -1.0, 1.0); w.w = (double) (w.b - w.a) / (WIN * SR) * (double) cnt / (double) (t1[k] - t0[k]); total += w.w * w.u;
            const double sc = (double) cnt / den;
            for (size_t t = t0[k]; t < t1[k]; ++t) if (fin (z[t])) w.e[t - t0[k]] = (float) (sc * (! gone.empty() && gone[t] ? -usual : std::abs (z[t] - m0) - std::abs (z[t] - pr[t] - m1)));
        }
        return W;
    };
    double so = 0, sl = 0, sw = 0, go = 0, gl = 0; std::vector<Window> Wo, Wl;
    if (rule) Wo = ask (true, so, go);
    if (! rule || (! isLead && M.pieces.size() > 1 && M.clock.t.size() >= 2 && L.lane[0].empty())) Wl = ask (false, sl, gl);
    for (const Window& w : Wl) sw += w.w;
    lead = ! rule || (! Wl.empty() && (sl > so + 0.1 * sw || (L.steps() && gl > 0.0 && gl > 1.08 * go)));
    for (auto& w : lead ? Wl : Wo) out.push_back (std::move (w));
}
// The place between two neighbouring stretches, one with the movement and one without, where the frames change sides
// (leaving: A has it, B has not). Where the frames say nothing the place falls on the side of the stretch with it.
inline size_t edgeAt (const Window& A, const Window& B, bool leaving)
{
    double c = 0, best = 0; size_t pos = A.a;
    for (const Window* w : { &A, &B })
        for (size_t i = 0; i < w->e.size(); ++i)
        {
            c += (double) w->e[i];
            if (leaving ? c > best : c <= best) { best = c; pos = std::min (B.b, w->first + (i + 1) * HOP); }
        }
    return pos;
}
// Which stretches have the movement: the answers that gather the most evidence, where every change between 'there'
// and 'not there' costs `price`; then each change is put where the frames around it say.
inline std::vector<Span> spansOf (std::vector<Window>& W, double price, double lean)
{
    std::vector<Span> out; const size_t m = W.size(); if (m == 0) return out;
    Vec b0 (m), b1 (m); std::vector<char> f0 (m, 0), f1 (m, 1);         // the most that can be gathered up to stretch k when it is not there / there, and the answer before it on that way
    for (size_t k = 0; k < m; ++k)
    {
        const double g = W[k].w * (W[k].u + (W[k].own ? lean : -lean));
        if (k == 0) { b0[0] = 0.0; b1[0] = g; continue; }
        f0[k] = b1[k - 1] - price > b0[k - 1]; b0[k] = std::max (b0[k - 1], b1[k - 1] - price);
        f1[k] = b1[k - 1] >= b0[k - 1] - price; b1[k] = std::max (b1[k - 1], b0[k - 1] - price) + g;
    }
    char st = b1[m - 1] > b0[m - 1]; for (size_t k = m; k-- > 0;) { W[k].on = st != 0; if (k > 0) st = st ? f1[k] : f0[k]; }
    for (size_t k = 0; k < m;)
    {
        if (! W[k].on) { ++k; continue; }
        size_t j = k; while (j + 1 < m && W[j + 1].on) ++j;
        Span sp { k > 0 ? edgeAt (W[k - 1], W[k], false) : W[k].a, j + 1 < m ? edgeAt (W[j], W[j + 1], true) : W[j].b };
        if (sp.b > sp.a + (size_t) (0.25 * SR)) out.push_back (sp);
        k = j + 1;
    }
    return out;
}
// How much of the movement is taken out at a place: all of it inside a stretch that has it, none outside, a short
// fade just inside each of its ends (not at the ends of the sound).
struct There
{
    const std::vector<Span>& sp; size_t n; mutable size_t k = 0;
    double at (size_t i) const
    {
        while (k < sp.size() && sp[k].b <= i) ++k;
        while (k > 0 && sp[k - 1].b > i) --k;
        if (k >= sp.size() || i < sp[k].a) return 0.0;
        const Span& s = sp[k]; const double G = EDGE_GUARD * SR, R = EDGE_RAMP * SR; double w = 1.0;
        if (s.a > 0) w = std::min (w, ((double) i - (double) s.a - G) / R);
        if (s.b < n) w = std::min (w, ((double) s.b - G - (double) i) / R);
        return std::clamp (w, 0.0, 1.0);
    }
};
// One piece, and the movement is on a part of it only: what was measured is that part blurred by the rest. The longest
// stretch that has the movement (PIECE seconds and four cycles at least) is measured again by itself at the found
// period, with the same measures and the same rule. If the rule is surer of it there even with ALONE taken off (a
// stretch picked because it shows the movement shows it more than one picked blindly), the sound is cut into that
// stretch and what lies before and after it, the movement is the one the stretch showed, and every stretch of the
// sound is asked again whether it has that movement (the windows are filled anew; true: something changed).
inline bool narrow (const Stereo& x, Movement& M, const Stop& stop)
{
    const Found* F = M.best(); if (! M.found || F == nullptr || ! F->lane[0].empty() || M.spans.empty()) return false;
    size_t in = 0; Span S; for (const Span& sp : M.spans) { in += sp.b - sp.a; if (sp.b - sp.a > S.b - S.a) S = sp; }
    if ((double) in > 0.85 * (double) M.n || (double) (S.b - S.a) < std::max (PIECE, 4.0 * F->period) * SR) return false;
    const Stereo s = x.slice (S.a, S.b - S.a); Found f = findAt (s, levels (s), F->against, F->period);
    if (stop && stop()) { M.stopped = true; return true; }
    if (! f.found || ! (f.score - ALONE > M.score)) return false;
    std::vector<Piece> ps; Vec pw; double top = 0; int lead = -1;
    for (const Span& c : { Span { 0, S.a }, S, Span { S.b, M.n } })
    {
        if (c.b <= c.a) continue;
        Piece p; p.a = c.a; p.b = c.b; double e = 0; for (size_t i = c.a; i < c.b; ++i) e += x.l[i] * x.l[i] + x.r[i] * x.r[i];
        pw.push_back (e / (double) (c.b - c.a)); top = std::max (top, pw.back());
        if (c.a == S.a && c.b == S.b) { lead = (int) ps.size(); p.own = f.score; p.use = p.guided = true; }
        ps.push_back (std::move (p));
    }
    M.sounding = 0; for (size_t j = 0; j < ps.size(); ++j) { ps[j].sounds = top > 0 && pw[j] > top * 1e-4; M.sounding += ps[j].sounds ? 1 : 0; }
    ps[(size_t) lead].f = std::move (f); M.pieces = std::move (ps); M.lead = lead; M.carrying = 1; const Found& B = *M.best();
    M.against = B.against; M.steps = B.steps(); M.rate = B.rate; M.depth = B.depth; M.score = B.score - ALONE; M.sure = sureness (M.score);
    M.clock = clockOf (M); M.windows.clear();
    for (Piece& p : M.pieces) { if (stop && stop()) { M.stopped = true; return true; } bool byLead = false; askPiece (x, M, p, M.windows, byLead); }
    return true;
}
// x in pieces of `seconds` to twice that; own (piece): what a piece shows by itself.
template <typename Own>
inline Movement lookPieces (const Stereo& x, double seconds, Own own, const Stop& stop, double carry, double price, double lean)
{
    const double none = -std::numeric_limits<double>::infinity();
    Movement M; const size_t n = x.size(); M.n = n; if (n == 0) return M;
    const size_t np = (double) n >= 2.0 * seconds * SR ? (size_t) ((double) n / (seconds * SR)) : 1; M.pieces.resize (np); Vec pw (np, 0.0); double top = 0;
    for (size_t j = 0; j < np; ++j)
    {
        Piece& p = M.pieces[j]; p.a = j * n / np; p.b = (j + 1) * n / np;
        double e = 0; for (size_t i = p.a; i < p.b; ++i) e += x.l[i] * x.l[i] + x.r[i] * x.r[i];
        pw[j] = e / (double) std::max<size_t> (1, p.b - p.a); top = std::max (top, pw[j]);
    }
    for (size_t j = 0; j < np; ++j)
    {
        Piece& p = M.pieces[j]; p.sounds = top > 0 && pw[j] > top * 1e-4; if (! p.sounds) continue;        // a piece 40 dB under the loudest one says nothing
        ++M.sounding; p.f = np == 1 ? own (x) : own (x.slice (p.a, p.b - p.a));
        if (p.f.stopped || (stop && stop())) { M.stopped = true; return M; }
        if (p.f.found) p.own = p.f.score;
    }
    // the movements the pieces found by themselves, surest first (at most three different ones)
    std::vector<size_t> order; for (size_t j = 0; j < np; ++j) if (M.pieces[j].f.found) order.push_back (j);
    std::stable_sort (order.begin(), order.end(), [&] (size_t a, size_t b) { return M.pieces[a].f.score > M.pieces[b].f.score; });
    std::vector<size_t> leads;
    for (size_t j : order)
    {
        bool known = false; for (size_t l : leads) known = known || sameMovement (M.pieces[l].f, M.pieces[j].f);
        if (! known && leads.size() < 3) leads.push_back (j);
    }
    if (leads.empty()) return M;
    // every other sounding piece measured at each of them
    const size_t nc = leads.size(); std::vector<std::vector<Found>> G (nc, std::vector<Found> (np));
    for (size_t j = 0; j < np && np > 1; ++j)
    {
        const Piece& p = M.pieces[j]; if (! p.sounds) continue;
        bool need = false; for (size_t c = 0; c < nc; ++c) need = need || ! sameMovement (M.pieces[leads[c]].f, p.f);
        if (! need) continue;
        const Stereo s = x.slice (p.a, p.b - p.a); const Levels E = levels (s);
        for (size_t c = 0; c < nc; ++c)
        {
            const Found& L = M.pieces[leads[c]].f; if (sameMovement (L, p.f) || (L.against && ! L.lane[0].empty())) continue;
            if (stop && stop()) { M.stopped = true; return M; }
            G[c][j] = findAt (s, E, L.against, L.period);
        }
    }
    // the movement whose carrying pieces have the best middle score
    size_t bc = 0; double bs = none;
    for (size_t c = 0; c < nc; ++c)
    {
        const Found& L = M.pieces[leads[c]].f; Vec v;
        for (size_t j = 0; j < np; ++j)
        {
            const Piece& p = M.pieces[j]; if (! p.sounds) continue;
            const Found& f = sameMovement (L, p.f) ? p.f : G[c][j];
            if (f.score >= carry && sameMovement (L, f)) v.push_back (f.score);
        }
        const double sc = (v.empty() ? none : med (v)) - (v.size() < 2 && M.sounding >= 2 ? ALONE : 0.0);
        if (c == 0 || sc > bs) { bc = c; bs = sc; }
    }
    const Found L = M.pieces[leads[bc]].f;
    for (size_t j = 0; j < np; ++j)
    {
        Piece& p = M.pieces[j]; if (! p.sounds) continue;
        if (! sameMovement (L, p.f)) { p.f = std::move (G[bc][j]); p.guided = true; }
        p.use = p.f.score >= carry && sameMovement (L, p.f); if (! p.use) continue;
        ++M.carrying; if (M.lead < 0 || p.f.score > M.pieces[(size_t) M.lead].f.score) M.lead = (int) j;
    }
    if (M.lead < 0) return M;
    const Found& B = M.pieces[(size_t) M.lead].f;
    M.found = true; M.against = B.against; M.steps = B.steps(); M.rate = B.rate; M.depth = B.depth; M.score = bs; M.sure = sureness (bs);
    M.clock = clockOf (M);
    for (Piece& p : M.pieces)                                           // where the movement is
    {
        if (stop && stop()) { M.stopped = true; return M; }
        bool lead = false; askPiece (x, M, p, M.windows, lead); if (p.use && lead) p.clocked = true;
    }
    M.spans = spansOf (M.windows, price, lean);
    if (np == 1 && narrow (x, M, stop)) { if (M.stopped) return M; M.spans = spansOf (M.windows, price, lean); }
    if (const Found* B = M.best(); B != nullptr && B->against && B->pan.adj < PAN_ADJ_MIN)      // the long look's hint did not hold on a stretch of its own (see findSlowPan): no find
    {
        for (Piece& p : M.pieces) { Piece q; q.a = p.a; q.b = p.b; q.sounds = p.sounds; p = std::move (q); }
        M.found = M.against = M.steps = false; M.score = none; M.sure = M.rate = M.depth = 0; M.lead = -1; M.carrying = 0; M.clock = Clock(); M.windows.clear(); M.spans.clear();
        return M;
    }
    for (Window& w : M.windows) { w.e.clear(); w.e.shrink_to_fit(); }
    for (Piece& p : M.pieces)
    {
        size_t in = 0; for (const Span& sp : M.spans) { const size_t a = std::max (sp.a, p.a), b = std::min (sp.b, p.b); if (b > a) in += b - a; }
        p.there = (double) in / (double) std::max<size_t> (1, p.b - p.a);
        if (p.sounds && ! p.use && p.there >= 0.5) { p.use = p.clocked = true; ++M.carrying; }
    }
    return M;
}
inline Movement analyse (const Stereo& x, const Stop& stop = {}, double carry = CARRY, double price = SWITCH, double lean = LEAN)
{
    const bool isLong = (double) x.size() >= 2.0 * PIECE * SR;
    Movement A = lookPieces (x, PIECE, [&] (const Stereo& s) { return find (s, stop, SCORE_MIN, ! isLong); }, stop, carry, price, lean);
    if (! isLong || A.stopped) return A;
    Movement B = lookPieces (x, LONG, [&] (const Stereo& s) { return findSlowPan (s, stop, PIECE / 4.0); }, stop, carry, price, lean);
    if (B.stopped) return B;
    B.wide = true;
    const bool a = A.found && A.score >= 0.0, b = B.found && B.score >= 0.0;       // as in find(): a sure pan before a shared movement
    if (a && b) return A.against && A.score > B.score ? A : B;
    if (a || b) return a ? A : B;
    if (A.found && B.found) return A.score >= B.score ? A : B;
    return B.found ? B : A;
}
// The gain in dB for the sound the movement was found on, as its faders ask (see faderDb), in the stretches that have
// the movement (spans) and nowhere else: in a piece that carries the movement by the rule its own measured movement,
// in any other piece the surest piece's on the clock; from one piece to the next the two are blended over 20 ms. At
// another speed the found movement comes out like that and the surest piece's movement goes back in at that speed, as
// one movement through every stretch that had it (no jump in its beat at the piece borders).
inline void ownDb (const Movement& M, size_t n, double amount, double speed, Vec& gl, Vec& gr)
{
    gl.assign (n, 0.0); gr.assign (n, 0.0); const Found* F = M.best(); if (! M.found || F == nullptr) return;
    const bool same = std::abs (speed - 1.0) < 1e-9; const long X = 480; Vec pl, pr, cover; if (! same) cover.assign (n, 0.0);
    const double s = same ? amount - 1.0 : -1.0, Ps = F->period * SR; const GainAt lead (*F); const There there { M.spans, M.n };
    for (const Piece& p : M.pieces)
    {
        if (p.a >= n) continue;
        const long lo = p.a == 0 ? 0 : (long) p.a - X, hi = std::min ((long) n, p.b >= M.n ? (long) n : (long) p.b + X); if (hi <= lo) continue;
        bool any = false; for (const Span& sp : M.spans) any = any || ((long) sp.a < hi && (long) sp.b > lo);
        if (! any) continue;
        if (p.use && ! p.clocked) strengthDb (p.f, (size_t) (hi - lo), s, pl, pr, 1.0, (double) (lo - (long) p.a));
        else
        {
            pl.resize ((size_t) (hi - lo)); pr.resize ((size_t) (hi - lo)); for (long i = lo; i < hi; ++i) lead (M.clock.at ((double) i) * Ps, pl[(size_t) (i - lo)], pr[(size_t) (i - lo)]);
            shapeDb (*F, s, pl, pr);
        }
        for (long i = lo; i < hi; ++i)
        {
            double w = there.at ((size_t) i); if (w <= 0.0) continue;
            if (p.a > 0 && i < (long) p.a + X) w *= (double) (i - ((long) p.a - X)) / (2.0 * X);
            if (p.b < M.n && i >= (long) p.b - X) w *= (double) ((long) p.b + X - i) / (2.0 * X);
            gl[(size_t) i] += w * pl[(size_t) (i - lo)]; gr[(size_t) i] += w * pr[(size_t) (i - lo)]; if (! same) cover[(size_t) i] += w;
        }
    }
    if (same) return;
    for (size_t i = 0; i < n; ++i) { if (cover[i] <= 0.0) continue; double al, ar; lead (M.clock.at ((double) i * speed) * Ps, al, ar); gl[i] += amount * cover[i] * al; gr[i] += amount * cover[i] * ar; }
}
// The gain in dB that puts the movement onto another sound: the surest piece's shape on the first sound's clock, at
// the times at which the first sound has the movement (spans) and at no others: a movement that comes in half way
// through the first sound comes in half way through the second. Past the first sound's end it goes on as it was at
// that end. amount: 0 = none, 1 = as found, 2 = twice as deep. speed: 1 = as found.
inline void ontoDb (const Movement& M, size_t n, double amount, double speed, Vec& gl, Vec& gr)
{
    gl.assign (n, 0.0); gr.assign (n, 0.0); const Found* F = M.best(); if (! M.found || F == nullptr || M.n == 0) return;
    const double s = std::max (amount, 0.0);
    if (M.pieces.size() == 1) strengthDb (*F, n, s, gl, gr, speed, 0.0);
    else { const GainAt at (*F); const double Ps = F->period * SR; for (size_t i = 0; i < n; ++i) { at (M.clock.at ((double) i * speed) * Ps, gl[i], gr[i]); gl[i] *= s; gr[i] *= s; } }
    const There there { M.spans, M.n };
    for (size_t i = 0; i < n; ++i) { const double w = there.at (std::min (i, M.n - 1)); gl[i] *= w; gr[i] *= w; }
}

} // namespace level
} // namespace dr
