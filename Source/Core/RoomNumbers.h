#pragma once
// The room in numbers, read off a sound with the help of the model's answer ("room or sound" per band
// and moment): per band a decay time (straight-line fits to the level in the stretches the answer
// calls "all room": the free decays) and a strength (how loud those stretches are next to the direct
// sound smeared out with that decay time). Port of the prototype's roomnum.py. Plain C++17, no JUCE.
// LE is the level picture and g the answer, both [frame * NB + band], T frames.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "Room.h"

namespace dr
{
namespace room
{
using Band = std::array<double, NB>;        // one number per band
constexpr double BUILD_UP_DB = 2.3;         // a real room builds up before it decays: a plain exponential that matches the tail holds about this much more energy
constexpr double ALL_ROOM_DB = -10.0;       // a cell counts as "all room" when the answer is under this, as holding direct sound when above
constexpr double FLOOR_DB = 57.0;           // cells further than this under the loudest one are not looked at

using Weighted = std::vector<std::pair<double, double>>;       // (value, weight)
// The middle of some values by weight: in rising order, the first at which the weight so far reaches half of all.
inline double middle (Weighted& v)
{
    double all = 0.0, sofar = 0.0; for (const auto& p : v) all += p.second;
    std::sort (v.begin(), v.end(), [] (const std::pair<double, double>& a, const std::pair<double, double>& b) { return a.first < b.first; });
    for (const auto& p : v) { sofar += p.second; if (sofar / all >= 0.5) return p.first; }
    return v.back().first;
}
inline bool any (const Band& v) { for (double x : v) if (std::isfinite (x)) return true; return false; }
inline double middle (Vec v)            // of an even count: half way between the two in the middle
{
    std::sort (v.begin(), v.end()); const size_t n = v.size();
    return n == 0 ? 0.0 : n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

// No band may stray further above the middle of the spectrum (450 Hz .. 7 kHz) than real reverbs do: a band that
// reads far beyond that has been misread. Decay time: at most capT times the mid-band middle; strength: at most
// capK dB above it.
inline bool midBand (double fc) { return fc > 450.0 && fc < 7000.0; }
inline double capT (double fc) { return fc < 450.0 ? 5.0 : fc <= 7000.0 ? 2.5 : 1.6; }
inline double capK (double fc) { return fc < 450.0 ? 12.0 : fc <= 7000.0 ? 6.0 : 4.0; }
// The middle of v over the mid bands that have evidence of their own; if none has, over all bands that have; if none, over all.
inline double mid (const Band& v, const Band& ev, const Bands& B)
{
    Vec a, b, c;
    for (size_t i = 0; i < NB; ++i) { c.push_back (v[i]); if (ev[i] > 0.0) { b.push_back (v[i]); if (midBand (B.fc[i])) a.push_back (v[i]); } }
    return middle (! a.empty() ? a : ! b.empty() ? b : c);
}

// The free decays of band b: (slope in dB/s, weight) for every run of at least minlen frames with the answer
// under gmax and the level above the floor fl. The first two frames (still holding the end of the direct sound)
// are left out. A decay counts more the further it falls and the straighter it is.
inline Weighted decays (const Vec& LE, const std::vector<float>& g, int T, int b, double fl, double gmax, int minlen)
{
    Weighted out; const int skip = 2;
    auto le = [&] (int t) { return LE[(size_t) t * NB + (size_t) b]; };
    auto in = [&] (int t) { return (double) g[(size_t) t * NB + (size_t) b] < gmax && le (t) > fl; };
    for (int t = 0; t < T; ++t)
    {
        if (! in (t)) continue;
        int e = t + 1; while (e < T && in (e)) ++e;         // the run is frames t .. e - 1
        const int a = t + skip, n = e - a; t = e;
        if (n < minlen) continue;
        double mx = 0.0, my = 0.0, sxx = 0.0, sxy = 0.0, res = 0.0;        // the least-squares line through the levels
        for (int i = a; i < e; ++i) { mx += i * DT; my += le (i); }
        mx /= n; my /= n;
        for (int i = a; i < e; ++i) { const double dx = i * DT - mx; sxx += dx * dx; sxy += dx * (le (i) - my); }
        const double slope = sxy / sxx, at0 = my - slope * mx;
        for (int i = a; i < e; ++i) { const double d = slope * (i * DT) + at0 - le (i); res += d * d; }
        const double fall = -slope * ((e - 1) * DT - a * DT);
        if (slope < -3.0 && fall > 6.0) out.push_back ({ slope, fall / (1.0 + std::sqrt (res / n)) });
    }
    return out;
}

// Decay time per band (s; not a number where the room never shows) and how much evidence each band has. If fewer
// than four bands qualify, the rules are relaxed step by step (shorter stretches, less clearly "all room").
inline void decayTimes (const Vec& LE, const std::vector<float>& g, int T, double top, Band& t60, Band& ev)
{
    const double gmax[3] = { ALL_ROOM_DB, -7.0, -4.0 }; const int minlen[3] = { 8, 6, 5 };
    for (int step = 0; step < 3; ++step)
    {
        int have = 0;
        for (int b = 0; b < NB; ++b)
        {
            Weighted d = decays (LE, g, T, b, top - FLOOR_DB, gmax[step], minlen[step]);
            t60[(size_t) b] = std::numeric_limits<double>::quiet_NaN(); ev[(size_t) b] = 0.0;
            if (d.empty()) continue;
            for (const auto& p : d) ev[(size_t) b] += p.second;
            t60[(size_t) b] = -60.0 / middle (d); ++have;
        }
        if (have >= 4) break;
    }
}

// Per-band numbers made to agree with their neighbours (the middle by evidence over +-n bands) and filled where a
// band has none (straight lines between the nearest bands that have, flat beyond the ends). All not a number: no
// band had any. log: the numbers are ratios and are treated as such.
inline Band smoothFill (const Band& v, const Band& ev, int n = 2, bool log = true)
{
    const double none = std::numeric_limits<double>::quiet_NaN();
    Band x, sm, out; bool ok[NB]; Weighted nearby; int first = -1, last = -1;
    for (int b = 0; b < NB; ++b) { const size_t i = (size_t) b; x[i] = log ? std::log (v[i]) : v[i]; ok[b] = std::isfinite (x[i]) && ev[i] > 0.0; }
    for (int b = 0; b < NB; ++b)
    {
        nearby.clear();
        for (int j = std::max (0, b - n); j < std::min (NB, b + n + 1); ++j) if (ok[j]) nearby.push_back ({ x[(size_t) j], ev[(size_t) j] });
        sm[(size_t) b] = nearby.empty() ? none : middle (nearby);
        if (! nearby.empty()) { if (first < 0) first = b; last = b; }
    }
    if (first < 0) { out.fill (none); return out; }
    for (int b = 0, lo = first; b < NB; ++b)
    {
        double& o = out[(size_t) b];
        if (b <= first || b >= last) o = sm[(size_t) (b <= first ? first : last)];
        else if (std::isfinite (sm[(size_t) b])) { lo = b; o = sm[(size_t) b]; }
        else
        {
            int hi = b + 1; while (! std::isfinite (sm[(size_t) hi])) ++hi;
            o = (sm[(size_t) hi] - sm[(size_t) lo]) / (double) (hi - lo) * (double) (b - lo) + sm[(size_t) lo];
        }
        if (log) o = std::exp (o);
    }
    return out;
}

// The direct sound of each band (where the answer says there is some) smeared out with the band's decay time:
// what a room with that decay and strength 1 would hold at every moment. [frame * NB + band]
inline Vec smear (const Vec& LE, const std::vector<float>& g, int T, double top, const Band& t60)
{
    Vec P ((size_t) T * NB, 0.0); const double fl = top - FLOOR_DB;
    for (int b = 0; b < NB; ++b)
    {
        if (! std::isfinite (t60[(size_t) b])) continue;
        const double r = std::pow (10.0, -6.0 * DT / t60[(size_t) b]); double acc = 0.0;
        for (int t = 0; t < T; ++t)
        {
            const size_t i = (size_t) t * NB + (size_t) b; const double le = LE[i], gg = (double) g[i];
            const double direct = gg > ALL_ROOM_DB && le + gg > fl + 5.0 ? std::pow (10.0, le / 10.0) * std::pow (10.0, gg / 10.0) : 0.0;
            acc = r * acc + (1.0 - r) * direct; P[i] = acc;
        }
    }
    return P;
}

// Per band: the strength (reverb energy per unit of direct energy, for a plain exponential decay with decay time
// t60) and the evidence (how many "all room" moments the band had). In those moments the level is the strength
// times the direct sound smeared with that decay: the middle of that ratio, the outermost tenths left out.
inline void strengths (const Vec& LE, const std::vector<float>& g, int T, double top, const Band& t60, Band& kap, Band& ev)
{
    const Vec P = smear (LE, g, T, top, t60); const double fl = top - FLOOR_DB; Vec q;
    for (int b = 0; b < NB; ++b)
    {
        kap[(size_t) b] = std::numeric_limits<double>::quiet_NaN(); ev[(size_t) b] = 0.0;
        double most = 0.0; for (int t = 0; t < T; ++t) most = std::max (most, P[(size_t) t * NB + (size_t) b]);
        if (! (most > 0.0)) continue;
        q.clear();
        for (int t = 0; t < T; ++t)
        {
            const size_t i = (size_t) t * NB + (size_t) b;
            if ((double) g[i] < ALL_ROOM_DB && LE[i] > fl && P[i] > 1e-4 * most) q.push_back (std::log (std::pow (10.0, LE[i] / 10.0) / P[i]));
        }
        if (q.size() < 8) continue;
        std::sort (q.begin(), q.end());
        const size_t k = (size_t) ((double) q.size() * 0.1), from = q.size() - 2 * k >= 4 ? k : 0, to = q.size() - from;
        double sum = 0.0; for (size_t i = from; i < to; ++i) sum += q[i];
        kap[(size_t) b] = std::exp (sum / (double) (to - from)); ev[(size_t) b] = (double) q.size();
    }
}

// (strength in dB, decay time in s, evidence) per band, smoothed over neighbouring bands, filled and capped; found =
// false if no room shows. The strength is that of the whole response (the tail level minus the build-up allowance).
struct Numbers { bool found = false; Band kdb {}, t60 {}, ev {}; };
inline Numbers roomNumbers (const Vec& LE, const std::vector<float>& g, int T)
{
    Numbers N; if (T < 1 || LE.size() != (size_t) T * NB || g.size() != LE.size()) return N;
    const double top = *std::max_element (LE.begin(), LE.end()); const Bands B; Band raw, ev, kap;
    decayTimes (LE, g, T, top, raw, ev); if (! any (raw)) return N;
    Band t60 = smoothFill (raw, ev); const double midT = mid (t60, ev, B);
    for (size_t b = 0; b < NB; ++b) t60[b] = std::min (t60[b], midT * capT (B.fc[b]));
    strengths (LE, g, T, top, t60, kap, ev); if (! any (kap)) return N;
    Band kdb = smoothFill (kap, ev); for (double& k : kdb) k = 10.0 * std::log10 (k) - BUILD_UP_DB;
    const double midK = mid (kdb, ev, B);
    for (size_t b = 0; b < NB; ++b) N.kdb[b] = std::min (kdb[b], midK + capK (B.fc[b]));
    N.t60 = t60; N.ev = ev; N.found = true;
    return N;
}

// Three more pictures for the model's second pass, made from its first answer g: (1) how loud the room should be
// at every band and moment next to what is there (from the found decay time and strength), (2) the found decay
// time and (3) the found strength, each band's number held over time. No room found: all three say so (-1).
// [plane][frame][band], to follow the four planes of inputs().
inline std::vector<float> secondLook (const Vec& LE, const std::vector<float>& g, int T)
{
    const size_t n = (size_t) T * NB; std::vector<float> x (3 * n, -1.0f);
    const Numbers N = roomNumbers (LE, g, T); if (! N.found) return x;
    const Vec P = smear (LE, g, T, *std::max_element (LE.begin(), LE.end()), N.t60);
    for (size_t b = 0; b < NB; ++b)
    {
        const double k = std::pow (10.0, (N.kdb[b] + BUILD_UP_DB) / 10.0);
        const float howLong = (float) (std::clamp (std::log2 (N.t60[b]), -3.3, 3.3) / 3.3), howStrong = (float) (std::clamp (N.kdb[b], -30.0, 20.0) / 25.0 + 0.2);
        for (size_t i = b; i < n; i += NB)
        {
            const double share = 10.0 * std::log10 (k * P[i] / std::max (std::pow (10.0, LE[i] / 10.0), 1e-30) + 1e-9);
            x[i] = (float) (std::clamp (share, -30.0, 10.0) / 20.0 + 0.5); x[n + i] = howLong; x[2 * n + i] = howStrong;
        }
    }
    return x;
}

} // namespace room
} // namespace dr
