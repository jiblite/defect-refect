#pragma once
// The reverb engine as the app runs it (port of the prototype's engine.py); this is the one file to include.
//
//   room::Found f = room::find (y, net);                     what was found on a sound: the "room or sound" answer and the room in numbers
//   Stereo ir  = room::roomIr (f, length, bright, width);    a stereo impulse response of the found room (what Export IR writes)
//   Stereo out = room::oneSound (y, f, amount, ...);         the sound with its own reverb turned down / up, and reshaped (length, brightness, width)
//   Stereo out = room::onto (b, f, amount, ...);             the found room put on a second sound
//   Stereo wet = room::convolved (b, ir);                    the sound through the room alone, to be mixed by the caller
//
// 48 kHz stereo. Every call builds the few tables it needs itself (a fraction of a millisecond): nothing is
// kept between calls, so any of them can run on any thread at any time. Plain C++17, no JUCE.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "RoomNumbers.h"

namespace dr
{
namespace room
{
// ---------------------------------------------------------------- what is found on a sound
struct Found
{
    bool stopped = false;               // given up on the way (see Stop): nothing below is complete
    Picture picture;                    // what the model saw
    std::vector<float> gains;           // its answer, dB per frame and band (after the second pass, if there is one)
    bool found = false;                 // a room shows: only then do the numbers below mean anything
    Band kdb {}, t60 {}, ev {}, width {};       // per band: strength (dB), decay time (s), evidence (moments that were all room), width of the tail (0 = mono .. 1 = fully wide)
};

// Everything that is worked out once per sound. `second` is a model for a second pass (7 input planes: the 4 of
// the first pass and the 3 that secondLook() makes from the first answer); its answer then replaces the first.
// `piece`: see Net::run. Meant for a worker thread: `stop` is asked every few hundred frames of the analysis and
// before every level of the model. Throws std::runtime_error (as Net::run does) when a model is not loaded or is
// made for another number of input planes.
inline Found find (const Stereo& y, const Net& net, const Net* second = nullptr, const Stop& stop = {}, const Progress& progress = {}, int piece = 4096)
{
    Found f; const double share[3] = { second != nullptr ? 0.2 : 0.3, second != nullptr ? 0.4 : 0.7, 0.4 };     // of the time: analysis, model, second model
    auto part = [&] (double from, double len) { return progress ? Progress ([&progress, from, len] (double p) { progress (from + len * p); }) : Progress(); };
    f.picture = analyse (y, stop, part (0.0, share[0])); const int T = f.picture.frames;
    if (T == 0) { f.stopped = true; return f; }
    std::vector<float> x = inputs (f.picture);
    f.gains = net.run (x, T, piece, stop, part (share[0], share[1]));
    if (second != nullptr && ! f.gains.empty())
    {
        const std::vector<float> more = secondLook (f.picture.LE, f.gains, T); x.insert (x.end(), more.begin(), more.end());
        f.gains = second->run (x, T, piece, stop, part (share[0] + share[1], share[2]));
    }
    if (f.gains.empty()) { f.stopped = true; return f; }

    const Numbers N = roomNumbers (f.picture.LE, f.gains, T); f.found = N.found;
    if (f.found)
    {
        f.kdb = N.kdb; f.t60 = N.t60; f.ev = N.ev;
        // How wide the tail is: per band the middle unshared share (256 ms) of the cells that are clearly room and
        // not too quiet; bands with fewer than ten such cells take the middle of the other bands.
        const double top = f.picture.top(), none = std::numeric_limits<double>::quiet_NaN(); Band u; Vec v, got;
        for (size_t b = 0; b < NB; ++b)
        {
            v.clear();
            for (size_t i = b; i < f.gains.size(); i += NB)
                if ((double) f.gains[i] < -12.0 && f.picture.LE[i] > top - 50.0) v.push_back (std::pow (10.0, std::clamp (f.picture.U256[i], -40.0, 0.0) / 10.0));
            u[b] = v.size() >= 10 ? middle (v) : none;
            if (v.size() >= 10) got.push_back (u[b]);
        }
        const double rest = got.empty() ? 1.0 : middle (got);
        for (size_t b = 0; b < NB; ++b) f.width[b] = std::clamp ((std::isfinite (u[b]) ? u[b] : rest) * 1.15, 0.0, 1.0);
    }
    if (progress) progress (1.0);
    return f;
}

// ---------------------------------------------------------------- the impulse response
// n numbers from a bell curve that are the same on every machine: a counter goes through a 64-bit mixer
// (splitmix64, arithmetic that wraps), and pairs of the results are turned into normal numbers (Box-Muller).
inline Vec gauss (size_t n, uint64_t seed)
{
    Vec out (n);
    auto unit = [seed] (uint64_t i)       // in (0, 1]
    {
        uint64_t z = seed + i * 0x9E3779B97F4A7C15ULL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; z ^= z >> 31;
        return ((double) (z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    };
    for (size_t i = 0; i < n; i += 2)
    {
        const double r = std::sqrt (-2.0 * std::log (unit (i + 1))), a = 2.0 * PI * unit (i + 2);
        out[i] = r * std::cos (a); if (i + 1 < n) out[i + 1] = r * std::sin (a);
    }
    return out;
}

// An energy response: per band the energy that arrives `lag` frames after the direct sound. [band * lags + lag]
struct Decay { int lags = 0; Vec w; };
constexpr int LAGS_MAX = 1500;          // the longest response: 16 s

// Per band a plain exponential decay with decay time t60 (s) holding 10^(kdb/10) in all; long enough for the
// slowest band to fall by dropDb.
inline Decay expo (const Band& kdb, const Band& t60, double dropDb = 70.0)
{
    double slowest = -std::numeric_limits<double>::infinity(); for (double t : t60) slowest = std::max (slowest, std::isfinite (t) ? t : 0.05);
    Decay d; d.lags = (int) std::clamp (std::ceil (slowest * dropDb / 60.0 / DT) + 8.0, 24.0, (double) LAGS_MAX); d.w.assign ((size_t) NB * (size_t) d.lags, 0.0);
    for (size_t b = 0; b < NB; ++b)
    {
        if (! std::isfinite (t60[b]) || ! std::isfinite (kdb[b])) continue;
        const double r = std::pow (10.0, -6.0 * DT / std::max (t60[b], 0.03)), all = std::pow (10.0, kdb[b] / 10.0); double* w = &d.w[b * (size_t) d.lags]; double sum = 0.0;
        for (int k = 0; k < d.lags; ++k) { w[k] = std::pow (r, (double) k); sum += w[k]; }
        for (int k = 0; k < d.lags; ++k) w[k] = all * w[k] / sum;
    }
    return d;
}

// A stereo tail whose band power follows the energy response lag by lag: noise shaped frame by frame. The two
// channels share a part of the noise so that the unshared share of the tail is `unshared` (0 .. 1 per band).
// The tail starts one frame step (10.7 ms) after the direct sound, which is not part of it.
inline Stereo synthIr (const Decay& W, const Band& unshared, uint64_t seed = 1)
{
    const int L = W.lags; if (L < 1 || W.w.size() != (size_t) NB * (size_t) L) return {};
    const Bands B; const Transform F; const Vec win = window();
    const int n = L * HOP + NF, T = frames ((size_t) n), cut = B.edge[NB], c0 = 3 * NF / (2 * HOP);      // bins from `cut` up stay silent; a click at sample NF sits in the middle of frame c0
    std::vector<size_t> of ((size_t) cut, 0);                       // the band each bin belongs to (the bins under the first band go with it)
    for (int b = 0; b < NB; ++b) for (int k = B.edge[b]; k < B.edge[b + 1]; ++k) of[(size_t) k] = (size_t) b;
    // A unit click lies under four windows: its power in one channel and one band, all frames together.
    double click = 0.0; for (int j = NF - HOP; j >= 0; j -= HOP) click += win[(size_t) j] * win[(size_t) j];
    Band pc, mixSame, mixOwn; for (size_t b = 0; b < NB; ++b) { pc[b] = (B.edge[b + 1] - B.edge[b]) * click; const double u = std::clamp (unshared[b], 0.0, 1.0); mixSame[b] = std::sqrt (1.0 - u); mixOwn[b] = std::sqrt (u); }

    Vec re (NF), im (NF), cr (NF), ci (NF);
    auto silenceAbove = [&] { std::fill (re.begin() + cut, re.end() - (cut - 1), 0.0); std::fill (im.begin() + cut, im.end() - (cut - 1), 0.0); };

    // 1. Noise with the wanted left / right relation, each frame's bands brought to the level the response asks for.
    Stereo own, same, h; own.l = gauss ((size_t) n, seed + 1000003); own.r = gauss ((size_t) n, seed + 2000003); same.l = gauss ((size_t) n, seed); same.r.assign ((size_t) n, 0.0);
    h.l.assign ((size_t) n, 0.0); h.r.assign ((size_t) n, 0.0);
    Vec xl ((size_t) (2 * cut)), xr ((size_t) (2 * cut));
    for (int t = c0; t < std::min (T, c0 + L); ++t)
    {
        frame (own, t, win, re, im); F.run (re, im); frame (same, t, win, cr, ci); F.run (cr, ci);
        double p[2][NB] = {};
        for (int k = 0; k < cut; ++k)
        {
            const size_t a = (size_t) k, b = of[a]; const Bin x = bin (re, im, k);
            xl[2 * a] = mixSame[b] * cr[a] + mixOwn[b] * x.lr; xl[2 * a + 1] = mixSame[b] * ci[a] + mixOwn[b] * x.li;
            xr[2 * a] = mixSame[b] * cr[a] + mixOwn[b] * x.rr; xr[2 * a + 1] = mixSame[b] * ci[a] + mixOwn[b] * x.ri;
            if (k >= B.edge[0]) { p[0][b] += xl[2 * a] * xl[2 * a] + xl[2 * a + 1] * xl[2 * a + 1]; p[1][b] += xr[2 * a] * xr[2 * a] + xr[2 * a + 1] * xr[2 * a + 1]; }
        }
        for (size_t b = 0; b < NB; ++b) for (int c = 0; c < 2; ++c) p[c][b] = std::sqrt (W.w[b * (size_t) L + (size_t) (t - c0)] * pc[b] / std::max (p[c][b], 1e-30));
        for (int k = 0; k < cut; ++k) { const size_t a = (size_t) k, b = of[a]; setBin (re, im, k, xl[2 * a] * p[0][b], xl[2 * a + 1] * p[0][b], xr[2 * a] * p[1][b], xr[2 * a + 1] * p[1][b]); }
        silenceAbove(); F.back (re, im); overlap (h, t, win, re, im);
    }
    overlapDone (h, win);

    // 2. Overlapping frames blur the shaping a little, so each band is brought to its exact energy (the response's sum times a unit click's).
    double have[2][NB] = {};
    for (int t = 0; t < T; ++t)
    {
        if (! frame (h, t, win, re, im)) continue;
        F.run (re, im);
        for (int k = B.edge[0]; k < cut; ++k) { const Bin x = bin (re, im, k); have[0][of[(size_t) k]] += x.lr * x.lr + x.li * x.li; have[1][of[(size_t) k]] += x.rr * x.rr + x.ri * x.ri; }
    }
    for (size_t b = 0; b < NB; ++b)
    {
        double want = 0.0; for (int k = 0; k < std::min (L, T - c0); ++k) want += W.w[b * (size_t) L + (size_t) k];
        for (int c = 0; c < 2; ++c) have[c][b] = std::sqrt (want * pc[b] / std::max (have[c][b], 1e-30));
    }
    Stereo out; out.l.assign ((size_t) n, 0.0); out.r.assign ((size_t) n, 0.0);
    for (int t = 0; t < T; ++t)
    {
        if (! frame (h, t, win, re, im)) continue;
        F.run (re, im);
        for (int k = 0; k < cut; ++k) { const size_t b = of[(size_t) k]; const Bin x = bin (re, im, k); setBin (re, im, k, x.lr * have[0][b], x.li * have[0][b], x.rr * have[1][b], x.ri * have[1][b]); }
        silenceAbove(); F.back (re, im); overlap (out, t, win, re, im);
    }
    overlapDone (out, win);
    out.l.erase (out.l.begin(), out.l.begin() + (NF - HOP)); out.r.erase (out.r.begin(), out.r.begin() + (NF - HOP));
    return out;
}

// The found numbers with the faders applied (each 0 .. 2, 1 = as found): Length scales the decay time (the level
// right after the direct sound stays where it was: a longer room holds more energy in all, a shorter one less);
// Brightness tilts the decay time and the strength of the upper bands against the lower ones (around 1 kHz);
// Width scales the tail's width.
struct Shaped { Band kdb, t60, width; };
inline Shaped shaped (const Found& f, double length = 1.0, double bright = 1.0, double width = 1.0)
{
    const Bands B; Shaped s; const double tilt = bright - 1.0;
    for (size_t b = 0; b < NB; ++b)
    {
        const double oct = std::log2 (B.fc[b] / 1000.0);
        s.t60[b] = std::clamp (f.t60[b] * std::max (length, 0.02) * std::pow (2.0, 0.35 * tilt * oct), 0.03, 20.0);
        const double r0 = std::pow (10.0, -6.0 * DT / f.t60[b]), r1 = std::pow (10.0, -6.0 * DT / s.t60[b]);
        s.kdb[b] = f.kdb[b] + 3.0 * tilt * oct + 10.0 * std::log10 ((1.0 - r0) / (1.0 - r1)); s.width[b] = std::clamp (f.width[b] * width, 0.0, 1.0);
    }
    return s;
}
// The found room as a stereo impulse response (empty if no room was found). The same seed gives the same noise.
inline Stereo roomIr (const Found& f, double length = 1.0, double bright = 1.0, double width = 1.0, uint64_t seed = 1)
{
    if (! f.found) return {};
    const Shaped s = shaped (f, length, bright, width); return synthIr (expo (s.kdb, s.t60), s.width, seed);
}

// ---------------------------------------------------------------- the room on a sound
// Each channel of x through its own channel of the tail h, cut to the length of x: the room's part alone.
// A long sound goes through in pieces: piece and tail are turned into spectra, multiplied and turned back.
// Stopped on the way (asked before every piece): comes back empty.
inline Stereo convolved (const Stereo& x, const Stereo& h, const Stop& stop = {})
{
    const size_t n = x.size(), m = h.size(); Stereo out; out.l.assign (n, 0.0); out.r.assign (n, 0.0);
    if (n == 0 || m == 0) return out;
    if (m > ((size_t) 1 << 28)) throw std::runtime_error ("room: the tail is too long");
    const size_t M = nextPow2 (std::max ((size_t) 2, std::min (n + m - 1, 2 * m))), step = M - m + 1;      // points per transform; samples of x per piece
    const Transform F ((int) M, false);
    Vec hr (M, 0.0), hi (M, 0.0), re (M), im (M);
    std::copy (h.l.begin(), h.l.end(), hr.begin()); std::copy (h.r.begin(), h.r.end(), hi.begin());
    F.spread (hr.data(), hi.data());
    // Left rides as the real and right as the imaginary part. In bit-reversed order a bin and its mirror image
    // lie at mirrored places p, q of a power-of-two stretch; from the two, left and right come apart:
    // left = (Z[p] + conj Z[q]) / 2, right = (Z[p] - conj Z[q]) / 2i. The tail is kept that way (left at p, right
    // at q), with the scale of the way back.
    const double sc = 1.0 / (double) M;
    hr[0] *= sc; hi[0] *= sc; hr[1] *= sc; hi[1] *= sc;
    for (size_t lo = 2; lo < M; lo *= 2)
        for (size_t p = lo, q = 2 * lo - 1; p < q; ++p, --q)
        {
            const double lr = 0.5 * (hr[p] + hr[q]), li = 0.5 * (hi[p] - hi[q]), rr = 0.5 * (hi[p] + hi[q]), ri = 0.5 * (hr[q] - hr[p]);
            hr[p] = lr * sc; hi[p] = li * sc; hr[q] = rr * sc; hi[q] = ri * sc;
        }
    for (size_t at = 0; at < n; at += step)
    {
        if (stop && stop()) return {};
        const size_t len = std::min (step, n - at); bool sound = false;
        for (size_t i = 0; i < len; ++i) { re[i] = x.l[at + i]; im[i] = x.r[at + i]; sound = sound || re[i] != 0.0 || im[i] != 0.0; }
        if (! sound) continue;
        std::fill (re.begin() + (std::ptrdiff_t) len, re.end(), 0.0); std::fill (im.begin() + (std::ptrdiff_t) len, im.end(), 0.0);
        F.spread (re.data(), im.data());
        re[0] *= hr[0]; im[0] *= hi[0]; re[1] *= hr[1]; im[1] *= hi[1];       // bins 0 and M/2 are their own mirror images
        for (size_t lo = 2; lo < M; lo *= 2)
            for (size_t p = lo, q = 2 * lo - 1; p < q; ++p, --q)
            {
                const double lr = 0.5 * (re[p] + re[q]), li = 0.5 * (im[p] - im[q]), rr = 0.5 * (im[p] + im[q]), ri = 0.5 * (re[q] - re[p]);
                const double ur = lr * hr[p] - li * hi[p], ui = lr * hi[p] + li * hr[p], vr = rr * hr[q] - ri * hi[q], vi = rr * hi[q] + ri * hr[q];
                re[p] = ur - vi; im[p] = ui + vr; re[q] = ur + vi; im[q] = vr - ui;
            }
        F.gather (im.data(), re.data());        // the way back: the same transform with real and imaginary part swapped
        const size_t got = std::min (n - at, len + m - 1);
        for (size_t i = 0; i < got; ++i) { out.l[at + i] += re[i]; out.r[at + i] += im[i]; }
    }
    return out;
}

// A room rebuilt from numbers is one even wash; a real short room is mostly a few early reflections, which the ear
// takes as part of the sound itself. At the same energy the copy therefore comes out wetter than the real thing
// when the room is short: on a second sound it goes on 4 dB quieter for mid-band decay times (as found) up to 1 s,
// less and less up to 2 s, and as found from there on.
inline double leanDb (const Found& f)
{
    const Bands B; Vec t; for (size_t b = 0; b < NB; ++b) if (midBand (B.fc[b])) t.push_back (f.t60[b]);
    return -4.0 * std::clamp (1.0 - std::log2 (std::max (middle (t), 1e-3)), 0.0, 1.0);
}
// The found room on a second sound: amount 0 = none, 1 = as found (see leanDb), 2 = twice as loud. Stopped: comes back empty.
// It is b + amount * 10^(leanDb (f) / 20) * convolved (b, roomIr (f, length, bright, width)): a caller that keeps
// the convolved part can mix other amounts itself.
inline Stereo onto (const Stereo& b, const Found& f, double amount = 1.0, double length = 1.0, double bright = 1.0, double width = 1.0, const Stop& stop = {})
{
    if (! f.found || amount <= 0.0) return b;
    Stereo out = convolved (b, roomIr (f, length, bright, width), stop); const double s = amount * std::pow (10.0, leanDb (f) / 20.0);
    if (out.size() != b.size()) return {};
    for (size_t i = 0; i < b.size(); ++i) { out.l[i] = b.l[i] + s * out.l[i]; out.r[i] = b.r[i] + s * out.r[i]; }
    return out;
}

constexpr double MORE_MAX = 16.0;       // the room part of a cell is never made more than 12 dB louder
// How much stronger (power) the room part of every band and moment has to be for the room to decay with t1 (s per
// band) instead of the found decay time, starting from the same level: the direct sound as the answer has it, run
// through both decays, one against the other. moreDb: a level change of the room part per band on top.
inline Vec reshapeFactor (const Found& f, const Band& t1, const Band& moreDb)
{
    const int T = f.picture.frames; const double top = f.picture.top();
    if (T < 1 || f.picture.LE.size() != (size_t) T * NB || f.gains.size() != f.picture.LE.size()) throw std::runtime_error ("room: what was found holds no picture of a sound");
    const Vec p0 = smear (f.picture.LE, f.gains, T, top, f.t60), p1 = smear (f.picture.LE, f.gains, T, top, t1);
    const double most = p0.empty() ? 0.0 : *std::max_element (p0.begin(), p0.end()); Band level; Vec F (p0.size());
    for (size_t b = 0; b < NB; ++b) level[b] = (1.0 - std::pow (10.0, -6.0 * DT / f.t60[b])) / (1.0 - std::pow (10.0, -6.0 * DT / t1[b])) * std::pow (10.0, moreDb[b] / 10.0);
    for (size_t i = 0; i < F.size(); ++i) F[i] = std::clamp ((p0[i] > 1e-9 * most ? p1[i] / std::max (p0[i], 1e-300) : 1.0) * level[i % NB], 0.0, MORE_MAX);
    return F;
}

// The sound itself: amount 0 = reverb out, 1 = untouched, 2 = reverb twice as loud. Length, Brightness and Width
// reshape the sound's own reverb (nothing synthetic is added): the room part of every band and moment is turned
// up or down so that it decays with the new decay time from the same starting level, tilted and widened as asked.
// f has to be what find() made of this very sound (std::runtime_error if it is not). Stopped: comes back empty.
inline Stereo oneSound (const Stereo& x, const Found& f, double amount = 1.0, double length = 1.0, double bright = 1.0, double width = 1.0, const Stop& stop = {})
{
    if (! f.found || (std::abs (length - 1.0) < 1e-9 && std::abs (bright - 1.0) < 1e-9 && std::abs (width - 1.0) < 1e-9)) return render (x, f.gains, amount, stop);
    const Bands B; Band t1, more; const double tilt = bright - 1.0;
    for (size_t b = 0; b < NB; ++b)
    {
        const double oct = std::log2 (B.fc[b] / 1000.0);
        t1[b] = std::clamp (f.t60[b] * std::max (length, 0.02) * std::pow (2.0, 0.35 * tilt * oct), 0.03, 20.0); more[b] = 3.0 * tilt * oct;
    }
    return reshaped (x, f.gains, amount, reshapeFactor (f, t1, more), std::clamp (width, 0.0, 2.0), stop);
}

} // namespace room
} // namespace dr
