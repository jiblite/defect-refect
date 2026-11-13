#pragma once
// What each found effect looks like: the usual picture for that kind of effect (a filter's
// frequency curve, a reverb's decay, a delay's echoes...), drawn from the settings the faders
// currently give. Plain data; the app only draws it.

#include <array>
#include <cstdio>

#include "Estimate.h"

namespace dr
{
struct Curve { std::vector<std::pair<double, double>> pts; int colour = 0; bool fill = false, dim = false; };   // colour 0 main, 1 second (right channel), 2 neutral
struct Bar { double x = 0, y0 = 0, y1 = 0, w = 0; int colour = 0; };
struct Plot
{
    std::string title, xLabel, yLabel, caption;
    bool xLog = false; double x0 = 0, x1 = 1, y0 = 0, y1 = 1;
    std::vector<double> xTicks, yTicks;
    std::vector<Curve> curves; std::vector<Bar> bars;
    bool empty() const { return curves.empty() && bars.empty(); }
};

namespace disp
{
inline std::string num (double v, int dec = 1) { char b[64]; std::snprintf (b, sizeof b, "%.*f", dec, v); return b; }
inline std::string hz (double f) { return f >= 1000 ? num (f / 1000, f >= 10000 ? 0 : 1) + " kHz" : num (f, 0) + " Hz"; }
inline std::complex<double> biq (const fx::Biquad& q, double f)
{
    const std::complex<double> z = std::polar (1.0, -2 * PI * f / SR);
    return (q.b0 + q.b1 * z + q.b2 * z * z) / (1.0 + q.a1 * z + q.a2 * z * z);
}
inline double db (double a) { return 20 * std::log10 (std::max (a, 1e-6)); }
inline void freqAxes (Plot& p, double lo = -36, double hi = 18)
{
    p.xLog = true; p.x0 = 20; p.x1 = 20000; p.y0 = lo; p.y1 = hi; p.xLabel = "Hz"; p.yLabel = "dB";
    p.xTicks = { 100, 1000, 10000 }; for (double d = std::ceil (lo / 12) * 12; d <= hi; d += 12) p.yTicks.push_back (d);
}
// round tick values (1, 2, 2.5 or 5 times a power of ten) strictly inside lo..hi
inline std::vector<double> ticks (double lo, double hi, int about = 4)
{
    std::vector<double> t; const double span = hi - lo; if (! (span > 0)) return t;
    const double raw = span / std::max (1, about), mag = std::pow (10.0, std::floor (std::log10 (raw))), r = raw / mag;
    const double step = (r < 1.5 ? 1 : r < 2.25 ? 2 : r < 3.5 ? 2.5 : r < 7.5 ? 5 : 10) * mag;
    for (double v = std::ceil ((lo + span * 0.02) / step) * step; v < hi - span * 0.02; v += step) t.push_back (std::abs (v) < step * 1e-6 ? 0.0 : v);
    return t;
}
template <typename F> Curve freqCurve (F mag, int colour = 0, bool dim = false)
{
    Curve c; c.colour = colour; c.dim = dim;
    for (int i = 0; i <= 240; ++i) { const double f = 20 * std::pow (1000.0, i / 240.0); c.pts.push_back ({ f, db (mag (f)) }); }
    return c;
}
} // namespace disp

// The found room (room engine): the direct sound as a bar at 0 dB and how the room dies away behind it, one line
// each for the lows (under 450 Hz), the mids and the highs (over 7 kHz). A line starts at the level the room has
// right after the direct sound (the energy of one 10.7 ms step against the direct sound's) and falls by 60 dB per
// decay time. s: the Amount fader; onSecond: the room goes onto a second sound (otherwise it is the sound's own).
inline Plot roomPlot (const FoundFx& F, const Params& p, double s, bool onSecond)
{
    using namespace disp;
    Plot P; P.title = F.title; const room::Found& f = *F.room; const room::Bands B;
    // what the Amount fader does to the room's level: on a second sound it is a plain level (with the lean of a
    // rebuilt room); on the sound itself the left half turns the room down as far as the engine typically gets it
    // down (roomAmountDb), the right half up by up to 6 dB
    const double amountDb = onSecond ? 20 * std::log10 (std::max (s, 1e-6)) + room::leanDb (f) : roomAmountDb (std::max (s, 0.0));
    struct Zone { double t = 0, start = 0, all = 0; bool any = false; };
    auto zones = [&] (const room::Shaped& sh, double moreDb)
    {
        std::array<Zone, 3> z;
        for (int i = 0; i < 3; ++i)
        {
            Vec t, k, a;
            for (size_t b = 0; b < (size_t) room::NB; ++b)
            {
                if ((B.fc[b] < 450.0 ? 0 : B.fc[b] <= 7000.0 ? 1 : 2) != i) continue;
                t.push_back (sh.t60[b]); a.push_back (sh.kdb[b]); k.push_back (sh.kdb[b] + 10 * std::log10 (1.0 - std::pow (10.0, -6.0 * room::DT / sh.t60[b])));
            }
            if (! t.empty()) { z[(size_t) i].t = room::middle (t); z[(size_t) i].start = room::middle (k) + moreDb; z[(size_t) i].all = room::middle (a) + moreDb; z[(size_t) i].any = true; }
        }
        return z;
    };
    const room::Shaped now = room::shaped (f, get (p, "length", 1.0), get (p, "bright", 1.0), get (p, "width", 1.0));
    const auto z = zones (now, amountDb), z0 = zones (room::shaped (f), onSecond ? room::leanDb (f) : 0.0);
    P.y0 = -72; P.y1 = 6; P.xLabel = "s"; P.yLabel = "dB"; P.yTicks = { -60, -40, -20, 0 };
    // the time axis holds the room as found and as the faders have it, down to the bottom of the picture
    double end = 0.25; for (const auto* zz : { &z, &z0 }) for (const auto& q : *zz) if (q.any) end = std::max (end, room::DT + q.t * std::clamp (q.start - P.y0, 6.0, 78.0) / 60.0);
    P.x1 = end * 1.06; P.x0 = -0.04 * P.x1; P.xTicks = ticks (0, P.x1);
    P.bars.push_back ({ 0, P.y0, 0, P.x1 * 0.012, 2 });
    const bool off = onSecond && s < 1e-3;
    if (! off)
        for (int i : { 0, 2, 1 })       // lows, highs, and the mids on top
        {
            const Zone& q = z[(size_t) i]; if (! q.any) continue;
            Curve c; c.colour = i == 1 ? 0 : i == 0 ? 2 : 1; c.fill = i == 1; const double fall = q.start - P.y0 + 12.0;
            if (i == 1) c.pts.push_back ({ room::DT, P.y0 - 12.0 });
            c.pts.push_back ({ room::DT, q.start }); c.pts.push_back ({ room::DT + q.t * fall / 60.0, q.start - fall }); P.curves.push_back (c);
        }
    Vec w (now.width.begin(), now.width.end()); const Zone& m = z[1];
    if (off) P.caption = "Amount at zero: the room is not put on the second sound.";
    else
    {
        P.caption = "Decay " + num (m.t, 2) + " s (lows " + num (z[0].t, 2) + " s, highs " + num (z[2].t, 2) + " s). Room " + num (std::abs (m.all), 1) + " dB " + (m.all < 0 ? "under" : "over")
                    + " the direct sound" + (! onSecond && s < 1.0 - 1e-9 ? ", taken down by about " + num (-amountDb, 1) + " dB (typical; less under the sound itself)" : std::string())
                    + ", width " + num (room::middle (w) * 100, 0) + "%. Teal = mids, grey = lows, orange = highs.";
    }
    return P;
}

// The tone engine's groups: the curve the faders give, in dB over frequency, bin by bin (a comb's teeth are evenly
// spaced in Hz, so on this axis they crowd together towards the top).
// COMB FILTER: the found reflections' curve at the Amount and Delay the faders give (Amount at zero: a flat line).
// TONE MATCH: what the second sound gets, from the last render: one curve, or teal = left and orange = right when
// the two sounds are versions of one recording.
inline Plot tonePlot (const FoundFx& F, const Params& p, double s, bool onSecond)
{
    using namespace disp;
    Plot P; P.title = F.title; if (F.tone == nullptr) return P;
    const bool match = F.name == "tonematch"; freqAxes (P, match ? -30 : -24, match ? 18 : 12);
    auto line = [&] (const Vec& c, double k, int colour, bool cap)
    {
        Curve q; q.colour = colour; if (c.size() != (size_t) tone::NB) return;
        for (int b = 4; b < tone::NB && (double) b * tone::binHz <= 20000.0; ++b) { const double v = k * c[(size_t) b]; q.pts.push_back ({ (double) b * tone::binHz, cap ? tone::softCap (v) : v }); }
        P.curves.push_back (std::move (q));
    };
    auto pct = [] (double v) { return num (v * 100, 0) + "%"; };
    if (! match)
    {
        const double stretch = std::clamp (get (p, "delay", 1.0), 0.25, 2.0); const auto& E = F.tone->echoes; if (E.empty()) return P;
        line (tone::combCurve (E, F.tone->spec.both, stretch), std::max (s, 0.0), 0, false);
        std::string c = E.size() == 1 ? "A reflection " : "Reflections ";
        for (size_t i = 0; i < E.size(); ++i)
            c += std::string (i ? " and " : "") + num (E[i].delay * stretch / SR * 1e3, 2) + " ms after the sound at " + pct (std::abs (E[i].gain)) + " of its level" + (E[i].gain < 0 ? " (flipped)" : "");
        c += ": notches every " + hz (SR / std::max (E[0].delay * stretch, 1e-9)) + ".";
        if (s < 1e-3) c += onSecond ? " Amount at zero: not put on the second sound." : " Amount at zero: taken out.";
        else if (std::abs (s - 1.0) > 0.005) c += std::string (" Shown at ") + pct (s) + (onSecond ? "." : " of what was found.");
        P.caption = c; return P;
    }
    const double detail = get (p, "detail", matchDetail), base = matchDetailAt (F, 1.0); const bool same = get (F.est, "same", 0.0) > 0.5, asFound = std::abs (detail - base) < 1e-9;
    const Vec& L = asFound && ! F.foundL.empty() ? F.foundL : F.curveL; const Vec& R = asFound && ! F.foundR.empty() ? F.foundR : F.curveR;
    if (L.size() != (size_t) tone::NB) { P.caption = "Working out the curve..."; return P; }
    if (s < 1e-3) { line (L, 0.0, 0, false); P.caption = "Amount at zero: the second sound keeps its own tone."; return P; }
    if (same && R.size() == L.size() && R != L) line (R, s, 1, true);
    line (L, s, 0, true);
    double hi = -1e9, lo = 1e9, fhi = 0, flo = 0;
    for (int b = 6; b < tone::NB && (double) b * tone::binHz <= 18000.0; ++b) { const double v = tone::softCap (s * L[(size_t) b]); if (v > hi) { hi = v; fhi = b * tone::binHz; } if (v < lo) { lo = v; flo = b * tone::binHz; } }
    const double oct = tone::detailOctaves (detail);
    const std::string fine = oct <= 0 ? "every detail" : oct >= 0.75 ? "broad strokes, an octave wide" : "smoothed over 1/" + num (1.0 / oct, 0) + " octave";
    P.caption = std::string (same ? "Two versions of one recording: what the second still needs, after the other effects, to get the tone of the first, for each side (teal left, orange right)"
                                  : "What the second sound needs, after the other effects, to take the long-term tone of the first")
                + ": up to " + (hi >= 0 ? "+" : "") + num (hi, 1) + " dB around " + hz (fhi) + ", down to " + num (lo, 1) + " dB around " + hz (flo) + ". Detail: " + fine + ".";
    return P;
}

// s: the effect's Amount fader (0..2); p: its settings with the settings faders applied. onSecond: the effect
// goes onto a second sound (only the room's picture depends on it).
inline Plot makePlot (const FoundFx& F, const Params& scaledP, double s, bool onSecond = false)
{
    using namespace disp;
    if (F.room != nullptr) return roomPlot (F, scaledP, s, onSecond);
    if (F.tone != nullptr) return tonePlot (F, scaledP, s, onSecond);
    const std::string& f = F.name; Plot P; P.title = F.title;
    bool fade = false; const Params e = mapAmount (f, scaledP, std::max (s, 0.0), fade); const double w = fade ? std::min (1.0, s) : 1.0;
    auto g = [&] (const char* k, double d) { return get (e, k, d); };
    if (s < 1e-3) { P.caption = "Amount at zero: this effect is off."; }

    if (f == "filter")
    {
        freqAxes (P); const int t = (int) std::lround (g ("type", 2)); const int kind = t == 0 ? 5 : t == 1 ? 4 : 3;
        fx::Biquad a, b; a.setRbj (kind, g ("cutoff", 1500), 0, std::clamp (g ("resonance", 0.707), 0.2, 20.0)); b.setRbj (kind, g ("cutoff", 1500), 0, 0.707); const bool two = g ("slope", 0) > 0.5;
        P.curves.push_back (freqCurve ([&] (double fr) { auto h = biq (a, fr); if (two) h *= biq (b, fr); return std::abs ((1 - w) + w * h); }));
        P.curves.back().fill = true;
        P.caption = std::string (t == 0 ? "Band-pass" : t == 1 ? "High-pass" : "Low-pass") + " at " + hz (g ("cutoff", 1500)) + ", resonance " + num (g ("resonance", 0.707)) + (two ? ", 24 dB/oct" : ", 12 dB/oct");
    }
    else if (f == "filtersweep")
    {
        freqAxes (P); const double lo = g ("fmin", 300), hi = g ("fmax", 5000), q = std::max (0.3, g ("q", 4));
        int i = 0;
        for (double u : { 0.0, 0.5, 1.0 })
        {
            const double fc = lo * std::pow (hi / lo, u); fx::Biquad lp, pk; lp.setLow (fc); pk.setPeak (fc, q);
            P.curves.push_back (freqCurve ([&] (double fr) { const auto h = biq (lp, fr); return std::abs ((1 - w) + w * (h + 0.8 * h * biq (pk, fr))); }, 0, i++ != 1));
        }
        P.caption = "Sweeps " + hz (lo) + " to " + hz (hi) + " every " + num (1.0 / std::max (0.01, g ("rate", 0.25))) + " s, resonance " + num (q);
    }
    else if (f == "phaser")
    {
        freqAxes (P, -36, 12); const int st = std::max (1, (int) std::lround (g ("stages", 6))); const double lo = g ("fmin", 200), hi = g ("fmax", 3000), fb = g ("fb", 0.3), mix = g ("mix", 0.5);
        int i = 0;
        for (double u : { 0.0, 0.5, 1.0 })
        {
            const double fc = lo * std::pow (hi / lo, u), t = std::tan (PI * fc / SR), a = (t - 1) / (t + 1);
            P.curves.push_back (freqCurve ([&] (double fr)
            {
                const std::complex<double> z = std::polar (1.0, -2 * PI * fr / SR); std::complex<double> ap = std::pow ((a + z) / (1.0 + a * z), st);
                const auto wet = ap / (1.0 - fb * ap * 0.98); return std::abs ((1 - w) + w * ((1 - mix) + mix * wet));
            }, 0, i++ != 1));
        }
        P.caption = num (st, 0) + " stages, notches move " + hz (lo) + " to " + hz (hi) + " every " + num (1.0 / std::max (0.01, g ("rate", 0.4))) + " s, feedback " + num (fb, 2);
    }
    else if (f == "flanger")
    {
        freqAxes (P, -36, 12); const double d0 = g ("dmin", 0.5e-3), d1 = g ("dmax", 5e-3), depth = g ("depth", 0.7);
        int i = 0;
        for (double d : { d0, d1 })
        { P.curves.push_back (freqCurve ([&] (double fr) { return std::abs (1.0 + depth * std::polar (1.0, -2 * PI * fr * d)) / (1 + depth); }, 0, i == 1)); ++i; }
        P.caption = "Comb sweeps " + num (d0 * 1e3, 2) + " to " + num (d1 * 1e3, 2) + " ms every " + num (1.0 / std::max (0.01, g ("rate", 0.3))) + " s, intensity " + num (depth, 2);
    }
    else if (f == "chorus")
    {
        const double r = std::max (0.01, g ("rate", 0.8)), base = g ("base", 15e-3), wd = g ("width", 4e-3); const int v = std::max (1, (int) std::lround (g ("voices", 2)));
        P.x0 = 0; P.x1 = 2.0 / r; P.y0 = 0; P.y1 = std::max (5.0, (base + wd) * 1e3 * 1.25); P.xLabel = "s"; P.yLabel = "ms";
        P.xTicks = ticks (P.x0, P.x1); P.yTicks = ticks (P.y0, P.y1);
        for (int k = 0; k < v; ++k) for (int side = 0; side < 2; ++side)
        {
            Curve c; c.colour = side; for (int i = 0; i <= 200; ++i) { const double t = P.x1 * i / 200; c.pts.push_back ({ t, std::max (0.0, base + wd * std::sin (2 * PI * r * (1 + 0.13 * k) * t + (side ? PI / 2 : 0) + k * 2.1)) * 1e3 }); }
            P.curves.push_back (c);
        }
        P.caption = num (v, 0) + (v == 1 ? " copy" : " copies") + " around " + num (base * 1e3) + " ms, drifting +-" + num (wd * 1e3) + " ms, mix " + num (g ("mix", 0.5), 2);
    }
    else if (f == "delay")
    {
        const double ms = g ("ms", 300), fb = g ("fb", 0.4), mix = g ("mix", 0.4); const bool pp = g ("pingpong", 0) > 0.5;
        P.x0 = -0.3 * ms; P.x1 = 7.6 * ms; P.y0 = -48; P.y1 = 3; P.xLabel = "ms"; P.yLabel = "dB"; P.xTicks = { ms, 2 * ms, 4 * ms, 6 * ms }; P.yTicks = { -36, -24, -12, 0 };
        P.bars.push_back ({ 0, -48, 0, ms * 0.12, 2 }); double a = mix;
        for (int k = 1; k < 8 && a > 1e-3; ++k) { P.bars.push_back ({ k * ms, -48, std::max (-48.0, db (a)), ms * 0.12, pp ? (k % 2 ? 1 : 0) : 0 }); a *= fb; }
        P.caption = "Echo every " + num (ms, 0) + " ms, feedback " + num (fb, 2) + ", level " + num (db (mix)) + " dB" + (pp ? ", ping-pong (teal left, orange right)" : "");
    }
    else if (f == "reverb")
    {
        const double rt = std::max (0.05, g ("rt60", 1.8)), mix = std::clamp (g ("mix", 0.35), 0.0, 0.99);
        P.x0 = -0.05 * rt; P.x1 = std::max (1.0, rt * 1.25); P.y0 = -66; P.y1 = 3; P.xLabel = "s"; P.yLabel = "dB"; P.xTicks = ticks (0, P.x1); P.yTicks = { -60, -40, -20, 0 };
        P.bars.push_back ({ 0, -66, db (1 - mix), P.x1 * 0.012, 2 });
        // tail level right after the direct sound: mix x (wet/dry level), spread over the tail (about -10 log10 of its length in 10 ms slices)
        const double start = db (mix) - 6;
        Curve c; c.fill = true; c.pts.push_back ({ 0.01, P.y0 }); c.pts.push_back ({ 0.01, start }); c.pts.push_back ({ 0.01 + rt * 1.2, start - 72 }); P.curves.push_back (c);
        P.caption = "Decay " + num (rt, 2) + " s (to -60 dB), room level " + num (mix * 100, 0) + "%";
    }
    else if (f == "distortion" || f == "saturation" || f == "bitcrush")
    {
        P.x0 = -1; P.x1 = 1; P.y0 = f == "saturation" ? -1.3 : -1.05; P.y1 = -P.y0; P.xLabel = "in"; P.yLabel = "out"; P.xTicks = { -0.5, 0, 0.5 }; P.yTicks = { -0.5, 0, 0.5 };
        Curve id; id.colour = 2; id.dim = true; id.pts = { { -1, -1 }, { 1, 1 } }; P.curves.push_back (id);
        Curve c;
        if (f == "distortion")
        {
            const double d = std::max (0.5, g ("drive", 8)); const bool th = g ("kind", 1) > 0.5; const double top = th ? std::tanh (d) : std::min (1.0, d);
            for (int i = 0; i <= 200; ++i) { const double x = -1 + 2.0 * i / 200, y = (th ? std::tanh (x * d) : std::clamp (x * d, -1.0, 1.0)) / top; c.pts.push_back ({ x, (1 - w) * x + w * y }); }
            P.caption = std::string (th ? "Soft clip" : "Hard clip") + ", drive x" + num (d);
        }
        else if (f == "saturation")
        {
            const double k = std::pow (10.0, g ("drive", 8) / 20) * 0.5, b = g ("kind", 1) > 0.5 ? 0.25 : 0.0, nrm = std::tanh (k);
            for (int i = 0; i <= 200; ++i) { const double x = -1 + 2.0 * i / 200, y = (std::tanh (x * k + b) - std::tanh (b)) / nrm; c.pts.push_back ({ x, (1 - w) * x + w * y }); }
            P.caption = std::string (b > 0 ? "Tape-like" : "Soft") + " saturation, drive " + num (g ("drive", 8)) + " dB, tone " + num (g ("tone", 0.5) * 100, 0) + "%";
        }
        else
        {
            const double bits = std::max (1.0, g ("bits", 6)), q = std::pow (2.0, bits - 1);
            for (int i = 0; i <= 400; ++i) { const double x = -1 + 2.0 * i / 400; c.pts.push_back ({ x, std::nearbyint (x * q) / q }); }
            P.caption = num (bits, 1) + " bits, every sample held " + num (std::max (1.0, g ("down", 4)), 1) + "x (" + hz (SR / std::max (1.0, g ("down", 4))) + " rate)";
        }
        P.curves.push_back (c);
    }
    else if (f == "tremolo" || f == "autopan" || f == "gatedrhythm")
    {
        const double r = std::max (0.02, g ("rate", 1)); const bool gate = f == "gatedrhythm";
        P.x0 = 0; P.x1 = gate ? 16.0 / r : 2.0 / r; P.xLabel = "s"; P.yLabel = gate ? "level" : "dB"; P.xTicks = ticks (P.x0, P.x1);
        if (gate)
        {
            const double duty = g ("duty", 0.5), depth = g ("depth", 1); P.y0 = 0; P.y1 = 1.1; P.yTicks = { 0.5, 1.0 };
            Curve c; c.fill = true; c.pts.push_back ({ 0, 0 });
            for (int k = 0; k < 16; ++k) { const double a = k / r, b = (k + duty) / r, e2 = (k + 1) / r; c.pts.push_back ({ a, 1 }); c.pts.push_back ({ b, 1 }); c.pts.push_back ({ b, 1 - depth }); c.pts.push_back ({ e2, 1 - depth }); }
            c.pts.push_back ({ P.x1, 0 }); P.curves.push_back (c);
            P.caption = num (r) + " steps per second, open " + num (duty * 100, 0) + "% of each step, cuts down by " + num (depth * 100, 0) + "%";
        }
        else
        {
            const double a = g ("amp_db", 0), ph = g ("phase", 0), span = std::max (3.0, a * 1.2); P.y0 = -span; P.y1 = span; P.yTicks = ticks (-span, span);
            for (int side = 0; side < (f == "autopan" ? 2 : 1); ++side)
            {
                Curve c; c.colour = side; const double k = f == "autopan" ? 0.5 : 1.0;
                for (int i = 0; i <= 200; ++i) { const double t = P.x1 * i / 200; c.pts.push_back ({ t, (side ? -k : k) * a * std::sin (2 * PI * (r * t + ph)) }); }
                P.curves.push_back (c);
            }
            P.caption = f == "autopan" ? "Moves left-right " + num (r, 2) + " times a second, " + num (a) + " dB between the sides (teal left, orange right)"
                                       : "Level wobbles " + num (r, 2) + " times a second, +-" + num (a) + " dB";
        }
    }
    else if (f == "haas" || f == "width")
    {
        if (f == "haas")
        {
            const double ms = g ("lag", 0) / SR * 1e3, a = std::abs (ms); P.x0 = -0.25 * std::max (a, 1.0); P.x1 = std::max (a, 1.0) * 1.5; P.y0 = 0; P.y1 = 1.2; P.xLabel = "ms"; P.xTicks = { a };
            P.bars.push_back ({ 0, 0, 1, P.x1 * 0.015, ms < 0 ? 0 : 1 }); P.bars.push_back ({ a, 0, 1, P.x1 * 0.015, ms < 0 ? 1 : 0 });
            P.caption = std::string (ms < 0 ? "Right" : "Left") + " side arrives " + num (a, 2) + " ms late (teal left, orange right)";
        }
        else
        {
            const double am = g ("amount", 0.5); P.x0 = -1.2; P.x1 = 1.2; P.y0 = 0; P.y1 = 1.2; P.xLabel = "left - centre - right"; P.xTicks = { 0 };
            Curve c; c.fill = true; const double sp = std::min (1.0, am / 1.2);    // half-ellipse: wider with more side
            for (int i = 0; i <= 120; ++i) { const double th = PI * i / 120; c.pts.push_back ({ std::cos (th) * (0.08 + sp), std::sin (th) }); }
            P.curves.push_back (c);
            P.caption = "Added spread " + num (am * 100, 0) + "% (side signal above 300 Hz, 7 ms behind)";
        }
    }
    else if (f == "exciter")
    {
        freqAxes (P, -24, 24); const double fr0 = g ("freq", 3000), am = g ("amount", 0.3); fx::Biquad hp; hp.setRbj (4, fr0);
        P.curves.push_back (freqCurve ([&] (double fr) { const double h = std::abs (biq (hp, fr)); return 1.0 + 2.5 * am * h * h; }));
        P.curves.back().fill = true;
        P.caption = "New harmonics above " + hz (fr0) + ", intensity " + num (am * 100, 0) + "% (curve is a sketch of where they land)";
    }
    else if (f == "ringmod")
    {
        const double c0 = g ("carrier", 200), mix = g ("mix", 0.8); P.xLog = true; P.x0 = 20; P.x1 = 20000; P.y0 = 0; P.y1 = 1.2; P.xLabel = "Hz"; P.xTicks = { 100, 1000, 10000 };
        P.bars.push_back ({ 1000, 0, 1 - mix, 0, 2 });
        for (double fr : { std::abs (1000 - c0), 1000 + c0 }) if (fr > 20 && fr < 20000) P.bars.push_back ({ fr, 0, mix * 0.71, 0, 0 });
        P.caption = "Multiplied by a " + hz (c0) + " tone: every pitch splits into two, " + hz (c0) + " above and below (shown for a 1 kHz tone going in). Mix " + num (mix * 100, 0) + "%";
    }
    else if (f == "stutter")
    {
        const double L = g ("slice_ms", 120) / 1e3, prob = g ("prob", 0.4); const int reps = std::max (1, (int) std::lround (g ("repeats", 4)));
        P.x0 = 0; P.x1 = L * 24; P.y0 = 0; P.y1 = 1.2; P.xLabel = "s"; P.xTicks = ticks (P.x0, P.x1);
        std::mt19937 rng (3); std::uniform_real_distribution<double> u (0, 1); int k = 0;
        while (k < 24)
        {
            if (u (rng) < prob) { for (int r = 0; r < reps && k < 24; ++r, ++k) P.bars.push_back ({ (k + 0.5) * L, 0, 1.0, L * 0.86, 0 }); }
            else { P.bars.push_back ({ (k + 0.5) * L, 0, 0.45, L * 0.86, 2 }); ++k; }
        }
        P.caption = num (L * 1e3, 0) + " ms slices, repeated " + num (reps, 0) + "x, on " + num (prob * 100, 0) + "% of the slices (tall = a repeat; where they land is random)";
    }
    return P;
}

} // namespace dr
