#pragma once
// What each found effect looks like: the usual picture for that kind of effect (a filter's
// frequency curve, a reverb's decay, a delay's echoes...), drawn from the settings the faders
// currently give. Plain data; the app only draws it.

#include <algorithm>
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
inline std::string timeText (double sec) { const int t = (int) std::floor (sec + 0.5); return std::to_string (t / 60) + ":" + (t % 60 < 10 ? "0" : "") + std::to_string (t % 60); }     // a time in a sound as M:SS
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

// The found echoes (echo engine): the sound itself as a bar at 0 dB and every repeat as a bar at its time after the
// sound and its level against it, as the faders have them. One row of bars when left and right carry the same
// echo, else teal = left and orange = right. With Feedback off the middle the repeats' levels follow a plain model
// (each repeat keeps the same share of the one before); the tone of the echoes is not in the picture.
inline Plot echoPlot (const FoundFx& F, const Params& p, double s, bool onSecond)
{
    using namespace disp;
    Plot P; P.title = F.title; if (F.echo == nullptr || ! F.echo->any()) return P;
    const echo::Found& f = *F.echo; const echo::Shape sh = echoShape (p, s); echo::Shape full = sh; full.amount = std::max (1.0, sh.amount);
    const bool both = f.ch[0].found && f.ch[1].found;
    const bool same = both && f.ch[0].rec.q == f.ch[1].rec.q && f.ch[0].level.size() == f.ch[1].level.size() && ! f.ch[0].level.empty() && std::abs (f.ch[0].level[0] - f.ch[1].level[0]) < 0.5;
    double step = 1e9, last = 0; for (int c = 0; c < 2; ++c) if (f.ch[c].found) { step = std::min (step, (double) f.ch[c].rec.q * sh.time / SR * 1e3); for (auto& r : echo::repeats (f.ch[c], full)) if (r.db > -54.0) last = std::max (last, r.ms); }
    P.x0 = -0.35 * step; P.x1 = std::max (last + 0.6 * step, 4.6 * step); P.y0 = -60; P.y1 = 3; P.xLabel = "ms"; P.yLabel = "dB"; P.xTicks = ticks (0, P.x1); P.yTicks = { -48, -36, -24, -12, 0 };
    const double w = std::min (0.14 * step, 0.012 * (P.x1 - P.x0) * (same || ! both ? 1.6 : 1.0));
    P.bars.push_back ({ 0, P.y0, 0, w, 2 });
    for (int c = 0; c < 2; ++c)
    {
        if (! f.ch[c].found || (same && c == 1)) continue;
        const double shift = same || ! both ? 0.0 : (c == 0 ? -0.55 : 0.55) * w;
        for (auto& r : echo::repeats (f.ch[c], sh)) if (r.db > P.y0 && r.ms + shift < P.x1) P.bars.push_back ({ r.ms + shift, P.y0, std::min (r.db, P.y1), w, same || ! both ? 0 : c });
    }
    const int lead = f.ch[0].found ? 0 : 1; const echo::Channel& L = f.ch[lead];
    auto ms = [&] (int c) { return num ((double) f.ch[c].rec.q * sh.time / SR * 1e3, 0) + " ms"; };
    std::string c = same || ! both ? "Echo every " + ms (lead) + (both ? "" : lead == 0 ? " on the left side only" : " on the right side only")
                                   : f.ch[0].rec.q == f.ch[1].rec.q ? "Echo every " + ms (0) + " (teal left, orange right)" : "Echoes every " + ms (0) + " on the left (teal) and every " + ms (1) + " on the right (orange)";
    if (! L.level.empty())
    {
        const double phi = echo::shareAt (L, sh.feedback);
        if (sh.amount > 1e-3) c += ", the first one " + num (-(L.level[0] + 20 * std::log10 (sh.amount)), 1) + " dB under the sound";
        c += phi > 0.03 ? ", each repeat about " + num (-20 * std::log10 (phi), 1) + " dB under the one before" : std::string (", no further repeats");
    }
    if (std::abs (sh.bright - 1.0) > 0.005) c += sh.bright < 1.0 ? ", echoes darker" : ", echoes brighter";
    c += ".";
    if (sh.amount < 1e-3) c += onSecond ? " Amount at zero: not put on the second sound." : " Amount at zero: taken out.";
    P.caption = c; return P;
}

// The found level movement (level engine): the gain in dB over two cycles as the faders have it, its top at 0 dB;
// teal = left and orange = right when the sides move against each other. A pan seen less than twice is drawn as the
// course that was measured over the sound.
inline Plot levelPlot (const FoundFx& F, const Params& p, double s, bool onSecond)
{
    using namespace disp;
    Plot P; P.title = F.title; if (F.level == nullptr || ! F.level->found || F.level->best() == nullptr) return P;
    const level::Movement& M = *F.level; const level::Found& f = *M.best(); const double speed = levelSpeed (p), amount = std::max (s, 0.0);
    const bool course = ! f.lane[0].empty();
    const double span = course ? (double) f.lane[0].size() * level::HOP / SR : 2.0 * f.period;
    P.x0 = 0; P.x1 = span; P.xLabel = "s"; P.yLabel = "dB"; P.xTicks = ticks (0, span);
    const level::GainAt at (f); const int N = 600; double low = 0;
    Curve cl, cr; cl.colour = 0; cr.colour = 1;
    for (int i = 0; i <= N; ++i)
    {
        const double t = span * i / N; double gl, gr, ol, orr; at (t * SR * speed, gl, gr); at (t * SR, ol, orr);
        cl.pts.push_back ({ t, amount * gl }); cr.pts.push_back ({ t, amount * gr }); low = std::min ({ low, ol, orr, amount * gl, amount * gr });      // the axis holds the movement as found too
    }
    P.y1 = 3; P.y0 = std::clamp (std::floor (1.15 * low - 1.0), -72.0, -6.0);
    const double step = P.y0 < -40 ? 12 : P.y0 < -16 ? 6 : 3; for (double d = 0; d > P.y0 + 0.3 * step; d -= step) P.yTicks.push_back (d);
    for (Curve* c : { &cl, &cr }) for (auto& q : c->pts) q.second = std::max (q.second, P.y0);
    P.curves.push_back (cl); if (M.against) P.curves.push_back (cr);
    const double rate = f.rate * speed, depth = f.depth * amount; std::string c;
    if (M.against)
        c = course ? "Moves from side to side about once in " + num (f.period / speed, 1) + " s, " + (depth >= 40 ? std::string ("all the way to each side") : "up to " + num (depth) + " dB between the sides") + " (teal left, orange right). Seen less than twice: drawn as it was measured."
                   : "Moves between the sides " + num (rate, 2) + " times a second, " + (depth >= 40 ? std::string ("all the way to each side") : "up to " + num (depth) + " dB between them") + " (teal left, orange right).";
    else if (M.steps) c = "Opens and closes " + num (rate, 2) + " times a second, " + num (depth) + " dB down when closed" + (f.leavesOut() ? ", some steps left out." : ".");
    else c = "The level goes down by " + num (depth) + " dB and back " + num (rate, 2) + " times a second.";
    {                                                                  // where in the sound it is, when that is not all of it
        const double all = (double) M.n / SR; double in = 0; for (const level::Span& sp : M.spans) in += (double) (sp.b - sp.a) / SR;
        if (! M.spans.empty() && in < 0.97 * all && all >= 1.0)
        {
            const std::string where = onSecond ? "On the first sound it is there " : "It is there ", rest = onSecond ? "; it goes onto the second sound at the same times." : "; the rest of the sound is left alone.";
            if (M.spans.size() == 1) c += " " + where + "from " + timeText ((double) M.spans[0].a / SR) + " to " + timeText ((double) M.spans[0].b / SR) + rest;
            else c += " " + where + "on " + num (in, 0) + " of " + num (all, 0) + " s, in " + std::to_string (M.spans.size()) + " stretches" + rest;
        }
    }
    if (amount < 1e-3) c += onSecond ? " Amount at zero: not put on the second sound." : " Amount at zero: taken out.";
    P.caption = c; return P;
}

// The flanger engine's group: the copy's delay over time as the faders have it (teal left, orange right).
inline Plot flangerPlot (const FoundFx& F, const Params& p, double s, bool onSecond)
{
    using namespace disp;
    Plot P; P.title = F.title; if (F.sweep == nullptr || ! F.sweep->found) return P;
    const sweep::Result& R = *F.sweep; const double speed = sweepSpeed (p), reach = sweepReach (p), fbAdd = sweepFeedback (p), amount = std::max (s, 0.0);
    const int c0 = R.has[0] ? 0 : 1; const sweep::Found& f = R.ch[c0]; const int T = (int) f.d.size(); if (T < 2) return P;
    const bool periodic = f.curve.valid; const double all = (double) T / sweep::FPS;
    // where along the first sound the copy is: the picture starts at its longest stretch
    const auto spans = sweep::where (R); double in = 0, from = 0, longest = 0;
    for (const auto& sp : spans) { const double len = (double) (sp.second - sp.first) / SR; in += len; if (len > longest) { longest = len; from = (double) sp.first / SR; } }
    const bool part = ! spans.empty() && in < 0.97 * (double) R.n / SR;
    const double span = periodic && f.curve.rate * speed > 1e-6 ? std::clamp (2.0 / (f.curve.rate * speed), 0.5, std::max (0.5, all)) : std::min (part ? std::max (longest, 2.0) : all, 12.0);
    const int n = std::max (2, (int) std::ceil (span * sweep::FPS)), off = part && ! periodic ? (int) (from * sweep::FPS / std::max (speed, 1e-3)) : 0; double hi = 0, lo = 1e9; Vec seen;
    for (int c = 0; c < 2; ++c)
    {
        if (! R.has[c] || (c == 1 && R.has[0] && R.ch[0].d == R.ch[1].d && speed == 1.0 && reach == 1.0)) continue;
        const Vec d = sweep::movement (R.ch[c], off + n, speed, reach); Curve cv; cv.colour = c;
        for (int t = 0; t < n; ++t) { const double ms = d[(size_t) (off + t)] / SR * 1e3; cv.pts.push_back ({ (double) (off + t) / sweep::FPS, ms }); hi = std::max (hi, ms); lo = std::min (lo, ms); seen.push_back (ms); }
        P.curves.push_back (cv);
    }
    P.x0 = (double) off / sweep::FPS; P.x1 = (double) (off + n - 1) / sweep::FPS; P.y0 = 0; P.y1 = std::max (1.0, hi * 1.2); P.xLabel = "s"; P.yLabel = "ms"; P.xTicks = ticks (P.x0, P.x1); P.yTicks = ticks (P.y0, P.y1);
    if (! periodic && seen.size() >= 200)       // a free line: the caption's range leaves out its outermost hundredths (the first frames of a stretch can be off)
    {
        std::sort (seen.begin(), seen.end()); const size_t k = seen.size() / 100;
        lo = seen[k]; hi = seen[seen.size() - 1 - k];
    }
    const double g = std::clamp (amount * std::abs (f.g), 0.0, 0.98), fb = std::clamp (f.fb + fbAdd, -0.9, 0.9);
    std::string c = "A copy of the sound at " + num (g * 100, 0) + "% of its level" + (f.g < 0 ? " (flipped)" : "") + ", its delay moving between " + num (lo, 2) + " and " + num (hi, 2) + " ms";
    c += periodic ? ", " + num (f.curve.rate * speed, 2) + " times a second" : ", freely";
    if (std::abs (fb) >= 0.05) c += ", feedback " + num (std::abs (fb) * 100, 0) + "%";
    c += P.curves.size() > 1 ? " (teal left, orange right)." : ".";
    if (part)           // a flanger on a part of the first sound: taken out there only; onto a second sound it goes all the way
    {
        const std::string where = onSecond ? "On the first sound it is there " : "It is there ", rest = onSecond ? "; it goes onto the whole of the second sound." : "; the rest of the sound is left alone.";
        if (spans.size() == 1) c += " " + where + "from " + timeText ((double) spans[0].first / SR) + " to " + timeText ((double) spans[0].second / SR) + rest;
        else c += " " + where + "on " + num (in, 0) + " of " + num ((double) R.n / SR, 0) + " s, in " + std::to_string (spans.size()) + " stretches" + rest;
    }
    if (amount < 1e-3) c = onSecond ? "Amount at zero: not put on the second sound." : "Amount at zero: the copy is taken out.";
    P.caption = c; return P;
}

// The chorus engine's group: every voice's delay over time as the faders have it (teal left, orange right).
inline Plot chorusPlot (const FoundFx& F, const Params& p, double s, bool onSecond)
{
    using namespace disp;
    Plot P; P.title = F.title; if (F.chorus == nullptr || ! F.chorus->found) return P;
    const chorus::Found& C = *F.chorus; const chorus::Piece* pc = C.best(); if (pc == nullptr) return P;
    const double speed = chorusSpeed (p), depth = chorusDepth (p), amount = std::max (s, 0.0);
    double slow = 1e9, fast = 0, loud = 0; int count = 0;
    for (int c = 0; c < 2; ++c) for (const auto& v : pc->voices[c]) { slow = std::min (slow, v.rate * speed); fast = std::max (fast, v.rate * speed); loud += v.ahead ? 1.0 / std::max (v.loud, 0.25) : v.loud; ++count; }      // a copy louder than the sound: its loudness holds the sound's level against the copy
    if (count == 0) return P;
    loud /= count;
    auto alike = [] (const chorus::Voice& a, const chorus::Voice& b) { return a.kind == b.kind && a.rate == b.rate && a.phase == b.phase && a.mid == b.mid && a.half == b.half; };
    bool same = pc->voices[0].size() == pc->voices[1].size(); for (size_t i = 0; same && i < pc->voices[0].size(); ++i) same = alike (pc->voices[0][i], pc->voices[1][i]);
    // two rounds of the slowest voice (half a second at least, twelve at most), from the start of the stretch the voices were read on
    const double span = std::clamp (2.0 / std::max (slow, 1e-3), 0.5, 12.0), start = (double) pc->a / SR, pivot = 0.5 * (double) C.n - (double) pc->a; const int n = 240; double lo = 1e9, hi = 0;
    for (int c = 0; c < 2; ++c)
    {
        if (c == 1 && same) continue;
        for (const auto& v : pc->voices[c])
        {
            chorus::Voice q = v; q.half = v.half * depth; q.rate = v.rate * speed; q.phase = v.phase + v.rate * (pivot / (chorus::HOP * chorus::DEC)) / chorus::FPS * (1.0 - speed);
            Curve cv; cv.colour = c;
            for (int i = 0; i < n; ++i) { const double sec = span * i / (n - 1); double d, r; chorus::at (q, sec * chorus::FPS, d, r); const double ms = std::max (d * chorus::DEC, 8.0) / SR * 1e3; cv.pts.push_back ({ start + sec, ms }); lo = std::min (lo, ms); hi = std::max (hi, ms); }
            P.curves.push_back (cv);
        }
    }
    P.x0 = start; P.x1 = start + span; P.y0 = 0; P.y1 = std::max (1.0, hi * 1.2); P.xLabel = "s"; P.yLabel = "ms"; P.xTicks = ticks (P.x0, P.x1); P.yTicks = ticks (P.y0, P.y1);
    const size_t nl = pc->voices[0].size(), nr = pc->voices[1].size(); const double level = std::clamp (amount * loud, 0.0, 4.0) * 100;
    std::string c;
    if (nl == nr || nl == 0 || nr == 0) { const size_t k = std::max (nl, nr); c = k == 1 ? "A copy of the sound" : std::to_string (k) + " copies of the sound"; if (nl == nr && ! same) c += " a side"; else if (nl == 0) c += " on the right"; else if (nr == 0) c += " on the left"; }
    else c = std::to_string (nl) + (nl == 1 ? " copy" : " copies") + " of the sound on the left and " + std::to_string (nr) + " on the right";
    c += std::string (count == 1 || (same && nl == 1) ? ", at about " : ", each at about ") + num (level, 0) + "% of its level, " + (count == 1 || (same && nl == 1) ? "its delay" : "their delays") + " moving between " + num (lo, 1) + " and " + num (hi, 1) + " ms, ";
    c += fast - slow < 0.02 * fast ? num (fast, 2) + " times a second" : num (slow, 2) + " to " + num (fast, 2) + " times a second";
    c += ! same && nl > 0 && nr > 0 ? " (teal left, orange right)." : ".";
    const auto spans = chorus::where (C); double in = 0; for (const auto& sp : spans) in += (double) (sp.second - sp.first) / SR;
    if (! spans.empty() && in < 0.97 * (double) C.n / SR)      // a chorus on a part of the first sound: taken out there only
    {
        const std::string where = onSecond ? "On the first sound they are there " : "They are there ", rest = onSecond ? "; onto the second sound they go at the same times." : "; the rest of the sound is left alone.";
        if (spans.size() == 1) c += " " + where + "from " + timeText ((double) spans[0].first / SR) + " to " + timeText ((double) spans[0].second / SR) + rest;
        else c += " " + where + "on " + num (in, 0) + " of " + num ((double) C.n / SR, 0) + " s, in " + std::to_string (spans.size()) + " stretches" + rest;
    }
    if (amount < 1e-3) c = onSecond ? "Amount at zero: not put on the second sound." : "Amount at zero: the copies are taken out.";
    P.caption = c; return P;
}

// The moving-tone engine's groups (PHASER, FILTER SWEEP): the found tone curve at three moments of its movement, where
// it sits lowest on the frequency axis, in the middle (the strong line) and highest, as the Amount fader has it. Where the
// curve sits at a moment: the middle of what it changes there, on the band axis.
inline Plot movingPlot (const FoundFx& F, const Params& p, double s, bool onSecond)
{
    using namespace disp;
    Plot P; P.title = F.title; if (F.moving == nullptr || ! F.moving->found) return P;
    const moving::Found& M = *F.moving; const double speed = movingSpeed (p), amount = std::max (s, 0.0); const int NB = moving::NB;
    auto cell = [&] (int t, int b) { const size_t i = (size_t) t * (size_t) NB + (size_t) b; return 0.5 * ((double) M.raw[0][i] + (double) M.raw[1][i]); };
    std::vector<std::pair<double, int>> place; double low = 0, high = 0;
    for (int t = 0; t < M.T; ++t)
    {
        if (M.gate[(size_t) t] < 1.0) continue;
        double a = 0, w = 0;
        for (int b = 0; b < NB; ++b) { const double v = cell (t, b); a += b * std::abs (v); w += std::abs (v); low = std::min (low, v); high = std::max (high, v); }
        if (w > 0.25 * NB) place.push_back ({ a / w, t });          // a frame where the curve does something (a quarter of a dB a band)
    }
    freqAxes (P, -36, 18);
    if (place.size() >= 3)
    {
        std::sort (place.begin(), place.end()); const size_t n = place.size(); int i = 0;
        for (size_t k : { n / 10, n - 1 - n / 10, n / 2 })          // lowest and highest faintly, the middle one last and strong
        {
            const int t = place[k].second, t0 = std::max (t - 1, 0), t1 = std::min (t + 1, M.T - 1); Curve c; c.dim = i++ < 2;
            for (int b = 0; b < NB; ++b) { double v = 0; for (int q = t0; q <= t1; ++q) v += cell (q, b); c.pts.push_back ({ moving::F0 * std::pow (2.0, b / moving::BPO), std::clamp (amount * v / (t1 - t0 + 1), P.y0, P.y1) }); }
            P.curves.push_back (std::move (c));
        }
        P.curves.back().fill = true;
    }
    const bool phaser = F.name == "phaser"; const double all = (double) M.n / SR;
    std::string c = phaser ? "Notches that travel up and down the spectrum" : "A filter whose curve travels up and down the spectrum";
    c += ", up to " + num (std::max (-low, high) * amount, 0) + " dB " + (-low >= high ? "down" : "up") + ". Drawn where it sits lowest, in the middle (strong line) and highest.";
    if (M.period > 0) c += " It comes round every " + num (M.period / moving::FPS / speed, 1) + " s.";
    double in = 0; for (const auto& sp : M.spans) in += (double) (sp.second - sp.first) / moving::FPS; in = std::min (in, all);
    if (! M.spans.empty() && in < 0.97 * all && all >= 1.0)             // where in the sound it is, when that is not all of it
    {
        const std::string where = onSecond ? "On the first sound it is there " : "It is there ", rest = onSecond ? "; it goes onto the second sound at the same times." : "; the rest of the sound is left alone.";
        if (M.spans.size() == 1) c += " " + where + "from " + timeText (M.spans[0].first / moving::FPS) + " to " + timeText (std::min (all, M.spans[0].second / moving::FPS)) + rest;
        else c += " " + where + "on " + num (in, 0) + " of " + num (all, 0) + " s, in " + std::to_string (M.spans.size()) + " stretches" + rest;
    }
    if (amount < 1e-3) c = onSecond ? "Amount at zero: not put on the second sound." : "Amount at zero: the moving filter is taken off.";
    P.caption = c; return P;
}

// The sides engine's groups (HAAS WIDTH, STEREO WIDTH). The pair: how the one sound reaches each side as the Amount fader
// has it: a bar per arrival, at its time and at its level against the plain sound (1 = the sound itself, under the line =
// upside down); teal = left, orange = right. The copy: the short filter through which the middle is put at the sides,
// drawn as what comes out of it for a single click.
inline Plot sidesPlot (const FoundFx& F, double s, bool onSecond)
{
    using namespace disp;
    Plot P; P.title = F.title; if (F.sides == nullptr || ! F.sides->found()) return P;
    const sides::Found& f = *F.sides; const double amount = std::clamp (s, 0.0, 2.0); std::string c;
    if (f.mode == sides::COPY)
    {
        const int KB = sides::KB; if ((int) f.h.size() != 2 * KB + 1) return P;
        double top = 1e-9, all = 0; for (double v : f.h) { top = std::max (top, std::abs (v)); all += v * v; }
        int half = 48; for (; half < KB; half += 24) { double in = 0; for (int i = KB - half; i <= KB + half; ++i) in += f.h[(size_t) i] * f.h[(size_t) i]; if (in >= 0.999 * all) break; }     // as wide as the filter has anything (1 ms at least)
        half = std::min (half, KB); const double hi = 1.15 * top * std::max (1.0, amount);
        P.x0 = -half / SR * 1e3; P.x1 = half / SR * 1e3; P.y0 = -hi; P.y1 = hi; P.xLabel = "ms"; P.yLabel = "level"; P.xTicks = ticks (P.x0, P.x1); P.yTicks = ticks (P.y0, P.y1);
        Curve z; z.colour = 2; z.dim = true; z.pts = { { P.x0, 0.0 }, { P.x1, 0.0 } }; P.curves.push_back (z);
        Curve cv; for (int i = KB - half; i <= KB + half; ++i) cv.pts.push_back ({ (i - KB) / SR * 1e3, amount * f.h[(size_t) i] }); P.curves.push_back (cv);
        c = "The middle of the sound goes through this short filter (drawn: what a single click becomes) and is added on the left side and taken away on the right.";
        if (amount >= 1e-3) c += " That puts the sides about " + num (std::abs (f.sideDb + 20 * std::log10 (amount)), 1) + " dB " + (f.sideDb + 20 * std::log10 (amount) <= 0 ? "under" : "over") + " the middle.";
    }
    else
    {
        const double lv = sides::levelOf (f.a, f.b); if (! (lv > 0)) return P;
        sides::Taps a = f.a, b = f.b; for (auto& t : a) t.v /= lv; for (auto& t : b) t.v /= lv;
        double da = 0, ga = 0, db = 0, gb = 0; const bool plain = sides::oneTap (a, da, ga) && sides::oneTap (b, db, gb) && ga > 0 && gb > 0;
        auto arrivals = [&] (double q)                                  // every arrival of each side with Amount at q: ms, level
        {
            std::array<std::vector<std::pair<double, double>>, 2> out; const sides::At A = sides::pairAt (a, b, q, 1.0);
            for (int ch = 0; ch < 2; ++ch)
            {
                if (A.warp) out[(size_t) ch].push_back ({ ((ch == 0 ? da : db) + A.move[ch]) / SR * 1e3, A.factor[ch] * (ch == 0 ? ga : gb) });
                else for (const auto& t : A.taps[ch]) out[(size_t) ch].push_back ({ t.lag / SR * 1e3, t.v });
            }
            return out;
        };
        const auto now = arrivals (amount), was = arrivals (1.0); double last = 0, lo = 0, hi = 1;      // the axes hold the pair as found too
        for (const auto* q : { &now, &was }) for (const auto& side : *q) for (const auto& [ms, v] : side) { last = std::max (last, ms); lo = std::min (lo, v); hi = std::max (hi, v); }
        const double span = std::max (1.15 * last, 2.0);
        P.x0 = -0.06 * span; P.x1 = span; P.y0 = lo < 0 ? 1.15 * lo : 0.0; P.y1 = 1.15 * hi; P.xLabel = "ms"; P.yLabel = "level"; P.xTicks = ticks (0, P.x1); P.yTicks = ticks (P.y0, P.y1);
        const double w = 0.014 * (P.x1 - P.x0);
        if (lo < 0) { Curve z; z.colour = 2; z.dim = true; z.pts = { { P.x0, 0.0 }, { P.x1, 0.0 } }; P.curves.push_back (z); }
        for (int ch = 0; ch < 2; ++ch) for (const auto& [ms, v] : now[(size_t) ch]) if (std::abs (v) > 1e-4) P.bars.push_back ({ ms + (ch == 0 ? -0.55 : 0.55) * w, 0.0, v, w, ch });
        if (plain && std::abs (db - da) >= 0.5)
        {
            const bool rightLate = db > da; const double ms = std::min (amount * std::abs (db - da), (double) sides::K) / SR * 1e3, dB = amount * 20 * std::log10 (rightLate ? gb / ga : ga / gb);
            c = std::string (rightLate ? "The right" : "The left") + " side arrives " + num (ms, ms < 10 ? 2 : 1) + " ms after the " + (rightLate ? "left" : "right") + " one";
            if (std::abs (dB) >= 0.5) c += ", " + num (std::abs (dB), 1) + " dB " + (dB < 0 ? "quieter" : "louder");
            c += " (teal left, orange right).";
        }
        else
        {
            auto mainLag = [] (const sides::Taps& v) { int lag = 0; double m = -1; for (const auto& t : v) if (std::abs (t.v) > m) { m = std::abs (t.v); lag = t.lag; } return lag; };
            int reach = 0; for (const auto* v : { &a, &b }) for (const auto& t : *v) reach = std::max (reach, t.lag);
            const int la = mainLag (a), lb = mainLag (b);
            if (f.kind == sides::HAAS) c = std::string ("The strongest part of the ") + (lb > la ? "right" : "left") + " side arrives " + num (std::abs (lb - la) / SR * 1e3, 1) + " ms after that of the " + (lb > la ? "left" : "right") + " one (teal left, orange right).";
            else c = "One sound reaches both sides, with copies of itself up to " + num (reach / SR * 1e3, 1) + " ms later that differ between the sides (teal left, orange right).";
            if (amount >= 1e-3 && std::abs (amount - 1.0) > 0.005) c += " Everything after the direct sound is at " + num (amount * 100, 0) + "% of the level found.";
        }
    }
    const auto spans = sides::where (f); const double all = (double) f.n / SR; double in = 0; for (const auto& sp : spans) in += sp.second - sp.first;
    if (! spans.empty() && sides::share (f) < 0.97 && all >= 1.0)       // where in the sound it is, when a part of the sound (not of its silences) is without it
    {
        const std::string where = onSecond ? "On the first sound it is there " : "It is there ", rest = onSecond ? "; it goes onto the whole of the second sound." : "; the rest of the sound is left alone.";
        if (spans.size() == 1) c += " " + where + "from " + timeText (spans[0].first) + " to " + timeText (spans[0].second) + rest;
        else c += " " + where + "on " + num (in, 0) + " of " + num (all, 0) + " s, in " + std::to_string (spans.size()) + " stretches" + rest;
    }
    if (amount < 1e-3) c = onSecond ? "Amount at zero: not put on the second sound." : "Amount at zero: the one sound behind both sides is left, in the middle.";
    P.caption = c; return P;
}

// The ring engine's group (RING MOD): the gain the sound is multiplied by, over two cycles of the tone, as the faders have
// it (1 = the sound as it is, under the line = upside down).
inline Plot ringPlot (const FoundFx& F, const Params& p, double s, bool onSecond)
{
    using namespace disp;
    Plot P; P.title = F.title; if (F.ring == nullptr || ! F.ring->found || ! (F.ring->f > 0)) return P;
    const ring::Found& f = *F.ring; const double amount = std::clamp (s, 0.0, 2.0), tone = f.f * ringCarrier (p);
    const ring::GainAt now (f, amount), was (f, 1.0); const double span = 2000.0 / tone;       // two cycles, ms
    double lo = std::min ({ 0.0, now.base - now.swing, was.base - was.swing }), hi = std::max ({ 1.0, now.base + now.swing, was.base + was.swing });     // the axis holds the gain as found too
    P.x0 = 0; P.x1 = span; P.y0 = lo < 0 ? 1.12 * lo : 0.0; P.y1 = 1.12 * hi; P.xLabel = "ms"; P.yLabel = "gain"; P.xTicks = ticks (0, span); P.yTicks = ticks (P.y0, P.y1);
    if (lo < 0) { Curve z; z.colour = 2; z.dim = true; z.pts = { { P.x0, 0.0 }, { P.x1, 0.0 } }; P.curves.push_back (z); }
    Curve c; for (int i = 0; i <= 240; ++i) { const double ms = span * i / 240.0; c.pts.push_back ({ ms, now.base + now.swing * std::sin (2 * PI * tone * ms * 1e-3) }); }
    P.curves.push_back (c);
    std::string t = "Multiplied by a tone at " + (tone < 100 ? num (tone, 1) + " Hz" : hz (tone)) + ": every pitch in the sound splits into two, that far above and below.";
    if (now.swing > 1e-9)
    {
        const double plain = now.base / (now.swing / std::sqrt (2.0));          // the plain sound against the split part
        if (plain < 0.03) t += " None of the plain sound is left beside them.";
        else { const double dB = 20 * std::log10 (plain); t += " The plain sound is still there, " + (std::abs (dB) < 0.5 ? std::string ("as loud as the split part.") : num (std::abs (dB), 1) + " dB " + (dB < 0 ? "under" : "over") + " the split part."); }
    }
    if (amount < 1e-3) t = onSecond ? "Amount at zero: not put on the second sound." : "Amount at zero: the tone is taken out (the sound divided by the gain).";
    P.caption = t; return P;
}

// s: the effect's Amount fader (0..2); p: its settings with the settings faders applied. onSecond: the effect
// goes onto a second sound (only the room's picture depends on it).
inline Plot makePlot (const FoundFx& F, const Params& scaledP, double s, bool onSecond = false)
{
    using namespace disp;
    if (F.room != nullptr) return roomPlot (F, scaledP, s, onSecond);
    if (F.tone != nullptr) return tonePlot (F, scaledP, s, onSecond);
    if (F.echo != nullptr) return echoPlot (F, scaledP, s, onSecond);
    if (F.level != nullptr) return levelPlot (F, scaledP, s, onSecond);
    if (F.sweep != nullptr) return flangerPlot (F, scaledP, s, onSecond);
    if (F.moving != nullptr) return movingPlot (F, scaledP, s, onSecond);
    if (F.chorus != nullptr) return chorusPlot (F, scaledP, s, onSecond);
    if (F.sides != nullptr) return sidesPlot (F, s, onSecond);
    if (F.ring != nullptr) return ringPlot (F, scaledP, s, onSecond);
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
