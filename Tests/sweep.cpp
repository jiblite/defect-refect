// The moving-filter engine (flanger) without the UI (48 kHz WAV in), for comparison with the Python prototype
// (Tools/sweep_parity.py):
//   drsweep a.wav b.wav prefix
// measures the moving copy on a, prints what it found and the decision's numbers of every candidate line, and writes as raw
// float64 files
//   prefix.d.f64       the copy's delay per frame, left then right (samples; zeros for a channel without a copy)
//   prefix.out.f64     a with the copy taken out (apply s = -1), left then right
//   prefix.half.f64    a with the copy half taken out (s = -0.5)
//   prefix.onto.f64    b with the copy put on (s = +1);  prefix.onto2.f64  with the copy twice as loud (s = +2)
//   prefix.fad.f64     b with the copy put on, the faders moved: speed 1.5, sweep 0.5, feedback +0.2
// and the time each step took. Nothing found: only prints.
//   drsweep both a.wav b.wav outA.f64 outB.f64
// the engine as the app uses it, on a sound of any length: a with its copy taken out, a's copy put on b (raw float64,
// left then right; written only if a copy was found), and one line of JSON about what was found, piece by piece:
// "there" = the share of the sound's frames that have the copy, "spans" = [channel, from s, to s] of each such stretch,
// and per piece "on" (the share of its frames that have it) and "w" (how well the line is seen, per half second).
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include "Wav.h"
#include "../Source/Core/Sweep.h"

static void writeRaw (const std::string& p, const std::vector<const dr::Vec*>& vs)
{
    std::ofstream f (p, std::ios::binary);
    for (auto* v : vs) f.write ((const char*) v->data(), (std::streamsize) (v->size() * sizeof (double)));
    if (! f) throw std::runtime_error ("cannot write " + p);
}
static dr::Stereo sound (const std::string& path)
{
    dr::Stereo y; double sr = 0;
    if (! dr::readWav (path, y, sr)) throw std::runtime_error ("cannot read " + path);
    if (std::abs (sr - 48000.0) > 0.5) throw std::runtime_error (path + " is not at 48 kHz");
    return y;
}
int main (int argc, char** argv)
{
    try
    {
        if (argc >= 6 && std::string (argv[1]) == "both")      // as the app uses the engine, a sound of any length: a without its copy, a's copy on b; one line of JSON
        {
            const dr::Stereo a = sound (argv[2]), b = sound (argv[3]); using clock = std::chrono::steady_clock; const auto t0 = clock::now();
            const dr::sweep::Result R = dr::sweep::find (a); const double took = std::chrono::duration<double> (clock::now() - t0).count();
            if (R.found)
            {
                const dr::Stereo out = dr::sweep::apply (a, R, -1.0); writeRaw (argv[4], { &out.l, &out.r });
                const dr::Stereo onto = dr::sweep::apply (b, R, 1.0); writeRaw (argv[5], { &onto.l, &onto.r });
            }
            const int c = R.has[0] ? 0 : 1; const dr::sweep::Found& f = R.ch[c];
            double there = R.found ? 1.0 : 0.0; if (R.found && ! f.there.empty()) { there = 0; for (double v : f.there) there += v; there /= (double) f.there.size(); }
            std::printf ("{\"took\": %.3f, \"found\": %s, \"shown\": %d, \"sounding\": %d, \"periodic\": %s, \"rate\": %.6f, \"g\": %.6f, \"fb\": %.6f, \"dmin\": %.3f, \"dmax\": %.3f, \"there\": %.4f, \"parts\": [", took, R.found ? "true" : "false", R.shown, R.sounding,
                         R.found && f.curve.valid ? "true" : "false", R.found ? f.curve.rate : 0.0, R.found ? f.g : 0.0, R.found ? f.fb : 0.0, R.found ? f.dmin : 0.0, R.found ? f.dmax : 0.0, there);
            for (size_t j = 0; j < R.parts.size(); ++j)
            {
                const auto& q = R.parts[j]; const int k = q.has[0] ? 0 : 1;
                double on = 1.0; if (q.found && ! q.ch[k].there.empty()) { on = 0; for (double v : q.ch[k].there) on += v; on /= (double) q.ch[k].there.size(); }
                std::printf ("%s{\"a\": %zu, \"b\": %zu, \"found\": %s, \"g\": %.6f, \"fb\": %.6f, \"periodic\": %s, \"rate\": %.6f, \"seen\": %.4f, \"on\": %.4f, \"w\": [", j ? ", " : "", q.a, q.b, q.found ? "true" : "false", q.found ? q.ch[k].g : 0.0, q.found ? q.ch[k].fb : 0.0,
                             q.found && q.ch[k].curve.valid ? "true" : "false", q.found ? q.ch[k].curve.rate : 0.0, q.found ? q.ch[k].seen : 0.0, q.found ? on : 0.0);
                if (q.found)                // how clearly the picture shows the line, half second by half second (-1: nothing sounds there)
                {
                    const auto& w = q.ch[k].weight; const auto& act = q.ch[k].act; const size_t step = 94;
                    for (size_t t = 0; t < w.size(); t += step)
                    {
                        double sum = 0; int na = 0; for (size_t i = t; i < std::min (w.size(), t + step); ++i) if (act[i]) { sum += w[i]; ++na; }
                        std::printf ("%s%.3f", t ? ", " : "", na ? sum / na : -1.0);
                    }
                }
                std::printf ("], \"spans\": [");          // where the copy is: channel, from, to (s)
                bool first = true;
                for (int c = 0; q.found && c < 2; ++c)
                {
                    if (! q.has[c]) continue;
                    const auto& th = q.ch[c].there; const size_t T = q.ch[c].d.size(); const double t0 = (double) q.a / 48000.0, fps = 48000.0 / dr::sweep::HOP;
                    for (size_t t = 0; t < T;)
                    {
                        if (! th.empty() && th[t] <= 0) { ++t; continue; }
                        size_t e = t; while (e < T && (th.empty() || th[e] > 0)) ++e;
                        std::printf ("%s[%d, %.2f, %.2f]", first ? "" : ", ", c, t0 + (double) t / fps, t0 + (double) e / fps); first = false; t = e;
                    }
                }
                std::printf ("]}");
            }
            std::printf ("]}\n"); return 0;
        }
        if (argc < 4) { std::cout << "drsweep a.wav b.wav prefix | drsweep both a.wav b.wav outA.f64 outB.f64\n"; return 1; }
        const dr::Stereo a = sound (argv[1]), b = sound (argv[2]); const std::string pre = argv[3];
        using clock = std::chrono::steady_clock; auto t0 = clock::now();
        auto lap = [&] (const char* what) { const auto t1 = clock::now(); std::printf ("  %s: %.2f s\n", what, std::chrono::duration<double> (t1 - t0).count()); t0 = t1; };
        const dr::sweep::Result R = dr::sweep::find (a); lap ("find");
        std::printf ("found %d\n", R.found ? 1 : 0);
        for (int c = 0; c < 2; ++c)
            for (auto& f : R.all[c])
                std::printf ("cand ch%d %s followed %d c1first %.17g move %.17g seen %.17g g %.17g fb %.17g res %.17g level %.17g expect %.17g levelB %.17g still %.17g jumps %.17g dmax %.17g curve %d rate %.17g lo %.17g hi %.17g accept %d\n", c, f.moving ? "moving" : "whole", f.followed ? 1 : 0,
                             f.c1First, f.move, f.seen, f.g, f.fb, f.res, f.level, f.expect, f.levelB, f.still, f.jumps, f.dmax, f.curve.valid ? f.curve.kind : -1, f.curve.rate, f.curve.lo, f.curve.hi, dr::sweep::accept (f) ? 1 : 0);
        if (! R.found) return 0;
        for (int c = 0; c < 2; ++c) if (R.has[c]) std::printf ("copy ch%d g %.6f fb %.6f delay %.4f .. %.4f samples %s\n", c, R.ch[c].g, R.ch[c].fb, R.ch[c].dmin, R.ch[c].dmax, R.ch[c].curve.valid ? (R.ch[c].curve.kind == 0 ? "sine" : "triangle") : "free");
        const size_t T = a.size() / dr::sweep::HOP + 1; const dr::Vec zero (T, 0.0);
        writeRaw (pre + ".d.f64", { R.has[0] ? &R.ch[0].d : &zero, R.has[1] ? &R.ch[1].d : &zero });
        const dr::Stereo out = dr::sweep::apply (a, R, -1.0); lap ("taken out"); writeRaw (pre + ".out.f64", { &out.l, &out.r });
        const dr::Stereo half = dr::sweep::apply (a, R, -0.5); writeRaw (pre + ".half.f64", { &half.l, &half.r });
        const dr::Stereo onto = dr::sweep::apply (b, R, 1.0); lap ("put on"); writeRaw (pre + ".onto.f64", { &onto.l, &onto.r });
        const dr::Stereo onto2 = dr::sweep::apply (b, R, 2.0); writeRaw (pre + ".onto2.f64", { &onto2.l, &onto2.r });
        const dr::Stereo fad = dr::sweep::apply (b, R, 1.0, 1.5, 0.5, 0.2); writeRaw (pre + ".fad.f64", { &fad.l, &fad.r });
        const dr::sweep::Check C = dr::sweep::check (a, R); lap ("check");
        std::printf ("check ok %d res %.6f %.6f level %.6f %.6f expect %.6f %.6f\n", C.ok ? 1 : 0, C.res[0], C.res[1], C.level[0], C.level[1], C.expect[0], C.expect[1]);
        return 0;
    }
    catch (const std::exception& e) { std::cerr << "drsweep: " << e.what() << "\n"; return 1; }
}
