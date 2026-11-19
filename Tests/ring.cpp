// The ring engine without the UI (48 kHz WAV in), for comparison with the Python prototype (Tools/ring_parity.py):
//   drring a.wav b.wav prefix
// reads the tone off a, prints what it found as one line of JSON, and writes as raw float64 (left then right)
//   prefix.x.f64       a without the gain
//   prefix.out.f64     a with the tone taken out (Amount 0);  prefix.half.f64  half taken out (Amount 0.5)
//   prefix.twice.f64   a with the tone part twice as strong (Amount 2);  prefix.same.f64  Amount in the middle (must be a itself)
//   prefix.up.f64      a with the tone a fifth higher (Carrier 1.5)
//   prefix.onto.f64    b with the gain put on (Amount 1);  prefix.onto2.f64  Amount 2;  prefix.onto05.f64  Amount 0.5
// Nothing found: writes nothing.
//   drring find a.wav ...
// only what is found on each sound, one line of JSON each.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include "Wav.h"
#include "../Source/Core/Ring.h"

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
static void say (const dr::ring::Found& F, double took)
{
    std::printf ("{\"took\": %.3f, \"found\": %s, \"looked\": %s, \"measured\": %s, \"n\": %zu, \"first_win\": %.12g, \"first_f\": %.12g, \"first_rho\": %.12g, \"first_v\": %.12g, ", took, F.found ? "true" : "false", F.looked ? "true" : "false", F.measured ? "true" : "false", F.n, F.firstWin, F.firstF, F.firstRho, F.firstV);
    std::printf ("\"f\": %.15g, \"w\": %.17g, \"phase\": %.15g, \"rho\": %.15g, \"eps\": %.12g, \"win\": %.12g, \"ideal\": %.12g, \"match\": %.12g, \"dip\": %.12g, \"offset\": %.12g, \"holes\": %.12g, \"passes\": %d}\n", F.f, F.w, F.phase, F.rho, F.eps, F.win, F.ideal, F.match, F.dip, F.offset, F.holes, F.passes);
}
int main (int argc, char** argv)
{
    try
    {
        using clock = std::chrono::steady_clock; namespace rg = dr::ring;
        if (argc >= 3 && std::string (argv[1]) == "find")      // only what is found: one line of JSON for each sound
        {
            for (int i = 2; i < argc; ++i) { const dr::Stereo a = sound (argv[i]); const auto t0 = clock::now(); const rg::Found F = rg::find (a); say (F, std::chrono::duration<double> (clock::now() - t0).count()); std::fflush (stdout); }
            return 0;
        }
        if (argc < 4) { std::cout << "drring a.wav b.wav prefix | drring find a.wav ...\n"; return 1; }
        const dr::Stereo a = sound (argv[1]), b = sound (argv[2]); const std::string pre = argv[3];
        auto t0 = clock::now(); const rg::Found F = rg::find (a); const double took = std::chrono::duration<double> (clock::now() - t0).count(); say (F, took);
        if (! F.found) return 0;
        const dr::Vec xl (F.xl.begin(), F.xl.end()), xr (F.xr.begin(), F.xr.end()); writeRaw (pre + ".x.f64", { &xl, &xr });
        t0 = clock::now();
        const dr::Stereo out = rg::own (a, F, 0.0); writeRaw (pre + ".out.f64", { &out.l, &out.r });
        std::fprintf (stderr, "  taken out: %.2f s\n", std::chrono::duration<double> (clock::now() - t0).count());
        const dr::Stereo half = rg::own (a, F, 0.5); writeRaw (pre + ".half.f64", { &half.l, &half.r });
        const dr::Stereo twice = rg::own (a, F, 2.0); writeRaw (pre + ".twice.f64", { &twice.l, &twice.r });
        const dr::Stereo same = rg::own (a, F, 1.0); writeRaw (pre + ".same.f64", { &same.l, &same.r });
        const dr::Stereo up = rg::own (a, F, 1.0, 1.5); writeRaw (pre + ".up.f64", { &up.l, &up.r });
        const dr::Stereo onto = rg::onto (b, F, 1.0); writeRaw (pre + ".onto.f64", { &onto.l, &onto.r });
        const dr::Stereo onto2 = rg::onto (b, F, 2.0); writeRaw (pre + ".onto2.f64", { &onto2.l, &onto2.r });
        const dr::Stereo onto05 = rg::onto (b, F, 0.5); writeRaw (pre + ".onto05.f64", { &onto05.l, &onto05.r });
        return 0;
    }
    catch (const std::exception& e) { std::cerr << "drring: " << e.what() << "\n"; return 1; }
}
