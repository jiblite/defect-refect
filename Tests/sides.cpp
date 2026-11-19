// The sides engine without the UI (48 kHz WAV in), for comparison with the Python prototype (Tools/sides_parity.py):
//   drsides a.wav b.wav prefix
// reads off a what was done between its two sides, prints what it found as one line of JSON, and writes as raw float64
//   prefix.x.f64       the one sound behind both sides (the pair) or the copy put at the sides (the copy)
//   prefix.out.f64     a with the effect taken out (Amount 0), left then right
//   prefix.half.f64    a with the effect half taken out (Amount 0.5);  prefix.twice.f64  a with twice the effect (Amount 2)
//   prefix.same.f64    a with Amount in the middle (must be a itself)
//   prefix.onto.f64    b with the effect put on (Amount 1);  prefix.onto2.f64  twice as strong (Amount 2);  prefix.onto05.f64  half (Amount 0.5)
// Nothing found: writes nothing.
//   drsides find a.wav ...
// only what is found on each sound, one line of JSON each.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include "Wav.h"
#include "../Source/Core/Sides.h"

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
static void taps (const char* name, const dr::sides::Taps& t)
{
    std::printf ("\"%s\": [", name);
    for (size_t i = 0; i < t.size(); ++i) std::printf ("%s[%d, %.12g]", i ? ", " : "", t[i].lag, t[i].v);
    std::printf ("], ");
}
static void say (const dr::sides::Found& F, double took)
{
    std::printf ("{\"took\": %.3f, \"kind\": %s, \"mode\": %s, \"n\": %zu, ", took, F.kind == dr::sides::HAAS ? "\"haas\"" : (F.kind == dr::sides::WIDTH ? "\"width\"" : "null"), F.mode == dr::sides::PAIR ? "\"pair\"" : (F.mode == dr::sides::COPY ? "\"copy\"" : "null"), F.n);
    std::printf ("\"has_pair\": %s, \"has_copy\": %s, \"way\": \"%s\", \"taps\": %d, \"apart\": %d, \"reach\": %d, \"made\": %.9g, \"res\": %.9g, \"fits\": %.9g, \"covered\": %.9g, \"inside\": %.9g, \"beyond\": %.9g, ", F.hasPair ? "true" : "false", F.hasCopy ? "true" : "false", F.way == 0 ? "a" : "b", F.taps, F.apart, F.reach, F.made, F.res, F.fits, F.covered, F.inside, F.beyond);
    taps ("a", F.a); taps ("b", F.b);
    std::printf ("\"share\": %.9g, \"all\": %.9g, \"balance\": %.9g, \"side_db\": %.9g, \"copy_checked\": %s, \"copy_covered\": %.9g, \"copy_inside\": %.9g, \"spans\": [", F.share, F.all, F.balance, F.sideDb, F.copyChecked ? "true" : "false", F.copyCovered, F.copyInside);
    for (size_t i = 0; i < F.spans.size(); ++i) std::printf ("%s[%d, %d]", i ? ", " : "", F.spans[i].first, F.spans[i].second);
    std::printf ("]}\n");
}
int main (int argc, char** argv)
{
    try
    {
        using clock = std::chrono::steady_clock; namespace sd = dr::sides;
        if (argc >= 3 && std::string (argv[1]) == "find")      // only what is found: one line of JSON for each sound
        {
            for (int i = 2; i < argc; ++i) { const dr::Stereo a = sound (argv[i]); const auto t0 = clock::now(); const sd::Found F = sd::find (a); say (F, std::chrono::duration<double> (clock::now() - t0).count()); std::fflush (stdout); }
            return 0;
        }
        if (argc < 4) { std::cout << "drsides a.wav b.wav prefix | drsides find a.wav ...\n"; return 1; }
        const dr::Stereo a = sound (argv[1]), b = sound (argv[2]); const std::string pre = argv[3];
        auto t0 = clock::now(); const sd::Found F = sd::find (a); const double took = std::chrono::duration<double> (clock::now() - t0).count(); say (F, took);
        if (! F.found()) return 0;
        t0 = clock::now();
        const dr::Vec x (F.x.begin(), F.x.end()); writeRaw (pre + ".x.f64", { &x });
        const dr::Stereo out = sd::own (a, F, 0.0); writeRaw (pre + ".out.f64", { &out.l, &out.r });
        std::fprintf (stderr, "  taken out: %.2f s\n", std::chrono::duration<double> (clock::now() - t0).count());
        const dr::Stereo half = sd::own (a, F, 0.5); writeRaw (pre + ".half.f64", { &half.l, &half.r });
        const dr::Stereo twice = sd::own (a, F, 2.0); writeRaw (pre + ".twice.f64", { &twice.l, &twice.r });
        const dr::Stereo same = sd::own (a, F, 1.0); writeRaw (pre + ".same.f64", { &same.l, &same.r });
        t0 = clock::now();
        const dr::Stereo onto = sd::onto (b, F, 1.0); writeRaw (pre + ".onto.f64", { &onto.l, &onto.r });
        std::fprintf (stderr, "  put on: %.2f s\n", std::chrono::duration<double> (clock::now() - t0).count());
        const dr::Stereo onto2 = sd::onto (b, F, 2.0); writeRaw (pre + ".onto2.f64", { &onto2.l, &onto2.r });
        const dr::Stereo onto05 = sd::onto (b, F, 0.5); writeRaw (pre + ".onto05.f64", { &onto05.l, &onto05.r });
        return 0;
    }
    catch (const std::exception& e) { std::cerr << "drsides: " << e.what() << "\n"; return 1; }
}
