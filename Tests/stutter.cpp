// The stutter engine without the UI (48 kHz WAV in), for comparison with the Python prototype (Tools/stutter_parity.py):
//   drstutter a.wav b.wav prefix
// reads the events off a, prints what it found as one line of JSON, and writes as raw float64 (left then right)
//   prefix.onto.f64    b with the events played (Amount 1);  prefix.onto2.f64  Amount 2;  prefix.onto05.f64  Amount 0.5
//   prefix.fine.f64    b with the events played, the slices half as long (Slice 0.5)
//   prefix.same.f64    a with Amount in the middle (must be a itself);  prefix.own2.f64  Amount 2
//   prefix.own0.f64    a with the repeats cut out (Amount 0);  prefix.own05.f64  Amount 0.5
//   prefix.ownfine.f64 a with the slices half as long (Slice 0.5)
// Nothing found: writes nothing.
//   drstutter find a.wav ...
// only what is found on each sound, one line of JSON each.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include "Wav.h"
#include "../Source/Core/Stutter.h"

static void writeRaw (const std::string& p, const dr::Stereo& s)
{
    std::ofstream f (p, std::ios::binary);
    f.write ((const char*) s.l.data(), (std::streamsize) (s.l.size() * sizeof (double))); f.write ((const char*) s.r.data(), (std::streamsize) (s.r.size() * sizeof (double)));
    if (! f) throw std::runtime_error ("cannot write " + p);
}
static dr::Stereo sound (const std::string& path, double& sr)
{
    dr::Stereo y; sr = 0;
    if (! dr::readWav (path, y, sr)) throw std::runtime_error ("cannot read " + path);
    return y;
}
static void say (const dr::stutter::Found& F, double took)
{
    std::printf ("{\"took\": %.3f, \"found\": %s, \"n\": %zu, \"fades\": %s, \"share\": %.9g, \"events\": [", took, F.found ? "true" : "false", F.n, F.fades ? "true" : "false", F.share);
    for (size_t i = 0; i < F.events.size(); ++i) std::printf ("%s[%ld, %ld, %ld]", i ? ", " : "", F.events[i].s0, F.events[i].L, F.events[i].end);
    std::printf ("], \"own\": [");
    for (size_t i = 0; i < F.own.size(); ++i) std::printf ("%s%ld", i ? ", " : "", F.own[i]);
    std::printf ("], \"left_out\": [");
    for (size_t i = 0; i < F.leftOut.size(); ++i) std::printf ("%s[%ld, %ld, %ld, \"%s\"]", i ? ", " : "", F.leftOut[i].s0, F.leftOut[i].L, F.leftOut[i].end, F.leftOut[i].why.c_str());
    std::printf ("], \"stretches\": [");
    for (size_t i = 0; i < F.stretches.size(); ++i) std::printf ("%s[%ld, %ld, %ld]", i ? ", " : "", F.stretches[i].lo, F.stretches[i].hi, F.stretches[i].L);
    std::printf ("]}\n");
}
int main (int argc, char** argv)
{
    try
    {
        using clock = std::chrono::steady_clock; namespace st = dr::stutter;
        if (argc >= 3 && std::string (argv[1]) == "find")      // only what is found: one line of JSON for each sound
        {
            for (int i = 2; i < argc; ++i) { double sr; const dr::Stereo a = sound (argv[i], sr); const auto t0 = clock::now(); const st::Found F = st::find (a, sr); say (F, std::chrono::duration<double> (clock::now() - t0).count()); std::fflush (stdout); }
            return 0;
        }
        if (argc < 4) { std::cout << "drstutter a.wav b.wav prefix | drstutter find a.wav ...\n"; return 1; }
        double sra, srb; const dr::Stereo a = sound (argv[1], sra), b = sound (argv[2], srb); const std::string pre = argv[3];
        const auto t0 = clock::now(); const st::Found F = st::find (a, sra); say (F, std::chrono::duration<double> (clock::now() - t0).count());
        if (! F.found) return 0;
        writeRaw (pre + ".onto.f64", st::onto (b, F, 1.0)); writeRaw (pre + ".onto2.f64", st::onto (b, F, 2.0)); writeRaw (pre + ".onto05.f64", st::onto (b, F, 0.5));
        writeRaw (pre + ".fine.f64", st::onto (b, F, 1.0, 0.5));
        writeRaw (pre + ".same.f64", st::own (a, F, 1.0)); writeRaw (pre + ".own2.f64", st::own (a, F, 2.0));
        writeRaw (pre + ".own0.f64", st::own (a, F, 0.0)); writeRaw (pre + ".own05.f64", st::own (a, F, 0.5));
        writeRaw (pre + ".ownfine.f64", st::own (a, F, 1.0, 0.5));
        return 0;
    }
    catch (const std::exception& e) { std::cerr << "drstutter: " << e.what() << "\n"; return 1; }
}
