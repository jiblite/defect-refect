// render ResourcesDir src.wav [target.wav|-] out.wav [fader=value ...]  : the app's processing without the UI
//   out.wav = "-": only what was found is printed (every effect's score, the room engine's verdict), nothing is rendered
//   sure=NN: the Sureness slider (default 65); ir=file.wav: also writes the found room as Export IR does
//   The room engine is used when ResourcesDir holds room.bin (as in the app), with a second look when it also holds room2.bin.
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include "Wav.h"
#include "../Source/Core/Estimate.h"
static void writeWav (const std::string& p, const dr::Stereo& y)
{
    std::ofstream f (p, std::ios::binary); const uint32_t n = (uint32_t) y.size(), data = n * 2 * 4;
    auto w32 = [&] (uint32_t v) { f.write ((const char*) &v, 4); }; auto w16 = [&] (uint16_t v) { f.write ((const char*) &v, 2); };
    f.write ("RIFF", 4); w32 (36 + data); f.write ("WAVEfmt ", 8); w32 (16); w16 (3); w16 (2); w32 (48000); w32 (48000 * 8); w16 (8); w16 (32); f.write ("data", 4); w32 (data);
    double pk = std::max (y.peak(), 1e-9); const double g = pk > 0.99 ? 0.99 / pk : 1.0;
    for (uint32_t i = 0; i < n; ++i) { float a = (float) (y.l[i] * g), b = (float) (y.r[i] * g); f.write ((const char*) &a, 4); f.write ((const char*) &b, 4); }
}
int main (int argc, char** argv)
{
    using clk = std::chrono::steady_clock; auto t0 = clk::now();
    auto loadM = [] (const std::string& p) { std::ifstream f (p, std::ios::binary); std::vector<char> b ((std::istreambuf_iterator<char> (f)), {}); dr::Models m; m.load (b.data(), b.size()); return m; };
    const std::string dir = argv[1]; const dr::Models M = loadM (dir + "/models.bin"), M27 = loadM (dir + "/models27.bin");
    dr::Stereo src, tgt; double sr; dr::readWav (argv[2], src, sr);
    const bool two = std::string (argv[3]) != "-"; if (two) dr::readWav (argv[3], tgt, sr);
    std::map<std::string, double> fad; double cut = 0.65; std::string irPath; const bool only = std::string (argv[4]) == "-";
    for (int i = 5; i < argc; ++i)
    {
        std::string a = argv[i]; auto e = a.find ('=');
        if (a.rfind ("sure=", 0) == 0) cut = std::stod (a.substr (5)) / 100.0; else if (a.rfind ("ir=", 0) == 0) irPath = a.substr (3); else fad[a.substr (0, e)] = std::stod (a.substr (e + 1));
    }
    auto I = dr::inspect (M, src); dr::mergeNew (I, dr::inspect (M27, src));
    std::shared_ptr<const dr::room::Found> rf;
    {
        std::ifstream f (dir + "/room.bin", std::ios::binary);
        if (f)
        {
            const std::vector<char> b ((std::istreambuf_iterator<char> (f)), {}); dr::room::Net net, net2; net.load (b.data(), b.size());
            std::ifstream f2 (dir + "/room2.bin", std::ios::binary);        // the model for the second look, when there is one
            if (f2) { const std::vector<char> b2 ((std::istreambuf_iterator<char> (f2)), {}); net2.load (b2.data(), b2.size()); }
            auto r = std::make_shared<dr::room::Found> (dr::room::find (src, net, net2.ready() ? &net2 : nullptr));
            const dr::RoomEvidence ev = dr::roomEvidence (*r);
            std::cout << "room: found=" << r->found << " detector=" << (I.prob.count ("reverb") ? I.prob.at ("reverb") : 0.0) << " sureness=" << dr::roomSureness (*r)
                      << " share=" << ev.share << " deep=" << ev.deep << " bands=" << ev.bands << " moments=" << ev.moments << " decay=" << ev.decay << " strength=" << ev.strength << " own=" << ev.own << "\n";
            if (dr::mergeRoom (I, *r)) rf = r; else std::cout << "room: hardly any difference between left and right, the old reverb handling stays\n";
        }
    }
    auto t1 = clk::now();
    if (only) { for (auto& [f, p] : I.prob) std::cout << f << "=" << p << " "; std::cout << "\n"; return 0; }
    auto found = dr::estimate (I, src, cut, nullptr, rf); auto t2 = clk::now();
    for (auto& F : found) { std::cout << F.name << " p=" << F.prob << (F.room ? " (room engine)" : ""); for (auto& [k, v] : F.est) std::cout << " " << k << "=" << v; std::cout << "\n"; }
    if (! irPath.empty() && rf && rf->found)
    {
        const dr::RoomIr ir = dr::exportIr (*rf, dr::get (fad, "reverb.length", 1.0), dr::get (fad, "reverb.bright", 1.0), dr::get (fad, "reverb.width", 1.0)); const dr::Stereo& h = ir.h;
        std::cout << "impulse response " << (double) h.size() / 48000.0 << " s, rms " << h.rms() << " peak " << h.peak() << ", as found = wet " << ir.levelDb << " dB against dry\n"; writeWav (irPath, h);
    }
    auto out = two ? dr::applyTo (tgt, found, fad) : dr::reshape (src, found, fad); auto t3 = clk::now();
    { double e = 0; size_t hot = 0; for (size_t i = 0; i < out.size(); ++i) { e += out.l[i] * out.l[i] + out.r[i] * out.r[i]; hot += std::max (std::abs (out.l[i]), std::abs (out.r[i])) > 1.0; }
      const dr::Stereo& in = two ? tgt : src;
      std::cout << "level in rms " << in.rms() << " peak " << in.peak() << " | out rms " << out.rms() << " peak " << out.peak() << " over-full-scale samples " << hot << "\n"; }
    writeWav (argv[4], out);
    auto ms = [] (auto a, auto b) { return (int) std::chrono::duration_cast<std::chrono::milliseconds> (b - a).count(); };
    std::cout << "inspect " << ms (t0, t1) << " ms, estimate " << ms (t1, t2) << " ms, render " << ms (t2, t3) << " ms\n";
}
