// render ResourcesDir src.wav [target.wav|-] out.wav [fader=value ...]  : the app's processing without the UI
//   More renders of the same first sound in one go (it is analysed once): add  -- [target.wav|-] out.wav [fader=value ...]  as often as needed.
//   out.wav = "-": only what was found is printed (every effect's score, the room engine's verdict), nothing is rendered
//   sure=NN: the Sureness slider (default 65); ir=file.wav: also writes the found room as Export IR does
//   The room engine is used when ResourcesDir holds room.bin (as in the app), with a second look when it also holds room2.bin.
//   match=1: two sounds with Tone Match switched on (its faders: tonematch.amount, tonematch.detail)
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
    dr::Stereo src; double sr; dr::readWav (argv[2], src, sr);
    struct Block { std::string tgt, out, ir; std::map<std::string, double> fad; bool match = false; };
    std::vector<Block> blocks; double cut = 0.65;
    for (int i = 3; i + 1 < argc;)
    {
        Block b; b.tgt = argv[i]; b.out = argv[i + 1]; i += 2;
        for (; i < argc && std::string (argv[i]) != "--"; ++i)
        {
            std::string a = argv[i]; auto e = a.find ('=');
            if (a.rfind ("sure=", 0) == 0) cut = std::stod (a.substr (5)) / 100.0; else if (a.rfind ("ir=", 0) == 0) b.ir = a.substr (3); else if (a.rfind ("match=", 0) == 0) b.match = a.substr (6) != "0";
            else b.fad[a.substr (0, e)] = std::stod (a.substr (e + 1));
        }
        blocks.push_back (std::move (b)); if (i < argc) ++i;
    }
    if (blocks.empty()) { std::cout << "usage: render ResourcesDir src.wav [target.wav|-] out.wav [fader=value ...] [-- [target.wav|-] out.wav [fader=value ...]] ...\n"; return 1; }
    const bool only = blocks[0].out == "-";
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
            if (dr::mergeRoom (I, *r)) rf = r;
        }
    }
    const auto tf = std::make_shared<const dr::tone::Found> (dr::tone::analyse (src)); dr::mergeTone (I, *tf);
    std::cout << "tone: reflections=" << tf->echoes.size(); for (auto& e : tf->echoes) std::cout << " delay_ms=" << e.delay / 48.0 << " gain=" << e.gain; std::cout << "\n";
    auto t1 = clk::now();
    if (only) { for (auto& [f, p] : I.prob) std::cout << f << "=" << p << " "; std::cout << "\n"; return 0; }
    const auto found0 = dr::estimate (I, src, cut, nullptr, rf, tf);
    auto t2 = clk::now();
    for (auto& F : found0) { std::cout << F.name << " p=" << F.prob << (F.room ? " (room engine)" : F.tone ? " (tone engine)" : ""); for (auto& [k, v] : F.est) std::cout << " " << k << "=" << v; std::cout << "\n"; }
    for (auto& B : blocks)
    {
        const bool two = B.tgt != "-"; dr::Stereo tgt; if (two) dr::readWav (B.tgt, tgt, sr);
        auto found = found0; const auto& fad = B.fad;
        if (two && B.match && ! tf->spec.silent)
        {
            long lag = 0; const double c = dr::tone::pairCoherence (src, tgt, &lag); found.push_back (dr::matchFx (tf, c > dr::tone::sameCoherence));
            std::cout << "tone match: coherence=" << c << (c > dr::tone::sameCoherence ? " (two versions of one recording)" : " (two different sounds)") << " shift=" << lag << "\n";
            auto& F = found.back(); std::cout << F.name << " p=" << F.prob << " (tone engine)"; for (auto& [k, v] : F.est) std::cout << " " << k << "=" << v; std::cout << "\n";
        }
        if (! B.ir.empty() && rf && rf->found)
        {
            const dr::RoomIr ir = dr::exportIr (*rf, dr::get (fad, "reverb.length", 1.0), dr::get (fad, "reverb.bright", 1.0), dr::get (fad, "reverb.width", 1.0)); const dr::Stereo& h = ir.h;
            std::cout << "impulse response " << (double) h.size() / 48000.0 << " s, rms " << h.rms() << " peak " << h.peak() << ", as found = wet " << ir.levelDb << " dB against dry\n"; writeWav (B.ir, h);
        }
        auto out = two ? dr::applyTo (tgt, found, fad) : dr::reshape (src, found, fad);
        { size_t hot = 0; for (size_t i = 0; i < out.size(); ++i) hot += std::max (std::abs (out.l[i]), std::abs (out.r[i])) > 1.0;
          const dr::Stereo& in = two ? tgt : src;
          std::cout << "level in rms " << in.rms() << " peak " << in.peak() << " | out rms " << out.rms() << " peak " << out.peak() << " over-full-scale samples " << hot << "\n"; }
        writeWav (B.out, out);
    }
    auto t3 = clk::now();
    auto ms = [] (auto a, auto b) { return (int) std::chrono::duration_cast<std::chrono::milliseconds> (b - a).count(); };
    std::cout << "inspect " << ms (t0, t1) << " ms, estimate " << ms (t1, t2) << " ms, render " << ms (t2, t3) << " ms\n";
}
