// render ResourcesDir src.wav [target.wav|-] out.wav [fader=value ...]  : the app's processing without the UI
//   More renders of the same first sound in one go (it is analysed once): add  -- [target.wav|-] out.wav [fader=value ...]  as often as needed.
//   out.wav = "-": only what was found is printed (every effect's score, the room engine's verdict), nothing is rendered
//   sure=NN: the Sureness slider (default 65); ir=file.wav: also writes the found room as Export IR does;
//   echoir=file.wav: also writes the found echoes as Export IR does for the DELAY group
//   The room engine is used when ResourcesDir holds room.bin (as in the app), with a second look when it also holds room2.bin.
//   The tone engine and the echo engine always run (the DELAY group's faders: delay.amount, delay.time, delay.feedback, delay.bright).
//   The level engine always runs: a found movement is the TREMOLO, AUTO PAN or GATED RHYTHM group (faders: <group>.amount, <group>.speed).
//   The flanger engine always runs: a found moving copy is the FLANGER group (faders: flanger.amount, flanger.speed, flanger.sweep, flanger.feedback).
//   The chorus engine always runs: found voices are the CHORUS group (faders: chorus.amount, chorus.speed, chorus.depth).
//   The sides engine always runs: a delay or a widening between the two sides is the HAAS WIDTH or STEREO WIDTH group (faders: haas.amount, width.amount).
//   The ring engine always runs: a tone the sound was multiplied by is the RING MOD group (faders: ringmod.amount, ringmod.carrier).
//   The moving-tone engine runs when ResourcesDir holds moving.bin (as in the app): a found moving filter is the PHASER or FILTER SWEEP
//   group (faders: <group>.amount, <group>.speed).
//   The verifier's models are read from ResourcesDir/verify.bin (Source/Core/Verify.h): the scores are its scores, as in the app.
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
    dr::Stereo src; double sr = 0; if (! dr::readWav (argv[2], src, sr) || src.size() == 0) { std::cout << "cannot read " << argv[2] << "\n"; return 1; }
    struct Block { std::string tgt, out, ir, echoIr; std::map<std::string, double> fad; bool match = false; };
    std::vector<Block> blocks; double cut = 0.65;
    for (int i = 3; i + 1 < argc;)
    {
        Block b; b.tgt = argv[i]; b.out = argv[i + 1]; i += 2;
        for (; i < argc && std::string (argv[i]) != "--"; ++i)
        {
            std::string a = argv[i]; auto e = a.find ('=');
            if (a.rfind ("sure=", 0) == 0) cut = std::stod (a.substr (5)) / 100.0; else if (a.rfind ("ir=", 0) == 0) b.ir = a.substr (3); else if (a.rfind ("echoir=", 0) == 0) b.echoIr = a.substr (7); else if (a.rfind ("match=", 0) == 0) b.match = a.substr (6) != "0";
            else b.fad[a.substr (0, e)] = std::stod (a.substr (e + 1));
        }
        blocks.push_back (std::move (b)); if (i < argc) ++i;
    }
    if (blocks.empty()) { std::cout << "usage: render ResourcesDir src.wav [target.wav|-] out.wav [fader=value ...] [-- [target.wav|-] out.wav [fader=value ...]] ...\n"; return 1; }
    const bool only = blocks[0].out == "-";
    auto I = dr::inspect (M, src); dr::mergeNew (I, dr::inspect (M27, src, {}, &I));
    {
        std::ifstream f (dir + "/verify.bin", std::ios::binary); if (! f) { std::cout << "no verify.bin in " << dir << "\n"; return 1; }
        const std::vector<char> b ((std::istreambuf_iterator<char> (f)), {}); dr::Verifier vf; vf.load (b.data(), b.size());
        dr::verify (I, src, vf);        // every effect's score is now the verifier's; reverb, the comb filter and the delay stay with their engines below
    }
    std::shared_ptr<const dr::room::Found> rf; dr::room::Net net, net2;
    {
        std::ifstream f (dir + "/room.bin", std::ios::binary);
        if (f)
        {
            const std::vector<char> b ((std::istreambuf_iterator<char> (f)), {}); net.load (b.data(), b.size());
            std::ifstream f2 (dir + "/room2.bin", std::ios::binary);        // the model for the second look, when there is one
            if (f2) { const std::vector<char> b2 ((std::istreambuf_iterator<char> (f2)), {}); net2.load (b2.data(), b2.size()); }
            auto r = std::make_shared<dr::room::Found> (dr::room::find (src, net, net2.ready() ? &net2 : nullptr));
            const dr::RoomEvidence ev = dr::roomEvidence (*r); std::vector<dr::RoomPiece> pieces;
            const double sure = dr::roomVerdict (src, *r, net, net2.ready() ? &net2 : nullptr, {}, &pieces);       // a sound of more than 9 s is judged in pieces
            std::cout << "room: found=" << r->found << " detector=" << (I.prob.count ("reverb") ? I.prob.at ("reverb") : 0.0) << " sureness=" << sure
                      << " share=" << ev.share << " deep=" << ev.deep << " bands=" << ev.bands << " moments=" << ev.moments << " decay=" << ev.decay << " strength=" << ev.strength << " own=" << ev.own << "\n";
            if (! pieces.empty())
            {
                std::cout << "room in pieces: whole=" << dr::roomSureness (*r) << " pieces_at_s:sureness:weight=";
                for (size_t i = 0; i < pieces.size(); ++i) std::cout << (i ? "," : "") << (double) pieces[i].from / 48000.0 << ":" << pieces[i].sure << ":" << pieces[i].weight;
                std::cout << "\n";
            }
            if (dr::mergeRoom (I, sure)) rf = r;
        }
    }
    const auto tf = std::make_shared<const dr::tone::Found> (dr::tone::analyse (src)); dr::mergeTone (I, *tf);
    std::cout << "tone: reflections=" << tf->echoes.size(); for (auto& e : tf->echoes) std::cout << " delay_ms=" << e.delay / 48.0 << " gain=" << e.gain; std::cout << "\n";
    const auto ef = std::make_shared<const dr::echo::Found> (dr::echo::analyse (src)); dr::mergeEcho (I, *ef);
    std::cout << "echo: sure=" << ef->sure; for (int c = 0; c < 2; ++c) if (ef->ch[c].found) std::cout << (c ? " right" : " left") << "_ms=" << ef->ch[c].rec.q / 48.0 << " p=" << ef->ch[c].rec.p; std::cout << "\n";
    const auto lf = std::make_shared<const dr::level::Movement> (dr::level::analyse (src)); dr::mergeLevel (I, *lf);
    std::cout << "level: found=" << lf->found; if (lf->found) std::cout << " kind=" << dr::levelName (*lf) << " sure=" << lf->sure << " rate_hz=" << lf->rate << " depth_db=" << lf->depth << " pieces=" << lf->carrying << "/" << lf->sounding;
    if (lf->found) { std::cout << " there_s="; for (size_t i = 0; i < lf->spans.size(); ++i) std::cout << (i ? "," : "") << (double) lf->spans[i].a / 48000.0 << ".." << (double) lf->spans[i].b / 48000.0; }       // where along the sound the movement is
    std::cout << "\n";
    const auto wf = std::make_shared<const dr::sweep::Result> (dr::sweep::find (src)); dr::mergeSweep (I, *wf);
    std::cout << "flanger: found=" << wf->found; if (wf->found) { const auto& f = wf->ch[wf->has[0] ? 0 : 1]; std::cout << " gain=" << f.g << " feedback=" << f.fb << " delay_ms=" << f.dmin / 48.0 << ".." << f.dmax / 48.0 << (f.curve.valid ? " periodic rate_hz=" : " free") << (f.curve.valid ? std::to_string (f.curve.rate) : "");
        std::cout << " pieces=" << wf->shown << "/" << wf->sounding << " there_s="; const auto sp = dr::sweep::where (*wf); for (size_t i = 0; i < sp.size(); ++i) std::cout << (i ? "," : "") << (double) sp[i].first / 48000.0 << ".." << (double) sp[i].second / 48000.0; }     // where along the sound the copy is
    std::cout << "\n";
    std::shared_ptr<const dr::moving::Found> mf; dr::moving::Net mnet;
    {
        std::ifstream f (dir + "/moving.bin", std::ios::binary);
        if (f)
        {
            const std::vector<char> b ((std::istreambuf_iterator<char> (f)), {}); mnet.load (b.data(), b.size());
            mf = std::make_shared<const dr::moving::Found> (dr::moving::find (mnet, src)); dr::mergeMoving (I, *mf, wf.get());
            std::cout << "moving filter: found=" << mf->found;
            if (mf->found) { std::cout << " kind=" << mf->kind << " sure=" << mf->sure << " period_s=" << mf->period / dr::moving::FPS << " there_s="; const auto sp = dr::moving::where (*mf); for (size_t i = 0; i < sp.size(); ++i) std::cout << (i ? "," : "") << (double) sp[i].first / 48000.0 << ".." << (double) sp[i].second / 48000.0; }
            std::cout << "\n";
        }
    }
    const auto cf = std::make_shared<const dr::chorus::Found> (dr::chorus::find (src)); dr::mergeChorus (I, *cf, wf.get());
    std::cout << "chorus: found=" << cf->found;
    if (cf->found)
    {
        const dr::chorus::Piece* pc = cf->best(); std::cout << " sure=" << cf->sure << " voices=" << pc->voices[0].size() << "+" << pc->voices[1].size();
        for (int c = 0; c < 2; ++c) for (auto& v : pc->voices[c]) std::cout << " [" << (c ? "R " : "L ") << (v.kind ? "tri " : "sin ") << v.rate << " Hz " << v.mid / dr::chorus::FS * 1e3 << "+-" << std::abs (v.half) / dr::chorus::FS * 1e3 << " ms loud " << v.loud << (v.ahead ? " ahead" : "") << "]";
        std::cout << " there_s="; const auto sp = dr::chorus::where (*cf); for (size_t i = 0; i < sp.size(); ++i) std::cout << (i ? "," : "") << (double) sp[i].first / 48000.0 << ".." << (double) sp[i].second / 48000.0;
    }
    std::cout << "\n";
    dr::More more;
    {
        const auto sd = std::make_shared<const dr::sides::Found> (dr::sides::find (src)); dr::mergeSides (I, *sd, tf.get()); more.sides = sd;
        std::cout << "sides: found=" << (sd->found() ? dr::sidesName (*sd) : "0");
        if (sd->hasPair) { std::cout << " pair taps=" << sd->taps << " apart=" << sd->apart << " reach=" << sd->reach << " made_db=" << sd->made << " covered=" << sd->covered << " inside_db=" << 10.0 * std::log10 (std::max (sd->inside, 1e-30)) << " beyond_db=" << sd->beyond; }
        if (sd->hasCopy) std::cout << " copy share=" << sd->share << " side_db=" << sd->sideDb;
        if (sd->mode == dr::sides::PAIR) for (int c = 0; c < 2; ++c) { std::cout << (c ? " R" : " L"); for (auto& t : (c ? sd->b : sd->a)) std::cout << " " << t.v << "@" << t.lag / 48.0 << "ms"; }
        if (sd->found()) { std::cout << " there_s="; const auto sp = dr::sides::where (*sd); for (size_t i = 0; i < sp.size(); ++i) std::cout << (i ? "," : "") << sp[i].first << ".." << sp[i].second; }
        std::cout << "\n";
    }
    {
        const auto rg = std::make_shared<const dr::ring::Found> (dr::ring::find (src)); dr::mergeRing (I, *rg); more.ring = rg;
        std::cout << "ring: found=" << (rg->found ? 1 : 0);
        if (rg->looked) std::cout << " first_win=" << rg->firstWin << " first_hz=" << rg->firstF;
        if (rg->measured) std::cout << " hz=" << rg->f << " rho=" << rg->rho << " floor=" << rg->eps << " win=" << rg->win << " ideal=" << rg->ideal << " match_db=" << rg->match << " dip_db=" << rg->dip << " offset_db=" << rg->offset << " holes=" << rg->holes;
        std::cout << "\n";
    }
    {       // an echo on a sound that is multiplied by a tone: is it still there with the tone taken out?
        const double s = dr::recheckEchoWithout (I, ef, more);
        if (s >= 0) std::cout << "echo without the tone: sureness=" << s << " counts=" << I.prob.at ("delay") << "\n";
    }
    if (rf != nullptr && rf->found && net.ready())      // a room on a sound that shows other effects: is it still there with those taken out?
    {
        std::string what; const double s = dr::recheckRoomWithout (I, src, ef, lf, wf, mf, cf, more, net, net2.ready() ? &net2 : nullptr, {}, &what);
        if (s >= 0) std::cout << "room without the other effects (" << what << "): sureness=" << s << " counts=" << I.prob.at ("reverb") << "\n";
    }
    auto t1 = clk::now();
    if (only) { for (auto& [f, p] : I.prob) std::cout << f << "=" << p << " "; std::cout << "\n"; return 0; }
    const auto found0 = dr::estimate (I, src, cut, nullptr, rf, tf, ef, lf, wf, mf, cf, more);
    auto t2 = clk::now();
    for (auto& F : found0) { std::cout << F.name << " p=" << F.prob << (F.room ? " (room engine)" : F.tone ? " (tone engine)" : F.echo ? " (echo engine)" : F.level ? " (level engine)" : F.sweep ? " (flanger engine)" : F.moving ? " (moving-tone engine)" : F.chorus ? " (chorus engine)" : F.sides ? " (sides engine)" : F.ring ? " (ring engine)" : ""); for (auto& [k, v] : F.est) std::cout << " " << k << "=" << v; std::cout << "\n"; }
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
        if (! B.echoIr.empty())
            for (auto& F : found) if (F.echo != nullptr)
            {
                const dr::EchoIr ir = dr::echoIr (*F.echo, dr::scaled (F, fad)); const dr::Stereo& h = ir.h;
                std::cout << "echoes as an impulse response " << (double) h.size() / 48000.0 << " s, peak " << h.peak() << ", turned down by " << ir.downDb << " dB\n"; writeWav (B.echoIr, h);
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
