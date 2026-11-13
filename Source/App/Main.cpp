// Defect Refect: drop a sound, get faders for the effects found on it.
// One sound: left takes the effect out, middle = as found, right = double.
// Two sounds: the first sound's effects go onto the second. Left = not applied, right = double.

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_gui_extra/juce_gui_extra.h>

#include <atomic>
#include <set>
#include <thread>

#include "BinaryData.h"
#include "Core/Display.h"

namespace app
{
namespace theme
{
const juce::Colour bg { 0xff000000 }, panel { 0xff0a0c0f }, lane { 0xff06080a }, outline { 0xff1c222a };
const juce::Colour text { 0xffdce3ea }, textDim { 0xff6d7783 }, teal { 0xff19e3d3 }, orange { 0xffff8a1f };
}

class LookAndFeel : public juce::LookAndFeel_V4
{
public:
    LookAndFeel()
    {
        setColour (juce::ResizableWindow::backgroundColourId, theme::bg);
        setColour (juce::Label::textColourId, theme::text);
        setColour (juce::TextButton::buttonColourId, theme::panel);
        setColour (juce::TextButton::textColourOffId, theme::text);
        setColour (juce::TextButton::buttonOnColourId, theme::teal.withAlpha (0.35f));
        setColour (juce::Slider::textBoxTextColourId, theme::text);
        setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        setColour (juce::Slider::backgroundColourId, theme::outline);
        setColour (juce::TooltipWindow::backgroundColourId, theme::panel);
        setColour (juce::TooltipWindow::textColourId, theme::text);
        setColour (juce::TooltipWindow::outlineColourId, theme::outline);
        setColour (juce::ScrollBar::thumbColourId, theme::textDim);
    }
    void drawLinearSlider (juce::Graphics& g, int x, int y, int w, int h, float pos, float, float, juce::Slider::SliderStyle, juce::Slider& s) override
    {
        const auto col = s.findColour (juce::Slider::thumbColourId);
        const float cy = (float) y + (float) h * 0.5f, mid = (float) x + (float) w * 0.5f;
        g.setColour (theme::outline); g.fillRoundedRectangle ((float) x, cy - 2.0f, (float) w, 4.0f, 2.0f);
        if (s.getProperties().contains ("fromLeft")) { g.setColour (col.withAlpha (0.45f)); g.fillRect ((float) x, cy - 2.0f, pos - (float) x, 4.0f); }
        else
        {
            g.setColour (col.withAlpha (0.6f)); g.fillRect (juce::jmin (mid, pos), cy - 2.0f, std::abs (pos - mid), 4.0f);
            g.setColour (theme::textDim); g.fillRect (mid - 0.5f, cy - 7.0f, 1.0f, 14.0f);
        }
        g.setColour (col); g.fillRoundedRectangle (pos - 5.0f, cy - 8.0f, 10.0f, 16.0f, 2.0f);
    }
};

// ---------------------------------------------------------------- audio
inline bool loadStereo48k (const juce::File& f, dr::Stereo& out, juce::String& err)
{
    juce::AudioFormatManager fm; fm.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> r (fm.createReaderFor (f));
    if (r == nullptr) { err = "Cannot read " + f.getFileName(); return false; }
    if (r->lengthInSamples > (juce::int64) (r->sampleRate * 60 * 20)) { err = "Files up to 20 minutes, please."; return false; }
    const int n = (int) r->lengthInSamples; juce::AudioBuffer<float> b ((int) juce::jmax (1u, r->numChannels), n);
    r->read (&b, 0, n, 0, true, true);
    const int c1 = b.getNumChannels() > 1 ? 1 : 0;
    std::vector<double> L ((size_t) n), R ((size_t) n);
    for (int i = 0; i < n; ++i) { L[(size_t) i] = b.getSample (0, i); R[(size_t) i] = b.getSample (c1, i); }
    if (std::abs (r->sampleRate - dr::SR) > 0.5)
    {
        const double ratio = r->sampleRate / dr::SR; const int m = (int) std::floor (n / ratio);
        auto rs = [&] (const std::vector<double>& x)
        {
            std::vector<float> in (x.begin(), x.end()), o ((size_t) m); juce::WindowedSincInterpolator ip;
            ip.process (ratio, in.data(), o.data(), m, (int) in.size(), 0);
            return std::vector<double> (o.begin(), o.end());
        };
        L = rs (L); R = rs (R);
    }
    out.l = std::move (L); out.r = std::move (R);
    return true;
}

class Player : public juce::AudioSource
{
public:
    void set (std::shared_ptr<const dr::Stereo> s) { const juce::SpinLock::ScopedLockType l (lock); buf = std::move (s); if (buf && pos >= buf->size()) pos = 0; }
    void play (bool p) { playing = p; }
    bool isPlaying() const { return playing; }
    void rewind() { pos = 0; }
    double position() const { const juce::SpinLock::ScopedLockType l (lock); return buf && buf->size() ? (double) pos / (double) buf->size() : 0.0; }
    void seek (double t) { const juce::SpinLock::ScopedLockType l (lock); if (buf) pos = (size_t) juce::jlimit (0.0, (double) buf->size() - 1, t * (double) buf->size()); }
    void prepareToPlay (int, double) override {}
    void releaseResources() override {}
    void getNextAudioBlock (const juce::AudioSourceChannelInfo& info) override
    {
        info.clearActiveBufferRegion();
        const juce::SpinLock::ScopedLockType l (lock);
        if (! playing || ! buf || buf->size() == 0) return;
        auto* L = info.buffer->getWritePointer (0, info.startSample);
        auto* R = info.buffer->getNumChannels() > 1 ? info.buffer->getWritePointer (1, info.startSample) : nullptr;
        for (int i = 0; i < info.numSamples; ++i)
        {
            if (pos >= buf->size()) pos = 0;      // loop
            L[i] = (float) (buf->l[pos] * gain); if (R) R[i] = (float) (buf->r[pos] * gain); ++pos;
        }
    }
    double gain = 1.0;
private:
    mutable juce::SpinLock lock; std::shared_ptr<const dr::Stereo> buf; size_t pos = 0; std::atomic<bool> playing { false };
};

// ---------------------------------------------------------------- drop zone
class DropZone : public juce::Component, public juce::FileDragAndDropTarget, public juce::SettableTooltipClient
{
public:
    DropZone (juce::String t, juce::String h, juce::Colour c) : title (std::move (t)), hint (std::move (h)), col (c)
    {
        addChildComponent (clear); clear.setButtonText ("x"); clear.setTooltip ("Remove this sound");
        clear.onClick = [this] { if (onClear) onClear(); };
    }
    std::function<void (const juce::File&)> onFile;
    std::function<void()> onClear, onBrowse;
    std::function<void (double)> onSeek;
    void setSound (const juce::String& name, std::shared_ptr<const dr::Stereo> s)
    {
        file = name; peaks.clear();
        if (s && s->size())
        {
            const int W = 600; peaks.resize (W);
            for (int i = 0; i < W; ++i)
            {
                const size_t a = s->size() * (size_t) i / W, b = juce::jmax (a + 1, s->size() * (size_t) (i + 1) / W); float m = 0;
                for (size_t j = a; j < b; ++j) m = juce::jmax (m, (float) std::abs (0.5 * (s->l[j] + s->r[j])));
                peaks[(size_t) i] = m;
            }
        }
        clear.setVisible (s != nullptr); repaint();
    }
    void setPlayhead (double t) { if (std::abs (t - head) > 0.001) { head = t; repaint(); } }
    void setStatus (const juce::String& s) { status = s; repaint(); }
    bool isInterestedInFileDrag (const juce::StringArray& f) override { return f.size() > 0 && juce::File (f[0]).hasFileExtension ("wav;aif;aiff;flac;mp3;ogg"); }
    void fileDragEnter (const juce::StringArray&, int, int) override { over = true; repaint(); }
    void fileDragExit (const juce::StringArray&) override { over = false; repaint(); }
    void filesDropped (const juce::StringArray& f, int, int) override { over = false; repaint(); if (onFile) onFile (juce::File (f[0])); }
    void mouseDown (const juce::MouseEvent& e) override
    {
        if (peaks.empty()) { if (onBrowse) onBrowse(); return; }
        if (onSeek && wave.contains (e.getPosition())) onSeek ((double) (e.x - wave.getX()) / juce::jmax (1, wave.getWidth()));
    }
    void resized() override
    {
        clear.setBounds (getWidth() - 30, 8, 22, 22);
        wave = getLocalBounds().reduced (14).withTrimmedTop (30).withTrimmedBottom (18);
    }
    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds().toFloat().reduced (1);
        g.setColour (theme::lane); g.fillRoundedRectangle (r, 6);
        g.setColour (over ? col : theme::outline); g.drawRoundedRectangle (r, 6, over ? 2.0f : 1.0f);
        g.setColour (col); g.setFont (juce::FontOptions (14.0f, juce::Font::bold));
        g.drawText (title, 14, 8, getWidth() - 60, 20, juce::Justification::centredLeft);
        if (peaks.empty())
        {
            g.setColour (theme::textDim); g.setFont (juce::FontOptions (13.0f));
            g.drawFittedText (hint, getLocalBounds().reduced (20), juce::Justification::centred, 3);
            return;
        }
        g.setColour (theme::textDim); g.setFont (juce::FontOptions (12.0f));
        g.drawText (file, 14, getHeight() - 18, getWidth() - 28, 14, juce::Justification::centredLeft);
        g.setColour (col.withAlpha (0.8f)); g.setFont (juce::FontOptions (12.0f));
        g.drawText (status, 14, getHeight() - 18, getWidth() - 28, 14, juce::Justification::centredRight);
        const float cy = (float) wave.getCentreY(), hh = (float) wave.getHeight() * 0.5f; float mx = 1e-6f; for (float p : peaks) mx = juce::jmax (mx, p);
        g.setColour (col.withAlpha (0.55f));
        for (int x = 0; x < wave.getWidth(); ++x)
        {
            const float p = peaks[(size_t) (x * (int) peaks.size() / juce::jmax (1, wave.getWidth()))] / mx * hh;
            g.drawVerticalLine (wave.getX() + x, cy - p, cy + p);
        }
        if (head > 0) { g.setColour (theme::text); g.drawVerticalLine (wave.getX() + (int) (head * wave.getWidth()), (float) wave.getY(), (float) wave.getBottom()); }
    }
private:
    juce::String title, hint, file, status; juce::Colour col; bool over = false; std::vector<float> peaks; double head = 0; juce::Rectangle<int> wave;
    juce::TextButton clear;
};

// ---------------------------------------------------------------- the picture of the selected effect
class DisplayPanel : public juce::Component
{
public:
    // now: the effect as the faders have it; found: as it was found (drawn faintly behind when the faders have moved)
    void set (dr::Plot now, dr::Plot found, juce::String hintText = {}) { plot = std::move (now); ghost = std::move (found); hint = std::move (hintText); repaint(); }
    void paint (juce::Graphics& g) override
    {
        auto r = getLocalBounds().toFloat().reduced (1);
        g.setColour (theme::lane); g.fillRoundedRectangle (r, 6); g.setColour (theme::outline); g.drawRoundedRectangle (r, 6, 1.0f);
        if (plot.title.empty())
        {
            g.setColour (theme::textDim); g.setFont (juce::FontOptions (13.0f));
            g.drawFittedText (hint.isEmpty() ? juce::String ("Click an effect or move one of its faders to see what was found.") : hint, getLocalBounds().reduced (24), juce::Justification::centred, 4);
            return;
        }
        auto area = getLocalBounds().reduced (12);
        g.setColour (theme::text); g.setFont (juce::FontOptions (13.0f, juce::Font::bold));
        g.drawText (juce::String (plot.title), area.removeFromTop (20), juce::Justification::centredLeft);
        auto cap = area.removeFromBottom (48);
        g.setColour (theme::textDim); g.setFont (juce::FontOptions (12.0f));
        g.drawFittedText (juce::String (plot.caption), cap.withTrimmedTop (6), juce::Justification::topLeft, 3);
        if (plot.empty()) return;
        auto pr = area.withTrimmedLeft (38).withTrimmedBottom (20).withTrimmedTop (16).withTrimmedRight (6).toFloat();
        auto X = [&] (double v) { const double u = plot.xLog ? std::log (juce::jmax (v, 1e-9) / plot.x0) / std::log (plot.x1 / plot.x0) : (v - plot.x0) / (plot.x1 - plot.x0); return pr.getX() + (float) u * pr.getWidth(); };
        auto Y = [&] (double v) { return pr.getBottom() - (float) ((v - plot.y0) / (plot.y1 - plot.y0)) * pr.getHeight(); };
        auto label = [] (double v, bool hz)
        {
            if (hz && std::abs (v) >= 1000) return juce::String (v / 1000.0, std::fmod (v, 1000.0) == 0.0 ? 0 : 1) + "k";
            juce::String t (v, std::abs (v) < 10 ? 2 : (std::abs (v) < 100 ? 1 : 0));
            if (t.containsChar ('.')) { while (t.endsWithChar ('0')) t = t.dropLastCharacters (1); if (t.endsWithChar ('.')) t = t.dropLastCharacters (1); }
            return t;
        };
        // grid and axis labels
        g.setFont (juce::FontOptions (10.5f));
        for (double t : plot.xTicks)
        {
            const float x = X (t); if (x < pr.getX() - 1 || x > pr.getRight() + 1) continue;
            g.setColour (theme::outline); g.drawVerticalLine ((int) x, pr.getY(), pr.getBottom());
            g.setColour (theme::textDim); g.drawText (label (t, plot.xLog), (int) x - 30, (int) pr.getBottom() + 2, 60, 14, juce::Justification::centred);
        }
        for (double t : plot.yTicks)
        {
            const float y = Y (t); if (y < pr.getY() - 1 || y > pr.getBottom() + 1) continue;
            g.setColour (theme::outline); g.drawHorizontalLine ((int) y, pr.getX(), pr.getRight());
            g.setColour (theme::textDim); g.drawText (label (t, false), (int) pr.getX() - 40, (int) y - 7, 36, 14, juce::Justification::centredRight);
        }
        g.setColour (theme::textDim);
        g.drawText (juce::String (plot.xLabel), (int) pr.getX(), (int) pr.getBottom() + 2, (int) pr.getWidth(), 14, juce::Justification::centredRight);
        g.drawText (juce::String (plot.yLabel), (int) pr.getX() - 40, (int) pr.getY() - 17, 60, 12, juce::Justification::centredLeft);
        g.setColour (theme::outline); g.drawRect (pr, 1.0f);
        juce::Graphics::ScopedSaveState save (g); g.reduceClipRegion (pr.toNearestInt());
        auto colourOf = [] (int c) { return c == 0 ? theme::teal : c == 1 ? theme::orange : theme::textDim; };
        auto draw = [&] (const dr::Plot& P, float alpha)
        {
            for (auto& b : P.bars)
            {
                const float x = X (b.x), ya = Y (b.y0), yb = Y (b.y1); g.setColour (colourOf (b.colour).withAlpha (alpha));
                if (b.w > 0 && ! plot.xLog) { const float xa = X (b.x - b.w / 2), xb = X (b.x + b.w / 2); g.fillRect (juce::Rectangle<float> (xa, juce::jmin (ya, yb), juce::jmax (2.0f, xb - xa), std::abs (ya - yb))); }
                else { g.fillRect (juce::Rectangle<float> (x - 1.5f, juce::jmin (ya, yb), 3.0f, std::abs (ya - yb))); g.fillEllipse (x - 3.5f, yb - 3.5f, 7.0f, 7.0f); }
            }
            for (auto& c : P.curves)
            {
                if (c.pts.size() < 2) continue;
                juce::Path path; bool first = true;
                for (auto& [px, py] : c.pts) { const float x = X (px), y = juce::jlimit (pr.getY() - 50.0f, pr.getBottom() + 50.0f, Y (py)); if (first) path.startNewSubPath (x, y); else path.lineTo (x, y); first = false; }
                const auto col = colourOf (c.colour);
                if (c.fill && alpha > 0.9f)
                {
                    juce::Path f = path; f.lineTo (X (c.pts.back().first), pr.getBottom() + 50.0f); f.lineTo (X (c.pts.front().first), pr.getBottom() + 50.0f); f.closeSubPath();
                    g.setColour (col.withAlpha (0.14f)); g.fillPath (f);
                }
                g.setColour (col.withAlpha (alpha * (c.dim ? 0.5f : 1.0f))); g.strokePath (path, juce::PathStrokeType (c.dim ? 1.0f : 1.7f));
            }
        };
        if (! ghost.empty()) draw (ghost, 0.28f);
        draw (plot, 1.0f);
    }
private:
    dr::Plot plot, ghost; juce::String hint;
};

// ---------------------------------------------------------------- the faders
class FaderPanel : public juce::Component
{
public:
    std::function<void()> onChange;
    std::function<void()> onFocus;          // another effect was selected (clicked, or one of its faders touched)
    std::map<std::string, double> values;
    std::string focus;                      // the effect whose picture is shown
    void setFocus (const std::string& n) { if (focus == n) return; focus = n; repaint(); if (onFocus) onFocus(); }
    void mouseDown (const juce::MouseEvent& e) override { for (auto& g : groups) if (g.bounds.contains (e.getPosition())) { setFocus (g.name); return; } }
    void paint (juce::Graphics& g) override
    {
        for (auto& grp : groups)
            if (grp.name == focus)
            {
                g.setColour (theme::panel); g.fillRoundedRectangle (grp.bounds.toFloat(), 5.0f);
                g.setColour (accent.withAlpha (0.55f)); g.drawRoundedRectangle (grp.bounds.toFloat().reduced (0.5f), 5.0f, 1.0f);
            }
    }
    void build (const std::vector<dr::FoundFx>& found, bool twoSounds, bool keepValues = false)
    {
        const auto old = values; accent = twoSounds ? theme::orange : theme::teal;
        groups.clear(); rows.clear(); heads.clear(); values.clear();
        for (auto& F : found)
        {
            auto h = std::make_unique<juce::Label>(); h->setText (juce::String (F.title) + "   " + juce::String (juce::roundToInt (F.prob * 100)) + "% sure", juce::dontSendNotification);
            h->setFont (juce::FontOptions (13.0f, juce::Font::bold)); h->setColour (juce::Label::textColourId, theme::text); addAndMakeVisible (*h);
            h->setInterceptsMouseClicks (false, false);
            Group grp; grp.head = h.get(); grp.name = F.name; heads.push_back (std::move (h));
            const std::string fxName = F.name;
            for (auto& k : F.knobs)
            {
                auto row = std::make_unique<Row>();
                row->label.setText (k.label, juce::dontSendNotification); row->label.setFont (juce::FontOptions (12.0f)); row->label.setColour (juce::Label::textColourId, theme::textDim);
                auto& s = row->slider; s.setSliderStyle (juce::Slider::LinearHorizontal); s.setRange (0.0, 2.0, 0.0); s.setValue (1.0, juce::dontSendNotification);
                s.setDoubleClickReturnValue (true, 1.0); s.setTextBoxStyle (juce::Slider::TextBoxRight, true, 64, 18);
                s.setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack); s.setColour (juce::Slider::textBoxTextColourId, theme::textDim);
                s.setColour (juce::Slider::thumbColourId, twoSounds ? theme::orange : theme::teal);
                const bool amount = k.key.empty();
                juce::String tip = k.tip;
                if (amount) tip = twoSounds ? "How much of this effect goes onto the second sound. Middle = as found on the first sound. Left = not applied. Right = double."
                                            : (dr::canRemove (F.name) ? "Middle = as found. Left = take it out (best effort). Right = double."
                                                                      : "Middle = as found. Right = double. Taking this one out is not possible yet, so the left half leaves it as found.");
                else tip += " Middle = as found. Left = none. Right = double. Double-click for the middle.";
                s.setTooltip (tip); row->label.setTooltip (tip);
                s.textFromValueFunction = [] (double v) { return std::abs (v - 1.0) < 0.005 ? juce::String ("as found") : juce::String (juce::roundToInt (v * 100)) + "%"; };
                s.updateText();
                const std::string id = k.id; values[id] = 1.0;
                if (keepValues) { auto it = old.find (id); if (it != old.end()) { values[id] = it->second; s.setValue (it->second, juce::dontSendNotification); } }
                s.onValueChange = [this, id, fxName, &s] { values[id] = s.getValue(); setFocus (fxName); if (onChange) onChange(); };
                s.onDragStart = [this, fxName] { setFocus (fxName); };
                row->label.setInterceptsMouseClicks (false, false);
                addAndMakeVisible (row->label); addAndMakeVisible (s);
                grp.rows.push_back (row.get()); rows.push_back (std::move (row));
            }
            groups.push_back (grp);
        }
        if (found.empty()) groups.clear();
        layout();
        // keep the selection if that effect is still shown, otherwise select the first one
        bool have = false; for (auto& g : groups) have = have || g.name == focus;
        if (! have) focus = groups.empty() ? std::string() : groups.front().name;
        repaint(); if (onFocus) onFocus();
    }
    void resetAll() { for (auto& r : rows) r->slider.setValue (1.0, juce::sendNotificationSync); }
    int count() const { return (int) rows.size(); }
    void layout()
    {
        // groups flow in columns
        const int colW = 322, gap = 22, rowH = 24, headH = 26; const int cols = juce::jmax (1, (getParentWidth() - 14) / (colW + gap));
        std::vector<int> colY ((size_t) cols, 4);
        for (auto& g : groups)
        {
            const size_t c = (size_t) (std::min_element (colY.begin(), colY.end()) - colY.begin()); const int x = 8 + (int) c * (colW + gap); int y = colY[c];
            const int top = y;
            g.head->setBounds (x, y, colW, headH); y += headH;
            for (auto* r : g.rows) { r->label.setBounds (x, y, 92, rowH); r->slider.setBounds (x + 92, y, colW - 92, rowH); y += rowH; }
            g.bounds = { x - 8, top - 2, colW + 14, y - top + 6 };
            colY[c] = y + 14;
        }
        setSize (juce::jmax (getParentWidth(), 10), *std::max_element (colY.begin(), colY.end()) + 10);
    }
private:
    struct Row { juce::Label label; juce::Slider slider; };
    struct Group { juce::Label* head = nullptr; std::vector<Row*> rows; std::string name; juce::Rectangle<int> bounds; };
    juce::Colour accent { theme::teal };
    std::vector<std::unique_ptr<Row>> rows; std::vector<std::unique_ptr<juce::Label>> heads; std::vector<Group> groups;
};

// ---------------------------------------------------------------- main screen
class Main : public juce::Component, private juce::Timer
{
public:
    Main() : zoneA ("FIRST SOUND", "Drop a WAV here (or click). Its effects become faders.", theme::teal),
             zoneB ("SECOND SOUND", "Optional: drop a second WAV here (or click). The first sound's effects then go onto it.", theme::orange)
    {
        setLookAndFeel (&lnf);
        for (auto* z : { &zoneA, &zoneB }) addAndMakeVisible (*z);
        zoneA.onFile = [this] (const juce::File& f) { load (0, f); };
        zoneB.onFile = [this] (const juce::File& f) { load (1, f); };
        zoneA.onBrowse = [this] { browse (0); }; zoneB.onBrowse = [this] { browse (1); };
        zoneA.onClear = [this] { clearSlot (0); }; zoneB.onClear = [this] { clearSlot (1); };
        zoneA.onSeek = zoneB.onSeek = [this] (double t) { player.seek (t); };
        for (auto* b : { &swapBtn, &playBtn, &abBtn, &exportBtn, &resetBtn }) addAndMakeVisible (*b);
        swapBtn.setButtonText ("Swap"); swapBtn.setTooltip ("Swap the two sounds. The new first sound is inspected.");
        swapBtn.onClick = [this] { swapSlots(); };
        playBtn.setButtonText ("Play"); playBtn.setTooltip ("Play or stop the result (space bar). Loops.");
        playBtn.onClick = [this] { togglePlay(); };
        abBtn.setButtonText ("Result"); abBtn.setClickingTogglesState (true); abBtn.setTooltip ("Listen to the result, or to the untouched sound to compare.");
        abBtn.onClick = [this] { abBtn.setButtonText (abBtn.getToggleState() ? "Original" : "Result"); updatePlayerBuffer(); };
        exportBtn.setButtonText ("Export"); exportBtn.setTooltip ("Save the result as a 24-bit 48 kHz WAV.");
        exportBtn.onClick = [this] { exportResult(); };
        resetBtn.setButtonText ("Reset Faders"); resetBtn.setTooltip ("All faders back to the middle (as found).");
        resetBtn.onClick = [this] { faders.resetAll(); };
        addAndMakeVisible (modeLabel); modeLabel.setFont (juce::FontOptions (13.0f)); modeLabel.setColour (juce::Label::textColourId, theme::textDim);
        addAndMakeVisible (sureLabel); sureLabel.setText ("Sureness", juce::dontSendNotification); sureLabel.setFont (juce::FontOptions (12.0f)); sureLabel.setColour (juce::Label::textColourId, theme::textDim);
        addAndMakeVisible (sure); sure.setSliderStyle (juce::Slider::LinearHorizontal); sure.setRange (30.0, 99.0, 1.0); sure.setValue (65.0, juce::dontSendNotification); sure.getProperties().set ("fromLeft", true);
        sure.setDoubleClickReturnValue (true, 65.0); sure.setTextBoxStyle (juce::Slider::TextBoxRight, true, 64, 18); sure.setTextValueSuffix ("%");
        sure.setColour (juce::Slider::thumbColourId, theme::text); sure.setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack); sure.setColour (juce::Slider::textBoxTextColourId, theme::textDim);
        {
            const juce::String tip = "How sure the app must be before it shows an effect. Left shows more effects (including guesses), right shows only the clear ones. "
                                     "The six newer effects (saturation, exciter, ring mod, filter, stutter, gated rhythm) need 15% more. Double-click for 65%.";
            sure.setTooltip (tip); sureLabel.setTooltip (tip);
        }
        sure.onValueChange = [this] { sureQueued = true; sureDelay = 6; };
        addAndMakeVisible (view); view.setViewedComponent (&faders, false); view.setScrollBarsShown (true, false);
        addAndMakeVisible (display);
        faders.onChange = [this] { renderSoon(); updateDisplay(); };
        faders.onFocus = [this] { updateDisplay(); };
        dev.initialiseWithDefaultDevices (0, 2);
        { auto setup = dev.getAudioDeviceSetup(); setup.sampleRate = 48000; dev.setAudioDeviceSetup (setup, true); }
        resampler = std::make_unique<juce::ResamplingAudioSource> (&player, false, 2);
        srcPlayer.setSource (resampler.get()); dev.addAudioCallback (&srcPlayer);
        dev.addChangeListener (&devListener); devListener.fn = [this] { updateRatio(); }; updateRatio();
        setWantsKeyboardFocus (true);
        setSize (1120, 820);
        updateUi();
        // models load in the background (the 12-effect detector and the 27-effect one for the six added in 0.2.0)
        busy = true; modeLabel.setText ("Loading the detector...", juce::dontSendNotification);
        startWorker ([this] { auto m = std::make_shared<dr::Models>(); m->load (BinaryData::models_bin, (size_t) BinaryData::models_binSize);
                             auto m27 = std::make_shared<dr::Models>(); m27->load (BinaryData::models27_bin, (size_t) BinaryData::models27_binSize); models27 = m27; models = m; },
                     [this] { busy = false; updateUi(); if (pending[0].exists()) load (0, pending[0]); if (pending[1].exists()) load (1, pending[1]); });
        startTimerHz (20);
    }
    ~Main() override
    {
        stopTimer(); ++generation; if (worker.joinable()) worker.join();
        dev.removeAudioCallback (&srcPlayer); srcPlayer.setSource (nullptr); setLookAndFeel (nullptr);
    }
    void loadFromCommandLine (const juce::File& a, const juce::File& b) { pending[0] = a; pending[1] = b; }
    void setSureness (double v) { sure.setValue (v, juce::sendNotificationSync); }   // test hook: sure=NN on the command line

    void resized() override
    {
        auto r = getLocalBounds().reduced (16);
        auto top = r.removeFromTop (juce::jmin (260, getHeight() / 3));
        const int sw = 70; auto left = top.removeFromLeft ((top.getWidth() - sw) / 2); auto right = top.removeFromRight (left.getWidth());
        zoneA.setBounds (left); zoneB.setBounds (right); swapBtn.setBounds (top.withSizeKeepingCentre (sw - 12, 30));
        r.removeFromTop (10);
        auto bar = r.removeFromTop (32);
        playBtn.setBounds (bar.removeFromLeft (80)); bar.removeFromLeft (8); abBtn.setBounds (bar.removeFromLeft (90)); bar.removeFromLeft (8);
        exportBtn.setBounds (bar.removeFromRight (90)); bar.removeFromRight (8); resetBtn.setBounds (bar.removeFromRight (110)); bar.removeFromRight (8);
        modeLabel.setBounds (bar.reduced (6, 0));
        r.removeFromTop (8);
        auto sr = r.removeFromTop (26); sureLabel.setBounds (sr.removeFromLeft (92)); sure.setBounds (sr.removeFromLeft (360));
        r.removeFromTop (8);
        // the picture of the selected effect sits to the right of the faders
        const int dw = juce::jlimit (300, 430, (int) (r.getWidth() * 0.38));
        display.setBounds (r.removeFromRight (dw).removeFromTop (juce::jmin (r.getHeight(), 330)));
        r.removeFromRight (10);
        view.setBounds (r); faders.setSize (r.getWidth() - 12, faders.getHeight()); faders.layout();
    }
    void paint (juce::Graphics& g) override
    {
        g.fillAll (theme::bg);
        if (faders.count() == 0)
        {
            g.setColour (theme::textDim); g.setFont (juce::FontOptions (14.0f));
            g.drawFittedText (emptyText, view.getBounds().reduced (40), juce::Justification::centred, 4);
        }
    }
    bool keyPressed (const juce::KeyPress& k) override { if (k == juce::KeyPress::spaceKey) { togglePlay(); return true; } return false; }

private:
    // ---- state
    LookAndFeel lnf; juce::TooltipWindow tips { this, 500 };
    DropZone zoneA, zoneB;
    juce::TextButton swapBtn, playBtn, abBtn, exportBtn, resetBtn;
    juce::Label modeLabel, sureLabel; juce::Slider sure; juce::Viewport view; FaderPanel faders; DisplayPanel display;
    juce::AudioDeviceManager dev; juce::AudioSourcePlayer srcPlayer; Player player; std::unique_ptr<juce::ResamplingAudioSource> resampler;
    struct DevListener : juce::ChangeListener { std::function<void()> fn; void changeListenerCallback (juce::ChangeBroadcaster*) override { if (fn) fn(); } } devListener;
    std::shared_ptr<dr::Models> models, models27;
    std::shared_ptr<const dr::Stereo> sound[2], result;
    juce::String names[2]; juce::File pending[2], lastDir;
    std::vector<dr::FoundFx> found; bool inspected = false;
    std::shared_ptr<const dr::Inspection> inspection; std::map<std::string, dr::FoundFx> cache;   // every effect read so far, by name
    bool sureQueued = false; int sureDelay = 0;
    double cut() const { return sure.getValue() / 100.0; }
    std::thread worker; std::atomic<int> generation { 0 }; bool busy = false, renderQueued = false; int renderDelay = 0;
    juce::String emptyText = "Drop a sound on the left. Faders for the effects found on it appear here.";
    std::unique_ptr<juce::FileChooser> chooser;

    void updateRatio()
    {
        if (auto* d = dev.getCurrentAudioDevice()) resampler->setResamplingRatio (dr::SR / d->getCurrentSampleRate());
    }
    // Background work: one job at a time; a newer job makes older results stale.
    void startWorker (std::function<void()> job, std::function<void()> done)
    {
        if (worker.joinable()) worker.join();
        const int gen = generation.load();
        juce::Component::SafePointer<Main> safe (this);
        worker = std::thread ([job, done, gen, safe, this]
        {
            try { job(); } catch (const std::exception& e) { const juce::String msg (e.what()); juce::MessageManager::callAsync ([safe, msg] { if (safe) safe->modeLabel.setText ("Error: " + msg, juce::dontSendNotification); }); }
            juce::MessageManager::callAsync ([safe, done, gen] { if (safe && safe->generation.load() == gen) done(); });
        });
    }
    void browse (int slot)
    {
        chooser = std::make_unique<juce::FileChooser> ("Open a sound", lastDir, "*.wav;*.aif;*.aiff;*.flac;*.mp3;*.ogg");
        chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                              [this, slot] (const juce::FileChooser& c) { if (c.getResult().existsAsFile()) load (slot, c.getResult()); });
    }
    void load (int slot, const juce::File& f)
    {
        if (! models) { pending[slot] = f; return; }
        lastDir = f.getParentDirectory();
        dr::Stereo s; juce::String err;
        if (! loadStereo48k (f, s, err)) { modeLabel.setText (err, juce::dontSendNotification); return; }
        sound[slot] = std::make_shared<const dr::Stereo> (std::move (s)); names[slot] = f.getFileName();
        (slot == 0 ? zoneA : zoneB).setSound (names[slot], sound[slot]);
        if (slot == 0) inspect(); else modeChanged();
    }
    void clearSlot (int slot)
    {
        sound[slot].reset(); names[slot] = {}; (slot == 0 ? zoneA : zoneB).setSound ({}, nullptr);
        if (slot == 0) { ++generation; busy = false; found.clear(); inspected = false; faders.build ({}, false); result.reset(); updatePlayerBuffer(); updateUi(); }
        else modeChanged();
    }
    void swapSlots()
    {
        std::swap (sound[0], sound[1]); std::swap (names[0], names[1]);
        zoneA.setSound (names[0], sound[0]); zoneB.setSound (names[1], sound[1]);
        if (sound[0]) inspect(); else clearSlot (0);
    }
    void inspect()
    {
        ++generation; busy = true; inspected = false; found.clear(); cache.clear(); inspection.reset(); faders.build ({}, false); updateUi();
        zoneA.setStatus ("inspecting..."); modeLabel.setText ("Looking for effects on the first sound...", juce::dontSendNotification);
        auto src = sound[0]; auto M = models; auto M27 = models27; auto out = std::make_shared<std::vector<dr::FoundFx>>();
        auto insp = std::make_shared<dr::Inspection>(); const double c = cut();
        const int gen = generation.load();
        startWorker ([src, M, M27, out, insp, c, gen, this]
                     {
                         auto alive = [this, gen] (float) { return generation.load() == gen; };
                         auto I = dr::inspect (*M, *src, alive);
                         if (generation.load() != gen) return;
                         dr::mergeNew (I, dr::inspect (*M27, *src, alive));
                         if (generation.load() != gen) return;
                         *out = dr::estimate (I, *src, c); *insp = I;
                     },
                     [this, out, insp]
                     {
                         busy = false; found = *out; inspected = true; inspection = insp;
                         for (auto& F : found) cache[F.name] = F;
                         zoneA.setStatus (juce::String ((int) found.size()) + (found.size() == 1 ? " effect found" : " effects found"));
                         emptyText = "No effects found on this sound (from the 18 it knows so far).";
                         faders.build (found, sound[1] != nullptr); resized(); renderNow(); updateUi();
                     });
    }
    // The Sureness slider moved: show the effects above it, reading the settings of any not read yet.
    void applySureness()
    {
        if (! inspected || ! inspection) return;
        const auto names = dr::namesAbove (*inspection, cut());
        std::set<std::string> missing; for (auto& n : names) if (! cache.count (n)) missing.insert (n);
        auto show = [this, names]
        {
            std::vector<dr::FoundFx> f; for (auto& n : names) f.push_back (cache.at (n));
            dr::pruneFound (f);
            std::vector<std::string> a, b; for (auto& x : f) a.push_back (x.name); for (auto& x : found) b.push_back (x.name);
            if (a == b) return;
            found = f; faders.build (found, sound[1] != nullptr, true); resized();
            zoneA.setStatus (juce::String ((int) found.size()) + (found.size() == 1 ? " effect shown" : " effects shown"));
            renderNow(); updateUi();
        };
        if (missing.empty()) { show(); return; }
        busy = true; ++generation; modeLabel.setText ("Reading the settings of the newly shown effects...", juce::dontSendNotification);
        auto I = inspection; auto src = sound[0]; auto out = std::make_shared<std::vector<dr::FoundFx>>();
        startWorker ([I, src, missing, out] { *out = dr::estimate (*I, *src, 0.0, &missing); },
                     [this, out, show] { busy = false; for (auto& F : *out) cache[F.name] = F; show(); updateUi(); });
    }
    // The picture for the selected effect: as the faders have it now, with the found state faintly behind.
    void updateDisplay()
    {
        const dr::FoundFx* F = nullptr; for (auto& x : found) if (x.name == faders.focus) F = &x;
        if (F == nullptr) { display.set ({}, {}, inspected && found.empty() ? juce::String ("Nothing found to show.") : juce::String()); return; }
        const double s = dr::amountOf (*F, faders.values); const bool moved = dr::settingsMoved (*F, faders.values) || std::abs (s - 1.0) > 1e-9;
        display.set (dr::makePlot (*F, dr::scaled (*F, faders.values), s), moved ? dr::makePlot (*F, F->est, 1.0) : dr::Plot());
    }
    void modeChanged()     // one sound <-> two sounds: the faders mean something else, so they start from the middle
    {
        if (inspected) { faders.build (found, sound[1] != nullptr); resized(); }
        result.reset(); updatePlayerBuffer(); renderNow(); updateUi();
    }
    void renderSoon() { renderQueued = true; renderDelay = 3; }    // ~150 ms after the last move
    void renderNow()
    {
        if (! inspected) { updatePlayerBuffer(); return; }
        if (busy) { renderSoon(); return; }
        const bool two = sound[1] != nullptr; auto src = two ? sound[1] : sound[0]; if (! src) return;
        busy = true; ++generation; const int gen = generation.load();
        auto fx = found; auto vals = faders.values; auto out = std::make_shared<dr::Stereo>();
        modeLabel.setText ("Working...", juce::dontSendNotification);
        startWorker ([src, two, fx, vals, out, gen, this]
                     {
                         auto cancel = [this, gen] { return generation.load() != gen; };
                         *out = two ? dr::applyTo (*src, fx, vals, cancel) : dr::reshape (*src, fx, vals, cancel);
                     },
                     [this, out] { busy = false; result = out; updatePlayerBuffer(); updateUi(); });
    }
    void updatePlayerBuffer()
    {
        const bool orig = abBtn.getToggleState(); const int slot = sound[1] ? 1 : 0;
        auto s = (orig || ! result) ? sound[slot] : result;
        // results come out of the limiter under full scale; an original louder than full scale is turned down
        player.gain = (s && s->peak() > 1.0) ? 0.99 / s->peak() : 1.0; player.set (s);
    }
    void togglePlay()
    {
        if (! sound[0] && ! sound[1]) return;
        player.play (! player.isPlaying()); playBtn.setButtonText (player.isPlaying() ? "Stop" : "Play");
    }
    void exportResult()
    {
        if (! result) return;
        auto res = result; const juce::String base = juce::File::createLegalFileName ((sound[1] ? names[1] : names[0]).upToLastOccurrenceOf (".", false, false) + " refect.wav");
        chooser = std::make_unique<juce::FileChooser> ("Export the result", lastDir.getChildFile (base), "*.wav");
        chooser->launchAsync (juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::warnAboutOverwriting, [this, res] (const juce::FileChooser& c)
        {
            auto f = c.getResult(); if (f == juce::File()) return; f = f.withFileExtension ("wav");
            juce::AudioBuffer<float> b (2, (int) res->size()); const double g = res->peak() > 0.999 ? 0.999 / res->peak() : 1.0;
            for (int i = 0; i < b.getNumSamples(); ++i) { b.setSample (0, i, (float) (res->l[(size_t) i] * g)); b.setSample (1, i, (float) (res->r[(size_t) i] * g)); }
            f.deleteFile(); juce::WavAudioFormat wav;
            std::unique_ptr<juce::AudioFormatWriter> w (wav.createWriterFor (new juce::FileOutputStream (f), dr::SR, 2, 24, {}, 0));
            if (w) { w->writeFromAudioSampleBuffer (b, 0, b.getNumSamples()); modeLabel.setText ("Saved " + f.getFileName(), juce::dontSendNotification); }
            else modeLabel.setText ("Could not write " + f.getFullPathName(), juce::dontSendNotification);
        });
    }
    void updateUi()
    {
        const bool two = sound[1] != nullptr;
        swapBtn.setEnabled (sound[0] || sound[1]); exportBtn.setEnabled (result != nullptr); resetBtn.setEnabled (faders.count() > 0);
        playBtn.setEnabled (sound[0] || sound[1]);
        if (! busy)
        {
            juce::String m;
            if (! sound[0]) m = sound[1] ? "Drop a first sound: its effects go onto the second." : "";
            else if (! inspected) m = "";
            else if (found.empty()) m = "Nothing found to change.";
            else if (two) m = "Two sounds: the first sound's effects on the second. Left = not applied, middle = as found, right = double.";
            else m = "One sound: left = take the effect out, middle = as found, right = double.";
            modeLabel.setText (m, juce::dontSendNotification);
        }
        repaint();
    }
    void timerCallback() override
    {
        if (sureQueued && --sureDelay <= 0 && ! busy) { sureQueued = false; applySureness(); }
        if (renderQueued && --renderDelay <= 0 && ! busy) { renderQueued = false; renderNow(); }
        const double t = player.isPlaying() ? player.position() : 0.0; const bool two = sound[1] != nullptr;
        zoneA.setPlayhead (two ? 0.0 : t); zoneB.setPlayhead (two ? t : 0.0);
    }
};

// test hook: plots=<folder> on the command line writes a picture of every effect's display, as found and with the faders moved
inline void dumpPlots (const juce::File& dir)
{
    dir.createDirectory();
    const std::map<std::string, dr::Params> typical = {
        { "flanger", { { "rate", 0.3 }, { "dmin", 0.5e-3 }, { "dmax", 5e-3 }, { "depth", 0.7 }, { "fb", 0.2 } } },
        { "chorus", { { "rate", 0.8 }, { "base", 15e-3 }, { "width", 4e-3 }, { "mix", 0.5 }, { "voices", 2 } } },
        { "phaser", { { "rate", 0.4 }, { "fmin", 200 }, { "fmax", 3000 }, { "fb", 0.3 }, { "mix", 0.5 }, { "stages", 6 } } },
        { "filtersweep", { { "rate", 0.25 }, { "fmin", 300 }, { "fmax", 5000 }, { "q", 4 } } },
        { "width", { { "amount", 0.6 } } }, { "haas", { { "lag", 720 } } },
        { "delay", { { "ms", 320 }, { "fb", 0.45 }, { "mix", 0.4 }, { "pingpong", 1 } } },
        { "reverb", { { "rt60", 1.8 }, { "mix", 0.35 } } },
        { "distortion", { { "drive", 8 }, { "kind", 1 }, { "top", -1 }, { "drive_scale", 1 } } },
        { "bitcrush", { { "bits", 6 }, { "down", 4 } } },
        { "tremolo", { { "rate", 4 }, { "amp_db", 6 }, { "phase", 0.1 } } }, { "autopan", { { "rate", 0.7 }, { "amp_db", 9 }, { "phase", 0 } } },
        { "saturation", { { "drive", 12 }, { "kind", 1 }, { "tone", 0.5 } } }, { "exciter", { { "amount", 0.4 }, { "freq", 3000 } } },
        { "ringmod", { { "carrier", 220 }, { "mix", 0.8 } } }, { "filter", { { "cutoff", 1500 }, { "resonance", 2.5 }, { "type", 2 }, { "slope", 1 } } },
        { "stutter", { { "slice_ms", 120 }, { "repeats", 4 }, { "prob", 0.4 } } }, { "gatedrhythm", { { "rate", 8 }, { "duty", 0.5 }, { "depth", 0.9 } } } };
    for (auto& [name, est] : typical)
    {
        dr::FoundFx F; F.name = name; F.title = dr::titleFor (name); F.est = est; F.knobs = dr::knobsFor (name);
        for (int moved = 0; moved < 2; ++moved)
        {
            DisplayPanel p; p.setSize (420, 330);
            std::map<std::string, double> v; for (auto& k : F.knobs) v[k.id] = 1.0;
            if (moved) { int i = 0; for (auto& k : F.knobs) v[k.id] = (i++ == 0 ? 1.6 : 1.45); }
            p.set (dr::makePlot (F, dr::scaled (F, v), dr::amountOf (F, v)), moved ? dr::makePlot (F, F.est, 1.0) : dr::Plot());
            juce::Image img = p.createComponentSnapshot (p.getLocalBounds(), true, 2.0f);
            juce::File out = dir.getChildFile (juce::String (name) + (moved ? "-moved" : "") + ".png"); out.deleteFile();
            juce::FileOutputStream os (out); juce::PNGImageFormat png; if (os.openedOk()) png.writeImageToStream (img, os);
        }
    }
}
} // namespace app

class DefectRefectApp : public juce::JUCEApplication
{
public:
    const juce::String getApplicationName() override { return JUCE_APPLICATION_NAME_STRING; }
    const juce::String getApplicationVersion() override { return JUCE_APPLICATION_VERSION_STRING; }
    bool moreThanOneInstanceAllowed() override { return true; }
    void initialise (const juce::String& cmd) override
    {
        window = std::make_unique<Window> (getApplicationName() + "  v" + getApplicationVersion());
        auto args = juce::StringArray::fromTokens (cmd, true); args.trim(); for (auto& a : args) a = a.unquoted();
        auto cwd = juce::File::getCurrentWorkingDirectory();
        for (int i = args.size(); --i >= 0;)
            if (args[i].startsWith ("plots=")) { app::dumpPlots (cwd.getChildFile (args[i].fromFirstOccurrenceOf ("=", false, false))); quit(); return; }
        for (int i = args.size(); --i >= 0;)
            if (args[i].startsWith ("sure="))
            {
                const double v = args[i].fromFirstOccurrenceOf ("=", false, false).getDoubleValue(); args.remove (i);
                juce::Component::SafePointer<juce::Component> c (window->getContentComponent());
                juce::Timer::callAfterDelay (20000, [c, v] { if (auto* m = dynamic_cast<app::Main*> (c.getComponent())) m->setSureness (v); });
            }
        if (args.size() > 0 && args[0].isNotEmpty())
            if (auto* m = dynamic_cast<app::Main*> (window->getContentComponent()))
                m->loadFromCommandLine (cwd.getChildFile (args[0]), args.size() > 1 ? cwd.getChildFile (args[1]) : juce::File());
    }
    void shutdown() override { window.reset(); }
    void systemRequestedQuit() override { quit(); }
private:
    struct Window : juce::DocumentWindow
    {
        explicit Window (const juce::String& name) : DocumentWindow (name, juce::Colours::black, DocumentWindow::allButtons)
        {
            setUsingNativeTitleBar (true); setContentOwned (new app::Main(), true); setResizable (true, true);
            setResizeLimits (820, 600, 4000, 3000); centreWithSize (getWidth(), getHeight()); setVisible (true);
        }
        void closeButtonPressed() override { juce::JUCEApplication::getInstance()->systemRequestedQuit(); }
    };
    std::unique_ptr<Window> window;
};

START_JUCE_APPLICATION (DefectRefectApp)
