#include "raw_nanoaod_writer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_map>

#include "skelana/pscluj.hpp"

namespace sk = skelana;

// PSHMC: SKELANA fortran entry point that reads the DELPHI simulation banks
// (LQ(LDTOP-28)/(LDTOP-29) on shortDST, the fullDST equivalents otherwise)
// and populates the PSCLUJ common block with a LUJETS-style event record.
// We call it from fillGenPart(); we do NOT otherwise run any SKELANA logic
// from this PHDST-level writer.
extern "C" void pshmc_();

namespace
{
std::string trimBankName(std::string name)
{
    while (!name.empty() && name.back() == ' ')
    {
        name.pop_back();
    }
    return name;
}

std::string hollerith4(int word)
{
    char bytes[4];
    std::memcpy(bytes, &word, sizeof(bytes));
    std::string little(bytes, bytes + 4);
    std::string big{bytes[3], bytes[2], bytes[1], bytes[0]};

    auto printableScore = [](const std::string &s) {
        int score = 0;
        for (unsigned char c : s)
        {
            if (std::isupper(c) || std::isdigit(c) || c == ' ')
            {
                ++score;
            }
        }
        return score;
    };

    return printableScore(little) >= printableScore(big) ? little : big;
}

std::string bankName(int l)
{
    return l > 0 ? hollerith4(ph::IQ(l - 4)) : "";
}

int bankNLinks(int l)
{
    return l > 0 ? ph::IQ(l - 2) : 0;
}

int bankNData(int l)
{
    return l > 0 ? ph::IQ(l - 1) : 0;
}

bool isRawLikeBank(const std::string &name)
{
    const std::string trimmed = trimBankName(name);
    return trimmed == "RAW" || (!trimmed.empty() && trimmed[0] == 'R');
}
} // namespace

// -----------------------------------------------------------------------------
// Field-registration helpers, copied from delphi-nanoaod/src/nanoaod_writer.cpp.
// Keeping the exact same idiom so the two writers look identical to read.
// -----------------------------------------------------------------------------
template <typename T>
static void MakeField(std::unique_ptr<RNTupleModel> &model,
                      const std::string &name, const std::string &description,
                      std::shared_ptr<T> &ptr)
{
    ptr = model->MakeField<T>({name, description});
}

// -----------------------------------------------------------------------------
// Singleton plumbing. phdst::Analysis::instance_ is the static the user00_()
// shim in phdst_analysis.cpp dispatches through, so the override chain works
// transparently.
// -----------------------------------------------------------------------------
RawNanoAODWriter::RawNanoAODWriter() : output_("raw_nanoaod.root") {}
RawNanoAODWriter::~RawNanoAODWriter() = default;

RawNanoAODWriter *RawNanoAODWriter::getInstance()
{
    if (instance_ == nullptr)
    {
        instance_ = new RawNanoAODWriter();
    }
    return static_cast<RawNanoAODWriter *>(instance_);
}

void RawNanoAODWriter::setOutput(const std::filesystem::path &output)
{
    output_ = output;
}

void RawNanoAODWriter::configureRawBankScan(bool enabled, int maxDepth,
                                            int maxLinks, int maxWordsPerBank,
                                            bool allBanks)
{
    rawBankScanEnabled_ = enabled;
    rawBankMaxDepth_ = std::max(0, maxDepth);
    rawBankMaxLinks_ = std::max(0, maxLinks);
    rawBankMaxWordsPerBank_ = maxWordsPerBank;
    rawBankAllBanks_ = allBanks;
}

// -----------------------------------------------------------------------------
// PHDST callbacks.
// -----------------------------------------------------------------------------
void RawNanoAODWriter::user00()
{
    super::user00();

    std::unique_ptr<RNTupleModel> model = RNTupleModel::Create();
    defineEvent(model);
    defineRawBank(model);
    defineEmShower(model);
    defineHadShower(model);
    defineStic(model);
    defineMuidEl(model);
    defineTrac(model);
    defineTrackElement(model);
    defineVdHit(model);
    defineMtpc(model);
    defineBeamSpot(model);
    defineVtx(model);
    defineGenPart(model);

    writer_ = RNTupleWriter::Recreate(std::move(model), "Events", output_.string());
    std::cout << "RawNanoAODWriter: opened " << output_
              << " (Event, RawBank/RawWord, EmShower/Layer, HadShower/Hit, Stic, Muid/ElidRaw, "
              << "TracRaw + TrackElement + Vd{Assoc,Unassoc}Hit + MtpcRaw, "
              << "GenPart)"
              << std::endl;
}

int RawNanoAODWriter::user01()
{
    return super::user01();
}

void RawNanoAODWriter::user02()
{
    super::user02();
    fillEvent();
    fillRawBanks();
    fillEmShowers();
    fillHadShowers();
    fillStic();
    fillMuidEl();
    fillTrac();
    fillTrackElement();
    fillVdHit();
    fillMtpc();
    fillBeamSpot();
    fillVtx();
    fillGenPart();
    if (!writer_)
    {
        std::cerr << "RawNanoAODWriter::user02: writer_ is null!" << std::endl;
        return;
    }
    writer_->Fill();
    static long filled = 0;
    ++filled;
    if (filled <= 5 || filled % 500 == 0 || *nRawBank_ > 0)
    {
        std::cout << "RawNanoAODWriter: filled event " << filled
                  << "  run="    << ph::IIIRUN
                  << "  evt="    << ph::IIIEVT
                  << "  nEmS="   << *nEmShower_
                  << "  nHadS="  << *nHadShower_
                  << "  nStic="  << *nStic_
                  << "  nMuid="  << *nMuidRaw_
                  << "  nElid="  << *nElidRaw_
                  << "  nTrac="  << *nTracRaw_
                  << "  nRawBank=" << *nRawBank_
                  << "  nRawWord=" << *nRawWord_
                  << "  isMC="   << static_cast<int>(*Event_isMC_)
                  << "  nGen="   << *nGenPart_
                  << std::endl;
    }
}

void RawNanoAODWriter::user99()
{
    super::user99();
    if (writer_)
    {
        writer_.reset();   // triggers the RNTuple commit
    }
    std::cout << "RawNanoAODWriter: wrote " << output_ << std::endl;
}

// -----------------------------------------------------------------------------
// Event_* definition + per-event fill (M0 surface).
// -----------------------------------------------------------------------------
void RawNanoAODWriter::defineEvent(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "Event_experimentNumber",   "IIIEXP: experiment number",       Event_experimentNumber_);
    MakeField(model, "Event_runNumber",          "IIIRUN: run number",              Event_runNumber_);
    MakeField(model, "Event_fileSequenceNumber", "IIFILE: file sequence number",    Event_fileSequenceNumber_);
    MakeField(model, "Event_eventNumber",        "IIIEVT: event number",            Event_eventNumber_);
    MakeField(model, "Event_date",               "IIIDAT: event date (yymmdd)",     Event_date_);
    MakeField(model, "Event_time",               "IIITIM: event time (hhmmss)",     Event_time_);
    MakeField(model, "Event_fillNumber",         "IIFILL: LEP fill number",         Event_fillNumber_);
    MakeField(model, "Event_bFieldTesla",        "BPILOT output: solenoid B (Tesla)", Event_bFieldTesla_);
    MakeField(model, "Event_bFieldGevCm",        "BPILOT output: 1/R [1/cm] = BGEVCM / pT [GeV]", Event_bFieldGevCm_);
    MakeField(model, "Event_recordType",         "PHRTY record type, e.g. BOF/CPT/RAW/TAN/DST", Event_recordType_);
    MakeField(model, "Event_isMC",               "1 if the simulation structure (PSCLUJ) is populated for this event, 0 for real data", Event_isMC_);
}

void RawNanoAODWriter::fillEvent()
{
    *Event_experimentNumber_   = ph::IIIEXP;
    *Event_runNumber_          = ph::IIIRUN;
    *Event_fileSequenceNumber_ = ph::IIFILE;
    *Event_eventNumber_        = ph::IIIEVT;
    *Event_date_               = ph::IIIDAT;
    *Event_time_               = ph::IIITIM;
    *Event_fillNumber_         = ph::IIFILL;
    auto bfield = ph::BPILOT();
    *Event_bFieldTesla_ = bfield.first;
    *Event_bFieldGevCm_ = bfield.second;
    *Event_recordType_ = ph::PHRTY();
}

// -----------------------------------------------------------------------------
// RawBank / RawWord -- generic ZEBRA bank manifest for DETRAW raw-DST.
//
// delphi-raw-bank-lister is the human-readable probe. This writer stores the
// same RAW/Rxxx tree in the RNTuple so downstream ML code can first learn which
// detector raw banks are present and, when desired, consume a bounded payload
// word sample without a detector-specific unpacker.
// -----------------------------------------------------------------------------
void RawNanoAODWriter::defineRawBank(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nRawBank", "Number of scanned RAW/Rxxx ZEBRA banks", nRawBank_);
    MakeField(model, "RawBank_name", "Trimmed 4-character ZEBRA bank name", RawBank_name_);
    MakeField(model, "RawBank_nameCode", "Raw Hollerith bank-name word IQ(L-4)", RawBank_nameCode_);
    MakeField(model, "RawBank_top",
        "Top store code: 0=LRTOP, 1=LDTOP, 2=LTTOP, 3=LITOP, 4=LRTINT", RawBank_top_);
    MakeField(model, "RawBank_addr", "ZEBRA link address in the current event store", RawBank_addr_);
    MakeField(model, "RawBank_parentIdx",
        "Index of the closest emitted parent bank, or -1 for a top-level/raw-only orphan", RawBank_parentIdx_);
    MakeField(model, "RawBank_linkIndex", "Link number under the parent bank/top chain", RawBank_linkIndex_);
    MakeField(model, "RawBank_depth", "Depth below the scanned top bank", RawBank_depth_);
    MakeField(model, "RawBank_nlinks", "Bank link count IQ(L-2)", RawBank_nlinks_);
    MakeField(model, "RawBank_ndata", "Bank data-word count IQ(L-1)", RawBank_ndata_);
    MakeField(model, "RawBank_firstRawWordIdx", "First RawWord_* row for this bank", RawBank_firstRawWordIdx_);
    MakeField(model, "RawBank_nRawWords",
        "Number of RawWord_* rows saved for this bank after the per-bank limit", RawBank_nRawWords_);

    MakeField(model, "nRawWord", "Total saved raw bank payload words", nRawWord_);
    MakeField(model, "RawWord_rawBankIdx", "Index into RawBank_*", RawWord_rawBankIdx_);
    MakeField(model, "RawWord_offset", "1-based word offset inside the source ZEBRA bank", RawWord_offset_);
    MakeField(model, "RawWord_i", "Integer view IQ(L+offset) of the payload word", RawWord_i_);
    MakeField(model, "RawWord_f", "Float view Q(L+offset) of the payload word", RawWord_f_);
}

void RawNanoAODWriter::fillRawBanks()
{
    RawBank_name_->clear();
    RawBank_nameCode_->clear();
    RawBank_top_->clear();
    RawBank_addr_->clear();
    RawBank_parentIdx_->clear();
    RawBank_linkIndex_->clear();
    RawBank_depth_->clear();
    RawBank_nlinks_->clear();
    RawBank_ndata_->clear();
    RawBank_firstRawWordIdx_->clear();
    RawBank_nRawWords_->clear();

    RawWord_rawBankIdx_->clear();
    RawWord_offset_->clear();
    RawWord_i_->clear();
    RawWord_f_->clear();

    if (!rawBankScanEnabled_)
    {
        *nRawBank_ = 0;
        *nRawWord_ = 0;
        return;
    }

    std::unordered_set<int> seen;
    scanRawBankTop(0, ph::LRTOP, seen);
    scanRawBankTop(1, ph::LDTOP, seen);
    scanRawBankTop(2, ph::LTTOP, seen);
    scanRawBankTop(3, ph::LITOP, seen);
    scanRawBankTop(4, ph::LRTINT, seen);

    *nRawBank_ = static_cast<std::int32_t>(RawBank_name_->size());
    *nRawWord_ = static_cast<std::int32_t>(RawWord_rawBankIdx_->size());
}

void RawNanoAODWriter::scanRawBankTop(int topCode, int topLink,
                                      std::unordered_set<int> &seen)
{
    if (topLink <= 0)
    {
        return;
    }
    scanRawBankChain(topLink, topCode, -1, 0, 0, seen);
}

void RawNanoAODWriter::scanRawBankChain(int l, int topCode,
                                        std::int32_t parentIdx,
                                        std::int16_t linkIndex,
                                        int depth,
                                        std::unordered_set<int> &seen)
{
    for (int current = l; current > 0; current = ph::LQ(current))
    {
        if (seen.count(current) != 0)
        {
            return;
        }
        seen.insert(current);

        const std::string name = bankName(current);
        const bool emit = rawBankAllBanks_ || isRawLikeBank(name);
        std::int32_t childParentIdx = parentIdx;
        if (emit)
        {
            childParentIdx = appendRawBank(current, topCode, parentIdx,
                                           linkIndex, depth);
        }

        if (depth >= rawBankMaxDepth_)
        {
            continue;
        }

        const int nlinks = std::clamp(bankNLinks(current), 0, rawBankMaxLinks_);
        for (int i = 1; i <= nlinks; ++i)
        {
            const int child = ph::LQ(current - i);
            if (child <= 0)
            {
                continue;
            }
            scanRawBankChain(child, topCode, childParentIdx,
                             static_cast<std::int16_t>(i), depth + 1, seen);
        }
    }
}

std::int32_t RawNanoAODWriter::appendRawBank(int l, int topCode,
                                             std::int32_t parentIdx,
                                             std::int16_t linkIndex,
                                             int depth)
{
    const auto bankIdx = static_cast<std::int32_t>(RawBank_name_->size());
    const int ndata = std::max(0, bankNData(l));
    const int firstWord = static_cast<int>(RawWord_rawBankIdx_->size());
    const int wordLimit = rawBankMaxWordsPerBank_ < 0
        ? ndata
        : std::min(ndata, rawBankMaxWordsPerBank_);

    RawBank_name_->push_back(trimBankName(bankName(l)));
    RawBank_nameCode_->push_back(ph::IQ(l - 4));
    RawBank_top_->push_back(topCode);
    RawBank_addr_->push_back(l);
    RawBank_parentIdx_->push_back(parentIdx);
    RawBank_linkIndex_->push_back(linkIndex);
    RawBank_depth_->push_back(static_cast<std::int16_t>(depth));
    RawBank_nlinks_->push_back(static_cast<std::int16_t>(bankNLinks(l)));
    RawBank_ndata_->push_back(ndata);
    RawBank_firstRawWordIdx_->push_back(firstWord);
    RawBank_nRawWords_->push_back(wordLimit);

    for (int offset = 1; offset <= wordLimit; ++offset)
    {
        RawWord_rawBankIdx_->push_back(bankIdx);
        RawWord_offset_->push_back(offset);
        RawWord_i_->push_back(ph::IQ(l + offset));
        RawWord_f_->push_back(ph::Q(l + offset));
    }

    return bankIdx;
}

// -----------------------------------------------------------------------------
// EM showers / layers — from the EMNC extra-module under every PA (per-track)
// bank. We walk PV -> PA -> EMNC following the SKELANA PSHEMC pattern
// (skelana.car line 3268).
// -----------------------------------------------------------------------------

void RawNanoAODWriter::defineEmShower(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nEmShower",               "Number of EM-cluster showers (HPC detector code 9, FEMC 26)", nEmShower_);
    MakeField(model, "EmShower_paIdx",          "Index (0-based) of the source PA track bank in this event", EmShower_paIdx_);
    MakeField(model, "EmShower_detector",       "IDET: 9 = HPC (barrel), 26 = EMF/FEMC (endcap merged)",      EmShower_detector_);
    MakeField(model, "EmShower_energy",         "Q(LSHOWR+1) shower energy (GeV)",                            EmShower_energy_);
    MakeField(model, "EmShower_position",       "Q(LSHOWR+2..4) shower impact position (cm)",                 EmShower_position_);
    MakeField(model, "EmShower_nLayers",        "Q(LSHOWR+5) number of HPC layers hit",                       EmShower_nLayers_);
    MakeField(model, "EmShower_layerPattern",   "Q(LSHOWR+6) 10-bit HPC layer hit pattern",                   EmShower_layerPattern_);
    MakeField(model, "EmShower_nLayerEnergies", "Number of EmLayer_* rows emitted for this shower",           EmShower_nLayerEnergies_);

    MakeField(model, "nEmLayer",                "Total EM-layer rows across all showers", nEmLayer_);
    MakeField(model, "EmLayer_emShowerIdx",     "Index into EmShower_* (same row order as the parent)",       EmLayer_emShowerIdx_);
    MakeField(model, "EmLayer_layer",           "HPC layer number 1..NHPLAY",                                 EmLayer_layer_);
    MakeField(model, "EmLayer_energy",
        "Per-layer raw HPC deposit (GeV). NOTE: sum over an EmShower's layer rows "
        "is NOT guaranteed to equal EmShower_energy: the latter is the calibrated, "
        "shower-containment-corrected cluster energy, while the former are raw "
        "per-layer readings. For high-energy EM showers the two agree to <<1%, but "
        "they can differ by factors of a few for soft / laterally-spread showers.",
        EmLayer_energy_);
}

void RawNanoAODWriter::fillEmShowers()
{
    EmShower_paIdx_->clear();
    EmShower_detector_->clear();
    EmShower_energy_->clear();
    EmShower_position_->clear();
    EmShower_nLayers_->clear();
    EmShower_layerPattern_->clear();
    EmShower_nLayerEnergies_->clear();

    EmLayer_emShowerIdx_->clear();
    EmLayer_layer_->clear();
    EmLayer_energy_->clear();

    // NHPLAY = number of HPC layers (9 in DELPHI, verified via pschpc.hpp
    // LENHPC=8 having layer hit pattern at index 6 etc.). Use 10 as a
    // generous loop bound to cover historical 9- and 10-layer variants.
    const int kMaxLayersToScan = 10;

    // Walk: LDTOP -> PV bank(s) -> PA bank(s) -> EMNC.
    if (ph::LDTOP <= 0) { *nEmShower_ = 0; *nEmLayer_ = 0; return; }
    int lpv = ph::LQ(ph::LDTOP - 1);
    int paIdx = 0;

    for (; lpv > 0; lpv = ph::LQ(lpv))
    {
        // Each PV links to a chain of PA (per-track) banks via LQ(LPV - 1).
        for (int lpa = ph::LQ(lpv - 1); lpa > 0; lpa = ph::LQ(lpa), ++paIdx)
        {
            int lemnc = ph::LPHPA("EMNC", lpa, 0);
            if (lemnc == 0) continue;

            int nsidet = static_cast<int>(std::lround(ph::Q(lemnc + 2)));
            int nshowr = nsidet % 100;
            int idet   = nsidet / 100;
            int lshowr = lemnc + 2;

            for (int ns = 0; ns < nshowr; ++ns)
            {
                float e      = ph::Q(lshowr + 1);
                float px     = ph::Q(lshowr + 2);
                float py     = ph::Q(lshowr + 3);
                float pz     = ph::Q(lshowr + 4);
                int   nlay   = static_cast<int>(std::lround(ph::Q(lshowr + 5)));
                int   patt   = static_cast<int>(std::lround(ph::Q(lshowr + 6)));

                EmShower_paIdx_->push_back(static_cast<std::int16_t>(paIdx));
                EmShower_detector_->push_back(static_cast<std::int8_t>(idet));
                EmShower_energy_->push_back(e);
                EmShower_position_->push_back(XYZVectorF(px, py, pz));
                EmShower_nLayers_->push_back(static_cast<std::int16_t>(nlay));
                EmShower_layerPattern_->push_back(patt);

                const std::int16_t showerIdx =
                    static_cast<std::int16_t>(EmShower_paIdx_->size() - 1);

                // Per-layer energies follow the shower header when IDET == 9
                // (HPC, older PXDST flat packing). For EMF or newer packed
                // HPC we skip the walk here; those packings will be handled
                // in follow-up milestones.
                int nEmittedLayers = 0;
                if (idet == 9 && nlay > 0 && nlay <= kMaxLayersToScan)
                {
                    int filledBits = 0;
                    for (int nl = 1; nl <= kMaxLayersToScan && filledBits < nlay; ++nl)
                    {
                        int bit = (patt >> (nl - 1)) & 1;
                        if (!bit) continue;
                        ++filledBits;
                        // Layer energy is Q(LSHOWR + 6 + filledBits).
                        float eLayer = ph::Q(lshowr + 6 + filledBits);
                        EmLayer_emShowerIdx_->push_back(showerIdx);
                        EmLayer_layer_->push_back(static_cast<std::int8_t>(nl));
                        EmLayer_energy_->push_back(eLayer);
                        ++nEmittedLayers;
                    }
                }
                EmShower_nLayerEnergies_->push_back(
                    static_cast<std::int16_t>(nEmittedLayers));

                // Advance LSHOWR to the next shower within this EMNC block
                // (mirrors the SKELANA PSHEMC pattern: 6 header words + NLAY
                // per-layer energies for HPC; 4 words only for EMF).
                if (idet == 9)
                {
                    lshowr += 6 + nlay;
                }
                else
                {
                    lshowr += 4;
                }
            }
        }
    }

    *nEmShower_ = static_cast<std::int16_t>(EmShower_paIdx_->size());
    *nEmLayer_  = static_cast<std::int16_t>(EmLayer_emShowerIdx_->size());
}

// -----------------------------------------------------------------------------
// Hadronic calorimeter — PA.HCNC (shortDST) per PSHHAC (skelana.car L3442).
// -----------------------------------------------------------------------------
void RawNanoAODWriter::defineHadShower(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nHadShower",          "Number of HCAL showers (PA.HCNC)",                   nHadShower_);
    MakeField(model, "HadShower_paIdx",     "PA-track index in the event (0-based)",              HadShower_paIdx_);
    MakeField(model, "HadShower_energy",    "Q(LSHOWR+1) shower energy (GeV)",                    HadShower_energy_);
    MakeField(model, "HadShower_direction", "Q(LSHOWR+2..4) shower 3-direction",                  HadShower_direction_);
    MakeField(model, "HadShower_nHits",     "Q(LSHOWR+5) number of layer-hit pairs",              HadShower_nHits_);
    MakeField(model, "HadShower_nHitRows",  "Number of HadHit_* rows emitted for this shower",    HadShower_nHitRows_);

    MakeField(model, "nHadHit",             "Total HadHit rows", nHadHit_);
    MakeField(model, "HadHit_hadShowerIdx", "Index into HadShower_*",  HadHit_hadShowerIdx_);
    MakeField(model, "HadHit_layer",        "HAC layer number (1..N_HCAL_LAYERS) from packed Q",  HadHit_layer_);
    MakeField(model, "HadHit_nTowers",      "Number of towers in this layer-hit from packed Q",   HadHit_nTowers_);
    MakeField(model, "HadHit_energy",       "Energy of this layer-hit (GeV)",                     HadHit_energy_);
}

void RawNanoAODWriter::fillHadShowers()
{
    HadShower_paIdx_->clear();
    HadShower_energy_->clear();
    HadShower_direction_->clear();
    HadShower_nHits_->clear();
    HadShower_nHitRows_->clear();
    HadHit_hadShowerIdx_->clear();
    HadHit_layer_->clear();
    HadHit_nTowers_->clear();
    HadHit_energy_->clear();

    if (ph::LDTOP <= 0) { *nHadShower_ = 0; *nHadHit_ = 0; return; }
    int paIdx = 0;
    for (int lpv = ph::LQ(ph::LDTOP - 1); lpv > 0; lpv = ph::LQ(lpv))
    {
        for (int lpa = ph::LQ(lpv - 1); lpa > 0; lpa = ph::LQ(lpa), ++paIdx)
        {
            int lhcnc = ph::LPHPA("HCNC", lpa, 0);
            if (lhcnc == 0) continue;

            int nshowr = static_cast<int>(std::lround(ph::Q(lhcnc + 2)));
            int lshowr = lhcnc + 2;

            for (int ns = 0; ns < nshowr; ++ns)
            {
                float e      = ph::Q(lshowr + 1);
                float px     = ph::Q(lshowr + 2);
                float py     = ph::Q(lshowr + 3);
                float pz     = ph::Q(lshowr + 4);
                int   nhits  = static_cast<int>(std::lround(ph::Q(lshowr + 5)));

                HadShower_paIdx_->push_back(static_cast<std::int16_t>(paIdx));
                HadShower_energy_->push_back(e);
                HadShower_direction_->push_back(XYZVectorF(px, py, pz));
                HadShower_nHits_->push_back(static_cast<std::int16_t>(nhits));

                const std::int16_t showerIdx =
                    static_cast<std::int16_t>(HadShower_paIdx_->size() - 1);

                int emitted = 0;
                // Sanity-cap: each hit costs 2 words; a shower's nhits
                // shouldn't be negative or absurdly large.
                if (nhits > 0 && nhits < 200)
                {
                    for (int nl = 1; nl <= nhits; ++nl)
                    {
                        float hitE  = ph::Q(lshowr + 5 + 2*nl - 1);
                        int   packed = static_cast<int>(std::lround(ph::Q(lshowr + 5 + 2*nl)));
                        int   layer = packed / 1000;
                        int   ntow  = packed % 1000;

                        HadHit_hadShowerIdx_->push_back(showerIdx);
                        HadHit_layer_->push_back(static_cast<std::int8_t>(layer));
                        HadHit_nTowers_->push_back(static_cast<std::int16_t>(ntow));
                        HadHit_energy_->push_back(hitE);
                        ++emitted;
                    }
                }
                HadShower_nHitRows_->push_back(static_cast<std::int16_t>(emitted));

                lshowr += 5 + 2*std::max(0, nhits);
            }
        }
    }
    *nHadShower_ = static_cast<std::int16_t>(HadShower_paIdx_->size());
    *nHadHit_    = static_cast<std::int16_t>(HadHit_hadShowerIdx_->size());
}

// -----------------------------------------------------------------------------
// STIC — one row per track with a PA.SSTC extra-module, plus MAIN fields.
// See PSHSTC (skelana.car L4180). We save the commonly-used fields only.
// -----------------------------------------------------------------------------
void RawNanoAODWriter::defineStic(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nStic",                   "Number of tracks with a STIC shower (PA.SSTC)", nStic_);
    MakeField(model, "Stic_paIdx",              "PA-track index",                                Stic_paIdx_);
    MakeField(model, "Stic_energyFromMain",     "Q(LMAIN+6) energy reported by the MAIN block",  Stic_energyFromMain_);
    MakeField(model, "Stic_directionFromMain",  "Q(LMAIN+3..5) direction reported by MAIN",      Stic_directionFromMain_);
    MakeField(model, "Stic_numHitTowers",       "Q(LSSTC+2) encoded tower-hit count (/10)",      Stic_numHitTowers_);
}

void RawNanoAODWriter::fillStic()
{
    Stic_paIdx_->clear();
    Stic_energyFromMain_->clear();
    Stic_directionFromMain_->clear();
    Stic_numHitTowers_->clear();

    if (ph::LDTOP <= 0) { *nStic_ = 0; return; }
    int paIdx = 0;
    for (int lpv = ph::LQ(ph::LDTOP - 1); lpv > 0; lpv = ph::LQ(lpv))
    {
        for (int lpa = ph::LQ(lpv - 1); lpa > 0; lpa = ph::LQ(lpa), ++paIdx)
        {
            int lsstc = ph::LPHPA("SSTC", lpa, 0);
            if (lsstc == 0) continue;

            int lmain = ph::LPHPA("MAIN", lpa, 0);
            if (lmain == 0) continue;

            // SSTC payload: header + per-shower block (SKELANA only looks at
            // the first shower, with the loop commented out in PSHSTC).
            int lshowr = lsstc + 2;

            Stic_paIdx_->push_back(static_cast<std::int16_t>(paIdx));
            Stic_energyFromMain_->push_back(ph::Q(lmain + 6));
            Stic_directionFromMain_->push_back(
                XYZVectorF(ph::Q(lmain + 3), ph::Q(lmain + 4), ph::Q(lmain + 5)));
            int rawHitCount = static_cast<int>(std::lround(ph::Q(lshowr + 2) / 10.0));
            Stic_numHitTowers_->push_back(static_cast<std::int16_t>(rawHitCount));
        }
    }
    *nStic_ = static_cast<std::int16_t>(Stic_paIdx_->size());
}

// -----------------------------------------------------------------------------
// Per-track Muon + Electron raw ID (PA.MUID + PA.ELID).
// Mirrors PSCMUD / PSCELD but emitted by our PHDST reader directly, without
// going through SKELANA. Useful when the SKELANA-based pipeline is not in
// play but you still want the standard per-track lepton-ID words.
// -----------------------------------------------------------------------------
void RawNanoAODWriter::defineMuidEl(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nMuidRaw",            "Number of tracks with PA.MUID (muon-ID)", nMuidRaw_);
    MakeField(model, "MuidRaw_paIdx",       "PA-track index",                          MuidRaw_paIdx_);
    MakeField(model, "MuidRaw_tag",         "Q(LMUID+1) NINT: MUCAL2 tag",             MuidRaw_tag_);
    MakeField(model, "MuidRaw_looseChi2",   "Q(LMUID+2) global chi2 of very loose refit", MuidRaw_looseChi2_);
    MakeField(model, "MuidRaw_hitPattern",  "Q(LMUID+3) NINT: hit pattern with inefficiencies", MuidRaw_hitPattern_);

    MakeField(model, "nElidRaw",            "Number of tracks with PA.ELID (electron-ID)", nElidRaw_);
    MakeField(model, "ElidRaw_paIdx",       "PA-track index",                              ElidRaw_paIdx_);
    MakeField(model, "ElidRaw_tag",         "Q(LELID+1) NINT: ELECID tag (0..5)",          ElidRaw_tag_);
    MakeField(model, "ElidRaw_gammaConvTag","Q(LELID+2) NINT: gamma-conversion tag",       ElidRaw_gammaConvTag_);
    MakeField(model, "ElidRaw_refitMomentum","Q(LELID+3..5): electron refit 3-momentum at vertex", ElidRaw_refitMomentum_);
}

void RawNanoAODWriter::fillMuidEl()
{
    MuidRaw_paIdx_->clear();
    MuidRaw_tag_->clear();
    MuidRaw_looseChi2_->clear();
    MuidRaw_hitPattern_->clear();

    ElidRaw_paIdx_->clear();
    ElidRaw_tag_->clear();
    ElidRaw_gammaConvTag_->clear();
    ElidRaw_refitMomentum_->clear();

    if (ph::LDTOP <= 0) { *nMuidRaw_ = 0; *nElidRaw_ = 0; return; }
    int paIdx = 0;
    for (int lpv = ph::LQ(ph::LDTOP - 1); lpv > 0; lpv = ph::LQ(lpv))
    {
        for (int lpa = ph::LQ(lpv - 1); lpa > 0; lpa = ph::LQ(lpa), ++paIdx)
        {
            int lmuid = ph::LPHPA("MUID", lpa, 0);
            if (lmuid > 0)
            {
                MuidRaw_paIdx_->push_back(static_cast<std::int16_t>(paIdx));
                MuidRaw_tag_->push_back(
                    static_cast<std::int32_t>(std::lround(ph::Q(lmuid + 1))));
                MuidRaw_looseChi2_->push_back(ph::Q(lmuid + 2));
                MuidRaw_hitPattern_->push_back(
                    static_cast<std::int32_t>(std::lround(ph::Q(lmuid + 3))));
            }

            int lelid = ph::LPHPA("ELID", lpa, 0);
            if (lelid > 0)
            {
                ElidRaw_paIdx_->push_back(static_cast<std::int16_t>(paIdx));
                ElidRaw_tag_->push_back(
                    static_cast<std::int32_t>(std::lround(ph::Q(lelid + 1))));
                ElidRaw_gammaConvTag_->push_back(
                    static_cast<std::int32_t>(std::lround(ph::Q(lelid + 2))));
                ElidRaw_refitMomentum_->push_back(XYZVectorF(
                    ph::Q(lelid + 3), ph::Q(lelid + 4), ph::Q(lelid + 5)));
            }
        }
    }
    *nMuidRaw_ = static_cast<std::int16_t>(MuidRaw_paIdx_->size());
    *nElidRaw_ = static_cast<std::int16_t>(ElidRaw_paIdx_->size());
}

// -----------------------------------------------------------------------------
// Track raw — PA.TRAC + PA.MAIN (per PSHTRA / PSCTRA).
// One row per charged track. Perigee parameters + weight matrix + track
// length, chi2, detector flags, first-measured-point, number of d.o.f.
// Uncharged tracks are filtered out: SKELANA's PSHTRA gates on
//   IF ( NINT(Q(LMAIN+8)) .NE. 0 )
// and we do the same.
// -----------------------------------------------------------------------------
void RawNanoAODWriter::defineTrac(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nTracRaw",              "Number of charged-track rows (PA.TRAC + PA.MAIN)", nTracRaw_);
    MakeField(model, "TracRaw_paIdx",         "PA-track index in the event",                      TracRaw_paIdx_);
    MakeField(model, "TracRaw_impactRPhi",    "QTRAC(4): impact parameter in R-phi",              TracRaw_impactRPhi_);
    MakeField(model, "TracRaw_impactZ",       "QTRAC(5): impact parameter in Z",                  TracRaw_impactZ_);
    MakeField(model, "TracRaw_theta",         "QTRAC(6): theta at perigee",                       TracRaw_theta_);
    MakeField(model, "TracRaw_phi",           "QTRAC(7): phi at perigee",                         TracRaw_phi_);
    MakeField(model, "TracRaw_invR",          "QTRAC(8): 1/R curvature with sign at perigee",     TracRaw_invR_);
    MakeField(model, "TracRaw_weightMatrix",  "QTRAC(9..23): 15-element symmetric 5x5 weight matrix", TracRaw_weightMatrix_);
    MakeField(model, "TracRaw_trackLength",   "|Q(LMAIN+9)|: track length in cm",                 TracRaw_trackLength_);
    MakeField(model, "TracRaw_detectorsUsed",
        "IQ(LPA+2): PSCTRA documents this as bits 1-VD, 2-ID, 3-TPC, 4-OD, "
        "5-FCA, 6-FCB, but on the 100-event Z->hadrons smoke the observed "
        "distribution had 0 VD / 0 TPC hits and heavy forward-detector "
        "activity, which is clearly not the physical pattern. Treat this "
        "field as raw and decode against the actual PSCTRA in your MC "
        "sample before relying on the bit layout.",
        TracRaw_detectorsUsed_);
    MakeField(model, "TracRaw_firstPointR",   "Q(LMAIN+23..24): R of first measured point",       TracRaw_firstPointR_);
    MakeField(model, "TracRaw_firstPointZ",   "Q(LMAIN+25): Z of first measured point",           TracRaw_firstPointZ_);
    MakeField(model, "TracRaw_chi2NoVD",      "Q(LMAIN+16): track fit chi2 without VD",           TracRaw_chi2NoVD_);
    MakeField(model, "TracRaw_chi2VD",        "Q(LMAIN+26): track fit chi2 with VD",              TracRaw_chi2VD_);
    MakeField(model, "TracRaw_ndfNoVD",       "Q(LMAIN+17): d.o.f. of fit without VD",            TracRaw_ndfNoVD_);
    MakeField(model, "TracRaw_ndfVD",         "Q(LMAIN+27): d.o.f. of fit with VD",               TracRaw_ndfVD_);
    MakeField(model, "TracRaw_chi2VDHits",    "Q(LMAIN+18): chi2 of VD-associated hits",          TracRaw_chi2VDHits_);
    MakeField(model, "TracRaw_charge",        "sign of Q(LMAIN+8): +1 / 0 / -1",                  TracRaw_charge_);
}

void RawNanoAODWriter::fillTrac()
{
    TracRaw_paIdx_->clear();
    TracRaw_impactRPhi_->clear();
    TracRaw_impactZ_->clear();
    TracRaw_theta_->clear();
    TracRaw_phi_->clear();
    TracRaw_invR_->clear();
    TracRaw_weightMatrix_->clear();
    TracRaw_trackLength_->clear();
    TracRaw_detectorsUsed_->clear();
    TracRaw_firstPointR_->clear();
    TracRaw_firstPointZ_->clear();
    TracRaw_chi2NoVD_->clear();
    TracRaw_chi2VD_->clear();
    TracRaw_ndfNoVD_->clear();
    TracRaw_ndfVD_->clear();
    TracRaw_chi2VDHits_->clear();
    TracRaw_charge_->clear();

    if (ph::LDTOP <= 0) { *nTracRaw_ = 0; return; }
    int paIdx = 0;
    for (int lpv = ph::LQ(ph::LDTOP - 1); lpv > 0; lpv = ph::LQ(lpv))
    {
        for (int lpa = ph::LQ(lpv - 1); lpa > 0; lpa = ph::LQ(lpa), ++paIdx)
        {
            int lmain = ph::LPHPA("MAIN", lpa, 0);
            if (lmain == 0) continue;

            // DELPHI charge encoding: NINT(Q(LMAIN+8)) == 1 -> +, == 2 -> -,
            // == 0 -> neutral; anything else is "unknown" (PSHVEC sets 999).
            int chargeCode = static_cast<int>(std::lround(ph::Q(lmain + 8)));
            if (chargeCode == 0) continue;   // neutral -- skip (same gate as PSHTRA)
            std::int8_t signedCharge = (chargeCode == 1) ? 1
                                     : (chargeCode == 2) ? -1
                                     : 0;

            int ltrac = ph::LPHPA("TRAC", lpa, 0);

            TracRaw_paIdx_->push_back(static_cast<std::int16_t>(paIdx));

            // Perigee (Q(LTRAC+2..+6) ↔ QTRAC(4..8)) + weight matrix
            // (LTRAC+7..+21 ↔ QTRAC(9..23)). UCOPY moves 20 floats in PSHTRA.
            if (ltrac > 0)
            {
                TracRaw_impactRPhi_->push_back(ph::Q(ltrac + 2));
                TracRaw_impactZ_->push_back(ph::Q(ltrac + 3));
                TracRaw_theta_->push_back(ph::Q(ltrac + 4));
                TracRaw_phi_->push_back(ph::Q(ltrac + 5));
                TracRaw_invR_->push_back(ph::Q(ltrac + 6));
                std::array<float, 15> w{};
                for (int k = 0; k < 15; ++k) w[k] = ph::Q(ltrac + 7 + k);
                TracRaw_weightMatrix_->push_back(w);
            }
            else
            {
                TracRaw_impactRPhi_->push_back(0.f);
                TracRaw_impactZ_->push_back(0.f);
                TracRaw_theta_->push_back(0.f);
                TracRaw_phi_->push_back(0.f);
                TracRaw_invR_->push_back(0.f);
                TracRaw_weightMatrix_->push_back(std::array<float, 15>{});
            }

            // PSHTRA sign-flips track length depending on primary-vertex
            // participation; we keep the raw magnitude (downstream can
            // re-apply the convention if it wants).
            TracRaw_trackLength_->push_back(std::fabs(ph::Q(lmain + 9)));
            TracRaw_detectorsUsed_->push_back(ph::IQ(lpa + 2));

            // First measured point: R is a 2-norm of (Q+23, Q+24). PSHTRA
            // uses VMOD(Q(LMAIN+23), 2).
            float r23 = ph::Q(lmain + 23);
            float r24 = ph::Q(lmain + 24);
            TracRaw_firstPointR_->push_back(std::sqrt(r23*r23 + r24*r24));
            TracRaw_firstPointZ_->push_back(ph::Q(lmain + 25));

            TracRaw_chi2NoVD_->push_back(ph::Q(lmain + 16));
            TracRaw_chi2VD_->push_back(ph::Q(lmain + 26));

            int ndf1 = static_cast<int>(std::lround(ph::Q(lmain + 17)));
            int ndf2 = static_cast<int>(std::lround(ph::Q(lmain + 27)));
            // SKELANA sanitises this: negative or enormous ndof is set to 0.
            if (ndf1 < 0 || ndf1 > 1000) ndf1 = 0;
            if (ndf2 < 0 || ndf2 > 1000) ndf2 = 0;
            TracRaw_ndfNoVD_->push_back(static_cast<std::int16_t>(ndf1));
            TracRaw_ndfVD_->push_back(static_cast<std::int16_t>(ndf2));
            TracRaw_chi2VDHits_->push_back(ph::Q(lmain + 18));

            TracRaw_charge_->push_back(signedCharge);
        }
    }
    *nTracRaw_ = static_cast<std::int16_t>(TracRaw_paIdx_->size());
}

// -----------------------------------------------------------------------------
// Track elements — M7 payload for track refitting. Same layout across
// PA.TETP(13, TPC), PA.TEID(12, ID), PA.TEOD(14, OD), PA.TEFA(15, FCA),
// PA.TEFB(16, FCB). Each PA track has at most one TE per sub-detector.
// See shortdst.des block PA.TETP(13) for the word layout; we save the
// eight common header words and leave the variable-length error matrix
// and PXDST-251+ footer (nDoF, chi2, length) as a TODO.
// -----------------------------------------------------------------------------
void RawNanoAODWriter::defineTrackElement(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nTrackElement",                 "Per-sub-detector TE rows across all charged tracks",     nTrackElement_);
    MakeField(model, "TrackElement_tracRawIdx",       "Index into TracRaw_* (same event)",                      TrackElement_tracRawIdx_);
    MakeField(model, "TrackElement_subDetector",      "0=TPC (TETP), 1=ID (TEID), 2=OD (TEOD), 3=FCA (TEFA), 4=FCB (TEFB)", TrackElement_subDetector_);
    MakeField(model, "TrackElement_dataDescriptor",   "Q(LTE+2) NINT: RB bitmask, coord system + measurement code", TrackElement_dataDescriptor_);
    MakeField(model, "TrackElement_coord1",           "Q(LTE+3): X or R at reference point",                    TrackElement_coord1_);
    MakeField(model, "TrackElement_coord2",           "Q(LTE+4): Y or R*Phi",                                   TrackElement_coord2_);
    MakeField(model, "TrackElement_coord3",           "Q(LTE+5): Z",                                            TrackElement_coord3_);
    MakeField(model, "TrackElement_theta",            "Q(LTE+6): theta at reference point",                     TrackElement_theta_);
    MakeField(model, "TrackElement_phi",              "Q(LTE+7): phi at reference point",                       TrackElement_phi_);
    MakeField(model, "TrackElement_invPOrPt",         "Q(LTE+8): 1/P or 1/Pt at reference point",               TrackElement_invPOrPt_);
}

namespace {
struct TEProbe { const char *name; std::int8_t code; };
static const TEProbe kTEProbes[] = {
    {"TETP", 0},   // TPC
    {"TEID", 1},   // ID
    {"TEOD", 2},   // OD
    {"TEFA", 3},   // FCA
    {"TEFB", 4},   // FCB
};
}

void RawNanoAODWriter::fillTrackElement()
{
    TrackElement_tracRawIdx_->clear();
    TrackElement_subDetector_->clear();
    TrackElement_dataDescriptor_->clear();
    TrackElement_coord1_->clear();
    TrackElement_coord2_->clear();
    TrackElement_coord3_->clear();
    TrackElement_theta_->clear();
    TrackElement_phi_->clear();
    TrackElement_invPOrPt_->clear();

    if (ph::LDTOP <= 0 || TracRaw_paIdx_->empty()) { *nTrackElement_ = 0; return; }

    // We need to match each PA bank to its TracRaw_ index. TracRaw_ was
    // filled in the same order, guarded on charged tracks. Rather than
    // re-walking the PV/PA chain twice, iterate in sync.
    int paIdx = 0;
    std::size_t tracRow = 0;
    for (int lpv = ph::LQ(ph::LDTOP - 1); lpv > 0; lpv = ph::LQ(lpv))
    {
        for (int lpa = ph::LQ(lpv - 1); lpa > 0; lpa = ph::LQ(lpa), ++paIdx)
        {
            // Only rows present in TracRaw_ (i.e. charged tracks) get TE entries.
            if (tracRow >= TracRaw_paIdx_->size()) break;
            if ((*TracRaw_paIdx_)[tracRow] != paIdx) continue;

            const std::int16_t thisTracRawIdx = static_cast<std::int16_t>(tracRow);
            ++tracRow;

            for (const auto &probe : kTEProbes)
            {
                int lte = ph::LPHPA(probe.name, lpa, 0);
                if (lte == 0) continue;

                TrackElement_tracRawIdx_->push_back(thisTracRawIdx);
                TrackElement_subDetector_->push_back(probe.code);
                TrackElement_dataDescriptor_->push_back(
                    static_cast<std::int32_t>(std::lround(ph::Q(lte + 2))));
                TrackElement_coord1_->push_back(ph::Q(lte + 3));
                TrackElement_coord2_->push_back(ph::Q(lte + 4));
                TrackElement_coord3_->push_back(ph::Q(lte + 5));
                TrackElement_theta_->push_back(ph::Q(lte + 6));
                TrackElement_phi_->push_back(ph::Q(lte + 7));
                TrackElement_invPOrPt_->push_back(ph::Q(lte + 8));
            }
        }
    }
    *nTrackElement_ = static_cast<std::int16_t>(TrackElement_tracRawIdx_->size());
}

// -----------------------------------------------------------------------------
// VD hits — M7. The MVDH event-level bank is reached via LQ(LDTOP - 21)
// (an indirect link on the DST top store). SKELANA walks it in PSHVDH
// (skelana.car L4379): each associated hit carries a back-pointer
// LQ(LQMVDH - N) to its parent PA track bank; we translate that to a
// TracRaw_ index here (-1 if the track was dropped as neutral).
// After the NVDAS associated hits comes a block of NVDUN unassociated
// hits, emitted into a separate collection.
//
// Words per hit is encoded in IQ(LQMVDH+1) as 1000*NWPH + NVDHT:
//   NWPH == 4 -> module, localX, R, RPhi
//   NWPH == 5 -> as above + signalToNoise
// -----------------------------------------------------------------------------
void RawNanoAODWriter::defineVdHit(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nVdAssocHit",                 "VD hits associated to a PA track",           nVdAssocHit_);
    MakeField(model, "VdAssocHit_tracRawIdx",       "Index into TracRaw_*, or -1",                VdAssocHit_tracRawIdx_);
    MakeField(model, "VdAssocHit_module",           "KVDAS(1): module number with sign of Z",     VdAssocHit_module_);
    MakeField(model, "VdAssocHit_localX",           "QVDAS(2): local X (or Z since 94) coord (cm)", VdAssocHit_localX_);
    MakeField(model, "VdAssocHit_R",                "QVDAS(3): R coordinate (-R if R-Z)",         VdAssocHit_R_);
    MakeField(model, "VdAssocHit_RPhi",             "QVDAS(4): R*Phi (or Z since 94) coord",      VdAssocHit_RPhi_);
    MakeField(model, "VdAssocHit_signalToNoise",    "QVDAS(5): S/N ratio of the hit (0 if NWPH<=4)", VdAssocHit_signalToNoise_);

    MakeField(model, "nVdUnassocHit",               "VD hits NOT associated to any track",        nVdUnassocHit_);
    MakeField(model, "VdUnassocHit_module",         "KVDUN(1): module number with sign of Z",     VdUnassocHit_module_);
    MakeField(model, "VdUnassocHit_localX",         "QVDUN(2): local X / Z (cm)",                 VdUnassocHit_localX_);
    MakeField(model, "VdUnassocHit_R",              "QVDUN(3): R coord",                          VdUnassocHit_R_);
    MakeField(model, "VdUnassocHit_RPhi",           "QVDUN(4): R*Phi coord",                      VdUnassocHit_RPhi_);
    MakeField(model, "VdUnassocHit_signalToNoise",  "QVDUN(5): S/N ratio (0 if NWPH<=4)",         VdUnassocHit_signalToNoise_);
}

void RawNanoAODWriter::fillVdHit()
{
    VdAssocHit_tracRawIdx_->clear();
    VdAssocHit_module_->clear();
    VdAssocHit_localX_->clear();
    VdAssocHit_R_->clear();
    VdAssocHit_RPhi_->clear();
    VdAssocHit_signalToNoise_->clear();
    VdUnassocHit_module_->clear();
    VdUnassocHit_localX_->clear();
    VdUnassocHit_R_->clear();
    VdUnassocHit_RPhi_->clear();
    VdUnassocHit_signalToNoise_->clear();

    *nVdAssocHit_ = 0;
    *nVdUnassocHit_ = 0;
    if (ph::LDTOP <= 0) return;
    int lqmvdh = ph::LQ(ph::LDTOP - 21);
    if (lqmvdh <= 0) return;

    int header     = ph::IQ(lqmvdh + 1);
    int nvdht      = header % 1000;
    int nwph       = header / 1000;
    int nvdas      = ph::IQ(lqmvdh - 3);

    // Sanity: SKELANA's tweak for old files that overflowed the 3-digit
    // NVDHT field. See PSHVDH.
    if (header == 5000) { nvdht = 1000; nwph = 4; }
    if (header == 6000) { nvdht = 1000; nwph = 5; }

    // Build a PA-bank-pointer -> TracRaw index mapping for association.
    // TracRaw_paIdx_ is a 0-based PA index; we need a map LPA -> tracRawIdx.
    // Rewalk the PV/PA tree to build the LPA -> paIdx map, then cross to
    // TracRaw_paIdx_ which stores paIdx values in row order.
    std::unordered_map<int, int> paPointerToTracRawIdx;
    {
        int paIdx = 0;
        std::size_t tracRow = 0;
        for (int lpv = ph::LQ(ph::LDTOP - 1); lpv > 0; lpv = ph::LQ(lpv))
        {
            for (int lpa = ph::LQ(lpv - 1); lpa > 0; lpa = ph::LQ(lpa), ++paIdx)
            {
                if (tracRow < TracRaw_paIdx_->size() &&
                    (*TracRaw_paIdx_)[tracRow] == paIdx)
                {
                    paPointerToTracRawIdx[lpa] = static_cast<int>(tracRow);
                    ++tracRow;
                }
            }
        }
    }

    int idat = 1;
    for (int n = 1; n <= nvdas; ++n)
    {
        int lpa = ph::LQ(lqmvdh - n);
        auto it = paPointerToTracRawIdx.find(lpa);
        std::int16_t tracRawIdx = (it == paPointerToTracRawIdx.end())
            ? static_cast<std::int16_t>(-1) : static_cast<std::int16_t>(it->second);

        VdAssocHit_tracRawIdx_->push_back(tracRawIdx);
        VdAssocHit_module_->push_back(static_cast<std::int32_t>(
            std::lround(ph::Q(lqmvdh + idat + 1))));
        VdAssocHit_localX_->push_back(ph::Q(lqmvdh + idat + 2));
        VdAssocHit_R_->push_back(ph::Q(lqmvdh + idat + 3));
        VdAssocHit_RPhi_->push_back(ph::Q(lqmvdh + idat + 4));
        VdAssocHit_signalToNoise_->push_back(nwph > 4 ? ph::Q(lqmvdh + idat + 5) : 0.f);
        idat += nwph;
    }

    int nvdun = std::max(0, nvdht - nvdas);
    for (int i = 1; i <= nvdun; ++i)
    {
        VdUnassocHit_module_->push_back(static_cast<std::int32_t>(
            std::lround(ph::Q(lqmvdh + idat + 1))));
        VdUnassocHit_localX_->push_back(ph::Q(lqmvdh + idat + 2));
        VdUnassocHit_R_->push_back(ph::Q(lqmvdh + idat + 3));
        VdUnassocHit_RPhi_->push_back(ph::Q(lqmvdh + idat + 4));
        VdUnassocHit_signalToNoise_->push_back(nwph > 4 ? ph::Q(lqmvdh + idat + 5) : 0.f);
        idat += nwph;
    }

    *nVdAssocHit_   = static_cast<std::int16_t>(VdAssocHit_tracRawIdx_->size());
    *nVdUnassocHit_ = static_cast<std::int16_t>(VdUnassocHit_module_->size());
}

// -----------------------------------------------------------------------------
// TPC per-track summary — PA.MTPC(7). One row per charged track for which
// MTPC info is present. Layout from shortdst.des line 1894.
// -----------------------------------------------------------------------------
void RawNanoAODWriter::defineMtpc(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nMtpcRaw",                   "Rows in MtpcRaw_*",                              nMtpcRaw_);
    MakeField(model, "MtpcRaw_tracRawIdx",         "Index into TracRaw_*",                           MtpcRaw_tracRawIdx_);
    MakeField(model, "MtpcRaw_dEdx80Max",          "Q(LMTPC+2): dE/dx, 80% trunc, max amplitude",    MtpcRaw_dEdx80Max_);
    MakeField(model, "MtpcRaw_dEdx80Sigma",        "Q(LMTPC+3): sigma of the 80%-trunc Landau fit",  MtpcRaw_dEdx80Sigma_);
    MakeField(model, "MtpcRaw_dEdx65Max",          "Q(LMTPC+4): dE/dx, 65% trunc, max amplitude",    MtpcRaw_dEdx65Max_);
    MakeField(model, "MtpcRaw_dEdx65Sigma",        "Q(LMTPC+5): sigma of the 65%-trunc Landau fit",  MtpcRaw_dEdx65Sigma_);
    MakeField(model, "MtpcRaw_dEdx80Integrated",   "Q(LMTPC+6): dE/dx, 80%, integrated amplitude",   MtpcRaw_dEdx80Integrated_);
    MakeField(model, "MtpcRaw_packedPadsSectors",  "IQ(LMTPC+7): nPads + 100*sectorIn + 10000*sectorOut", MtpcRaw_packedPadsSectors_);
    MakeField(model, "MtpcRaw_packedWiresHits",    "IQ(LMTPC+8): nWiresCrossed + 1000*nHitsClose + 1000000*method", MtpcRaw_packedWiresHits_);
    MakeField(model, "MtpcRaw_zFitChi2",           "Q(LMTPC+14): chi2 of the Z-direction fit",       MtpcRaw_zFitChi2_);
}

void RawNanoAODWriter::fillMtpc()
{
    MtpcRaw_tracRawIdx_->clear();
    MtpcRaw_dEdx80Max_->clear();
    MtpcRaw_dEdx80Sigma_->clear();
    MtpcRaw_dEdx65Max_->clear();
    MtpcRaw_dEdx65Sigma_->clear();
    MtpcRaw_dEdx80Integrated_->clear();
    MtpcRaw_packedPadsSectors_->clear();
    MtpcRaw_packedWiresHits_->clear();
    MtpcRaw_zFitChi2_->clear();

    if (ph::LDTOP <= 0 || TracRaw_paIdx_->empty()) { *nMtpcRaw_ = 0; return; }

    int paIdx = 0;
    std::size_t tracRow = 0;
    for (int lpv = ph::LQ(ph::LDTOP - 1); lpv > 0; lpv = ph::LQ(lpv))
    {
        for (int lpa = ph::LQ(lpv - 1); lpa > 0; lpa = ph::LQ(lpa), ++paIdx)
        {
            if (tracRow >= TracRaw_paIdx_->size()) break;
            if ((*TracRaw_paIdx_)[tracRow] != paIdx) continue;
            const std::int16_t thisTracRawIdx = static_cast<std::int16_t>(tracRow);
            ++tracRow;

            int lmtpc = ph::LPHPA("MTPC", lpa, 0);
            if (lmtpc == 0) continue;

            MtpcRaw_tracRawIdx_->push_back(thisTracRawIdx);
            MtpcRaw_dEdx80Max_->push_back(ph::Q(lmtpc + 2));
            MtpcRaw_dEdx80Sigma_->push_back(ph::Q(lmtpc + 3));
            MtpcRaw_dEdx65Max_->push_back(ph::Q(lmtpc + 4));
            MtpcRaw_dEdx65Sigma_->push_back(ph::Q(lmtpc + 5));
            MtpcRaw_dEdx80Integrated_->push_back(ph::Q(lmtpc + 6));
            MtpcRaw_packedPadsSectors_->push_back(ph::IQ(lmtpc + 7));
            MtpcRaw_packedWiresHits_->push_back(ph::IQ(lmtpc + 8));
            MtpcRaw_zFitChi2_->push_back(ph::Q(lmtpc + 14));
        }
    }
    *nMtpcRaw_ = static_cast<std::int16_t>(MtpcRaw_tracRawIdx_->size());
}

// -----------------------------------------------------------------------------
// Beamspot -- event-level scalars from the beam-spot bank at LQ(LDTOP-25).
// The bank word layout is XYZ (3 floats) + sigma XYZ (3 floats), as reused
// from DELPHI's PSBEAM (skelana.car L1574). When the bank is absent we set
// all seven fields to 0 and errorFlag = -1 so downstream can filter.
// -----------------------------------------------------------------------------
void RawNanoAODWriter::defineBeamSpot(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "Event_beamSpotX",         "LQ(LDTOP-25)+1: beam spot X (cm)",      Event_beamSpotX_);
    MakeField(model, "Event_beamSpotY",         "LQ(LDTOP-25)+2: beam spot Y (cm)",      Event_beamSpotY_);
    MakeField(model, "Event_beamSpotZ",         "LQ(LDTOP-25)+3: beam spot Z (cm)",      Event_beamSpotZ_);
    MakeField(model, "Event_beamSpotSigmaX",    "LQ(LDTOP-25)+4: beam spot sigma X (cm)", Event_beamSpotSigmaX_);
    MakeField(model, "Event_beamSpotSigmaY",    "LQ(LDTOP-25)+5: beam spot sigma Y (cm)", Event_beamSpotSigmaY_);
    MakeField(model, "Event_beamSpotSigmaZ",    "LQ(LDTOP-25)+6: beam spot sigma Z (cm)", Event_beamSpotSigmaZ_);
    MakeField(model, "Event_beamSpotErrorFlag", "0 if bank present, -1 if missing",      Event_beamSpotErrorFlag_);
}

void RawNanoAODWriter::fillBeamSpot()
{
    *Event_beamSpotX_ = 0.f;
    *Event_beamSpotY_ = 0.f;
    *Event_beamSpotZ_ = 0.f;
    *Event_beamSpotSigmaX_ = 0.f;
    *Event_beamSpotSigmaY_ = 0.f;
    *Event_beamSpotSigmaZ_ = 0.f;
    *Event_beamSpotErrorFlag_ = -1;

    if (ph::LDTOP <= 0) return;
    int lqspot = ph::LQ(ph::LDTOP - 25);
    if (lqspot <= 0) return;

    *Event_beamSpotX_      = ph::Q(lqspot + 1);
    *Event_beamSpotY_      = ph::Q(lqspot + 2);
    *Event_beamSpotZ_      = ph::Q(lqspot + 3);
    *Event_beamSpotSigmaX_ = ph::Q(lqspot + 4);
    *Event_beamSpotSigmaY_ = ph::Q(lqspot + 5);
    *Event_beamSpotSigmaZ_ = ph::Q(lqspot + 6);
    *Event_beamSpotErrorFlag_ = 0;
}

// -----------------------------------------------------------------------------
// Reconstructed vertices -- PV bank chain starting at LQ(LDTOP-1) and
// advancing via LQ(LPV). Per PSHVTX (skelana.car L2836), each PV has:
//   IQ(LPV)      status bitmask  (bits 1 dummy, 2 secondary, 3 sec-hadronic, 4 sim)
//   IQ(LPV+2)    nOutgoing
//   IQ(LPV+3)    ndof
//   Q(LPV+4) NINT  mass code of origin particle
//   Q(LPV+5..7)    position (cm)
//   Q(LPV+8)       chi2
//   Q(LPV+9..14)   3x3 error matrix as (XX, XY, YY, XZ, YZ, ZZ)
// -----------------------------------------------------------------------------
void RawNanoAODWriter::defineVtx(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nVtx",           "Number of reconstructed vertices (PV chain from LDTOP-1)", nVtx_);
    MakeField(model, "Vtx_position",   "Q(LPV+5..7) vertex (X, Y, Z) in cm", Vtx_position_);
    MakeField(model, "Vtx_chi2",       "Q(LPV+8) vertex fit chi2",           Vtx_chi2_);
    MakeField(model, "Vtx_ndf",        "IQ(LPV+3) vertex fit ndof",          Vtx_ndf_);
    MakeField(model, "Vtx_nOutgoing",  "IQ(LPV+2) number of outgoing particles", Vtx_nOutgoing_);
    MakeField(model, "Vtx_massCode",   "Q(LPV+4) NINT: origin-particle mass code", Vtx_massCode_);
    MakeField(model, "Vtx_statusBits", "IQ(LPV) status (bit1=dummy, 2=secondary, 3=sec-hadronic, 4=sim)", Vtx_statusBits_);
    MakeField(model, "Vtx_errXX",      "Q(LPV+9):  error matrix XX", Vtx_errXX_);
    MakeField(model, "Vtx_errXY",      "Q(LPV+10): error matrix XY", Vtx_errXY_);
    MakeField(model, "Vtx_errYY",      "Q(LPV+11): error matrix YY", Vtx_errYY_);
    MakeField(model, "Vtx_errXZ",      "Q(LPV+12): error matrix XZ", Vtx_errXZ_);
    MakeField(model, "Vtx_errYZ",      "Q(LPV+13): error matrix YZ", Vtx_errYZ_);
    MakeField(model, "Vtx_errZZ",      "Q(LPV+14): error matrix ZZ", Vtx_errZZ_);
}

void RawNanoAODWriter::fillVtx()
{
    Vtx_position_->clear();
    Vtx_chi2_->clear();
    Vtx_ndf_->clear();
    Vtx_nOutgoing_->clear();
    Vtx_massCode_->clear();
    Vtx_statusBits_->clear();
    Vtx_errXX_->clear();
    Vtx_errXY_->clear();
    Vtx_errYY_->clear();
    Vtx_errXZ_->clear();
    Vtx_errYZ_->clear();
    Vtx_errZZ_->clear();

    *nVtx_ = 0;
    if (ph::LDTOP <= 0) return;

    int lpv = ph::LQ(ph::LDTOP - 1);
    int count = 0;
    const int kMax = 150;   // DELPHI's NVTXMX cap
    while (lpv > 0 && count < kMax)
    {
        int status     = ph::IQ(lpv);
        int nOut       = ph::IQ(lpv + 2);
        int ndof       = ph::IQ(lpv + 3);
        int massCode   = static_cast<int>(std::lround(ph::Q(lpv + 4)));
        float x        = ph::Q(lpv + 5);
        float y        = ph::Q(lpv + 6);
        float z        = ph::Q(lpv + 7);
        float chi2     = ph::Q(lpv + 8);

        Vtx_position_->push_back(XYZVectorF(x, y, z));
        Vtx_chi2_->push_back(chi2);
        Vtx_ndf_->push_back(static_cast<std::int16_t>(ndof));
        Vtx_nOutgoing_->push_back(static_cast<std::int16_t>(nOut));
        Vtx_massCode_->push_back(massCode);
        Vtx_statusBits_->push_back(status);
        Vtx_errXX_->push_back(ph::Q(lpv +  9));
        Vtx_errXY_->push_back(ph::Q(lpv + 10));
        Vtx_errYY_->push_back(ph::Q(lpv + 11));
        Vtx_errXZ_->push_back(ph::Q(lpv + 12));
        Vtx_errYZ_->push_back(ph::Q(lpv + 13));
        Vtx_errZZ_->push_back(ph::Q(lpv + 14));

        ++count;
        lpv = ph::LQ(lpv);
    }
    *nVtx_ = static_cast<std::int16_t>(Vtx_position_->size());
}

// -----------------------------------------------------------------------------
// GEN-truth particle list -- SKELANA's PSHMC reads the simulation banks at
// LQ(LDTOP-28)/(LDTOP-29) on shortDST (PSHLUJ) or the fullDST equivalents
// (PSFLUJ) and populates the PSCLUJ common block with a LUJETS-style record.
//
// We call PSHMC from user02 and then copy the populated fields out. On real
// data PSCLUJ stays at its previous state, so we manually zero NP first and
// rely on NP == 0 after PSHMC to mean "not MC".
//
// Fields mirror the ones the SKELANA-based NanoAODWriter writes in its
// GenPart_* collection, so downstream analysis code can be schema-agnostic
// across the two writers.
// -----------------------------------------------------------------------------
void RawNanoAODWriter::defineGenPart(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nGenPart",               "Number of LUJETS generator particles (0 on real data)", nGenPart_);
    MakeField(model, "GenPart_status",         "KP(i,1): JETSET status code (1 = final-state stable)",  GenPart_status_);
    MakeField(model, "GenPart_pdgId",          "KP(i,2): PDG particle code",                            GenPart_pdgId_);
    MakeField(model, "GenPart_parentIdx",      "KP(i,3) - 1: parent index (-1 if none)",                GenPart_parentIdx_);
    MakeField(model, "GenPart_firstChildIdx",  "KP(i,4) - 1: first-daughter index",                     GenPart_firstChildIdx_);
    MakeField(model, "GenPart_lastChildIdx",   "KP(i,5) - 1: last-daughter index",                      GenPart_lastChildIdx_);
    MakeField(model, "GenPart_fourMomentum",   "PP(i,1..4): (px, py, pz, E) in GeV",                    GenPart_fourMomentum_);
    MakeField(model, "GenPart_mass",           "PP(i,5): generator mass in GeV",                        GenPart_mass_);
    MakeField(model, "GenPart_vertex",
        "VP(i,1..3): production vertex (mm). Units differ from the reco Vtx_position "
        "(cm) and TracRaw impact params (cm) -- LUJETS uses mm by convention.",
        GenPart_vertex_);
    MakeField(model, "GenPart_productionTime", "VP(i,4): time of production",       GenPart_productionTime_);
    MakeField(model, "GenPart_properLifetime", "VP(i,5): proper lifetime",          GenPart_properLifetime_);
}

void RawNanoAODWriter::fillGenPart()
{
    GenPart_status_->clear();
    GenPart_pdgId_->clear();
    GenPart_parentIdx_->clear();
    GenPart_firstChildIdx_->clear();
    GenPart_lastChildIdx_->clear();
    GenPart_fourMomentum_->clear();
    GenPart_mass_->clear();
    GenPart_vertex_->clear();
    GenPart_productionTime_->clear();
    GenPart_properLifetime_->clear();

    // PSHMC/PSHLUJ/PSFLUJ return early on real data without touching NP, so
    // we zero it manually and re-check afterwards -- NP > 0 after the call
    // means the simulation structure was present and fully unpacked.
    sk::NP = 0;
    pshmc_();

    const int np = sk::NP;
    *nGenPart_ = np;
    *Event_isMC_ = (np > 0) ? 1 : 0;

    for (int i = 1; i <= np; ++i)
    {
        GenPart_status_->push_back(static_cast<std::int16_t>(sk::KP(i, 1)));
        GenPart_pdgId_->push_back(sk::KP(i, 2));
        GenPart_parentIdx_->push_back(sk::KP(i, 3) - 1);
        GenPart_firstChildIdx_->push_back(sk::KP(i, 4) - 1);
        GenPart_lastChildIdx_->push_back(sk::KP(i, 5) - 1);
        GenPart_fourMomentum_->push_back(
            XYZTVectorF(sk::PP(i, 1), sk::PP(i, 2), sk::PP(i, 3), sk::PP(i, 4)));
        GenPart_mass_->push_back(sk::PP(i, 5));
        GenPart_vertex_->push_back(
            XYZVectorF(sk::VP(i, 1), sk::VP(i, 2), sk::VP(i, 3)));
        GenPart_productionTime_->push_back(sk::VP(i, 4));
        GenPart_properLifetime_->push_back(sk::VP(i, 5));
    }
}
