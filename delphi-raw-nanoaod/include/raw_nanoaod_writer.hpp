#ifndef RAW_NANOAOD_WRITER_HPP
#define RAW_NANOAOD_WRITER_HPP

// PHDST-level raw-hit nanoAOD writer.
// Subclasses phdst::Analysis (NOT skelana::Analysis) so that no SKELANA
// aggregation runs: we read the ZEBRA banks directly. M0 just surfaces the
// PHCIII event-header fields; M1+ add PA.EMCA / PA.HCAL / PA.STIC / … hit
// collections. See docs/PHDST_RAW_NANOAOD_PLAN.md.

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "phdst_analysis.hpp"
#include "phdst.hpp"

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <Math/Vector3D.h>
#include <Math/Vector4D.h>

using XYZVectorF = ROOT::Math::DisplacementVector3D<
    ROOT::Math::Cartesian3D<float>,
    ROOT::Math::DefaultCoordinateSystemTag>;

using XYZTVectorF = ROOT::Math::LorentzVector<
    ROOT::Math::PxPyPzE4D<float>>;

using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleModel = ROOT::Experimental::RNTupleModel;

namespace ph = phdst;

class RawNanoAODWriter : public ph::Analysis
{
private:
    typedef ph::Analysis super;

public:
    RawNanoAODWriter(const RawNanoAODWriter &) = delete;
    RawNanoAODWriter &operator=(const RawNanoAODWriter &) = delete;
    virtual ~RawNanoAODWriter();
    static RawNanoAODWriter *getInstance();
    void setOutput(const std::filesystem::path &output);
    void configureRawBankScan(bool enabled, int maxDepth, int maxLinks,
                              int maxWordsPerBank, bool allBanks);

protected:
    RawNanoAODWriter();
    virtual void user00() override;
    virtual int  user01() override;
    virtual void user02() override;
    virtual void user99() override;

private:
    void defineEvent(std::unique_ptr<RNTupleModel> &model);
    void fillEvent();
    void defineEmShower(std::unique_ptr<RNTupleModel> &model);
    void fillEmShowers();   // walks PA banks, emits EmShower_* + EmLayer_*
    void defineHadShower(std::unique_ptr<RNTupleModel> &model);
    void fillHadShowers();  // PA.HCNC -> HadShower_* + HadHit_*
    void defineStic(std::unique_ptr<RNTupleModel> &model);
    void fillStic();        // PA.SSTC -> Stic_* (one row per track with STIC)
    void defineMuidEl(std::unique_ptr<RNTupleModel> &model);
    void fillMuidEl();      // PA.MUID + PA.ELID -> MuidRaw_* + ElidRaw_*
    void defineTrac(std::unique_ptr<RNTupleModel> &model);
    void fillTrac();        // PA.TRAC + PA.MAIN -> TracRaw_* per charged track
    void defineTrackElement(std::unique_ptr<RNTupleModel> &model);
    void fillTrackElement();// PA.{TETP,TEID,TEOD,TEFA,TEFB} -> TrackElement_*
    void defineVdHit(std::unique_ptr<RNTupleModel> &model);
    void fillVdHit();       // MVDH (LDTOP-21) -> VdAssocHit_* + VdUnassocHit_*
    void defineMtpc(std::unique_ptr<RNTupleModel> &model);
    void fillMtpc();        // PA.MTPC -> MtpcRaw_* (TPC per-track dE/dx etc.)
    void defineBeamSpot(std::unique_ptr<RNTupleModel> &model);
    void fillBeamSpot();    // LQ(LDTOP-25) -> Event_beamSpot* scalars
    void defineVtx(std::unique_ptr<RNTupleModel> &model);
    void fillVtx();         // LQ(LDTOP-1) walk -> Vtx_* collection
    void defineGenPart(std::unique_ptr<RNTupleModel> &model);
    void fillGenPart();     // calls SKELANA's PSHMC to fill PSCLUJ, then emits GenPart_*
    void defineRawBank(std::unique_ptr<RNTupleModel> &model);
    void fillRawBanks();    // generic RAW/Rxxx bank manifest + payload words
    void scanRawBankTop(int topCode, int topLink, std::unordered_set<int> &seen);
    void scanRawBankChain(int l, int topCode, std::int32_t parentIdx,
                          std::int16_t linkIndex, int depth,
                          std::unordered_set<int> &seen);
    std::int32_t appendRawBank(int l, int topCode, std::int32_t parentIdx,
                               std::int16_t linkIndex, int depth);

    std::filesystem::path              output_;
    std::unique_ptr<RNTupleWriter>     writer_;
    bool rawBankScanEnabled_ = true;
    bool rawBankAllBanks_ = false;
    int rawBankMaxDepth_ = 8;
    int rawBankMaxLinks_ = 64;
    int rawBankMaxWordsPerBank_ = 64;

    // Event_* fields (M0): from PHCIII common block.
    std::shared_ptr<int>   Event_experimentNumber_;
    std::shared_ptr<int>   Event_runNumber_;
    std::shared_ptr<int>   Event_fileSequenceNumber_;
    std::shared_ptr<int>   Event_eventNumber_;
    std::shared_ptr<int>   Event_date_;
    std::shared_ptr<int>   Event_time_;
    std::shared_ptr<int>   Event_fillNumber_;
    std::shared_ptr<float> Event_bFieldTesla_;   // BTESLA from BPILOT
    std::shared_ptr<float> Event_bFieldGevCm_;   // BGEVCM from BPILOT — use as 1/R [1/cm] = BGEVCM / pT [GeV]
    std::shared_ptr<std::string> Event_recordType_; // PHRTY(): BOF/CPT/RAW/TAN/DST/etc.

    // Generic raw-DST bank scan. With DETRAW-enabled DELANA output, detector
    // raw banks appear as RAW/Rxxx ZEBRA banks. RawBank_* records the compact
    // tree printed by delphi-raw-bank-lister; RawWord_* stores a bounded copy
    // of the bank data words as both integer and float views.
    std::shared_ptr<std::int32_t>                         nRawBank_;
    std::shared_ptr<std::vector<std::string>>             RawBank_name_;
    std::shared_ptr<std::vector<std::int32_t>>            RawBank_nameCode_;
    std::shared_ptr<std::vector<std::int32_t>>            RawBank_top_;
    std::shared_ptr<std::vector<std::int32_t>>            RawBank_addr_;
    std::shared_ptr<std::vector<std::int32_t>>            RawBank_parentIdx_;
    std::shared_ptr<std::vector<std::int16_t>>            RawBank_linkIndex_;
    std::shared_ptr<std::vector<std::int16_t>>            RawBank_depth_;
    std::shared_ptr<std::vector<std::int16_t>>            RawBank_nlinks_;
    std::shared_ptr<std::vector<std::int32_t>>            RawBank_ndata_;
    std::shared_ptr<std::vector<std::int32_t>>            RawBank_firstRawWordIdx_;
    std::shared_ptr<std::vector<std::int32_t>>            RawBank_nRawWords_;

    std::shared_ptr<std::int32_t>                         nRawWord_;
    std::shared_ptr<std::vector<std::int32_t>>            RawWord_rawBankIdx_;
    std::shared_ptr<std::vector<std::int32_t>>            RawWord_offset_;
    std::shared_ptr<std::vector<std::int32_t>>            RawWord_i_;
    std::shared_ptr<std::vector<float>>                   RawWord_f_;

    // EM-shower collections (M2).
    //
    // Sourced from the EMNC extra-module under every PA (per-track) bank in
    // the shortDST. Each EMNC may contain 0, 1, ... showers; each shower
    // unpacks an energy, a position-vector, a 10-bit HPC layer pattern, and
    // optionally (for HPC showers in older PXDST versions) one per-layer
    // energy per set bit.
    std::shared_ptr<std::int16_t>                         nEmShower_;
    std::shared_ptr<std::vector<std::int16_t>>            EmShower_paIdx_;         // index of source PA track (0-based)
    std::shared_ptr<std::vector<std::int8_t>>             EmShower_detector_;      // 9 = HPC, 26 = EMF(FEMC)
    std::shared_ptr<std::vector<float>>                   EmShower_energy_;        // Q(LSHOWR+1)
    std::shared_ptr<std::vector<XYZVectorF>>              EmShower_position_;      // Q(LSHOWR+2..4) in cm
    std::shared_ptr<std::vector<std::int16_t>>            EmShower_nLayers_;       // Q(LSHOWR+5)
    std::shared_ptr<std::vector<std::int32_t>>            EmShower_layerPattern_;  // Q(LSHOWR+6) 10-bit HPC hit pattern
    std::shared_ptr<std::vector<std::int16_t>>            EmShower_nLayerEnergies_;// number of EmLayer rows for this shower

    std::shared_ptr<std::int16_t>                         nEmLayer_;
    std::shared_ptr<std::vector<std::int16_t>>            EmLayer_emShowerIdx_;    // index into EmShower_*
    std::shared_ptr<std::vector<std::int8_t>>             EmLayer_layer_;          // 1..NHPLAY (DELPHI convention)
    std::shared_ptr<std::vector<float>>                   EmLayer_energy_;         // per-layer deposit

    // --- Hadronic-calorimeter collections (M3): PA.HCNC shortDST extra-module.
    // Layout per PSHHAC (skelana.car line 3442): per shower,
    //   Q(LSHOWR+1) = shower E
    //   Q(LSHOWR+2..4) = 3-momentum (x, y, z) of the shower direction
    //   Q(LSHOWR+5) = NLHITS = number of layer-hit pairs
    //   Q(LSHOWR+5+2*nl-1) = energy of hit nl
    //   Q(LSHOWR+5+2*nl)   = packed 1000*layer + nTowers
    // Advance by 5 + 2*NLHITS.
    std::shared_ptr<std::int16_t>                         nHadShower_;
    std::shared_ptr<std::vector<std::int16_t>>            HadShower_paIdx_;
    std::shared_ptr<std::vector<float>>                   HadShower_energy_;
    std::shared_ptr<std::vector<XYZVectorF>>              HadShower_direction_;
    std::shared_ptr<std::vector<std::int16_t>>            HadShower_nHits_;
    std::shared_ptr<std::vector<std::int16_t>>            HadShower_nHitRows_;

    std::shared_ptr<std::int16_t>                         nHadHit_;
    std::shared_ptr<std::vector<std::int16_t>>            HadHit_hadShowerIdx_;
    std::shared_ptr<std::vector<std::int8_t>>             HadHit_layer_;     // 1..N_HCAL_LAYERS
    std::shared_ptr<std::vector<std::int16_t>>            HadHit_nTowers_;
    std::shared_ptr<std::vector<float>>                   HadHit_energy_;

    // --- STIC small-angle EM collection (M4): PA.SSTC shortDST.
    // One row per track with a STIC shower (SKELANA's CCC DO NS loop is
    // commented out so there's effectively one shower per track). No layer
    // decomposition — STIC is thin enough that SKELANA only keeps totals.
    std::shared_ptr<std::int16_t>                         nStic_;
    std::shared_ptr<std::vector<std::int16_t>>            Stic_paIdx_;
    std::shared_ptr<std::vector<float>>                   Stic_energyFromMain_;   // Q(LMAIN+6)
    std::shared_ptr<std::vector<XYZVectorF>>              Stic_directionFromMain_;
    std::shared_ptr<std::vector<std::int16_t>>            Stic_numHitTowers_;

    // --- Muon / Electron per-track raw ID (M5): PA.MUID + PA.ELID records.
    // Keep it simple — one row per track that has that extra-module present,
    // carrying the raw ID words. The Muid_* / Elid_* collections in the
    // SKELANA-based delphi-nanoaod carry the parsed/refitted versions; these
    // sit alongside them and expose the un-interpreted PSCMUD/PSCELD words
    // for downstream ML / calibration work.
    std::shared_ptr<std::int16_t>                         nMuidRaw_;
    std::shared_ptr<std::vector<std::int16_t>>            MuidRaw_paIdx_;
    std::shared_ptr<std::vector<std::int32_t>>            MuidRaw_tag_;        // Q(LMUID+1) NINT = tag
    std::shared_ptr<std::vector<float>>                   MuidRaw_looseChi2_;  // Q(LMUID+2)
    std::shared_ptr<std::vector<std::int32_t>>            MuidRaw_hitPattern_; // Q(LMUID+3) NINT

    std::shared_ptr<std::int16_t>                         nElidRaw_;
    std::shared_ptr<std::vector<std::int16_t>>            ElidRaw_paIdx_;
    std::shared_ptr<std::vector<std::int32_t>>            ElidRaw_tag_;             // Q(LELID+1) NINT
    std::shared_ptr<std::vector<std::int32_t>>            ElidRaw_gammaConvTag_;    // Q(LELID+2) NINT
    std::shared_ptr<std::vector<XYZVectorF>>              ElidRaw_refitMomentum_;   // Q(LELID+3..5)

    // --- Track raw (M6): PA.TRAC + PA.MAIN. One row per charged track
    // (Q(LMAIN+8) != 0). Follows SKELANA PSHTRA / PSCTRA (stdcdes.car
    // +KEEP,PSCTRA). The 20-word TRAC payload carries the perigee
    // parameters and the 15-element symmetric weight matrix.
    std::shared_ptr<std::int16_t>                         nTracRaw_;
    std::shared_ptr<std::vector<std::int16_t>>            TracRaw_paIdx_;
    std::shared_ptr<std::vector<float>>                   TracRaw_impactRPhi_;    // QTRAC( 4)
    std::shared_ptr<std::vector<float>>                   TracRaw_impactZ_;       // QTRAC( 5)
    std::shared_ptr<std::vector<float>>                   TracRaw_theta_;         // QTRAC( 6)
    std::shared_ptr<std::vector<float>>                   TracRaw_phi_;           // QTRAC( 7)
    std::shared_ptr<std::vector<float>>                   TracRaw_invR_;          // QTRAC( 8)  curvature w/ sign at perigee
    std::shared_ptr<std::vector<std::array<float, 15>>>   TracRaw_weightMatrix_;  // QTRAC(9..23) 5x5 symmetric
    std::shared_ptr<std::vector<float>>                   TracRaw_trackLength_;   // |Q(LMAIN+9)| (sign flipped by SKELANA)
    std::shared_ptr<std::vector<std::int32_t>>            TracRaw_detectorsUsed_; // IQ(LPA+2) bits 1-VD,2-ID,3-TPC,4-OD,5-FCA,6-FCB
    std::shared_ptr<std::vector<float>>                   TracRaw_firstPointR_;   // |Q(LMAIN+23..24)|
    std::shared_ptr<std::vector<float>>                   TracRaw_firstPointZ_;   // Q(LMAIN+25)
    std::shared_ptr<std::vector<float>>                   TracRaw_chi2NoVD_;      // Q(LMAIN+16)
    std::shared_ptr<std::vector<float>>                   TracRaw_chi2VD_;        // Q(LMAIN+26)
    std::shared_ptr<std::vector<std::int16_t>>            TracRaw_ndfNoVD_;       // Q(LMAIN+17)
    std::shared_ptr<std::vector<std::int16_t>>            TracRaw_ndfVD_;         // Q(LMAIN+27)
    std::shared_ptr<std::vector<float>>                   TracRaw_chi2VDHits_;    // Q(LMAIN+18)
    std::shared_ptr<std::vector<std::int8_t>>             TracRaw_charge_;        // sign of Q(LMAIN+8)

    // --- Track elements (M7): per-track-per-sub-detector bank entries.
    // Shared layout across PA.TETP(TPC), PA.TEID(ID), PA.TEOD(OD),
    // PA.TEFA(FCA), PA.TEFB(FCB) — see shortdst.des. Each PA track has at
    // most one TE per sub-detector. We save the 8 header words; the
    // variable-length error-matrix tail and the PXDST-251+ (nDoF, chi2,
    // length) footer are NOT saved in this first pass — they need the
    // measurement-code popcount to locate. If downstream wants them, an
    // M7+ commit can add a blob field.
    //
    // IMPORTANT — fadana vs shortDST: these banks are only populated on
    // the DELANA "full-DST" output (the .fadana file). The shortDST
    // (.sdst) produced by shortdst.exe strips them. Correspondingly,
    // the MVDH VD-hit bank exposed as VdAssocHit_* / VdUnassocHit_*
    // below is only on the shortDST and is empty on .fadana. For a
    // complete refit one therefore needs BOTH files. The Delphi-Sim-
    // Pipeline branch feature/cmssw-el9-base now preserves both so
    // downstream analyses can merge them by (run, event) number.
    std::shared_ptr<std::int16_t>                         nTrackElement_;
    std::shared_ptr<std::vector<std::int16_t>>            TrackElement_tracRawIdx_;
    std::shared_ptr<std::vector<std::int8_t>>             TrackElement_subDetector_;  // 0=TPC, 1=ID, 2=OD, 3=FCA, 4=FCB
    std::shared_ptr<std::vector<std::int32_t>>            TrackElement_dataDescriptor_;
    std::shared_ptr<std::vector<float>>                   TrackElement_coord1_;
    std::shared_ptr<std::vector<float>>                   TrackElement_coord2_;
    std::shared_ptr<std::vector<float>>                   TrackElement_coord3_;
    std::shared_ptr<std::vector<float>>                   TrackElement_theta_;
    std::shared_ptr<std::vector<float>>                   TrackElement_phi_;
    std::shared_ptr<std::vector<float>>                   TrackElement_invPOrPt_;

    // --- VD hits (M7): the MVDH event-level bank at LQ(LDTOP - 21).
    // Per SKELANA PSHVDH (skelana.car L4379) and PSCVDA / PSCVDU
    // commons (stdcdes.car). Associated hits carry a back-reference to
    // a PA track bank; we translate that to an index into TracRaw_*.
    // Unassociated hits live in a flat per-event list.
    std::shared_ptr<std::int16_t>                         nVdAssocHit_;
    std::shared_ptr<std::vector<std::int16_t>>            VdAssocHit_tracRawIdx_;
    std::shared_ptr<std::vector<std::int32_t>>            VdAssocHit_module_;
    std::shared_ptr<std::vector<float>>                   VdAssocHit_localX_;    // cm
    std::shared_ptr<std::vector<float>>                   VdAssocHit_R_;
    std::shared_ptr<std::vector<float>>                   VdAssocHit_RPhi_;
    std::shared_ptr<std::vector<float>>                   VdAssocHit_signalToNoise_;

    std::shared_ptr<std::int16_t>                         nVdUnassocHit_;
    std::shared_ptr<std::vector<std::int32_t>>            VdUnassocHit_module_;
    std::shared_ptr<std::vector<float>>                   VdUnassocHit_localX_;
    std::shared_ptr<std::vector<float>>                   VdUnassocHit_R_;
    std::shared_ptr<std::vector<float>>                   VdUnassocHit_RPhi_;
    std::shared_ptr<std::vector<float>>                   VdUnassocHit_signalToNoise_;

    // --- TPC per-track summary (M7): PA.MTPC(7) -- dE/dx fields + wire /
    // pad counts. Present on shortDST even though the per-hit TETP bank
    // is not. Essential input for dE/dx-based particle ID when refitting
    // tracks or running particle-flow reco.
    std::shared_ptr<std::int16_t>                         nMtpcRaw_;
    std::shared_ptr<std::vector<std::int16_t>>            MtpcRaw_tracRawIdx_;
    std::shared_ptr<std::vector<float>>                   MtpcRaw_dEdx80Max_;        // Q(LMTPC+2) 80% trunc max-amp dE/dx
    std::shared_ptr<std::vector<float>>                   MtpcRaw_dEdx80Sigma_;      // Q(LMTPC+3)
    std::shared_ptr<std::vector<float>>                   MtpcRaw_dEdx65Max_;        // Q(LMTPC+4)
    std::shared_ptr<std::vector<float>>                   MtpcRaw_dEdx65Sigma_;      // Q(LMTPC+5)
    std::shared_ptr<std::vector<float>>                   MtpcRaw_dEdx80Integrated_; // Q(LMTPC+6)
    std::shared_ptr<std::vector<std::int32_t>>            MtpcRaw_packedPadsSectors_;// IQ(LMTPC+7)
    std::shared_ptr<std::vector<std::int32_t>>            MtpcRaw_packedWiresHits_;  // IQ(LMTPC+8)
    std::shared_ptr<std::vector<float>>                   MtpcRaw_zFitChi2_;         // Q(LMTPC+14)

    // --- Beamspot (M9): event-level scalars from LQ(LDTOP-25).
    // PSBEAM (skelana.car L1574) reads Q(LQSPOT+1..3) = XYZ, Q(LQSPOT+4..6)
    // = sigma XYZ. When the bank is absent (older data / bad run), SKELANA
    // falls back to a VDBSPT default; here we only surface what the bank
    // actually carries, with an errorFlag = 0 "bank present" / -1 "missing".
    std::shared_ptr<float>                                Event_beamSpotX_;
    std::shared_ptr<float>                                Event_beamSpotY_;
    std::shared_ptr<float>                                Event_beamSpotZ_;
    std::shared_ptr<float>                                Event_beamSpotSigmaX_;
    std::shared_ptr<float>                                Event_beamSpotSigmaY_;
    std::shared_ptr<float>                                Event_beamSpotSigmaZ_;
    std::shared_ptr<std::int8_t>                          Event_beamSpotErrorFlag_;

    // --- Reconstructed vertices (M9): PV-bank chain at LQ(LDTOP-1).
    // See PSHVTX (skelana.car L2836) for the walk. Each PV contributes one
    // row. Offsets, all 1-based from LPV:
    //   IQ(LPV)      status bits (1=dummy, 2=secondary, 3=sec-hadronic, 4=sim)
    //   IQ(LPV+2)    nOutgoing (multiplicity)
    //   IQ(LPV+3)    ndof
    //   Q(LPV+4) NINT  mass code of origin particle
    //   Q(LPV+5..7)    X, Y, Z
    //   Q(LPV+8)       chi2
    //   Q(LPV+9..14)   error matrix (XX, XY, YY, XZ, YZ, ZZ)
    std::shared_ptr<std::int16_t>                         nVtx_;
    std::shared_ptr<std::vector<XYZVectorF>>              Vtx_position_;
    std::shared_ptr<std::vector<float>>                   Vtx_chi2_;
    std::shared_ptr<std::vector<std::int16_t>>            Vtx_ndf_;
    std::shared_ptr<std::vector<std::int16_t>>            Vtx_nOutgoing_;
    std::shared_ptr<std::vector<std::int32_t>>            Vtx_massCode_;
    std::shared_ptr<std::vector<std::int32_t>>            Vtx_statusBits_;
    std::shared_ptr<std::vector<float>>                   Vtx_errXX_;
    std::shared_ptr<std::vector<float>>                   Vtx_errXY_;
    std::shared_ptr<std::vector<float>>                   Vtx_errYY_;
    std::shared_ptr<std::vector<float>>                   Vtx_errXZ_;
    std::shared_ptr<std::vector<float>>                   Vtx_errYZ_;
    std::shared_ptr<std::vector<float>>                   Vtx_errZZ_;

    // --- MC GEN-truth particle list (Tier A of the MC-truth plan).
    // Filled by calling SKELANA's PSHMC fortran routine, which populates the
    // PSCLUJ common block (a LUJETS-like LUND event record) from the DELPHI
    // simulation banks at LQ(LDTOP-28)/(LDTOP-29) (shortDST) or the
    // equivalent fullDST banks. Empty on real-data events; Event_isMC is a
    // convenience scalar so downstream code can gate without counting.
    std::shared_ptr<std::int8_t>                          Event_isMC_;     // 1 if PSCLUJ populated, 0 otherwise
    std::shared_ptr<std::int32_t>                         nGenPart_;
    std::shared_ptr<std::vector<std::int16_t>>            GenPart_status_;       // KP(i,1): JETSET status code
    std::shared_ptr<std::vector<std::int32_t>>            GenPart_pdgId_;        // KP(i,2): PDG particle code
    std::shared_ptr<std::vector<std::int32_t>>            GenPart_parentIdx_;    // KP(i,3) - 1, -1 if none
    std::shared_ptr<std::vector<std::int32_t>>            GenPart_firstChildIdx_;// KP(i,4) - 1
    std::shared_ptr<std::vector<std::int32_t>>            GenPart_lastChildIdx_; // KP(i,5) - 1
    std::shared_ptr<std::vector<XYZTVectorF>>             GenPart_fourMomentum_; // PP(i,1..4) (px, py, pz, E) in GeV
    std::shared_ptr<std::vector<float>>                   GenPart_mass_;         // PP(i,5) in GeV
    std::shared_ptr<std::vector<XYZVectorF>>              GenPart_vertex_;       // VP(i,1..3) production point (mm)
    std::shared_ptr<std::vector<float>>                   GenPart_productionTime_;// VP(i,4) production time
    std::shared_ptr<std::vector<float>>                   GenPart_properLifetime_;// VP(i,5) proper lifetime
};

#endif // RAW_NANOAOD_WRITER_HPP
