#ifndef NANOAOD_WRITER_HPP
#define NANOAOD_WRITER_HPP

#include <filesystem>
#include <vector>

#include "skelana_analysis.hpp"
#include "skelana.hpp"
#include "skelana/pscvda.hpp"
#include "skelana/pscvdu.hpp"
#include "skelana/pscrv0.hpp"
#include "skelana/pschv0.hpp"
#include "skelana/pscphc.hpp"
#include "skelana/pscpho.hpp"
#include "skelana/pscpi0.hpp"
#include "skelana/pscter.hpp"

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <Math/Vector4D.h>
#include <Math/Vector3D.h>
#include <Math/SMatrixFfwd.h>
#include <Math/SVector.h>

#include <TTree.h>
#include <TFile.h>
#include <TDatabasePDG.h>
#include <TParticlePDG.h>

//DataProcessing dependencies
#include "DataProcessing/include/particleData.h"
#include "DataProcessing/include/eventSelection.h"
#include "DataProcessing/include/eventData.h"
#include "DataProcessing/include/thrustTools.h"
#include "DataProcessing/include/sphericityTools.h"

using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleModel = ROOT::Experimental::RNTupleModel;

namespace sk = skelana;

typedef ROOT::Math::XYZTVectorF XYZTVectorF;
typedef ROOT::Math::DisplacementVector3D< ROOT::Math::Cartesian3D<float>, ROOT::Math::DefaultCoordinateSystemTag > XYZVectorF;
typedef ROOT::Math::PositionVector3D< ROOT::Math::Cartesian3D<float>, ROOT::Math::DefaultCoordinateSystemTag > XYZPointF;

typedef ROOT::Math::SMatrixSym3F SMatrixSym3F;
typedef ROOT::Math::SMatrixSym5F SMatrixSym5F;
typedef ROOT::Math::SVector<float, 5> SVector5F;

enum class DataKind {
    data,
    gen,
    sim
};

class NanoAODWriter : public sk::Analysis
{
private:
    typedef sk::Analysis super;

public:
    NanoAODWriter(const NanoAODWriter &) = delete;
    NanoAODWriter &operator=(const NanoAODWriter &) = delete;
    virtual ~NanoAODWriter();
    static NanoAODWriter *getInstance();
    void setOutput(const std::filesystem::path &output);
    void setMC();

protected:
    NanoAODWriter();
    virtual void user00();
    virtual int user01();
    virtual void user02();
    virtual void user99();

private:
    void defineEvent(std::unique_ptr<RNTupleModel> &model);
    void fillEvent();
    void definePart(std::unique_ptr<RNTupleModel> &model);
    void fillPart();
    void defineJet(std::unique_ptr<RNTupleModel> &model);
    void fillJet();
    void defineVtx(std::unique_ptr<RNTupleModel> &model);
    void fillVtx();
    void defineSimPart(std::unique_ptr<RNTupleModel> &model);
    void fillSimPart();
    void defineGenPart(std::unique_ptr<RNTupleModel> &model);
    void fillGenPart();
    void defineSimVtx(std::unique_ptr<RNTupleModel> &model);
    void fillSimVtx();
    void defineBeamSpot(std::unique_ptr<RNTupleModel> &model);
    void fillBeamSpot();
    void defineTrac(std::unique_ptr<RNTupleModel> &model);
    void fillTrac();
    void defineMuid(std::unique_ptr<RNTupleModel> &model);
    void fillMuid();
    void defineElid(std::unique_ptr<RNTupleModel> &model);
    void fillElid();
    void defineHadid(std::unique_ptr<RNTupleModel> &model);
    void fillHadid();
    void defineBtag(std::unique_ptr<RNTupleModel> &model);
    void fillBtag();
    void definePhoton(std::unique_ptr<RNTupleModel> &model);
    void fillPhoton();
    void defineV0(std::unique_ptr<RNTupleModel> &model);
    void fillV0();
    void definePhotonConv(std::unique_ptr<RNTupleModel> &model);
    void fillPhotonConv();
    void defineUter(std::unique_ptr<RNTupleModel> &model);
    void fillUter();
    void fillPartLoop(particleData& pData, eventData& eData, DataKind cat = DataKind::data);
    void fillSelection(particleData& pData, eventData& eData);

    std::filesystem::path output_;
    std::unique_ptr<RNTupleWriter> writer_;
    bool mc_;

    // Create a TTree to store event-based data
    TFile* file_ = nullptr;
    TTree* out_t = nullptr;
    particleData    out_pData;
    eventData       out_eData;

    TTree* out_tgen = nullptr;
    particleData    out_pData_gen;
    eventData       out_eData_gen;

    TTree* out_tsim = nullptr;
    particleData    out_pData_sim;
    eventData       out_eData_sim;

    // Variables for event information and particle data
    float emf[particleData::nMaxPart];
    float hpc[particleData::nMaxPart];
    float hac[particleData::nMaxPart];
    float stic[particleData::nMaxPart];
    int lock[particleData::nMaxPart];
    TDatabasePDG* pdgDatabase = nullptr;

    // ADD these three:
    float chi2ndf[particleData::nMaxPart];    // chi2/ndf (no VD)
    float chi2ndfVD[particleData::nMaxPart];  // chi2/ndf (with VD)
    float dpp[particleData::nMaxPart];        // delta p / p

    std::shared_ptr<int> Event_runNumber_;
    std::shared_ptr<int> Event_eventNumber_;
    std::shared_ptr<int> Event_fillNumber_;
    std::shared_ptr<int> Event_date_;
    std::shared_ptr<int> Event_time_;
    std::shared_ptr<float> Event_magField_;
    std::shared_ptr<float> Event_cmEnergy_;
    std::shared_ptr<int8_t> Event_shortDstVersion_;
    std::shared_ptr<bool> Event_hadronTagT4_;
    std::shared_ptr<int16_t> Event_chargedMultT4_;
    std::shared_ptr<int16_t> Event_chargedMult_;
    std::shared_ptr<int16_t> Event_neutralMult_;
    std::shared_ptr<float> Event_totalChargedEnergy_;
    std::shared_ptr<float> Event_totalEMEnergy_;
    std::shared_ptr<float> Event_totalHadronicEnergy_;
    std::shared_ptr<std::string> Event_DSTType_;

    std::shared_ptr<int16_t> nPart_;
    std::shared_ptr<std::vector<XYZTVectorF>> Part_fourMomentum_;
    std::shared_ptr<std::vector<int8_t>> Part_charge_;
    std::shared_ptr<std::vector<int16_t>> Part_pdgId_;
    std::shared_ptr<std::vector<int>> Part_massId_;
    std::shared_ptr<std::vector<int8_t>> Part_jetIdx_;
    std::shared_ptr<std::vector<int8_t>> Part_hemisphereIdx_;
    std::shared_ptr<std::vector<int8_t>> Part_vtxCode_;
    std::shared_ptr<std::vector<int8_t>> Part_vtxIdx_;
    std::shared_ptr<std::vector<int>> Part_lock_;
    std::shared_ptr<std::vector<float>> Part_hpcShowerEnergy_;
    std::shared_ptr<std::vector<float>> Part_hpcShowerTheta_;
    std::shared_ptr<std::vector<float>> Part_hpcShowerPhi_;
    std::shared_ptr<std::vector<int>> Part_hpcParticleCode_;
    std::shared_ptr<std::vector<int>> Part_hpcNumLayers_;
    std::shared_ptr<std::vector<int>> Part_hpcLayerHitPattern_;
    std::shared_ptr<std::vector<int>> Part_hpcNumAssociatedShowers_;
    std::shared_ptr<std::vector<float>> Part_hpcTotalShowerEnergy_;
    std::shared_ptr<std::vector<float>> Part_hacShowerEnergy_;
    std::shared_ptr<std::vector<float>> Part_hacShowerTheta_;
    std::shared_ptr<std::vector<float>> Part_hacShowerPhi_;
    std::shared_ptr<std::vector<int>> Part_hacParticleCode_;
    std::shared_ptr<std::vector<int>> Part_hacNumTowers_;
    std::shared_ptr<std::vector<int>> Part_hacTowerHitPattern_;
    std::shared_ptr<std::vector<int>> Part_hacNumAssociatedShowers_;
    std::shared_ptr<std::vector<float>> Part_hacTotalShowerEnergy_;
    std::shared_ptr<std::vector<float>> Part_sticShowerEnergy_;
    std::shared_ptr<std::vector<float>> Part_sticShowerTheta_;
    std::shared_ptr<std::vector<float>> Part_sticShowerPhi_;
    std::shared_ptr<std::vector<int>> Part_sticNumTowers_;
    std::shared_ptr<std::vector<int>> Part_sticChargedTag_;
    std::shared_ptr<std::vector<int>> Part_sticSiliconVertexPos_;
    std::shared_ptr<std::vector<int16_t>> Part_simIdx_;
    std::shared_ptr<std::vector<int16_t>> Part_originVtxIdx_;
    std::shared_ptr<std::vector<int16_t>> Part_decayVtxIdx_;

    // --- Photon (Tier 1: neutral EM clusters, derived from existing Part_* commons) ---
    // Filter: VECP(7,i) == 0 (neutral) AND any of HPC/STIC/FEMC energy > 0.
    // No new skelana bindings. π0 / converted-photon tags are Tier 2
    // (PSCPHO / PSCPHC bindings, not implemented here).
    std::shared_ptr<int16_t>                        nPhoton_;
    std::shared_ptr<std::vector<int16_t>>           Photon_partIdx_;         // index into Part_*
    std::shared_ptr<std::vector<XYZTVectorF>>       Photon_fourMomentum_;    // VECP(1..4,i)
    std::shared_ptr<std::vector<int>>               Photon_lock_;            // LVLOCK quality word
    std::shared_ptr<std::vector<bool>>              Photon_isHPC_;           // barrel EM (42-138 deg)
    std::shared_ptr<std::vector<bool>>              Photon_isSTIC_;          // forward small-angle EM
    std::shared_ptr<std::vector<bool>>              Photon_isFEMC_;          // endcap EM (FEMC), non-HPC/STIC
    std::shared_ptr<std::vector<float>>             Photon_hpcShowerEnergy_; // QHPC(1) most-energetic
    std::shared_ptr<std::vector<float>>             Photon_hpcTotalShowerEnergy_; // QHPC(8)
    std::shared_ptr<std::vector<int>>               Photon_hpcNumLayers_;    // KHPC(5) shower depth
    std::shared_ptr<std::vector<int>>               Photon_hpcLayerHitPattern_;  // KHPC(6) 9-layer bits
    std::shared_ptr<std::vector<int>>               Photon_hpcNumAssociatedShowers_; // KHPC(7)
    std::shared_ptr<std::vector<int>>               Photon_hpcParticleCode_; // KHPC(4)
    std::shared_ptr<std::vector<float>>             Photon_sticShowerEnergy_;// QSTIC(1)
    std::shared_ptr<std::vector<int>>               Photon_sticNumTowers_;   // KSTIC(4)
    std::shared_ptr<std::vector<int>>               Photon_sticChargedTag_;  // KSTIC(5) charged-track veto
    std::shared_ptr<std::vector<float>>             Photon_emEnergy_;        // QEMF(8) total EM (FEMC merged)
    std::shared_ptr<std::vector<float>>             Photon_hadEnergy_;       // QHAC(8) leakage for EM/HAD

    // Tier 2 (photon ID in HPC, per-track from PSCPHO / PSCPI0)
    std::shared_ptr<std::vector<float>>             Photon_hpcEnergyWeightedDepth_;     // QPHOT(1)
    std::shared_ptr<std::vector<int>>               Photon_hpcPhotNumClusters_;         // KPHOT(2)
    std::shared_ptr<std::vector<int>>               Photon_hpcPhotFirstLayer_;          // KPHOT(3)
    std::shared_ptr<std::vector<int>>               Photon_hpcPhotMaxConsecutiveLayers_;// KPHOT(5)
    std::shared_ptr<std::vector<float>>             Photon_hpcTransverseFluctuation_;   // QPHOT(6)
    std::shared_ptr<std::vector<float>>             Photon_pi0FittedMass_;              // QPI0(12) — THE π⁰ discriminator
    std::shared_ptr<std::vector<float>>             Photon_pi0Chi2_;                    // QPI0(22)
    std::shared_ptr<std::vector<float>>             Photon_pi0GammaChi2_;               // QPI0(24)
    std::shared_ptr<std::vector<int>>               Photon_pi0NumConnectedMaxima_;      // KPI0(5)
    std::shared_ptr<std::vector<int>>               Photon_pi0NumExpectedMaxima_;       // KPI0(6)
    std::shared_ptr<std::vector<float>>             Photon_pi0RelE1stGaussian_;         // QPI0(7)

    // --- Photon-conversion collection (PSCPHC, per-conversion — separate from Photon_*) ---
    std::shared_ptr<int16_t>                        nPhotonConv_;
    std::shared_ptr<std::vector<int16_t>>           PhotonConv_firstDaughterIdx_;       // KPHOC(1)-1
    std::shared_ptr<std::vector<int16_t>>           PhotonConv_secondDaughterIdx_;      // KPHOC(2)-1
    std::shared_ptr<std::vector<int16_t>>           PhotonConv_simPhotonIdx_;           // KPHOC(3)-1
    std::shared_ptr<std::vector<int8_t>>            PhotonConv_pairType_;               // KPHOC(4): -21/-22/-23=2-TPC I/II/III, -24=1-TPC
    std::shared_ptr<std::vector<XYZTVectorF>>       PhotonConv_fourMomentum_;           // QPHOC(5..8)
    std::shared_ptr<std::vector<float>>             PhotonConv_trackLength_;            // QPHOC(9)
    std::shared_ptr<std::vector<XYZPointF>>         PhotonConv_vertex_;                 // QPHOC(10..12)

    // --- Unassociated Track Element (UTER) collection (PSCTER, SKELANA A.1.7) ---
    // Lowest-level hit-like info that SKELANA exposes. For particle-flow reco,
    // these are track-elements that the standard pattern recognition did NOT
    // associate with any existing charged track object. Useful as a starting
    // point for extended PF, MET recovery, forward-pointing γ tracking, etc.
    // SKELANA does NOT expose per-cell HPC/HAC/STIC calo hits — those are in
    // raw SDST banks and would need a PHDST-level reader below SKELANA.
    std::shared_ptr<int16_t>                        nUter_;
    std::shared_ptr<std::vector<int8_t>>            Uter_detector_;     // 0 = OD, 1 = FCA, 2 = FCB
    std::shared_ptr<std::vector<int>>               Uter_dataDescriptor_;  // K..(1)
    std::shared_ptr<std::vector<float>>             Uter_coord1_;       // Q..(2)
    std::shared_ptr<std::vector<float>>             Uter_coord2_;       // Q..(3)
    std::shared_ptr<std::vector<float>>             Uter_coord3_;       // Q..(4)
    std::shared_ptr<std::vector<float>>             Uter_theta_;        // Q..(5)
    std::shared_ptr<std::vector<float>>             Uter_phi_;          // Q..(6)
    std::shared_ptr<std::vector<float>>             Uter_invPOrPt_;     // Q..(7)
    std::shared_ptr<std::vector<float>>             Uter_elementLength_;// Q..(8)

    // --- Reconstructed V0 collection (PSCRV0, SKELANA A.1.5) ---
    std::shared_ptr<int16_t>                              nV0_;
    std::shared_ptr<std::vector<int16_t>>                 V0_firstPartIdx_;      // KRV0(1)-1
    std::shared_ptr<std::vector<int16_t>>                 V0_secondPartIdx_;     // KRV0(2)-1
    std::shared_ptr<std::vector<int16_t>>                 V0_incomingIdx_;       // KRV0(3)-1 (0 means 'none' — we emit -1)
    std::shared_ptr<std::vector<int16_t>>                 V0_mcIncomingIdx_;     // KRV0(4)-1
    std::shared_ptr<std::vector<int>>                     V0_tagFlag_;           // KRV0(5) bitmask (22=K0, 33=Lambda, 0=loose, 1=tight, 2/3=bkg)
    std::shared_ptr<std::vector<float>>                   V0_momentum_;          // QRV0(6)
    std::shared_ptr<std::vector<float>>                   V0_vertexChi2Prob_;    // QRV0(7)
    std::shared_ptr<std::vector<XYZPointF>>               V0_vertex_;            // QRV0(8..10), cm
    std::shared_ptr<std::vector<float>>                   V0_xyFlightDistanceOverError_;  // QRV0(11)
    std::shared_ptr<std::vector<float>>                   V0_xyFlightAngle_;     // QRV0(12), rad
    std::shared_ptr<std::vector<XYZVectorF>>              V0_pPlus_;             // QRV0(13..15), GeV
    std::shared_ptr<std::vector<XYZVectorF>>              V0_pMinus_;            // QRV0(16..18), GeV
    std::shared_ptr<std::vector<float>>                   V0_suggestedMass_;     // QRV0(19) (neg for antipart)
    std::shared_ptr<std::vector<float>>                   V0_impactEps_;         // QRV0(20)
    std::shared_ptr<std::vector<float>>                   V0_impactZ_;           // QRV0(21)
    std::shared_ptr<std::vector<float>>                   V0_neutralTheta_;      // QRV0(22)
    std::shared_ptr<std::vector<float>>                   V0_neutralPhi_;        // QRV0(23)

    // --- V0 hypothesis sub-collection (PSCHV0, up to 8 per V0) ---
    std::shared_ptr<int16_t>                              nV0Hyp_;
    std::shared_ptr<std::vector<int16_t>>                 V0Hyp_v0Idx_;          // index into V0_*
    std::shared_ptr<std::vector<int8_t>>                  V0Hyp_slot_;           // 1..8 (n in the manual)
    std::shared_ptr<std::vector<int8_t>>                  V0Hyp_kind_;           // 1,2,11,12,21,22,31,32 or -1=failed
    std::shared_ptr<std::vector<float>>                   V0Hyp_probability_;
    std::shared_ptr<std::vector<XYZPointF>>               V0Hyp_vertex_;
    std::shared_ptr<std::vector<XYZVectorF>>              V0Hyp_momentum_;
    std::shared_ptr<std::vector<float>>                   V0Hyp_impactXY_;
    std::shared_ptr<std::vector<float>>                   V0Hyp_impactXYErr_;
    std::shared_ptr<std::vector<float>>                   V0Hyp_impactZ_;
    std::shared_ptr<std::vector<float>>                   V0Hyp_impactZErr_;
    std::shared_ptr<std::vector<float>>                   V0Hyp_impact3D_;
    std::shared_ptr<std::vector<float>>                   V0Hyp_impact3DErr_;

    std::shared_ptr<int16_t> nJet_;
    std::shared_ptr<std::vector<XYZTVectorF>> Jet_fourMomentum_;
    std::shared_ptr<std::vector<int8_t>> Jet_charge_;

    std::shared_ptr<float> Jet_thrust_;
    std::shared_ptr<float> Jet_oblatness_;
    std::shared_ptr<std::vector<XYZVectorF>> Jet_thrustVector_;
    std::shared_ptr<float> Jet_sphericity_;
    std::shared_ptr<float> Jet_aplanarity_;
    std::shared_ptr<std::vector<XYZVectorF>> Jet_sphericityVector_;

    std::shared_ptr<int16_t> nSimPart_;
    std::shared_ptr<std::vector<XYZTVectorF>> SimPart_fourMomentum_;
    std::shared_ptr<std::vector<int16_t>> SimPart_charge_;
    std::shared_ptr<std::vector<int16_t>> SimPart_pdgId_;
    std::shared_ptr<std::vector<int16_t>> SimPart_partIdx_;
    std::shared_ptr<std::vector<int16_t>> SimPart_genIdx_;
    std::shared_ptr<std::vector<int16_t>> SimPart_originVtxIdx_;
    std::shared_ptr<std::vector<int16_t>> SimPart_decayVtxIdx_;

    std::shared_ptr<int16_t> nGenPart_;
    std ::shared_ptr<std::vector<int16_t>> GenPart_status_;
    std::shared_ptr<std::vector<int16_t>> GenPart_pdgId_;
    std::shared_ptr<std::vector<int16_t>> GenPart_parentIdx_;
    std::shared_ptr<std::vector<int16_t>> GenPart_firstChildIdx_;
    std::shared_ptr<std::vector<int16_t>> GenPart_lastChildIdx_;
    std::shared_ptr<std::vector<XYZTVectorF>> GenPart_fourMomentum_;
    std::shared_ptr<std::vector<XYZTVectorF>> GenPart_fourPosition_;
    std::shared_ptr<std::vector<float>> GenPart_tau_;
    std::shared_ptr<std::vector<int16_t>> GenPart_simIdx_;
    std::shared_ptr<std::vector<float>> GenPart_mass_;

    std::shared_ptr<int16_t> nVtx_;
    std::shared_ptr<std::vector<int16_t>> Vtx_outgoingIdx_;
    std::shared_ptr<std::vector<int16_t>> Vtx_incomingIdx_;
    std::shared_ptr<std::vector<int16_t>> Vtx_nOutgoing_;
    std::shared_ptr<std::vector<int16_t>> Vtx_ndf_;
    std::shared_ptr<std::vector<XYZPointF>> Vtx_position_;
    std::shared_ptr<std::vector<float>> Vtx_chi2_;
    std::shared_ptr<std::vector<SMatrixSym3F>> Vtx_errorMatrix_;
    std::shared_ptr<std::vector<int16_t>> Vtx_errorFlag_;
    std::shared_ptr<std::vector<int16_t>> Vtx_status_;

    std::shared_ptr<int16_t> nSimVtx_;
    std::shared_ptr<std::vector<int16_t>> SimVtx_outgoingIdx_;
    std::shared_ptr<std::vector<int16_t>> SimVtx_incomingIdx_;
    std::shared_ptr<std::vector<int16_t>> SimVtx_nOutgoing_;
    std::shared_ptr<std::vector<int16_t>> SimVtx_masscode_;
    std::shared_ptr<std::vector<XYZPointF>> SimVtx_position_;

    std::shared_ptr<XYZVectorF> Beam_position_;
    std::shared_ptr<XYZVectorF> Beam_size_;

    std::shared_ptr<std::vector<int16_t>> Trac_partIdx_;
    std::shared_ptr<std::vector<int16_t>> Trac_originVtxIdx_;
    std::shared_ptr<std::vector<int16_t>> Trac_decayVtxIdx_;
    std::shared_ptr<std::vector<SVector5F>> Trac_perigee_;
    std::shared_ptr<std::vector<ROOT::Math::SMatrixSym5F>> Trac_weightMatrix_;
    std::shared_ptr<std::vector<float>> Trac_length_;
    std::shared_ptr<std::vector<int16_t>> Trac_detectors_;
    std::shared_ptr<std::vector<float>> Trac_firstPointR_;
    std::shared_ptr<std::vector<float>> Trac_firstPointZ_;
    std::shared_ptr<std::vector<float>> Trac_chi2NoVD_;
    std::shared_ptr<std::vector<float>> Trac_chi2VD_;
    std::shared_ptr<std::vector<int16_t>> Trac_ndfNoVD_;
    std::shared_ptr<std::vector<int16_t>> Trac_ndfVD_;
    std::shared_ptr<std::vector<int16_t>> Trac_vdHitsRPhi_;
    std::shared_ptr<std::vector<int16_t>> Trac_vdHitsZ_;
    std::shared_ptr<std::vector<float>> Trac_resRPhiFirstPoint_;
    std::shared_ptr<std::vector<float>> Trac_errorResRPhiFirstPoint_;
    std::shared_ptr<std::vector<float>> Trac_resZFirstPoint_;
    std::shared_ptr<std::vector<float>> Trac_errorResZFirstPoint_;
    std::shared_ptr<std::vector<float>> Trac_impParRPhi_;
    std::shared_ptr<std::vector<float>> Trac_impParZ_;
    std::shared_ptr<std::vector<float>> Trac_impParToVertexRPhi_;
    std::shared_ptr<std::vector<float>> Trac_impParToVertexZ_;
    std::shared_ptr<std::vector<float>> Trac_impParToBeamSpotRPhi_;
    std::shared_ptr<std::vector<float>> Trac_chi2VDHits_;

    std::shared_ptr<std::vector<int>> Muid_partIdx_;
    std::shared_ptr<std::vector<int>> Muid_tag_;
    std::shared_ptr<std::vector<float>> Muid_looseChi2_;
    std::shared_ptr<std::vector<int16_t>> Muid_hitPattern_;

    std::shared_ptr<std::vector<int>> Elid_partIdx_;
    std::shared_ptr<std::vector<int>> Elid_tag_;
    std::shared_ptr<std::vector<int16_t>> Elid_gammaConversion_;
    std::shared_ptr<std::vector<float>> Elid_px_;
    std::shared_ptr<std::vector<float>> Elid_py_;
    std::shared_ptr<std::vector<float>> Elid_pz_;

    std::shared_ptr<std::vector<int>> Haid_sign_;
    std::shared_ptr<std::vector<int>> Haid_kaonDedx_;
    std::shared_ptr<std::vector<int>> Haid_protonDedx_;
    std::shared_ptr<std::vector<int>> Haid_kaonRich_;
    std::shared_ptr<std::vector<int>> Haid_protonRich_;
    std::shared_ptr<std::vector<int>> Haid_pionRich_;
    std::shared_ptr<std::vector<float>> Haid_kaonCombined_;
    std::shared_ptr<std::vector<float>> Haid_protonCombined_;
    std::shared_ptr<std::vector<int>>Haid_richQuality_;

    std::shared_ptr<std::vector<int8_t>> Haidn_pionTag_;
    std::shared_ptr<std::vector<int8_t>> Haidn_kaonTag_;
    std::shared_ptr<std::vector<int8_t>> Haidn_protonTag_;
    std::shared_ptr<std::vector<int8_t>> Haidn_heavyTag_;
    std::shared_ptr<std::vector<int8_t>> Haidn_pionTrackSelection_;
    std::shared_ptr<std::vector<int8_t>> Haidn_kaonTrackSelection_;
    std::shared_ptr<std::vector<int8_t>> Haidn_protonTrackSelection_;
    std::shared_ptr<std::vector<int8_t>> Haidn_heavyTrackSelection_;

    std::shared_ptr<std::vector<int8_t>> Haidr_pionTag_;
    std::shared_ptr<std::vector<int8_t>> Haidr_kaonTag_;
    std::shared_ptr<std::vector<int8_t>> Haidr_protonTag_;
    std::shared_ptr<std::vector<int8_t>> Haidr_heavyTag_;
    std::shared_ptr<std::vector<int8_t>> Haidr_electronTag_;
    std::shared_ptr<std::vector<int8_t>> Haidr_selectionFlag_;

    std::shared_ptr<std::vector<int8_t>> Haide_pionTag_;
    std::shared_ptr<std::vector<int8_t>> Haide_kaonTag_;
    std::shared_ptr<std::vector<int8_t>> Haide_protonTag_;
    std::shared_ptr<std::vector<int8_t>> Haide_heavyTag_;
    std::shared_ptr<std::vector<int8_t>> Haide_electronTag_;
    std::shared_ptr<std::vector<int8_t>> Haide_selectionFlag_;

    std::shared_ptr<std::vector<int8_t>> Haidc_pionTag_;
    std::shared_ptr<std::vector<int8_t>> Haidc_kaonTag_;
    std::shared_ptr<std::vector<int8_t>> Haidc_protonTag_;
    std::shared_ptr<std::vector<int8_t>> Haidc_heavyTag_;
    std::shared_ptr<std::vector<int8_t>> Haidc_electronTag_;
    std::shared_ptr<std::vector<int8_t>> Haidc_selectionFlag_;

    std::shared_ptr<std::vector<float>>  Dedx_value_;
    std::shared_ptr<std::vector<float>>  Dedx_width_;
    std::shared_ptr<std::vector<int16_t>> Dedx_nrWires_; 
    std::shared_ptr<std::vector<float>> Dedx_gapWires_; 
    std::shared_ptr<std::vector<float>>  Dedx_error_;
    std::shared_ptr<std::vector<float>>  Dedx_valueVD_; 
    std::shared_ptr<std::vector<int16_t>> Dedx_nrVDHits_;

    std::shared_ptr<std::vector<float>> Rich_theg_;
    std::shared_ptr<std::vector<float>> Rich_sigg_;
    std::shared_ptr<std::vector<int16_t>> Rich_nphg_;
    std::shared_ptr<std::vector<float>> Rich_nepg_;
    std::shared_ptr<std::vector<int16_t>> Rich_flagg_;
    std::shared_ptr<std::vector<float>> Rich_thel_;
    std::shared_ptr<std::vector<float>> Rich_sigl_;
    std::shared_ptr<std::vector<int16_t>> Rich_nphl_;
    std::shared_ptr<std::vector<float>> Rich_nepl_;
    std::shared_ptr<std::vector<int16_t>> Rich_flagl_;

    std::shared_ptr<std::vector<float>> Btag_probNegIP_;
    std::shared_ptr<std::vector<float>> Btag_probPosIP_;
    std::shared_ptr<std::vector<float>> Btag_probAllIP_;
    std::shared_ptr<XYZVectorF> Btag_thrustVector_; 

    // --- VD Associated hits ---
    std::shared_ptr<int> nVdHit_;
    std::shared_ptr<std::vector<int>>   VdHit_trackIdx_;
    std::shared_ptr<std::vector<int>>   VdHit_module_;
    std::shared_ptr<std::vector<float>> VdHit_localX_;
    std::shared_ptr<std::vector<float>> VdHit_R_;
    std::shared_ptr<std::vector<float>> VdHit_RPhi_;
    std::shared_ptr<std::vector<float>> VdHit_signalToNoise_;
    
    // --- VD Unassociated hits ---
    std::shared_ptr<int> nVdUnHit_;
    std::shared_ptr<std::vector<int>>   VdUnHit_module_;
    std::shared_ptr<std::vector<float>> VdUnHit_localX_;
    std::shared_ptr<std::vector<float>> VdUnHit_R_;
    std::shared_ptr<std::vector<float>> VdUnHit_RPhi_;
    std::shared_ptr<std::vector<float>> VdUnHit_signalToNoise_;

    // --- Beam spot ---
    std::shared_ptr<int>   BeamSpot_errorFlag_;
    std::shared_ptr<float> BeamSpot_x_;
    std::shared_ptr<float> BeamSpot_y_;
    std::shared_ptr<float> BeamSpot_z_;
    std::shared_ptr<float> BeamSpot_sigmaX_;
    std::shared_ptr<float> BeamSpot_sigmaY_;
    std::shared_ptr<float> BeamSpot_sigmaZ_;

    // --- MC hadron tagging (FCC-style, only when mc_ && IFLSIM) ---
    std::shared_ptr<std::vector<int8_t>> Part_fromBc_;
    std::shared_ptr<std::vector<int8_t>> Part_fromBs_;
    std::shared_ptr<std::vector<int8_t>> Part_fromBu_;
    std::shared_ptr<std::vector<int8_t>> Part_fromBd_;
    std::shared_ptr<std::vector<int8_t>> Part_fromLb_;
    std::shared_ptr<int> n_Bc_Emin_;
    std::shared_ptr<int> n_Bs_Emin_;
    std::shared_ptr<int> n_Bu_Emin_;
    std::shared_ptr<int> n_Bd_Emin_;
    std::shared_ptr<int> n_Lb_Emin_;
    std::shared_ptr<int> n_Bc_Emax_;
    std::shared_ptr<int> n_Bs_Emax_;
    std::shared_ptr<int> n_Bu_Emax_;
    std::shared_ptr<int> n_Bd_Emax_;
    std::shared_ptr<int> n_Lb_Emax_;
    std::shared_ptr<int> label_Bc_Emin_;
    std::shared_ptr<int> label_Bs_Emin_;
    std::shared_ptr<int> label_Bu_Emin_;
    std::shared_ptr<int> label_Bd_Emin_;
    std::shared_ptr<int> label_Lb_Emin_;
    std::shared_ptr<int> label_light_Emin_;
    std::shared_ptr<int> label_hasBc_Emin_;
    std::shared_ptr<int> label_has1Bc_Emin_;
    std::shared_ptr<int> label_Bc_Emax_;
    std::shared_ptr<int> label_Bs_Emax_;
    std::shared_ptr<int> label_Bu_Emax_;
    std::shared_ptr<int> label_Bd_Emax_;
    std::shared_ptr<int> label_Lb_Emax_;
    std::shared_ptr<int> label_light_Emax_;
    std::shared_ptr<int> label_hasBc_Emax_;
    std::shared_ptr<int> label_has1Bc_Emax_;
    
    // --- Method declarations ---
    void defineVdHit(std::unique_ptr<RNTupleModel> &model);
    void fillVdHit();
    void defineVdUnHit(std::unique_ptr<RNTupleModel> &model);
    void fillVdUnHit();

    void defineHadronTagging(std::unique_ptr<RNTupleModel> &model);
    void fillHadronTagging(const eventData &eData);
};

#endif // NANOAOD_WRITER_HPP
