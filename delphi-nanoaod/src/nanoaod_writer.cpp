#include "nanoaod_writer.hpp"
#include "hadron_tagging.hpp"
#include "phdst.hpp"
#include <Math/PositionVector3D.h>
#include <Math/DisplacementVector3D.h>
#include <Math/Vector4D.h>
#include <Math/Vector3D.h>
#include <Math/SMatrix.h>

#include <TTree.h>

#include "TDatabasePDG.h"
#include "TParticlePDG.h"

//Additional common block to extract jet releateted variables
extern "C" struct
{
    float thr;
    float obl;
    float sph;
    float apl;
} psjval_;

float &SPH = psjval_.sph;
float &APL = psjval_.apl;
float &THR = psjval_.thr;
float &OBL = psjval_.obl;

// MakeField with type deduction
template <typename T>
void MakeField(std::unique_ptr<RNTupleModel> &model, const std::string &name, const std::string &description, std::shared_ptr<T> &ptr)
{
    ptr = model->MakeField<T>({name, description});
}

// Fill a vector with n elements using a function f
template <typename T, typename Func>
void fillVector(std::shared_ptr<std::vector<T>> &ptr, int i, int n, Func f)
{
    ptr->clear();
    std::generate_n(std::back_inserter(*ptr), n, [i, f]() mutable
                    { return f(i++); });
}

NanoAODWriter::NanoAODWriter() : output_("nanoaod.root"), mc_(false) {};

NanoAODWriter::~NanoAODWriter() {};

NanoAODWriter *NanoAODWriter::getInstance()
{
    if (instance_ == nullptr)
    {
        instance_ = new NanoAODWriter();
    }
    return static_cast<NanoAODWriter *>(instance_);
};

void NanoAODWriter::setOutput(const std::filesystem::path &output)
{
    output_ = output;
};

void NanoAODWriter::setMC()
{
    mc_ = true;
};

void NanoAODWriter::user00()
{
    super::user00();

    std::unique_ptr<RNTupleModel> model = RNTupleModel::Create();

    defineEvent(model);
    definePart(model);
    definePhoton(model);
    defineVtx(model);
    if (sk::IFLRV0 > 0)
    {
        defineV0(model);
    }
    if (sk::IFLPHC > 0)
    {
        definePhotonConv(model);
    }
    if (sk::IFLUTE > 0)
    {
        defineUter(model);
    }

    if (sk::IFLJET > 0)
    {
        defineJet(model);
    }
    if (mc_ && sk::IFLSIM > 0)
    {
        defineSimPart(model);
        defineGenPart(model);
        defineSimVtx(model);
        defineHadronTagging(model);
    }
    if (sk::IFLTRA > 0)
    {
        defineTrac(model);
        defineVdHit(model);
        defineVdUnHit(model);
    }
    if (sk::IFLMUO > 0)
    {
        defineMuid(model);
    }
    if (sk::IFLELE > 0)
    {
        defineElid(model);
    }
    if (sk::IFLHAD > 0)
    {
        defineHadid(model);
    }
    if (sk::IFLBTG > 0)
    {
        defineBtag(model);
    }
    if (sk::IFLBSP > 0)
    {
        defineBeamSpot(model);
    }
    if (sk::IFLBSP > 0)
    {
        fillBeamSpot();
    }

    // writer_ = RNTupleWriter::Recreate(std::move(model), "Events", output_.string());

    file_ = TFile::Open(output_.string().c_str(), "recreate");
    writer_ = RNTupleWriter::Append(std::move(model), "Events", *file_);
    // file_ = TFile::CurrentFile();
    // file_->cd();
    out_t = new TTree("t", "t");
    out_tgen = new TTree("tgen", "tgen");
    out_tsim = new TTree("tgenBefore", "tgenBefore");
    out_t->SetDirectory(file_);
    out_tgen->SetDirectory(file_);
    out_tsim->SetDirectory(file_);

    // register branches
    do_chThrust           = false;
    do_neuThrust          = false;
    do_thrustCorr         = false;
    do_thrustCorrInverse  = false;
    do_thrustMissP        = true;
    out_pData.SetBranchWrite(out_t, 1);
    out_eData.SetBranchWrite(out_t, 0);

    out_pData_gen.SetBranchWrite(out_tgen, 1);
    out_eData_gen.SetBranchWrite(out_tgen, 0);

    out_pData_sim.SetBranchWrite(out_tsim, 1);
    out_eData_sim.SetBranchWrite(out_tsim, 0);

    out_t->Branch("EMF", emf, "EMF[nParticle]/F");
    out_t->Branch("HPC", hpc, "HPC[nParticle]/F");
    out_t->Branch("HAC", hac, "HAC[nParticle]/F");
    out_t->Branch("STIC", stic, "STIC[nParticle]/F");
    out_t->Branch("LOCK", lock, "LOCK[nParticle]/I");

    out_t->Branch("chi2ndf",   chi2ndf,   "chi2ndf[nParticle]/F");
    out_t->Branch("chi2ndfVD", chi2ndfVD, "chi2ndfVD[nParticle]/F");
    out_t->Branch("dpp",       dpp,       "dpp[nParticle]/F");

    if (mc_ && sk::IFLSIM > 0)
    {
        out_t->Branch("Part_fromBc", Part_fromBc_.get());
        out_t->Branch("Part_fromBs", Part_fromBs_.get());
        out_t->Branch("Part_fromBu", Part_fromBu_.get());
        out_t->Branch("Part_fromBd", Part_fromBd_.get());
        out_t->Branch("Part_fromLb", Part_fromLb_.get());
        out_t->Branch("n_Bc_Emin", n_Bc_Emin_.get(), "n_Bc_Emin/I");
        out_t->Branch("n_Bs_Emin", n_Bs_Emin_.get(), "n_Bs_Emin/I");
        out_t->Branch("n_Bu_Emin", n_Bu_Emin_.get(), "n_Bu_Emin/I");
        out_t->Branch("n_Bd_Emin", n_Bd_Emin_.get(), "n_Bd_Emin/I");
        out_t->Branch("n_Lb_Emin", n_Lb_Emin_.get(), "n_Lb_Emin/I");
        out_t->Branch("n_Bc_Emax", n_Bc_Emax_.get(), "n_Bc_Emax/I");
        out_t->Branch("n_Bs_Emax", n_Bs_Emax_.get(), "n_Bs_Emax/I");
        out_t->Branch("n_Bu_Emax", n_Bu_Emax_.get(), "n_Bu_Emax/I");
        out_t->Branch("n_Bd_Emax", n_Bd_Emax_.get(), "n_Bd_Emax/I");
        out_t->Branch("n_Lb_Emax", n_Lb_Emax_.get(), "n_Lb_Emax/I");
        out_t->Branch("label_Bc_Emin", label_Bc_Emin_.get(), "label_Bc_Emin/I");
        out_t->Branch("label_Bs_Emin", label_Bs_Emin_.get(), "label_Bs_Emin/I");
        out_t->Branch("label_Bu_Emin", label_Bu_Emin_.get(), "label_Bu_Emin/I");
        out_t->Branch("label_Bd_Emin", label_Bd_Emin_.get(), "label_Bd_Emin/I");
        out_t->Branch("label_Lb_Emin", label_Lb_Emin_.get(), "label_Lb_Emin/I");
        out_t->Branch("label_light_Emin", label_light_Emin_.get(), "label_light_Emin/I");
        out_t->Branch("label_hasBc_Emin", label_hasBc_Emin_.get(), "label_hasBc_Emin/I");
        out_t->Branch("label_has1Bc_Emin", label_has1Bc_Emin_.get(), "label_has1Bc_Emin/I");
        out_t->Branch("label_Bc_Emax", label_Bc_Emax_.get(), "label_Bc_Emax/I");
        out_t->Branch("label_Bs_Emax", label_Bs_Emax_.get(), "label_Bs_Emax/I");
        out_t->Branch("label_Bu_Emax", label_Bu_Emax_.get(), "label_Bu_Emax/I");
        out_t->Branch("label_Bd_Emax", label_Bd_Emax_.get(), "label_Bd_Emax/I");
        out_t->Branch("label_Lb_Emax", label_Lb_Emax_.get(), "label_Lb_Emax/I");
        out_t->Branch("label_light_Emax", label_light_Emax_.get(), "label_light_Emax/I");
        out_t->Branch("label_hasBc_Emax", label_hasBc_Emax_.get(), "label_hasBc_Emax/I");
        out_t->Branch("label_has1Bc_Emax", label_has1Bc_Emax_.get(), "label_has1Bc_Emax/I");
    }

    pdgDatabase = TDatabasePDG::Instance();
};

int NanoAODWriter::user01()
{
    return super::user01();
};

void NanoAODWriter::user02()
{
    super::user02();

    fillEvent();
    fillPart();
    fillPhoton();
    fillVtx();
    if (sk::IFLRV0 > 0)
    {
        fillV0();
    }
    if (sk::IFLPHC > 0)
    {
        fillPhotonConv();
    }
    if (sk::IFLUTE > 0)
    {
        fillUter();
    }
    if (sk::IFLJET > 0)
    {
        fillJet();
    }
    if (mc_ && sk::IFLSIM > 0)
    {
        fillSimPart();
        fillGenPart();
        fillSimVtx();
    }
    if (sk::IFLTRA > 0)
    {
        fillTrac();
        fillVdHit();
        fillVdUnHit();
    }
    if (sk::IFLMUO > 0)
    {
        fillMuid();
    }
    if (sk::IFLELE > 0)
    {
        fillElid();
    }
    if (sk::IFLBTG > 0)
    {
        fillBtag();
    }
    if (sk::IFLHAD > 0)
    {
        fillHadid();
    }

    fillPartLoop(out_pData, out_eData, DataKind::data);
    fillSelection(out_pData, out_eData);
    if (mc_ && sk::IFLSIM > 0)
    {
        fillHadronTagging(out_eData);
    }

    writer_->Fill();
    out_t->Fill();
    if (mc_) {
        fillPartLoop(out_pData_gen, out_eData_gen, DataKind::gen);
        fillSelection(out_pData_gen, out_eData_gen);
        out_tgen->Fill();

        fillPartLoop(out_pData_sim, out_eData_sim, DataKind::sim);
        fillSelection(out_pData_sim, out_eData_sim);
        out_tsim->Fill();
    }
};

void NanoAODWriter::defineEvent(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "Event_runNumber", "Run number", Event_runNumber_);
    MakeField(model, "Event_evtNumber", "Event number", Event_eventNumber_);
    MakeField(model, "Event_fillNumber", "Fill number", Event_fillNumber_);
    MakeField(model, "Event_date", "Date", Event_date_);
    MakeField(model, "Event_time", "Time", Event_time_);
    MakeField(model, "Event_magField", "Magnetic field", Event_magField_);
    MakeField(model, "Event_shortDstVersion", "Short DST version", Event_shortDstVersion_);
    MakeField(model, "Event_hadronTagT4", "Team4 Hadron tag", Event_hadronTagT4_);
    MakeField(model, "Event_chargedMultT4", "Charged Multiplicity (Team 4)", Event_chargedMultT4_);
    MakeField(model, "Event_chargedMult", "Charged Multiplicity", Event_chargedMult_);
    MakeField(model, "Event_neutralMult", "Neutral Multiplicity", Event_neutralMult_);
    MakeField(model, "Event_cmEnergy", "Center of mass energy", Event_cmEnergy_);
    MakeField(model, "Event_totalChargedEnergy", "Total charged energy", Event_totalChargedEnergy_);
    MakeField(model, "Event_totalEMEnergy", "Total electromagnetic energy", Event_totalEMEnergy_);
    MakeField(model, "Event_totalHadronicEnergy", "Total hadronic energy", Event_totalHadronicEnergy_);
    MakeField(model, "Event_DSTType", "DST type", Event_DSTType_);
}

void NanoAODWriter::fillEvent()
{
    *Event_runNumber_ = phdst::IIIRUN;
    *Event_eventNumber_ = phdst::IIIEVT;
    *Event_fillNumber_ = phdst::IIFILL;
    *Event_date_ = phdst::IIIDAT;
    *Event_time_ = phdst::IIITIM;
    *Event_magField_ = sk::BMAG;
    *Event_shortDstVersion_ = sk::ISVER;
    *Event_hadronTagT4_ = sk::IHAD4 > 0;
    *Event_chargedMultT4_ = sk::NCTR4;
    *Event_chargedMult_ = sk::NCTRK;
    *Event_neutralMult_ = sk::NNTRK;
    *Event_cmEnergy_ = sk::ECMAS;
    *Event_totalChargedEnergy_ = sk::ECHAR;
    *Event_totalEMEnergy_ = sk::EMNEU;
    *Event_totalHadronicEnergy_ = sk::EHNEU;
    *Event_DSTType_ = sk::CDTYPE();

    out_pData.RunNo = *Event_runNumber_;
    out_pData.EventNo = *Event_eventNumber_;
    out_pData_gen.RunNo = out_pData.RunNo;
    out_pData_gen.EventNo = out_pData.EventNo;
    out_pData_sim.RunNo = out_pData.RunNo;
    out_pData_sim.EventNo = out_pData.EventNo;

    out_pData.year = 0; // temp //
	  // out_pData.subDir = -999;
	  // out_pData.process = -999;
    out_pData.source = 0; // temp //
    out_pData.isMC = mc_;
    out_pData.isOnres = false;
    out_pData_gen.isMC = true;
    out_pData_gen.isOnres = false;
    out_pData_sim.isMC = true;
    out_pData_sim.isOnres = false;
    out_pData.particleWeight = 1;
    out_pData_gen.particleWeight = 1;
    out_pData_sim.particleWeight = 1;
    out_pData.bx  = sk::XYZBS(1);
    out_pData.by  = sk::XYZBS(2);
    out_pData.ebx = sk::DXYZBS(1);
    out_pData.eby = sk::DXYZBS(2);
    out_pData.Energy = *Event_cmEnergy_;
    out_pData_gen.Energy = *Event_cmEnergy_;
    out_pData_sim.Energy = *Event_cmEnergy_;
}

void NanoAODWriter::definePart(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nPart", "Number of particles", nPart_);
    MakeField(model, "Part_fourMomentum", "Particle 4-momentum", Part_fourMomentum_);
    MakeField(model, "Part_charge", "Particle charge", Part_charge_);
    MakeField(model, "Part_pdgId", "Particle PDG ID", Part_pdgId_);
    MakeField(model, "Part_massId", "Particle mass ID", Part_massId_);
    MakeField(model, "Part_jetIdx", "Jet index", Part_jetIdx_);
    MakeField(model, "Part_hemisphereIdx", "Hemisphere index", Part_hemisphereIdx_);
    MakeField(model, "Part_vtxCode", "Vertex code", Part_vtxCode_);
    MakeField(model, "Part_vtxIdx", "Vertex index", Part_vtxIdx_);
    MakeField(model, "Part_lock", "Particle lock", Part_lock_);
    MakeField(model, "Part_hpcShowerEnergy", "QHPC(1): energy of the most energetic HPC shower", Part_hpcShowerEnergy_);
    MakeField(model, "Part_hpcShowerTheta", "QHPC(2): theta of the most energetic HPC shower", Part_hpcShowerTheta_);
    MakeField(model, "Part_hpcShowerPhi", "QHPC(3): phi of the most energetic HPC shower", Part_hpcShowerPhi_);
    MakeField(model, "Part_hpcParticleCode", "KHPC(4): particle identification code for the HPC shower", Part_hpcParticleCode_);
    MakeField(model, "Part_hpcNumLayers", "KHPC(5): number of HPC layers", Part_hpcNumLayers_);
    MakeField(model, "Part_hpcLayerHitPattern", "KHPC(6): HPC layer hit pattern", Part_hpcLayerHitPattern_);
    MakeField(model, "Part_hpcNumAssociatedShowers", "KHPC(7): number of associated HPC showers", Part_hpcNumAssociatedShowers_);
    MakeField(model, "Part_hpcTotalShowerEnergy", "QHPC(8): total HPC shower energy", Part_hpcTotalShowerEnergy_);
    MakeField(model, "Part_hacShowerEnergy", "QHAC(1): energy of the most energetic HAC shower", Part_hacShowerEnergy_);
    MakeField(model, "Part_hacShowerTheta", "QHAC(2): theta of the most energetic HAC shower", Part_hacShowerTheta_);
    MakeField(model, "Part_hacShowerPhi", "QHAC(3): phi of the most energetic HAC shower", Part_hacShowerPhi_);
    MakeField(model, "Part_hacParticleCode", "KHAC(4): particle identification code for the HAC shower", Part_hacParticleCode_);
    MakeField(model, "Part_hacNumTowers", "KHAC(5): number of HAC towers", Part_hacNumTowers_);
    MakeField(model, "Part_hacTowerHitPattern", "KHAC(6): HAC tower hit pattern", Part_hacTowerHitPattern_);
    MakeField(model, "Part_hacNumAssociatedShowers", "KHAC(7): number of associated HAC showers", Part_hacNumAssociatedShowers_);
    MakeField(model, "Part_hacTotalShowerEnergy", "QHAC(8): total HAC shower energy", Part_hacTotalShowerEnergy_);
    MakeField(model, "Part_sticShowerEnergy", "QSTIC(1): STIC shower energy", Part_sticShowerEnergy_);
    MakeField(model, "Part_sticShowerTheta", "QSTIC(2): STIC shower theta", Part_sticShowerTheta_);
    MakeField(model, "Part_sticShowerPhi", "QSTIC(3): STIC shower phi", Part_sticShowerPhi_);
    MakeField(model, "Part_sticNumTowers", "KSTIC(4): number of STIC towers in the shower", Part_sticNumTowers_);
    MakeField(model, "Part_sticChargedTag", "KSTIC(5): charged tag from the STIC veto", Part_sticChargedTag_);
    MakeField(model, "Part_sticSiliconVertexPos", "KSTIC(6): STIC silicon strip vertex position", Part_sticSiliconVertexPos_);

    if (mc_ && sk::IFLSIM > 0)
    {
        MakeField(model, "Part_simIdx", "Particle simulation index", Part_simIdx_);
    }
    MakeField(model, "Part_originVtxIdx", "Particle origin vertex index", Part_originVtxIdx_);
    MakeField(model, "Part_decayVtxIdx", "Particle decay vertex index", Part_decayVtxIdx_);
}

void NanoAODWriter::fillPart()
{
    *nPart_ = sk::NVECP;
    fillVector(Part_fourMomentum_, sk::LVPART, sk::NVECP, [](int i)
               { return XYZTVectorF(sk::VECP(1, i), sk::VECP(2, i), sk::VECP(3, i), sk::VECP(4, i)); });
    fillVector(Part_charge_, sk::LVPART, sk::NVECP, [](int i)
               { return int(sk::VECP(7, i)); });
    fillVector(Part_pdgId_, sk::LVPART, sk::NVECP, [](int i)
               { return int(sk::VECP(8, i)); });
    fillVector(Part_massId_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::IVECP(9, i); });
    fillVector(Part_jetIdx_, sk::LVPART, sk::NVECP, [](int i)
               { return (sk::IVECP(10, i) / 1000) % 10 - 1; });
    fillVector(Part_hemisphereIdx_, sk::LVPART, sk::NVECP, [](int i)
               { return (sk::IVECP(10, i) / 100) % 10 - 1; });
    fillVector(Part_vtxCode_, sk::LVPART, sk::NVECP, [](int i)
               { return (sk::IVECP(10, i) / 10) % 10; });
    fillVector(Part_vtxIdx_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::IVECP(10, i) % 10 - 1; });
    fillVector(Part_lock_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::LVLOCK(i); });
    fillVector(Part_hpcShowerEnergy_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::QHPC(1, i); });
    fillVector(Part_hpcShowerTheta_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::QHPC(2, i); });
    fillVector(Part_hpcShowerPhi_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::QHPC(3, i); });
    fillVector(Part_hpcParticleCode_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::KHPC(4, i); });
    fillVector(Part_hpcNumLayers_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::KHPC(5, i); });
    fillVector(Part_hpcLayerHitPattern_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::KHPC(6, i); });
    fillVector(Part_hpcNumAssociatedShowers_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::KHPC(7, i); });
    fillVector(Part_hpcTotalShowerEnergy_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::QHPC(8, i); });
    fillVector(Part_hacShowerEnergy_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::QHAC(1, i); });
    fillVector(Part_hacShowerTheta_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::QHAC(2, i); });
    fillVector(Part_hacShowerPhi_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::QHAC(3, i); });
    fillVector(Part_hacParticleCode_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::KHAC(4, i); });
    fillVector(Part_hacNumTowers_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::KHAC(5, i); });
    fillVector(Part_hacTowerHitPattern_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::KHAC(6, i); });
    fillVector(Part_hacNumAssociatedShowers_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::KHAC(7, i); });
    fillVector(Part_hacTotalShowerEnergy_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::QHAC(8, i); });
    fillVector(Part_sticShowerEnergy_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::QSTIC(1, i); });
    fillVector(Part_sticShowerTheta_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::QSTIC(2, i); });
    fillVector(Part_sticShowerPhi_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::QSTIC(3, i); });
    fillVector(Part_sticNumTowers_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::KSTIC(4, i); });
    fillVector(Part_sticChargedTag_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::KSTIC(5, i); });
    fillVector(Part_sticSiliconVertexPos_, sk::LVPART, sk::NVECP, [](int i)
               { return sk::KSTIC(6, i); });

    if (mc_ && sk::IFLSIM > 0)
    {
        fillVector(Part_simIdx_, sk::LVPART, sk::NVECP, [](int i)
                   { return sk::IPAST(i) - 1; });
    }
    fillVector(Part_originVtxIdx_, sk::LVPART, sk::NVECP, [](int i)
                { return sk::IPAPV(1, i) - 1; });
    fillVector(Part_decayVtxIdx_, sk::LVPART, sk::NVECP, [](int i)
                { return sk::IPAPV(2, i) - 1; });
}

// -----------------------------------------------------------------------------
// Photon collection (Tier 1).
//
// Neutral VECP entries with any calorimeter energy — a photon-*candidate* view
// of the existing Part_* collection, indexed separately for downstream
// convenience. No new skelana bindings; this is all derivable from what
// fillPart already reads. Tier 2 (π0 / converted-photon tags from PSCPHO /
// PSCPHC) would extend this once the Fortran-side bindings are added; see
// delphi-nanoaod-photon-plan.md for that follow-up.
// -----------------------------------------------------------------------------
void NanoAODWriter::definePhoton(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nPhoton", "Number of neutral EM clusters (photon candidates)", nPhoton_);
    MakeField(model, "Photon_partIdx", "Index into Part_*", Photon_partIdx_);
    MakeField(model, "Photon_fourMomentum", "Photon 4-momentum", Photon_fourMomentum_);
    MakeField(model, "Photon_lock", "SKELANA LVLOCK quality word", Photon_lock_);
    MakeField(model, "Photon_isHPC", "Fired HPC barrel calorimeter", Photon_isHPC_);
    MakeField(model, "Photon_isSTIC", "Fired STIC small-angle calorimeter", Photon_isSTIC_);
    MakeField(model, "Photon_isFEMC", "Fired FEMC endcap (and not HPC/STIC)", Photon_isFEMC_);
    MakeField(model, "Photon_hpcShowerEnergy", "QHPC(1): energy of most-energetic HPC shower", Photon_hpcShowerEnergy_);
    MakeField(model, "Photon_hpcTotalShowerEnergy", "QHPC(8): total HPC shower energy associated", Photon_hpcTotalShowerEnergy_);
    MakeField(model, "Photon_hpcNumLayers", "KHPC(5): number of HPC layers hit (shower depth)", Photon_hpcNumLayers_);
    MakeField(model, "Photon_hpcLayerHitPattern", "KHPC(6): HPC 9-layer hit pattern", Photon_hpcLayerHitPattern_);
    MakeField(model, "Photon_hpcNumAssociatedShowers", "KHPC(7): number of associated HPC showers", Photon_hpcNumAssociatedShowers_);
    MakeField(model, "Photon_hpcParticleCode", "KHPC(4): HPC particle identification code", Photon_hpcParticleCode_);
    MakeField(model, "Photon_sticShowerEnergy", "QSTIC(1): STIC shower energy", Photon_sticShowerEnergy_);
    MakeField(model, "Photon_sticNumTowers", "KSTIC(4): number of STIC towers hit", Photon_sticNumTowers_);
    MakeField(model, "Photon_sticChargedTag", "KSTIC(5): STIC charged-track veto. Code values (SKELANA v1.05 A.3.3): -2 = tight photon, -1 = loose photon, 0 = no information, +1 = electron", Photon_sticChargedTag_);
    MakeField(model, "Photon_emEnergy", "QEMF(8): total EM energy (FEMC/HPC merged)", Photon_emEnergy_);
    MakeField(model, "Photon_hadEnergy", "QHAC(8): HAC leakage, for EM/HAD cut", Photon_hadEnergy_);

    // Tier 2: per-track photon- and pi0-ID in HPC (only populated where SKELANA filled
    // PSCPHO / PSCPI0; require IFLPHO > 0 in the yaml config).
    MakeField(model, "Photon_hpcEnergyWeightedDepth",      "QPHOT(1): energy-weighted shower depth (PSCPHO)", Photon_hpcEnergyWeightedDepth_);
    MakeField(model, "Photon_hpcPhotNumClusters",          "KPHOT(2): number of HPC clusters in PSCPHO", Photon_hpcPhotNumClusters_);
    MakeField(model, "Photon_hpcPhotFirstLayer",           "KPHOT(3): number of the first HPC layer", Photon_hpcPhotFirstLayer_);
    MakeField(model, "Photon_hpcPhotMaxConsecutiveLayers", "KPHOT(5): max number of consecutive HPC layers", Photon_hpcPhotMaxConsecutiveLayers_);
    MakeField(model, "Photon_hpcTransverseFluctuation",    "QPHOT(6): transverse shower fluctuation measure", Photon_hpcTransverseFluctuation_);
    MakeField(model, "Photon_pi0FittedMass",               "QPI0(12): fitted pi0 mass — key pi0 discriminator", Photon_pi0FittedMass_);
    MakeField(model, "Photon_pi0Chi2",                     "QPI0(22): chi2 of the pi0 fit", Photon_pi0Chi2_);
    MakeField(model, "Photon_pi0GammaChi2",                "QPI0(24): chi2 of the single-gamma fit", Photon_pi0GammaChi2_);
    MakeField(model, "Photon_pi0NumConnectedMaxima",       "KPI0(5): number of connected maxima in HPCANA", Photon_pi0NumConnectedMaxima_);
    MakeField(model, "Photon_pi0NumExpectedMaxima",        "KPI0(6): number of expected maxima in HPCANA", Photon_pi0NumExpectedMaxima_);
    MakeField(model, "Photon_pi0RelE1stGaussian",          "QPI0(7): relative energy of the first Gaussian", Photon_pi0RelE1stGaussian_);
}

void NanoAODWriter::fillPhoton()
{
    Photon_partIdx_->clear();
    Photon_fourMomentum_->clear();
    Photon_lock_->clear();
    Photon_isHPC_->clear();
    Photon_isSTIC_->clear();
    Photon_isFEMC_->clear();
    Photon_hpcShowerEnergy_->clear();
    Photon_hpcTotalShowerEnergy_->clear();
    Photon_hpcNumLayers_->clear();
    Photon_hpcLayerHitPattern_->clear();
    Photon_hpcNumAssociatedShowers_->clear();
    Photon_hpcParticleCode_->clear();
    Photon_sticShowerEnergy_->clear();
    Photon_sticNumTowers_->clear();
    Photon_sticChargedTag_->clear();
    Photon_emEnergy_->clear();
    Photon_hadEnergy_->clear();
    Photon_hpcEnergyWeightedDepth_->clear();
    Photon_hpcPhotNumClusters_->clear();
    Photon_hpcPhotFirstLayer_->clear();
    Photon_hpcPhotMaxConsecutiveLayers_->clear();
    Photon_hpcTransverseFluctuation_->clear();
    Photon_pi0FittedMass_->clear();
    Photon_pi0Chi2_->clear();
    Photon_pi0GammaChi2_->clear();
    Photon_pi0NumConnectedMaxima_->clear();
    Photon_pi0NumExpectedMaxima_->clear();
    Photon_pi0RelE1stGaussian_->clear();

    for (int i = sk::LVPART; i <= sk::NVECP; ++i)
    {
        if (int(sk::VECP(7, i)) != 0) continue;        // charged: skip
        const float hpcTot = sk::QHPC(8, i);
        const float sticE  = sk::QSTIC(1, i);
        const float emfTot = sk::QEMF(8, i);
        const bool isHPC  = hpcTot > 0.0f;
        const bool isSTIC = sticE  > 0.0f;
        const bool isFEMC = emfTot > 0.0f && !isHPC && !isSTIC;
        if (!(isHPC || isSTIC || isFEMC)) continue;    // no EM energy: not a photon candidate

        Photon_partIdx_->push_back(static_cast<int16_t>(i - sk::LVPART));
        Photon_fourMomentum_->push_back(XYZTVectorF(
            sk::VECP(1, i), sk::VECP(2, i), sk::VECP(3, i), sk::VECP(4, i)));
        Photon_lock_->push_back(sk::LVLOCK(i));
        Photon_isHPC_->push_back(isHPC);
        Photon_isSTIC_->push_back(isSTIC);
        Photon_isFEMC_->push_back(isFEMC);
        Photon_hpcShowerEnergy_->push_back(sk::QHPC(1, i));
        Photon_hpcTotalShowerEnergy_->push_back(hpcTot);
        Photon_hpcNumLayers_->push_back(sk::KHPC(5, i));
        Photon_hpcLayerHitPattern_->push_back(sk::KHPC(6, i));
        Photon_hpcNumAssociatedShowers_->push_back(sk::KHPC(7, i));
        Photon_hpcParticleCode_->push_back(sk::KHPC(4, i));
        Photon_sticShowerEnergy_->push_back(sticE);
        Photon_sticNumTowers_->push_back(sk::KSTIC(4, i));
        Photon_sticChargedTag_->push_back(sk::KSTIC(5, i));
        Photon_emEnergy_->push_back(emfTot);
        Photon_hadEnergy_->push_back(sk::QHAC(8, i));
        Photon_hpcEnergyWeightedDepth_->push_back(sk::QPHOT(1, i));
        Photon_hpcPhotNumClusters_->push_back(sk::KPHOT(2, i));
        Photon_hpcPhotFirstLayer_->push_back(sk::KPHOT(3, i));
        Photon_hpcPhotMaxConsecutiveLayers_->push_back(sk::KPHOT(5, i));
        Photon_hpcTransverseFluctuation_->push_back(sk::QPHOT(6, i));
        Photon_pi0FittedMass_->push_back(sk::QPI0(12, i));
        Photon_pi0Chi2_->push_back(sk::QPI0(22, i));
        Photon_pi0GammaChi2_->push_back(sk::QPI0(24, i));
        Photon_pi0NumConnectedMaxima_->push_back(sk::KPI0(5, i));
        Photon_pi0NumExpectedMaxima_->push_back(sk::KPI0(6, i));
        Photon_pi0RelE1stGaussian_->push_back(sk::QPI0(7, i));
    }
    *nPhoton_ = static_cast<int16_t>(Photon_partIdx_->size());
}

// -----------------------------------------------------------------------------
// PhotonConv collection (PSCPHC, one row per reconstructed γ→e+e- conversion).
// Separate from Photon_* because the indexing is per-event, not per-track;
// matching to a Photon_* row would require kinematic association, which is
// intentionally left to downstream analysis.
// -----------------------------------------------------------------------------
void NanoAODWriter::definePhotonConv(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nPhotonConv",                  "Number of γ→e+e- conversions (PSCPHC)", nPhotonConv_);
    MakeField(model, "PhotonConv_firstDaughterIdx",  "KPHOC(1)-1: index of first decay product", PhotonConv_firstDaughterIdx_);
    MakeField(model, "PhotonConv_secondDaughterIdx", "KPHOC(2)-1: index of second decay product", PhotonConv_secondDaughterIdx_);
    MakeField(model, "PhotonConv_simPhotonIdx",      "KPHOC(3)-1: index of simulated photon (MC only)", PhotonConv_simPhotonIdx_);
    MakeField(model, "PhotonConv_pairType",          "KPHOC(4): -21/-22/-23 = 2-TPC I/II/III, -24 = 1-TPC I", PhotonConv_pairType_);
    MakeField(model, "PhotonConv_fourMomentum",      "QPHOC(5..8): photon 4-momentum at conversion", PhotonConv_fourMomentum_);
    MakeField(model, "PhotonConv_trackLength",       "QPHOC(9): photon track length beam spot → conversion", PhotonConv_trackLength_);
    MakeField(model, "PhotonConv_vertex",            "QPHOC(10..12): conversion vertex (cm)", PhotonConv_vertex_);
}

void NanoAODWriter::fillPhotonConv()
{
    PhotonConv_firstDaughterIdx_->clear();
    PhotonConv_secondDaughterIdx_->clear();
    PhotonConv_simPhotonIdx_->clear();
    PhotonConv_pairType_->clear();
    PhotonConv_fourMomentum_->clear();
    PhotonConv_trackLength_->clear();
    PhotonConv_vertex_->clear();

    const int n = sk::NPHOC;
    for (int i = 1; i <= n; ++i)
    {
        PhotonConv_firstDaughterIdx_->push_back(static_cast<int16_t>(sk::KPHOC(1, i) - 1));
        PhotonConv_secondDaughterIdx_->push_back(static_cast<int16_t>(sk::KPHOC(2, i) - 1));
        PhotonConv_simPhotonIdx_->push_back(static_cast<int16_t>(sk::KPHOC(3, i) - 1));
        PhotonConv_pairType_->push_back(static_cast<int8_t>(sk::KPHOC(4, i)));
        PhotonConv_fourMomentum_->push_back(XYZTVectorF(
            sk::QPHOC(5, i), sk::QPHOC(6, i), sk::QPHOC(7, i), sk::QPHOC(8, i)));
        PhotonConv_trackLength_->push_back(sk::QPHOC(9, i));
        PhotonConv_vertex_->push_back(XYZPointF(
            sk::QPHOC(10, i), sk::QPHOC(11, i), sk::QPHOC(12, i)));
    }
    *nPhotonConv_ = static_cast<int16_t>(PhotonConv_firstDaughterIdx_->size());
}

// -----------------------------------------------------------------------------
// Unassociated track-element (UTER) collection — PSCTER/OD + FCA + FCB merged.
// For particle-flow reco: lowest-level "hits-like" objects that SKELANA surfaces.
// Per-cell calorimeter hits are not here (SKELANA aggregates to cluster level);
// that would need a PHDST-level reader bypassing SKELANA.
// -----------------------------------------------------------------------------
void NanoAODWriter::defineUter(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nUter",               "Number of unassociated track elements (OD + FCA + FCB)", nUter_);
    MakeField(model, "Uter_detector",       "0 = OD, 1 = FCA, 2 = FCB", Uter_detector_);
    MakeField(model, "Uter_dataDescriptor", "K..(1): data descriptor + measurement code <TER(04)>", Uter_dataDescriptor_);
    MakeField(model, "Uter_coord1",         "Q..(2): coordinate 1 <TER(10)>", Uter_coord1_);
    MakeField(model, "Uter_coord2",         "Q..(3): coordinate 2 <TER(11)>", Uter_coord2_);
    MakeField(model, "Uter_coord3",         "Q..(4): coordinate 3 <TER(12)>", Uter_coord3_);
    MakeField(model, "Uter_theta",          "Q..(5): theta at reference point", Uter_theta_);
    MakeField(model, "Uter_phi",            "Q..(6): phi at reference point", Uter_phi_);
    MakeField(model, "Uter_invPOrPt",       "Q..(7): 1/P or 1/Pt at reference point", Uter_invPOrPt_);
    MakeField(model, "Uter_elementLength",  "Q..(8): length of the track element", Uter_elementLength_);
}

void NanoAODWriter::fillUter()
{
    Uter_detector_->clear();
    Uter_dataDescriptor_->clear();
    Uter_coord1_->clear();
    Uter_coord2_->clear();
    Uter_coord3_->clear();
    Uter_theta_->clear();
    Uter_phi_->clear();
    Uter_invPOrPt_->clear();
    Uter_elementLength_->clear();

    for (int i = 1; i <= sk::NTEOD; ++i)
    {
        Uter_detector_->push_back(0);
        Uter_dataDescriptor_->push_back(sk::KTEOD(1, i));
        Uter_coord1_->push_back(sk::QTEOD(2, i));
        Uter_coord2_->push_back(sk::QTEOD(3, i));
        Uter_coord3_->push_back(sk::QTEOD(4, i));
        Uter_theta_->push_back(sk::QTEOD(5, i));
        Uter_phi_->push_back(sk::QTEOD(6, i));
        Uter_invPOrPt_->push_back(sk::QTEOD(7, i));
        Uter_elementLength_->push_back(sk::QTEOD(8, i));
    }
    for (int i = 1; i <= sk::NTEFCA; ++i)
    {
        Uter_detector_->push_back(1);
        Uter_dataDescriptor_->push_back(sk::KTEFCA(1, i));
        Uter_coord1_->push_back(sk::QTEFCA(2, i));
        Uter_coord2_->push_back(sk::QTEFCA(3, i));
        Uter_coord3_->push_back(sk::QTEFCA(4, i));
        Uter_theta_->push_back(sk::QTEFCA(5, i));
        Uter_phi_->push_back(sk::QTEFCA(6, i));
        Uter_invPOrPt_->push_back(sk::QTEFCA(7, i));
        Uter_elementLength_->push_back(sk::QTEFCA(8, i));
    }
    for (int i = 1; i <= sk::NTEFCB; ++i)
    {
        Uter_detector_->push_back(2);
        Uter_dataDescriptor_->push_back(sk::KTEFCB(1, i));
        Uter_coord1_->push_back(sk::QTEFCB(2, i));
        Uter_coord2_->push_back(sk::QTEFCB(3, i));
        Uter_coord3_->push_back(sk::QTEFCB(4, i));
        Uter_theta_->push_back(sk::QTEFCB(5, i));
        Uter_phi_->push_back(sk::QTEFCB(6, i));
        Uter_invPOrPt_->push_back(sk::QTEFCB(7, i));
        Uter_elementLength_->push_back(sk::QTEFCB(8, i));
    }

    *nUter_ = static_cast<int16_t>(Uter_detector_->size());
}

// -----------------------------------------------------------------------------
// V0 collection (SKELANA A.1.5 PSCRV0 + A.1.6 PSCHV0).
//
// Guarded by IFLRV0 > 0. Emits one row per reconstructed V0 in V0_*, plus a
// flat V0Hyp_* side table with up to 8 fit hypotheses per V0 (rows linked back
// via V0Hyp_v0Idx). Hypothesis kinds: 1/2 = VTX(+DIR) no mass constraint,
// 11/12 = K0, 21/22 = Lambda, 31/32 = anti-Lambda, -1 = failed.
// -----------------------------------------------------------------------------
void NanoAODWriter::defineV0(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nV0", "Number of reconstructed V0s (PSCRV0)", nV0_);
    MakeField(model, "V0_firstPartIdx",  "Index into Part_* of the + daughter", V0_firstPartIdx_);
    MakeField(model, "V0_secondPartIdx", "Index into Part_* of the - daughter", V0_secondPartIdx_);
    MakeField(model, "V0_incomingIdx",   "Index of incoming particle (-1 if none)", V0_incomingIdx_);
    MakeField(model, "V0_mcIncomingIdx", "Index of MC incoming particle (-1 if none)", V0_mcIncomingIdx_);
    MakeField(model, "V0_tagFlag",       "KRV0(5) bit mask: 0=loose, 1=tight, 2=K0bkg, 3=Lbkg, 22=K0, 33=Lambda", V0_tagFlag_);
    MakeField(model, "V0_momentum",      "QRV0(6): scalar V0 momentum (GeV/c)", V0_momentum_);
    MakeField(model, "V0_vertexChi2Prob","QRV0(7): chi2 probability of PXFVTX fit", V0_vertexChi2Prob_);
    MakeField(model, "V0_vertex",        "QRV0(8..10): V0 vertex position (cm)", V0_vertex_);
    MakeField(model, "V0_xyFlightDistanceOverError", "QRV0(11): xy flight distance normalized to its error", V0_xyFlightDistanceOverError_);
    MakeField(model, "V0_xyFlightAngle", "QRV0(12): angle (rad) in xy w.r.t. primary-vertex line", V0_xyFlightAngle_);
    MakeField(model, "V0_pPlus",         "QRV0(13..15): post-fit + daughter 3-momentum (GeV/c)", V0_pPlus_);
    MakeField(model, "V0_pMinus",        "QRV0(16..18): post-fit - daughter 3-momentum (GeV/c)", V0_pMinus_);
    MakeField(model, "V0_suggestedMass", "QRV0(19): suggested V0 mass (negative for antiparticle)", V0_suggestedMass_);
    MakeField(model, "V0_impactEps",     "QRV0(20): epsilon impact parameter of the neutral track", V0_impactEps_);
    MakeField(model, "V0_impactZ",       "QRV0(21): Z impact parameter of the neutral track", V0_impactZ_);
    MakeField(model, "V0_neutralTheta",  "QRV0(22): theta of the neutral track", V0_neutralTheta_);
    MakeField(model, "V0_neutralPhi",    "QRV0(23): phi of the neutral track", V0_neutralPhi_);

    MakeField(model, "nV0Hyp",               "Number of V0 hypothesis rows (flattened)", nV0Hyp_);
    MakeField(model, "V0Hyp_v0Idx",          "Index into V0_*", V0Hyp_v0Idx_);
    MakeField(model, "V0Hyp_slot",           "Slot 1..8 in PSCHV0 (n in the manual)", V0Hyp_slot_);
    MakeField(model, "V0Hyp_kind",           "KRV0Hn(1): 1,2,11,12,21,22,31,32 or -1=failed", V0Hyp_kind_);
    MakeField(model, "V0Hyp_probability",    "QRV0Hn(2): fit probability", V0Hyp_probability_);
    MakeField(model, "V0Hyp_vertex",         "QRV0Hn(3..5): fitted V0 vertex", V0Hyp_vertex_);
    MakeField(model, "V0Hyp_momentum",       "QRV0Hn(6..8): fitted V0 3-momentum", V0Hyp_momentum_);
    MakeField(model, "V0Hyp_impactXY",       "QRV0Hn(9):  impact xy", V0Hyp_impactXY_);
    MakeField(model, "V0Hyp_impactXYErr",    "QRV0Hn(10): error of impact xy", V0Hyp_impactXYErr_);
    MakeField(model, "V0Hyp_impactZ",        "QRV0Hn(11): impact z", V0Hyp_impactZ_);
    MakeField(model, "V0Hyp_impactZErr",     "QRV0Hn(12): error of impact z", V0Hyp_impactZErr_);
    MakeField(model, "V0Hyp_impact3D",       "QRV0Hn(13): impact 3D", V0Hyp_impact3D_);
    MakeField(model, "V0Hyp_impact3DErr",    "QRV0Hn(14): error of impact 3D", V0Hyp_impact3DErr_);
}

void NanoAODWriter::fillV0()
{
    V0_firstPartIdx_->clear();
    V0_secondPartIdx_->clear();
    V0_incomingIdx_->clear();
    V0_mcIncomingIdx_->clear();
    V0_tagFlag_->clear();
    V0_momentum_->clear();
    V0_vertexChi2Prob_->clear();
    V0_vertex_->clear();
    V0_xyFlightDistanceOverError_->clear();
    V0_xyFlightAngle_->clear();
    V0_pPlus_->clear();
    V0_pMinus_->clear();
    V0_suggestedMass_->clear();
    V0_impactEps_->clear();
    V0_impactZ_->clear();
    V0_neutralTheta_->clear();
    V0_neutralPhi_->clear();

    V0Hyp_v0Idx_->clear();
    V0Hyp_slot_->clear();
    V0Hyp_kind_->clear();
    V0Hyp_probability_->clear();
    V0Hyp_vertex_->clear();
    V0Hyp_momentum_->clear();
    V0Hyp_impactXY_->clear();
    V0Hyp_impactXYErr_->clear();
    V0Hyp_impactZ_->clear();
    V0Hyp_impactZErr_->clear();
    V0Hyp_impact3D_->clear();
    V0Hyp_impact3DErr_->clear();

    const int nV0 = sk::NRV0;
    for (int i = 1; i <= nV0; ++i)
    {
        V0_firstPartIdx_->push_back(static_cast<int16_t>(sk::KRV0(1, i) - 1));
        V0_secondPartIdx_->push_back(static_cast<int16_t>(sk::KRV0(2, i) - 1));
        V0_incomingIdx_->push_back(static_cast<int16_t>(sk::KRV0(3, i) - 1));
        V0_mcIncomingIdx_->push_back(static_cast<int16_t>(sk::KRV0(4, i) - 1));
        V0_tagFlag_->push_back(sk::KRV0(5, i));
        V0_momentum_->push_back(sk::QRV0(6, i));
        V0_vertexChi2Prob_->push_back(sk::QRV0(7, i));
        V0_vertex_->push_back(XYZPointF(sk::QRV0(8, i), sk::QRV0(9, i), sk::QRV0(10, i)));
        V0_xyFlightDistanceOverError_->push_back(sk::QRV0(11, i));
        V0_xyFlightAngle_->push_back(sk::QRV0(12, i));
        V0_pPlus_->push_back(XYZVectorF(sk::QRV0(13, i), sk::QRV0(14, i), sk::QRV0(15, i)));
        V0_pMinus_->push_back(XYZVectorF(sk::QRV0(16, i), sk::QRV0(17, i), sk::QRV0(18, i)));
        V0_suggestedMass_->push_back(sk::QRV0(19, i));
        V0_impactEps_->push_back(sk::QRV0(20, i));
        V0_impactZ_->push_back(sk::QRV0(21, i));
        V0_neutralTheta_->push_back(sk::QRV0(22, i));
        V0_neutralPhi_->push_back(sk::QRV0(23, i));

        // Hypothesis kind codes, indexed by slot 1..8.
        static const int slot_kind_expected[8] = {1, 2, 11, 12, 21, 22, 31, 32};
        for (int n = 1; n <= 8; ++n)
        {
            int kind = sk::KRV0H(n, 1, i);
            // Emit every slot (including failed) so the table has fixed shape
            // per V0; downstream code filters on kind != -1.
            V0Hyp_v0Idx_->push_back(static_cast<int16_t>(i - 1));
            V0Hyp_slot_->push_back(static_cast<int8_t>(n));
            V0Hyp_kind_->push_back(static_cast<int8_t>(kind));
            V0Hyp_probability_->push_back(sk::QRV0H(n, 2, i));
            V0Hyp_vertex_->push_back(XYZPointF(
                sk::QRV0H(n, 3, i), sk::QRV0H(n, 4, i), sk::QRV0H(n, 5, i)));
            V0Hyp_momentum_->push_back(XYZVectorF(
                sk::QRV0H(n, 6, i), sk::QRV0H(n, 7, i), sk::QRV0H(n, 8, i)));
            V0Hyp_impactXY_->push_back(sk::QRV0H(n,  9, i));
            V0Hyp_impactXYErr_->push_back(sk::QRV0H(n, 10, i));
            V0Hyp_impactZ_->push_back(sk::QRV0H(n, 11, i));
            V0Hyp_impactZErr_->push_back(sk::QRV0H(n, 12, i));
            V0Hyp_impact3D_->push_back(sk::QRV0H(n, 13, i));
            V0Hyp_impact3DErr_->push_back(sk::QRV0H(n, 14, i));
            // Sanity: if kind != -1 it should match the slot's expected value.
            (void)slot_kind_expected;
        }
    }
    *nV0_    = static_cast<int16_t>(V0_firstPartIdx_->size());
    *nV0Hyp_ = static_cast<int16_t>(V0Hyp_v0Idx_->size());
}

void NanoAODWriter::defineJet(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nJet", "Number of jets", nJet_);
    MakeField(model, "Jet_fourMomentum", "Jet 4-momentum", Jet_fourMomentum_);
    MakeField(model, "Jet_charge", "Jet charge", Jet_charge_);

    if ((sk::IFLJET / 10) % 10 == 0)
    {
        MakeField(model, "Jet_oblatness", "Jet oblatness", Jet_oblatness_);
        MakeField(model, "Jet_thrust", "Jet thrust", Jet_thrust_);
        MakeField(model, "Jet_thrustVector", "Jet thrust vector", Jet_thrustVector_);
    }
    if ((sk::IFLJET / 100) % 10 == 0)
    {
        MakeField(model, "Jet_sphericity", "Jet sphericity", Jet_sphericity_);
        MakeField(model, "Jet_aplanarity", "Jet aplanarity", Jet_aplanarity_);
        MakeField(model, "Jet_sphericityVector", "Jet sphericity vector", Jet_sphericityVector_);
    }
}

void NanoAODWriter::fillJet()
{
    *nJet_ = sk::NJET;
    fillVector(Jet_fourMomentum_, sk::LVJET + 1, sk::NJET, [](int i)
               { return XYZTVectorF(sk::VECP(1, i), sk::VECP(2, i), sk::VECP(3, i), sk::VECP(4, i)); });
    fillVector(Jet_charge_, sk::LVJET + 1, sk::NJET, [](int i)
               { return int(sk::VECP(6, i)); });

    if ((sk::IFLJET / 10) % 10 == 0)
    {
        *Jet_thrust_ = THR;
        *Jet_oblatness_ = OBL;
        fillVector(Jet_thrustVector_, sk::LVTHRU, 3, [](int i)
                   { return XYZVectorF(sk::VECP(1, i), sk::VECP(2, i), sk::VECP(3, i)); });
    }

    if ((sk::IFLJET / 100) % 10 == 0)
    {
        *Jet_sphericity_ = SPH;
        *Jet_aplanarity_ = APL;
        fillVector(Jet_sphericityVector_, sk::LVSPHE, 3, [](int i)
                   { return XYZVectorF(sk::VECP(1, i), sk::VECP(2, i), sk::VECP(3, i)); });
    }
}

void NanoAODWriter::defineVtx(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nVtx", "Number of vertices", nVtx_);
    MakeField(model, "Vtx_outgoingIdx", "First outgoing particle index", Vtx_outgoingIdx_);
    MakeField(model, "Vtx_incomingIdx", "Incoming particle index", Vtx_incomingIdx_);
    MakeField(model, "Vtx_nOutgoing", "Number of outgoing particles", Vtx_nOutgoing_);
    MakeField(model, "Vtx_ndf", "Number of degrees of freedom of the Vertex Fit", Vtx_ndf_);
    MakeField(model, "Vtx_position", "Vertex position", Vtx_position_);
    MakeField(model, "Vtx_chi2", "Vertex fit chi2", Vtx_chi2_);
    MakeField(model, "Vtx_errorMatrix", "Vertex error matrix", Vtx_errorMatrix_);
    MakeField(model, "Vtx_errorFlag", "Vertex error flag", Vtx_errorFlag_);
    MakeField(model, "Vtx_status", "Vertex status", Vtx_status_);
}

void NanoAODWriter::fillVtx()
{
    *nVtx_ = sk::NVTX;
    fillVector(Vtx_outgoingIdx_, 1, sk::NVTX, [](int i)
               { return sk::KVTX(1, i); });
    fillVector(Vtx_incomingIdx_, 1, sk::NVTX, [](int i)
               { return sk::KVTX(2, i); });
    fillVector(Vtx_nOutgoing_, 1, sk::NVTX, [](int i)
               { return sk::KVTX(3, i); });
    fillVector(Vtx_ndf_, 1, sk::NVTX, [](int i)
               { return sk::KVTX(4, i); });
    fillVector(Vtx_position_, 1, sk::NVTX, [](int i)
               { return XYZPointF(sk::QVTX(6, i), sk::QVTX(7, i), sk::QVTX(8, i)); });
    fillVector(Vtx_chi2_, 1, sk::NVTX, [](int i)
               { return sk::QVTX(9, i); });
    fillVector(Vtx_errorMatrix_, 1, sk::NVTX, [](int i)
               { return ROOT::Math::SMatrixSym3F(&sk::QVTX(10, i), 6); });
    fillVector(Vtx_errorFlag_, 1, sk::NVTX, [](int i)
               { return sk::KVTX(16, i); });
    fillVector(Vtx_status_, 1, sk::NVTX, [](int i)
               { return sk::KVTX(17, i); });
}

void NanoAODWriter::defineSimPart(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nSimPart", "Number of simulated particles", nSimPart_);
    MakeField(model, "SimPart_fourMomentum", "Simulated particle 4-momentum", SimPart_fourMomentum_);
    MakeField(model, "SimPart_charge", "Simulated particle charge", SimPart_charge_);
    MakeField(model, "SimPart_pdgId", "Simulated particle PDG ID", SimPart_pdgId_);
    MakeField(model, "SimPart_partIdx", "Simulated particle index", SimPart_partIdx_);
    MakeField(model, "SimPart_genIdx", "Simulated particle generation index", SimPart_genIdx_);
    MakeField(model, "SimPart_originVtxIdx", "Simulated particle origin vertex index", SimPart_originVtxIdx_);
    MakeField(model, "SimPart_decayVtxIdx", "Simulated particle decay vertex index", SimPart_decayVtxIdx_);
}

void NanoAODWriter::fillSimPart()
{
    *nSimPart_ = sk::NVECMC;
    fillVector(SimPart_fourMomentum_, sk::MTRACK + sk::LVPART, sk::NVECMC, [](int i)
    { return XYZTVectorF(sk::VECP(1, i), sk::VECP(2, i), sk::VECP(3, i), sk::VECP(4, i)); });
    fillVector(SimPart_charge_, sk::MTRACK + sk::LVPART, sk::NVECMC, [](int i)
    { return int(sk::VECP(7, i)); });
    fillVector(SimPart_pdgId_, sk::MTRACK + sk::LVPART, sk::NVECMC, [](int i)
    { return int(sk::VECP(8, i)); });
    fillVector(SimPart_partIdx_, sk::LVPART, sk::NVECMC, [](int i)
    { return sk::ISTPA(i) - 1; });
    fillVector(SimPart_genIdx_, sk::LVPART, sk::NVECMC, [](int i)
    { return sk::ISTSH(i) - 1; });
    fillVector(SimPart_originVtxIdx_, sk::LVPART, sk::NVECMC, [](int i)
    { return sk::ISTVX(1, i) - 1; });
    fillVector(SimPart_decayVtxIdx_, sk::LVPART, sk::NVECMC, [](int i)
    { return sk::ISTVX(2, i) - 1; });
}

void NanoAODWriter::defineGenPart(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nGenPart", "Number of generated particles", nGenPart_);
    MakeField(model, "GenPart_status", "Generated particle status", GenPart_status_);
    MakeField(model, "GenPart_pdgId", "Generated particle PDG ID", GenPart_pdgId_);
    MakeField(model, "GenPart_parentIdx", "Generated particle parent index", GenPart_parentIdx_);
    MakeField(model, "GenPart_firstChildIdx", "Generated particle first child index", GenPart_firstChildIdx_);
    MakeField(model, "GenPart_lastChildIdx", "Generated particle last child index", GenPart_lastChildIdx_);
    MakeField(model, "GenPart_vector", "Generated particle 4-momentum", GenPart_fourMomentum_);
    MakeField(model, "GenPart_vertex", "Generated particle 4-position", GenPart_fourPosition_);
    MakeField(model, "GenPart_tau", "Generated particle lifetime", GenPart_tau_);
    MakeField(model, "GenPart_simIdx", "Generated particle simulation index", GenPart_simIdx_);
    MakeField(model, "GenPart_mass", "Generated particle mass", GenPart_mass_);
}

void NanoAODWriter::fillGenPart()
{
    *nGenPart_ = sk::NP;
    fillVector(GenPart_status_, 1, sk::NP, [](int i)
               { return sk::KP(i, 1); });
    fillVector(GenPart_pdgId_, 1, sk::NP, [](int i)
               { return sk::KP(i, 2); });
    fillVector(GenPart_parentIdx_, 1, sk::NP, [](int i)
               { return sk::KP(i, 3) - 1; });
    fillVector(GenPart_firstChildIdx_, 1, sk::NP, [](int i)
               { return sk::KP(i, 4) - 1; });
    fillVector(GenPart_lastChildIdx_, 1, sk::NP, [](int i)
               { return sk::KP(i, 5) - 1; });
    fillVector(GenPart_fourMomentum_, 1, sk::NP, [](int i)
               { return XYZTVectorF(sk::PP(i, 1), sk::PP(i, 2), sk::PP(i, 3), sk::PP(i, 4)); });
    fillVector(GenPart_fourPosition_, 1, sk::NP, [](int i)
               { return XYZTVectorF(sk::VP(i, 1), sk::VP(i, 2), sk::VP(i, 3), sk::VP(i, 4)); });
    fillVector(GenPart_tau_, 1, sk::NP, [](int i)
               { return sk::VP(i, 5); });
    fillVector(GenPart_simIdx_, 1, sk::NP, [](int i)
               { return sk::ISHST(i) - 1; });

    fillVector(GenPart_mass_, 1, sk::NP, [](int i)
    { return sk::PP(i, 5); });

}

void NanoAODWriter::defineSimVtx(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nSimVtx", "Number of simulated vertices", nSimVtx_);
    MakeField(model, "SimVtx_outgoingIdx", "First outgoing particle index", SimVtx_outgoingIdx_);
    MakeField(model, "SimVtx_incomingIdx", "Incoming particle index", SimVtx_incomingIdx_);
    MakeField(model, "SimVtx_nOut", "Number of outgoing particles", SimVtx_nOutgoing_);
    MakeField(model, "SimVtx_masscode", "Mass code", SimVtx_masscode_);
    MakeField(model, "SimVtx_position", "Position", SimVtx_position_);
}

void NanoAODWriter::fillSimVtx()
{
    *nSimVtx_ = sk::NVTXMC;
    fillVector(SimVtx_outgoingIdx_, sk::NVTXMX + 1, sk::NVTXMC, [](int i)
    { return sk::KVTX(1, i) - 1; });
    fillVector(SimVtx_incomingIdx_, sk::NVTXMX + 1, sk::NVTXMC, [](int i)
    { return sk::KVTX(2, i) - 1; });
    fillVector(SimVtx_nOutgoing_, sk::NVTXMX + 1, sk::NVTXMC, [](int i)
    { return sk::KVTX(3, i); });
    fillVector(SimVtx_masscode_, sk::NVTXMX + 1, sk::NVTXMC, [](int i)
    { return sk::KVTX(4, i); });
    fillVector(SimVtx_position_, sk::NVTXMX + 1, sk::NVTXMC, [](int i)
    { return XYZPointF(sk::QVTX(1, i), sk::QVTX(2, i), sk::QVTX(3, i)); });
}

void NanoAODWriter::defineTrac(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "Trac_partIdx", "Particle track index", Trac_partIdx_);
    MakeField(model, "Trac_originVtxIdx", "Origin vertex index", Trac_originVtxIdx_);
    MakeField(model, "Trac_decayVtxIdx", "Decay vertex index", Trac_decayVtxIdx_);
    MakeField(model, "Trac_perigee", "Perigee parameter", Trac_perigee_);
    MakeField(model, "Trac_weightMatrix", "Track weight matrix", Trac_weightMatrix_);
    MakeField(model, "Trac_length", "Track length", Trac_length_);
    MakeField(model, "Trac_detectors", "Track detectors", Trac_detectors_);
    MakeField(model, "Trac_firstPointR", "First point in R", Trac_firstPointR_);
    MakeField(model, "Trac_firstPointZ", "First point in z", Trac_firstPointZ_);
    MakeField(model, "Trac_chi2NoVD", "Track fit chi2 no vertex detector", Trac_chi2NoVD_);
    MakeField(model, "Trac_chi2VD", "Track fit chi2 vertex detector", Trac_chi2VD_);
    MakeField(model, "Trac_ndfNoVD", "Track fit ndf no vertex detector", Trac_ndfNoVD_);
    MakeField(model, "Trac_ndfVD", "Track fit ndf vertex detector", Trac_ndfVD_);
    MakeField(model, "Trac_vdHitsRPhi", "Hits in vertex detector RPhi", Trac_vdHitsRPhi_);
    MakeField(model, "Trac_vdHitsZ", "Hits in vertex detector Z", Trac_vdHitsZ_);
    MakeField(model, "Trac_resRPhiFirstPoint", "Residual in RPhi to first point", Trac_resRPhiFirstPoint_);
    MakeField(model, "Trac_errorResRPhiFirstPoint", "Error on residual RPhi to first point", Trac_errorResRPhiFirstPoint_);
    MakeField(model, "Trac_resZFirstPoint", "Residual in Z to first point", Trac_resZFirstPoint_);
    MakeField(model, "Trac_errorResZFirstPoint", "Error on residual Z to first point", Trac_errorResZFirstPoint_);
    MakeField(model, "Trac_impParRPhi", "QTRAC(4): impact parameter in RPhi at perigee", Trac_impParRPhi_);
    MakeField(model, "Trac_impParZ", "QTRAC(5): impact parameter in Z at perigee", Trac_impParZ_);
    MakeField(model, "Trac_impParToVertexRPhi", "Impact parameter to vertex in RPhi (geometric sign)", Trac_impParToVertexRPhi_);
    MakeField(model, "Trac_impParToVertexZ", "Impact parameter to vertex in Z (geometric sign)", Trac_impParToVertexZ_);
    MakeField(model, "Trac_impParToBeamSpotRPhi", "Impact parameter to beam spot in RPhi (geometric sign)", Trac_impParToBeamSpotRPhi_);
    MakeField(model, "Trac_chi2VDHits", "Chi2 vertex detector", Trac_chi2VDHits_);
}

void NanoAODWriter::fillTrac()
{
    Trac_partIdx_->clear();
    Trac_originVtxIdx_->clear();
    Trac_decayVtxIdx_->clear();
    Trac_perigee_->clear();
    Trac_weightMatrix_->clear();
    Trac_length_->clear();
    Trac_detectors_->clear();
    Trac_firstPointR_->clear();
    Trac_firstPointZ_->clear();
    Trac_chi2NoVD_->clear();
    Trac_chi2VD_->clear();
    Trac_ndfNoVD_->clear();
    Trac_ndfVD_->clear();
    Trac_vdHitsRPhi_->clear();
    Trac_vdHitsZ_->clear();
    Trac_resRPhiFirstPoint_->clear();
    Trac_errorResRPhiFirstPoint_->clear();
    Trac_resZFirstPoint_->clear();
    Trac_errorResZFirstPoint_->clear();
    Trac_impParRPhi_->clear();
    Trac_impParZ_->clear();
    Trac_impParToVertexZ_->clear();
    Trac_impParToVertexRPhi_->clear();
    Trac_impParToBeamSpotRPhi_->clear();
    Trac_chi2VDHits_->clear();

    for (int i = sk::LVPART; i <= sk::NVECP; i++)
    {
        Trac_partIdx_->push_back(i - 1);
        Trac_originVtxIdx_->push_back(sk::KTRAC(1, i) - 1);
        Trac_decayVtxIdx_->push_back(sk::KTRAC(2, i) - 1);
        Trac_perigee_->push_back(SVector5F(&sk::QTRAC(3, i), 5));
        Trac_weightMatrix_->push_back(SMatrixSym5F(&sk::QTRAC(8, i), 15));
        Trac_length_->push_back(sk::QTRAC(24, i));
        Trac_detectors_->push_back(sk::KTRAC(25, i));
        Trac_firstPointR_->push_back(sk::QTRAC(26, i));
        Trac_firstPointZ_->push_back(sk::QTRAC(27, i));
        Trac_chi2NoVD_->push_back(sk::QTRAC(28, i));
        Trac_chi2VD_->push_back(sk::QTRAC(29, i));
        Trac_ndfNoVD_->push_back(sk::KTRAC(30, i));
        Trac_ndfVD_->push_back(sk::KTRAC(31, i));
        Trac_vdHitsRPhi_->push_back(sk::KTRAC(32, i));
        Trac_vdHitsZ_->push_back(sk::KTRAC(33, i));
        Trac_resRPhiFirstPoint_->push_back(sk::QTRAC(34, i));
        Trac_errorResRPhiFirstPoint_->push_back(sk::QTRAC(35, i));
        Trac_resZFirstPoint_->push_back(sk::QTRAC(36, i));
        Trac_errorResZFirstPoint_->push_back(sk::QTRAC(37, i));
        Trac_impParRPhi_->push_back(sk::QTRAC(4, i));
        Trac_impParZ_->push_back(sk::QTRAC(5, i));
        Trac_impParToVertexRPhi_->push_back(sk::QTRAC(38, i));
        Trac_impParToVertexZ_->push_back(sk::QTRAC(39, i));
        Trac_impParToBeamSpotRPhi_->push_back(sk::QTRAC(40, i));
        Trac_chi2VDHits_->push_back(sk::QTRAC(42, i));
    }
}

void NanoAODWriter::defineMuid(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "Muid_partIdx", "Muon ID link to track ID", Muid_partIdx_);
    MakeField(model, "Muid_tag", "Muon ID tag", Muid_tag_);
    MakeField(model, "Muid_looseChi2", "Muon ID loose chi2", Muid_looseChi2_);
    MakeField(model, "Muid_hitPattern", "Muon ID hit pattern", Muid_hitPattern_);
}

void NanoAODWriter::fillMuid()
{
    Muid_partIdx_->clear();
    Muid_tag_->clear();
    Muid_looseChi2_->clear();
    Muid_hitPattern_->clear();

    for (int i = sk::LVPART; i <= sk::NVECP; i++)
    {
        Muid_partIdx_->push_back(i - 1);
        Muid_tag_->push_back(sk::KMUID(1, i));
        Muid_looseChi2_->push_back(sk::QMUID(2, i));
        Muid_hitPattern_->push_back(sk::KMUID(3, i));
    }
}

void NanoAODWriter::defineElid(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "Elid_partIdx", "Electron ID link to track ID", Elid_partIdx_);
    MakeField(model, "Elid_tag", "Electron ID tag", Elid_tag_);
    MakeField(model, "Elid_gammaConversion", "Electron ID gamma conversion", Elid_gammaConversion_);
    MakeField(model, "Elid_px", "Best electron px estimation", Elid_px_);
    MakeField(model, "Elid_py", "Best electron py estimation", Elid_py_);
    MakeField(model, "Elid_pz", "Best electron pz estimation", Elid_pz_);
}

void NanoAODWriter::fillElid()
{
    Elid_partIdx_->clear();
    Elid_tag_->clear();
    Elid_gammaConversion_->clear();
    Elid_px_->clear();
    Elid_py_->clear();
    Elid_pz_->clear();
    for (int i = sk::LVPART; i <= sk::NVECP; i++)
    {
        Elid_partIdx_->push_back(i - 1);
        Elid_tag_->push_back(sk::KELID(1, i));
        Elid_gammaConversion_->push_back(sk::KELID(2, i));
        Elid_px_->push_back(sk::QELID(3, i));
        Elid_py_->push_back(sk::QELID(4, i));
        Elid_pz_->push_back(sk::QELID(5, i));
    }
}

void NanoAODWriter::defineBtag(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "Btag_probNegIP", "B-tagging probability for tracks with negative impact parameter, hemisphere 1", Btag_probNegIP_);
    MakeField(model, "Btag_probPosIP", "B-tagging probability for tracks with negative impact parameter, hemisphere 2", Btag_probPosIP_);
    MakeField(model, "Btag_probAllIP", "B-tagging probability for tracks with negative impact parameter, whole event", Btag_probAllIP_);
    MakeField(model, "Btag_thrustVector", "B-tagging probability for tracks with positive impact parameter, hemisphere 1", Btag_thrustVector_);
}

void NanoAODWriter::fillBtag()
{
    fillVector(Btag_probNegIP_, 1, 3, [](int i)
               { return sk::QBTPRN(i); });
    fillVector(Btag_probPosIP_, 1, 3, [](int i)
               { return sk::QBTPRP(i); });
    fillVector(Btag_probAllIP_, 1, 3, [](int i)
               { return sk::QBTPRS(i); });

    *Btag_thrustVector_ = XYZVectorF(sk::QBTTHR(1), sk::QBTTHR(2), sk::QBTTHR(3));
}

void NanoAODWriter::defineHadid(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "Haid_sign", "Used for combined tag", Haid_sign_);
    MakeField(model, "Haid_kaonDedx", "Kaon signature with DEDX", Haid_kaonDedx_);
    MakeField(model, "Haid_protonDedx", "Proton signature with DEDX", Haid_protonDedx_);
    MakeField(model, "Haid_kaonRich", "Kaon signature with RICH", Haid_kaonRich_);
    MakeField(model, "Haid_protonRich", "Proton signature with RICH", Haid_protonRich_);
    MakeField(model, "Haid_pionRich", "Pion signature with RICH", Haid_pionRich_);
    MakeField(model, "Haid_kaonCombined", "Kaon signature with combined tag", Haid_kaonCombined_);
    MakeField(model, "Haid_protonCombined", "Proton signature with combined tag", Haid_protonCombined_);
    MakeField(model, "Haid_richQuality", "RICH quality status", Haid_richQuality_);

    MakeField(model, "Haidn_pionTag", "Pion tag", Haidn_pionTag_);
    MakeField(model, "Haidn_kaonTag", "Kaon tag", Haidn_kaonTag_);
    MakeField(model, "Haidn_protonTag", "Proton tag", Haidn_protonTag_);
    MakeField(model, "Haidn_heavyTag", "Heavy particle tag", Haidn_heavyTag_);
    MakeField(model, "Haidn_pionTrackSelection", "Pion track selection", Haidn_pionTrackSelection_);
    MakeField(model, "Haidn_kaonTrackSelection", "Kaon track selection", Haidn_kaonTrackSelection_);
    MakeField(model, "Haidn_protonTrackSelection", "Proton track selection", Haidn_protonTrackSelection_);
    MakeField(model, "Haidn_heavyTrackSelection", "Heavy particle track selection", Haidn_heavyTrackSelection_);

    MakeField(model, "Haidr_pionTag", "Pion tag", Haidr_pionTag_);
    MakeField(model, "Haidr_kaonTag", "Kaon tag", Haidr_kaonTag_);
    MakeField(model, "Haidr_protonTag", "Proton tag", Haidr_protonTag_);
    MakeField(model, "Haidr_heavyTag", "Heavy particle tag", Haidr_heavyTag_);
    MakeField(model, "Haidr_electronTag", "Electron tag", Haidr_electronTag_);
    MakeField(model, "Haidr_selectionFlag", "Selection flag", Haidr_selectionFlag_);

    MakeField(model, "Haide_pionTag", "Pion tag", Haide_pionTag_);
    MakeField(model, "Haide_kaonTag", "Kaon tag", Haide_kaonTag_);
    MakeField(model, "Haide_protonTag", "Proton tag", Haide_protonTag_);
    MakeField(model, "Haide_heavyTag", "Heavy particle tag", Haide_heavyTag_);
    MakeField(model, "Haide_electronTag", "Electron tag", Haide_electronTag_);
    MakeField(model, "Haide_selectionFlag", "Selection flag", Haide_selectionFlag_);

    MakeField(model, "Haidc_pionTag", "Pion tag", Haidc_pionTag_);
    MakeField(model, "Haidc_kaonTag", "Kaon tag", Haidc_kaonTag_);
    MakeField(model, "Haidc_protonTag", "Proton tag", Haidc_protonTag_);
    MakeField(model, "Haidc_heavyTag", "Heavy particle tag", Haidc_heavyTag_);
    MakeField(model, "Haidc_electronTag", "Electron tag", Haidc_electronTag_);
    MakeField(model, "Haidc_selectionFlag", "Selection flag", Haidc_selectionFlag_);

    MakeField(model, "Dedx_value", "Dedx value (1 for mips)", Dedx_value_);
    MakeField(model, "Dedx_width", "Dedx Landau width", Dedx_width_);
    MakeField(model, "Dedx_nrWires", "Dedx number of wires", Dedx_nrWires_);
    MakeField(model, "Dedx_gapWires", "Mean distance between wires", Dedx_gapWires_);
    MakeField(model, "Dedx_error", "Dedx value error", Dedx_error_);
    MakeField(model, "Dedx_valueVD", "Dedx VD value", Dedx_valueVD_);
    MakeField(model, "Dedx_nrVDHits", "Number of VD hits", Dedx_nrVDHits_);

    MakeField(model, "Rich_theg", "Cherenkov angle gas radiator", Rich_theg_);
    MakeField(model, "Rich_sigg", "Sigma of Cherenkov angle gas radiator", Rich_sigg_);
    MakeField(model, "Rich_nphg", "Observed number of photons gas radiator", Rich_nphg_);
    MakeField(model, "Rich_nepg", "Expected number of photons gas radiator", Rich_nepg_);
    MakeField(model, "Rich_flagg", "Flag gas radiator", Rich_flagg_);
    MakeField(model, "Rich_thel", "Cherenkov angle liquid radiator", Rich_thel_);
    MakeField(model, "Rich_sigl", "Sigma of Cherenkov angle liquid radiator", Rich_sigl_);
    MakeField(model, "Rich_nphl", "Observed number of photons liquid radiator", Rich_nphl_);
    MakeField(model, "Rich_nepl", "Expected number of photons liquid radiator", Rich_nepl_);
    MakeField(model, "Rich_flagl", "Flag liquid radiator", Rich_flagl_);
}

void NanoAODWriter::fillHadid()
{
    Haid_sign_->clear();
    Haid_kaonDedx_->clear();
    Haid_protonDedx_->clear();
    Haid_kaonRich_->clear();
    Haid_protonRich_->clear();
    Haid_pionRich_->clear();
    Haid_kaonCombined_->clear();
    Haid_protonCombined_->clear();
    Haid_richQuality_->clear();

    Haidn_pionTag_->clear();
    Haidn_kaonTag_->clear();
    Haidn_protonTag_->clear();
    Haidn_heavyTag_->clear();
    Haidn_pionTrackSelection_->clear();
    Haidn_kaonTrackSelection_->clear();
    Haidn_protonTrackSelection_->clear();
    Haidn_heavyTrackSelection_->clear();

    Haidr_pionTag_->clear();
    Haidr_kaonTag_->clear();
    Haidr_protonTag_->clear();
    Haidr_heavyTag_->clear();
    Haidr_electronTag_->clear();
    Haidr_selectionFlag_->clear();

    Haide_pionTag_->clear();
    Haide_kaonTag_->clear();
    Haide_protonTag_->clear();
    Haide_heavyTag_->clear();
    Haide_electronTag_->clear();
    Haide_selectionFlag_->clear();

    Haidc_pionTag_->clear();
    Haidc_kaonTag_->clear();
    Haidc_protonTag_->clear();
    Haidc_heavyTag_->clear();
    Haidc_electronTag_->clear();
    Haidc_selectionFlag_->clear();

    Dedx_value_->clear();
    Dedx_width_->clear();
    Dedx_nrWires_->clear();
    Dedx_gapWires_->clear();
    Dedx_error_->clear();
    Dedx_valueVD_->clear();
    Dedx_nrVDHits_->clear();

    Rich_theg_->clear();
    Rich_sigg_->clear();
    Rich_nphg_->clear();
    Rich_nepg_->clear();
    Rich_flagg_->clear();
    Rich_thel_->clear();
    Rich_sigl_->clear();
    Rich_nphl_->clear();
    Rich_nepl_->clear();
    Rich_flagl_->clear();

    for (int i = sk::LVPART; i <= sk::NVECP; ++i)
    {
        Haid_sign_->push_back(sk::KHAID(1, i));
        Haid_kaonDedx_->push_back(sk::KHAID(2, i));
        Haid_protonDedx_->push_back(sk::KHAID(3, i));
        Haid_kaonRich_->push_back(sk::KHAID(4, i));
        Haid_protonRich_->push_back(sk::KHAID(5, i));
        Haid_pionRich_->push_back(sk::KHAID(6, i));
        Haid_kaonCombined_->push_back(sk::QHAID(7, i));
        Haid_protonCombined_->push_back(sk::QHAID(8, i));
        Haid_richQuality_->push_back(sk::KHAID(9, i));

        Haidn_pionTag_->push_back(sk::KHAIDN(1, i));
        Haidn_kaonTag_->push_back(sk::KHAIDN(2, i));
        Haidn_protonTag_->push_back(sk::KHAIDN(3, i));
        Haidn_heavyTag_->push_back(sk::KHAIDN(4, i));
        Haidn_pionTrackSelection_->push_back(sk::KHAIDT(1, i));
        Haidn_kaonTrackSelection_->push_back(sk::KHAIDT(2, i));
        Haidn_protonTrackSelection_->push_back(sk::KHAIDT(3, i));
        Haidn_heavyTrackSelection_->push_back(sk::KHAIDT(4, i));

        Haidr_pionTag_->push_back(sk::KHAIDR(1, i));
        Haidr_kaonTag_->push_back(sk::KHAIDR(2, i));
        Haidr_protonTag_->push_back(sk::KHAIDR(3, i));
        Haidr_heavyTag_->push_back(sk::KHAIDR(4, i));
        Haidr_electronTag_->push_back(sk::KHAIDR(5, i));
        Haidr_selectionFlag_->push_back(sk::KHAIDR(6, i));

        Haide_pionTag_->push_back(sk::KHAIDE(1, i));
        Haide_kaonTag_->push_back(sk::KHAIDE(2, i));
        Haide_protonTag_->push_back(sk::KHAIDE(3, i));
        Haide_heavyTag_->push_back(sk::KHAIDE(4, i));
        Haide_electronTag_->push_back(sk::KHAIDE(5, i));
        Haide_selectionFlag_->push_back(sk::KHAIDE(6, i));

        Haidc_pionTag_->push_back(sk::KHAIDC(1, i));
        Haidc_kaonTag_->push_back(sk::KHAIDC(2, i));
        Haidc_protonTag_->push_back(sk::KHAIDC(3, i));
        Haidc_heavyTag_->push_back(sk::KHAIDC(4, i));
        Haidc_electronTag_->push_back(sk::KHAIDC(5, i));
        Haidc_selectionFlag_->push_back(sk::KHAIDC(6, i));

        Dedx_value_->push_back(sk::QDEDX(1, i));
        Dedx_width_->push_back(sk::QDEDX(2, i));
        Dedx_nrWires_->push_back(sk::KDEDX(3, i));
        Dedx_gapWires_->push_back(sk::QDEDX(4, i));
        Dedx_error_->push_back(sk::QDEDX(5, i));
        Dedx_valueVD_->push_back(sk::QDEDX(6, i));
        Dedx_nrVDHits_->push_back(sk::KDEDX(7, i));

        Rich_theg_->push_back(sk::THEG(i));
        Rich_sigg_->push_back(sk::SIGG(i));
        Rich_nphg_->push_back(sk::NPHG(i));
        Rich_nepg_->push_back(sk::NEPG(i));
        Rich_flagg_->push_back(sk::FLAGG(i));
        Rich_thel_->push_back(sk::THEL(i));
        Rich_sigl_->push_back(sk::SIGL(i));
        Rich_nphl_->push_back(sk::NPHL(i));
        Rich_nepl_->push_back(sk::NEPL(i));
        Rich_flagl_->push_back(sk::FLAGL(i));
    }
}

void NanoAODWriter::defineVdHit(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nVdHit", "Number of VD associated hits", nVdHit_);
    MakeField(model, "VdHit_trackIdx", "Track index for this VD hit", VdHit_trackIdx_);
    MakeField(model, "VdHit_module", "KVDAS(1): Module number (sign = Z side)", VdHit_module_);
    MakeField(model, "VdHit_localX", "QVDAS(2): Local X (or Z since 94) coordinate", VdHit_localX_);
    MakeField(model, "VdHit_R", "QVDAS(3): R coordinate (-R if R-Z measured)", VdHit_R_);
    MakeField(model, "VdHit_RPhi", "QVDAS(4): RPhi (or Z since 94) coordinate", VdHit_RPhi_);
    MakeField(model, "VdHit_signalToNoise", "QVDAS(5): Signal to noise ratio", VdHit_signalToNoise_);
}
 
void NanoAODWriter::fillVdHit()
{
    VdHit_trackIdx_->clear();
    VdHit_module_->clear();
    VdHit_localX_->clear();
    VdHit_R_->clear();
    VdHit_RPhi_->clear();
    VdHit_signalToNoise_->clear();
 
    int totalHits = 0;
    for (int i = sk::LVPART; i <= sk::NVECP; i++)
    {
        int nHits = sk::NASHT(i);
        for (int n = 1; n <= nHits; n++)
        {
            VdHit_trackIdx_->push_back(i - 1);
            VdHit_module_->push_back(sk::KVDAS(1, i, n));
            VdHit_localX_->push_back(sk::QVDAS(2, i, n));
            VdHit_R_->push_back(sk::QVDAS(3, i, n));
            VdHit_RPhi_->push_back(sk::QVDAS(4, i, n));
            VdHit_signalToNoise_->push_back(sk::QVDAS(5, i, n));
            totalHits++;
        }
    }
    *nVdHit_ = totalHits;
}

void NanoAODWriter::defineVdUnHit(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "nVdUnHit", "Number of VD unassociated hits", nVdUnHit_);
    MakeField(model, "VdUnHit_module", "KVDUN(1): Module number (sign = Z side)", VdUnHit_module_);
    MakeField(model, "VdUnHit_localX", "QVDUN(2): Local X (or Z since 94) coordinate", VdUnHit_localX_);
    MakeField(model, "VdUnHit_R", "QVDUN(3): R coordinate (-R if R-Z measured)", VdUnHit_R_);
    MakeField(model, "VdUnHit_RPhi", "QVDUN(4): RPhi (or Z since 94) coordinate", VdUnHit_RPhi_);
    MakeField(model, "VdUnHit_signalToNoise", "QVDUN(5): Signal to noise ratio", VdUnHit_signalToNoise_);
}
 
void NanoAODWriter::fillVdUnHit()
{
    VdUnHit_module_->clear();
    VdUnHit_localX_->clear();
    VdUnHit_R_->clear();
    VdUnHit_RPhi_->clear();
    VdUnHit_signalToNoise_->clear();
 
    *nVdUnHit_ = sk::NVDUN;
    for (int i = 1; i <= sk::NVDUN; i++)
    {
        VdUnHit_module_->push_back(sk::KVDUN(1, i));
        VdUnHit_localX_->push_back(sk::QVDUN(2, i));
        VdUnHit_R_->push_back(sk::QVDUN(3, i));
        VdUnHit_RPhi_->push_back(sk::QVDUN(4, i));
        VdUnHit_signalToNoise_->push_back(sk::QVDUN(5, i));
    }
}

void NanoAODWriter::defineBeamSpot(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "BeamSpot_errorFlag", "Beam spot error flag (0=OK)", BeamSpot_errorFlag_);
    MakeField(model, "BeamSpot_x", "Beam spot x position", BeamSpot_x_);
    MakeField(model, "BeamSpot_y", "Beam spot y position", BeamSpot_y_);
    MakeField(model, "BeamSpot_z", "Beam spot z position", BeamSpot_z_);
    MakeField(model, "BeamSpot_sigmaX", "Beam spot sigma x", BeamSpot_sigmaX_);
    MakeField(model, "BeamSpot_sigmaY", "Beam spot sigma y", BeamSpot_sigmaY_);
    MakeField(model, "BeamSpot_sigmaZ", "Beam spot sigma z", BeamSpot_sigmaZ_);
}

void NanoAODWriter::fillBeamSpot()
{
    *BeamSpot_errorFlag_ = sk::IERRBS;
    *BeamSpot_x_ = sk::XYZBS(1);
    *BeamSpot_y_ = sk::XYZBS(2);
    *BeamSpot_z_ = sk::XYZBS(3);
    *BeamSpot_sigmaX_ = sk::DXYZBS(1);
    *BeamSpot_sigmaY_ = sk::DXYZBS(2);
    *BeamSpot_sigmaZ_ = sk::DXYZBS(3);
}

void NanoAODWriter::fillPartLoop(particleData& pData,
                                 eventData& eData,
                                 DataKind cat) {
    int nParticle = 0;
    int nParticleHP = 0;
    int nChargedParticle = 0;
    int nChargedParticleHP = 0;
    TVector3 netP(0, 0, 0);
    TVector3 netChargedP(0, 0, 0);
    TVector3 netPGen(0, 0, 0);
    TVector3 netChargedPGen(0, 0, 0);
    std::vector<int> llp;
    int nSize = (cat == DataKind::data) ? *nPart_
        : (cat == DataKind::gen) ? *nGenPart_
        : (cat == DataKind::sim) ? *nSimPart_
        : 0;
    for (auto iSize = 0; iSize < nSize; ++iSize) {
        bool pass = false;
        // calculate charge
        float q = 0;
        if (cat == DataKind::gen) {
            int status = sk::KP(iSize + 1, 1);
            pass = (status == 1 || status == 4); // final state particle, decayed or not decayed
        } else if (cat == DataKind::sim) {
            int status = sk::KP(sk::ISTSH(iSize+1), 1);
            int particleID = sk::KP(sk::ISTSH(iSize+1), 2);
            int mom = sk::ISTVX(1, iSize+1);
            int dau = sk::ISTVX(2, iSize+1);
            if (status == 1 || (status == 4 && dau == 0)) {
                pass = 1;
            } else if (status == 4) {
                llp.push_back(dau);
                pass = 0;
            } else if (particleID == 0 && std::find(llp.begin(), llp.end(), mom) != llp.end()) {
                pass = 1;
            } else {
                pass = 0;
            }
        } else {
            pass = !Part_lock_->at(iSize); // bad particle
        }
        if (pass) {
            // FORTRAN index starts with 1
            int i = iSize + 1;
            XYZTVectorF temp;
            if (cat == DataKind::gen) {
                temp = GenPart_fourMomentum_->at(iSize);
                int pdgid = sk::KP(i, 2);
                TParticlePDG* particle = pdgDatabase->GetParticle(pdgid);
                if (particle) {
                    q = particle->Charge() / 3.0;
                } else {
                    particle = pdgDatabase->GetParticle(-pdgid);
                    q = -particle->Charge() / 3.0;
                }
                pData.pid[nParticle] = pdgid;

                // for gen, use this var to store Lund status
                pData.pwflag[nParticle] = (sk::KP(i, 1) == 1) ? 1 : 4; 
                pData.highPurity[nParticle]= 1;
                pData.index[nParticle] = i;
                pData.correspondenceIndex[nParticle] = sk::ILUST(i);
                // No track quality info for gen/sim
                chi2ndf[nParticle]   = -1.f;
                chi2ndfVD[nParticle] = -1.f;
                dpp[nParticle]       = -1.f;

                nParticleHP++;
                if (abs(q) > 0.5) {
                    nChargedParticle++;
                    nChargedParticleHP++;
                }
            } else if (cat == DataKind::sim) {
                temp = SimPart_fourMomentum_->at(iSize);
                q = sk::VECP(7, sk::MTRACK+i);
                pData.pid[nParticle] = sk::KP(sk::ISTSH(iSize+1), 2);
                // store mass code. -- 2 for electron (only thing useful as of June 11 2025)
                pData.pwflag[nParticle] = sk::VECP(8, sk::MTRACK+i);
                pData.highPurity[nParticle]= 1;
                pData.index[nParticle] = i;
                pData.correspondenceIndex[nParticle] = sk::ISTPA(i);

                // No track quality info for gen/sim
                chi2ndf[nParticle]   = -1.f;
                chi2ndfVD[nParticle] = -1.f;
                dpp[nParticle]       = -1.f;
                
                nParticleHP++;
                nChargedParticle++;
                nChargedParticleHP++;
            } else {
                q = Part_charge_->at(iSize);
                temp = Part_fourMomentum_->at(iSize);
                lock[nParticle] = Part_lock_->at(iSize);
                // LEPTON ID with standard MUID and ELID
                if (Part_charge_->at(iSize) == 0) {
                    pData.pwflag[nParticle] = 4;
                    // standard muon selection (3rd bit from the right)
                } else if (sk::KMUID(1, i) & (1 << 2)) {
                    pData.pwflag[nParticle] = 1;
                    // standard electron selection
                } else if (sk::KELID(1, i) >= 4) {
                    pData.pwflag[nParticle] = 2;
                    // loose conversion ele tag
                } else if (sk::KELID(2, i) >= 1) {
                    pData.pwflag[nParticle] = 3;
                } else {
                    pData.pwflag[nParticle] = 0;
                }
                // TODO: use vdHits?
                pData.ntpc[nParticle] = (pData.charge[nParticle]!=0)? 7: 0;
                pData.d0[nParticle] = sk::QTRAC(38, i);
                pData.z0[nParticle] = sk::QTRAC(39, i);
                // use this variable to store track length for DELPHI
                pData.weight[nParticle] = sk::QTRAC(24, i);

                // --- Track quality variables ---
                // chi2/ndf (no VD)
                int ndf_noVD = sk::KTRAC(30, i);
                float chi2_noVD = sk::QTRAC(28, i);
                chi2ndf[nParticle] = (ndf_noVD > 0) ? chi2_noVD / ndf_noVD : -1.f;

                // chi2/ndf (with VD)
                int ndf_VD = sk::KTRAC(31, i);
                float chi2_VD = sk::QTRAC(29, i);
                chi2ndfVD[nParticle] = (ndf_VD > 0) ? chi2_VD / ndf_VD : -1.f;

                // delta p / p from the weight matrix
                // Perigee order: (d0, z0, theta, phi, 1/R)
                // Weight matrix QTRAC(8..22) is the UPPER triangle of 5x5 symmetric
                // stored as: W11, W12, W13, W14, W15,
                //                  W22, W23, W24, W25,
                //                       W33, W34, W35,
                //                            W44, W45,
                //                                 W55
                // Element W55 = sigma^{-2}(1/R) is at index 22 (offset 8 + 14)
                // 1/R is related to pT by: pT = 0.3 * B * R  (B in Tesla, R in meters)
                // so sigma(pT)/pT = sigma(1/R) / (1/R) = sigma(p)/p for a helix
                //
                // W55 = 1/sigma^2(1/R), and 1/R = QTRAC(7,i) (the 5th perigee param)
                // But the perigee vector starts at QTRAC(3,i):
                //   QTRAC(3) = d0, QTRAC(4) = z0 (at perigee, not vertex!),
                //   QTRAC(5) = theta, QTRAC(6) = phi, QTRAC(7) = 1/R
                float invR = sk::QTRAC(7, i);  // 1/R (curvature)
                float W55  = sk::QTRAC(22, i); // weight matrix element for 1/R

                if (std::abs(invR) > 1e-10f && W55 > 0.f) {
                    float sigma_invR = 1.f / std::sqrt(W55);
                    dpp[nParticle] = sigma_invR / std::abs(invR);
                } else {
                    dpp[nParticle] = -1.f;  // undefined (neutral or degenerate)
                }
                
                // below we follow the same definition in eventSelection.h
                // TODO: Use DELPHI selections
                if (pData.pwflag[nParticle]<=2) {
                    pData.highPurity[nParticle]= pData.pwflag[nParticle]<=2 && temp.Pt() >= 0.2;
                } else if (pData.pwflag[nParticle]==4) {
                    pData.highPurity[nParticle]= pData.pwflag[nParticle]==4 && temp.Pt() >= 0.4;
                }
                
                pData.index[nParticle] = i;
                pData.correspondenceIndex[nParticle] = sk::IPAST(i);
                
                if(pData.pwflag[nParticle]<=2) {
                    nChargedParticle++;
                    if (pData.highPurity[nParticle]) nChargedParticleHP++;
                }
                if (pData.highPurity[nParticle]) {
                    nParticleHP++;
                }
            }
            
            netP -= TVector3(temp.x(), temp.y(), temp.z());
            if (abs(q) > 0.5) {
                netChargedP -= TVector3(temp.x(), temp.y(), temp.z());
            }

            pData.px[nParticle] = temp.x();
            pData.py[nParticle] = temp.y();
            pData.pz[nParticle] = temp.z();
            emf[nParticle] = sk::QEMF(8,i);
            hpc[nParticle] = sk::QHPC(8,i);
            hac[nParticle] = sk::QHAC(8,i);
            stic[nParticle] = sk::QSTIC(1,i);
            pData.pt[nParticle] = temp.Pt();
            pData.pmag[nParticle]   = temp.P();
            pData.rap[nParticle]    = temp.Rapidity();
            pData.eta[nParticle]    = temp.Eta();
            pData.theta[nParticle]  = temp.Theta();
            pData.phi[nParticle]    = temp.Phi();
            pData.mass[nParticle]   = temp.M();
            pData.charge[nParticle] = q;
            
            ++nParticle;
        }
    }
    pData.nParticle         = nParticle;
    eData.nChargedParticle  = nChargedParticle;
    eData.nParticleHP       = nParticleHP;
    eData.nChargedParticleHP= nChargedParticleHP;

    eData.missP = netP.Mag();
    eData.missPt = netP.Perp();
    eData.missTheta = netP.Theta();
    eData.missPhi = netP.Phi();

    eData.missChargedP = netChargedP.Mag();
    eData.missChargedPt = netChargedP.Perp();
    eData.missChargedTheta = netChargedP.Theta();
    eData.missChargedPhi = netChargedP.Phi();
}

void NanoAODWriter::fillSelection(particleData& pData,
                                  eventData& eData) {
    TVector3 thrust             = getThrust(pData.nParticle, pData.px, pData.py, pData.pz, THRUST::OPTIMAL);
    TVector3 thrustWithMissP    = getThrust(pData.nParticle, pData.px, pData.py, pData.pz, THRUST::OPTIMAL,false,false,NULL,true,pData.pwflag);

    eData.Thrust        = thrust.Mag();
    eData.TTheta        = thrust.Theta();
    eData.TPhi          = thrust.Phi();
    eData.ThrustWithMissP   = thrustWithMissP.Mag();
    eData.TThetaWithMissP   = thrustWithMissP.Theta();
    eData.TPhiWithMissP     = thrustWithMissP.Phi();
    if ( do_thrustMissP ) {
        setThrustVariables(&pData, &eData, TVector3(),
                           TVector3(), TVector3(), TVector3(), TVector3(),
                           thrustWithMissP);
    }

    Sphericity spher = Sphericity(pData.nParticle,
                        pData.px,
                        pData.py,
                        pData.pz,
                        pData.pwflag,
                        false);
    spher.setTree(&eData);

    eventSelection eSelection;
    eSelection.setEventSelection(&pData, &eData);

    eData.passesTotalChgEnergyMin = eSelection.getPassesTotalChgEnergyMin();
    eData.passesNeuNch = eSelection.getPassesNeuNch();
    eData.passesNTrkMin = eSelection.getPassesNTrkMin();
    eData.passesSTheta = eSelection.getPassesSTheta();
    eData.passesBELLE = eSelection.getPassesNeuNch() && 
      eSelection.getPassesSTheta() &&	
      eSelection.getPassesTotalChgEnergyMin() &&
      eSelection.getPassesNTrkMin();
    eData.passesISR = eSelection.getPassesISR();
    eData.passesWW = eSelection.getPassesWW();
    eData.Mvis = eSelection.getMvis();
    eData.sPrime = eSelection.getsPrime();
    eData.d2 = eSelection.getd2();
    eData.cW = eSelection.getcW();
}

void NanoAODWriter::defineHadronTagging(std::unique_ptr<RNTupleModel> &model)
{
    MakeField(model, "Part_fromBc", "Reco particle has Bc ancestor in gen (via sim match)", Part_fromBc_);
    MakeField(model, "Part_fromBs", "Reco particle has Bs ancestor in gen", Part_fromBs_);
    MakeField(model, "Part_fromBu", "Reco particle has Bu ancestor in gen", Part_fromBu_);
    MakeField(model, "Part_fromBd", "Reco particle has Bd ancestor in gen", Part_fromBd_);
    MakeField(model, "Part_fromLb", "Reco particle has Lb ancestor in gen", Part_fromLb_);
    MakeField(model, "n_Bc_Emin", "Gen Bc count thrust-min hemisphere (w.r.t. ThrustWithMissP)", n_Bc_Emin_);
    MakeField(model, "n_Bs_Emin", "Gen Bs count thrust-min hemisphere", n_Bs_Emin_);
    MakeField(model, "n_Bu_Emin", "Gen Bu count thrust-min hemisphere", n_Bu_Emin_);
    MakeField(model, "n_Bd_Emin", "Gen Bd count thrust-min hemisphere", n_Bd_Emin_);
    MakeField(model, "n_Lb_Emin", "Gen Lb count thrust-min hemisphere", n_Lb_Emin_);
    MakeField(model, "n_Bc_Emax", "Gen Bc count thrust-max hemisphere", n_Bc_Emax_);
    MakeField(model, "n_Bs_Emax", "Gen Bs count thrust-max hemisphere", n_Bs_Emax_);
    MakeField(model, "n_Bu_Emax", "Gen Bu count thrust-max hemisphere", n_Bu_Emax_);
    MakeField(model, "n_Bd_Emax", "Gen Bd count thrust-max hemisphere", n_Bd_Emax_);
    MakeField(model, "n_Lb_Emax", "Gen Lb count thrust-max hemisphere", n_Lb_Emax_);
    MakeField(model, "label_Bc_Emin", "FCC-style exclusive Bc label Emin", label_Bc_Emin_);
    MakeField(model, "label_Bs_Emin", "FCC-style exclusive Bs label Emin", label_Bs_Emin_);
    MakeField(model, "label_Bu_Emin", "FCC-style exclusive Bu label Emin", label_Bu_Emin_);
    MakeField(model, "label_Bd_Emin", "FCC-style exclusive Bd label Emin", label_Bd_Emin_);
    MakeField(model, "label_Lb_Emin", "FCC-style exclusive Lb label Emin", label_Lb_Emin_);
    MakeField(model, "label_light_Emin", "FCC-style light-B label Emin", label_light_Emin_);
    MakeField(model, "label_hasBc_Emin", "Any Bc in Emin hemisphere", label_hasBc_Emin_);
    MakeField(model, "label_has1Bc_Emin", "Exactly one Bc in Emin hemisphere", label_has1Bc_Emin_);
    MakeField(model, "label_Bc_Emax", "FCC-style exclusive Bc label Emax", label_Bc_Emax_);
    MakeField(model, "label_Bs_Emax", "FCC-style exclusive Bs label Emax", label_Bs_Emax_);
    MakeField(model, "label_Bu_Emax", "FCC-style exclusive Bu label Emax", label_Bu_Emax_);
    MakeField(model, "label_Bd_Emax", "FCC-style exclusive Bd label Emax", label_Bd_Emax_);
    MakeField(model, "label_Lb_Emax", "FCC-style exclusive Lb label Emax", label_Lb_Emax_);
    MakeField(model, "label_light_Emax", "FCC-style light-B label Emax", label_light_Emax_);
    MakeField(model, "label_hasBc_Emax", "Any Bc in Emax hemisphere", label_hasBc_Emax_);
    MakeField(model, "label_has1Bc_Emax", "Exactly one Bc in Emax hemisphere", label_has1Bc_Emax_);
}

void NanoAODWriter::fillHadronTagging(const eventData &eData)
{
    const int nPart = static_cast<int>(*nPart_);
    const int nGen = static_cast<int>(*nGenPart_);
    const int nSim = static_cast<int>(*nSimPart_);

    delphi_hadron_tagging::fillEventHemisphereCounts(
        nGen, *GenPart_pdgId_, *GenPart_fourMomentum_, eData.TThetaWithMissP, eData.TPhiWithMissP,
        *n_Bc_Emin_, *n_Bs_Emin_, *n_Bu_Emin_, *n_Bd_Emin_, *n_Lb_Emin_,
        *n_Bc_Emax_, *n_Bs_Emax_, *n_Bu_Emax_, *n_Bd_Emax_, *n_Lb_Emax_);

    delphi_hadron_tagging::fillExclusiveLabels(
        *n_Bc_Emin_, *n_Bs_Emin_, *n_Bu_Emin_, *n_Bd_Emin_, *n_Lb_Emin_,
        *label_Bc_Emin_, *label_Bs_Emin_, *label_Bu_Emin_, *label_Bd_Emin_, *label_Lb_Emin_,
        *label_light_Emin_, *label_hasBc_Emin_, *label_has1Bc_Emin_,
        *n_Bc_Emax_, *n_Bs_Emax_, *n_Bu_Emax_, *n_Bd_Emax_, *n_Lb_Emax_,
        *label_Bc_Emax_, *label_Bs_Emax_, *label_Bu_Emax_, *label_Bd_Emax_, *label_Lb_Emax_,
        *label_light_Emax_, *label_hasBc_Emax_, *label_has1Bc_Emax_);

    delphi_hadron_tagging::fillPartFlavourFlags(
        nPart, nSim, nGen, *Part_simIdx_, *SimPart_genIdx_, *GenPart_parentIdx_, *GenPart_pdgId_,
        *Part_fromBc_, *Part_fromBs_, *Part_fromBu_, *Part_fromBd_, *Part_fromLb_);
}

void NanoAODWriter::user99()
{
    file_->cd();
    out_t->Write();
    out_tgen->Write();
    out_tsim->Write();
    writer_.reset();
    file_->Close();
};
