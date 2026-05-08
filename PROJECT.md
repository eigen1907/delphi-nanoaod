# DELPHI exclusive two-body hadronic final state search

## Current Workspace
>https://github.com/eigen1907/delphi-nanoaod/tree/dev-fullDST
>https://github.com/eigen1907/DeepMuonReco

## 0. Background

### 0.1 Physics Background

Starting point of this project is the LEP1 Z-pole dataset.

The final DELPHI LEP1 sample corresponds to an integrated luminosity of about $116~\mathrm{pb}^{-1}$ collected around the $Z^0$ resonance.  
For DELPHI LEP2, the full dataset corresponds to about $660~\mathrm{pb}^{-1}$ collected at centre-of-mass energies from $161~\mathrm{GeV}$ to $209~\mathrm{GeV}$.

For the target channels $e^+e^- \to \pi^+\pi^-$ and $e^+e^- \to K^+K^-$, the Standard Model expectation at LEP energies is extremely small.  
So the numbers below should be read only as order-of-magnitude scale estimates, not as DELPHI measurements.

| Running period | Representative $\sqrt{s}$ | DELPHI luminosity | $\sigma(e^+e^- \to \pi^+\pi^-)$ | $\sigma(e^+e^- \to K^+K^-)$ | expected $N(\pi^+\pi^-)$ | expected $N(K^+K^-)$ |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| LEP1 | $91.2~\mathrm{GeV}$ | $116~\mathrm{pb}^{-1}$ | $\sim 2.7\times 10^{-8}~\mathrm{pb}$ | $\sim 2.2\times 10^{-8}~\mathrm{pb}$ | $\sim 3.1\times 10^{-6}$ | $\sim 2.6\times 10^{-6}$ |
| LEP2 | $200~\mathrm{GeV}$ | $660~\mathrm{pb}^{-1}$ | $\sim 2.4\times 10^{-10}~\mathrm{pb}$ | $\sim 2.0\times 10^{-10}~\mathrm{pb}$ | $\sim 1.6\times 10^{-7}$ | $\sim 1.3\times 10^{-7}$ |

These estimates show the main point clearly: the expected signal yield is far below one event for DELPHI.

So this project should be framed as an upper-limit search, not as a realistic observation of the Standard Model signal.

In the literature checked for this project, there does not seem to be a dedicated published DELPHI or LEP upper limit specifically for the exclusive two-body charged-hadron final states:
- $Z \to \pi^+\pi^-$
- $Z \to K^+K^-$

### 0.2 Detector Background

| Family | Detector | Meaning | Main role in the process-label model | Main measurable quantities |
| :-- | :-- | :-- | :-- | :-- |
| Tracking | VD | Vertex Detector | hit-level structure near the interaction point | hit position<br>vertex-near activity<br>impact-parameter-related information |
| Tracking | ID / OD | Inner / Outer tracking detectors | charged-track pattern and trajectory support | track points<br>direction<br>lever arm |
| Tracking | FCA / FCB | Forward Chamber A / B | forward charged-track reconstruction | forward track points<br>forward direction |
| Tracking | VFT | Very Forward Tracker | very forward charged-particle information | forward hit / track information |
| Tracking | TPC | Time Projection Chamber | main charged-track and hadron-PID detector | momentum<br>charge<br>curvature<br>`dE/dx` |
| Calorimetry | HPC | High Density Projection Chamber | barrel EM response and neutral-activity information | EM shower energy<br>position<br>shower shape<br>layer response |
| Calorimetry | FEMC | Forward Electromagnetic Calorimeter | forward EM response and veto | forward EM energy<br>position<br>forward veto activity |
| Calorimetry | HCAL / HAC | Hadron Calorimeter | hadronic activity and punch-through information | hadronic energy<br>shower depth<br>extra hadronic activity |
| Calorimetry / Forward veto | STIC | Small Angle Tile Calorimeter | very forward veto detector | very forward energy deposit<br>small-angle veto activity |
| PID / Muon | BRICH | Barrel Ring Imaging Cherenkov | barrel `\pi/K` separation if added later | Cherenkov-angle-related PID<br>likelihood-like PID<br>photon-count-like quantities |
| PID / Muon | FRICH | Forward Ring Imaging Cherenkov | forward `\pi/K` separation if added later | Cherenkov-angle-related PID<br>likelihood-like PID<br>photon-count-like quantities |
| PID / Muon | MUB / MUF / SMC | Barrel / Forward / Surround Muon Chambers | muon rejection and punch-through tagging | muon matching<br>muon-ID response<br>outer muon activity |
| PID / Electron | Electron-ID system | electron rejection and classification support | electron-ID tag<br>conversion-related flag<br>refit momentum information | detector-level electron-ID response |
| Trigger / Lumi | TOF | Time Of Flight | possible timing support | timing<br>limited PID support |
| Trigger / Lumi | HOF / VSAT | Forward activity taggers | forward activity support and veto | forward hit/activity<br>tag/veto support |

---

## 1. Physics objective

This project targets an upper-limit search for rare exclusive two-body charged-hadron final states in DELPHI data:
- $e^+e^- \to K^+K^-$
- $e^+e^- \to \pi^+\pi^-$

Because the expected Standard Model yield is far below one event for DELPHI, the goal is not a realistic direct observation of the signal.  
Instead, the analysis should be framed as a rare-event discrimination problem followed by a final upper-limit result.

In this project, the analysis is formulated as an event-level classification problem using detector-near raw information.  
The model is trained to predict the underlying process label directly from the full detector response of each event.

From a physics point of view, one may expect the model to learn discriminating behavior such as:
- exclusive low-multiplicity topology
- two-track signal-like structure
- rejection of leptonic backgrounds
- rejection of generic hadronic backgrounds
- separation between $\pi^+\pi^-$-like and $K^+K^-$-like events

However, these are only possible emergent behaviors of the model, not explicit intermediate tasks imposed by the analysis design.

The main point of this project is therefore not to enforce a manually factorized sequence of physics selections, but to let the model process the detector-level raw information end-to-end and directly infer the most probable process class.

The classifier should therefore learn to distinguish process classes such as:
- $e^+e^- \to \pi^+\pi^-$
- $e^+e^- \to K^+K^-$
- $e^+e^- \to q\bar{q}$
- $e^+e^- \to \mu^+\mu^-$
- $e^+e^- \to e^+e^-$
- $e^+e^- \to \tau^+\tau^-$
- $\gamma\gamma \to \mathrm{hadrons}$
- $\gamma\gamma \to \ell^+\ell^-$
- other relevant backgrounds

Its output will then be used for signal selection and upper-limit extraction.

---

## 2. Technical Scope

This project starts from the current DELPHI extraction framework in the two `delphi-nanoaod` branches and aims to build a first detector-level ML pipeline without large software changes.

The first technical goal is simple:
- build an event-level ML dataset from the information already accessible in `shortDST` and `full-DST`
- train a first deep learning model that predicts the process label directly from that event-level input

So the first stage is mainly about turning heterogeneous detector information into one usable event representation.

Initial rules:
- do not modify the existing extraction modules for the first result
- use only what is already extractable from `shortDST` and `full-DST`
- define signal and background samples first
- build the first detector-level dataset first
- add new detector content only when clearly needed

### Practical meaning of "raw"

Here, "raw" means detector-near bank-level or object-level information from `DST`.  
It does not mean true detector readout.

So the first version should focus on:
- event-level information
- vertex information
- charged-track information
- TPC PID information
- calorimeter information
- lepton-ID information
- MC truth labels

Derived quantities such as opening angle, acoplanarity, momentum balance, $E/p$, event exclusivity, or pair-level PID scores are not required in the first version.

### Deep learning model direction

For the first model, we will follow the general idea of the `latent_attention` model and adapt it to the DELPHI event-classification problem.

The basic idea for this project is:
- treat each detector block as a separate input group
- project each group into a common embedding space
- compress the information with a small set of latent tokens
- fuse the detector blocks with attention
- predict the final process label from the fused event representation

For DELPHI, the detector blocks can be organized as:
- event / vertex
- charged-track
- TPC PID
- EM calorimetry
- hadronic calorimetry
- STIC / forward veto
- muon-ID
- electron-ID
- optional RICH block later

So the model should not rely on a long sequence of hand-made physics cuts.  
Instead, it should learn how to combine the full detector response and directly predict the most probable process class.

### First implementation plan

The first implementation should follow this order:
1. extract the information that is already available
2. organize it into one event-level representation
3. assign a process label to each MC event
4. train a first latent-attention-style classifier
5. check which detector blocks are useful
6. add more detector content only if necessary

So the technical starting point of the project is:

**available detector information $\to$ event representation $\to$ process label prediction**

---

## 3. DELPHI Dataset Summary

This project should keep detector-near raw information as much as possible.

Here, "raw" means bank-level or object-level information available from `shortDST` or `full-DST`.  
It does not mean true detector readout.

To make the structure clearer, the accessible information is organized into four layers:

1. detector-native information
2. object-level information
3. summary-level information
4. MC supervision information

### 3.1 Event Information

| Category       | Built from               | Available now                                  | To be added | If needed |
| :------------- | :----------------------- | :--------------------------------------------- | :---------- | :-------- |
| event metadata | event header             | `(both) Event_*`                               |             |           |
| run conditions | beam / field information | `(both) beam spot`,<br>`(both) magnetic field` |             |           |

### 3.2 Detector Information

| Subdetector | Used for                                                    | Available now                                                                                                                                                           | To be added                                              | If needed            |
| :---------- | :---------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------- | :------------------- |
| VD          | charged-track reconstruction                                | `(short) VdAssocHit_*`,<br>`(short) VdUnassocHit_*`                                                                                                                     |                                                          |                      |
| ID          | charged-track object                                        | `(full) TrackElement_*`                                                                                                                                                 |                                                          | finer track elements |
| TPC         | charged-track object,<br>hadron PID,<br>electron-ID support | `(both) MtpcRaw_*` raw `dE/dx`,<br>`(both) truncated-mean variants`,<br>`(both) occupancy-related packed fields`,<br>`(both) Z-fit quality`,<br>`(full) TrackElement_*` |                                                          | `PA.MTPC(7)`         |
| OD          | charged-track object                                        | `(full) TrackElement_*`                                                                                                                                                 |                                                          | finer track elements |
| FCA         | charged-track object                                        | `(full) TrackElement_*`                                                                                                                                                 |                                                          | finer track elements |
| FCB         | charged-track object                                        | `(full) TrackElement_*`                                                                                                                                                 |                                                          | finer track elements |
| VFT         | forward charged-track support                               |                                                                                                                                                                         |                                                          | VFT information      |
| HPC         | EM shower object,<br>electron-ID support                    | `(both) EmShower_*`,<br>`(both) EmLayer_*`                                                                                                                              | `(short) EmCluster_*`-style inner EM-cluster information | cluster details      |
| FEMC        | forward EM shower object,<br>electron-ID support            | `(both) EmShower_*`                                                                                                                                                     |                                                          | cluster details      |
| HCAL / HAC  | hadronic shower object,<br>muon-ID support                  | `(both) HadShower_*`,<br>`(both) HadHit_*`                                                                                                                              |                                                          | layer details        |
| STIC        | forward veto / small-angle EM object                        | `(both) Stic_*`                                                                                                                                                         |                                                          | tower details        |
| BRICH       | hadron PID                                                  |                                                                                                                                                                         | BRICH PID information                                    | bank-level BRICH     |
| FRICH       | hadron PID                                                  |                                                                                                                                                                         | FRICH PID information                                    | bank-level FRICH     |
| MUB         | muon-ID object                                              |                                                                                                                                                                         |                                                          | chamber hits         |
| MUF         | muon-ID object                                              |                                                                                                                                                                         |                                                          | chamber hits         |
| SMC         | muon-ID object                                              |                                                                                                                                                                         |                                                          | chamber hits         |
| TOF         | timing / PID support                                        |                                                                                                                                                                         |                                                          | timing information   |
| HOF         | forward veto support                                        |                                                                                                                                                                         |                                                          | veto information     |
| VSAT        | very-forward veto support                                   |                                                                                                                                                                         |                                                          | veto information     |

### 3.3 Object Information

| Object          | Built from                                        | Available now                                                                                                                                                                | To be added                                              | If needed       |
| :-------------- | :------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------- | :-------------- |
| primary vertex  | vertex reconstruction                             | `(both) Vtx_*`                                                                                                                                                               |                                                          |                 |
| charged-track   | VD,<br>ID,<br>TPC,<br>OD,<br>FCA,<br>FCB          | `(both) TracRaw_*` track parameters,<br>`(both) fit quality`,<br>`(both) detector-use flags`,<br>`(both) track length`,<br>`(both) first measured point`,<br>`(both) charge` |                                                          | track blocks    |
| EM shower       | HPC,<br>FEMC                                      | `(both) EmShower_*`,<br>`(both) EmLayer_*`                                                                                                                                   | `(short) EmCluster_*`-style inner EM-cluster information | cluster details |
| hadronic shower | HCAL / HAC                                        | `(both) HadShower_*`,<br>`(both) HadHit_*`                                                                                                                                   |                                                          | layer details   |
| STIC shower     | STIC                                              | `(both) Stic_*`                                                                                                                                                              |                                                          | tower details   |
| muon-ID         | MUB,<br>MUF,<br>SMC,<br>HCAL / HAC,<br>HPC / FEMC | `(both) MuidRaw_*`                                                                                                                                                           |                                                          | `PA.MU(4)`      |
| electron-ID     | TPC,<br>HPC,<br>FEMC                              | `(both) ElidRaw_*`                                                                                                                                                           |                                                          | `PA.EL(5)`      |
| hadron-PID      | TPC,<br>BRICH,<br>FRICH                           |                                                                                                                                                                              | BRICH / FRICH PID information                            | bank-level PID  |
### 3.4 MC Information

| Category | Built from | Available now | To be added | If needed |
| :-- | :-- | :-- | :-- | :-- |
| MC truth / supervision label | simulation banks | `(both, MC only) GenPart_*` | reco-truth matching table | truth back-pointers |


the analysis can combine:
- detector-native information
- object-level information
- summary-level information
- MC supervision labels

within one event-level representation.

### Note on DST formats

For this project, the relevant DELPHI formats are:
- `short DST`: mostly used during LEP1
- `long DST`: a richer DST format documented separately from short DST
- `full DST`: the direct output from the reconstruction process

In the current `delphi-raw-nanoaod` implementation, the extractor is built around:
- `shortDST (.sdst)`
- `full-DST (.fadana)`

On branch `dev-rawDST`, the intended data path is extended to:
- raw open-data (`/eos/opendata/delphi/raw-data/.../*.sl`)
- DELANA DETRAW raw-DST (`RAW`/`TAN`/`DST` records)
- raw nanoAOD with both generic `RawBank_*` / `RawWord_*` banks and the
  existing object-level PHDST collections

In this mode, `Event_recordType` is important: detector raw banks are stored on
`RAW` records, while reconstructed PA/DST objects are stored on `DST` records
for the same `(run, event)`.

with one shared output schema, while the actually populated collections depend on the input type.

### References

Current implementation and schema:
- `https://github.com/jingyucms/delphi-nanoaod/tree/dev-fullDST/delphi-raw-nanoaod/README.md`

Planned lower-level bank extensions:
- `https://github.com/jingyucms/delphi-nanoaod/tree/dev-fullDST/docs/PHDST_RAW_NANOAOD_PLAN.md`

DELPHI format manuals:
- short DST manual record: `https://opendata.cern.ch/record/80506`
- long DST manual record: `https://opendata.cern.ch/record/80507`
- full DST manuals record: `https://opendata.cern.ch/record/80504`

DELPHI open-data overview:
- `https://opendata.cern.ch/docs/about-delphi`

