# 6buwan [alpha]: A RAW-vs-HOG Pipeline for OCSVM Signature Verification
 
![6buwan logo](/docs/images/logo.png)
 
## Motivation
 
A signature verification system deployed in practice rarely has more than a handful of enrolled genuine signatures per person to learn from, and it almost never has access to that person's forgeries in advance. This rules out standard two-class classification and forces a one-class formulation: a model per writer, fit only to that writer's genuine samples, that must still separate genuine signatures from forgery attempts it has never seen. Given that constraint, an obvious engineering question follows — is it worth extracting a hand-designed feature like HOG at all, or does a one-class model do just as well, or better, working directly on raw pixels? HOG was built to be robust to exactly the kind of variation (stroke thickness, minor translation, local contrast) that raw pixels are sensitive to, so the intuition favors HOG. Whether that intuition survives contact with a genuinely one-class, low-data regime is the question this pipeline is built to answer, not assume.
 
## Methodology
 
### Preprocessing
 
Every scan is binarized with an Otsu threshold to locate ink, cropped to that ink's bounding box, then resized with aspect preserved and padded to a fixed frame. This matters more than it looks: a naive resize of the full scan (blank margin included) puts the signature at a different scale and position in the frame for almost every image, since scans don't share a consistent margin. That inconsistency would add noise to both feature representations, but not necessarily the same amount of noise to each — which would confound any RAW vs. HOG comparison before it starts. Cropping to ink first removes that confound; both representations are then computed from the same, consistently-framed image.
 
### Feature Representations
 
**RAW** is the preprocessed image flattened and normalized — a fixed-length vector with no built-in invariance to anything. A signature's absolute pixel arrangement is the entire signal.
 
**HOG** (Dalal & Triggs: oriented gradient histograms over local cells and blocks) discards absolute pixel intensity and instead encodes local stroke direction and edge structure, which is designed to tolerate small shifts, thickness variation, and lighting differences that RAW cannot.
 
These encode different, explicit assumptions about what makes a signature identifiable — position and intensity for RAW, local gradient structure for HOG — which is exactly what makes comparing them informative rather than incidental.
 
### Verification Model
 
One RBF-kernel One-Class SVM is trained per writer, on that writer's genuine training signatures only, mirroring the enrollment-only constraint from the Motivation section above: `nu` bounds the expected fraction of training points treated as outliers, and `gamma` sets the kernel's locality. Both are tuned from a grid centered on a data-driven reference value (sklearn's `gamma='scale'` heuristic) fixed before any validation results are seen, rather than searched and widened reactively.
 
### What Makes the RAW vs. HOG Comparison Fair
 
Everything except the feature representation is held identical between the two conditions: the same preprocessing, the same writer-specific train/validation/test split (fixed random seed), the same scaler-fitting rule (fit only on pooled training-genuine signatures, applied everywhere else), and the same hyperparameter search procedure. Final-test data is touched exactly once, after both representations' hyperparameters are locked. Any performance gap that remains is therefore attributable to the representation itself, not to an inconsistency in how the two conditions were run.
 
### Evaluation
 
Each writer's model is scored on that writer's held-out genuine and forged signatures using False Acceptance Rate, False Rejection Rate, Accuracy, Precision, Recall, and F1, then aggregated across writers. Results are also compared against trivial reject-all and accept-all baselines, so that any reported gain is checked against what a non-model would achieve on the same class balance.
 
## Dataset
 
Experiments are run on the **CEDAR** offline signature dataset (55 writers, genuine and forged signatures per writer). Per writer, genuine signatures are split into training, validation, and held-out final-test sets; forged signatures are used only for validation and final testing, never for training, consistent with the one-class formulation above. The pipeline is designed to generalize to additional signature datasets, and extending it beyond CEDAR is part of ongoing work.
 
