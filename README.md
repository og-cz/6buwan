# 6buwan [alpha]: A RAW-vs-HOG Pipeline for OCSVM Signature Verification

![6buwan logo](/docs/images/logo.png)

## Motivation

A signature verification system deployed in practice rarely has more than a handful of enrolled genuine signatures per person to learn from, and it almost never has access to that person's forgeries in advance. This rules out standard two-class classification and forces a one-class formulation: a model per writer, fit only to that writer's genuine samples, that must still separate genuine signatures from forgery attempts it has never seen. Given that constraint, an obvious engineering question follows: is it worth extracting a hand-designed feature like HOG at all, or does a one-class model do just as well, or better, working directly on raw pixels? HOG was built to be robust to exactly the kind of variation (stroke thickness, minor translation, local contrast) that raw pixels are sensitive to, so the intuition favors HOG. Whether that intuition survives contact with a genuinely one-class, low-data regime is the question this pipeline is built to answer, not assume.

## Methodology

![Preprocessing steps: original → Otsu threshold → density-crop → aspect cap → resize & pad](/docs/diagrams/6buwan-chart-full-preprocessing.png)

### Preprocessing

Every scan is binarized with an Otsu threshold to locate ink, cropped to that ink's bounding box, then resized with aspect preserved and padded to a fixed 100x100 frame. A naive resize of the full scan (blank margin included) puts the signature at a different scale and position in the frame for almost every image, since scans don't share a consistent margin that inconsistency would add noise to both feature representations, but not necessarily the same amount to each, which would confound any RAW vs. HOG comparison before it starts. Cropping to ink first removes that confound; both representations are then computed from the same, consistently-framed image.

Cropping itself is density-thresholded rather than triggered by any single ink pixel: a row or column only counts toward the bounding box if its ink count reaches a fixed fraction of the busiest row/column's count. This keeps a thin trailing flourish stroke from dragging the crop out into a sliver, while a hard cap on the crop's aspect ratio guards against the same failure mode surviving thresholding. Output is left in grayscale rather than re-binarized after cropping, so HOG still sees soft gradient information at stroke edges rather than a hard edge map.

![Signature cleaning example: original CEDAR scans vs. cropped 100x100 output](/docs/diagrams/signature-cleaning-result.png)

### Feature Representations

**RAW** is the preprocessed image flattened and normalized to a fixed-length vector, with no built-in invariance to anything. A signature's absolute pixel arrangement is the entire signal.

**HOG** (Dalal & Triggs: oriented gradient histograms over local cells and blocks) discards absolute pixel intensity and instead encodes local stroke direction and edge structure, which is designed to tolerate small shifts, thickness variation, and lighting differences that RAW cannot.

These encode different, explicit assumptions about what makes a signature identifiable position and intensity for RAW, local gradient structure for HOG which is exactly what makes comparing them informative rather than incidental.

### Verification Model

One RBF-kernel One-Class SVM is trained per writer, on that writer's genuine training signatures only, mirroring the enrollment-only constraint from the Motivation section above: `nu` bounds the expected fraction of training points treated as outliers, and `gamma` sets the kernel's locality. Both are tuned from a grid centered on a data-driven reference value (sklearn's `gamma='scale'` heuristic, `1 / (n_features * variance)`) fixed before any validation results are seen, rather than searched and widened reactively. RAW and HOG get their own reference gamma, computed from their own dimensionality and variance, so the two conditions are anchored fairly rather than sharing one arbitrary grid.

### What Makes the RAW vs. HOG Comparison Fair

Everything except the feature representation is held identical between the two conditions: the same preprocessing, the same writer-specific train/validation/final-test split (fixed random seed), the same scaler-fitting rule (fit only on pooled training-genuine signatures, applied everywhere else), and the same hyperparameter search procedure. Final-test data is touched exactly once, after both representations' hyperparameters are locked. Any performance gap that remains is therefore attributable to the representation itself, not to an inconsistency in how the two conditions were run.

### Evaluation

Each writer's model is scored on that writer's held-out genuine and forged signatures using False Acceptance Rate, False Rejection Rate, Accuracy, Precision, Recall, F1, and Matthews Correlation Coefficient, then aggregated across writers. Results are compared against trivial reject-all and accept-all baselines and, where the test set is closer to balanced, against balanced accuracy, so that any reported gain is checked against what a non-model would achieve on the same class balance. A per-writer ROC-AUC computed from each model's `decision_function` score is used as a threshold-independent check, to separate whether a representation is weaker in general from whether it was simply weaker at the specific operating point selected by the locked hyperparameters.

## Datasets

### CEDAR

Experiments are run on the **CEDAR** offline signature dataset (55 writers, 24 genuine and 24 forged signatures each). Per writer, genuine signatures are split 12 train / 3 validation / 3 final-test; forged signatures are used only for validation (12) and final testing (12), never for training, consistent with the one-class formulation above. This leaves the final-test set skewed 80% forged (165 genuine / 660 forged overall), for which trivial reject-all and accept-all baselines are reported alongside balanced accuracy. A supplementary experiment uses the 6 genuine signatures per writer left untouched by the main split, growing training to 18 genuine signatures per writer while validation and final-test stay at 3 each, to ask directly whether more enrolled reference signatures improves either representation.

### UTSig

**UTSig** is used as a bonus cross-dataset check, run only after the CEDAR pipeline was locked. It was selected after surveying the available alternatives (GPDS, withdrawn for GDPR reasons; its replacements and BHSig260, capped at the same 24 genuine/writer as CEDAR; MCYT-75, only 15 genuine/writer) as the only dataset offering more genuine references per writer than CEDAR: 115 writers with 27 genuine and 6 skilled-forged signatures each, confirmed against the downloaded data rather than assumed from documentation. Genuine signatures are split 21 train / 3 validation / 3 final-test, using every available signature with none left over; skilled forgeries are split 3 validation / 3 final-test, with none held out for training. Skilled forgeries are used rather than UTSig's simple or opposite-hand forgeries because they are the closest analog to CEDAR's forged class and the standard hardest-case benchmark in this literature.

UTSig's final-test set is class-balanced (345 genuine / 345 forged), unlike CEDAR's 80%-forged split, so its trivial baselines and what counts as "beating the baseline" are not directly comparable to CEDAR's without accounting for this. The pipeline (preprocessing, feature extraction, scaler-fitting rule, reference-gamma method, and hyperparameter grid) is reused from CEDAR's locked configuration rather than re-tuned for UTSig, to keep the cross-dataset comparison methodological rather than a second round of fitting to validation.

## References

- CEDAR signature dataset
- UTSig signature dataset
- Dalal, N. & Triggs, B. Histograms of Oriented Gradients for Human Detection
- [signver](https://github.com/victordibia/signver) signature verification library