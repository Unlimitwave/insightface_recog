# Commercial Evaluation Mode

Enterprise Evaluation Mode helps teams evaluate InsightFace locally with their
own data before choosing a commercial model license, private model evaluation,
SDK/API access, SLA, or custom training path.

All processing is local by default. No images, embeddings, videos, or reports
are uploaded automatically.

## KYC / 1:1 Verification

Prepare a CSV:

```csv
image1_path,image2_path,label
/data/kyc/a1.jpg,/data/kyc/a2.jpg,1
/data/kyc/a1.jpg,/data/kyc/b1.jpg,0
```

Labels:

- `1`: same person
- `0`: different person

The evaluation computes total pairs, positive pairs, negative pairs, failed
detections, accuracy, FAR, FRR, TAR at FAR targets when data is sufficient,
recommended threshold, false positive samples, false negative samples, and
average latency.

## Access Control / 1:N Identification

Prepare a gallery folder with one subfolder per person:

```text
gallery/
  alice/
    1.jpg
    2.jpg
  bob/
    1.jpg
```

Prepare a probe folder:

```text
probe/
  lobby_001.jpg
  lobby_002.jpg
```

Optional ground truth CSV:

```csv
image_path,person_name
lobby_001.jpg,alice
lobby_002.jpg,bob
```

The evaluation reports gallery persons, gallery face samples, probe images,
detected probe faces, Top-1 accuracy, Top-5 accuracy, unknown rejection rate,
and confusion cases when ground truth is available.

## Interpreting Metrics

- FAR: false accept rate. Lower means fewer different-person pairs accepted.
- FRR: false reject rate. Lower means fewer same-person pairs rejected.
- TAR: true accept rate at a target FAR.
- Top-1 accuracy: correct person is the first search result.
- Top-5 accuracy: correct person appears in the first five results.

Thresholds are business decisions. Higher thresholds usually reduce false
accepts but increase false rejects.

## Exporting Reports

Click **Export Report** after an evaluation. Reports include:

1. Executive Summary
2. Evaluation Scenario
3. Dataset Summary
4. Model and Runtime
5. License Status
6. Metrics
7. Threshold Recommendation
8. Error Analysis
9. Latency and Hardware
10. Deployment Considerations
11. Responsible Use and Compliance Notice
12. Commercial Licensing Next Steps
13. Appendix: Raw Results

Markdown, HTML, and PDF are supported when the GUI extra is installed.

## Commercial License Notice

This evaluation may use research or non-commercial model files. Production or
commercial deployment requires an appropriate commercial model license. Please
contact InsightFace for commercial model licensing, private model evaluation,
SDK/API access, SLA, or custom training.

This report does not provide legal advice. Users are responsible for consent,
privacy, retention, and compliance with applicable biometric regulations.
