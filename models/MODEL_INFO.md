# AERIS-X Model Information

## General Details
- **Model Name**: Drone Detection & Distance Estimation (YOLOv8)
- **Model File**: `models/drone_yolov8n.pt`
- **File Size**: 12,351,474 bytes (~12.4 MB)
- **SHA-256**: `bf24a20e69b28896a0c7e4855c72146149fa5c25845fa5e54beda2b93cf79824`
- **Source Repository**: `TomSmail/drone-yolo-v1`
- **Source URL**: [https://huggingface.co/TomSmail/drone-yolo-v1](https://huggingface.co/TomSmail/drone-yolo-v1)
- **Author/Organization**: TomSmail
- **License**: CC-BY-4.0
- **Date Obtained**: 2026-09-11

## Architecture
- **Framework**: Ultralytics (YOLOv8)
- **Variant**: YOLOv8n (nano)
- **Inference Device**: CPU (configurable)

## Class Mapping
The model natively exports the following classes:
- `0`: `quadcopter`
- `1`: `fixed-wing`

**Selected Target Class for AERIS-X**: `quadcopter`
*(The model retains the capability to detect fixed-wing UAVs, but AERIS-X Phase 4 is specifically configured to target `quadcopter` via `config.json`)*.

## Performance Metrics
> **WARNING**: Do NOT confuse published metrics with AERIS-X experimental results.

### PUBLISHED MODEL METRICS
- These metrics were reported by the model author upon their specific test dataset.
- Training: 30 epochs on RTX 3090.
- Specific precision, recall, and mAP values were not detailed on the model card beyond general dataset references.

### AERIS-X EXPERIMENTAL RESULTS
- **Accuracy**: (Pending full physical validation with a real drone).
- **Physical Integration Status**: Frame extraction and inference pipeline verified, but real-world drone detection accuracy is not yet established. 
- **Inference Latency/FPS**: (Pending physical evaluation).

## Limitations
- This model was downloaded for prototyping purposes and has not undergone rigorous military/defense certification.
- Detection relies solely on visual features matching the training data (commercial multirotors and military fixed-wing).
- Low light and high altitude performance degradation are expected and will be addressed by Phase 5 adaptations.
