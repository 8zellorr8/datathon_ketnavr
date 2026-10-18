"""Local research probe only: provisional per-camera door appearance calibration.

No model/API changes. Pixel vectors are never sent anywhere. Training references
are selected explicitly; prediction receives only image features and parameters.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

from PIL import Image

from forte_ai.video import extract_frame, probe_video


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results' / 'door_review' / 'cv_probe'
ROI = (0.0, 0.0, 0.65, 0.70)
ROI_SIZE = (64, 96)
ANCHOR_SIZE = (32, 48)
TRAIN = {'closed': ('X-31', [12, 14, 16, 18, 20]),
         'open': ('X-32', [8, 10, 12, 14, 16, 18])}
EVALUATE = {'X-31': [0, 2, 4, 6, 8, 10, 22],
            'X-32': [0, 2, 4, 6, 20],
            'X-33': list(range(0, 15, 2))}


def features_from_image(image):
    """Accept pixels only; neither source names nor labels enter this function."""
    image = image.convert('L')
    width, height = image.size
    box = tuple(round(value * (width if index % 2 == 0 else height))
                for index, value in enumerate(ROI))
    cropped = image.crop(box).resize(ROI_SIZE, Image.Resampling.BILINEAR)
    anchor = image.resize(ANCHOR_SIZE, Image.Resampling.BILINEAR)
    def centered_values(pixels):
        values = [pixel / 255.0 for pixel in pixels.tobytes()]
        level = statistics.fmean(values)
        return [value - level for value in values]
    return {'roi': centered_values(cropped), 'anchor': centered_values(anchor)}


def moments(samples, field):
    columns = zip(*(sample[field] for sample in samples))
    means, variances = [], []
    for column in columns:
        center = statistics.fmean(column)
        means.append(center)
        variances.append(statistics.fmean((value - center) ** 2 for value in column))
    return means, variances


def fit_calibration(samples):
    """Learn class moments from selected reference vectors, with motion masking."""
    means, variances = {}, {}
    for state in ('closed', 'open'):
        means[state], variances[state] = moments(samples[state], 'roi')
    weights = []
    for index in range(len(means['open'])):
        variance = max(variances[state][index] for state in means)
        contrast = abs(means['open'][index] - means['closed'][index])
        # Stable contrast only. A moving body should acquire large temporal
        # variance; stationary body/clothing remains an unresolved confound.
        weight = min(25.0, contrast ** 2 / (variance + 0.0004))
        weights.append(weight if variance <= .0025 and contrast >= .04 else 0.0)
    total = sum(weights)
    weights = [weight / total for weight in weights] if total else weights
    all_samples = samples['closed'] + samples['open']
    anchor_mean, anchor_variance = moments(all_samples, 'anchor')
    anchor_weights = []
    for index, variance in enumerate(anchor_variance):
        x, y = index % ANCHOR_SIZE[0], index // ANCHOR_SIZE[0]
        outside_roi = x / ANCHOR_SIZE[0] >= ROI[2] or y / ANCHOR_SIZE[1] >= ROI[3]
        anchor_weights.append(1.0 if outside_roi and variance <= .003 else 0.0)
    anchor_total = sum(anchor_weights)
    anchor_weights = [weight / anchor_total for weight in anchor_weights] if anchor_total else anchor_weights
    model = {'version': 'research_pixel_centroids_1', 'roi_normalized': ROI,
             'roi_size': ROI_SIZE, 'anchor_size': ANCHOR_SIZE, 'means': means,
             'within_class_variances': variances, 'weights': weights,
             'anchor_mean': anchor_mean, 'anchor_weights': anchor_weights,
             'min_margin': .20, 'min_stable_pixels': 40,
             'class_max_distance': {}, 'camera_max_distance': 0.0}
    for state in means:
        distances = [weighted_distance(sample['roi'], means[state], weights) for sample in samples[state]]
        model['class_max_distance'][state] = max(distances) * 2 + .015
    anchor_distances = [weighted_distance(sample['anchor'], anchor_mean, anchor_weights) for sample in all_samples]
    model['camera_max_distance'] = max(anchor_distances) * 2 + .04
    return model


def weighted_distance(values, center, weights):
    return math.sqrt(sum(weight * (value - baseline) ** 2
                         for value, baseline, weight in zip(values, center, weights)))


def predict(features, calibration):
    """Image vectors + learned parameters only. No IDs, hashes or annotations."""
    distances = {state: weighted_distance(features['roi'], center, calibration['weights'])
                 for state, center in calibration['means'].items()}
    nearest = min(distances, key=distances.get)
    farthest = max(distances.values())
    margin = (farthest - distances[nearest]) / max(farthest, 1e-9)
    anchor_distance = weighted_distance(features['anchor'], calibration['anchor_mean'],
                                        calibration['anchor_weights'])
    reasons = []
    if sum(weight > 0 for weight in calibration['weights']) < calibration['min_stable_pixels']:
        reasons.append('insufficient_stable_pixels')
    if not any(calibration['anchor_weights']) or anchor_distance > calibration['camera_max_distance']:
        reasons.append('camera_appearance_outside_reference')
    if margin < calibration['min_margin']:
        reasons.append('ambiguous_centroid_margin')
    if distances[nearest] > calibration['class_max_distance'][nearest]:
        reasons.append('door_appearance_outside_reference')
    return {'state': 'uncertain' if reasons else nearest, 'nearest_centroid': nearest,
            'distances': {state: round(value, 6) for state, value in distances.items()},
            'margin': round(margin, 6), 'anchor_distance': round(anchor_distance, 6),
            'abstention_reasons': reasons}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'data' / 'manifest.json').read_text(encoding='utf-8'))
    records = {record['id']: record for record in manifest['records']}
    samples, provenance, cache = {}, [], {}
    def load(identifier, requested_time):
        key = (identifier, requested_time)
        if key not in cache:
            frame = extract_frame(records[identifier]['video_path'], requested_time, max_side=560)
            with Image.open(io.BytesIO(base64.b64decode(frame['image_base64']))) as image:
                vectors = features_from_image(image)
            cache[key] = (frame['timestamp_sec'], vectors)
        return cache[key]
    for state, (identifier, times) in TRAIN.items():
        samples[state] = []
        for timestamp in times:
            actual, vector = load(identifier, timestamp)
            samples[state].append(vector)
            provenance.append({'source_id': identifier, 'requested_sec': timestamp,
                               'actual_pts_sec': actual, 'provisional_state': state})
    model = fit_calibration(samples)
    (OUTPUT / 'calibration.json').write_text(json.dumps(model, allow_nan=False), encoding='utf-8')
    importance_grid = [[0.0 for _ in range(4)] for _ in range(4)]
    for index, weight in enumerate(model['weights']):
        x, y = index % ROI_SIZE[0], index // ROI_SIZE[0]
        importance_grid[y * 4 // ROI_SIZE[1]][x * 4 // ROI_SIZE[0]] += weight
    training_results = []
    for state, vectors in samples.items():
        for vector in vectors:
            training_results.append({'reference_state': state, **predict(vector, model)})
    evaluations = []
    for identifier, times in EVALUATE.items():
        duration = probe_video(records[identifier]['video_path'])['duration_sec']
        for timestamp in times:
            if timestamp >= duration:
                continue
            actual, vector = load(identifier, timestamp)
            evaluations.append({'source_id': identifier, 'requested_sec': timestamp,
                                'actual_pts_sec': actual, **predict(vector, model)})
    report = {
        'purpose': 'Development research only; no API or production model changed.',
        'training_provenance': provenance, 'training_reference_predictions': training_results,
        'train_video_ids': ['X-31', 'X-32'], 'development_probe_video_id': 'X-33',
        'labels_provenance': 'Provisional prior human descriptions and supplied intervals; not independently verified.',
        'evaluation': evaluations, 'heldout_video_is_development_not_blind_test': True,
        'weights_sha256': hashlib.sha256((OUTPUT / 'calibration.json').read_bytes()).hexdigest(),
        'stable_pixels': sum(weight > 0 for weight in model['weights']),
        'weight_mass_grid_4x4_within_roi': [[round(value, 4) for value in row] for row in importance_grid],
        'camera_distance_threshold': round(model['camera_max_distance'], 6),
        'class_distance_thresholds': model['class_max_distance'],
        'production_usable': False,
        'unresolved_attribution': [
            'Without verified door segmentation, selected stable pixels may encode furniture, lighting, or stationary clothing.',
            'Two same-camera reference clips do not establish robustness or reject every different camera.',
            'No door-state accuracy is established; provisional times are not verified per-frame truth.',
        ],
        'no_training_on_x33': True, 'qwen_weights_modified': False, 'external_ai_calls': False,
    }
    (OUTPUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    compact = {'stable_pixels': report['stable_pixels'], 'weight_mass_grid_4x4': report['weight_mass_grid_4x4_within_roi'],
               'production_usable': False, 'evaluation': evaluations}
    print(json.dumps(compact, ensure_ascii=True, allow_nan=False))


if __name__ == '__main__':
    main()
