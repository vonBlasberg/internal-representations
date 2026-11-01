"""Post hoc geometry of saved A10 vectors, separate from candidate selection."""
import math
import struct
from .reuse import PACKAGE, legacy

REQUEST_SHA = 'a7b24e2ee43f19c475f3fcbdaa84726ee1df94cf58469993219a2ceb6f7ccd14'
RECEIPTS_SHA = 'ecdb33cb8213062349c908b35d0ed2a989f383a30f56f77b785a69594eb9ff9c'


def dot(left, right):
    if len(left) != len(right):
        raise legacy.AnalysisError('Incompatible vector dimensions')
    return math.fsum(x * y for x, y in zip(left, right))


def attach_a10_geometry(rows):
    request_path = PACKAGE / 'evidence/a10-request.json'
    receipt_path = PACKAGE / 'evidence/a10-geometry/realized_receipts.json'
    if legacy.sha(request_path) != REQUEST_SHA or legacy.sha(receipt_path) != RECEIPTS_SHA:
        raise legacy.AnalysisError('Changed geometry source')
    request, receipts = legacy.load_json(request_path), legacy.load_json(receipt_path)
    vectors = {}
    for kind, layers in request['directions'].items():
        for layer, record in layers.items():
            path = PACKAGE / 'evidence/a10-geometry' / record['path']
            if legacy.sha(path) != record['sha256']:
                raise legacy.AnalysisError('Changed saved direction')
            raw = path.read_bytes()
            vectors[(kind, int(layer))] = struct.unpack('<' + str(len(raw) // 4) + 'f', raw)
    geometry_rows = []
    for row in rows:
        edit = row['edit']
        if not edit:
            row['geometry'] = {'status': 'NO_INTERNAL_EDIT', 'realized_displacement': None}
            continue
        kind = 'raw' if row['kind'] in {'internal', 'sham', 'reverse'} else row['kind']
        vector, target = vectors[(kind, edit['layer'])], vectors[('raw', edit['layer'])]
        norm, target_norm = math.sqrt(dot(vector, vector)), math.sqrt(dot(target, target))
        cosine = dot(vector, target) / (norm * target_norm)
        receipt = receipts['receipts'][row['trial_id']]
        measured = [r['realized_l2'] for r in receipt['observations']]
        if not measured or not all(math.isfinite(v) and v >= 0 for v in measured):
            raise legacy.AnalysisError('Invalid measured displacement')
        data = {'status': 'SAVED_VECTOR_AND_MEASURED_RECEIPT', 'direction_norm': norm,
            'cosine_with_intended': cosine,
            'intended_displacement_l2': abs(edit['coefficient']) * norm,
            'intended_projection_on_target': edit['coefficient'] * dot(vector, target) / target_norm,
            'realized_displacement_l2_min': min(measured), 'realized_displacement_l2_max': max(measured),
            'realized_displacement_observations': len(measured),
            'realized_displacement': measured,
            'realized_displacement_source': receipt['source_path_in_delivery'],
            'realized_displacement_source_sha256': receipt['source_sha256'],
            'receipt_summary_sha256': RECEIPTS_SHA,
            'normalization_rule': 'Saved unnormalized direction; controls norm matched within layer',
            'scope': 'Intended geometry uses float32 vectors. Measured per-application L2 comes from saved BF16 execution observations; it is not a cumulative trajectory effect.'}
        row['geometry'] = data
        geometry_rows.append({'arm_id': row['arm_id'], 'layer': edit['layer'], 'kind': row['kind'],
            'strength': edit['coefficient'], 'valid': row['score']['valid'],
            'decision': row['score']['decision'] if row['score']['valid'] else 'INVALID', **data})
    return {'analysis_status': 'POST_HOC_ONE_CASE_DESCRIPTION', 'request_sha256': REQUEST_SHA,
        'receipt_summary_sha256': RECEIPTS_SHA,
        'rows': sorted(geometry_rows, key=lambda r: (r['layer'], r['intended_projection_on_target'], r['arm_id'])),
        'interpretation': 'Permuted controls retain a component along the intended direction. Their changes are compatible with a direction-related decision effect, not proof of a cooperation construct. Invalid answers are separate outcomes, not an ordinal midpoint. No new selection or significance test.'}
