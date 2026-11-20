"""Degradation labels parsed from file names (free side information, no MOS): distortion type id and up to
P_MAX numeric parameters (QP, level, sigma, ...), each rank-normalised to [0, 1] within its type.

  SJTU    obj_i.ply                    type = i // 6 (7 types), level = i % 6
  WPC     obj_level_7 | obj_gQP_1_tQP_2 | obj_gsigma_2_tsigma_8 | obj_pqs_1_qs_16_rounded | obj_tsl_4_tqs_64
  BASICS  p01_geocnn_r03.ply           type = codec, level = rate index
  LS      obj_<Type>_<level>.ply       type = distortion name, level

Used by the auxiliary degradation head (--aux_deg): the head regresses the parameters of the true type and
classifies the type, so the shared CLIP features have to order versions of the same content by severity.
"""
import re
import numpy as np

P_MAX = 2
_WPC = re.compile(r'_(level_\d+|gQP_\d+_tQP_\d+|gsigma_\d+_tsigma_\d+|pqs_\d+_qs_\d+_rounded|tsl_\d+_tqs_\d+)\.ply$')


def _parse(db, name):
    """-> (type string, list of raw numbers)"""
    if db == 'MIX':  # WPC + BASICS names (csvfiles/mix_data_info); their type names do not collide
        return _parse('WPC' if _WPC.search(name) else 'BASICS', name)
    if db == 'SJTU':
        i = int(name.rsplit('_', 1)[1].split('.')[0])
        return f't{i // 6}', [i % 6]
    if db == 'WPC':
        part = _WPC.search(name).group(1)
        nums = [int(x) for x in re.findall(r'\d+', part)]
        key = part.split('_')[0]
        return key, nums[1:] if key == 'pqs' else nums      # pqs is always 1 -> keep qs only
    if db == 'BASICS':
        _, codec, r = name[:-4].rsplit('_', 2)
        return codec, [int(r[1:])]
    t, lv = name[:-4].rsplit('_', 2)[-2:]
    return t, [int(lv)]


def build(db, names):
    """names -> dict name -> (type_id, params[P_MAX] with NaN where absent), plus the type list."""
    parsed = {n: _parse(db, n) for n in names}
    types = sorted({t for t, _ in parsed.values()})
    ranks = {}
    for t in types:
        vals = [p for tt, p in parsed.values() if tt == t]
        for j in range(max(len(v) for v in vals)):
            u = sorted({v[j] for v in vals if len(v) > j})
            ranks[(t, j)] = {x: (k / (len(u) - 1) if len(u) > 1 else 0.0) for k, x in enumerate(u)}
    out = {}
    for n, (t, p) in parsed.items():
        v = np.full(P_MAX, np.nan, np.float32)
        for j, x in enumerate(p[:P_MAX]):
            v[j] = ranks[(t, j)][x]
        out[n] = (types.index(t), v)
    return out, types


# ---- dataset-agnostic degradation factors (--aux_factor) -----------------------------------------------------------
# Instead of each database's own type ids, every distortion type of every database (and the synthetic codec samples)
# is described by the same multi-hot vector of physical factors, plus one severity in [0, 1] (0 = mildest level of its
# type, 1 = strongest), so the head learns what a degradation does rather than which dataset category it belongs to.
# Colour-only vs geometry types of LS-PCQA follow its paper (Table 4: geometry metrics undefined = colour only).
FACTORS = ['geom_noise', 'sparse', 'geom_quant', 'smooth', 'local_deform', 'colour_noise', 'colour_quant', 'colour_shift']
_F = {n: i for i, n in enumerate(FACTORS)}
_TYPE_FACTORS = {
    # SJTU-PCQA (readme): t0 octree pruning, t1 colour noise, t2 downscaling, t3 D+C, t4 D+G, t5 geometry Gaussian noise, t6 C+G
    ('SJTU', 't0'): ['sparse', 'geom_quant'], ('SJTU', 't1'): ['colour_noise'], ('SJTU', 't2'): ['sparse'],
    ('SJTU', 't3'): ['sparse', 'colour_noise'], ('SJTU', 't4'): ['sparse', 'geom_noise'], ('SJTU', 't5'): ['geom_noise'],
    ('SJTU', 't6'): ['colour_noise', 'geom_noise'],
    # WPC: downsampling, Gaussian noise on geometry and colour, V-PCC, G-PCC octree (lifting), G-PCC trisoup
    ('WPC', 'level'): ['sparse'], ('WPC', 'gsigma'): ['geom_noise', 'colour_noise'], ('WPC', 'gQP'): ['geom_quant', 'colour_quant'],
    ('WPC', 'pqs'): ['geom_quant', 'sparse', 'colour_quant'], ('WPC', 'tsl'): ['smooth', 'colour_quant'],
    # BASICS: learned geometry codec, G-PCC octree with predlift / RAHT attributes, V-PCC
    ('BASICS', 'geocnn'): ['smooth', 'colour_quant'], ('BASICS', 'gpcc-octree-predlift'): ['geom_quant', 'sparse', 'colour_quant'],
    ('BASICS', 'gpcc-octree-raht'): ['geom_quant', 'sparse', 'colour_quant'], ('BASICS', 'vpcc'): ['geom_quant', 'colour_quant'],
    # synthetic G-PCC (synth_codec/encode_all.sh): octree+RAHT, octree+lifting, trisoup+RAHT, lossless geometry + RAHT
    ('SYNTH', 'otr'): ['geom_quant', 'sparse', 'colour_quant'], ('SYNTH', 'otl'): ['geom_quant', 'sparse', 'colour_quant'],
    ('SYNTH', 'tri'): ['smooth', 'colour_quant'], ('SYNTH', 'atr'): ['colour_quant'],
    # generic synthetic bank (synth_codec/synth_more.py)
    ('SYNTH', 'cgn'): ['colour_noise'], ('SYNTH', 'csp'): ['colour_noise'], ('SYNTH', 'clm'): ['colour_noise'],
    ('SYNTH', 'cct'): ['colour_shift'], ('SYNTH', 'csat'): ['colour_shift'], ('SYNTH', 'cms'): ['colour_shift'],
    ('SYNTH', 'cq'): ['colour_quant'], ('SYNTH', 'lmiss'): ['local_deform'], ('SYNTH', 'loff'): ['local_deform'],
    ('SYNTH', 'lrot'): ['local_deform'], ('SYNTH', 'gshift'): ['geom_noise'],
    ('SYNTH', 'vpc'): ['geom_quant', 'colour_quant'],   # MPEG V-PCC (TMC2 v25, CTC all-intra r5..r1)
    ('SYNTH', 'ds'): ['sparse'],                        # random downsampling (synth_more.py)
}
_LS_COLOUR_NOISE = ['ColorNoise', 'CorrelatedGaussianNoise', 'GammaNoise', 'GaussianNoise', 'HighFrequencyNoise', 'LumaNoise',
                    'MultiplicativeGaussianNoise', 'PoissonNoise', 'RayleighNoise', 'SaltpepperNoise', 'UniformNoise']
_TYPE_FACTORS.update({('LS', t): ['colour_noise'] for t in _LS_COLOUR_NOISE})
_TYPE_FACTORS.update({
    ('LS', 'ColorQuantizationDither'): ['colour_quant'], ('LS', 'QuantizationNoise'): ['colour_quant'],
    ('LS', 'ContrastDistortion'): ['colour_shift'], ('LS', 'MeanShift'): ['colour_shift'], ('LS', 'SaturationDistortion'): ['colour_shift'],
    ('LS', 'DownSample'): ['sparse'], ('LS', 'GaussianShifting'): ['geom_noise'], ('LS', 'UniformShifting'): ['geom_noise'],
    ('LS', 'LocalLoss'): ['local_deform'], ('LS', 'LocalOffset'): ['local_deform'], ('LS', 'LocalRotation'): ['local_deform'],
    ('LS', 'Octree'): ['geom_quant', 'sparse'], ('LS', 'reconstruction'): ['smooth'],
    ('LS', 'C2AI-lossy-geom-lossy-attrs'): ['geom_quant', 'colour_quant'],                       # V-PCC
    ('LS', 'lossy-geom-lossy-attrs'): ['geom_quant', 'sparse', 'colour_quant'],                   # G-PCC
    ('LS', 'lossless-geom-lossy-attrs'): ['colour_quant'], ('LS', 'lossless-geom-nearlossless-attrs'): ['colour_quant'],
    ('LS', 'limitlossyG-lossyA'): ['geom_quant', 'colour_quant'],                                 # AVS
    ('LS', 'losslessG-limitlossyA'): ['colour_quant'], ('LS', 'losslessG-lossyA'): ['colour_quant'],
})
# parameter directions: +1 = a larger number is a stronger distortion (checked against the MOS: WPC 'level' and the
# BASICS rate index r00..r06 grow with quality, everything else with distortion)
_REVERSED = {('WPC', 'level'), ('BASICS', 'geocnn'), ('BASICS', 'gpcc-octree-predlift'), ('BASICS', 'gpcc-octree-raht'), ('BASICS', 'vpcc')}


def _family(db):
    return {'SJTU': 'SJTU', 'WPC': 'WPC', 'BASICS': 'BASICS', 'SYNTH': 'SYNTH'}.get(db, 'LS')


def _parse_any(db, name):
    if db == 'SYNTH':  # <ref>_<code>_L<level>.ply
        _, code, lv = name[:-4].rsplit('_', 2)
        return code, [int(lv[1:])]
    return _parse(db, name)


def build_factors(db, names):
    """names -> dict name -> (severity float32 in [0, 1], multi-hot float32[len(FACTORS)]). Severity = mean over the
    type's parameters of the parameter's rank within the type (normalised, oriented so that 1 = strongest)."""
    fam = _family(db)
    parsed = {n: _parse_any(db, n) for n in names}
    out = {}
    for t in sorted({t for t, _ in parsed.values()}):
        assert (fam, t) in _TYPE_FACTORS, f'no factor mapping for {fam} type {t}'
        members = [n for n, (tt, _) in parsed.items() if tt == t]
        npar = max(len(parsed[n][1]) for n in members)
        sev = np.zeros(len(members))
        for j in range(npar):
            u = sorted({parsed[n][1][j] for n in members if len(parsed[n][1]) > j})
            r = {x: (k / (len(u) - 1) if len(u) > 1 else 0.0) for k, x in enumerate(u)}
            sev += [r[parsed[n][1][j]] if len(parsed[n][1]) > j else 0.0 for n in members]
        sev /= npar
        if (fam, t) in _REVERSED:
            sev = 1 - sev
        vec = np.zeros(len(FACTORS), np.float32)
        vec[[_F[f] for f in _TYPE_FACTORS[(fam, t)]]] = 1
        for n, s in zip(members, sev):
            out[n] = (np.float32(s), vec)
    return out
