"""Freeze outcome-blind nested draws and a finite set of physical controls."""
import copy
import json
import h5py
import numpy as np
import extension as c


def main():
    assert not (c.ROOT/'cases.h5').exists()
    c.design.ready()
    data, population, _ = c.sample_data()
    rng = np.random.default_rng(2026110257503)
    ids, allocation = c.nested_draw(data['stratum'], 8192, rng)
    local, physical_allocation = c.nested_draw(data['stratum'][ids], 4096, rng)
    physical = ids[local]
    full, slug = data['actual'], data['slug']
    states = (full, slug, c.cross(full, slug), c.cross(slug, full))
    raw, sample, cell, scenario, branch, pi = [], [], [], [], [], []

    def add(indices, values, ce, sc, br, alloc):
        raw.append(values[indices]); sample.extend(indices)
        cell.extend([ce]*len(indices)); scenario.extend([sc]*len(indices)); branch.extend([br]*len(indices))
        pi.extend(alloc[data['stratum'][indices]]/population[data['stratum'][indices]])

    # Four factorial corners share both the draw and numerical protocol.
    for ce, values in enumerate(states):
        add(ids, values, ce, 0, 0, allocation)
    add(physical, full, 4, 0, 1, physical_allocation)
    for sc, (_, uv, cr, oxygen) in enumerate(c.SCENARIOS[1:], 1):
        for ce, br, values in ((0, 0, full), (1, 0, slug), (4, 1, full)):
            values = values.copy(); values[:, 5:7] *= oxygen
            add(physical, values, ce, sc, br, physical_allocation)
    raw = np.concatenate(raw)
    assert len(raw) == 110592
    with h5py.File(c.ROOT/'cases.h5', 'x') as f:
        for key, value in dict(raw=raw, sample_id=sample, cell=cell, scenario=scenario,
                               branch=branch, inclusion_probability=pi, factorial_ids=ids,
                               physical_ids=physical, factorial_allocation=allocation,
                               physical_allocation=physical_allocation).items():
            f.create_dataset(key, data=np.asarray(value))
    # Each epoch has its own context and outputs; frozen parent paths are never written.
    original = json.loads((c.ISO/'highres_snapshot575_temporal/config.json').read_text())
    for entry in json.loads((c.ROOT/'inventory.json').read_text()):
        snap = entry['snapshot']; root = c.ROOT/f'time/{snap}'
        root.mkdir(parents=True, exist_ok=True)
        settings = copy.deepcopy(original)
        settings['run_name'] = f'x8_time_sequence_{snap}'
        settings['snapshot']['index'] = snap
        # Use inventory as the authority for the data root.
        settings['snapshot']['file_pattern'] = str(c.Path(entry['files'][0]['path']).parent/f'snapshot_{snap}.*.hdf5')
        settings['outputs'] = {key: str(root/('context.npz' if key == 'context_npz' else key))
                               for key in original['outputs']}
        c.save(root/'config.json', settings)
    c.save(c.ROOT/'design.json', dict(seed=2026110257503, factorial_particles=len(ids),
           physical_particles=len(physical), cases=len(raw), cells=['FF','SS','FS','SF','Full_off'],
           scenarios=c.SCENARIOS, nested_probabilities='final n_h / original population N_h; SRSWOR within frozen strata',
           solver=dict(tEqGuess=None, smallabd=1e-15, relative_tolerance=.001,
                       maxTime=1e18, mxstep=5000, wall_seconds=600, damping=[.25,.1,.03]),
           selection_uses_direct_outcomes=False, snapshots=list(range(551,576))))
    print('Prepared 110592 direct controls and 25 epoch configurations; no solver called.')


if __name__ == '__main__':
    main()
