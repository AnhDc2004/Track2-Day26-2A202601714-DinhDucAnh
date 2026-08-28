// kit/arena_ui/tests/camera.test.js
//
// COLOSSEUM — the featured camera must point at a duel where something is
// HAPPENING.
//
// The original score was `closeness = 100 - |hpA - hpB|`, which an untouched
// 100-100 duel maximises. "No claim, no damage" makes scoreless the default
// (44% of rounds in the reference bracket), so the camera settled on duels in
// which nothing had been landed and, needing a 10-point margin to switch,
// stayed there. The room watched two idle agents while a real fight ran in
// another pane.
//
// Run with `node kit/arena_ui/tests/camera.test.js`.

import { scoreDuel } from '../core/camera.js';

let failures = 0;
function check(name, cond) {
  if (cond) { console.log(`  ok   ${name}`); return; }
  console.log(`  FAIL ${name}`); failures++;
}

const untouched = scoreDuel({ hpA: 100, hpB: 100, hitAgeMs: null });
const oneHit    = scoreDuel({ hpA: 100, hpB: 88,  hitAgeMs: null });
const oneSided  = scoreDuel({ hpA: 100, hpB: 40,  hitAgeMs: null });
const bloodbath = scoreDuel({ hpA: 60,  hpB: 58,  hitAgeMs: null });
const justHit   = scoreDuel({ hpA: 100, hpB: 88,  hitAgeMs: 0 });

console.log('camera.scoreDuel');
check('a duel where a hit landed beats an untouched one', oneHit > untouched);
check('one landed hit clears the 10-point switch margin', oneHit - untouched >= 10);
check('a bloodbath outranks a one-sided beatdown', bloodbath > oneSided);
check('a one-sided beatdown still outranks a single hit', oneSided > oneHit);
check('a hit landing RIGHT NOW lifts a duel far above its resting score',
      justHit - oneHit >= 30);
check('a hit that landed long ago no longer counts',
      scoreDuel({ hpA: 100, hpB: 88, hitAgeMs: 60000 }) === oneHit);
check('an untouched duel is never the top pick', 
      [oneHit, oneSided, bloodbath, justHit].every((s) => s > untouched));

// A 0-100 blowout must not outrank a close fight forever -- that is what the
// damage cap is for.
const blowout = scoreDuel({ hpA: 100, hpB: 0, hitAgeMs: null });
check('the damage cap keeps a blowout under a close bloodbath', bloodbath > blowout);

// Degenerate inputs must not produce NaN and silently poison the comparison.
check('missing hp defaults rather than NaN', Number.isFinite(scoreDuel({})));
check('null argument does not throw', Number.isFinite(scoreDuel(null)));

console.log(failures === 0 ? 'camera: all passed' : `camera: ${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
