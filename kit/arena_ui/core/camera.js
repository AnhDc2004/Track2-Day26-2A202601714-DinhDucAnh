// kit/arena_ui/core/camera.js
//
// COLOSSEUM — which duel the projector points at.
//
// Extracted from projector.src.html so it can be tested. It was inline, and
// inline is why nobody noticed that it preferred duels where NOTHING WAS
// HAPPENING: the score was `closeness = 100 - |hpA - hpB|`, which a duel
// sitting untouched at 100-100 maximises. In a format where "no claim, no
// damage" makes a scoreless round the default -- 44% of rounds in the
// reference bracket -- that is not a rare edge case, it is the common one.
// The camera would settle on a duel in which nothing had been landed and,
// because switching needs a 10-point margin, stay there.

/** Score one duel for the featured camera. Higher wins.
 *
 *  Ordering this is built to produce, worst to best:
 *
 *    100-100 untouched   nothing has happened            30.0
 *    100- 88 one hit     something just happened         40.8
 *    100- 40 one-sided   a lot has happened              84.0
 *     60- 58 bloodbath   a lot, and it is still close   101.4
 *
 *  `damage` leads because "has anything happened here" is the first question;
 *  `closeness` only separates duels that have both drawn blood. The single-hit
 *  case clears the caller's 10-point switch margin against the untouched case
 *  ON PURPOSE (40.8 - 30.0 = 10.8): one landed hit is enough to take the
 *  camera off a duel where nobody has scored.
 *
 *  @param {{hpA:number, hpB:number, hitAgeMs:(number|null)}} d
 *    hitAgeMs — ms since this duel last took its biggest hit, or null/negative
 *    if it never has. Recent hits dominate everything else for 6s so the room
 *    sees the blow land rather than reading about it afterwards.
 */
export function scoreDuel(d) {
  const hpA = Number.isFinite(d && d.hpA) ? d.hpA : 100;
  const hpB = Number.isFinite(d && d.hpB) ? d.hpB : 100;
  const damage = Math.max(0, 100 - hpA) + Math.max(0, 100 - hpB);
  const closeness = 100 - Math.abs(hpA - hpB);
  const age = d && d.hitAgeMs;
  const fresh = Number.isFinite(age) && age >= 0 && age < HIT_WINDOW_MS
    ? HIT_BONUS * (1 - age / HIT_WINDOW_MS)
    : 0;
  return Math.min(damage, DAMAGE_CAP) * DAMAGE_WEIGHT + closeness * CLOSENESS_WEIGHT + fresh;
}

//: Past this much total damage a duel is already worth watching; more does not
//: make it more so, and uncapped damage would let a 0-100 blowout outrank a
//: close fight forever.
export const DAMAGE_CAP = 60;
export const DAMAGE_WEIGHT = 1.2;
export const CLOSENESS_WEIGHT = 0.3;
export const HIT_BONUS = 40;
export const HIT_WINDOW_MS = 6000;
