# Method notes

Detail that would crowd the README, and the gotcha list — things that cost real
time to discover and are worth not rediscovering.

---

## Why derive the mode at all

Because the field that claims to answer it does not. Mode is set at load build by
whoever is typing, and nothing downstream ever checks it. It is not noisy around
the right answer; it is wrong in one direction, nearly everywhere.

The cost of getting it wrong is asymmetric. Truckload is a minority of loads and
the large majority of carrier spend, so a mode error misroutes most of the money
while looking like a small percentage of rows.

### The anchors

| Anchor | Basis | Why it is trustworthy |
|---|---|---|
| LTL common carrier | carrier name matches a terminal-network operator | Those companies do not haul full truckloads for a brokerage |
| Truckload equipment | flatbed, step deck, lowboy, Conestoga, tanker, power only | No LTL shipment moves on this equipment |

Anchors override the fitted boundary in `classify()`. They are evidence; the
boundary is inference, and inference does not get to overrule evidence.

The carrier patterns are anchored on word boundaries. Substring matching would
have `SAIA` match `ISAIAH TRANSPORT` and quietly delete a truckload carrier's
entire volume from the benchmark. There is a test for that specific case.

### The boundary

Per distance band, take the LTL 90th percentile and the truckload 10th percentile
from the anchor sets. The cut runs through the corridor between them, at the
**geometric** midpoint, fitted to a line across bands.

Geometric because cost is roughly log-distributed. The arithmetic mean of a $600
LTL ceiling and a $2,400 truckload floor is $1,500; the geometric mean is $1,200.
The arithmetic version sits too close to the truckload side and sweeps heavy LTL
into the benchmark, where it drags lane averages down.

Bands need at least 30 loads in *both* anchor sets to enter the fit. Thin bands
produce unstable percentiles that bend the line. If fewer than two bands qualify,
`fit_boundary()` raises rather than fitting a line to one point — failing loudly
beats a silent bad boundary that every downstream number then inherits.

### Confidence: two floors, and a load must clear both

**Absolute floor** — $2.20/mile under 250 miles, $1.60 to 500, $1.20 beyond. Below
this, a load is not plausibly a full truckload at any point in the last decade's
rate cycle.

**Relative floor** — 62% of the lane's own median rate per mile, applied only where
the lane carries at least 8 derived-truckload loads.

The relative floor exists because the absolute one has a blind spot that is
invisible until the book is concentrated. On a lane whose own market clears
$4.00/mile, a partial at $2.00/mile is obviously not a truckload, and it clears
the $1.60 absolute floor comfortably. The lanes where this happens are precisely
the high-premium rural lanes the benchmark cares most about.

The lane median is computed over the *derived-truckload* population — which is the
contaminated one. That is deliberate and it is why a median is used: it tolerates
the contamination it is being used to detect. A mean would be dragged by the same
loads it is supposed to catch.

Both floors fail toward exclusion. Holding out a genuine truckload costs a little
coverage. Letting a partial in manufactures a finding that the book buys below
market, which is a wrong answer rather than a smaller sample.

---

## Distance

No TMS export carries mileage — the system stores what the carrier invoiced, not
how far they drove. Great-circle distance between postal centroids × a circuity
factor of 1.17 is the standard stand-in.

- Local moves are floored at 25 miles. Origin equals destination on a real share
  of loads, and without a floor the rate per mile is infinite and every aggregate
  touching the lane becomes NaN. A floor, not a filter — those loads are real
  spend and belong in the totals, just not in a truckload lane benchmark.
- Canadian points reduce to the three-character forward sortation area, the finest
  granularity the free reference data carries.
- The estimator runs a few percent short of real road miles. Stage 09 measures the
  bias against the provider's own mileage and prefers the provider's number
  wherever it exists. This matters more than it sounds: a mileage error moves
  every derived rate per mile by the same proportion, in the same direction, on
  every lane at once.

---

## Lane granularity

Three-digit postal prefixes, directional.

Five-digit zips fragment one real lane into dozens of singletons, none with enough
volume to have a percentile. State-to-state buckets average across markets that
price nothing alike. Three digits is where repeat volume and market coherence
overlap.

Directional because a backhaul is a different market. Merging the two directions
averages a tight lane with a loose one and describes neither.

---

## The internal benchmark

Run before buying any external data, because it needs no subscription and cannot
be wrong about the market — it never refers to one.

A lane qualifies at 10+ loads from 3+ carriers. Both thresholds matter: ten loads
from one carrier is one negotiated rate repeated ten times, and its 25th
percentile is not evidence of anything.

Recoverable spend is computed **per load** against the lane's own p25, then summed
— not from lane averages. Averaging first lets a lane with a handful of very
expensive loads look identical to one that is uniformly slightly expensive, and
only the first is worth a phone call.

---

## Market-data gotchas

Check these every time. Each one silently corrupts an aggregate.

**Grade every lane by match rate.** Divide the provider's count of *our* reports by
our actual load count. Under roughly 60%, the comparison describes a handful of
loads while being presented with the same confidence as one built from hundreds.
A lane that matched 7 of 63 loads is unmeasured, not "12% above market".

**Providers silently aggregate to market pairs.** A lane too thin to price gets
answered with the surrounding market-to-market average, at the same granularity of
presentation and often with no flag. Two distinct lanes return identical rate,
report count, and mileage. Dedupe on the returned figures before summing, or that
market gets counted twice.

**Spot and contract are quoted differently.** Spot is published all-in; contract is
linehaul-only and needs fuel surcharge added before it can be compared to what a
carrier was actually paid. Confirm which one you are looking at.

**National averages have a distance floor.** They typically cover hauls of 250+
miles. Comparing a 150-mile lane to them measures nothing.

**Check the date on every published figure.** Rate press releases are heavily
SEO-optimised and search results routinely surface a release from the same month
of the *previous* year. Confirm the year on the page, not in the snippet.

---

## Weighting

Every headline is weighted by loads, never a mean of per-lane percentages.

On a concentrated book an unweighted mean lets a 4-load lane and a 400-load lane
vote equally, which is how a rounding error on the long tail overturns the
headline. There is a test asserting this specific behaviour, because the naive
version is genuinely tempting and reads fine.

---

## Reporting the unmeasurable

Lanes with too little repeat volume cannot be benchmarked by this method or by any
rate service. Their size gets reported, not folded into an average.

This is the part most easily skipped and it changes what the number means. An
average computed over 95% of spend and presented as covering the book carries a
precision it has not earned. Saying "these lanes are 20% of spend and cannot be
benchmarked by anything" is more useful to whoever has to act on it than a
headline that quietly includes them.

---

## Reproducibility

`carrier_lifespan()` measures "still active" against the **book's own last load
date**, not against today. An analysis that silently changes its answer as the
calendar advances cannot be re-run and checked next quarter.

The synthetic generator is seeded. Same seed, same book, same numbers.
