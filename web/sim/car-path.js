// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * CarPath — a single parameterized curve that a vehicle follows continuously.
 *
 * Replaces the old edge-based positioning (u coordinate along road edges)
 * with a unified path of straight and turn segments. The car has one
 * parameter `d` (distance traveled in meters) and derives position and
 * heading from the path. No state transitions, no "turning mode."
 *
 * Reference: docs/plans/vehicle-framework-rewrite.md
 *
 * Segment types:
 *   StraightSegment — linear from start to end
 *   TurnSegment — quadratic Bezier (entry, control, exit)
 *
 * Pure data — no Three.js dependency.
 */

// ── Segment Types ─────────────────────────────────────────────────

/**
 * StraightSegment — linear path between two points.
 */
export class StraightSegment {
    /**
     * @param {{ x: number, z: number }} from — start point
     * @param {{ x: number, z: number }} to — end point
     */
    constructor(from, to) {
        this.type = 'straight';
        this.from = { x: from.x, z: from.z };
        this.to = { x: to.x, z: to.z };
        this.length = Math.sqrt(
            (to.x - from.x) ** 2 + (to.z - from.z) ** 2
        );
    }

    /**
     * Get world position at local parameter t in [0, 1].
     * @param {number} t — normalized position (0 = start, 1 = end)
     * @returns {{ x: number, z: number }}
     */
    getPositionAt(t) {
        return {
            x: this.from.x + (this.to.x - this.from.x) * t,
            z: this.from.z + (this.to.z - this.from.z) * t,
        };
    }

    /**
     * Get heading angle (radians) at local parameter t.
     * For a straight segment, heading is constant.
     * Convention: 0 = East (+x), PI/2 = South (+z), PI = West, -PI/2 = North
     * @param {number} _t — unused (heading is constant for straights)
     * @returns {number} angle in radians
     */
    getHeadingAt(_t) {
        return Math.atan2(this.to.z - this.from.z, this.to.x - this.from.x);
    }

    /**
     * Get the endpoint of this segment.
     * @returns {{ x: number, z: number }}
     */
    getEndPoint() {
        return { x: this.to.x, z: this.to.z };
    }

    /**
     * Get the start point of this segment.
     * @returns {{ x: number, z: number }}
     */
    getStartPoint() {
        return { x: this.from.x, z: this.from.z };
    }
}

/**
 * TurnSegment — quadratic Bezier curve through an intersection.
 *
 * B(t) = (1-t)^2 * P0 + 2*(1-t)*t * P1 + t^2 * P2
 *
 * Where P0 = entry, P1 = control, P2 = exit.
 * Tangent at entry is toward control, tangent at exit is from control.
 */
export class TurnSegment {
    /**
     * @param {{ x: number, z: number }} entry — curve start (P0)
     * @param {{ x: number, z: number }} control — control point (P1)
     * @param {{ x: number, z: number }} exit — curve end (P2)
     */
    constructor(entry, control, exit) {
        this.type = 'turn';
        this.entry = { x: entry.x, z: entry.z };
        this.control = { x: control.x, z: control.z };
        this.exit = { x: exit.x, z: exit.z };
        this.length = this._approximateArcLength(32);
    }

    /**
     * Approximate arc length by sampling the Bezier and summing chord distances.
     * @param {number} samples — number of subdivisions (higher = more accurate)
     * @returns {number} approximate arc length in meters
     */
    _approximateArcLength(samples) {
        let length = 0;
        let prev = this._bezier(0);
        for (let i = 1; i <= samples; i++) {
            const t = i / samples;
            const curr = this._bezier(t);
            length += Math.sqrt((curr.x - prev.x) ** 2 + (curr.z - prev.z) ** 2);
            prev = curr;
        }
        return length;
    }

    /**
     * Evaluate the quadratic Bezier at parameter t.
     * @param {number} t — parameter in [0, 1]
     * @returns {{ x: number, z: number }}
     */
    _bezier(t) {
        const mt = 1 - t;
        return {
            x: mt * mt * this.entry.x + 2 * mt * t * this.control.x + t * t * this.exit.x,
            z: mt * mt * this.entry.z + 2 * mt * t * this.control.z + t * t * this.exit.z,
        };
    }

    /**
     * Evaluate the Bezier tangent (first derivative) at parameter t.
     * B'(t) = 2*(1-t)*(P1 - P0) + 2*t*(P2 - P1)
     * @param {number} t — parameter in [0, 1]
     * @returns {{ x: number, z: number }}
     */
    _bezierTangent(t) {
        const mt = 1 - t;
        return {
            x: 2 * mt * (this.control.x - this.entry.x) + 2 * t * (this.exit.x - this.control.x),
            z: 2 * mt * (this.control.z - this.entry.z) + 2 * t * (this.exit.z - this.control.z),
        };
    }

    /**
     * Get world position at local parameter t in [0, 1].
     * @param {number} t — normalized position
     * @returns {{ x: number, z: number }}
     */
    getPositionAt(t) {
        return this._bezier(t);
    }

    /**
     * Get heading angle (radians) at local parameter t.
     * @param {number} t — normalized position
     * @returns {number} angle in radians
     */
    getHeadingAt(t) {
        const tan = this._bezierTangent(t);
        return Math.atan2(tan.z, tan.x);
    }

    /**
     * Get the endpoint of this segment.
     * @returns {{ x: number, z: number }}
     */
    getEndPoint() {
        return { x: this.exit.x, z: this.exit.z };
    }

    /**
     * Get the start point of this segment.
     * @returns {{ x: number, z: number }}
     */
    getStartPoint() {
        return { x: this.entry.x, z: this.entry.z };
    }
}

// ── CarPath ───────────────────────────────────────────────────────

/**
 * CarPath — a vehicle's complete path as a sequence of segments.
 *
 * The car has a single distance parameter `d` that advances each frame.
 * Position and heading are derived from the path at `d`. No state machine,
 * no edge transitions — just one continuous curve.
 */
export class CarPath {
    constructor() {
        /** @type {Array<StraightSegment|TurnSegment>} */
        this.segments = [];
        /** Total path length in meters. */
        this.totalLength = 0;
        /**
         * Cumulative distances: _cumDist[i] is the distance from path
         * start to the START of segments[i]. This makes getSegmentAt O(log n).
         * @type {number[]}
         */
        this._cumDist = [];
    }

    /**
     * Append a straight segment.
     * @param {{ x: number, z: number }} from
     * @param {{ x: number, z: number }} to
     * @returns {StraightSegment} the segment that was added
     */
    addStraight(from, to) {
        const seg = new StraightSegment(from, to);
        this._cumDist.push(this.totalLength);
        this.segments.push(seg);
        this.totalLength += seg.length;
        return seg;
    }

    /**
     * Append a Bezier turn segment.
     * @param {{ x: number, z: number }} entry
     * @param {{ x: number, z: number }} control
     * @param {{ x: number, z: number }} exit
     * @returns {TurnSegment} the segment that was added
     */
    addTurn(entry, control, exit) {
        const seg = new TurnSegment(entry, control, exit);
        this._cumDist.push(this.totalLength);
        this.segments.push(seg);
        this.totalLength += seg.length;
        return seg;
    }

    /**
     * Find which segment contains distance `d`, plus the local parameter.
     *
     * @param {number} d — distance along path (meters from start)
     * @returns {{ segment: StraightSegment|TurnSegment, index: number, t: number, localD: number }|null}
     *   segment — the containing segment
     *   index — index in segments array
     *   t — normalized parameter within the segment [0, 1]
     *   localD — distance within the segment
     */
    getSegmentAt(d) {
        if (this.segments.length === 0) return null;

        // Clamp d to valid range
        const dc = Math.max(0, Math.min(d, this.totalLength));

        // Binary search for the segment containing dc
        let lo = 0, hi = this.segments.length - 1;
        while (lo < hi) {
            const mid = (lo + hi + 1) >> 1;
            if (this._cumDist[mid] <= dc) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }

        const seg = this.segments[lo];
        const localD = dc - this._cumDist[lo];
        const t = seg.length > 0 ? Math.min(1, localD / seg.length) : 0;

        return { segment: seg, index: lo, t, localD };
    }

    /**
     * Get world position at distance `d` along the path.
     * @param {number} d — distance in meters
     * @returns {{ x: number, z: number }|null}
     */
    getPosition(d) {
        const info = this.getSegmentAt(d);
        if (!info) return null;
        return info.segment.getPositionAt(info.t);
    }

    /**
     * Get heading angle at distance `d` along the path.
     * Convention: 0 = East (+x), PI/2 = South (+z)
     * @param {number} d — distance in meters
     * @returns {number|null} angle in radians
     */
    getHeading(d) {
        const info = this.getSegmentAt(d);
        if (!info) return null;
        return info.segment.getHeadingAt(info.t);
    }

    /**
     * Check if distance `d` is within a TurnSegment.
     * @param {number} d — distance in meters
     * @returns {boolean}
     */
    isInTurn(d) {
        const info = this.getSegmentAt(d);
        if (!info) return false;
        return info.segment.type === 'turn';
    }

    /**
     * Remove all segments that end entirely before distance `d`.
     * Used for garbage collection — the car never goes backwards.
     *
     * After trimming, `d` values for the remaining segments stay valid
     * because we don't re-index. The car's `d` parameter is absolute.
     *
     * @param {number} d — remove segments whose end is before this distance
     * @returns {number} number of segments removed
     */
    trimBefore(d) {
        let removed = 0;
        while (this.segments.length > 1) {
            const segEnd = this._cumDist[0] + this.segments[0].length;
            if (segEnd < d) {
                this.segments.shift();
                this._cumDist.shift();
                removed++;
            } else {
                break;
            }
        }
        return removed;
    }

    /**
     * Get the last point on the path.
     * @returns {{ x: number, z: number }|null}
     */
    getEndPoint() {
        if (this.segments.length === 0) return null;
        return this.segments[this.segments.length - 1].getEndPoint();
    }

    /**
     * Get the first point on the path.
     * @returns {{ x: number, z: number }|null}
     */
    getStartPoint() {
        if (this.segments.length === 0) return null;
        return this.segments[0].getStartPoint();
    }

    /**
     * Get remaining distance from `d` to end of path.
     * @param {number} d — current distance
     * @returns {number}
     */
    remainingDistance(d) {
        return Math.max(0, this.totalLength - d);
    }

    /**
     * Sample positions along the path for visualization.
     * @param {number} startD — start distance
     * @param {number} endD — end distance
     * @param {number} step — meters between samples
     * @returns {Array<{ x: number, z: number }>}
     */
    samplePositions(startD, endD, step = 5) {
        const points = [];
        const s = Math.max(0, startD);
        const e = Math.min(this.totalLength, endD);
        for (let d = s; d <= e; d += step) {
            const pos = this.getPosition(d);
            if (pos) points.push(pos);
        }
        // Always include the exact end point
        if (e > s) {
            const last = this.getPosition(e);
            if (last) {
                const prev = points[points.length - 1];
                if (!prev || Math.abs(prev.x - last.x) > 0.01 || Math.abs(prev.z - last.z) > 0.01) {
                    points.push(last);
                }
            }
        }
        return points;
    }

    /**
     * Build a CarPath from a route (sequence of intersection IDs).
     *
     * The route is a list of intersection node IDs from a pathfinding algorithm.
     * This method constructs alternating turn and straight segments between them,
     * offset by `laneOffset` from the road centerline.
     *
     * @param {Array<string>} route — sequence of intersection node IDs
     * @param {Object} roadNetwork — RoadNetwork instance with nodes and edges
     * @param {number} [laneOffset=2] — lateral offset from road centerline (meters)
     * @returns {CarPath}
     */
    static fromRoute(route, roadNetwork, laneOffset = 2) {
        const path = new CarPath();
        if (!route || route.length < 2) return path;
        if (!roadNetwork || !roadNetwork.nodes) return path;

        // Compute entry/exit points for each segment between intersections
        for (let i = 0; i < route.length - 1; i++) {
            const fromNode = roadNetwork.nodes[route[i]];
            const toNode = roadNetwork.nodes[route[i + 1]];
            if (!fromNode || !toNode) continue;

            // Direction vector from -> to
            const dx = toNode.x - fromNode.x;
            const dz = toNode.z - fromNode.z;
            const dist = Math.sqrt(dx * dx + dz * dz);
            if (dist < 0.01) continue;

            // Unit direction and perpendicular (for lane offset)
            const ux = dx / dist;
            const uz = dz / dist;
            // Perpendicular: rotate 90 degrees clockwise (right-hand offset)
            const px = uz;
            const pz = -ux;

            // Lane-offset entry and exit points
            const entryX = fromNode.x + px * laneOffset;
            const entryZ = fromNode.z + pz * laneOffset;
            const exitX = toNode.x + px * laneOffset;
            const exitZ = toNode.z + pz * laneOffset;

            // For the first segment, just add a straight
            if (i === 0) {
                path.addStraight(
                    { x: entryX, z: entryZ },
                    { x: exitX, z: exitZ }
                );
                continue;
            }

            // For subsequent segments, add a turn at the intersection,
            // then a straight to the next intersection
            const pathEnd = path.getEndPoint();
            const turnExit = { x: entryX, z: entryZ };

            // Compute turn control point based on direction change
            const control = _computeTurnControl(pathEnd, turnExit);

            path.addTurn(pathEnd, control, turnExit);
            path.addStraight(turnExit, { x: exitX, z: exitZ });
        }

        return path;
    }
}

// ── Turn Control Point Helpers ────────────────────────────────────

/**
 * Compute the control point for a turn Bezier curve.
 *
 * For a grid city, the control point is the L-corner — the point where
 * extending the entry direction meets extending backwards from the exit
 * direction. For straight-through, the control is the midpoint (degenerate
 * Bezier = straight line).
 *
 * @param {{ x: number, z: number }} entry — curve entry point
 * @param {{ x: number, z: number }} exit — curve exit point
 * @returns {{ x: number, z: number }} control point
 */
function _computeTurnControl(entry, exit) {
    // Determine dominant direction change
    const dx = Math.abs(exit.x - entry.x);
    const dz = Math.abs(exit.z - entry.z);

    // If mostly collinear (straight through), use midpoint
    if (dx < 0.5 || dz < 0.5) {
        return {
            x: (entry.x + exit.x) / 2,
            z: (entry.z + exit.z) / 2,
        };
    }

    // L-corner: (exit.x, entry.z) or (entry.x, exit.z)
    // Pick the one that makes the shorter arc (the one on the inside of the turn)
    // For grid cities, we use the convention: control = (exit.x, entry.z)
    // This places the control at the corner of the L, giving smooth 90-degree turns
    return { x: exit.x, z: entry.z };
}

/**
 * Compute a turn control point with explicit incoming and outgoing directions.
 * Useful when you know the heading before and after the turn.
 *
 * @param {{ x: number, z: number }} entry — curve entry point
 * @param {{ x: number, z: number }} exit — curve exit point
 * @param {number} entryHeading — heading in radians at entry
 * @param {number} exitHeading — heading in radians at exit
 * @returns {{ x: number, z: number }} control point
 */
export function computeTurnControlFromHeadings(entry, exit, entryHeading, exitHeading) {
    // Extend entry direction and exit direction, find intersection
    // Entry line: entry + t * (cos(entryHeading), sin(entryHeading))
    // Exit line: exit - s * (cos(exitHeading), sin(exitHeading))
    // Solve for intersection

    const ce = Math.cos(entryHeading);
    const se = Math.sin(entryHeading);
    const cx = Math.cos(exitHeading);
    const sx = Math.sin(exitHeading);

    // Solve:  entry + t * (ce, se) = exit - s * (cx, sx)
    // entry.x + t*ce = exit.x - s*cx
    // entry.z + t*se = exit.z - s*sx
    const det = ce * (-sx) - se * (-cx);

    if (Math.abs(det) < 1e-6) {
        // Parallel headings (straight through) — use midpoint
        return {
            x: (entry.x + exit.x) / 2,
            z: (entry.z + exit.z) / 2,
        };
    }

    const t = ((exit.x - entry.x) * (-sx) - (exit.z - entry.z) * (-cx)) / det;

    return {
        x: entry.x + t * ce,
        z: entry.z + t * se,
    };
}
