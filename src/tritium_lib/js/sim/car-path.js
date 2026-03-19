// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * CarPath — continuous parameterized path for vehicle movement.
 *
 * A car's entire route (current road + intersection turns + future roads) is
 * represented as ONE continuous curve. The car has a single parameter `d`
 * (distance in meters along the path) that only increases. Position (x, z)
 * and heading are derived from the path at `d`.
 *
 * The path is piecewise: alternating straight segments and quadratic Bezier
 * turn curves. There is no "turning state" — the car just advances `d` and
 * the path geometry handles everything.
 *
 * Architecture based on:
 * - volkhin/RoadTrafficSimulator trajectory.coffee (Bezier transitions)
 * - movsim/traffic-simulation-de road.js (road-to-road connect pattern)
 * - SUMO internal lanes (intersection as driveable geometry)
 *
 * See: docs/plans/vehicle-framework-rewrite.md for full design rationale.
 */

// ============================================================
// PATH SEGMENT TYPES
// ============================================================

/**
 * Straight path segment.
 */
class StraightSegment {
    /**
     * @param {{ x: number, z: number }} from - Start point
     * @param {{ x: number, z: number }} to - End point
     * @param {number} startD - Distance along total path where this segment begins
     */
    constructor(from, to, startD) {
        this.type = 'straight';
        this.from = from;
        this.to = to;
        this.startD = startD;
        this.dx = to.x - from.x;
        this.dz = to.z - from.z;
        this.length = Math.sqrt(this.dx * this.dx + this.dz * this.dz);
        // Pre-compute heading (constant for straight segments)
        this.heading = Math.atan2(this.dx, this.dz);
    }

    getPosition(localT) {
        return {
            x: this.from.x + this.dx * localT,
            z: this.from.z + this.dz * localT,
        };
    }

    getHeading(_localT) {
        return this.heading;
    }
}

/**
 * Quadratic Bezier turn segment.
 * Creates a smooth arc through an intersection.
 */
class TurnSegment {
    /**
     * @param {{ x: number, z: number }} entry - Entry point (end of incoming lane)
     * @param {{ x: number, z: number }} control - Control point (L-corner of turn)
     * @param {{ x: number, z: number }} exit - Exit point (start of outgoing lane)
     * @param {number} startD - Distance along total path where this segment begins
     */
    constructor(entry, control, exit, startD) {
        this.type = 'turn';
        this.entry = entry;
        this.control = control;
        this.exit = exit;
        this.startD = startD;
        this.length = this._approximateLength(20);
    }

    getPosition(localT) {
        const u = 1 - localT;
        return {
            x: u * u * this.entry.x + 2 * u * localT * this.control.x + localT * localT * this.exit.x,
            z: u * u * this.entry.z + 2 * u * localT * this.control.z + localT * localT * this.exit.z,
        };
    }

    getHeading(localT) {
        // Tangent of quadratic Bezier: B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1)
        const u = 1 - localT;
        const tx = 2 * u * (this.control.x - this.entry.x) + 2 * localT * (this.exit.x - this.control.x);
        const tz = 2 * u * (this.control.z - this.entry.z) + 2 * localT * (this.exit.z - this.control.z);
        return Math.atan2(tx, tz);
    }

    _approximateLength(samples) {
        let length = 0;
        let prev = this.entry;
        for (let i = 1; i <= samples; i++) {
            const t = i / samples;
            const curr = this.getPosition(t);
            const dx = curr.x - prev.x;
            const dz = curr.z - prev.z;
            length += Math.sqrt(dx * dx + dz * dz);
            prev = curr;
        }
        return length;
    }
}

// ============================================================
// CAR PATH CLASS
// ============================================================

export class CarPath {
    constructor() {
        this.segments = [];      // Array of StraightSegment | TurnSegment
        this.totalLength = 0;
        this.intersections = []; // Array of { x, z, id } — track which intersections are on this path
    }

    /**
     * Add a straight segment to the path.
     * @param {{ x: number, z: number }} from
     * @param {{ x: number, z: number }} to
     */
    addStraight(from, to) {
        const seg = new StraightSegment(from, to, this.totalLength);
        if (seg.length < 0.01) return; // skip degenerate segments
        this.segments.push(seg);
        this.totalLength += seg.length;
    }

    /**
     * Add a quadratic Bezier turn segment.
     * @param {{ x: number, z: number }} entry
     * @param {{ x: number, z: number }} control - L-corner control point
     * @param {{ x: number, z: number }} exit
     */
    addTurn(entry, control, exit) {
        const seg = new TurnSegment(entry, control, exit, this.totalLength);
        if (seg.length < 0.01) return;
        this.segments.push(seg);
        this.totalLength += seg.length;
    }

    /**
     * Get world position at distance d along the path.
     * @param {number} d - Distance in meters from path start
     * @returns {{ x: number, z: number }}
     */
    getPosition(d) {
        d = Math.max(0, Math.min(d, this.totalLength));
        for (const seg of this.segments) {
            const segEnd = seg.startD + seg.length;
            if (d <= segEnd) {
                const localT = seg.length > 0 ? (d - seg.startD) / seg.length : 0;
                return seg.getPosition(localT);
            }
        }
        // Past end — return last point
        const last = this.segments[this.segments.length - 1];
        return last ? last.getPosition(1) : { x: 0, z: 0 };
    }

    /**
     * Get heading (rotation.y) at distance d along the path.
     * @param {number} d
     * @returns {number} Angle in radians
     */
    getHeading(d) {
        d = Math.max(0, Math.min(d, this.totalLength));
        for (const seg of this.segments) {
            const segEnd = seg.startD + seg.length;
            if (d <= segEnd) {
                const localT = seg.length > 0 ? (d - seg.startD) / seg.length : 0;
                return seg.getHeading(localT);
            }
        }
        const last = this.segments[this.segments.length - 1];
        return last ? last.getHeading(1) : 0;
    }

    /**
     * Check if distance d is within a turn segment.
     * @param {number} d
     * @returns {boolean}
     */
    isInTurn(d) {
        for (const seg of this.segments) {
            if (d >= seg.startD && d <= seg.startD + seg.length) {
                return seg.type === 'turn';
            }
        }
        return false;
    }

    /**
     * Get the end point of the path.
     * @returns {{ x: number, z: number }}
     */
    getEndPoint() {
        return this.getPosition(this.totalLength);
    }

    /**
     * Get remaining distance from d to end of path.
     * @param {number} d
     * @returns {number}
     */
    remainingLength(d) {
        return Math.max(0, this.totalLength - d);
    }

    /**
     * Sample the path as an array of points (for rendering route lines).
     * @param {number} fromD - Start distance
     * @param {number} toD - End distance
     * @param {number} step - Sampling interval in meters
     * @returns {Array<{ x: number, z: number }>}
     */
    sample(fromD, toD, step = 5) {
        const points = [];
        for (let d = fromD; d <= toD; d += step) {
            points.push(this.getPosition(d));
        }
        // Always include the exact end point
        if (toD > fromD) {
            points.push(this.getPosition(toD));
        }
        return points;
    }

    /**
     * Remove segments that are entirely behind distance d.
     * Shifts startD values so d remains valid.
     * @param {number} d - Trim everything before this distance
     */
    trimBefore(d) {
        while (this.segments.length > 1) {
            const first = this.segments[0];
            if (first.startD + first.length < d) {
                this.segments.shift();
            } else {
                break;
            }
        }
    }

    /**
     * Get the last intersection stored on this path.
     */
    getLastIntersection() {
        return this.intersections[this.intersections.length - 1] || null;
    }

    /**
     * Get the second-to-last intersection (for direction calculation).
     */
    getPrevIntersection() {
        return this.intersections[this.intersections.length - 2] || null;
    }
}

// ============================================================
// PATH BUILDER — construct a CarPath from a road network route
// ============================================================

/**
 * Build a CarPath from a sequence of intersection nodes.
 *
 * @param {Array<{ x: number, z: number, id: string }>} route - Ordered intersections to visit
 * @param {number} laneOffset - Lateral offset from road center (positive = right of center)
 * @param {number} margin - Distance from intersection center to path entry/exit (meters)
 * @returns {CarPath}
 */
export function buildPathFromRoute(route, laneOffset = 3, margin = 8) {
    const path = new CarPath();
    if (route.length < 2) return path;

    // For each pair of adjacent intersections, compute the lane-following
    // entry and exit points for the straight road segment between them.
    // Turns fill the gaps BETWEEN consecutive road segments.

    const roadPoints = []; // Array of { entry, exit, horizontal, dir, lx, lz, node }

    for (let i = 0; i < route.length - 1; i++) {
        const curr = route[i];
        const next = route[i + 1];
        const dx = next.x - curr.x;
        const dz = next.z - curr.z;
        const horizontal = Math.abs(dx) > Math.abs(dz);
        const dir = horizontal ? Math.sign(dx) : Math.sign(dz);

        // Lane offset: perpendicular to travel, right-hand traffic
        const lx = horizontal ? 0 : -dir * laneOffset;
        const lz = horizontal ? dir * laneOffset : 0;

        // Entry: margin meters past the departure intersection, on lane
        const entry = {
            x: curr.x + (horizontal ? dir * margin : 0) + lx,
            z: curr.z + (horizontal ? 0 : dir * margin) + lz,
        };

        // Exit: margin meters before the arrival intersection, on lane
        const exit = {
            x: next.x - (horizontal ? dir * margin : 0) + lx,
            z: next.z - (horizontal ? 0 : dir * margin) + lz,
        };

        roadPoints.push({ entry, exit, horizontal, dir, lx, lz, fromNode: curr, toNode: next });
    }

    // Build path: straight segments with turn curves between them
    for (let i = 0; i < roadPoints.length; i++) {
        const rp = roadPoints[i];

        // Turn from previous road segment to this one
        if (i > 0) {
            const prevExit = path.getEndPoint();
            const control = computeTurnControl(prevExit, rp.entry, rp.fromNode);
            path.addTurn(prevExit, control, rp.entry);
        }

        // Straight along this road
        path.addStraight(rp.entry, rp.exit);
        path.intersections.push(rp.fromNode);
    }

    // Record the final destination intersection
    path.intersections.push(route[route.length - 1]);

    return path;
}

/**
 * Extend an existing path by appending a turn + straight to the next intersection.
 *
 * @param {CarPath} path
 * @param {{ x: number, z: number }} nextNode - Next intersection to drive to
 * @param {number} laneOffset
 * @param {number} margin
 */
export function extendPath(path, nextNode, laneOffset = 3, margin = 3) {
    const lastNode = path.getLastIntersection();
    if (!lastNode) return;

    const dx = nextNode.x - lastNode.x;
    const dz = nextNode.z - lastNode.z;
    const horizontal = Math.abs(dx) > Math.abs(dz);
    const dir = horizontal ? Math.sign(dx) : Math.sign(dz);

    const lx = horizontal ? 0 : -dir * laneOffset;
    const lz = horizontal ? dir * laneOffset : 0;

    const entry = {
        x: lastNode.x + (horizontal ? dir * margin : 0) + lx,
        z: lastNode.z + (horizontal ? 0 : dir * margin) + lz,
    };

    const exit = {
        x: nextNode.x - (horizontal ? dir * margin : 0) + lx,
        z: nextNode.z - (horizontal ? 0 : dir * margin) + lz,
    };

    // Turn from current path end to new segment entry
    const prevExit = path.getEndPoint();
    const control = computeTurnControl(prevExit, entry, lastNode);
    path.addTurn(prevExit, control, entry);

    // Straight along new road
    path.addStraight(entry, exit);
    path.intersections.push(nextNode);
}

/**
 * Compute the Bezier control point for a turn at an intersection.
 *
 * For perpendicular grid roads, the control point is the L-corner:
 * the point where extending the entry direction meets extending
 * backwards from the exit direction.
 *
 * @param {{ x: number, z: number }} prevExit - Where the car is coming from
 * @param {{ x: number, z: number }} nextEntry - Where the car is going to
 * @param {{ x: number, z: number }} intersection - The intersection center
 * @returns {{ x: number, z: number }}
 */
function computeTurnControl(prevExit, nextEntry, intersection) {
    // Direction from prevExit toward intersection
    const dx1 = intersection.x - prevExit.x;
    const dz1 = intersection.z - prevExit.z;

    // Direction from intersection toward nextEntry
    const dx2 = nextEntry.x - intersection.x;
    const dz2 = nextEntry.z - intersection.z;

    // Check if this is a straight-through (same direction)
    const dot = dx1 * dx2 + dz1 * dz2;
    const len1 = Math.sqrt(dx1 * dx1 + dz1 * dz1) || 1;
    const len2 = Math.sqrt(dx2 * dx2 + dz2 * dz2) || 1;
    const cosAngle = dot / (len1 * len2);

    if (cosAngle > 0.9) {
        // Nearly straight — control point is midpoint (Bezier degenerates to line)
        return {
            x: (prevExit.x + nextEntry.x) / 2,
            z: (prevExit.z + nextEntry.z) / 2,
        };
    }

    // For a 90° turn: L-corner
    // The control point shares one coordinate with prevExit and the other with nextEntry.
    // Determine which based on approach direction:
    const approachHorizontal = Math.abs(dx1) > Math.abs(dz1);
    if (approachHorizontal) {
        // Approaching horizontally → keep prevExit's z, use nextEntry's x
        return { x: nextEntry.x, z: prevExit.z };
    } else {
        // Approaching vertically → keep prevExit's x, use nextEntry's z
        return { x: prevExit.x, z: nextEntry.z };
    }
}
