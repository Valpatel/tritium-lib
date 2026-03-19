// Created by Matthew Valancy
// Copyright 2026 Valpatel Software LLC
// Licensed under AGPL-3.0 — see LICENSE for details.

/**
 * Path — parameterized curves for unit movement.
 *
 * A path is a sequence of segments (straight lines and Bezier curves).
 * A unit's position is defined by a single parameter `d` (distance in meters
 * from the start of the path). Position, heading, and altitude are derived
 * from the path at `d`.
 *
 * Path2D: segments in the XZ plane (ground units)
 * Path3D: segments in XYZ space (aircraft) — extends Path2D
 */

// ============================================================
// SEGMENT TYPES
// ============================================================

class StraightSegment {
    constructor(from, to, startD) {
        this.type = 'straight';
        this.startD = startD;
        this.from = from; // {x, z} or {x, y, z}
        this.to = to;
        this.dx = to.x - from.x;
        this.dz = to.z - from.z;
        this.dy = (to.y || 0) - (from.y || 0);
        this.length = Math.sqrt(this.dx * this.dx + this.dz * this.dz + this.dy * this.dy);
        this._heading = Math.atan2(this.dx, this.dz);
    }

    getPosition(t) {
        return {
            x: this.from.x + this.dx * t,
            y: (this.from.y || 0) + this.dy * t,
            z: this.from.z + this.dz * t,
        };
    }

    getHeading(_t) {
        return this._heading;
    }

    getPitch(_t) {
        return this.length > 0.01 ? Math.atan2(this.dy, Math.sqrt(this.dx * this.dx + this.dz * this.dz)) : 0;
    }
}

class BezierSegment {
    constructor(entry, control, exit, startD) {
        this.type = 'bezier';
        this.startD = startD;
        this.entry = entry;   // {x, z} or {x, y, z}
        this.control = control;
        this.exit = exit;
        this.length = this._measureLength(20);
    }

    getPosition(t) {
        const u = 1 - t;
        return {
            x: u * u * this.entry.x + 2 * u * t * this.control.x + t * t * this.exit.x,
            y: u * u * (this.entry.y || 0) + 2 * u * t * (this.control.y || 0) + t * t * (this.exit.y || 0),
            z: u * u * this.entry.z + 2 * u * t * this.control.z + t * t * this.exit.z,
        };
    }

    getHeading(t) {
        const u = 1 - t;
        const tx = 2 * u * (this.control.x - this.entry.x) + 2 * t * (this.exit.x - this.control.x);
        const tz = 2 * u * (this.control.z - this.entry.z) + 2 * t * (this.exit.z - this.control.z);
        return Math.atan2(tx, tz);
    }

    getPitch(t) {
        const u = 1 - t;
        const ty = 2 * u * ((this.control.y || 0) - (this.entry.y || 0)) + 2 * t * ((this.exit.y || 0) - (this.control.y || 0));
        const tx = 2 * u * (this.control.x - this.entry.x) + 2 * t * (this.exit.x - this.control.x);
        const tz = 2 * u * (this.control.z - this.entry.z) + 2 * t * (this.exit.z - this.control.z);
        const hLen = Math.sqrt(tx * tx + tz * tz);
        return hLen > 0.01 ? Math.atan2(ty, hLen) : 0;
    }

    _measureLength(samples) {
        let len = 0;
        let prev = this.entry;
        for (let i = 1; i <= samples; i++) {
            const t = i / samples;
            const curr = this.getPosition(t);
            const dx = curr.x - prev.x;
            const dy = curr.y - (prev.y || 0);
            const dz = curr.z - prev.z;
            len += Math.sqrt(dx * dx + dy * dy + dz * dz);
            prev = curr;
        }
        return len;
    }
}

// ============================================================
// PATH CLASS
// ============================================================

export class Path {
    constructor() {
        this.segments = [];
        this.totalLength = 0;
        this.waypoints = []; // intersection/node references for route tracking
    }

    /** Add a straight segment. */
    addStraight(from, to) {
        const seg = new StraightSegment(from, to, this.totalLength);
        if (seg.length < 0.01) return this;
        this.segments.push(seg);
        this.totalLength += seg.length;
        return this;
    }

    /** Add a quadratic Bezier turn/curve. */
    addBezier(entry, control, exit) {
        const seg = new BezierSegment(entry, control, exit, this.totalLength);
        if (seg.length < 0.01) return this;
        this.segments.push(seg);
        this.totalLength += seg.length;
        return this;
    }

    /** Get the segment containing distance d. Returns { segment, localT }. */
    _findSegment(d) {
        d = Math.max(0, Math.min(d, this.totalLength));
        for (const seg of this.segments) {
            const segEnd = seg.startD + seg.length;
            if (d <= segEnd) {
                const localT = seg.length > 0 ? (d - seg.startD) / seg.length : 0;
                return { segment: seg, localT };
            }
        }
        const last = this.segments[this.segments.length - 1];
        return last ? { segment: last, localT: 1 } : null;
    }

    /** Get world position at distance d. Returns {x, y, z}. */
    getPosition(d) {
        const found = this._findSegment(d);
        if (!found) return { x: 0, y: 0, z: 0 };
        return found.segment.getPosition(found.localT);
    }

    /** Get heading (rotation Y) at distance d. */
    getHeading(d) {
        const found = this._findSegment(d);
        if (!found) return 0;
        return found.segment.getHeading(found.localT);
    }

    /** Get pitch (rotation X) at distance d. For ground units this is always 0. */
    getPitch(d) {
        const found = this._findSegment(d);
        if (!found) return 0;
        return found.segment.getPitch(found.localT);
    }

    /** Check if distance d is within a Bezier (turn) segment. */
    isInCurve(d) {
        const found = this._findSegment(d);
        return found ? found.segment.type === 'bezier' : false;
    }

    /** Get remaining distance from d to end of path. */
    remaining(d) {
        return Math.max(0, this.totalLength - d);
    }

    /** Get the last point on the path. */
    getEndPoint() {
        return this.getPosition(this.totalLength);
    }

    /** Sample path as array of points. */
    sample(fromD, toD, step = 5) {
        const points = [];
        for (let d = fromD; d <= toD; d += step) {
            points.push(this.getPosition(d));
        }
        if (toD > fromD) points.push(this.getPosition(toD));
        return points;
    }

    /** Remove segments entirely behind distance d. */
    trimBefore(d) {
        while (this.segments.length > 1 && this.segments[0].startD + this.segments[0].length < d) {
            this.segments.shift();
        }
    }

    /** Record a waypoint (intersection node) on this path. */
    addWaypoint(node) {
        this.waypoints.push(node);
        return this;
    }

    /** Get last waypoint. */
    getLastWaypoint() {
        return this.waypoints[this.waypoints.length - 1] || null;
    }

    /** Get second-to-last waypoint. */
    getPrevWaypoint() {
        return this.waypoints[this.waypoints.length - 2] || null;
    }
}

// ============================================================
// PATH BUILDER HELPERS
// ============================================================

/**
 * Compute Bezier control point for a turn at an intersection.
 * For perpendicular grid roads: L-corner where entry tangent meets exit tangent.
 *
 * @param {{ x: number, z: number }} prevExit - End of previous segment
 * @param {{ x: number, z: number }} nextEntry - Start of next segment
 * @param {{ x: number, z: number }} intersection - Node center
 * @returns {{ x: number, z: number }}
 */
export function computeTurnControl(prevExit, nextEntry, intersection) {
    const dx = nextEntry.x - prevExit.x;
    const dz = nextEntry.z - prevExit.z;
    const absDx = Math.abs(dx);
    const absDz = Math.abs(dz);
    const dominant = Math.max(absDx, absDz);
    const minor = Math.min(absDx, absDz);

    // Nearly straight — midpoint gives straight Bezier
    if (dominant < 0.1 || minor / dominant < 0.3) {
        return { x: (prevExit.x + nextEntry.x) / 2, z: (prevExit.z + nextEntry.z) / 2 };
    }

    // 90° turn: L-corner based on approach direction
    const approachDx = Math.abs(intersection.x - prevExit.x);
    const approachDz = Math.abs(intersection.z - prevExit.z);
    if (approachDx > approachDz) {
        return { x: nextEntry.x, z: prevExit.z };
    } else {
        return { x: prevExit.x, z: nextEntry.z };
    }
}

/**
 * Build a path from a route (sequence of intersection nodes).
 *
 * @param {Array<{x, z, id}>} route - Ordered intersections
 * @param {number} laneOffset - Lateral offset from road center
 * @param {number} margin - Distance from intersection to segment start/end
 * @returns {Path}
 */
export function buildPath(route, laneOffset = 3, margin = 8) {
    const path = new Path();
    if (route.length < 2) return path;

    const points = [];
    for (let i = 0; i < route.length - 1; i++) {
        const curr = route[i];
        const next = route[i + 1];
        const dx = next.x - curr.x;
        const dz = next.z - curr.z;
        const horizontal = Math.abs(dx) > Math.abs(dz);
        const dir = horizontal ? Math.sign(dx) : Math.sign(dz);
        const lx = horizontal ? 0 : -dir * laneOffset;
        const lz = horizontal ? dir * laneOffset : 0;

        points.push({
            entry: {
                x: curr.x + (horizontal ? dir * margin : 0) + lx,
                z: curr.z + (horizontal ? 0 : dir * margin) + lz,
            },
            exit: {
                x: next.x - (horizontal ? dir * margin : 0) + lx,
                z: next.z - (horizontal ? 0 : dir * margin) + lz,
            },
            fromNode: curr,
        });
    }

    for (let i = 0; i < points.length; i++) {
        if (i > 0) {
            const prevExit = path.getEndPoint();
            const control = computeTurnControl(prevExit, points[i].entry, points[i].fromNode);
            path.addBezier(prevExit, control, points[i].entry);
        }
        path.addStraight(points[i].entry, points[i].exit);
        path.addWaypoint(points[i].fromNode);
    }
    path.addWaypoint(route[route.length - 1]);
    return path;
}

/**
 * Extend an existing path to the next intersection.
 *
 * @param {Path} path
 * @param {{x, z, id}} nextNode
 * @param {number} laneOffset
 * @param {number} margin
 */
export function extendPath(path, nextNode, laneOffset = 3, margin = 8) {
    const lastNode = path.getLastWaypoint();
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

    const prevExit = path.getEndPoint();
    const control = computeTurnControl(prevExit, entry, lastNode);
    path.addBezier(prevExit, control, entry);
    path.addStraight(entry, exit);
    path.addWaypoint(nextNode);
}
