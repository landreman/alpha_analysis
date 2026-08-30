"""Bounded local retriangulation of folded incoming-surface charts.

A repair preserves every vertex, the cavity's directed boundary, its component,
and its signed chart area. It is permitted only for a simple disk with a
one-to-one boundary chart. No global field-line chart is assumed.
"""

import numpy as np


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _area(points, triangle):
    a, b, c = points[list(triangle)]
    return _cross(b - a, c - a)


def _weights(points, triangle, point):
    a, b, c = points[list(triangle)]
    denominator = _cross(b - a, c - a)
    if denominator == 0:
        return None
    u = _cross(point - a, c - a) / denominator
    v = _cross(b - a, point - a) / denominator
    return np.array([1 - u - v, u, v])


def _boundary(triangles):
    edges = {}
    for triangle in triangles:
        for a, b in zip(triangle, np.roll(triangle, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(
                (int(a), int(b))
            )
    directed = [owners[0] for owners in edges.values() if len(owners) == 1]
    if any(len(owners) > 2 for owners in edges.values()):
        return None
    following = {a: b for a, b in directed}
    if len(following) != len(directed) or len(set(following.values())) != len(directed):
        return None
    if not following:
        return None
    start = min(following)
    order = [start]
    current = following[start]
    while current != start:
        if current in order or current not in following:
            return None
        order.append(current)
        current = following[current]
    return order if len(order) == len(directed) else None


def _simple_polygon(points, order):
    for i in range(len(order)):
        a, b = points[order[i]], points[order[(i + 1) % len(order)]]
        for j in range(i + 1, len(order)):
            if j == (i + 1) % len(order) or (j + 1) % len(order) == i:
                continue
            c, d = points[order[j]], points[order[(j + 1) % len(order)]]
            first = _cross(b - a, c - a) * _cross(b - a, d - a)
            second = _cross(d - c, a - c) * _cross(d - c, b - c)
            if first <= 0 and second <= 0:
                # Collinear disjoint segments are not intersections.
                if (
                    first == 0
                    and second == 0
                    and (
                        np.any(np.maximum(a, b) < np.minimum(c, d))
                        or np.any(np.maximum(c, d) < np.minimum(a, b))
                    )
                ):
                    continue
                return False
    return True


def _triangulate(points, order, vertices, sign):
    """Ear-clip one simple cavity and retain all of its interior vertices."""
    if not _simple_polygon(points, order):
        return None
    polygon = list(order)
    result = []
    while len(polygon) > 3:
        found = False
        for index, current in enumerate(polygon):
            triangle = [
                polygon[index - 1],
                current,
                polygon[(index + 1) % len(polygon)],
            ]
            if sign * _area(points, triangle) <= 0:
                continue
            if any(
                np.min(_weights(points, triangle, points[other])) >= 0
                for other in polygon
                if other not in triangle
            ):
                continue
            result.append(triangle)
            polygon.pop(index)
            found = True
            break
        if not found:
            return None
    if sign * _area(points, polygon) <= 0:
        return None
    result.append(polygon)
    for vertex in sorted(set(vertices) - set(order)):
        owner = None
        for index, triangle in enumerate(result):
            weights = _weights(points, triangle, points[vertex])
            if weights is not None and np.min(weights) >= -64 * np.finfo(float).eps:
                owner = (index, weights)
                break
        if owner is None:
            return None
        index, weights = owner
        triangle = result[index]
        zero = np.flatnonzero(np.abs(weights) <= 64 * np.finfo(float).eps)
        if len(zero) > 1:
            # Distinct coincident vertices could be different well branches.
            # Never identify them merely because their chart points coincide.
            return None
        if len(zero) == 1:
            edge = [triangle[i] for i in range(3) if i != zero[0]]
            next_result = []
            for current in result:
                if all(v in current for v in edge):
                    for a, b in zip(current, np.roll(current, -1)):
                        if {a, b} == set(edge):
                            third = next(v for v in current if v not in edge)
                            next_result.extend(([a, vertex, third], [vertex, b, third]))
                            break
                else:
                    next_result.append(current)
            result = next_result
        else:
            a, b, c = triangle
            result[index] = [a, b, vertex]
            result.extend(([b, c, vertex], [c, a, vertex]))
    if any(sign * _area(points, triangle) <= 0 for triangle in result):
        return None
    if set(v for triangle in result for v in triangle) != set(vertices):
        return None
    if _boundary(result) != order:
        return None
    return result


def repair_incoming_charts(mesh, max_cavity_faces=64):
    """Repair only certified local disk cavities before any cuts are inserted.

    The field-line chart is q=sqrt(s)*(cos(alpha),sin(alpha)), with zeta lifted
    locally around the cavity. This keeps the axis regular while avoiding a
    globally identified alpha. Unrepairable folds are returned, never removed.
    """
    repaired = 0
    attempted = set()
    while True:
        bad = []
        areas = []
        for triangle in mesh.triangles:
            points = np.asarray(mesh.points)[triangle]
            chart = mesh._field_line_chart(points, points[0, 2])
            areas.append(_cross(chart[1] - chart[0], chart[2] - chart[0]))
        expected = {}
        for component in set(mesh.component_ids):
            values = [
                area
                for area, comp in zip(areas, mesh.component_ids)
                if comp == component
            ]
            expected[component] = np.sign(np.median(values))
        for index, (area, component) in enumerate(zip(areas, mesh.component_ids)):
            if expected[component] * area <= 0:
                key = tuple(sorted(mesh.triangles[index]))
                if key not in attempted:
                    bad.append(index)
        if not bad:
            remaining = sum(
                expected[comp] * area <= 0
                for area, comp in zip(areas, mesh.component_ids)
            )
            return repaired, int(remaining)
        seed = bad[0]
        attempted.add(tuple(sorted(mesh.triangles[seed])))
        component = mesh.component_ids[seed]
        sign = expected[component]
        origin = mesh.points[mesh.triangles[seed][0]][2]
        chart = mesh._field_line_chart(np.asarray(mesh.points), origin)
        selected = {seed}
        replacement = None
        while len(selected) <= max_cavity_faces:
            triangles = [mesh.triangles[index] for index in sorted(selected)]
            vertices = set(v for triangle in triangles for v in triangle)
            boundary = _boundary(triangles)
            if boundary is not None:
                replacement = _triangulate(chart, boundary, vertices, sign)
                if replacement is not None:
                    before = sum(_area(chart, triangle) for triangle in triangles)
                    after = sum(_area(chart, triangle) for triangle in replacement)
                    if abs(before - after) > 256 * np.finfo(float).eps * max(
                        1, abs(before)
                    ):
                        replacement = None
                    else:
                        break
            # Expand through full stars, retaining a connected topological
            # disk when possible. Crossing to another component is forbidden.
            expanded = {
                index
                for index, triangle in enumerate(mesh.triangles)
                if mesh.component_ids[index] == component
                and any(v in vertices for v in triangle)
            }
            if expanded == selected:
                break
            selected = expanded
        if replacement is None or len(selected) > max_cavity_faces:
            continue
        keep = [index for index in range(len(mesh.triangles)) if index not in selected]
        mesh.triangles = [mesh.triangles[index] for index in keep] + replacement
        mesh.parent_tetrahedra = [mesh.parent_tetrahedra[index] for index in keep] + [
            -1
        ] * len(replacement)
        mesh.component_ids = [mesh.component_ids[index] for index in keep] + [
            component
        ] * len(replacement)
        repaired += len(selected)
