// @ts-nocheck
// Shared with FitMongo: Canny path tracing, simplification, and marching-squares silhouettes.
const NEIGHBORS = [
  [-1, -1], [0, -1], [1, -1], [-1, 0],
  [1, 0], [-1, 1], [0, 1], [1, 1],
];

function pointSegmentDistanceSquared(point, start, end) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  if (dx === 0 && dy === 0) return (point.x - start.x) ** 2 + (point.y - start.y) ** 2;
  const t = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy)));
  const x = start.x + t * dx;
  const y = start.y + t * dy;
  return (point.x - x) ** 2 + (point.y - y) ** 2;
}

function simplifyOpen(points, tolerance) {
  if (points.length <= 2) return points;
  const keep = new Uint8Array(points.length);
  keep[0] = keep[points.length - 1] = 1;
  const stack = [[0, points.length - 1]];
  const toleranceSquared = tolerance * tolerance;
  while (stack.length) {
    const [first, last] = stack.pop();
    let furthest = -1;
    let maximum = toleranceSquared;
    for (let index = first + 1; index < last; index += 1) {
      const distance = pointSegmentDistanceSquared(points[index], points[first], points[last]);
      if (distance > maximum) {
        maximum = distance;
        furthest = index;
      }
    }
    if (furthest >= 0) {
      keep[furthest] = 1;
      stack.push([first, furthest], [furthest, last]);
    }
  }
  return points.filter((_, index) => keep[index]);
}

function simplifyPath(path, tolerance) {
  if (!path.closed) return { ...path, points: simplifyOpen(path.points, tolerance) };
  const points = path.points.slice(0, -1);
  if (points.length < 6) return { ...path, points };
  const halfway = Math.floor(points.length / 2);
  const firstHalf = simplifyOpen(points.slice(0, halfway + 1), tolerance);
  const secondHalf = simplifyOpen([...points.slice(halfway), points[0]], tolerance);
  return { ...path, points: [...firstHalf, ...secondHalf.slice(1, -1)] };
}

function pathMetrics(path, edgeStrength, width) {
  let geometricLength = 0;
  let contrastTotal = 0;
  let turnTotal = 0;
  let turnChanges = 0;
  let previousTurn = 0;
  let centerX = 0;
  let centerY = 0;
  for (let index = 0; index < path.points.length; index += 1) {
    const point = path.points[index];
    centerX += point.x;
    centerY += point.y;
    contrastTotal += edgeStrength[Math.floor(point.y) * width + Math.floor(point.x)] || 0;
    if (index > 0) {
      const previous = path.points[index - 1];
      geometricLength += Math.hypot(point.x - previous.x, point.y - previous.y);
    }
    if (index <= 0 || index >= path.points.length - 1) continue;
    const before = path.points[index - 1];
    const after = path.points[index + 1];
    const firstAngle = Math.atan2(point.y - before.y, point.x - before.x);
    const secondAngle = Math.atan2(after.y - point.y, after.x - point.x);
    let turn = secondAngle - firstAngle;
    if (turn > Math.PI) turn -= Math.PI * 2;
    else if (turn < -Math.PI) turn += Math.PI * 2;
    turnTotal += Math.abs(turn);
    if (Math.abs(turn) > .08 && Math.abs(previousTurn) > .08 && Math.sign(turn) !== Math.sign(previousTurn)) {
      turnChanges += 1;
    }
    if (Math.abs(turn) > .08) previousTurn = turn;
  }
  const denominator = Math.max(1, path.points.length - 2);
  return {
    geometricLength,
    contrast: contrastTotal / Math.max(1, path.points.length) / 255,
    curvature: turnTotal / denominator,
    jaggedness: turnChanges / denominator,
    centerX: centerX / path.points.length,
    centerY: centerY / path.points.length,
  };
}

function perceptualCandidates(paths, edgeStrength, width, accuracy) {
  const cells = new Map();
  const candidates = paths.map((path) => {
    const metrics = pathMetrics(path, edgeStrength, width);
    const cellX = Math.floor(metrics.centerX / 32);
    const cellY = Math.floor(metrics.centerY / 32);
    const key = `${cellX}:${cellY}`;
    cells.set(key, (cells.get(key) || 0) + (metrics.geometricLength < 28 ? 1 : .25));
    return { path, metrics, cellX, cellY };
  });
  const baseTolerance = 4.5 - accuracy * .04;
  for (const candidate of candidates) {
    const { metrics } = candidate;
    let density = 0;
    for (let y = -1; y <= 1; y += 1) {
      for (let x = -1; x <= 1; x += 1) {
        density += cells.get(`${candidate.cellX + x}:${candidate.cellY + y}`) || 0;
      }
    }
    const shapeDetail = Math.min(1, metrics.curvature / .45);
    const contrastDetail = Math.min(1, metrics.contrast / .7);
    const shortClosedDetail = candidate.path.closed && metrics.geometricLength < 80 ? .72 : 1;
    const tolerance = Math.max(.35, baseTolerance
      * (1.5 - shapeDetail * .65)
      * (1.2 - contrastDetail * .35)
      * (1 + metrics.jaggedness * 1.8)
      * shortClosedDetail);
    candidate.simplified = simplifyPath(candidate.path, tolerance);
    const continuity = candidate.path.closed ? 1.16 : 1;
    const smoothness = 1 / (1 + metrics.jaggedness * 5);
    const densityPenalty = 1 + Math.max(0, density - 2) * .1;
    candidate.importance = (metrics.contrast ** 1.35)
      * Math.sqrt(Math.max(1, metrics.geometricLength))
      * continuity * smoothness / densityPenalty;
    candidate.cost = Math.max(2, candidate.simplified.points.length);
    candidate.valuePerPoint = candidate.importance / candidate.cost;
  }
  return candidates;
}

function pointBudget(accuracy) {
  // A smooth exponential curve gives the quality control predictable bandwidth
  // growth without the abrupt threshold jumps of the previous implementation.
  return Math.round(180 + 85 * Math.exp(accuracy / 24));
}

function edgeKey(first, second, pixelCount) {
  const low = Math.min(first, second);
  return low * pixelCount + Math.max(first, second);
}

function connectedNeighbors(index, edgeMap, width, height) {
  const x = index % width;
  const y = Math.floor(index / width);
  const result = [];
  for (const [dx, dy] of NEIGHBORS) {
    const nx = x + dx;
    const ny = y + dy;
    if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
      const neighbor = ny * width + nx;
      if (edgeMap[neighbor]) result.push(neighbor);
    }
  }
  return result;
}

function traceFrom(start, next, edgeMap, width, height, visited) {
  const pixelCount = width * height;
  const indices = [start];
  let previous = start;
  let current = next;
  visited.add(edgeKey(previous, current, pixelCount));
  while (true) {
    indices.push(current);
    const neighbors = connectedNeighbors(current, edgeMap, width, height);
    if (neighbors.length !== 2) break;
    const candidate = neighbors[0] === previous ? neighbors[1] : neighbors[0];
    const key = edgeKey(current, candidate, pixelCount);
    if (visited.has(key)) break;
    visited.add(key);
    previous = current;
    current = candidate;
  }
  const closed = indices.length > 3 && indices.at(-1) === indices[0];
  return {
    closed,
    points: indices.map((index) => ({ x: index % width + .5, y: Math.floor(index / width) + .5 })),
  };
}

function traceEdgeMap(edgeMap, width, height) {
  const paths = [];
  const visited = new Set();
  const pixelCount = width * height;
  for (let index = 0; index < pixelCount; index += 1) {
    if (!edgeMap[index]) continue;
    const neighbors = connectedNeighbors(index, edgeMap, width, height);
    if (neighbors.length === 2) continue;
    for (const neighbor of neighbors) {
      if (!visited.has(edgeKey(index, neighbor, pixelCount))) paths.push(traceFrom(index, neighbor, edgeMap, width, height, visited));
    }
  }
  // Trace closed loops, which contain no endpoint or junction.
  for (let index = 0; index < pixelCount; index += 1) {
    if (!edgeMap[index]) continue;
    for (const neighbor of connectedNeighbors(index, edgeMap, width, height)) {
      if (!visited.has(edgeKey(index, neighbor, pixelCount))) paths.push(traceFrom(index, neighbor, edgeMap, width, height, visited));
    }
  }
  return paths;
}

function gaussianBlur(gray, width, height) {
  const horizontal = new Float32Array(gray.length);
  const blurred = new Float32Array(gray.length);
  const weights = [1, 2, 1];
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      let total = 0;
      for (let offset = -1; offset <= 1; offset += 1) {
        total += gray[row + Math.max(0, Math.min(width - 1, x + offset))] * weights[offset + 1];
      }
      horizontal[row + x] = total / 4;
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let total = 0;
      for (let offset = -1; offset <= 1; offset += 1) {
        total += horizontal[Math.max(0, Math.min(height - 1, y + offset)) * width + x] * weights[offset + 1];
      }
      blurred[y * width + x] = total / 4;
    }
  }
  return blurred;
}

function detectEdges(
  rgba, maskRgba, width, height, maskThreshold = 128, method = 'sobel', includeMaskContour = true,
) {
  const pixelCount = width * height;
  let gray = new Float32Array(pixelCount);
  const inside = new Uint8Array(pixelCount);
  for (let pixel = 0, offset = 0; pixel < pixelCount; pixel += 1, offset += 4) {
    gray[pixel] = rgba[offset] * .299 + rgba[offset + 1] * .587 + rgba[offset + 2] * .114;
    inside[pixel] = maskRgba[offset] >= maskThreshold ? 1 : 0;
  }
  const boundaryBand = new Uint8Array(pixelCount);
  if (!includeMaskContour) {
    // Internal ink is sent alongside a dedicated marching-squares silhouette.
    // Exclude two source pixels around that silhouette so the RGB gradient at
    // the foreground/background transition cannot become a parallel outline.
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        if (!inside[index]) continue;
        for (let dy = -2; dy <= 2 && !boundaryBand[index]; dy += 1) {
          for (let dx = -2; dx <= 2; dx += 1) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= width || ny >= height || !inside[ny * width + nx]) {
              boundaryBand[index] = 1;
              break;
            }
          }
        }
      }
    }
  }
  if (method === 'canny') gray = gaussianBlur(gray, width, height);

  const magnitude = new Float32Array(pixelCount);
  const direction = new Uint8Array(pixelCount);
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      const top = index - width;
      const bottom = index + width;
      let gx;
      let gy;
      if (method === 'scharr') {
        // Divide Scharr's 3/10/3 response by four so its magnitude remains on
        // Sobel's scale and both methods can share thresholds and opacity.
        gx = (3 * gray[top + 1] + 10 * gray[index + 1] + 3 * gray[bottom + 1]
          - 3 * gray[top - 1] - 10 * gray[index - 1] - 3 * gray[bottom - 1]) / 4;
        gy = (3 * gray[bottom - 1] + 10 * gray[bottom] + 3 * gray[bottom + 1]
          - 3 * gray[top - 1] - 10 * gray[top] - 3 * gray[top + 1]) / 4;
      } else {
        gx = gray[top + 1] + 2 * gray[index + 1] + gray[bottom + 1]
          - gray[top - 1] - 2 * gray[index - 1] - gray[bottom - 1];
        gy = gray[bottom - 1] + 2 * gray[bottom] + gray[bottom + 1]
          - gray[top - 1] - 2 * gray[top] - gray[top + 1];
      }
      magnitude[index] = Math.sqrt(gx * gx + gy * gy);
      // Quantize the gradient into the same four sectors as atan2 without
      // evaluating an inverse trig function for every 640 px source pixel.
      const absoluteX = Math.abs(gx);
      const absoluteY = Math.abs(gy);
      if (absoluteY <= absoluteX * 0.41421356) direction[index] = 0;
      else if (absoluteY >= absoluteX * 2.41421356) direction[index] = 2;
      else direction[index] = gx * gy >= 0 ? 1 : 3;
    }
  }

  const suppressed = new Float32Array(pixelCount);
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      let first;
      let second;
      if (direction[index] === 0) [first, second] = [index - 1, index + 1];
      else if (direction[index] === 1) [first, second] = [index - width + 1, index + width - 1];
      else if (direction[index] === 2) [first, second] = [index - width, index + width];
      else [first, second] = [index - width - 1, index + width + 1];
      if (magnitude[index] >= magnitude[first] && magnitude[index] >= magnitude[second]) {
        suppressed[index] = magnitude[index];
      }
    }
  }

  const edgeMap = new Uint8Array(pixelCount);
  const edgeStrength = new Uint8ClampedArray(pixelCount);
  if (method === 'canny') {
    // A weak edge survives only when it connects to a strong edge. This is the
    // hysteresis step missing from the simpler Sobel/NMS implementation.
    const weak = new Uint8Array(pixelCount);
    const queue = new Int32Array(pixelCount);
    let head = 0;
    let tail = 0;
    const lowThreshold = 30;
    const highThreshold = 72;
    for (let index = 0; index < pixelCount; index += 1) {
      if (!inside[index] || boundaryBand[index] || suppressed[index] < lowThreshold) continue;
      weak[index] = 1;
      if (suppressed[index] >= highThreshold) {
        edgeMap[index] = 1;
        queue[tail] = index;
        tail += 1;
      }
    }
    while (head < tail) {
      const index = queue[head];
      head += 1;
      const x = index % width;
      const y = Math.floor(index / width);
      for (const [dx, dy] of NEIGHBORS) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const neighbor = ny * width + nx;
        if (!weak[neighbor] || edgeMap[neighbor]) continue;
        edgeMap[neighbor] = 1;
        queue[tail] = neighbor;
        tail += 1;
      }
    }
    for (let index = 0; index < pixelCount; index += 1) {
      if (edgeMap[index]) edgeStrength[index] = Math.min(255, Math.round(suppressed[index] * .8));
    }
  } else {
    const gradientThreshold = 72;
    for (let index = 0; index < pixelCount; index += 1) {
      if (inside[index] && !boundaryBand[index] && suppressed[index] >= gradientThreshold) {
        edgeMap[index] = 1;
        edgeStrength[index] = Math.min(255, Math.round(suppressed[index] * .65));
      }
    }
  }

  // The segmentation contour matters even where the source RGB has little
  // contrast, so retain it independently of the selected internal-edge method.
  if (includeMaskContour) {
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const index = y * width + x;
        if (inside[index] && (
          !inside[index - 1] || !inside[index + 1] || !inside[index - width] || !inside[index + width]
        )) {
          edgeMap[index] = 1;
          edgeStrength[index] = 255;
        }
      }
    }
  }

  return { edgeMap, edgeStrength };
}

export function rasterizeEdges(rgba, maskRgba, width, height, maskThreshold = 128, method = 'sobel') {
  return detectEdges(rgba, maskRgba, width, height, maskThreshold, method).edgeMap;
}

export function rasterizeSoftEdges(rgba, maskRgba, width, height, maskThreshold = 128, method = 'sobel') {
  return detectEdges(rgba, maskRgba, width, height, maskThreshold, method).edgeStrength;
}

export function vectorizeEdges(
  rgba, maskRgba, width, height, accuracy, maskThreshold = 128, method = 'sobel', selection = 'classic',
) {
  const normalizedAccuracy = Math.max(0, Math.min(100, accuracy));
  const { edgeMap, edgeStrength } = detectEdges(
    rgba, maskRgba, width, height, maskThreshold, method, false,
  );
  const traced = traceEdgeMap(edgeMap, width, height).filter((path) => path.points.length >= 2);
  if (selection !== 'perceptual') {
    const tolerance = 4.5 - normalizedAccuracy * .04;
    const minimumLength = normalizedAccuracy >= 90 ? 2 : normalizedAccuracy >= 70 ? 3 : 5;
    const minimumStrength = 22 + (100 - normalizedAccuracy) * .9;
    const paths = traced
      .filter((path) => path.points.length >= minimumLength)
      .map((path) => {
        const strength = Math.round(path.points.reduce((total, point) => (
          total + edgeStrength[Math.floor(point.y) * width + Math.floor(point.x)]
        ), 0) / path.points.length);
        return { ...simplifyPath(path, tolerance), strength };
      })
      .filter((path) => path.points.length >= 2 && path.strength >= minimumStrength);
    return {
      paths,
      pointCount: paths.reduce((total, path) => total + path.points.length, 0),
    };
  }
  const candidates = perceptualCandidates(traced, edgeStrength, width, normalizedAccuracy)
    .filter((candidate) => candidate.simplified.points.length >= 2 && candidate.metrics.contrast >= .07)
    .sort((first, second) => second.valuePerPoint - first.valuePerPoint);
  const budget = pointBudget(normalizedAccuracy);
  const selected = [];
  let usedPoints = 0;
  for (const candidate of candidates) {
    if (selected.length && usedPoints + candidate.cost > budget) continue;
    selected.push(candidate);
    usedPoints += candidate.cost;
    if (usedPoints >= budget) break;
  }
  // Render long, structural paths first while selection itself remains based
  // on perceptual value per encoded point.
  selected.sort((first, second) => second.metrics.geometricLength - first.metrics.geometricLength);
  const paths = selected.map((candidate) => ({
    ...candidate.simplified,
    strength: Math.max(0, Math.min(255, Math.round(candidate.metrics.contrast * 255))),
    importance: candidate.importance,
  }));
  return {
    paths,
    pointCount: paths.reduce((total, path) => total + path.points.length, 0),
  };
}

const MARCHING_SEGMENTS = {
  1: [[3, 0]],
  2: [[0, 1]],
  3: [[3, 1]],
  4: [[1, 2]],
  // Keep diagonally touching foreground pixels as separate components in the
  // two ambiguous marching-squares cases. Real masks rarely contain these,
  // but this avoids inventing a bridge across a one-pixel background gap.
  5: [[3, 0], [1, 2]],
  6: [[0, 2]],
  7: [[3, 2]],
  8: [[2, 3]],
  9: [[2, 0]],
  10: [[0, 1], [2, 3]],
  11: [[2, 1]],
  12: [[1, 3]],
  13: [[1, 0]],
  14: [[0, 3]],
};

function marchingPoint(cellX, cellY, edge) {
  if (edge === 0) return { x: cellX + .5, y: cellY };
  if (edge === 1) return { x: cellX + 1, y: cellY + .5 };
  if (edge === 2) return { x: cellX + .5, y: cellY + 1 };
  return { x: cellX, y: cellY + .5 };
}

function contourPointKey(point) {
  // Marching-squares coordinates are half integers. Integer keys avoid float
  // comparison and are also the natural future Uint16 WebRTC representation.
  return `${Math.round(point.x * 2)},${Math.round(point.y * 2)}`;
}

function polygonArea(points) {
  let twiceArea = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    twiceArea += current.x * next.y - next.x * current.y;
  }
  return Math.abs(twiceArea) / 2;
}

function stitchContourSegments(segments) {
  const atPoint = new Map();
  segments.forEach((segment, index) => {
    for (const point of segment) {
      const key = contourPointKey(point);
      if (!atPoint.has(key)) atPoint.set(key, []);
      atPoint.get(key).push(index);
    }
  });
  const used = new Uint8Array(segments.length);
  const contours = [];
  for (let start = 0; start < segments.length; start += 1) {
    if (used[start]) continue;
    used[start] = 1;
    const points = [segments[start][0], segments[start][1]];
    const firstKey = contourPointKey(points[0]);
    let currentKey = contourPointKey(points[1]);
    while (currentKey !== firstKey) {
      const nextIndex = (atPoint.get(currentKey) || []).find((index) => !used[index]);
      if (nextIndex === undefined) break;
      used[nextIndex] = 1;
      const segment = segments[nextIndex];
      const next = contourPointKey(segment[0]) === currentKey ? segment[1] : segment[0];
      points.push(next);
      currentKey = contourPointKey(next);
    }
    if (currentKey === firstKey && points.length >= 4) contours.push(points);
  }
  return contours;
}

export function vectorizeMaskContours(maskRgba, width, height, accuracy, maskThreshold = 128) {
  const normalizedAccuracy = Math.max(0, Math.min(100, accuracy));
  const inside = (x, y) => (
    x >= 0 && y >= 0 && x < width && y < height
      ? maskRgba[(y * width + x) * 4] >= maskThreshold
      : false
  );
  const segments = [];
  // One padded cell on the top/left closes silhouettes that touch an image
  // boundary instead of producing an open contour.
  for (let y = -1; y < height; y += 1) {
    for (let x = -1; x < width; x += 1) {
      const code = (inside(x, y) ? 1 : 0)
        | (inside(x + 1, y) ? 2 : 0)
        | (inside(x + 1, y + 1) ? 4 : 0)
        | (inside(x, y + 1) ? 8 : 0);
      for (const [firstEdge, secondEdge] of MARCHING_SEGMENTS[code] || []) {
        segments.push([marchingPoint(x, y, firstEdge), marchingPoint(x, y, secondEdge)]);
      }
    }
  }
  const tolerance = .35 + (100 - normalizedAccuracy) * .055;
  const minimumArea = 1 + (100 - normalizedAccuracy) * .65;
  const paths = stitchContourSegments(segments)
    .map((points) => simplifyPath({ closed: true, points }, tolerance))
    .filter((path) => path.points.length >= 3 && polygonArea(path.points) >= minimumArea)
    .map((path) => ({ ...path, strength: 255 }));
  return {
    paths,
    pointCount: paths.reduce((total, path) => total + path.points.length, 0),
  };
}

function sampledClassColor(sourceRgba, sourceWidth, sourceHeight, categories, width, height, category) {
  let red = 0;
  let green = 0;
  let blue = 0;
  let count = 0;
  // Sampling every other semantic pixel is ample for a flat palette and keeps
  // this worker inexpensive while the 640 px ink paths are also extracted.
  for (let y = 0; y < height; y += 2) {
    const sourceY = Math.min(sourceHeight - 1, Math.floor((y + .5) * sourceHeight / height));
    for (let x = 0; x < width; x += 2) {
      if (categories[y * width + x] !== category) continue;
      const sourceX = Math.min(sourceWidth - 1, Math.floor((x + .5) * sourceWidth / width));
      const offset = (sourceY * sourceWidth + sourceX) * 4;
      red += sourceRgba[offset];
      green += sourceRgba[offset + 1];
      blue += sourceRgba[offset + 2];
      count += 1;
    }
  }
  if (!count) return [128, 128, 128];
  const average = [red / count, green / count, blue / count];
  const luminance = average[0] * .299 + average[1] * .587 + average[2] * .114;
  return average.map((channel) => Math.max(0, Math.min(255, Math.round(
    luminance + (channel - luminance) * 1.22 + 4,
  ))));
}

function sampledMajorityClassColor(sourceRgba, sourceWidth, sourceHeight, categories, width, height, category) {
  const bins = new Map();
  for (let y = 0; y < height; y += 1) {
    const sourceY = Math.min(sourceHeight - 1, Math.floor((y + .5) * sourceHeight / height));
    for (let x = 0; x < width; x += 1) {
      if (categories[y * width + x] !== category) continue;
      const sourceX = Math.min(sourceWidth - 1, Math.floor((x + .5) * sourceWidth / width));
      const offset = (sourceY * sourceWidth + sourceX) * 4;
      const red = sourceRgba[offset];
      const green = sourceRgba[offset + 1];
      const blue = sourceRgba[offset + 2];
      const key = `${red >> 5}:${green >> 5}:${blue >> 5}`;
      const bin = bins.get(key) || [0, 0, 0, 0];
      bin[0] += red;
      bin[1] += green;
      bin[2] += blue;
      bin[3] += 1;
      bins.set(key, bin);
    }
  }
  const majority = [...bins.values()].sort((first, second) => second[3] - first[3])[0];
  if (!majority) return [128, 128, 128];
  const average = majority.slice(0, 3).map((total) => total / majority[3]);
  const luminance = average[0] * .299 + average[1] * .587 + average[2] * .114;
  return average.map((channel) => Math.max(0, Math.min(255, Math.round(
    luminance + (channel - luminance) * 1.22 + 4,
  ))));
}

function pointInPolygon(x, y, points) {
  let inside = false;
  for (let current = 0, previous = points.length - 1; current < points.length; previous = current, current += 1) {
    const first = points[current];
    const second = points[previous];
    if ((first.y > y) !== (second.y > y)
      && x < ((second.x - first.x) * (y - first.y)) / (second.y - first.y) + first.x) {
      inside = !inside;
    }
  }
  return inside;
}

function sampledComponentColor(
  sourceRgba, sourceWidth, sourceHeight, categories, width, height, category, path,
) {
  const xs = path.points.map((point) => point.x);
  const ys = path.points.map((point) => point.y);
  const left = Math.max(0, Math.floor(Math.min(...xs)));
  const right = Math.min(width - 1, Math.ceil(Math.max(...xs)));
  const top = Math.max(0, Math.floor(Math.min(...ys)));
  const bottom = Math.min(height - 1, Math.ceil(Math.max(...ys)));
  let red = 0;
  let green = 0;
  let blue = 0;
  let count = 0;
  for (let y = top; y <= bottom; y += 1) {
    const sourceY = Math.min(sourceHeight - 1, Math.floor((y + .5) * sourceHeight / height));
    for (let x = left; x <= right; x += 1) {
      if (categories[y * width + x] !== category || !pointInPolygon(x + .5, y + .5, path.points)) continue;
      const sourceX = Math.min(sourceWidth - 1, Math.floor((x + .5) * sourceWidth / width));
      const offset = (sourceY * sourceWidth + sourceX) * 4;
      red += sourceRgba[offset];
      green += sourceRgba[offset + 1];
      blue += sourceRgba[offset + 2];
      count += 1;
    }
  }
  if (!count) return null;
  const average = [red / count, green / count, blue / count];
  const luminance = average[0] * .299 + average[1] * .587 + average[2] * .114;
  return average.map((channel) => Math.max(0, Math.min(255, Math.round(
    luminance + (channel - luminance) * 1.22 + 4,
  ))));
}

function clothingColorMasks(sourceRgba, sourceWidth, sourceHeight, categories, width, height, category) {
  const samples = [];
  let darkest;
  let lightest;
  let clothingTop = height;
  let clothingBottom = 0;
  for (let y = 0; y < height; y += 1) {
    const sourceY = Math.min(sourceHeight - 1, Math.floor((y + .5) * sourceHeight / height));
    for (let x = 0; x < width; x += 1) {
      const pixel = y * width + x;
      if (categories[pixel] !== category) continue;
      const sourceX = Math.min(sourceWidth - 1, Math.floor((x + .5) * sourceWidth / width));
      const offset = (sourceY * sourceWidth + sourceX) * 4;
      const color = [sourceRgba[offset], sourceRgba[offset + 1], sourceRgba[offset + 2]];
      const luminance = color[0] * .299 + color[1] * .587 + color[2] * .114;
      const sample = { pixel, color, luminance };
      samples.push(sample);
      clothingTop = Math.min(clothingTop, y);
      clothingBottom = Math.max(clothingBottom, y);
      if (!darkest || luminance < darkest.luminance) darkest = sample;
      if (!lightest || luminance > lightest.luminance) lightest = sample;
    }
  }
  if (samples.length < 32) return null;
  let centers = [darkest.color.map(Number), lightest.color.map(Number)];
  const assignments = new Uint8Array(samples.length);
  for (let iteration = 0; iteration < 6; iteration += 1) {
    const totals = [[0, 0, 0, 0], [0, 0, 0, 0]];
    samples.forEach((sample, index) => {
      const distances = centers.map((center) => (
        (sample.color[0] - center[0]) ** 2
        + (sample.color[1] - center[1]) ** 2
        + (sample.color[2] - center[2]) ** 2
      ));
      const cluster = distances[1] < distances[0] ? 1 : 0;
      assignments[index] = cluster;
      totals[cluster][0] += sample.color[0];
      totals[cluster][1] += sample.color[1];
      totals[cluster][2] += sample.color[2];
      totals[cluster][3] += 1;
    });
    centers = centers.map((center, cluster) => totals[cluster][3]
      ? totals[cluster].slice(0, 3).map((total) => total / totals[cluster][3])
      : center);
  }
  const counts = [0, 0];
  const verticalTotals = [0, 0];
  assignments.forEach((cluster, index) => {
    counts[cluster] += 1;
    verticalTotals[cluster] += Math.floor(samples[index].pixel / width);
  });
  const separation = Math.hypot(
    centers[0][0] - centers[1][0], centers[0][1] - centers[1][1], centers[0][2] - centers[1][2],
  );
  const minimumCluster = Math.max(16, samples.length * .08);
  const verticalSeparation = Math.abs(verticalTotals[0] / counts[0] - verticalTotals[1] / counts[1]);
  const minimumVerticalSeparation = Math.max(6, (clothingBottom - clothingTop) * .18);
  if (separation < 58 || counts.some((count) => count < minimumCluster)
    || verticalSeparation < minimumVerticalSeparation) return null;
  const masks = [
    new Uint8ClampedArray(width * height * 4),
    new Uint8ClampedArray(width * height * 4),
  ];
  samples.forEach((sample, index) => {
    const offset = sample.pixel * 4;
    masks[assignments[index]].fill(255, offset, offset + 4);
  });
  return masks;
}

function pathCenter(path) {
  return path.points.reduce((center, point) => ({
    x: center.x + point.x / path.points.length,
    y: center.y + point.y / path.points.length,
  }), { x: 0, y: 0 });
}

export function vectorizeSemanticRegions(
  sourceRgba,
  sourceWidth,
  sourceHeight,
  categories,
  width,
  height,
  labels,
  accuracy,
  clothingColors = 'smart',
) {
  const regions = [];
  const classMask = new Uint8ClampedArray(width * height * 4);
  for (let category = 1; category < labels.length; category += 1) {
    classMask.fill(0);
    let pixelCount = 0;
    for (let pixel = 0, offset = 0; pixel < categories.length; pixel += 1, offset += 4) {
      if (categories[pixel] !== category) continue;
      classMask[offset] = 255;
      classMask[offset + 1] = 255;
      classMask[offset + 2] = 255;
      classMask[offset + 3] = 255;
      pixelCount += 1;
    }
    if (pixelCount < 4) continue;
    const vectors = vectorizeMaskContours(classMask, width, height, accuracy, 128);
    if (!vectors.paths.length) continue;
    if (labels[category] === 'clothes') {
      if (clothingColors === 'single') {
        regions.push({
          category,
          component: 0,
          label: labels[category],
          color: sampledMajorityClassColor(
            sourceRgba, sourceWidth, sourceHeight, categories, width, height, category,
          ),
          paths: vectors.paths,
          pointCount: vectors.pointCount,
        });
        continue;
      }
      // Preserve the original semantic geometry whenever garments are already
      // disconnected. RGB subdivision is only needed when two colors touch
      // inside one clothes component; applying it universally lets shadows
      // fragment and visibly shrink shorts or trousers.
      const colorMasks = vectors.paths.length === 1
        ? clothingColorMasks(
          sourceRgba, sourceWidth, sourceHeight, categories, width, height, category,
        )
        : null;
      const colorPaths = colorMasks?.flatMap((mask) => (
        vectorizeMaskContours(mask, width, height, accuracy, 128).paths
      ));
      const paths = [...(colorPaths?.length ? colorPaths : vectors.paths)]
        .sort((first, second) => polygonArea(second.points) - polygonArea(first.points))
        .slice(0, 4)
        .sort((first, second) => {
          const firstCenter = pathCenter(first);
          const secondCenter = pathCenter(second);
          return firstCenter.y - secondCenter.y || firstCenter.x - secondCenter.x;
        });
      const componentPaths = paths.map((path) => [path]);
      componentPaths.forEach((pathsForComponent, component) => {
        regions.push({
          category,
          component,
          label: labels[category],
          color: sampledComponentColor(
            sourceRgba, sourceWidth, sourceHeight, categories, width, height, category, pathsForComponent[0],
          ) || sampledClassColor(sourceRgba, sourceWidth, sourceHeight, categories, width, height, category),
          paths: pathsForComponent,
          pointCount: pathsForComponent.reduce((total, path) => total + path.points.length, 0),
        });
      });
      continue;
    }
    regions.push({
      category,
      component: 0,
      label: labels[category],
      color: sampledClassColor(
        sourceRgba, sourceWidth, sourceHeight, categories, width, height, category,
      ),
      paths: vectors.paths,
      pointCount: vectors.pointCount,
    });
  }
  return regions;
}
