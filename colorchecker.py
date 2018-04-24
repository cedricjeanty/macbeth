########################################################################################################################

import numpy as np
import cv2

import argparse
from collections import namedtuple
from itertools import combinations
import logging
import os
import sys
import traceback

# see tau manifest
TAU = np.pi * 2

# MSER detector params
MSER_PARAMS = dict(_delta=2, _max_variation=1.0)

# square acceptance criteria
MIN_SQUARENESS_RATIO = 0.95
MAX_ASPECT_RATIO = 1.2

# grid connectivity criterias
MAX_GRID_INTERVAL_RATIO = 1.5
MAX_GRID_CELL_AREA_RATIO = 1.5
MAX_GRID_ANGLE_DEVIATION_DEGREES = 5
MIN_GRID_CELL_COUNT = 10

MACBETH_BGR = [
    (67,   81, 115),  (129, 149, 196),  (157, 123,  93),  (65,  108,  90),  (176, 129, 130),  (171, 191,  99),
    (45,  123, 220),  (168,  92,  72),  ( 98,  84, 195),  (105,  59,  91),  ( 62, 189, 160),  ( 41, 161, 229),
    (147,  62,  43),  (72,  149,  71),  ( 56,  48, 176),  (22,  200, 238),  (150,  84, 188),  (166, 136,   0),
    (240, 245, 245),  (201, 201, 200),  (161, 161, 160),  (121, 121, 120),  ( 85,  84,  83),  ( 50, 50,   50),
]
MACBETH_BGR = np.array(MACBETH_BGR).reshape(4, 6, 3).astype(np.uint8)
MACBETH_LAB = cv2.cvtColor(MACBETH_BGR, cv2.COLOR_BGR2LAB)

RotatedRect = namedtuple("RotatedRect", "center size angle")
ColorBox = namedtuple("ColorBox", "index rect")

########################################################################################################################

def findSquares(msers, canvas=None):
    """ Find squares within MSERs using simple square tests
    """

    msers.sort(key=lambda points: len(points), reverse=True)

    squares = []
    for i, points in enumerate(msers):
        box = ColorBox(i, RotatedRect(*cv2.minAreaRect(points)))

        if not all(box.rect.size): continue
        if len(points) / (box.rect.size[0] * box.rect.size[1]) < MIN_SQUARENESS_RATIO: continue
        if max(box.rect.size) / min(box.rect.size) > MAX_ASPECT_RATIO: continue

        # check that this box don't intersect with accepted boxes

        separated = True
        for accepted in squares:
            retval, _ = cv2.rotatedRectangleIntersection(accepted.rect, box.rect)
            if retval != cv2.INTERSECT_NONE:
                separated = False
                break
        if not separated: continue

        squares.append(box)

        if canvas is not None:
            canvas[points[:, 1], points[:, 0]] = 128

    return squares

########################################################################################################################

def findGrids(squares, canvas=None):
    """ Find square grids
    """

    adjacency = np.zeros((len(squares),) * 2, dtype=np.float64)

    for i, j in combinations(range(len(squares)), 2):
        b1, b2 = squares[i].rect, squares[j].rect

        # are boxes close to each other?
        dist = np.linalg.norm(np.array(b1.center) - b2.center)
        if dist > max(max(b1.size), max(b2.size)) * MAX_GRID_INTERVAL_RATIO:
            continue

        # do boxes have similar areas?
        area1 = b1.size[0] * b1.size[1]
        area2 = b2.size[0] * b2.size[1]
        if max(area1, area2) / min(area1, area2) > MAX_GRID_CELL_AREA_RATIO:
            continue

        # do boxes have similar orientation?
        v1 = np.array([np.cos(np.deg2rad(b1.angle)), np.sin(np.deg2rad(b1.angle))])
        v2 = np.array([np.cos(np.deg2rad(b2.angle)), np.sin(np.deg2rad(b2.angle))])
        diff = np.rad2deg(np.arccos(v1.dot(v2)))
        if diff > MAX_GRID_ANGLE_DEVIATION_DEGREES and abs(diff - 90) > MAX_GRID_ANGLE_DEVIATION_DEGREES:
            continue

        adjacency[i, j] = adjacency[j, i] = 1

        if canvas is not None:
            cv2.line(canvas, tuple(map(int, b1.center)), tuple(map(int, b2.center)), 255, lineType=cv2.LINE_AA)

    # find connected components using spectral clustering algorithm

    # calculate eigenvalues/eigenvectors of Laplacian matrix
    degree = np.diag(cv2.reduce(adjacency, 0, cv2.REDUCE_SUM).flatten())
    laplacian = degree - adjacency
    retval, eigenVal, eigenVec = cv2.eigen(laplacian)
    assert retval

    # detect the number of clusters
    clustersNum = sum(eigenVal.flatten() < 1e-7)
    assert clustersNum >= 1

    # apply k-means clustering to find all connected components
    U = eigenVec[-clustersNum:].T.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 10, 0)
    _, labels, _ = cv2.kmeans(U, clustersNum, None, criteria, attempts=1, flags=cv2.KMEANS_PP_CENTERS)

    labels = labels.flatten()
    grids = [np.argwhere(labels == i).flatten().tolist() for i in range(clustersNum)]
    grids = list(filter(lambda value: len(value) >= MIN_GRID_CELL_COUNT, grids))
    return grids

########################################################################################################################

def restoreGridContour(squares, grid, canvas=None):
    """ Restore grid contour
    """

    # find outer grid cells

    centers = []
    for i in grid:
        centers.append(np.array(squares[i].rect.center).astype(int))
    outerRect = RotatedRect(*cv2.minAreaRect(np.array(centers)))

    outer = []
    for i in grid:
        retval, _ = cv2.rotatedRectangleIntersection(squares[i].rect, outerRect)
        if retval == cv2.INTERSECT_PARTIAL:
            outer.append(i)

    # connect outer grid cells in chain

    center = np.mean([squares[i].rect.center for i in outer], axis=0)
    vectors = (squares[i].rect.center - center for i in outer)
    angles = [cv2.fastAtan2(v[1], v[0]) for v in vectors]
    chain = [outer[i] for i in np.argsort(angles)]

    # find outer corner points

    outer = []
    contour = np.array([squares[i].rect.center for i in chain]).astype(int)
    minDistSum = 0
    for i in chain:
        corners = cv2.boxPoints(squares[i].rect).astype(int)
        minDist = max(squares[i].rect.size) / 3
        minDistSum += minDist
        for point in corners:
            if cv2.pointPolygonTest(contour, tuple(point), measureDist=True) < -minDist:
                outer.append(point)
    minDist = minDistSum / len(chain)

    # connect outer corner points in chain

    vectors = (point - center for point in outer)
    angles = [cv2.fastAtan2(v[1], v[0]) for v in vectors]
    chain = np.array([outer[i] for i in np.argsort(angles)]).reshape(-1, 2)

    # approximate outer contour to extract corners

    approx = cv2.approxPolyDP(chain, epsilon=minDist, closed=True).reshape(-1, 2).tolist()
    vectors = (point - center for point in approx)
    angles = [cv2.fastAtan2(v[1], v[0]) for v in vectors]
    approx = [approx[i] for i in np.argsort(angles)]
    approx.append(approx[0])

    # fit lines between outer contour corners

    lines = []
    for p1, p2 in zip(approx, approx[1:]):
        c1, c2 = np.where(np.all(chain == p1, axis=1))[0][0], np.where(np.all(chain == p2, axis=1))[0][0]
        if c2 < c1: c2 += len(chain)

        points = np.array([chain[i % len(chain)] for i in range(c1, c2 + 1)])
        line = cv2.fitLine(points, cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
        vdir, vpos = line[:2], line[2:]

        p1, p2 = vpos - vdir, vpos + vdir
        line = np.cross(np.append(p1, 1), np.append(p2, 1))
        lines.append(line)

    # find all intersection points between lines

    intersections = []
    lines += lines[:2]
    for l1, l2, l3 in zip(lines, lines[1:], lines[2:]):
        for z1, z2 in [(l1, l2), (l1, l3)]:
            cross = np.cross(z1, z2)
            if cross[-1] != 0:
                cross = (cross / cross[-1])[:2].astype(int)
                intersections.append(cross)

    # find all intersection points that are the closest to initial rotated rectangle corners

    corners = []
    rect = cv2.minAreaRect(np.array(approx))
    for point in cv2.boxPoints(rect):
        index = np.argmin([np.linalg.norm(point - cross) for cross in intersections])
        corners.append(intersections[index])
    corners = np.array(corners)

    if canvas is not None:
        cv2.polylines(canvas, [corners], isClosed=True, color=255, thickness=2, lineType=cv2.LINE_AA)

    return corners

########################################################################################################################

def findBestGridCells(image, squares, grids, canvas):
    """ Find best grid cells that has colors matching ColorChecker samples
    """

    h, w = MACBETH_LAB.shape[:2]
    boxPoints = np.array([0, 0, w, 0, w, h, 0, h]).reshape(4, 2).astype(np.float32)
    boxCells = np.array(list(np.ndindex(h, w)), dtype=np.float32)[:, ::-1] + 0.5

    BestGrid = namedtuple("BestGrid", "grid cells transform cost")
    best = BestGrid(None, None, None, np.inf)

    for grid in grids:
        corners = restoreGridContour(squares, grid, canvas).astype(np.float32)
        assert len(corners) == 4

        # find the most appropriate orientation

        for i in range(4):

            # find perspective transform matrix
            M = cv2.getPerspectiveTransform(boxPoints, corners)

            # find color cells centers
            cells = cv2.transform(boxCells.reshape(-1, 1, 2), M)
            cells = cv2.convertPointsFromHomogeneous(cells).astype(int).reshape(-1, 2)

            # find sum of distances from color cells samples to expected color
            bgr = image[cells[:, 1], cells[:, 0]].reshape(-1, 1, 3)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(int).reshape(MACBETH_LAB.shape)
            dist = np.sqrt(np.sum((MACBETH_LAB.astype(int) - lab) ** 2, axis=2))
            cost = np.sum(dist)

            if cost < best.cost:
                best = BestGrid(grid, cells, M, cost)

            corners = np.roll(corners, 1, axis=0)

    grid, cells, transform, _ = best
    assert grid is not None

    # find corresponding points between transformed and source grid cells centers
    srcPoints, dstPoints = [], []
    for i in grid:
        center = squares[i].rect.center
        dist = np.sqrt(np.sum((cells - center) ** 2, axis=1))
        srcPoints.append(cells[np.argmin(dist)])
        dstPoints.append(center)

    # find homography matrix to minimize cells positioning error
    srcPoints, dstPoints = np.array(srcPoints, dtype=np.float32), np.array(dstPoints, dtype=np.float32)
    M, _ = cv2.findHomography(srcPoints, dstPoints)
    cells = cv2.transform(boxCells.reshape(-1, 1, 2), M.dot(transform))
    cells = cv2.convertPointsFromHomogeneous(cells).astype(int).reshape(-1, 2)

    return cells

########################################################################################################################

def setup_opts():

    # read command line arguments
    opts = argparse.ArgumentParser(description="find the Macbeth ColorChecker chart in an image")
    opts.add_argument("image", help="input image")
    opts.add_argument("--show-debug", action="store_true", help="show debug window")

    return opts

def pixel_color(image, pixel, sample_region=3):
    min_x = max(pixel[0] - sample_region, 0)
    max_x = min(pixel[0] + sample_region, image.shape[1])
    min_y = max(pixel[1] - sample_region, 0)
    max_y = min(pixel[1] + sample_region, image.shape[0])
    pixels = image[min_x:max_x, min_y:max_y, :]
    pixels = np.reshape(pixels, (-1,3))
    color = np.median(pixels, axis=0)
    return color


def find_macbeth(image):

    if image.shape[2] != 3:
        raise RuntimeError("image is expected to have three channels")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canvas = np.zeros(image.shape[:2], dtype=np.uint8)

    mser = cv2.MSER_create(**MSER_PARAMS)
    regions, _ = mser.detectRegions(grayscale)
    logging.info("found {} MSERs".format(len(regions)))

    squares = findSquares(regions, canvas)
    logging.info("found {} squares".format(len(squares)))

    grids = findGrids(squares, canvas)
    logging.info("found {} grid{}".format(len(grids), "" if len(grids) == 1 else "s"))

    if not grids:
        raise Exception("No grids found")
    
    cells = findBestGridCells(image, squares, grids, canvas)

    return cells, canvas


if __name__ == "__main__":
    # setup logs
    logging.getLogger().setLevel(logging.DEBUG)
    cout = logging.StreamHandler(sys.stdout)
    cout.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(cout)

    opts = setup_opts()
    params = vars(opts.parse_args())

    showDebug = params["show_debug"]

    logging.info("read image from {}".format(params["image"]))

    image = cv2.imread(params["image"], cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to read {}".format(params["image"]))

    logging.info("image is {0[1]}x{0[0]}".format(image.shape))

    try:
        cells, canvas = find_macbeth(image)

        if showDebug:
            temp = image.copy()
            for point in cells:
                cv2.circle(temp, tuple(point), 4, (255, 255, 255), cv2.FILLED)
                cv2.circle(temp, tuple(point), 2, (0, 0, 0), cv2.FILLED)
                cv2.circle(canvas, tuple(point), 4, 255, 1)
            cv2.circle(temp, tuple(cells[0]), 10, (255, 255, 255))
            cv2.imshow("image", temp)

        logging.info("done")

        if showDebug:
            cv2.imshow("debug", canvas)
            cv2.waitKey()

    except SystemExit: pass

    except BaseException as e:
        logging.error(e)
        print("")
        traceback.print_exc()

########################################################################################################################