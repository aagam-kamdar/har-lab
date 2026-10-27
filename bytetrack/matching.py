import cv2
import numpy as np
import time

# ---------------------------------------------------------------------------
# scipy-free replacements  (scipy has a broken install in this environment)
# ---------------------------------------------------------------------------

def cdist(XA, XB, metric='cosine'):
    """Minimal cdist replacement supporting 'cosine' and 'euclidean'."""
    XA = np.asarray(XA, dtype=float)
    XB = np.asarray(XB, dtype=float)
    if metric == 'cosine':
        # cosine distance = 1 - cosine_similarity
        na = np.linalg.norm(XA, axis=1, keepdims=True)
        nb = np.linalg.norm(XB, axis=1, keepdims=True)
        na = np.where(na < 1e-12, 1e-12, na)
        nb = np.where(nb < 1e-12, 1e-12, nb)
        sim = (XA / na) @ (XB / nb).T
        return np.maximum(0.0, 1.0 - sim)
    elif metric == 'euclidean':
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a@b.T
        aa = np.sum(XA ** 2, axis=1)[:, None]
        bb = np.sum(XB ** 2, axis=1)[None, :]
        dist2 = aa + bb - 2.0 * (XA @ XB.T)
        return np.sqrt(np.maximum(0.0, dist2))
    else:
        # Generic fallback
        out = np.zeros((len(XA), len(XB)), dtype=float)
        for i, a in enumerate(XA):
            for j, b in enumerate(XB):
                out[i, j] = np.linalg.norm(a - b)
        return out


class _SparseDummy:
    """Minimal sparse matrix replacement for merge_matches."""
    def __init__(self, data, coords, shape):
        self._data = np.array(data, dtype=float)
        self._row = np.array(coords[0], dtype=int)
        self._col = np.array(coords[1], dtype=int)
        self._shape = shape

    def __matmul__(self, other):
        """Dense multiplication for small matrices used in merge_matches."""
        R = np.zeros(self._shape, dtype=float)
        for d, r, c in zip(self._data, self._row, self._col):
            R[r, c] = d
        S = np.zeros(other._shape, dtype=float)
        for d, r, c in zip(other._data, other._row, other._col):
            S[r, c] = d
        result_dense = R @ S
        # Return an object with nonzero()
        class _Result:
            def __init__(self, dense):
                self._d = dense
            def nonzero(self):
                nz = np.argwhere(self._d != 0)
                if len(nz) == 0:
                    return (np.array([], dtype=int), np.array([], dtype=int))
                return (nz[:, 0], nz[:, 1])
        return _Result(result_dense)


class _SparseModule:
    @staticmethod
    def coo_matrix(arg, shape):
        data, coords = arg
        return _SparseDummy(data, coords, shape)


class _ScipyStub:
    sparse = _SparseModule()


scipy = _ScipyStub()

from . import kalman_filter

# Attempt to import cython_bbox; fallback to pure NumPy if unavailable on Windows
try:
    from cython_bbox import bbox_overlaps as bbox_ious
except ImportError:
    def bbox_ious(boxes, query_boxes):
        """
        Pure NumPy replacement for cython_bbox bbox_overlaps.
        boxes: (N, 4) in [x1, y1, x2, y2]
        query_boxes: (K, 4) in [x1, y1, x2, y2]
        Returns: (N, K) IoU matrix
        """
        boxes = np.ascontiguousarray(boxes, dtype=float)
        query_boxes = np.ascontiguousarray(query_boxes, dtype=float)
        n = boxes.shape[0]
        k = query_boxes.shape[0]
        if n == 0 or k == 0:
            return np.zeros((n, k), dtype=float)

        b1 = boxes[:, None, :]       # (N, 1, 4)
        b2 = query_boxes[None, :, :]  # (1, K, 4)

        ix1 = np.maximum(b1[..., 0], b2[..., 0])
        iy1 = np.maximum(b1[..., 1], b2[..., 1])
        ix2 = np.minimum(b1[..., 2], b2[..., 2])
        iy2 = np.minimum(b1[..., 3], b2[..., 3])

        iw = np.maximum(0.0, ix2 - ix1)
        ih = np.maximum(0.0, iy2 - iy1)
        inter = iw * ih

        area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
        area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
        union = area1 + area2 - inter

        iou = np.zeros((n, k), dtype=float)
        valid = union > 0
        iou[valid] = inter[valid] / union[valid]
        return iou


# Attempt to import lap; fallback to scipy linear_sum_assignment if lap is missing
try:
    import lap

    def linear_assignment(cost_matrix, thresh):
        if cost_matrix.size == 0:
            return (
                np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])),
            )
        matches, unmatched_a, unmatched_b = [], [], []
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatched_a, unmatched_b

except ImportError:
    from scipy.optimize import linear_sum_assignment

    def linear_assignment(cost_matrix, thresh):
        if cost_matrix.size == 0:
            return (
                np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])),
            )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= thresh:
                matches.append([r, c])
        matches = np.asarray(matches, dtype=int).reshape(-1, 2) if matches else np.empty((0, 2), dtype=int)
        matched_a = set(matches[:, 0]) if len(matches) > 0 else set()
        matched_b = set(matches[:, 1]) if len(matches) > 0 else set()
        unmatched_a = np.array([i for i in range(cost_matrix.shape[0]) if i not in matched_a], dtype=int)
        unmatched_b = np.array([j for j in range(cost_matrix.shape[1]) if j not in matched_b], dtype=int)
        return matches, unmatched_a, unmatched_b


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 @ M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type btlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious_mat = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious_mat.size == 0:
        return ious_mat

    ious_mat = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious_mat


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
