import math


def cluster_sparse_vectors(matrix, cluster_count, max_iterations=20):
    row_count = matrix.shape[0]
    if row_count == 0:
        return []

    rows = []
    for index in range(row_count):
        row = matrix.getrow(index)
        rows.append(
            {
                int(feature): float(value)
                for feature, value in zip(row.indices, row.data)
            }
        )
    if not any(rows):
        return [0] * row_count

    cluster_count = min(max(1, cluster_count), row_count)
    centers = [rows[0]]
    while len(centers) < cluster_count:
        farthest_index = None
        farthest_similarity = float("inf")
        for index, row in enumerate(rows):
            nearest_similarity = max(_cosine_similarity(row, center) for center in centers)
            if nearest_similarity < farthest_similarity:
                farthest_index = index
                farthest_similarity = nearest_similarity
        if farthest_index is None or farthest_similarity >= 1.0 - 1e-12:
            break
        centers.append(rows[farthest_index])

    labels = [-1] * row_count
    for _ in range(max_iterations):
        new_labels = [
            max(
                range(len(centers)),
                key=lambda cluster: _cosine_similarity(row, centers[cluster]),
            )
            for row in rows
        ]
        if new_labels == labels:
            break
        labels = new_labels

        totals = [{} for _ in centers]
        for row, label in zip(rows, labels):
            for feature, value in row.items():
                totals[label][feature] = totals[label].get(feature, 0.0) + value

        new_centers = []
        for total, previous_center in zip(totals, centers):
            norm = math.sqrt(sum(value * value for value in total.values()))
            if not norm:
                new_centers.append(previous_center)
                continue
            new_centers.append(
                {feature: value / norm for feature, value in total.items()}
            )
        centers = new_centers

    return [
        max(
            range(len(centers)),
            key=lambda cluster: _cosine_similarity(row, centers[cluster]),
        )
        for row in rows
    ]


def _cosine_similarity(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(feature, 0.0) for feature, value in left.items())
