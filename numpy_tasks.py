"""Решения задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("matrices must have shape (p, n, n)")
    p, n, _ = matrices.shape
    if vectors.shape[0] != p or vectors.size != p * n:
        raise ValueError("vectors must have shape (p, n, 1)")
    vectors = vectors.reshape(p, n, 1)
    # A_1 x_1 + ... + A_p x_p  ==  сумма по p и по j от A[p, i, j] * x[p, j, 0]
    return np.einsum("pij,pjk->ik", matrices, vectors)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (np.asarray(matrix) > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    return [np.unique(column).tolist() for column in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    if rows <= 0 or columns <= 0:
        raise ValueError("matrix dimensions must be positive")
    if std < 0:
        raise ValueError("std must be non-negative")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def plot_matrix_histograms(stats: MatrixStatistics, path: str = "histograms.png", bins: int = 10) -> str:
    """Гистограммы значений каждой строки и каждого столбца матрицы."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(stats.matrix)
    fig, (ax_rows, ax_cols) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, row in enumerate(matrix):
        ax_rows.hist(row, bins=bins, alpha=0.5, label=f"строка {i}")
    ax_rows.set_title("Значения по строкам")
    for j, column in enumerate(matrix.T):
        ax_cols.hist(column, bins=bins, alpha=0.5, label=f"столбец {j}")
    ax_cols.set_title("Значения по столбцам")
    for ax in (ax_rows, ax_cols):
        ax.set_xlabel("значение")
        ax.set_ylabel("частота")
        if len(ax.get_legend_handles_labels()[0]) <= 10:
            ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    if rows < 0 or columns < 0:
        raise ValueError("matrix dimensions must be non-negative")
    parity = np.indices((rows, columns)).sum(axis=0) % 2
    return np.where(parity == 0, first, second)


def _background(image_height: int, image_width: int, color) -> np.ndarray:
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")
    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    if width < 0 or height < 0:
        raise ValueError("rectangle size must be non-negative")
    if width > image_width or height > image_height:
        raise ValueError("rectangle does not fit into the image")
    image = _background(image_height, image_width, background_color)
    top = (image_height - height) // 2
    left = (image_width - width) // 2
    image[top:top + height, left:left + width] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    if semi_axis_x <= 0 or semi_axis_y <= 0:
        raise ValueError("semi-axes must be positive")
    image = _background(image_height, image_width, background_color)
    y, x = np.ogrid[:image_height, :image_width]
    x0, y0 = (image_width - 1) / 2, (image_height - 1) / 2
    mask = ((x - x0) / semi_axis_x) ** 2 + ((y - y0) / semi_axis_y) ** 2 <= 1
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or series.size == 0:
        raise ValueError("values must be a non-empty one-dimensional series")
    if not 1 <= window <= series.size:
        raise ValueError("window must be between 1 and len(values)")

    middle, left, right = series[1:-1], series[:-2], series[2:]
    maxima = np.flatnonzero((middle > left) & (middle > right)) + 1
    minima = np.flatnonzero((middle < left) & (middle < right)) + 1

    cumulative = np.concatenate(([0.0], np.cumsum(series)))
    moving_average = (cumulative[window:] - cumulative[:-window]) / window

    variance = float(series.var())
    return TimeSeriesStatistics(
        mean=float(series.mean()),
        variance=variance,
        std=float(np.sqrt(variance)),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional vector")
    if labels.size and (not np.all(labels == np.floor(labels)) or labels.min() < 0):
        raise ValueError("labels must be non-negative integers")
    labels = labels.astype(int)
    needed = int(labels.max()) + 1 if labels.size else 0
    if class_count is None:
        class_count = needed
    elif class_count < needed:
        raise ValueError("class_count is less than the number of classes in labels")
    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels] = 1
    return result
