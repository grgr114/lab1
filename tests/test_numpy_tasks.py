import numpy as np
import pytest

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixVectorBatchInput,
    OneHotInput, RandomMatrixInput, RectangleInput, TimeSeriesInput,
)
from numpy_tasks import (
    analyze_time_series, binarize, chess, draw_ellipse, draw_rectangle, matrix_statistics,
    one_hot, plot_matrix_histograms, sum_prod, unique_columns, unique_rows,
)


def test_sum_prod_matches_loop():
    rng = np.random.default_rng(0)
    matrices = rng.normal(size=(4, 3, 3))
    vectors = rng.normal(size=(4, 3, 1))
    expected = sum(a @ x for a, x in zip(matrices, vectors))
    result = sum_prod(MatrixVectorBatchInput(matrices, vectors))
    assert result.shape == (3, 1)
    np.testing.assert_allclose(result, expected)


def test_sum_prod_lists_and_single():
    result = sum_prod(MatrixVectorBatchInput([[[2]]], [[[3]]]))
    np.testing.assert_array_equal(result, [[6]])
    ident = [np.eye(2), 2 * np.eye(2)]
    vecs = [[[1], [2]], [[3], [4]]]
    np.testing.assert_array_equal(sum_prod(MatrixVectorBatchInput(ident, vecs)), [[7], [10]])


def test_sum_prod_bad_shapes():
    with pytest.raises(ValueError):
        sum_prod(MatrixVectorBatchInput(np.ones((2, 2, 2)), np.ones((3, 2, 1))))


def test_binarize():
    m = np.array([[0.1, 0.5], [0.6, 1.0]])
    result = binarize(BinarizeInput(m))
    np.testing.assert_array_equal(result, [[0, 0], [1, 1]])
    np.testing.assert_array_equal(m, [[0.1, 0.5], [0.6, 1.0]])  # исходная не изменилась
    np.testing.assert_array_equal(binarize(BinarizeInput([[1, 2, 3]], threshold=2)), [[0, 0, 1]])
    np.testing.assert_array_equal(binarize(BinarizeInput([[-5]], threshold=-10)), [[1]])


def test_unique_rows_columns():
    m = [[1, 2, 2], [3, 3, 3], [1, 2, 3]]
    assert unique_rows(MatrixInput(m)) == [[1, 2], [3], [1, 2, 3]]
    assert unique_columns(MatrixInput(m)) == [[1, 3], [2, 3], [2, 3]]
    assert unique_rows(MatrixInput([[5]])) == [[5]]
    assert unique_columns(MatrixInput([[1, 1]])) == [[1], [1]]


def test_matrix_statistics():
    stats = matrix_statistics(RandomMatrixInput(200, 300, mean=5, std=2, seed=1))
    assert stats.matrix.shape == (200, 300)
    assert stats.row_means.shape == (200,) and stats.column_means.shape == (300,)
    np.testing.assert_allclose(stats.row_variances, stats.matrix.var(axis=1))
    np.testing.assert_allclose(stats.column_means, stats.matrix.mean(axis=0))
    assert abs(stats.matrix.mean() - 5) < 0.05
    assert abs(stats.matrix.std() - 2) < 0.05
    again = matrix_statistics(RandomMatrixInput(200, 300, mean=5, std=2, seed=1))
    np.testing.assert_array_equal(stats.matrix, again.matrix)


def test_matrix_statistics_one_cell_and_histograms(tmp_path):
    stats = matrix_statistics(RandomMatrixInput(1, 1))
    np.testing.assert_array_equal(stats.row_variances, [0.0])
    path = plot_matrix_histograms(matrix_statistics(RandomMatrixInput(4, 5)), str(tmp_path / "h.png"))
    assert (tmp_path / "h.png").stat().st_size > 0 and path.endswith("h.png")


def test_chess():
    np.testing.assert_array_equal(
        chess(ChessInput(3, 4, 0, 1)),
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
    )
    np.testing.assert_array_equal(chess(ChessInput(1, 1, 7, 9)), [[7]])
    np.testing.assert_array_equal(chess(ChessInput(2, 1, 7, 9)), [[7], [9]])
    assert chess(ChessInput(5, 6, 2.5, -1)).shape == (5, 6)


def test_draw_rectangle():
    red, black = (255, 0, 0), (0, 0, 0)
    img = draw_rectangle(RectangleInput(width=4, height=2, image_height=6, image_width=8,
                                        shape_color=red, background_color=black))
    assert img.shape == (6, 8, 3)
    mask = np.all(img == red, axis=2)
    assert mask.sum() == 8
    rows, cols = np.nonzero(mask)
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (2, 3, 2, 5)
    full = draw_rectangle(RectangleInput(3, 3, 3, 3, red, black))
    assert np.all(full == red)
    with pytest.raises(ValueError):
        draw_rectangle(RectangleInput(10, 1, 3, 3, red, black))


def test_draw_ellipse():
    white, blue = (255, 255, 255), (0, 0, 255)
    img = draw_ellipse(EllipseInput(semi_axis_x=4, semi_axis_y=2, image_height=11, image_width=11,
                                    shape_color=blue, background_color=white))
    assert img.shape == (11, 11, 3)
    mask = np.all(img == blue, axis=2)
    assert mask[5, 5] and mask[5, 1] and mask[5, 9] and mask[3, 5] and mask[7, 5]
    assert not mask[5, 0] and not mask[2, 5] and not mask[0, 0]
    np.testing.assert_array_equal(mask, mask[::-1, ::-1])  # симметрия относительно центра
    assert np.all(img[0, 0] == white)


def test_analyze_time_series():
    values = [1, 3, 2, 4, 1, 1, 5]
    res = analyze_time_series(TimeSeriesInput(values, window=3))
    assert res.mean == pytest.approx(np.mean(values))
    assert res.variance == pytest.approx(np.var(values))
    assert res.std == pytest.approx(np.std(values))
    np.testing.assert_array_equal(res.local_maxima_indices, [1, 3])
    np.testing.assert_array_equal(res.local_minima_indices, [2])  # плато 1,1 — не строгий минимум
    np.testing.assert_allclose(res.moving_average, [2, 3, 7 / 3, 2, 7 / 3])
    assert len(res.moving_average) == len(values) - 3 + 1


def test_analyze_time_series_edges():
    res = analyze_time_series(TimeSeriesInput([5], window=1))
    assert res.variance == 0 and res.local_maxima_indices.size == 0
    np.testing.assert_array_equal(res.moving_average, [5])
    res = analyze_time_series(TimeSeriesInput([1, 2, 3], window=3))
    np.testing.assert_array_equal(res.moving_average, [2])
    for window in (0, 4):
        with pytest.raises(ValueError):
            analyze_time_series(TimeSeriesInput([1, 2, 3], window=window))


def test_one_hot():
    np.testing.assert_array_equal(
        one_hot(OneHotInput([0, 2, 3, 0])),
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
    )
    assert one_hot(OneHotInput([0, 1], class_count=5)).shape == (2, 5)
    np.testing.assert_array_equal(one_hot(OneHotInput([0])), [[1]])
    assert one_hot(OneHotInput([])).shape == (0, 0)
    with pytest.raises(ValueError):
        one_hot(OneHotInput([0, 3], class_count=2))
    with pytest.raises(ValueError):
        one_hot(OneHotInput([-1]))
