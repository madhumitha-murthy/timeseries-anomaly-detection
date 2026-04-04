"""
Tests for src/dataset.py

Covers:
  - train_val_split: size, order, no overlap
  - make_windows: shape, stride, content correctness
  - TimeSeriesDataset: len, reconstruction pair contract
  - get_dataloaders: output shapes and loader properties
  - load_labels: binary label array construction
"""

import numpy as np
import pandas as pd

from dataset import (
    TimeSeriesDataset,
    get_dataloaders,
    load_labels,
    make_windows,
    train_val_split,
)


# ---------------------------------------------------------------------------
# train_val_split
# ---------------------------------------------------------------------------

class TestTrainValSplit:
    def test_default_split_sizes(self):
        data = np.random.randn(1000, 5).astype(np.float32)
        tr, val = train_val_split(data)
        assert len(tr) == 800
        assert len(val) == 200

    def test_custom_val_frac(self):
        data = np.random.randn(500, 3).astype(np.float32)
        tr, val = train_val_split(data, val_frac=0.3)
        assert len(tr) == 350
        assert len(val) == 150

    def test_sizes_sum_to_total(self):
        data = np.random.randn(333, 2).astype(np.float32)
        tr, val = train_val_split(data, val_frac=0.2)
        assert len(tr) + len(val) == len(data)

    def test_temporal_order_preserved(self):
        """Earlier timesteps must be in train, later in val — no shuffling."""
        data = np.arange(100).reshape(100, 1).astype(np.float32)
        tr, val = train_val_split(data, val_frac=0.2)
        np.testing.assert_array_equal(tr, data[:80])
        np.testing.assert_array_equal(val, data[80:])

    def test_no_overlap(self):
        """No row should appear in both train and val."""
        data = np.arange(200).reshape(200, 1).astype(np.float32)
        tr, val = train_val_split(data, val_frac=0.25)
        # If there's no overlap, max of train < min of val (data is monotonic)
        assert tr.max() < val.min()

    def test_feature_dimension_unchanged(self):
        data = np.random.randn(100, 7).astype(np.float32)
        tr, val = train_val_split(data, val_frac=0.2)
        assert tr.shape[1] == 7
        assert val.shape[1] == 7


# ---------------------------------------------------------------------------
# make_windows
# ---------------------------------------------------------------------------

class TestMakeWindows:
    def test_shape_stride_1(self):
        data = np.random.randn(100, 5).astype(np.float32)
        windows = make_windows(data, window_size=10, stride=1)
        # n_windows = (100 - 10) // 1 + 1 = 91
        assert windows.shape == (91, 10, 5)

    def test_shape_stride_5(self):
        data = np.random.randn(100, 5).astype(np.float32)
        windows = make_windows(data, window_size=10, stride=5)
        # n_windows = (100 - 10) // 5 + 1 = 19
        assert windows.shape == (19, 10, 5)

    def test_window_content_is_correct(self):
        """Each window should be a contiguous slice of the original data."""
        data = np.arange(20).reshape(20, 1).astype(np.float32)
        windows = make_windows(data, window_size=5, stride=1)
        np.testing.assert_array_equal(windows[0, :, 0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(windows[1, :, 0], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(windows[-1, :, 0], [15, 16, 17, 18, 19])

    def test_dtype_is_float32(self):
        data = np.random.randn(50, 3).astype(np.float64)
        windows = make_windows(data, window_size=5)
        assert windows.dtype == np.float32

    def test_single_feature(self):
        data = np.random.randn(50, 1).astype(np.float32)
        windows = make_windows(data, window_size=10, stride=1)
        assert windows.shape == (41, 10, 1)

    def test_window_size_equals_length(self):
        data = np.random.randn(10, 3).astype(np.float32)
        windows = make_windows(data, window_size=10, stride=1)
        assert windows.shape == (1, 10, 3)


# ---------------------------------------------------------------------------
# TimeSeriesDataset
# ---------------------------------------------------------------------------

class TestTimeSeriesDataset:
    def test_len(self):
        windows = np.random.randn(42, 10, 3).astype(np.float32)
        ds = TimeSeriesDataset(windows)
        assert len(ds) == 42

    def test_getitem_returns_reconstruction_pair(self):
        """__getitem__ must return (window, window) — input equals target."""
        windows = np.random.randn(10, 5, 3).astype(np.float32)
        ds = TimeSeriesDataset(windows)
        x, y = ds[0]
        assert x.shape == y.shape
        np.testing.assert_array_equal(x.numpy(), y.numpy())

    def test_getitem_shape(self):
        windows = np.random.randn(20, 30, 5).astype(np.float32)
        ds = TimeSeriesDataset(windows)
        x, _ = ds[0]
        assert x.shape == (30, 5)

    def test_all_items_are_identical_input_target(self):
        windows = np.random.randn(5, 8, 2).astype(np.float32)
        ds = TimeSeriesDataset(windows)
        for i in range(len(ds)):
            x, y = ds[i]
            np.testing.assert_array_equal(x.numpy(), y.numpy())


# ---------------------------------------------------------------------------
# get_dataloaders
# ---------------------------------------------------------------------------

class TestGetDataloaders:
    def test_window_shapes(self, train_data, test_data):
        _, _, tr_w, te_w = get_dataloaders(train_data, test_data, window_size=10, batch_size=16)
        assert tr_w.shape[1] == 10   # window_size
        assert tr_w.shape[2] == 5    # features
        assert te_w.shape[1] == 10
        assert te_w.shape[2] == 5

    def test_train_loader_shuffled(self, train_data, test_data):
        """train_loader must have shuffle=True (drop_last also True)."""
        train_loader, _, _, _ = get_dataloaders(train_data, test_data, window_size=10, batch_size=16)
        assert train_loader.dataset is not None
        # DataLoader with shuffle=True has a RandomSampler
        from torch.utils.data import RandomSampler
        assert isinstance(train_loader.sampler, RandomSampler)

    def test_test_loader_not_shuffled(self, train_data, test_data):
        """test_loader must be sequential to preserve temporal order for scoring."""
        from torch.utils.data import SequentialSampler
        _, test_loader, _, _ = get_dataloaders(train_data, test_data, window_size=10, batch_size=16)
        assert isinstance(test_loader.sampler, SequentialSampler)

    def test_batch_feature_dim_matches(self, train_data, test_data):
        train_loader, _, _, _ = get_dataloaders(train_data, test_data, window_size=10, batch_size=8)
        x, y = next(iter(train_loader))
        assert x.shape[-1] == train_data.shape[1]  # features match


# ---------------------------------------------------------------------------
# load_labels
# ---------------------------------------------------------------------------

class TestLoadLabels:
    def test_returns_correct_length(self, tmp_path):
        csv_path = tmp_path / "labeled_anomalies.csv"
        pd.DataFrame({
            "chan_id": ["X-1"],
            "anomaly_sequences": ["[[10, 20]]"],
        }).to_csv(csv_path, index=False)
        labels = load_labels("X-1", test_length=100, data_dir=str(tmp_path))
        assert len(labels) == 100

    def test_anomaly_region_is_flagged(self, tmp_path):
        csv_path = tmp_path / "labeled_anomalies.csv"
        pd.DataFrame({
            "chan_id": ["X-1"],
            "anomaly_sequences": ["[[10, 20]]"],
        }).to_csv(csv_path, index=False)
        labels = load_labels("X-1", test_length=50, data_dir=str(tmp_path))
        assert labels[10] == 1
        assert labels[20] == 1
        assert labels[9] == 0
        assert labels[21] == 0

    def test_multiple_anomaly_segments(self, tmp_path):
        csv_path = tmp_path / "labeled_anomalies.csv"
        pd.DataFrame({
            "chan_id": ["X-1"],
            "anomaly_sequences": ["[[5, 10], [30, 35]]"],
        }).to_csv(csv_path, index=False)
        labels = load_labels("X-1", test_length=50, data_dir=str(tmp_path))
        assert labels[5:11].sum() == 6
        assert labels[30:36].sum() == 6
        assert labels[11:30].sum() == 0

    def test_unknown_channel_returns_zeros(self, tmp_path):
        csv_path = tmp_path / "labeled_anomalies.csv"
        pd.DataFrame({
            "chan_id": ["X-1"],
            "anomaly_sequences": ["[[0, 10]]"],
        }).to_csv(csv_path, index=False)
        labels = load_labels("UNKNOWN", test_length=50, data_dir=str(tmp_path))
        assert labels.sum() == 0
        assert len(labels) == 50
