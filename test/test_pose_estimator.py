import numpy as np

from jed_teleop.pose_estimator import PoseEstimator

class TestPoseEstimator(PoseEstimator):
    def run(self):
        pass

def test_set_position_and_update_deltas():
    estimator = TestPoseEstimator()
    position = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0])
    delta = np.array([0.2, 0.3, 0.1, 0.4, 0.5, 0.7, 0.0])
    estimator.set_position_and_update_deltas(position)
    position += delta
    estimator.set_position_and_update_deltas(position)
    position -= 2*delta
    estimator.set_position_and_update_deltas(position)
    actual_delta = estimator.get_deltas()
    expected_delta = -delta
    expected_delta[-1] = 1.0 # gripper state is not relative
    assert np.linalg.norm(actual_delta - expected_delta) < 1e-12

def test_set_position_and_update_deltas_with_intermediate_calls():
    estimator = TestPoseEstimator()
    position = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0])
    delta = np.array([0.2, 0.3, 0.1, 0.4, 0.5, 0.7, 0.0])
    estimator.set_position_and_update_deltas(position)
    position += delta
    estimator.set_position_and_update_deltas(position)
    actual_delta = estimator.get_deltas()
    expected_delta = delta.copy()
    expected_delta[-1] = -1.0
    assert np.linalg.norm(actual_delta - expected_delta) < 1e-12

    position -= 2 * delta
    estimator.set_position_and_update_deltas(position)

    actual_delta = estimator.get_deltas()
    expected_delta = -2*delta
    expected_delta[-1] = -1.0
    assert np.linalg.norm(actual_delta - expected_delta) < 1e-12

def test_is_paused():
    estimator = TestPoseEstimator()
    estimator.set_position_and_update_deltas(np.array([1.0, 2, 3]))
    estimator.is_paused = True
    estimator.set_position_and_update_deltas(np.array([1.0, 2, 4]))
    assert estimator.latest_deltas is None

    # the hand is moved to somewhere else
    estimator.set_position_and_update_deltas(np.array([0.0, -1.0, 0.0]))
    # when pause is resumed, the deltas should be calculated from the new position.
    estimator.is_paused = False
    estimator.set_position_and_update_deltas(np.array([1.0, -1.0, 0.0]))
    actual_delta = estimator.get_deltas()
    expected_delta = np.array([1.0, 0.0, 0.0])
    assert np.linalg.norm(actual_delta - expected_delta) < 1e-12