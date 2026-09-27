import unittest

import numpy as np
import pandas as pd

import validate_pumps
from test_pumps import make_frame
import pumps_data


class StuckTest(unittest.TestCase):
    def test_sensor_frozen_at_pump_normal_median_only(self):
        frame = pumps_data.reconstruct(make_frame({'NPV_2_1': [12], 'NPV_2_2': [12]}, normals_by_pump=3))
        result = validate_pumps.stuck(frame, 'motor_current')
        for pump in ('NPV_2_1', 'NPV_2_2'):
            expected = frame[(frame['trajectory'] < 0) & (frame['pump_id'] == pump)]['motor_current'].median()
            self.assertTrue(np.allclose(result.loc[result['pump_id'] == pump, 'motor_current'], expected))
        pd.testing.assert_series_equal(result['pressure_suction'], frame['pressure_suction'])
        self.assertFalse(np.allclose(frame['motor_current'], result['motor_current']))


if __name__ == '__main__':
    unittest.main()
