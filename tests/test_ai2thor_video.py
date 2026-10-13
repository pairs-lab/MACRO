import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from simulations.ai2thor.video import MissionVideoRecorder


class VideoRecorderTests(unittest.TestCase):
    def test_elapsed_planning_waits_keep_previous_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            now = [0.]
            writer = Mock()
            writer.isOpened.return_value = True
            first = Mock(shape=(720, 1280, 3))
            first.copy.return_value = first
            second = Mock(shape=(720, 1280, 3))
            second.copy.return_value = second
            with patch('simulations.ai2thor.video.cv2.VideoWriter', return_value=writer):
                recorder = MissionVideoRecorder(Path(directory) / 'test.mp4', fps=10.,
                                                 clock=lambda: now[0])
                recorder.update(first)
                now[0] = .3
                recorder.update(second)
                self.assertTrue(all(c.args[0] is first for c in writer.write.call_args_list))
                now[0] = .5
                recorder.update()
                self.assertIs(writer.write.call_args.args[0], second)
                recorder.close()
                recorder.close()
                writer.release.assert_called_once()

    def test_existing_video_is_not_overwritten(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as existing:
            with self.assertRaises(FileExistsError):
                MissionVideoRecorder(existing.name)


if __name__ == '__main__':
    unittest.main()
