"""Module providing a Gaze Events."""


class Gevent:
    """Class representing gaze event, with tracked points scaled to screen, blink and fixation."""

    def __init__(self,
                 point,
                 blink,
                 fixation,
                 l_eye = None,
                 r_eye = None,
                 screen_man = None,
                 roi = None,
                 edges = None,
                 cluster = None,
                 context = None,
                 saccades = False,
                 sub_frame = None):

        self.point = point
        self.blink = blink
        self.fixation = fixation
        self.saccades = saccades

        # ALL DEBUG DATA
        self.roi = roi
        self.edges = edges
        self.l_eye = l_eye
        self.r_eye = r_eye
        self.cluster = cluster
        self.context = context
        self.screen_man = screen_man
        self.sub_frame = sub_frame


class Cevent:
    """Class representing gaze event, with tracked points scaled to screen, blink and fixation."""

    def __init__(self,
                 point,
                 acceptance_radius,
                 calibration_radius,
                 calibration = False,
                 calibration_target = None,
                 calibration_point_idx = 0,
                 calibration_points = 0,
                 is_last_sample = False):

        self.point = point
        self.acceptance_radius = acceptance_radius
        self.calibration_radius = calibration_radius
        self.calibration = calibration
        self.calibration_target = calibration_target
        self.calibration_point_idx = calibration_point_idx
        self.calibration_points = calibration_points
        self.is_last_sample = is_last_sample
