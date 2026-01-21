from utils import clamp_int

class FollowController:
    def __init__(self, w, h,
                 max_lr, max_fb,
                 kp_fb, kd_fb, desired_area_ratio, area_tol,
                 kp_lr, kd_lr, deadband_x):
        self.w = w
        self.h = h

        self.max_lr = max_lr
        self.max_fb = max_fb

        self.kp_fb = kp_fb
        self.kd_fb = kd_fb
        self.desired_area_ratio = desired_area_ratio
        self.area_tol = area_tol

        self.kp_lr = kp_lr
        self.kd_lr = kd_lr
        self.deadband_x = deadband_x

        self.prev_ex = 0.0
        self.prev_ea = 0.0
        self.prev_ex_lr = 0.0

    def reset(self):
        self.prev_ex = 0.0
        self.prev_ea = 0.0
        self.prev_ex_lr = 0.0

    def compute(self, user_xy, area_ratio, dt):
        cx, cy = self.w // 2, self.h // 2

        ex = float(user_xy[0] - cx)                 
        ea = float(self.desired_area_ratio - area_ratio)  

        dex = (ex - self.prev_ex) / dt
        dea = (ea - self.prev_ea) / dt
        self.prev_ex = ex
        self.prev_ea = ea

        dlr = (ex - self.prev_ex_lr) / dt
        self.prev_ex_lr = ex

        lr_cmd = self.kp_lr * ex + self.kd_lr * dlr
        lr = clamp_int(lr_cmd, -self.max_lr, self.max_lr)
        if abs(ex) < self.deadband_x:
            lr = 0

        if abs(ea) < self.area_tol:
            fb_cmd = 0.0
        else:
            fb_cmd = self.kp_fb * ea + self.kd_fb * dea
        fb = clamp_int(fb_cmd, -self.max_fb, self.max_fb)

        return lr, fb
