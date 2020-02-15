# Contains mathematical helpers for trajectory optimization
import numpy as np
from pydrake.math import cos, sin


class RollPitchYaw:
    """ Represent the roll, pitch, yaw
        Contains useful operations
    """
    def __init__(self, rpy):
        self.roll_pitch_yaw_ = rpy

    def roll_angle(self):
        return self.roll_pitch_yaw_[0]

    def pitch_angle(self):
        return self.roll_pitch_yaw_[1]

    def yaw_angle(self):
        return self.roll_pitch_yaw_[2]

    def CalcAngularVelocityInChildFromRpyDt(self, rpyDt):
        """ Calculates angular velocity from this rpy angles and rpyDt
        :param rpyDt: time derivative of [r, p, y]
        :return: w_AD_D, frame D's angular velocity in frame A, expressed in frame D
        """
        # Get the 3x3 matrix taht contains partial derivatives of rotation angles with respect to r, p, y
        M = self.CalcMatrixRelatingAngularVelocityInChildToRpyDt()
        return M.dot(rpyDt)

    def CalcMatrixRelatingAngularVelocityInChildToRpyDt(self):
        """
        :return: the partial derivative of rotation angles (in own frame) with respect to roll, pitch, yaw
                 M.dot(rpyDt) gives angular velocities in own frame
        """
        r = self.roll_angle()
        p = self.pitch_angle()
        sr = sin(r)
        cr = cos(r)
        sp = sin(p)
        cp = cos(p)
        return np.array([[1, 0, -sp],
                         [0, cr, sr*cp],
                         [0, -sr, cr*cp]])

    def CalcRpyDDtFromRpyDtAndAngularAccelInParent(self, rpyDt, alpha_AD_A):
        """
        :param rpyDt: time derivative of roll, pitch, yaw
        :param alpha_AD_A: angular acceleration expressed in parent frame
        :return: 2nd time derivative of roll, pitch, yaw
        """
        Minv = self.CalcMatrixRelatingRpyDtToAngularVelocityInParent()
        MDt = self.CalcDtMatrixRelatingAngularVelocityInParentToRpyDt(rpyDt)
        return Minv.dot(alpha_AD_A - MDt.dot(rpyDt))

    def CalcMatrixRelatingRpyDtToAngularVelocityInParent(self):
        """
        :return: compute the matrix of partial derivatives relating rpyDt to angular velocities in parent frame
                 M.dot(w) gives rpyDt
        """
        p = self.pitch_angle()
        y = self.yaw_angle()
        sp = sin(p)
        cp = cos(p)
        # Test for Gimbal Lock. Can't use here for trajectory optimization dynamics
        # if self.DoesCosPitchAngleViolateGimbalLockTolerance(cp):
        #     raise Exception("Gimbal Lock")
        one_over_cp = 1.0/cp
        sy = sin(y)
        cy = cos(y)
        cy_over_cp = cy*one_over_cp
        sy_over_cp = sy*one_over_cp
        return np.array([[cy_over_cp, sy_over_cp, 0],
                         [-sy, cy, 0],
                         [cy_over_cp*sp, sy_over_cp*sp, 1]])

    def DoesCosPitchAngleViolateGimbalLockTolerance(self, cos_pitch_angle):
        """ For testing gimbal lock
        :param cos_pitch_angle: cosine(pitch)
        :return: whether too close to gimbal lock
        """
        kGimbalLockToleranceCosPitchAngle = 0.008
        return abs(cos_pitch_angle) < kGimbalLockToleranceCosPitchAngle

    def CalcDtMatrixRelatingAngularVelocityInParentToRpyDt(self, rpyDt):
        """
        :param rpyDt: roll, pitch, yaw, time derivative
        :return: time-derivative of the matrix returned by MatrixRelatingAngularVelocityAToRpyDt()
        """
        p = self.pitch_angle()
        y = self.yaw_angle()
        sp = sin(p)
        cp = cos(p)
        sy = sin(y)
        cy = cos(y)
        pDt = rpyDt[1]
        yDt = rpyDt[2]
        sp_pDt = sp*pDt
        cp_yDt = cp*yDt
        return np.array([[-cy*sp_pDt-sy*cp_yDt, -cy*yDt, 0],
                         [-sy*sp_pDt+cy*cp_yDt, -sy*yDt, 0],
                         [-cp*pDt, 0, 0]])


class RotationMatrix:
    """ Represent the rotation matrix
    """
    def __init__(self, rpy):
        """ Convert from roll, pitch, yaw to rotation matrix
        :param rpy: RollPitchYaw
        """
        r = rpy.roll_angle()
        p = rpy.pitch_angle()
        y = rpy.yaw_angle()

        c0, c1, c2 = cos(r), cos(p), cos(y)
        s0, s1, s2 = sin(r), sin(p), sin(y)

        c2_s1 = c2*s1
        s2_s1 = s2*s1

        Rxx = c2*c1
        Rxy = c2_s1 * s0 - s2 * c0
        Rxz = c2_s1 * c0 + s2 * s0
        Ryx = s2 * c1
        Ryy = s2_s1 * s0 + c2 * c0
        Ryz = s2_s1 * c0 - c2 * s0
        Rzx = -s1
        Rzy = c1 * s0
        Rzz = c1 * c0
        self.R_AB_ = np.array([[Rxx, Rxy, Rxz],
                               [Ryx, Ryy, Ryz],
                               [Rzx, Rzy, Rzz]])

    def matrix(self):
        """
        :return: the rotation matrix as numpy array
        """
        return self.R_AB_
