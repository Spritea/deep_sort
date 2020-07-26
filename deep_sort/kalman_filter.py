# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        #np.linalg.multi_dot用于多个矩阵连续相乘
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        #以下2步是用cholesky分解，解线性方程组AX=B中的X，X即为A的逆乘B
        #A为projected_cov，B为np.dot(covariance, self._update_mat.T).T
        #这里的kaiman_gain是X还取了个转置scipy.linalg.cho_solve().T
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        #(A*B.T).T=B*A.T,当A*B.T结果为一维数组，在numpy中一维数组取转置不起作用，还是自身
        #所以A*B.T=(A*B.T).T=B*A.T
        #即这里的np.dot(innovation, kalman_gain.T)就是np.dot(kalman_gain,innovation.T)
        #innovation又是一维数组，取转置没影响，所以innovation.T=innovation
        #写在一起，有np.dot(innovation, kalman_gain.T)=np.dot(kalman_gain,innovation.T)
        #=np.dot(kalman_gain,innovation),最后的式子就是kalman滤波器中的标准公式
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        #这个multi_dot指的是K*S*K.T,见chrome收藏的网页
        #zhuanlan.zhihu.com/p/45238681
        #S=H*P'*H.T+R,K=P'*H.T*S.-1,S.-1是S的逆矩阵
        #K*S*K.T=P'*H.T*S.-1*S*K.T=P'*H.T*K.T
        #在标准式子中，这个multi_dot是K*H*P'
        #所以要证明P'*H.T*K.T=K*H*P'
        #即要证明P'*H.T*K.T的结果是个对称阵，并且P'是对称阵
        #对于P'=F*P*F.T+Q,P'确实是对称阵(用例子算出来的)，虽然第一次predict后就不是对角阵了
        #S=H*P'*H.T+R过程中一直是对角阵，K=P'*H.T*S.-1，K的形式也没变过
        #对于P'*H.T*K.T的结果，确实是对称阵(也用例子算出来的，再加上K的形式没变)
        #所以P'*H.T*K.T=K*H*P'成立
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        #cholesky分解，得L*L.T=S，L为下三角阵
        #这里的S就是论文中的S_i
        cholesky_factor = np.linalg.cholesky(covariance)
        #measurements维度(11,4),mean(4,),d(11,4)
        #这里的d就是deep sort论文中的d_j-y_i
        #本来d应该是(4,1)的列向量，只包含一个检测器得到的检测框
        #但这里mearuments包含该帧所有检测框，且为行向量，所以d为(11,4)
        #此时论文中公式要改为(d_j-y_i)*S_i(-1)*(d_j-y_i).T
        #代入d，得d*S_i.(-1)*d.T,即d*S*d.T=d*(L*L.T).(-1)*d.T
        #即=d*(L.T).(-1)*L.(-1)*d.T
        d = measurements - mean
        #解方程cholesky_factor*z=d.T,需cholesky_factor为三角阵
        #即L*z=d.T,z=L.(-1)*d.T
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        #当d只处理1个检测框时，z为(4,1)
        #z.T*z=(L.(-1)*d.T).T*L.(-1)*d.T=d*L.(-1).T*L.(-1)*d.T
        #故z.T*z为论文中的d*S_i.(-1)*d.T，计算上z.T*z也就是各个元素平方然后求和
        #但当d处理多个检测框时，如d为(1,4),z为(4,11)
        #下面这个写成np.sum形式，先平方再对各列求和
        #这样就同时计算一个track的预测值和一帧上所有检测框的马氏距离
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
