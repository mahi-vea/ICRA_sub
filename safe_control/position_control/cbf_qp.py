import numpy as np
import cvxpy as cp

class CBFQP:
    def __init__(self, robot, robot_spec, num_obs=1):
        self.robot = robot
        self.robot_spec = robot_spec
        self.num_obs = num_obs

        self.cbf_param = {}

        if self.robot_spec['model'] == "SingleIntegrator2D":
            self.cbf_param['alpha'] = 1.0
        elif self.robot_spec['model'] == 'Unicycle2D':
            self.cbf_param['alpha'] = 1.0
        elif self.robot_spec['model'] == 'DynamicUnicycle2D':
            self.cbf_param['alpha1'] = 10.0
            self.cbf_param['alpha2'] = 10.0
        elif self.robot_spec['model'] == 'DoubleIntegrator2D':
            self.cbf_param['alpha1'] = 1.5
            self.cbf_param['alpha2'] = 1.5
        elif self.robot_spec['model'] == 'KinematicBicycle2D':
            self.cbf_param['alpha1'] = 1.5
            self.cbf_param['alpha2'] = 1.5
        elif self.robot_spec['model'] == 'KinematicBicycle2D_C3BF':
            self.cbf_param['alpha'] = 1.5
        elif self.robot_spec['model'] == 'KinematicBicycle2D_DPCBF':
            self.cbf_param['alpha'] = 1.5
        elif self.robot_spec['model'] == 'DynamicUnicycle2D_DPCBF':
            self.cbf_param['alpha'] = 10
        elif self.robot_spec['model'] == 'Quad2D':
            self.cbf_param['alpha1'] = 1.5
            self.cbf_param['alpha2'] = 1.5
        elif self.robot_spec['model'] == 'Quad3D':
            self.cbf_param['alpha'] = 1.5
        elif self.robot_spec['model'] == 'Manipulator2D':
            self.cbf_param['alpha'] = 1.0

        # Optional per-scenario CBF gain overrides.
        if 'cbf_alpha' in self.robot_spec:
            self.cbf_param['alpha'] = float(self.robot_spec['cbf_alpha'])
        if 'cbf_alpha1' in self.robot_spec:
            self.cbf_param['alpha1'] = float(self.robot_spec['cbf_alpha1'])
        if 'cbf_alpha2' in self.robot_spec:
            self.cbf_param['alpha2'] = float(self.robot_spec['cbf_alpha2'])

        self.setup_control_problem()

    def setup_control_problem(self):
        self.u = cp.Variable((2, 1))
        self.u_ref = cp.Parameter((2, 1), value=np.zeros((2, 1)))
        self.A1 = cp.Parameter((self.num_obs, 2), value=np.zeros((self.num_obs, 2)))
        self.b1 = cp.Parameter((self.num_obs, 1), value=np.zeros((self.num_obs, 1)))
        objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))

        if self.robot_spec['model'] == 'SingleIntegrator2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <=  self.robot_spec['v_max'],
                           cp.abs(self.u[1]) <=  self.robot_spec['v_max']]
        elif self.robot_spec['model'] == 'Unicycle2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['v_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['w_max']]
        elif self.robot_spec['model'] == 'DynamicUnicycle2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['w_max']]
        elif self.robot_spec['model'] == 'DoubleIntegrator2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['a_max']]
        elif 'KinematicBicycle2D' in self.robot_spec['model']:
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['beta_max']]
        elif self.robot_spec['model'] == 'DynamicUnicycle2D_DPCBF':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u[0]) <= self.robot_spec['a_max'],
                           cp.abs(self.u[1]) <= self.robot_spec['w_max']]
        elif self.robot_spec['model'] == 'Quad2D':
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           self.robot_spec["f_min"] <= self.u[0],
                           self.u[0] <= self.robot_spec["f_max"],
                           self.robot_spec["f_min"] <= self.u[1],
                           self.u[1] <= self.robot_spec["f_max"]]
        elif self.robot_spec['model'] == 'Quad3D':
            # overwrite the variables
            self.u = cp.Variable((4, 1))
            self.u_ref = cp.Parameter((4, 1), value=np.zeros((4, 1)))
            self.A1 = cp.Parameter((1, 4), value=np.zeros((1, 4)))
            self.b1 = cp.Parameter((1, 1), value=np.zeros((1, 1)))
            objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           self.u[0] <= self.robot_spec['u_max'],
                           self.u[0] >= 0.0,
                           cp.abs(self.u[1]) <= self.robot_spec['u_max'],
                           cp.abs(self.u[2]) <= self.robot_spec['u_max'],
                           cp.abs(self.u[3]) <= self.robot_spec['u_max']]
        elif self.robot_spec['model'] == 'Manipulator2D':
            # Manipulator 3DOF
            self.u = cp.Variable((3, 1))
            self.u_ref = cp.Parameter((3, 1), value=np.zeros((3, 1)))
            self.A1 = cp.Parameter((self.num_obs, 3), value=np.zeros((self.num_obs, 3)))
            self.b1 = cp.Parameter((self.num_obs, 1), value=np.zeros((self.num_obs, 1)))
            objective = cp.Minimize(cp.sum_squares(self.u - self.u_ref))
            constraints = [self.A1 @ self.u + self.b1 >= 0,
                           cp.abs(self.u) <= self.robot_spec['w_max']]

        self.cbf_controller = cp.Problem(objective, constraints)

    def solve_control_problem(self, robot_state, control_ref, obs_list):
        # Reset ALL parameters first — CVXPY requires every Parameter to have
        # a value before solve(), regardless of what happens in the loop below.
        self.A1.value[:] = 0
        self.b1.value[:] = 0
        # self.u_ref.value = control_ref['u_ref']\
        self.u_ref.value = control_ref['u_ref']
        # print(f"[DEBUG] control_ref['u_ref'] = {control_ref['u_ref']}, type = {type(control_ref['u_ref'])}")
        # print(f"[DEBUG] self.u_ref.value after assign = {self.u_ref.value}")

        if obs_list is None:
            self.status = 'optimal'
            return self.u_ref.value

        mode = self.robot_spec.get('cbf_mode', 'cbf')
        row_idx = 0
        for i, obs in enumerate(obs_list):
            if obs is None:
                continue

            # Stop if we exceed allocated constraints
            if row_idx >= self.num_obs:
                break

            # Handle Manipulator2D (Multiple constraints per obstacle)
            if self.robot_spec['model'] == 'Manipulator2D':
                h_list, dh_dx_list = self.robot.agent_barrier(obs)
                for h, dh_dx in zip(h_list, dh_dx_list):
                    if row_idx >= self.num_obs:
                        break

                    if mode == 'hard':
                        dt = self.robot.dt
                        self.A1.value[row_idx, :] = dh_dx @ self.robot.g()
                        self.b1.value[row_idx, :] = h / dt + (dh_dx @ self.robot.f())
                    else:
                        self.A1.value[row_idx, :] = dh_dx
                        self.b1.value[row_idx, :] = self.cbf_param['alpha'] * h

                    row_idx += 1

            # Handle Standard Robots (Single constraint per obstacle)
            else:
                dt = self.robot.dt
                if self.robot_spec['model'] in ['SingleIntegrator2D', 'Unicycle2D',
                                                 'KinematicBicycle2D_C3BF',
                                                 'KinematicBicycle2D_DPCBF',
                                                 'DynamicUnicycle2D_DPCBF',
                                                 'Quad3D']:
                    h, dh_dx = self.robot.agent_barrier(obs)

                    if mode == 'hard':
                        self.A1.value[row_idx, :] = dh_dx @ self.robot.g()
                        self.b1.value[row_idx, :] = h / dt + dh_dx @ self.robot.f()
                    else:
                        # CBF: A*u + b >= 0  where b = Lf_h + alpha*h
                        self.A1.value[row_idx, :] = dh_dx @ self.robot.g()
                        self.b1.value[row_idx, :] = dh_dx @ self.robot.f() + self.cbf_param['alpha'] * h

                elif self.robot_spec['model'] in ['DynamicUnicycle2D', 'DoubleIntegrator2D',
                                                   'KinematicBicycle2D', 'Quad2D']:
                    h, h_dot, dh_dot_dx = self.robot.agent_barrier(obs)

                    if mode == 'hard':
                        dt = self.robot.dt
                        self.A1.value[row_idx, :] = dh_dot_dx @ self.robot.g()
                        self.b1.value[row_idx, :] = h / (dt**2) + 2 * h_dot / dt + dh_dot_dx @ self.robot.f()
                    else:
                        # CBF for Relative Degree 2
                        gamma1 = self.cbf_param['alpha1'] + self.cbf_param['alpha2']
                        gamma2 = self.cbf_param['alpha1'] * self.cbf_param['alpha2']
                        self.A1.value[row_idx, :] = dh_dot_dx @ self.robot.g()
                        self.b1.value[row_idx, :] = dh_dot_dx @ self.robot.f() + gamma1 * h_dot + gamma2 * h

                row_idx += 1

        # Solve — all Parameters are guaranteed to have values from above
        # for param in self.cbf_controller.parameters():
        #     if param.value is None:
        #         print(f"[DEBUG] Unset parameter: name={param.name()}, shape={param.shape}, id={id(param)}")
        #     else:
        #         print(f"[DEBUG] OK parameter: name={param.name()}, shape={param.shape}, value=\n{param.value}")
        # print(f"[DEBUG] id(self.u_ref)={id(self.u_ref)}, unset param id={[id(p) for p in self.cbf_controller.parameters() if p.value is None]}")        
        self.cbf_controller.solve(solver=cp.GUROBI, reoptimize=True)

        self.status = self.cbf_controller.status

        return self.u.value