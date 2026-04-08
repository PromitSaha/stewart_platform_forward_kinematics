import numpy as np

class inv_kinematics:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.home_pos= np.array([0, 0, 0.5628]) # home position of the platform

        pi = np.pi

        ## Define the Geometry of the platform

        # Coordinate of the points where servo arms 
        # are attached to the corresponding servo axis.
        self.B = np.array([
            [0.0440, -0.1642, -0.1642, 0.0440, 0.1202, 0.1202],
            [0.1642, 0.0440, -0.0440, -0.1642, -0.1202, 0.1202],
            [0, 0, 0, 0, 0, 0]
        ])
            
        # Coordinates of the points where the rods 
        # are attached to the platform.
        self.P = np.array([
            [-0.0391, -0.0878, -0.0878, -0.0391, 0.1269, 0.1269],
            [0.1240, 0.0959, -0.0959, -0.1240, -0.0281, 0.0281],
            [0, 0, 0, 0, 0, 0]
        ])

    # Rotation matrices used later
    def rotX(self, theta):
        rotx = np.array([
            [1,     0    ,    0    ],
            [0,  np.cos(theta), -np.sin(theta)],
            [0,  np.sin(theta), np.cos(theta)] ])
        return rotx

    def rotY(self, theta):    
        roty = np.array([
            [np.cos(theta), 0,  np.sin(theta) ],
            [0         , 1,     0       ],
            [-np.sin(theta), 0,  np.cos(theta) ] ])   
        return roty

    def rotZ(self, theta):    
        rotz = np.array([
            [ np.cos(theta),-np.sin(theta), 0 ],
            [ np.sin(theta), np.cos(theta), 0 ],
            [   0        ,     0      , 1 ] ])   
        return rotz
    
    def solve(self, trans, rotation):
        # Get rotation matrix of platform. RotZ* RotY * RotX -> matmul
        # R = np.matmul( np.matmul(rotZ(rotation[2]), rotY(rotation[1])), rotX(rotation[0]) )
        R = np.matmul( np.matmul(self.rotX(rotation[0]), self.rotY(rotation[1])), self.rotZ(rotation[2]) )

        platform_center = trans + self.home_pos

        angle_offset = np.deg2rad(0)  # or +30 depending on your test
        R_offset = np.array([
            [np.cos(angle_offset), -np.sin(angle_offset), 0],
            [np.sin(angle_offset),  np.cos(angle_offset), 0],
            [0, 0, 1]
        ])
        P_aligned = R_offset @ self.P


        # Get leg length for each leg
        # leg = np.repeat(trans[:, np.newaxis], 6, axis=1) + np.repeat(home_pos[:, np.newaxis], 6, axis=1) + np.matmul(np.transpose(R), P) - B 
        l = np.repeat(platform_center[:, np.newaxis], 6, axis=1) + np.matmul(R, self.P) - self.B

        lll = np.linalg.norm(l, axis=0)

        # Actuator specs
        rest_length = 0.57       # in meters
        stroke_length = 0.202    # in meters

        # Convert to actuator extension: how far from rest position
        extension = lll - rest_length

        # Clamp to actuator range
        #extension = np.clip(extension, 0.0, stroke_length)

        if self.verbose:
            print("I.K 1 Raw leg lengths (m):", lll)
            print("I.K 1 Actuator extensions (m):", extension)

        return extension