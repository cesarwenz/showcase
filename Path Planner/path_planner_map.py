
import numpy as np
import random
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as Rot

class HoughBundler:
    '''Clasterize and merge each cluster of cv2.HoughLinesP() output
    a = HoughBundler()
    foo = a.process_lines(houghP_lines, binary_image)
    '''

    def get_orientation(self, line):
        '''get orientation of a line, using its length
        https://en.wikipedia.org/wiki/Atan2
        '''
        orientation = np.arctan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
        return np.degrees(orientation)

    def checker(self, line_new, groups, min_distance_to_merge, min_angle_to_merge):
        '''Check if line have enough distance and angle to be count as similar
        '''
        for group in groups:
            # walk through existing line groups
            for line_old in group:
                # check distance
                if self.get_distance(line_old, line_new) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_new = self.get_orientation(line_new)
                    orientation_old = self.get_orientation(line_old)
                    # if all is ok -- line is similar to others in group
                    if abs(orientation_new - orientation_old) < min_angle_to_merge:
                        group.append(line_new)
                        return False
        # if it is totally different line
        return True

    def DistancePointLine(self, point, line):
        """Get distance between point and line
        http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        """
        px, py = point
        x1, y1, x2, y2 = line

        def lineMagnitude(x1, y1, x2, y2):
            'Get line (aka vector) length'
            lineMagnitude = np.sqrt(np.power((x2 - x1), 2) + np.power((y2 - y1), 2))
            return lineMagnitude

        LineMag = lineMagnitude(x1, y1, x2, y2)
        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = lineMagnitude(px, py, x1, y1)
            iy = lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = lineMagnitude(px, py, ix, iy)

        return DistancePointLine

    def get_distance(self, a_line, b_line):
        """Get all possible distances between each dot of two lines and second line
        return the shortest
        """
        dist1 = self.DistancePointLine(a_line[:2], b_line)
        dist2 = self.DistancePointLine(a_line[2:], b_line)
        dist3 = self.DistancePointLine(b_line[:2], a_line)
        dist4 = self.DistancePointLine(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_pipeline_2(self, lines):
        'Clusterize (group) lines'
        groups = []  # all lines groups are here
        # Parameters to play with
        min_distance_to_merge = 20
        min_angle_to_merge = 30
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
                groups.append([line_new])

        return groups

    def merge_lines_segments1(self, lines):
        """Sort lines cluster and return first and last coordinates
        """
        orientation = self.get_orientation(lines[0])

        # special case
        if(len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical
        # if 45 < orientation < 135:
        #     #sort by y
        #     points = sorted(points, key=lambda point: point[1])
        # else:
            #sort by x
        points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    def process_lines(self, lines, img):
        '''Main function for lines from cv.HoughLinesP() output merging
        for OpenCV 3
        lines -- cv.HoughLinesP() output
        img -- binary image
        '''
        lines_x = []
        lines_y = []
        # for every line of cv2.HoughLinesP()
        for line_i in [l[0] for l in lines]:
                orientation = self.get_orientation(line_i)
                # if vertical
                if 45 < orientation < 135:
                    lines_y.append(line_i)
                else:
                    lines_x.append(line_i)

        lines_y = sorted(lines_y, key=lambda line: line[1])
        lines_x = sorted(lines_x, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_x, lines_y]:
                if len(i) > 0:
                    groups = self.merge_lines_pipeline_2(i)
                    merged_lines = []
                    for group in groups:
                        merged_lines.append(self.merge_lines_segments1(group))

                    merged_lines_all.extend(merged_lines)

        return merged_lines_all


class RowPlanner(object):
    def __init__(self, start_pos, lines, lin_vel, meter_per_pix, circ_radius, c_r, w_r, dt, x_offset, y_offset, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        """
        Starts the pathplanner class.
        """
        # Define planner parameters
        self.start_pos = start_pos
        self.lines = lines
        self.radius = circ_radius
        self.meter_per_pix = meter_per_pix
        self.pix_vel = lin_vel/meter_per_pix
        self.lin_vel = lin_vel
        self.c_r = c_r
        self.w_r = w_r
        self.dt = dt
        self.x_offset = x_offset # x-offset distance from rows
        self.y_offset = y_offset # y-offset distance from rows

        # Initialize nodes and cost matrix
        self.create_nodes()
        self.starting_node_dist()
        self.calculate_cost_matrix()
        
        # Set up simulated annealing parameters

        self.N = len(self.coords)
        self.T = np.sqrt(self.N) if T == -1 else T
        self.T_save = self.T  # save inital T to reset if batch annealing is used
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 1e-8 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1
        self.start = 0 # maybe replaced by a better process?
        ##
        ##


        self.nodes = [i for i in range(self.N)]

        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []


    def initial_solution(self):
        """
        Greedy algorithm to get an initial solution (closest-neighbour).
        """
        # Start with the shortest distance node from current position


        if self.start == 0:
            if self.start_dir == 0:
                cur_node = np.argmin(self.even_dist) * 2 
                self.start = 1
            else:
                cur_node = np.argmin(self.odd_dist)*2 + 1
                self.start = 1
        # Pick random starting node
        else:
            if self.start_dir == 0:
                cur_node = random.choice(range(0,self.N,2))
            else:
                cur_node = random.choice(range(1,self.N,2))
        solution = [cur_node]

        free_nodes = set(self.nodes)
        free_nodes.remove(cur_node)
        while free_nodes:
            temp_cost = np.copy(self.cost_matrix[cur_node])
            temp_cost[solution] =  float('Inf')
            next_node = np.argmin(temp_cost)
            free_nodes.remove(next_node)
            solution.append(next_node)
            cur_node = next_node
        
        cur_fit = self.fitness(solution)
        if cur_fit < self.best_fitness:  # If best found so far, update best fitness
            self.best_fitness = cur_fit
            self.best_solution = solution
        self.fitness_list.append(cur_fit)
        return solution, cur_fit

    def starting_node_dist(self):
        """
        Defines starting direction of the path planner and the distance
        from current location to the next node.
        """
        # Even nodes (start at left)
        left = np.copy(self.coords[np.arange(len(self.coords))%2==0])[:,2:4]
        # Odd nodes (start at right)
        right = np.copy(self.coords[np.arange(len(self.coords))%2!=0])[:,0:2]
        
        self.even_dist = [self.euc_dist(self.start_pos, left[x]) for x in range(len(left))]
        self.odd_dist = [self.euc_dist(self.start_pos, right[x]) for x in range(len(right))]
        
        if min(self.odd_dist) > min(self.even_dist):
            # Starts to the left (even)
            self.start_dir = 0
        else:
            # Starts to the right (odd)
            self.start_dir = 1

    def create_nodes(self):
        """
        Creates nodes using the tree row line points. The tree row line points
        are transformed to a local coordinate and offset by a x and y distance.
        """
        # Define rotation vector
        p = sum(self.lines[:,:])/len(self.lines[:,:])
        dx = p[0]-p[2]
        dy = p[1]-p[3]
        self.th = np.arctan2(dy, dx)
        # Rotation Matrix
        self.rot_local = Rot.from_euler('z', -self.th).as_dcm()[0:2, 0:2]
        self.rot_global = Rot.from_euler('z', self.th).as_dcm()[0:2, 0:2]
        
        self.coords = np.empty((0,4))

        for points in self.lines:
            # Convert to local coordinate space
            converted_left = np.dot(self.rot_local, points.reshape((2,2)).T)
            converted_right = np.copy(converted_left)

            # Add offset to create nodes
            converted_left[0] = np.array([converted_left[0,0]+self.x_offset, converted_left[0,1]-self.x_offset])
            converted_right[0] = np.array([converted_right[0,0]+self.x_offset, converted_right[0,1]-self.x_offset])
            
            converted_left[1] = np.array([converted_left[1,0]-self.y_offset, converted_left[1,1]-self.y_offset], dtype=int)
            converted_right[1] = np.array([converted_right[1,0]+self.y_offset, converted_right[1,1]+self.y_offset], dtype=int)
            
            # Convert back to global coordinate frame
            converted_global_left =  np.dot(self.rot_global, converted_left).astype(int)
            converted_global_right =  np.dot(self.rot_global, converted_right).astype(int)

            # Return newly created nodes
            self.coords = np.vstack((self.coords, converted_global_right.T.reshape(1,4)))
            self.coords = np.vstack((self.coords, converted_global_left.T.reshape(1,4)))

    def calculate_cost_matrix(self):
        """
        Calculates the track distance and turning distance, and embeds it to a 
        cost matrix. Turns only go from even to odd or vise versa.
        """
        # Distances
        line_dist = []
        # Calculate track distance of each node pair
        for x1, y1, x2, y2 in self.coords:
            dist = np.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)
            line_dist = np.append(line_dist, dist)

        # Horizontal distance cost matrix
        A = np.ones((len(self.coords), len(self.coords)))*float("Inf")
        # Vertical distance cost matrix 
        B = np.ones((len(self.coords), len(self.coords)))*float("Inf")

        for i in range(len(self.coords)):
            for j in range(len(self.coords)):
                # If node i to j, then skip
                if i==j:
                    continue
                # If starting from left
                # Travelling Straight cost
                elif (i % 2) != 0 and (j % 2) == 0:
                    A[i,j] = line_dist[i] + line_dist[j]
                    B[i,j] = self.euc_dist(self.coords[i, 2:4], self.coords[j, 2:4])
                # If even to odd
                elif (i % 2) == 0 and (j % 2) != 0:
                    A[i,j] = line_dist[i] + line_dist[j]
                    B[i,j] = self.euc_dist(self.coords[i, 0:2], self.coords[j, 0:2])
                    
                # Turning costs
                # If vertical distance is lower than U shaped turn, add T turn cost
                if B[i,j] < (2 * self.radius):
                    B[i,j] = (np.pi * self.radius) + (2 * self.radius) - B[i,j]
                # Else Add U turn cost
                else: 
                    B[i,j] = B[i,j] + (np.pi - 2) * self.radius
        self.cost_matrix = A + B

    def rearrange_nodes(self):
        """
        Rearranges the final solution of the SA algorithm, and matches
        vertical distance from current node to the next node.
        """
        # Rearrange nodes to the final solution order
        self.final_nodes = self.coords[self.best_solution].astype(int)
        # Define temp variable to hold initial node
        temp = np.array([self.final_nodes[0]])

        for idx, node_points in enumerate(self.final_nodes[1:]):
            # Convert current node (node_points) and previous node (temp) to local coord
            p = np.dot(self.rot_local, np.reshape(temp, (2,2)).T) # previous node
            m = np.dot(self.rot_local, np.reshape(node_points, (2,2)).T) # current node
            # If current node is odd
            if self.best_solution[idx + 1] % 2 != 0:
                # x1 of previous = x1 of current node
                p[0,0] = m[0,0]
                # Redefine final solution nodes
                self.final_nodes[idx] = np.dot(self.rot_global, p).T.reshape((1,4))

            # If current node is even
            else:
                # x2 of previous = x2 of current node
                p[0,1] = m[0,1]
                # Redefine final solution nodes
                self.final_nodes[idx] = np.dot(self.rot_global, p).T.reshape((1,4))

            temp = node_points

    def euc_dist(self, node_0, node_1):
        """
        Euclidean distance between two nodes.
        """
        return np.sqrt((node_0[0] - node_1[0]) ** 2 + (node_0[1] - node_1[1]) ** 2)

    def fitness(self, solution):
        """
        Total distance of the current solution path.
        """
        if self.start_dir == 0:
            cur_fit = self.even_dist[int(solution[0]/2)]
        else:
            cur_fit = self.odd_dist[int(solution[0]/2)]
        
        for i in range(self.N-1):
            cur_fit += self.cost_matrix[solution[i], solution[(i + 1)]] 
        return cur_fit

    def p_accept(self, candidate_fitness):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        return np.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)

    def accept(self, candidate):
        """
        Accept with probability 1 if candidate is better than current.
        Accept with probabilty p_accept(..) if candidate is worse.
        """
        candidate_fitness = self.fitness(candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness, self.cur_solution = candidate_fitness, candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_fitness, candidate
        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate

    def anneal(self):
        """
        Execute simulated annealing algorithm.
        """
        # Initialize with the greedy solution.
        self.cur_solution, self.cur_fitness = self.initial_solution()

        print("Starting annealing.")
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = np.array(self.cur_solution)
            l = random.randint(0, (self.N/2) - 1)
            i = random.randint(0, (self.N/2) - l)
            candidate[candidate % 2 != 0][i:(i+l)] = list(reversed(candidate[candidate % 2 != 0][i:(i+l)]))
            candidate[candidate % 2 == 0][i:(i+l)] = list(reversed(candidate[candidate % 2 == 0][i:(i+l)]))

            self.accept(candidate)
            self.T *= self.alpha
            self.iteration += 1
            self.fitness_list.append(self.cur_fitness)

        print("Best fitness obtained: ", self.best_fitness)
        improvement = 100 * (self.fitness_list[0] - self.best_fitness) / (self.fitness_list[0])
        print ("Improvement over greedy heuristic: "), improvement

    def batch_anneal(self, times=15):
        """
        Execute simulated annealing algorithm `times` times, with random initial solutions.
        """
        for i in range(1, times + 1):
            print ('Batch: ', i,'/',times, '-------------------------------')
            self.T = self.T_save
            self.iteration = 1
            self.cur_solution, self.cur_fitness = self.initial_solution()
            self.anneal()

    # def plot_arrow(self, )
    def trajectory(self, x_offset=0, y_offset=0, num_iters=1):
        """
        !!!
        """
        fig = plt.figure(figsize=(12.5, 15), dpi=72)
        fig.add_axes([0, 0, 1, 1])
        
        num_points = lambda nodes: np.ceil((np.sqrt((nodes[0]-nodes[2])**2 + (nodes[1]-nodes[3])**2)/self.pix_vel)/self.dt).astype(int)
        t = np.array([])
        waypoints_x = []
        waypoints_y = []    
        t_f = (np.pi/2)/self.w_r
        t_k = np.arange(0, t_f, self.dt)
        
        self.rearrange_nodes()
        final_nodes = np.copy(self.final_nodes)
        # # Define the starting position depending on the side
        # if self.start_dir == 0:
        #     waypoints_x = self.start_pos[0]
        #     waypoints_y = self.start_pos[1]
        # else:
        #     waypoints_x = self.start_pos[0]
        #     waypoints_y = self.start_pos[1]
            
        temp = final_nodes[0]
        vel_x = []
        vel_y = []
        theta = []
        dtheta = []
        # Define the direction for the first node
        if self.best_solution[0] % 2 != 0:
            # Plot right to left arrow from current solution
            waypoints_x = np.append(waypoints_x, np.linspace(temp[0], temp[2], num_points(temp)))
            waypoints_y = np.append(waypoints_y, np.linspace(temp[1], temp[3], num_points(temp)))
            theta = np.append(theta, (self.th-np.pi)*np.ones(num_points(temp)))
            dx = self.pix_vel*np.cos(self.th-np.pi)*np.ones(num_points(temp))
            dy = self.pix_vel*np.sin(self.th-np.pi)*np.ones(num_points(temp))
            
        else:
            waypoints_x = np.append(waypoints_x, np.linspace(temp[2], temp[0], num_points(temp)))
            waypoints_y = np.append(waypoints_y, np.linspace(temp[3], temp[1], num_points(temp)))  
            theta = np.append(theta, self.th*np.ones(num_points(temp)))
            dx = self.pix_vel*np.cos(self.th)*np.ones(num_points(temp))
            dy = self.pix_vel*np.sin(self.th)*np.ones(num_points(temp))
            
        vel_x = np.append(vel_x, dx)
        vel_y = np.append(vel_y, dy)
        dtheta = np.append(dtheta, np.zeros(num_points(temp)))

        # Create trajectories of turns and straight lines
        for idx, node_points in enumerate(final_nodes[1:]):
            # Define 
            p = np.dot(self.rot_local, np.reshape(temp, (2,2)).T)
            m = np.dot(self.rot_local, np.reshape(node_points, (2,2)).T)
            if self.best_solution[idx+1] % 2 != 0:
                # Calculate local coordinate of previous and current node
                # Defined the x coordinates of arc 1 and arc 2 turn
                # x1r = np.linspace(p[0,0], p[0,0]+self.radius, turn_res)
                # x2r = np.linspace(m[0,0], m[0,0]+self.radius, turn_res)
                
                # if previous node to next node y axis is greater, then arc 1 goes up and 
                # arc 2 goes down
                if p[1,0] > m[1,0]:
                    # y1r = np.sqrt(abs(self.radius**2-np.power((x1r-p[0,0]),2)))+p[1,0]-self.radius
                    # y2r = -np.sqrt(abs(self.radius**2-np.power((x2r-m[0,0]),2)))+m[1,0]+self.radius
                    x1r = self.c_r*np.sin(self.w_r*t_k) + p[0,0]                    
                    y1r = self.c_r*np.cos(self.w_r*t_k) + p[1,0] - self.radius
                    x2r = self.c_r*np.sin(-self.w_r*t_k + np.pi) + m[0,0]
                    y2r = self.c_r*np.cos(-self.w_r*t_k + np.pi) + m[1,0] + self.radius
                    theta_r = -self.w_r*t_k + theta[-1]
                    theta_2r = theta_r[-1] - self.w_r*t_k

                    dx1r = self.c_r*self.w_r*np.cos(self.w_r*t_k)             
                    dy1r = -self.c_r*self.w_r*np.sin(self.w_r*t_k) 
                    dx2r = -self.c_r*self.w_r*np.cos(-self.w_r*t_k + np.pi)
                    dy2r = self.c_r*self.w_r*np.sin(-self.w_r*t_k + np.pi)
                    dtheta1r = -self.w_r*np.ones(np.size(x1r))
                    dtheta2r = -self.w_r*np.ones(np.size(x1r))
                    

                # Vice versa
                else:
                    # y1r = -np.sqrt(abs(self.radius**2-np.power((x1r-p[0,0]),2)))+p[1,0]+self.radius
                    # y2r = np.sqrt(abs(self.radius**2-np.power((x2r-m[0,0]),2)))+m[1,0]-self.radius
                    x1r = self.c_r*np.sin(-self.w_r*t_k + np.pi) + p[0,0]                    
                    y1r = self.c_r*np.cos(-self.w_r*t_k + np.pi) + p[1,0] + self.radius
                    x2r = self.c_r*np.sin(self.w_r*t_k) + m[0,0]
                    y2r = self.c_r*np.cos(self.w_r*t_k) + m[1,0] - self.radius
                    theta_r = self.w_r*t_k + theta[-1]
                    theta_2r = theta_r[-1] + self.w_r*t_k
                
                    dx1r = -self.c_r*self.w_r*np.cos(-self.w_r*t_k + np.pi)                   
                    dy1r = self.c_r*self.w_r*np.sin(-self.w_r*t_k + np.pi) 
                    dx2r = self.c_r*self.w_r*np.cos(self.w_r*t_k)
                    dy2r = -self.c_r*self.w_r*np.sin(self.w_r*t_k)
                    dtheta1r = self.w_r*np.ones(np.size(x1r))
                    dtheta2r = self.w_r*np.ones(np.size(x1r))
                
                yt = np.linspace(y1r[-1], y2r[-1], num_points([x1r[-1], y1r[-1], x2r[-1], y2r[-1]]))
                xt = np.linspace(x1r[-1], x2r[-1], num_points([x1r[-1], y1r[-1], x2r[-1], y2r[-1]]))
                theta_t = theta_r[-1]*np.ones(np.size(xt))

                dxt = self.pix_vel*np.cos(theta_r[-1])*np.ones(num_points([x1r[-1], y1r[-1], x2r[-1], y2r[-1]]))
                dyt = self.pix_vel*np.sin(theta_r[-1])*np.ones(num_points([x1r[-1], y1r[-1], x2r[-1], y2r[-1]]))
                dtheta_t = np.zeros(num_points([x1r[-1], y1r[-1], x2r[-1], y2r[-1]]))

                x = np.append(np.append(x1r, xt), np.flip(x2r))
                y = np.append(np.append(y1r, yt), np.flip(y2r))
                theta_temp = np.append(np.append(theta_r, theta_t), theta_2r)
                
                dx = np.append(np.append(dx1r, dxt), dx2r)
                dy = np.append(np.append(dy1r, dyt), dy2r)
                dtheta_temp = np.append(np.append(dtheta1r, dtheta_t), dtheta2r)

                t = np.dot(self.rot_global, np.append(x,y).reshape((2,len(x)))).astype(int)

                waypoints_x = np.append(waypoints_x, t[0])
                waypoints_y = np.append(waypoints_y, t[1])
                theta = np.append(theta, theta_temp)
                
                vel_x = np.append(vel_x, dx)
                vel_y = np.append(vel_y, dy)
                dtheta = np.append(dtheta, dtheta_temp)

                waypoints_x = np.append(waypoints_x, np.linspace(node_points[0], node_points[2], num_points(node_points)))
                waypoints_y = np.append(waypoints_y, np.linspace(node_points[1], node_points[3], num_points(node_points)))
                theta = np.append(theta, theta[-1]*np.ones(num_points(node_points)))
                # Straight velocity traj
                vel_x = np.append(vel_x, self.pix_vel*np.cos(theta[-1])*np.ones(num_points(node_points)))
                vel_y = np.append(vel_y, self.pix_vel*np.sin(theta[-1])*np.ones(num_points(node_points)))
                dtheta = np.append(dtheta, np.zeros(num_points(node_points)))
            else:
                # x1r = np.linspace(p[0,1], p[0,1]-self.radius, turn_res)
                # x2r = np.linspace(m[0,1], m[0,1]-self.radius, turn_res)
                
                if p[1,1] > m[1,1]:
                    x1r = self.c_r*np.sin(-self.w_r*t_k) + p[0,1]
                    y1r = self.c_r*np.cos(-self.w_r*t_k) + p[1,1] - self.radius 
                    x2r = self.c_r*np.sin(self.w_r*t_k + np.pi) + m[0,1]
                    y2r = self.c_r*np.cos(self.w_r*t_k + np.pi) + m[1,1] + self.radius 
                    theta_r = self.w_r*t_k + theta[-1]
                    theta_2r = theta_r[-1] + self.w_r*t_k

                    dx1r = -self.c_r*self.w_r*np.cos(-self.w_r*t_k)                    
                    dy1r = self.c_r*self.w_r*np.sin(-self.w_r*t_k)
                    dx2r = self.c_r*self.w_r*np.cos(self.w_r*t_k + np.pi)
                    dy2r = -self.c_r*self.w_r*np.sin(self.w_r*t_k + np.pi)
                    dtheta1r = self.w_r*np.ones(np.size(x1r))
                    dtheta2r = self.w_r*np.ones(np.size(x1r))
                    

                    # y1r = np.sqrt(abs(self.radius**2-np.power((x1r-p[0,1]),2)))+p[1,1]-self.radius
                    # y2r = -np.sqrt(abs(self.radius**2-np.power((x2r-m[0,1]),2)))+m[1,1]+self.radius
                else:
                    # y1r = -np.sqrt(abs(self.radius**2-np.power((x1r-p[0,1]),2)))+p[1,1]+self.radius
                    # y2r = np.sqrt(abs(self.radius**2-np.power((x2r-m[0,1]),2)))+m[1,1]-self.radius
                    x1r = self.c_r*np.sin(self.w_r*t_k + np.pi) + p[0,1]                    
                    y1r = self.c_r*np.cos(self.w_r*t_k + np.pi) + p[1,1] + self.radius 
                    x2r = self.c_r*np.sin(-self.w_r*t_k) + m[0,1]
                    y2r = self.c_r*np.cos(-self.w_r*t_k) + m[1,1] - self.radius 
                    theta_r = -self.w_r*t_k + theta[-1]
                    theta_2r = theta_r[-1] - self.w_r*t_k

                    dx1r = self.c_r*self.w_r*np.cos(self.w_r*t_k + np.pi)                    
                    dy1r = -self.c_r*self.w_r*np.sin(self.w_r*t_k + np.pi)
                    dx2r = -self.c_r*self.w_r*np.cos(-self.w_r*t_k)
                    dy2r = self.c_r*self.w_r*np.sin(-self.w_r*t_k)
                    dtheta1r = -self.w_r*np.ones(np.size(x1r))
                    dtheta2r = -self.w_r*np.ones(np.size(x1r))
                    
                    

                yt = np.linspace(y1r[-1], y2r[-1], num_points(np.array([x1r[-1], y1r[-1], x2r[-1], y2r[-1]])))
                xt = np.linspace(x1r[-1], x2r[-1], num_points(np.array([x1r[-1], y1r[-1], x2r[-1], y2r[-1]])))
                theta_t = theta_r[-1]*np.ones(np.size(xt))

                dxt = self.pix_vel*np.cos(theta_r[-1])*np.ones(num_points([x1r[-1], y1r[-1], x2r[-1], y2r[-1]]))
                dyt = self.pix_vel*np.sin(theta_r[-1])*np.ones(num_points([x1r[-1], y1r[-1], x2r[-1], y2r[-1]]))
                dtheta_t = np.zeros(num_points([x1r[-1], y1r[-1], x2r[-1], y2r[-1]]))

                x = np.append(np.append(x1r, xt), np.flip(x2r))
                y = np.append(np.append(y1r, yt), np.flip(y2r))
                theta_temp = np.append(np.append(theta_r, theta_t), theta_2r)

                dx = np.append(np.append(dx1r, dxt), dx2r)
                dy = np.append(np.append(dy1r, dyt), dy2r)
                dtheta_temp = np.append(np.append(dtheta1r, dtheta_t), dtheta2r)

                t = np.dot(self.rot_global, np.append(x,y).reshape((2,len(x)))).astype(int)

                waypoints_x = np.append(waypoints_x, t[0])
                waypoints_y = np.append(waypoints_y, t[1])
                theta = np.append(theta, theta_temp)

                vel_x = np.append(vel_x, dx)
                vel_y = np.append(vel_y, dy)
                dtheta = np.append(dtheta, dtheta_temp)

                waypoints_x = np.append(waypoints_x, np.linspace(node_points[2], node_points[0], num_points(node_points)))
                waypoints_y = np.append(waypoints_y, np.linspace(node_points[3], node_points[1], num_points(node_points))) 
                theta = np.append(theta, theta[-1]*np.ones(num_points(node_points)))
                # Straight velocity traj
                vel_x = np.append(vel_x, self.pix_vel*np.cos(theta[-1])*np.ones(num_points(node_points)))
                vel_y = np.append(vel_y,  self.pix_vel*np.sin(theta[-1])*np.ones(num_points(node_points)))
                dtheta = np.append(dtheta, np.zeros(num_points(node_points)))
            temp = node_points
            
        # Plot ending node
        if self.start_dir == 0:
            # plt.plot(final_nodes[-1][2], final_nodes[-1][3], 'bx', ms=5, mew=2)
            waypoints_x = np.append(waypoints_x, final_nodes[-1][2])
            waypoints_y = np.append(waypoints_y, final_nodes[-1][3])
        else:
            # plt.plot(final_nodes[-1][0], final_nodes[-1][1], 'bx', ms=5, mew=2)
            waypoints_x = np.append(waypoints_x, final_nodes[-1][0])
            waypoints_y = np.append(waypoints_y, final_nodes[-1][1])

        waypoints_x = waypoints_x*self.meter_per_pix + x_offset
        waypoints_y = waypoints_y*self.meter_per_pix + y_offset
        vel_x = vel_x*self.meter_per_pix
        vel_y = vel_y*self.meter_per_pix

        lines_global = self.lines*self.meter_per_pix
        lines_global[:,0::2] = lines_global[:,0::2] + x_offset
        lines_global[:,1::2] = lines_global[:,1::2] + y_offset
        # Plot tree rows
        for x1, y1, x2, y2 in lines_global:
            plt.plot((x1, x2), (y1, y2), c = '#4CAF50', label='Tree rows')
        # plt.axis('off')
        plt.scatter(waypoints_x[1:], waypoints_y[1:], marker='.')
        return waypoints_x, waypoints_y, theta, vel_x, vel_y, dtheta

    
    def plot_learning(self):
        """
        Plot the fitness through iterations.
        """
        plt.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.show()
