%====================================================================
% Author: Cesar Wen Zhu
% Date: April 13, 2021
%====================================================================
% Grid map creation class

classdef RowPlanning < handle
    properties
        width
        height
        resolution
        init_x
        init_y
        row_width
        offset
        radius
        map_offset
        ox
        oy
        vec
        ref_line
        rx
        ry
        tpx
        tpy
        tnx
        tny
        theta
    end
    methods
        % Class constructor - creates trajectory given the inputs
        function obj = RowPlanning(ox, oy, resolution,...
                row_width, offset, radius)
            obj.ox = ox;
            obj.oy = oy;
            obj.resolution = resolution;
            obj.row_width = row_width;
            obj.offset = offset;
            obj.radius = radius; 
            obj.map_offset = round((40+obj.radius+obj.offset)/obj.resolution, -1);
            obj.width = ceil((max(obj.ox)-min(obj.ox)) / resolution) + obj.map_offset;
            obj.height = ceil((max(obj.oy)-min(obj.oy)) / resolution) + obj.map_offset;
            obj.init_x = (max(obj.ox)+min(obj.ox))/2;
            obj.init_y = (max(obj.oy)+min(obj.oy))/2;
            find_row_direction(obj);
            main(obj);
        end
        
    
        %{
        Finds the largest polygon line. This line will define the direction of tree
        rows.
        Input: polygon edge points (ox, oy)
        Output: Vector of largest polygon line and starting point of largest line
        %}
        function find_row_direction(obj)
            dist_max = 0;
            d = [0, 0];
            obj.ref_line = [0, 0];
            % For all points in polygon, find the largest eucledian dist.
            for i = drange(1:length(obj.ox)-1)
                dx = obj.ox(i+1) - obj.ox(i);
                dy = obj.oy(i+1) - obj.oy(i);
                d = sqrt(dx^2+dy^2);
                if d > dist_max
                    dist_max = d;
                    obj.vec = [dx dy];
                    obj.ref_line = [obj.ox(i) obj.oy(i)];
                end
            end
        end
        %{
        Transform xy coordinates back to global coordinate
        Input: Transformed xy coordinates, vector, and reference point
        Output: Global xy coordinates
        %}
        function [trans_x, trans_y] = transform_coordinates_back(obj, x, y)
            % Find the rotation angle using largest polygon line vector to transform coordinate 
            angle = rad2deg(atan2(obj.vec(2),obj.vec(1)));
            rot = rotz(-angle);
            % Transform xy coordinate by rotating in z
            transformed_xy = [x; y]' * rot(1:2, 1:2);
            % Translate xy coordinate by origin of largest line
            trans_x = transformed_xy(:,1) + obj.ref_line(1);
            trans_y = transformed_xy(:,2) + obj.ref_line(2);

        end
        
        %s
        %{
        Transform xy coordinates with respect to vector and reference point
        Input: xy coordinates, vector, and reference point
        Output: Transformed xy coordinates
        %}
        function [trans_x, trans_y] = transform_coordinates(obj, x, y)
            tranlate_x = x - obj.ref_line(1);
            tranlate_y = y - obj.ref_line(2);
            angle = rad2deg(atan2(obj.vec(2),obj.vec(1)));
            rot = rotz(angle);
            transformed_xy = [tranlate_x; tranlate_y]' * rot(1:2, 1:2);
            trans_x = transformed_xy(:,1);
            trans_y = transformed_xy(:,2);
        end

        %{
        Creates T-shaped agricultural turns in between rows.
        Input: Starting/ending point of the current row, and starting/ending
        point of the next row; radius of turns; resolution of grids; flip 
        direction boolean.
        Output: Trajectory points of the turn.
        %}
        function [x, y, wt] = turn(obj, x1, y1, x2, y2, flip)
            if flip
                % If flip is true, create turns from the start of current 
                % row to the start of next row
                
                % Define xy-quarter-circle of first turn
                x1t = x1:-obj.resolution:x1-obj.radius;
                y1t = -sqrt(obj.radius^2-(x1t-x1).^2)+ y1+obj.radius;
                % Define xy-quarter-cirlce of second turn
                x2t = x2-obj.radius:obj.resolution:x2;
                y2t = sqrt(obj.radius^2-(x2t-x2).^2)+ y2-obj.radius;
                % Define heading angle of turns
                w1t = zeros(1, length(x1t));
                w2t = zeros(1, length(x1t));
                for i = drange(1:length(x1t)-1)
                    w1t(i) = atan2((y1t(i+1)-y1t(i)),(x1t(i+1)-x1t(i)));
                    w2t(i) = atan2((y2t(i+1)-y2t(i)),(x2t(i+1)-x2t(i)));
                end
                w1t(end) = pi/2;
                w2t(end) = 0;
                % Define straight line reverse from first turn to second
                
                % If radius < row_width create line from end of first turn
                % to beginning of second row
                if obj.radius < obj.row_width
                    y3t = y1t(end):obj.resolution:y2t(1);
                    x3t = ones(1, length(y3t))*x1t(end);
                else
                    y3t = y1t(end):-obj.resolution:y2t(1);
                    x3t = ones(1, length(y3t))*x1t(end);
                end
                % Heading angle during when robot is reversing
                w3t = ones(1, length(y3t))*w1t(end);
                
                % Return coordinates
                x = [x1t x3t x2t];
                y = [y1t y3t y2t];
                wt = [w1t w3t w2t];
            else
                % If flip is false, create turns from the end of current 
                % row to the end of next row
                
                % Same steps as above but reversed
                x1t = x1:obj.resolution:x1+obj.radius;
                y1t = -sqrt(obj.radius^2-(x1t-x1).^2)+ y1+obj.radius;
                x2t = x2+obj.radius:-obj.resolution:x2;
                y2t = sqrt(obj.radius^2-(x2t-x2).^2)+ y2-obj.radius;
                w1t = zeros(1, length(x1t));
                w2t = zeros(1, length(x1t));
                for i = drange(1:length(x1t)-1)
                    w1t(i) = atan2((y1t(i+1)-y1t(i)),(x1t(i+1)-x1t(i)));
                    w2t(i) = atan2((y2t(i+1)-y2t(i)),(x2t(i+1)-x2t(i)));
                end
                w1t(end) = pi/2;
                w2t(end) = pi ; 
                if obj.radius < obj.row_width
                    y3t = y1t(end):obj.resolution:y2t(1);
                    x3t = ones(1, length(y3t))*x1t(end);
                else
                    y3t = y1t(end):-obj.resolution:y2t(1);
                    x3t = ones(1, length(y3t))*x1t(end);
                end
                w3t = ones(1, length(y3t))*w1t(end);
                
                x = [x1t x3t x2t];
                y = [y1t y3t y2t];
                wt = [w1t w3t w2t];
            end
        end

        %{
        Creates path trajectory using paired row points
        Input: Route nodes, resolution, radius of turns, offset of turns
        Output: Formed path trajectory between rows
        %}
        function [route_x, route_y, theta] = create_path(obj, route_x_pts, route_y_pts)
            switch_moving = false;
            route_x = [];
            route_y = [];
            theta = [];
            % Iterate for all row lines to be travelled
            for i = drange(1:length(route_x_pts))
                % Define the max and min points of paired row-points and
                % add offset between turns
                max_x = max(route_x_pts(i,:));
                min_x = min(route_x_pts(i,:));
                max_x  = max_x + obj.offset;
                min_x  = min_x - obj.offset;
                % If switch_moving is false create row to be travelled
                % from left to right
                if ~switch_moving
                    % If last iteration skip
                    if i==length(route_x_pts)
                    % elseif max point is less than next point + offset
                    % define it as new max
                    elseif max_x < max(route_x_pts(i+1,:)+obj.offset)
                        max_x = max(route_x_pts(i+1,:)+obj.offset);
                    end
                    % If array is empty, ignore
                    if isempty(route_x)
                    % Elseif the end of previous row points is not equal to
                    % current row min val, then define as new min
                    elseif route_x(end) ~= min_x;
                        min_x = route_x(end);
                    end
                    % Create path from left to right, spanned by resolution
                    row_path_x = min_x:obj.resolution:max_x;
                    % Switch direction
                    switch_moving = ~switch_moving;
                    % If not end row, create turn
                    if i~=length(route_x_pts)
                        [turnx, turny wt] = turn(obj, max_x, route_y_pts(i,1), ...
                            max_x, route_y_pts(i+1,1), false);
                    else
                        turnx = [];
                        turny = [];
                        wt = [];
                    end
                    % Create heading angle during row traversal
                    row_theta = ones(1,length(row_path_x))*0;
                    
                else
                    % If switch_moving is true create row to be travelled
                    % from right to left
                    
                    % Same steps as above
                    if i==length(route_x_pts)
                    elseif min_x < min(route_x_pts(i+1,:)-obj.offset)
                        min_x =  min(route_x_pts(i+1,:))-obj.offset;
                    end
                    if isempty(route_x)
                    elseif route_x(end) ~= max_x;
                        max_x = route_x(end);
                    end
                    row_path_x = max_x:-obj.resolution:min_x;
                    switch_moving = ~switch_moving;

                    if i~=length(route_x_pts)
                        [turnx, turny, wt] = turn(obj, min_x, route_y_pts(i,1), ...
                            min_x, route_y_pts(i+1,1), true);
                    else
                        turnx = [];
                        turny = [];
                        wt = [];
                    end
                    row_theta = ones(1,length(row_path_x))*pi;
                end
                % Output the formed path with xy coordinate and heading
                row_path_y = ones(1,length(row_path_x))*route_y_pts(i,1);
                route_x = real([route_x row_path_x turnx]);
                route_y = real([route_y row_path_y turny]);
                theta = [theta row_theta wt];
            end
        end

        %{
        Converts tree nodes to line points to to map obstacles in grid map
        Input: Tree row nodes, resolution
        Output: Spanned points between rows
        %}
        function [tree_row_x_pts, tree_row_y_pts] = tree_nodes_to_points(obj, tree_row_x_nodes, tree_row_y_nodes)
            tree_row_x_pts = [];
            tree_row_y_pts = [];
            % For all tree rows, span points between rows
            for i = drange(1:length(tree_row_x_nodes))
                max_x = max(tree_row_x_nodes(i,:));
                min_x = min(tree_row_x_nodes(i,:));
                tree_row_x_nodes(i,1) = min_x;
                tree_row_x_nodes(i,2) = max_x;
                max_y = max(tree_row_y_nodes(i,:));
                min_y = min(tree_row_y_nodes(i,:));
                tree_row_y_nodes(i,1) = min_y;
                tree_row_y_nodes(i,2) = max_y;

                tree_row_x_pts = [tree_row_x_pts min_x:obj.resolution:max_x];
                tree_row_y_pts = [tree_row_y_pts ones(1,length(min_x:obj.resolution:max_x))*tree_row_y_nodes(i,1)];
            end
        end
        
        %{
        Main row planning function 
        Input: Class parameters
        Output: Trajectory of path and cost map of the trajectory
        %}
        function main(obj)
            % Transform the polygon line coordinates with respect to the
            % largest line
            [tx, ty] = transform_coordinates(obj, obj.ox, obj.oy);
            % Define the number of rows to be produced
            rows = floor((max(ty)-min(ty))* 2/obj.row_width);
            % Define points spanned in the y axis to create rows
            row_y = linspace(min(ty)+(obj.offset/2), max(ty)+(obj.offset/2), rows);
            % Create paired-row xy-points based on the spanned y-axis points
            [row_x_pts, row_y_pts] = obj.separate_rows(tx,ty,row_y);
            % Create rows to be travelled
            route_x_nodes = row_x_pts(1:2:length(row_x_pts),:);
            route_y_nodes = row_y_pts(1:2:length(row_y_pts),:);
            % Create rows populated by trees
            tree_row_x_nodes = row_x_pts(2:2:length(row_x_pts),:);
            tree_row_y_nodes = row_y_pts(2:2:length(row_y_pts),:);
            tree_row_x = reshape(tree_row_x_nodes, [], 1);
            tree_row_y = reshape(tree_row_y_nodes, [], 1);   
            % Convert tree row lines to points
            [tree_row_x_pts, tree_row_y_pts] = tree_nodes_to_points(obj, tree_row_x_nodes, tree_row_y_nodes);
            % Create path using trajectory rows lines
            [route_x, route_y, theta] = create_path(obj, route_x_nodes, route_y_nodes);
            % Transform the coordinates back to global
            [obj.rx, obj.ry] = transform_coordinates_back(obj, route_x, route_y);
            [obj.tpx, obj.tpy] = transform_coordinates_back(obj, tree_row_x_pts, tree_row_y_pts);
            [obj.tnx, obj.tny] = transform_coordinates_back(obj, tree_row_x', tree_row_y');
            % Reshape tree points for mapping
            obj.tnx = reshape(obj.tnx, [], 2);
            obj.tny = reshape(obj.tny, [], 2);
            % Return heading angles
            obj.theta = theta' + atan2(obj.vec(2),obj.vec(1));
        end
        
        function plot_path(obj)
            close all;
            set(0, 'DefaultFigurePosition', [300 300  800 600]);
            figure(1);
            plot(obj.rx, obj.ry, '--k')
            hold on 
            plot(obj.ox, obj.oy, '-xb')
            hold on 
            for i = drange(1:length(obj.tnx))
            line(obj.tnx(i,:), obj.tny(i,:), 'Color','red','LineStyle','-', 'Marker', 'x')
            end
            
            map = Map(obj.width, obj.height, obj.resolution, obj.init_x, obj.init_y, 1);
            map.formulate_polygon(obj.ox, obj.oy);
            map.expand_grid((obj.radius+obj.offset)/obj.resolution+5)
            map.set_value_from_cartesian_array(obj.rx, obj.ry, 0.5)
            map.set_value_from_cartesian_array(obj.tpx, obj.tpy, 1)         
            figure(2);
            map.plot_gridmap()
        end
    end
    methods(Static)
        %{
        Creates rows using the transformed polygon points. Equally spaced points 
        are then used to formulate rows.
        Input: Transformed polygon points (tx, ty), and spaced y points (row_y)
        Output: Array of two points defining the row line.
        %}
        function [row_x_pts, row_y_pts] = separate_rows(tx, ty, row_y)
            pol_points = length(ty)-1;
            row_num = length(row_y);
            row_x_pts = zeros(row_num-1, 2);
            row_y_pts = zeros(row_num-1, 2);
            k = 1;
            % For all y-axis row points, find the intersection with polygon 
            % lines
            for r = drange(1:row_num)
                % Check each polygon point
                for i = drange(1:pol_points)
                    l = i+1;
                    if l > pol_points
                        l = 1;
                    end
                    max_y = max(ty(i), ty(l));
                    min_y = min(ty(i), ty(l));
                    % If y_row point intersects polygon line, define the
                    % point to be equal to the intersection
                    if row_y(r) == min_y
                        if row_y_pts(r,k)~=0
                            k = 2;
                        else
                            k = 1;
                        end
                        row_x_pts(r,k) = tx(l);
                        row_y_pts(r,k) = row_y(r);
                        continue
                    % If y_row point is not between two polygon points,
                    % skip current iteration
                    elseif ~(min_y < row_y(r) && row_y(r) < max_y)
                        continue
                    % Else calculate the intersection between y_row point
                    % and the polygon line
                    else
                        m = (ty(l) - ty(i)) / (tx(l) - tx(i));
                        interp = (row_y(r)-ty(i))/m + tx(i);
                        if row_x_pts(r,k)~=0
                            k = 2;
                        else
                            k = 1;
                        end
                        row_x_pts(r,k) = interp;
                        row_y_pts(r,k) = row_y(r);
                    end
                end
            end
        end
    end
end
    
    




