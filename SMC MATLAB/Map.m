%====================================================================
% Author: Cesar Wen Zhu
% Date: April 13, 2021
%====================================================================
% Inter-row coverage path planner

classdef Map < handle
    properties
       grid_width
       grid_height
       grid_resolution
       init_x
       init_y
       init_val
       lower_limit_x
       lower_limit_y
       grid_array
       
    end
    methods
        % Class constructor - creates map given the inputs
        function obj = Map(width, height, resolution,...
                init_x, init_y, init_val)
            
            obj.grid_width = width;
            obj.grid_height = height;
            obj.grid_resolution = resolution;
            
            obj.init_x = init_x;
            obj.init_y = init_y; 
            obj.init_val = init_val;
            
            obj.lower_limit_x = obj.init_x - obj.grid_width/2 * obj.grid_resolution;
            obj.lower_limit_y = obj.init_y - obj.grid_height/2 * obj.grid_resolution;
            
            obj.grid_array = zeros(1, obj.grid_width * obj.grid_height);
        end

        % Sets cost value to the map using indexes
        function set_value_from_index(obj, x_ind, y_ind, val)
            grid_ind = index_to_grid_array_ind(obj, x_ind, y_ind);
            % Check if position in map array has a greater cost
            if 1 <= grid_ind && grid_ind < length(obj.grid_array)
                % If not, set the new cost value
                obj.grid_array(grid_ind) = val;
            end
        end
        
        % Sets cost value to the map using xy coordinates
        function set_value_from_cartesian(obj, x_val, y_val, val)
            [x_ind, y_ind] = cartesian_to_index(obj, x_val, y_val);
            set_value_from_index(obj, x_ind, y_ind, val)

        end
        
        % Sets value to the map using multiple xy coordinates
        function set_value_from_cartesian_array(obj, x_val, y_val, val)
            for i = drange(1:length(x_val))
                set_value_from_cartesian(obj, x_val(i), y_val(i), val)
            end 
        end
        
        % Converts indexes to map array number
        function grid_array_ind = index_to_grid_array_ind(obj, x_ind, y_ind)
            grid_array_ind = y_ind * obj.grid_width + x_ind;
            if isempty(grid_array_ind)
                disp('Indexes not in array')
            end
        end
        
        % Converts indexes to xy coordinates
        function [x_val, y_val] = index_to_cartesian(obj, x_ind, y_ind)
            x_val = obj.lower_limit_x + x_ind * obj.grid_resolution + ...
                obj.grid_resolution / 2;
            y_val = obj.lower_limit_y + y_ind * obj.grid_resolution + ...
                obj.grid_resolution / 2;
        end
        
        % Converts xy coordinates to indexes
        function [x_ind, y_ind] = cartesian_to_index(obj, x_val, y_val)
            x_ind = floor((x_val - obj.lower_limit_x) / obj.grid_resolution);
            y_ind = floor((y_val - obj.lower_limit_y) / obj.grid_resolution);
        end
        
        % Checks if grid is occupied
        function bool = check_occupied(obj, x_ind, y_ind, occupied_val)
            grid_ind = index_to_grid_array_ind(obj, x_ind, y_ind);
            val = obj.grid_array(grid_ind);
            if val >= occupied_val || isempty(val)
                bool = true;
            else
                bool = false;
            end
        end
        
        % Creates polygon using points
        function formulate_polygon(obj, x_pol, y_pol)
            % If the first and last point do not match, polygon is not
            % closed
            if x_pol(1) ~= x_pol(end) || y_pol(1) ~= y_pol(end)
                x_pol = [x_pol x_pol(1)];
                y_pol = [y_pol y_pol(1)];
            end
            % For each index find if it is inside defined polygon using
            % find_indexes_inside_polygon
            for x_ind = drange(0:obj.grid_width)
                for y_ind = drange(0:obj.grid_height)
                    [x_val, y_val] = index_to_cartesian(obj, x_ind, y_ind);
                    bool = find_indexes_inside_polygon(obj, x_val, y_val, x_pol, y_pol);
                    % If not inside circle, set cost as 1 (occupied)
                    if ~bool
                        set_value_from_index(obj, x_ind, y_ind, 1)
                    end
                end
            end
        end
        
        % Find areas of the map that lie inside the polygon
        function bool = find_indexes_inside_polygon(obj, x_val, y_val, x_pol, y_pol)
            pol_points = length(x_pol)-1;
            bool = false;
            % Loop for all polygon points
            for i = drange(1:pol_points)
                l = i+1;
                % Index exceeds bounds, set as 1 (returns to the first
                % polygon point which equal to each other)
                if l > pol_points
                    l = 1;
                end
                % Find max and min x-points between a polygon line (2 points)
                if x_pol(i) >= x_pol(l)
                    max_x = x_pol(i);
                    min_x = x_pol(l);
                else
                    max_x = x_pol(l);
                    min_x = x_pol(i);
                end
                % If the given value does not span between the max and min,
                % it is not inside the current line. Continue to next
                % iteration
                if ~(min_x < x_val && x_val < max_x)
                    continue
                % Else check the the point is above the line
                else
                    m = (y_pol(l) - y_pol(i)) / (x_pol(l) - x_pol(i));
                    inside = y_pol(i) + m * (x_val - x_pol(i)) - y_val;
                    % If not above line, then return false  
                    if (inside) > 0
                        bool = ~bool;
                    end
                end
            end
        end
        
        % Expand the grid map by the provided index
        function expand_grid(obj, index)
            for j = drange(1:index)
                x_ind = [];
                y_ind = [];
                % Find all occupied indexes
                for x = drange(1:obj.grid_width-1)
                    for y = drange(1:obj.grid_height-1)
                        if ~check_occupied(obj, x, y, 1)
                            x_ind = [x_ind x];
                            y_ind = [y_ind y];
                        end
                    end
                end
                % Set all occupied indexes and its neighbours to zero 
                for i = [x_ind; y_ind]
                    set_value_from_index(obj, i(1)+1, i(2), 0)
                    set_value_from_index(obj, i(1), i(2)+1, 0)
                    set_value_from_index(obj, i(1)+1, i(2)+1, 0)
                    set_value_from_index(obj, i(1)-1, i(2), 0)
                    set_value_from_index(obj, i(1), i(2)-1, 0)
                    set_value_from_index(obj, i(1)-1, i(2)-1, 0)
                    set_value_from_index(obj, i(1)-1, i(2)+1, 0)
                    set_value_from_index(obj, i(1)+1, i(2)-1, 0)
                end         
            end
        end
        
        % Plot gridmap
        function plot_gridmap(obj)
            grid_map = reshape(obj.grid_array, [obj.grid_width, obj.grid_height])';
            pcolor(grid_map);
            cmap = colormap('parula');
            grid off;
        end
    end
end

