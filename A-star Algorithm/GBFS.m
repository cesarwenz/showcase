%% Main Loop
function [pathback, costMap]  = GBFS(gridMap, costMap, pathMap, goalPos, startPos)

% Initialize node variables
openNode = [startPos]; % assign first open node to start position
openNodeCosts = [0]; % zero cost for start position
openNodeHeuristics = [Inf]; % heuristic cost of infinity for start position
closedNode = []; % empty array for close node
closedNodeCosts = []; % empty array for close node cost
movementdirections = [1, 2, 3, 4, 5, 6, 7, 8]; % movement cases for path matrix

% Movement direction describe all 8 directions from a node
% (for switch cases in functions)
% case 1 - Right
% case 2 - Left
% case 3 - Down
% case 4 - Up
% case 5 - Left/Down Diag
% case 6 - Right/Down Diag
% case 7 - Left/Up Diag
% case 8 - Right/Up Diag

while ~any(openNode == goalPos) && ~isempty(openNode)
    % The while statements always checks if the goal node has been reached
    % or if no more open nodes can be explored (no solution)
    [~, ind] = min(openNodeHeuristics); % identifies minimum cost min(h(n)+g(n))
    [neighbourCosts,neighbourHeuristics,neighbourPositions] = calculateAllCosts(openNode(ind), ...
        gridMap,goalPos, startPos); % calls function to determine costs of neighbouring grids
    closedNode = [closedNode; openNode(ind)]; % stores open nodes to the closed node array
    closedNodeCosts = [closedNodeCosts; openNodeCosts(ind)]; % stores open nodes costs to closed node cost array
    openNode(ind) = []; % removes open nodes that were closed
    openNodeCosts(ind) = []; % removes open node cost that were closed
    openNodeHeuristics(ind) = [];  % removes open node heuristic that were closed
    for i=1:length(neighbourPositions)  % loops for all neighbouring grids
        if neighbourCosts(i) ~= Inf % if the neighbour is not a wall
            if ~any([closedNode; openNode] == neighbourPositions(i)) % if any of the neighbour is not open or closed
                pathMap(neighbourPositions(i)) = movementdirections(i); % update path direction for neighbour position
                costMap(neighbourPositions(i)) = neighbourCosts(i) + neighbourHeuristics(i); % update cost chart for neighbour position
                openNode = [openNode; neighbourPositions(i)]; % adds unnasigned neighbour to open node
                openNodeCosts = [openNodeCosts; neighbourCosts(i)]; % updates open node costs
                openNodeHeuristics = [openNodeHeuristics; neighbourHeuristics(i)]; % updates open node heuristics
            elseif any(openNode == neighbourPositions(i)) % else if the open node was already observed/assigned to open
                I = find(openNode == neighbourPositions(i)); % store position where node has been seen
                if openNodeCosts(I) > neighbourCosts(i) % check if the seen node has a lower value than the old
                    costMap(openNode(I)) = neighbourCosts(i) + neighbourHeuristics(i); % update cost
                    openNodeCosts(I) = neighbourCosts(i); % update node cost
                    openNodeHeuristics(I) = neighbourHeuristics(i); % update heuristics
                    pathMap(openNode(I)) = movementdirections(i); % update direction
                end
            else % else the node is an observed closed node
                I = find(closedNode == neighbourPositions(i)); % store position where closed node has been seen
                if closedNodeCosts(I) > neighbourCosts(i) % check if the seen node has a lower value than the old
                    costMap(closedNode(I)) = neighbourCosts(i) + neighbourHeuristics(i); % update closed cost
                    closedNodeCosts(I) = neighbourCosts(i); % update node cost array
                    pathMap(closedNode(I)) = movementdirections(i); % update direction
                end
            end
        end
    end
end
if any(openNode==goalPos)
    disp('Solution found!'); % display solution was found
    pathback = createPath(goalPos,pathMap); % call function to determine the path back
    costMap(goalPos) = 0; % set goal position for cost gradient graphics
    costMap(startPos) = 0; % set start position for cost gradient graphics

else 
    disp('No Solution!');
    pathback = []; % return an empty array if no solution is found
end
end

%% Cost Calculation
function [neighbourCosts,neighbourHeuristics,neighbourPositions] = calculateAllCosts(openNode,gridMap, ...
    goalPos, startPos)
% This function calculates costs and heuristics of all neighbours and store its position
% Neighbours are determined based on the openNode positions
n = length(gridMap); % determine length of grid map
[openNodey openNodex] = ind2sub([n n],openNode); % convert open node subscripts to indeces
[goalPosy goalPosx] = ind2sub([n n],goalPos); % convert goal node subscripts to indeces
[startPosy startPosx] = ind2sub([n n],startPos); % convert startnode subscripts to indeces
neighbourCosts = Inf*ones(8,1); % create infinity array of 8x1 since cost are not known
neighbourHeuristics = Inf*ones(8,1); % create infinity array of 8x1 since heuristics are not known
pos = ones(8,2); % create empty position array
for i=1:8 % Evaluate each of the 8 neighbours
    switch i
        case 1 % check right
            newx = openNodex - 1;
            newy = openNodey;
            if newx > 0
                % if neightbour does not exceed grid bound
                pos(i,:) = [newy newx]; % store neighbour position
                gx = abs(goalPosx-newx); % calculate row distance between current node and goal
                gy = abs(goalPosy-newy); % calculate column distance between current node and goal
                neighbourHeuristics(i) = (gx + gy) - min(gx,gy); % calculate heuristics using octile distance
                neighbourCosts(i) = gridMap(newy,newx); % calculate cost using octile distance
            end
        case 2 % check left
            newx = openNodex + 1;
            newy = openNodey;
            if newx <= n
                % if neightbour does not exceed grid bound
                pos(i,:) = [newy newx]; % store neighbour position
                gx = abs(goalPosx-newx); % calculate row distance between current node and goal
                gy = abs(goalPosy-newy); % calculate column distance between current node and goal
                neighbourHeuristics(i) = (gx + gy) - min(gx,gy); % calculate heuristics using octile distance
                neighbourCosts(i) = gridMap(newy,newx); % calculate cost using octile distance
            end
        case 3 % check down
            newx = openNodex;
            newy = openNodey - 1;
            if newy > 0
                % if neightbour does not exceed grid bound
                pos(i,:) = [newy newx]; % store neighbour position
                gx = abs(goalPosx-newx); % calculate row distance between current node and goal
                gy = abs(goalPosy-newy); % calculate column distance between current node and goal
                neighbourHeuristics(i) = (gx + gy) - min(gx,gy); % calculate heuristics using octile distance
                neighbourCosts(i) = gridMap(newy,newx); % calculate cost using octile distance
            end
        case 4 % check up
            newx = openNodex;
            newy = openNodey + 1;
            if newy <= n
                % if neightbour does not exceed grid bound
                pos(i,:) = [newy newx]; % store neighbour position
                gx = abs(goalPosx-newx); % calculate row distance between current node and goal
                gy = abs(goalPosy-newy); % calculate column distance between current node and goal
                neighbourHeuristics(i) = (gx + gy) - min(gx,gy); % calculate heuristics using octile distance
                neighbourCosts(i) = gridMap(newy,newx); % calculate cost using octile distance
            end
        case 5 % check left/down
            newx = openNodex + 1;
            newy = openNodey - 1;
            if (newx <= n) && (newy > 0) && (gridMap(newy+1,newx)~= Inf && gridMap(newy,newx-1)~= Inf)
                % if neightbour does not exceed grid bound and does not have two
                % adjacenet walls between diagonal
                pos(i,:) = [newy newx]; % store neighbour position
                gx = abs(goalPosx-newx); % calculate row distance between current node and goal
                gy = abs(goalPosy-newy); % calculate column distance between current node and goal
                neighbourHeuristics(i) = (gx + gy) - min(gx,gy); % calculate heuristics using octile distance
                neighbourCosts(i) = gridMap(newy,newx); % calculate cost using octile distance
            end
        case 6 % check right/down
            newx = openNodex - 1;
            newy = openNodey - 1;
            if (newx > 0) && (newy > 0) && (gridMap(newy+1,newx)~= Inf && gridMap(newy,newx+1)~= Inf)
                % if neightbour does not exceed grid bound and does not have two
                % adjacenet walls between diagonal
                pos(i,:) = [newy newx]; % store neighbour position
                gx = abs(goalPosx-newx); % calculate row distance between current node and goal
                gy = abs(goalPosy-newy); % calculate column distance between current node and goal
                neighbourHeuristics(i) = (gx + gy) - min(gx,gy); % calculate heuristics using octile distance
                neighbourCosts(i) = gridMap(newy,newx); % calculate cost using octile distance
            end
        case 7 % check left/up
            newx = openNodex + 1;
            newy = openNodey + 1;
            if (newx <= n) && (newy <= n) && (gridMap(newy-1,newx)~= Inf && gridMap(newy,newx-1)~= Inf)
                % if neightbour does not exceed grid bound and does not have two
                % adjacenet walls between diagonal
                pos(i,:) = [newy newx]; % store neighbour position
                gx = abs(goalPosx-newx); % calculate row distance between current node and goal
                gy = abs(goalPosy-newy); % calculate column distance between current node and goal
                neighbourHeuristics(i) = (gx + gy) - min(gx,gy); % calculate heuristics using octile distance
                neighbourCosts(i) = gridMap(newy,newx); % calculate cost using octile distance
            end
        case 8 % check right/up
            newx = openNodex - 1;
            newy = openNodey + 1;
            if (newx > 0) && (newy <= n) && (gridMap(newy-1,newx)~= Inf && gridMap(newy,newx+1)~= Inf)
                % if neightbour does not exceed grid bound and does not have two
                % adjacenet walls between diagonal
                pos(i,:) = [newy newx]; % store neighbour position
                gx = abs(goalPosx-newx); % calculate row distance between current node and goal
                gy = abs(goalPosy-newy); % calculate column distance between current node and goal
                neighbourHeuristics(i) = (gx + gy) - min(gx,gy); % calculate heuristics using octile distance
                neighbourCosts(i) = gridMap(newy,newx); % calculate cost using octile distance
            end
    end
end
neighbourPositions = sub2ind([n n],pos(:,1),pos(:,2)); % convert neighbour position to subscript
end
