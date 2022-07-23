clear;
close all;
clc;

n = 20; % identify grid size
wallpercent = 0.3; % identify wall percentage

% Create grid map matrix
gridMap = zeros(n, n); % create map with n x n field

% Create randomly distributed walls
for i=0:n*n*wallpercent
    gridMap(ceil(n.*rand),ceil(n.*rand)) = Inf; % create walls with infinity values
end

% Create user defined walls
% gridMap([3:18],10) = Inf;
% gridMap([3,18],[10:18]) = Inf;

% Create user defined start and goal position
% startPos = sub2ind([n,n],10,2); % create start position with random row & column
% goalPos = sub2ind([n,n],10,12); % create goal position with random row & column

% Start & goal initialization
startPos = sub2ind([n,n],ceil(n.*rand),ceil(n.*rand)); % create start position with random row & column
goalPos = sub2ind([n,n],ceil(n.*rand),ceil(n.*rand)); % create goal position with random row & column
gridMap(startPos) = 0; % zero value for start position
gridMap(goalPos) = 0; % zero value for goal position

% Cost matrix
costMap = NaN*ones(n,n); % initialize cost of grid map
costMap(startPos) = 1; % assign a value of 1 to start position
costMap(goalPos) = 1; % assign a value of 1 to start position

% Path matrix
pathMap = zeros(n,n); % create empty path matrix
pathMap(startPos) = 9; % assign a value of 9 to indicate start position
pathMap(goalPos) = 10; % assign a value of 10 to indicate goal position
pathMap(gridMap == Inf) = 0; % all walls assign a value of 0

% Pathfinding algorithms
[pathbackAStar, costMapAStar]  = astar(gridMap, costMap, pathMap, ...
    goalPos, startPos);
[pathbackDijkstra, costMapDijkstra]  = dijkstra(gridMap, costMap, pathMap, ...
    goalPos, startPos);
[pathbackGBFS, costMapGBFS]  = GBFS(gridMap, costMap, pathMap, ...
    goalPos, startPos);


if ~isempty(pathbackAStar)
    % Determine number of movements
    movementNumAStar = length(pathbackAStar)-1;
    movementNumDijkstra = length(pathbackDijkstra)-1;
    movementNumGBFS = length(pathbackGBFS)-1;
    movementNum = [movementNumAStar; movementNumDijkstra; movementNumGBFS];
    % Determine number of grids explored
    gridNumAStar = sum(sum(costMapAStar > 0));
    gridNumDijkstra = sum(sum(costMapDijkstra > 0));
    gridNumGBFS = sum(sum(costMapGBFS > 0));
    gridNum = [gridNumAStar; gridNumDijkstra; gridNumGBFS];
    % Create table
    Name = {'A Star','Dijkstra','GBFS'};
    table(movementNum, gridNum, 'RowNames', Name) 
end
%% Create figure
createMap(gridMap, costMapAStar, pathbackAStar, goalPos, startPos, 'astar')
createMap(gridMap, costMapDijkstra, pathbackDijkstra, goalPos, startPos, 'dijkstra')
createMap(gridMap, costMapGBFS, pathbackGBFS, goalPos, startPos, 'GBFS')

