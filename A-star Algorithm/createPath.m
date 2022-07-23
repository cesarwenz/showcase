function pathBack = createPath(goalPos,pathMap)
% This function determines the path back to the start position
% Start position is indicated by a integer of 9 in the path map
n = length(pathMap);  % length of the field
[goalPosy,goalPosx] = ind2sub([n,n],goalPos); % find respective x and y goal position
pathBack = [goalPosy goalPosx]; % store current position

while pathMap(goalPos) ~= 9 % while start position is not reached
    switch pathMap(goalPos) % determine the movement direction in the path map
        case 1 % move right
            goalPosx = goalPosx + 1;
        case 2 % move left
            goalPosx = goalPosx - 1;
        case 3 % move down
            goalPosy = goalPosy + 1;
        case 4 % move up
            goalPosy = goalPosy - 1;
        case 5 % move left/down
            goalPosx = goalPosx - 1;
            goalPosy = goalPosy + 1;
        case 6 % move right/down
            goalPosx = goalPosx + 1;
            goalPosy = goalPosy + 1;
        case 7 % move left/up
            goalPosx = goalPosx - 1;
            goalPosy = goalPosy - 1;
        case 8 % move right/up
            goalPosx = goalPosx + 1;
            goalPosy = goalPosy - 1;
    end
    pathBack = [pathBack; goalPosy goalPosx]; % store path to get back
    goalPos = sub2ind([n n],goalPosy,goalPosx); % convert x and y position back to subscript
end
end
