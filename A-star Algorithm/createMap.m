function createMap(gridMap, costMap, pathback, goalPos, startPos, method)
figure;
n = length(gridMap); % find length of grid

[goalposy,goalposx] = ind2sub([n,n],goalPos); % convert goal position to row and columns
[startposy,startposx] = ind2sub([n,n],startPos); % convert start position to row and columns

pcolor([gridMap gridMap(:,end); gridMap(end,:) gridMap(end,end)]); % create a grid plot with walls
cmap = colormap('parula'); % assign color spectrum
cmap(1,:) = zeros(3,1); % make the color spectrum to be white at a value of zero
cmap(end,:) = ones(3,1); % make the color spectrum to be black at a value of one
colormap(flipud(cmap)); % re-map spectrum
switch method % create specific titles for each method
    case 'astar'
        title('A Star');
    case 'dijkstra'
        title('Dijkstra');
    case 'GBFS'
        title('Greedy Best First'); 
        % flip cost matrix since heuristic calculate the lowest cost
        % towards goal
        cmap(1,:) = ones(3,1); % make the color spectrum to be black at a value of one
        cmap(end,:) = zeros(3,1); % make the color spectrum to be white at a value of zero 
        colormap(cmap); % re-map spectrum
end
hold on;
axishandle = pcolor([costMap costMap(:,end); costMap(end,:) costMap(end,end)]); % map cost spectrum to figure
set(axishandle,'CData',[costMap costMap(:,end); costMap(end,:) costMap(end,end)]); % adjust figure
set(gca,'CLim',[0 1.1*max(costMap(find(costMap < Inf)))]); % adjust limits of cost map

% re-align axis label
grid off
set(gca,'XTick',1.5:1:n+.5)
set(gca,'XTickLabel',1:n)
set(gca,'YTick',1.5:1:n+.5)
set(gca,'YTickLabel',1:n)
plot(goalposx+0.5,goalposy+0.5,'rx','MarkerSize',13,'LineWidth',4); % plot goal position
plot(startposx+0.5,startposy+0.5,'b+','MarkerSize',10,'LineWidth',4); % plot start position
if ~isempty(pathback)
    plot(pathback(:,2)+0.5,pathback(:,1)+0.5,'g','LineWidth',4); % plot path
end
